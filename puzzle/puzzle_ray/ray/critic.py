import os
import logging
import time
from functools import partial

import ray

import torch

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options
from megatron.core.enums import ModelType
from megatron.core import parallel_state, tensor_parallel
from megatron.utils import average_losses_across_data_parallel_group

from puzzle.pipeline.inference import forward_value
from puzzle.utils.data.data_utils import MiniDataset
from puzzle.pipeline.training import train_step

from puzzle_ray.ray.group import BaseRole
from puzzle_ray.ray.group import get_reward_model, get_megatron_optimizer, get_optimizer_param_scheduler, model_provider


class ExpDataset():
    def __init__(self, exp_data):
        self.exp_data = exp_data

    def __len__(self):
        return self.exp_data['prompts'].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.exp_data.items()}


@ray.remote(num_gpus=1)
class CriticModelRayRole(BaseRole):

    def init_model_from_pretrained(self, *args, **kwargs):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        config = {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 8,
            "load_model_from_hf_config": True,
            "model_name_or_path": "/home/kinman/code/RLHF/puzzle/puzzle/example/config/Llama2-350m-hf",
            "critic_model_name_or_path": "/home/kinman/code/RLHF/puzzle/puzzle/example/config/Llama2-350m-hf",
            "seq_length": 512,
            "max_position_embeddings": 512,
            "micro_batch_size": 2,
            "global_batch_size": 16,
            "inference_batch_times_seqlen_threshold": 1024,
            "lr": 0.00015,
            "train_iters": 500000,
            "lr_decay_iters": 320000,
            "lr_decay_style": "cosine",
            "min_lr": 1.0e-5,
            "weight_decay": 1e-2,
            "add_bias_linear": False,
            "add_position_embedding": False,
            "lr_warmup_fraction": 0.01,
            "clip_grad": 1.0,
            "use_contiguous_buffers_in_local_ddp": False,
            "fp16": True,

            # Tokenizer args
            'tokenizer_type': 'PretrainedFromHF',
            'tokenizer_model': "/home/dataset/llama-2-hf-all/Llama-2-7b-hf",
            'data_path': "/home/dataset/rlhf-data/Dahoas/rm-static",
            'split': "949,50,1",

            # RLHF args
            'generation_batches': 1,
        }

        self._init_megatron(extra_args_provider=None,
                            args_defaults=config)

        args = get_args()

        # TODO(lkm): need to update
        args.per_device_generation_batch_size = args.global_batch_size // parallel_state.get_data_parallel_world_size()
        args.per_device_training_batch_size = args.global_batch_size // parallel_state.get_data_parallel_world_size()

        # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
        self.exp_mini_dataset = MiniDataset(args.generation_batches,
                                        args.per_device_training_batch_size)

        self.model, self.optimizer, self.opt_param_scheduler = self._init_critic(model_provider, model_type=ModelType.encoder_or_decoder)

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

    def _init_megatron(self, extra_args_provider, args_defaults):
        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults, ignore_unknown_args=True)

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

        args = get_args()

        args.global_rank = torch.distributed.get_rank()

    def _init_critic(self, model_provider_func, model_type,
                    no_wd_decay_cond=None,
                    scale_lr_cond=None,
                    lr_mult=1.0):
        model = get_reward_model(model_provider_func, model_type)
        optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        return model, optimizer, opt_param_scheduler

    def forward_value(self, seq, attention_mask, batch_size, prompt_length, seq_length):
        self.eval()
        seq = seq.cuda()
        attention_mask = attention_mask.cuda()
        with torch.no_grad():
            values = forward_value(self.model,
                                    seq,
                                    attention_mask,
                                    batch_size=batch_size,
                                    prompt_length=prompt_length,
                                    seq_length=seq_length,
                                    return_value_only=True)
        self.train()
        if parallel_state.is_pipeline_first_stage():
            values = values.cpu()
        return values

    def training_step(self, step):
        self.train()

        args = get_args()

        if parallel_state.get_tensor_model_parallel_rank() == 0:
            exp_data = self.exp_dataset[step]
            dataloader = torch.utils.data.DataLoader(ExpDataset(exp_data), batch_size=args.micro_batch_size, shuffle=True)
            train_data_iterator = iter(dataloader)

        critic_loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(self.forward_critic_step,
                        self.model,
                        self.optimizer,
                        self.opt_param_scheduler,
                        train_data_iterator)

        return critic_loss_dict

    def add_exp_dataset(self, batch_exp):
        self.exp_dataset = self.exp_mini_dataset.add(batch_exp)
        num_total_exp_iters = len(self.exp_dataset)
        return num_total_exp_iters

    def forward_critic_step(self, data_iterator, critic_model):
        """Critic forward step"""
        inputs = self.get_batch(data_iterator)

        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                            ref_log_probs, reward_score,
                                            action_mask)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        output_tensor = critic_model(input_ids=seq, position_ids=None, attention_mask=attention_mask)

        if parallel_state.is_pipeline_last_stage():
            output_tensor = output_tensor[:, :-1]
            return output_tensor[:, start:], partial(self.critic_loss_fn, old_values=old_values[:,
                                                                    start:], returns=returns, mask=action_mask[:, start:])

        return output_tensor, None

    def get_batch(self, data_iterator):
        """Generate a batch"""
        args = get_args()

        # Broadcast data.
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        # Items and their type.
        key_int64 = ['prompts', 'input_ids', 'attention_mask']
        datatype = torch.int64
        data_b = tensor_parallel.broadcast_data(key_int64, data, datatype)

        key_float32 = ['logprobs', 'ref_logprobs', 'value', 'rewards']
        datatype = torch.float32
        data_b.update(tensor_parallel.broadcast_data(key_float32, data, datatype))

        return data_b

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        # Adopted from https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/ppo_trainer.py
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def critic_loss_fn(self, values, old_values, returns, mask):
        # Adopted from https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/ppo_trainer.py
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        averaged_vf_loss = average_losses_across_data_parallel_group([vf_loss])
        return vf_loss, {'critic_loss': averaged_vf_loss[0]}

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def eval(self):
        def _eval(model):
            for module in model:
                module.eval()
        _eval(self.model)

    def train(self):
        def _train(model):
            for module in model:
                module.train()
        _train(self.model)

    def print_mem(self):
        if torch.distributed.get_rank() == 0:
            print(f"current memory allocated: {torch.cuda.memory_summary()}")

    def empty_cache(self):
        torch.cuda.empty_cache()
