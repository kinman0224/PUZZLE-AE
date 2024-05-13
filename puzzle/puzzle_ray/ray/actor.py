import os
import logging
import time
from functools import partial

import ray

import torch
import torch.distributed

from megatron import print_rank_0
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.training import get_megatron_optimizer, get_optimizer_param_scheduler
from megatron import get_args, get_timers
from megatron import get_tokenizer
from megatron.core import parallel_state, tensor_parallel
from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options
from megatron.utils import average_losses_across_data_parallel_group

from puzzle.utils.utils import gather_log_probs
from puzzle.pipeline.generation import generate, dis_bubble_generate
from puzzle.pipeline.training import train_step, train_step_dual, train_step_dual_bulk
from puzzle.utils.data.data_utils import create_datasets, MiniDataset

from puzzle_ray.ray.group import BaseRole
from puzzle_ray.ray.group import get_model, get_megatron_optimizer, get_optimizer_param_scheduler, model_provider


class ExpDataset():
    def __init__(self, exp_data):
        self.exp_data = exp_data

    def __len__(self):
        return self.exp_data['prompts'].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.exp_data.items()}


@ray.remote(num_gpus=1)
class ActorModelRayRole(BaseRole):

    # def init_model_from_pretrained(self, model_name, model_config, model_weights, model_type):
    def init_model_from_pretrained(self, *args, **kwargs):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        config = {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 8,
            "load_model_from_hf_config": True,
            # "model_name_or_path": "/home/dataset/llama-2-hf-all/Llama-2-7b-hf",
            # "actor_model_name_or_path": "/home/dataset/llama-2-hf-all/Llama-2-7b-hf",
            "model_name_or_path": "/home/kinman/code/RLHF/puzzle/puzzle/example/config/Llama2-350m-hf",
            "actor_model_name_or_path": "/home/kinman/code/RLHF/puzzle/puzzle/example/config/Llama2-350m-hf",
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
            'data_path': ["/home/dataset/rlhf-data/Dahoas/rm-static"],
            'data_split': "2,4,4",
            'max_prompt_seq_len': 256,
            'max_answer_seq_len': 256,

            "use_dis_bubble_generation": True,
            "pf_stage_mbs": 4,
            "ar_stage_mbs": 16,

            # RLHF args
            'data_output_path': '/tmp/data_files',
            'generation_batches': 1,
            'ppo_epochs': 1,
            'num_train_epochs': 1,
            'gradient_accumulation_steps': 1
        }

        self._init_megatron(extra_args_provider=None,
                            args_defaults=config)

        args = get_args()

        args.global_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)

        # TODO(lkm): need to update
        args.per_device_generation_batch_size = args.global_batch_size // parallel_state.get_data_parallel_world_size()
        args.per_device_training_batch_size = args.global_batch_size // parallel_state.get_data_parallel_world_size()

        self.tokenizer = get_tokenizer()

        prompt_train_dataloader, num_total_iters = create_datasets(args=args, tokenizer=self.tokenizer, train_phase=3)

        if prompt_train_dataloader is not None:
            self.prompt_train_data_iterator = iter(prompt_train_dataloader)

        # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
        self.exp_mini_dataset = MiniDataset(args.generation_batches,
                                        args.per_device_training_batch_size)

        self.model, self.optimizer, self.opt_param_scheduler = self._init_actor(model_provider, model_type=ModelType.encoder_or_decoder)

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

        return num_total_iters

    def _init_megatron(self, extra_args_provider, args_defaults):
        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults, ignore_unknown_args=True)

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

    def _init_actor(self, model_provider_func, model_type,
                    no_wd_decay_cond=None,
                    scale_lr_cond=None,
                    lr_mult=1.0):
        model = get_model(model_provider_func, model_type)
        optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        return model, optimizer, opt_param_scheduler

    def generate_sequences(self, step):
        self.eval()

        batch_prompt = None
        if parallel_state.get_tensor_model_parallel_rank() == 0:
            batch_prompt = next(self.prompt_train_data_iterator)
        keys = ['prompt', 'prompt_att_mask']
        batch_prompt = tensor_parallel.broadcast_data(keys, batch_prompt, torch.int64)

        prompt = batch_prompt['prompt']
        attention_maske = batch_prompt['prompt_att_mask']

        # only first pipeline stage would contain generated sequence
        seq, logprobs = self._generate_sequence(batch_prompt['prompt'], batch_prompt['prompt_att_mask'])

        if parallel_state.is_pipeline_first_stage():
            seq = seq.cpu()
            logprobs = logprobs.cpu()
            prompt = prompt.cpu()

        return seq, logprobs, prompt

    def _generate_sequence(self, prompts, mask):
        args = get_args()

        timers = get_timers()
        timers('generate-seq', log_level=0).start(barrier=True)
        torch.cuda.synchronize()
        st = time.time()
        if args.use_dis_bubble_generation:
            seq, output_log_probs = dis_bubble_generate(self.model, prompts, mask, max_length=args.max_answer_seq_len, return_output_log_probs=True)
        else:
            seq, output_log_probs = generate(self.model, prompts, mask, max_length=args.max_answer_seq_len, return_output_log_probs=True)
        torch.cuda.synchronize()
        ed = time.time()
        self.generate_time = ed - st
        timers('generate-seq').stop(barrier=True)

        out_seq = []
        if parallel_state.is_pipeline_first_stage():
            batch_size = prompts.shape[0]
            prompt_length = prompts.shape[1]

            ans = seq[:, prompt_length:]
            valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

            for i in range(batch_size):
                if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
                    continue
                else:
                    out_seq.append(seq[i:i + 1])
            out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim
        out_seq = None if isinstance(out_seq, list) else out_seq
        return out_seq, output_log_probs

    def training_step(self, step):
        self.train()

        args = get_args()

        if parallel_state.get_tensor_model_parallel_rank() == 0:
            exp_data = self.exp_dataset[step]
            dataloader = torch.utils.data.DataLoader(ExpDataset(exp_data), batch_size=args.micro_batch_size, shuffle=True)
            train_data_iterator = iter(dataloader)

        actor_loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(self.forward_actor_step,
                        self.model,
                        self.optimizer,
                        self.opt_param_scheduler,
                        train_data_iterator)

        return actor_loss_dict

    def add_exp_dataset(self, batch_exp):
        self.exp_dataset = self.exp_mini_dataset.add(batch_exp)
        num_total_exp_iters = len(self.exp_dataset)
        return num_total_exp_iters

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def forward_actor_step(self, data_iterator, actor_model):
        """Actor forward step"""
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

        output_tensor = actor_model(input_ids=seq, position_ids=None, attention_mask=attention_mask)

        if parallel_state.is_pipeline_last_stage():
            output_tensor = gather_log_probs(output_tensor[:, :-1, :], seq[:, 1:])
            return output_tensor[:, start:], partial(self.actor_loss_fn, old_logprobs=log_probs[:, start:], advantages=advantages, mask=action_mask[:, start:])

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

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        # Adopted from https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/ppo_trainer.py
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        averaged_pg_loss = average_losses_across_data_parallel_group([pg_loss])
        return pg_loss, {'actor_loss': averaged_pg_loss[0]}

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

    def train(self):
        def _train(model):
            for module in model:
                module.train()
        _train(self.model)

    def eval(self):
        def _eval(model):
            for module in model:
                module.eval()
        _eval(self.model)

    def get_tensor_model_parallel_size(self):
        return parallel_state.get_tensor_model_parallel_world_size()

    def get_pipeline_model_parallel_size(self):
        return parallel_state.get_pipeline_model_parallel_world_size()

    def get_data_parallel_size(self):
        return parallel_state.get_data_parallel_world_size()

    def print_mem(self):
        if torch.distributed.get_rank() == 0:
            print(f"current memory allocated: {torch.cuda.memory_summary()}")

    def empty_cache(self):
        torch.cuda.empty_cache()
