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
from megatron.hf import load_hf_ckpt

from puzzle.utils.utils import gather_log_probs
from puzzle.core.utils import set_model_mpu
from puzzle.utils.data.data_utils import create_datasets, MiniDataset
from puzzle.pipeline.generation import generate, dis_bubble_generate
from puzzle.pipeline.training import train_step, train_step_dual, train_step_dual_bulk
from puzzle.pipeline.inference import forward, forward_value
from puzzle.core.utils import apply_model_mpu
import puzzle.core.shadow as shadow

from puzzle_ray.ray.group import BaseRole
from puzzle_ray.ray.group import get_model, get_reward_model, get_megatron_optimizer, get_optimizer_param_scheduler, model_provider


class ExpDataset():
    def __init__(self, exp_data):
        self.exp_data = exp_data

    def __len__(self):
        return self.exp_data['prompts'].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.exp_data.items()}


@ray.remote(num_gpus=1)
class TimeSharedModelRayRole(BaseRole):

    def init_model_from_pretrained(self, *args, **kwargs):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["MAX_JOBS"] = "64" # ninja compiler jobs

        print(kwargs)

        self._init_megatron(extra_args_provider=None,
                            args_defaults=kwargs)

        args = get_args()

        args.global_rank = self._rank
        args.local_rank = self._local_rank

        # TODO(lkm): need to update
        args.per_device_generation_batch_size = args.global_batch_size // parallel_state.get_data_parallel_world_size()
        args.per_device_training_batch_size = args.global_batch_size // parallel_state.get_data_parallel_world_size()

        self.tokenizer = get_tokenizer()

        # Create datasets
        prompt_train_dataloader, num_total_iters = create_datasets(args=args, tokenizer=self.tokenizer, train_phase=3)
        if prompt_train_dataloader is not None:
            self.prompt_train_data_iterator = iter(prompt_train_dataloader)

        # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
        self.exp_mini_dataset = MiniDataset(args.generation_batches,
                                        args.per_device_training_batch_size)

        # Init models
        self.actor, self.actor_optimizer, self.actor_opt_param_scheduler = \
            self._init_actor(model_provider, model_type=ModelType.encoder_or_decoder)

        self.critic, self.critic_optimizer, self.critic_opt_param_scheduler = \
            self._init_critic(model_provider, model_type=ModelType.encoder_or_decoder)

        self.ref = self._init_ref(model_provider, model_type=ModelType.encoder_or_decoder)

        self.reward = self._init_reward(model_provider, model_type=ModelType.encoder_or_decoder)

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
                    lr_mult=1.0,
                    mpu=None):
        args = get_args()
        mpu = mpu or parallel_state.get_mpu_by_index(0)
        assert type(mpu) is parallel_state.ModelParallismUtils
        parallel_state.switch_mpu_by_index(mpu.get_index())

        model = get_model(model_provider_func, model_type)
        optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

        # load_hf_ckpt(model)

        set_model_mpu(model, mpu)

        if args.use_shadow:
            # shadow
            if torch.distributed.get_rank() == 0:
                print(f"begin init shadow model, {torch.cuda.memory_allocated()}")
            shadow.init_shadow_model(
                                model,
                                mpu=parallel_state.initialize_model_parallel(args.shadow_tensor_model_parallel_size, args.shadow_pipeline_model_parallel_size),
                                shadow_model_provider=lambda: get_model(model_provider_func, model_type, wrap_with_ddp=False))
            if torch.distributed.get_rank() == 0:
                print(f"after init shadow model, {torch.cuda.memory_allocated()}")

        return model, optimizer, opt_param_scheduler

    def _init_ref(self, model_provider_func, model_type):
        model = get_model(model_provider_func, model_type, wrap_with_ddp=False)
        # load_hf_ckpt(model)
        return model

    def _init_critic(self, model_provider_func, model_type,
                    no_wd_decay_cond=None,
                    scale_lr_cond=None,
                    lr_mult=1.0):
        model = get_reward_model(model_provider_func, model_type)
        optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        return model, optimizer, opt_param_scheduler

    def _init_reward(self, model_provider_func, model_type):
        model = get_reward_model(model_provider_func, model_type, wrap_with_ddp=False)
        return model

    #

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def get_batch_prompt(self):
        batch_prompt = None
        if parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.is_pipeline_first_stage():
            batch_prompt = next(self.prompt_train_data_iterator)
        return batch_prompt

    def generate_sequences(self, batch_prompt):
        self.eval()

        # to device
        for k in batch_prompt.keys():
            batch_prompt[k] = batch_prompt[k].cuda()

        prompt = batch_prompt['prompt']
        attention_maske = batch_prompt['prompt_att_mask']

        # only first pipeline stage would contain generated sequence
        seq, logprobs = self._generate_sequence(prompt, attention_maske)

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
        if args.use_dis_bubble_generation and parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if args.use_shadow:
                with shadow.ApplyShadowModel(self.actor) as shadow_model:
                    seq, output_log_probs = dis_bubble_generate(shadow_model, prompts, mask, max_length=args.max_answer_seq_len, return_output_log_probs=True)
            else:
                seq, output_log_probs = dis_bubble_generate(self.actor, prompts, mask, max_length=args.max_answer_seq_len, return_output_log_probs=True)
        else:
            seq, output_log_probs = generate(self.actor, prompts, mask, max_length=args.max_answer_seq_len, return_output_log_probs=True)
        torch.cuda.synchronize()
        ed = time.time()
        self.generate_seq_time = ed - st
        # print(f"Generate sequence time: {self.generate_seq_time}")
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

    def forward(self, seq, attention_mask, batch_size, seq_length, return_output_log_probs=True):
        self.eval()

        seq = seq.cuda()
        attention_mask = attention_mask.cuda()

        with torch.no_grad():
            ref_logprobs = forward(self.ref, seq, attention_mask, batch_size, seq_length, return_output_log_probs=True)

        if parallel_state.is_pipeline_first_stage():
            ref_logprobs = ref_logprobs.cpu()
        return ref_logprobs

    def forward_value_reward(self, seq, attention_mask, batch_size, prompt_length, seq_length, PAD_ID):
        self.eval()

        seq = seq.cuda()
        attention_mask = attention_mask.cuda()
        with torch.no_grad():
            reward_score = forward_value(self.reward,
                                         seq, attention_mask,
                                         batch_size=batch_size,
                                         prompt_length=prompt_length,
                                         seq_length=seq_length,
                                         PAD_ID=PAD_ID)['chosen_end_scores']
        if parallel_state.is_pipeline_first_stage():
            reward_score = reward_score.cpu()
        return reward_score

    def forward_value(self, seq, attention_mask, batch_size, prompt_length, seq_length):
        self.eval()

        seq = seq.cuda()
        attention_mask = attention_mask.cuda()
        with torch.no_grad():
            values = forward_value(self.critic,
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

    def add_exp_dataset(self, batch_exp):
        self.exp_dataset = self.exp_mini_dataset.add(batch_exp)
        num_total_exp_iters = len(self.exp_dataset)
        return num_total_exp_iters

    def training_step(self, step):
        self.train()
        args = get_args()
        if not args.bulk_switch_on:
            actor_loss_dict, critic_loss_dict = self.training_step_sequence(step)
        else:
            actor_loss_dict, critic_loss_dict = self.training_step_bulk(step)
        def to_cpu(d):
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    d[k] = v.cpu()
            return d
        return to_cpu(actor_loss_dict), to_cpu(critic_loss_dict)

    def training_step_sequence(self, step):
        args = get_args()

        train_data_iterator = None
        if parallel_state.get_tensor_model_parallel_rank() == 0:
            exp_data = self.exp_dataset[step]
            dataloader = torch.utils.data.DataLoader(ExpDataset(exp_data), batch_size=args.micro_batch_size, shuffle=True)
            train_data_iterator = iter(dataloader)

        actor_loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(self.forward_actor_step,
                        self.actor,
                        self.actor_optimizer,
                        self.actor_opt_param_scheduler,
                        train_data_iterator)

        if parallel_state.get_tensor_model_parallel_rank() == 0:
                train_data_iterator = iter(dataloader)

        critic_loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(self.forward_critic_step,
                        self.critic,
                        self.critic_optimizer,
                        self.critic_opt_param_scheduler,
                        train_data_iterator)
        return actor_loss_dict, critic_loss_dict

    def training_step_bulk(self, step):
        args = get_args()

        train_data_iterator = None
        train_data_iterator_1 = []
        train_data_iterator_2 = []
        if parallel_state.get_tensor_model_parallel_rank() == 0:
            exp_data = self.exp_dataset[step]
            dataloader = torch.utils.data.DataLoader(ExpDataset(exp_data), batch_size=args.micro_batch_size, shuffle=True)
            train_data_iterator_1.append(iter(dataloader))
            dataloader = torch.utils.data.DataLoader(ExpDataset(exp_data), batch_size=args.micro_batch_size, shuffle=True)
            train_data_iterator_2.append(iter(dataloader))
        else:
            train_data_iterator_1.append(None)
            train_data_iterator_2.append(None)

        actor_loss_dict, critic_loss_dict, _, _, _, _, _, _ = \
            train_step_dual_bulk(self.forward_actor_step,
                            self.forward_critic_step,
                            self.actor,
                            self.critic,
                            self.actor_optimizer,
                            self.critic_optimizer,
                            self.actor_opt_param_scheduler,
                            self.critic_opt_param_scheduler,
                            train_data_iterator_1,
                            train_data_iterator_2)

        return actor_loss_dict, critic_loss_dict

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
        # if torch.isnan(prompts).all():
        #     print(f"[compute_rewards()] prompts is nan")
        # if torch.isnan(log_probs).all():
        #     print(f"[compute_rewards()] log_probs is nan")
        # # ref_log_probs is nan
        # if torch.isnan(ref_log_probs).all():
        #     print(f"[compute_rewards()] ref_log_probs is nan")
        # if torch.isnan(reward_score).all():
        #     print(f"[compute_rewards()] reward_score is nan")
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

    #

    def get_tensor_model_parallel_size(self):
        return parallel_state.get_tensor_model_parallel_world_size()

    def get_pipeline_model_parallel_size(self):
        return parallel_state.get_pipeline_model_parallel_world_size()

    def get_data_paralle_rank(self):
        return parallel_state.get_data_parallel_rank()

    def get_data_parallel_size(self):
        return parallel_state.get_data_parallel_world_size()

    def get_generate_seq_time(self):
        return self.generate_seq_time

    def train(self):
        def _train(model):
            for module in model:
                module.train()
        _train(self.actor)
        _train(self.critic)

    def eval(self):
        def _eval(model):
            for module in model:
                module.eval()
        _eval(self.actor)
        _eval(self.ref)
        _eval(self.critic)
        _eval(self.reward)
