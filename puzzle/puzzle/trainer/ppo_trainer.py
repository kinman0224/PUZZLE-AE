from functools import partial
import torch
import time

from megatron import get_args, get_timers
from megatron.core import parallel_state, tensor_parallel
from megatron.utils import average_losses_across_data_parallel_group

from puzzle.utils.utils import gather_log_probs
from puzzle.pipeline.generation import generate, dis_bubble_generate
from puzzle.pipeline.inference import forward, forward_value
from puzzle.pipeline.training import train_step, train_step_dual, train_step_dual_bulk

class ExpDataset():
    def __init__(self, exp_data):
        self.exp_data = exp_data

    def __len__(self):
        return self.exp_data['prompts'].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.exp_data.items()}


class PPOTrainer():
    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine

        self.actor_model = self.rlhf_engine.actor
        self.actor_optimizer = self.rlhf_engine.actor_optimizer
        self.actor_opt_param_scheduler = self.rlhf_engine.actor_opt_param_scheduler
        # self.actor_skeleton_model = self.rlhf_engine.actor_skeleton_model

        self.ref_model = self.rlhf_engine.ref

        self.critic_model = self.rlhf_engine.critic
        self.critic_optimizer = self.rlhf_engine.critic_optimizer
        self.critic_opt_param_scheduler = self.rlhf_engine.critic_opt_param_scheduler

        self.reward_model = self.rlhf_engine.reward

        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_prompt_seq_len = args.max_prompt_seq_len
        self.max_answer_seq_len = args.max_answer_seq_len

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

        # perf
        self.generate_time = 0.0

    def _generate_sequence(self, prompts, mask):
        args = get_args()

        timers = get_timers()
        timers('generate-seq', log_level=0).start(barrier=True)
        torch.cuda.synchronize()
        st = time.time()
        if args.use_dis_bubble_generation:
            seq, output_log_probs = dis_bubble_generate(self.actor_model, prompts, mask, max_length=self.max_answer_seq_len, return_output_log_probs=True)
        else:
            seq, output_log_probs = generate(self.actor_model, prompts, mask, max_length=self.max_answer_seq_len, return_output_log_probs=True)
        torch.cuda.synchronize()
        ed = time.time()
        self.generate_time = ed - st
        # print(f"Generate time: {self.generate_time}")
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

    def generate_experience(self, prompts, mask, step):
        args = get_args()
        self.eval()

        # only first pipeline stage would contain generated sequence
        seq, logprobs = self._generate_sequence(prompts, mask)

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long() if seq is not None else None

        batch_size = prompts.shape[0]
        prompt_length = prompts.shape[1]
        seq_length = prompts.shape[1] + self.max_answer_seq_len

        with torch.no_grad():
            ref_logprobs = forward(self.ref_model, seq, attention_mask, batch_size, seq_length, return_output_log_probs=True)
            reward_score = forward_value(self.reward_model,
                                                seq,
                                                attention_mask,
                                                batch_size=batch_size,
                                                prompt_length=prompt_length,
                                                seq_length=seq_length,
                                                PAD_ID=self.tokenizer.pad_token_id)['chosen_end_scores']
            values = forward_value(self.critic_model,
                                    seq,
                                    attention_mask,
                                    batch_size=batch_size,
                                    prompt_length=prompt_length,
                                    seq_length=seq_length,
                                    return_value_only=True)

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        ret = None
        if parallel_state.is_pipeline_first_stage():
            # only first pipeline stage would return whold experience
            ret = {
                'prompts': prompts,
                'logprobs': logprobs,
                'ref_logprobs': ref_logprobs,
                'value': values.detach()[:, :-1],
                'rewards': reward_score.detach(),
                'input_ids': seq,
                "attention_mask": attention_mask
            }
        return ret

    def train_rlhf(self, exp_data=None, merge_pipeline=False):
        self.train()

        args = get_args()

        train_data_iterator = None
        train_data_iterator_1 = None
        train_data_iterator_2 = None
        if parallel_state.get_tensor_model_parallel_rank() == 0:
            dataloader = torch.utils.data.DataLoader(ExpDataset(exp_data), batch_size=args.micro_batch_size, shuffle=True)
            train_data_iterator_1 = iter(dataloader)
            train_data_iterator_2 = iter(dataloader)
            train_data_iterator = iter(dataloader)

        timers = get_timers()
        if merge_pipeline:
            timers('train-actor-critic', log_level=0).start(barrier=True)
            actor_loss_dict, critic_loss_dict, _, _, _, _, _, _ = \
                train_step_dual(self.forward_actor_step,
                                self.forward_critic_step,
                                self.actor_model,
                                self.critic_model,
                                self.actor_optimizer,
                                self.critic_optimizer,
                                self.actor_opt_param_scheduler,
                                self.critic_opt_param_scheduler,
                                train_data_iterator_1,
                                train_data_iterator_2)
            timers('train-actor-critic').stop(barrier=True)
        else:
            timers('train-actor', log_level=0).start(barrier=True)
            actor_loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(self.forward_actor_step,
                        self.actor_model,
                        self.actor_optimizer,
                        self.actor_opt_param_scheduler,
                        train_data_iterator)
            timers('train-actor').stop(barrier=True)

            if parallel_state.get_tensor_model_parallel_rank() == 0:
                train_data_iterator = iter(dataloader)

            timers('train-critic', log_level=0).start(barrier=True)
            critic_loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(self.forward_critic_step,
                        self.critic_model,
                        self.critic_optimizer,
                        self.critic_opt_param_scheduler,
                        train_data_iterator)
            timers('train-critic').stop(barrier=True)

        return actor_loss_dict, critic_loss_dict


    def train_rlhf_bulk(self, exp_data=None):
        self.train()

        args = get_args()

        train_data_iterator = None
        train_data_iterator_1 = []
        train_data_iterator_2 = []
        if parallel_state.get_tensor_model_parallel_rank() == 0:
            dataloader = torch.utils.data.DataLoader(ExpDataset(exp_data), batch_size=args.micro_batch_size, shuffle=True)
            train_data_iterator_1.append(iter(dataloader))
            dataloader = torch.utils.data.DataLoader(ExpDataset(exp_data), batch_size=args.micro_batch_size, shuffle=True)
            train_data_iterator_2.append(iter(dataloader))
        else:
            train_data_iterator_1.append(None)
            train_data_iterator_2.append(None)
        timers = get_timers()

        timers('train-actor-critic', log_level=0).start(barrier=True)
        actor_loss_dict, critic_loss_dict, _, _, _, _, _, _ = \
            train_step_dual_bulk(self.forward_actor_step,
                            self.forward_critic_step,
                            self.actor_model,
                            self.critic_model,
                            self.actor_optimizer,
                            self.critic_optimizer,
                            self.actor_opt_param_scheduler,
                            self.critic_opt_param_scheduler,
                            train_data_iterator_1,
                            train_data_iterator_2)
        timers('train-actor-critic').stop(barrier=True)

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

    def train(self):
        def _train(model):
            for module in model:
                module.train()
        _train(self.actor_model)
        _train(self.critic_model)

    def eval(self):
        def _eval(model):
            for module in model:
                module.eval()
        _eval(self.actor_model)
        _eval(self.ref_model)
        _eval(self.reward_model)
        _eval(self.critic_model)
