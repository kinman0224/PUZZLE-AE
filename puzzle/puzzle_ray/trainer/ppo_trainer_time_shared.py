from typing import List, Dict, Tuple

import ray
import torch

class PPOTrainerTimeShared:
    #
    # A naive implementation of PPOTrainerTimeShared
    #

    def __init__(self, time_shared_models):
        self.time_shared_models = time_shared_models

        self.pad_token_id = ray.get(self.time_shared_models.async_run_method("get_pad_token_id"))[0]

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

    def generate_experience(self, step):
        batch_prompt_refs = self.time_shared_models.get_batch_prompt(step)
        batch_exp_refs = self.time_shared_models.generate_sequences_with_prompt(batch_prompt_refs)
        batch_exps: List[Dict[torch.Tensor]] = ray.get(batch_exp_refs)

        seqs, logprobs, prompts = list(zip(*batch_exps))
        attention_masks = [seq.not_equal(self.pad_token_id).long() for seq in seqs]
        batch_sizes = [seq.shape[0] for seq in seqs]
        seq_lengths = [seq.shape[1] for seq in seqs]
        prompt_lengths = [prompt.shape[1] for prompt in prompts]

        ref_logprobs_refs = self.time_shared_models.forward(seqs, attention_masks, batch_sizes, seq_lengths, True)
        reward_score_refs = self.time_shared_models.forward_value_reward(seqs, attention_masks, batch_sizes, prompt_lengths, seq_lengths, self.pad_token_id)
        values_refs = self.time_shared_models.forward_value(seqs, attention_masks, batch_sizes, prompt_lengths, seq_lengths)

        ref_logprobs = ray.get(ref_logprobs_refs)
        reward_score = ray.get(reward_score_refs)
        values = ray.get(values_refs)

        average_reward_list = []
        ret = []
        for i in range(len(batch_exps)):
            ret.append({
                'prompts': prompts[i],
                'logprobs': logprobs[i],
                'ref_logprobs': ref_logprobs[i],
                'value': values[i].detach()[:, :-1],
                'rewards': reward_score[i].detach(),
                'input_ids': seqs[i],
                'attention_mask': attention_masks[i],
            })
            average_reward_list.append(reward_score[i].mean())
        self.average_reward = sum(average_reward_list) / len(average_reward_list)

        # Single DP case
        # attention_mask = seq.not_equal(self.pad_token_id).long() if seq is not None else None
        # batch_size = seq.shape[0]
        # seq_length = seq.shape[1]
        # prompt_length = prompt.shape[1]

        # # allow the model to run in parallel
        # ref_logprobs_refs = self.time_shared_models.async_run_method("forward", seq, attention_mask, batch_size, seq_length, True)
        # reward_score_refs = self.time_shared_models.async_run_method("forward_value_reward", seq, attention_mask, batch_size, prompt_length, seq_length, self.pad_token_id)
        # values_refs = self.time_shared_models.async_run_method("forward_value", seq, attention_mask, batch_size, prompt_length, seq_length)

        # ref_logprobs = ray.get(ref_logprobs_refs)[0]
        # reward_score = ray.get(reward_score_refs)[0]
        # values = ray.get(values_refs)[0]

        # ret = {
        #     'prompts': prompt,
        #     'logprobs': logprobs,
        #     'ref_logprobs': ref_logprobs,
        #     'value': values.detach()[:, :-1],
        #     'rewards': reward_score.detach(),
        #     'input_ids': seq,
        #     'attention_mask': attention_mask,
        # }

        return ret

    @property
    def generate_seq_time(self):
        return ray.get(self.time_shared_models.async_run_method("get_generate_seq_time"))[0]

    def add_exp_dataset(self, batch_exp: List[Dict]):
        refs = []
        # Single DP case
        # refs.extend(self.time_shared_models.async_run_method("add_exp_dataset", batch_exp[0]))
        refs.extend(self.time_shared_models.add_exp_dataset(batch_exp))
        num_total_exp_iters = ray.get(refs)[0]
        return num_total_exp_iters

    def train_rlhf(self, step):
        actor_loss_dict, critic_loss_dict = ray.get(self.time_shared_models.async_run_method("training_step", step))[-1]
        return actor_loss_dict, critic_loss_dict
