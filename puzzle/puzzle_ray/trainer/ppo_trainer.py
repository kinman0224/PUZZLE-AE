import ray

class PPOTrainer:
    #
    # A naive implementation of PPOTrainer
    #

    def __init__(self, actor_models, critic_models, reward_models, ref_models):
        self.actor_models = actor_models
        self.critic_models = critic_models
        self.reward_models = reward_models
        self.ref_models = ref_models

        self.pad_token_id = ray.get(self.actor_models.async_run_method("get_pad_token_id"))[0]

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

    def generate_experience(self, step):
        seq, logprobs, prompt = ray.get(self.actor_models.generate_sequences(step))[0]

        attention_mask = seq.not_equal(self.pad_token_id).long() if seq is not None else None

        batch_size = seq.shape[0]
        seq_length = seq.shape[1]
        prompt_length = prompt.shape[1]

        ref_logprobs = ray.get(self.ref_models.async_run_method("forward", seq, attention_mask, batch_size, seq_length, True))[0]
        reward_score = ray.get(self.reward_models.async_run_method("forward_value", seq, attention_mask, batch_size, prompt_length, seq_length, self.pad_token_id))[0]
        values = ray.get(self.critic_models.async_run_method("forward_value", seq, attention_mask, batch_size, prompt_length, seq_length))[0]

        ret = {
            'prompts': prompt,
            'logprobs': logprobs,
            'ref_logprobs': ref_logprobs,
            'value': values.detach()[:, :-1],
            'rewards': reward_score.detach(),
            'input_ids': seq,
            'attention_mask': attention_mask,
        }
        return ret

    def add_exp_dataset(self, batch_exp):
        refs = []
        refs.extend(self.actor_models.async_run_method("add_exp_dataset", batch_exp))
        refs.extend(self.critic_models.async_run_method("add_exp_dataset", batch_exp))
        num_total_exp_iters = ray.get(refs)[0]
        return num_total_exp_iters

    def train_rlhf(self, step):
        actor_loss_dict = ray.get(self.actor_models.async_run_method("training_step", step))[-1]
        critic_loss_dict = ray.get(self.critic_models.async_run_method("training_step", step))[-1]

        ray.get(self.actor_models.async_run_method("empty_cache"))
        return actor_loss_dict, critic_loss_dict
