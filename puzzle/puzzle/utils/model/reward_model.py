import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.model.module import MegatronModule

class RewardWrapper(MegatronModule):

    def __init__(self, base_model, num_padding_at_beginning=0):
        super().__init__(share_word_embeddings=False)

        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning

        self.base_model = base_model

        if parallel_state.is_pipeline_last_stage():
            self.v_head = tensor_parallel.RowParallelLinear(
                self.config.hidden_size,
                1,
                bias=False,
                init_method=self.config.init_method).to(torch.cuda.current_device())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            return getattr(self.base_model, name)

    def __call__(self, *args, **kwargs):
        hidden_states = self.base_model.forward(*args, **kwargs)
        if parallel_state.is_pipeline_last_stage():
            hidden_states, _ = self.v_head(hidden_states)
            hidden_states = hidden_states.squeeze(-1) # (batch_size, seq_len)
        return hidden_states
