# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Forward step utilities."""

from collections.abc import Iterable

import torch

from megatron import get_args
from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model
from megatron.core.pipeline_parallel import p2p_communication

class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    def __init__(self, max_batch_size, max_sequence_len):
        """Note that offsets are set to zero and we always set the
        flag to allocate memory. After the first call, make sure to
        set this flag to False."""
        self.max_sequence_len = max_sequence_len
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_idx):
        "swap between batches"
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")

        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            assert len(batch_idx) == inference_key_memory.shape[1] ## make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (
                    new_inference_key_memory, new_inference_value_memory)

class ForwardStep:
    """Forward step function with all the communications.
    We use a class here to hide the inference parameters
    from the outside caller."""

    def __init__(self, model, max_batch_size, max_sequence_len):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        assert not isinstance(model, Iterable), \
            'interleaving schedule is not supported for inference'
        model.eval()
        self.model = model
        # Initialize inference parameters.
        self.inference_params = InferenceParams(max_batch_size,
                                                max_sequence_len)
        # Pipelining arguments.
        args = get_args()
        self.pipeline_size_larger_than_one = (
            args.pipeline_model_parallel_size > 1)
        # Threshold of pipelining.
        self.pipelining_batch_x_seqlen = \
            args.inference_batch_times_seqlen_threshold


    def __call__(self, tokens, position_ids, attention_mask):
        """Invocation of the forward methods. Note that self.inference_params
        is being modified by the forward step."""
        # Pipelining case.
        if self.pipeline_size_larger_than_one:
            current_batch_x_seqlen = tokens.size(0) * tokens.size(1)
            if current_batch_x_seqlen >= self.pipelining_batch_x_seqlen:
                micro_batch_size = \
                    max(1, self.pipelining_batch_x_seqlen // tokens.size(1))
                return _with_pipelining_forward_step(self.model,
                                                     tokens,
                                                     position_ids,
                                                     attention_mask,
                                                     self.inference_params,
                                                     micro_batch_size)

        return _no_pipelining_forward_step(self.model,
                                           tokens,
                                           position_ids,
                                           attention_mask,
                                           self.inference_params)



def _get_recv_buffer_dtype(args):
    """Receive happens between the layers."""
    if args.fp32_residual_connection:
        return torch.float
    return args.params_dtype



def _forward_step_helper(model, tokens, position_ids, attention_mask,
                         inference_params, recv_buffer=None):
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    args = get_args()
    config = get_attr_wrapped_model(model, 'config')

    batch_size = tokens.size(0)
    sequence_length = tokens.size(1)

    # Receive from previous stage.
    recv_tensor_shape = (sequence_length, batch_size, config.hidden_size)
    input_tensor = p2p_communication.recv_forward(recv_tensor_shape, _get_recv_buffer_dtype(args))

    # Forward pass through the model.
    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)
    output_tensor = model(tokens, position_ids, attention_mask,
                          inference_params=inference_params)

    # Send output to the next stage.
    p2p_communication.send_forward(output_tensor)

    return output_tensor



def _no_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                inference_params, recv_buffer=None):
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    output_tensor = _forward_step_helper(model, tokens, position_ids,
                                         attention_mask, inference_params,
                                         recv_buffer=recv_buffer)
    # Update the sequence length offset.
    inference_params.sequence_len_offset += tokens.size(1)

    logits = None
    if parallel_state.is_pipeline_last_stage():
        logits = output_tensor

    return logits



def _with_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                  inference_params, micro_batch_size):
    """No interleaving is supported."""
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size,
                                           micro_batch_size)
    if last_chunk > 0:
        num_micro_batches += 1

    # Preallocate memory for output logits.
    logits = None
    if parallel_state.is_pipeline_last_stage():
        args = get_args()
        logits = torch.empty(
            (batch_size, sequence_length, args.padded_vocab_size),
            dtype=torch.float32, device=torch.cuda.current_device())

    # Preallocate recv buffer.
    recv_buffer = None

    for micro_batch_index in range(num_micro_batches):
        # Slice among the batch dimenion.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        tokens2use = tokens[start:end, ...]
        attention_mask2use = attention_mask[start:end, ...]
        position_ids2use = None

        # Run a simple forward pass.
        if this_micro_batch_size != micro_batch_size:
            recv_buffer = None
        output = _forward_step_helper(model, tokens2use, position_ids2use,
                                      attention_mask2use, inference_params,
                                      recv_buffer=recv_buffer)

        # Adjust the batch size offset to account for the micro-batch.
        inference_params.batch_size_offset += this_micro_batch_size

        # Copy logits.
        if parallel_state.is_pipeline_last_stage():
            logits[start:end, ...] = output

    # Once we are done with all the micro-batches, we can
    # adjust the sequence length offset.
    inference_params.sequence_len_offset += sequence_length
    # and reset the batch size offset
    inference_params.batch_size_offset = 0

    return logits
