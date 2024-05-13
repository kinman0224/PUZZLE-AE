# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Forward step utilities."""

from collections.abc import Iterable

import torch

from megatron import get_args
from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model
from megatron.core.pipeline_parallel import p2p_communication

class ForwardValueStep:
    """Forward step function with all the communications.
    We use a class here to hide the inference parameters
    from the outside caller."""

    def __init__(self, model, batch_size, seq_length):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        assert not isinstance(model, Iterable), \
            'interleaving schedule is not supported for inference'
        model.eval()
        self.model = model
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Pipelining arguments.
        args = get_args()
        self.pipeline_size_larger_than_one = (
            args.pipeline_model_parallel_size > 1)
        # Threshold of pipelining.
        self.pipelining_batch_x_seqlen = \
            args.inference_batch_times_seqlen_threshold


    def __call__(self, tokens, position_ids, attention_mask):
        """Invocation of the forward methods."""
        # Pipelining case.
        if self.pipeline_size_larger_than_one:
            current_batch_x_seqlen = self.batch_size * self.seq_length
            if current_batch_x_seqlen >= self.pipelining_batch_x_seqlen:
                micro_batch_size = \
                    max(1, self.pipelining_batch_x_seqlen // self.seq_length)
                return _with_pipelining_forward_step(self.model,
                                                     tokens,
                                                     position_ids,
                                                     attention_mask,
                                                     self.batch_size,
                                                     self.seq_length,
                                                     micro_batch_size)

        return _no_pipelining_forward_step(self.model,
                                           tokens,
                                           position_ids,
                                           attention_mask,
                                           self.batch_size,
                                           self.seq_length)



def _get_recv_buffer_dtype(args):
    """Receive happens between the layers."""
    if args.fp32_residual_connection:
        return torch.float
    return args.params_dtype



def _allocate_recv_buffer(batch_size, sequence_length):
    """Receive happens between the layers with size [s, b, h]."""
    if parallel_state.is_pipeline_first_stage():
        return None
    args = get_args()
    recv_size = (sequence_length, batch_size, args.hidden_size)
    return torch.empty(recv_size,
                       dtype=_get_recv_buffer_dtype(args),
                       device=torch.cuda.current_device())



def _forward_step_helper(model, tokens, position_ids, attention_mask, batch_size, sequence_length, recv_buffer=None):
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    args = get_args()
    config = get_attr_wrapped_model(model, 'config')

    # Receive from previous stage.
    recv_tensor_shape = (sequence_length, batch_size, config.hidden_size)
    input_tensor = p2p_communication.recv_forward(recv_tensor_shape, _get_recv_buffer_dtype(args))

    # Forward pass through the model.
    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)
    output_tensor = model(tokens, position_ids, attention_mask)

    # Send output to the next stage.
    p2p_communication.send_forward(output_tensor)

    return output_tensor



def _no_pipelining_forward_step(model, tokens, position_ids, attention_mask, batch_size, sequence_length):
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    output_tensor = _forward_step_helper(model, tokens, position_ids,
                                         attention_mask, batch_size, sequence_length)

    logits = None
    if parallel_state.is_pipeline_last_stage():
        logits = output_tensor

    return logits



def _with_pipelining_forward_step(model, tokens, position_ids, attention_mask, batch_size, sequence_length, micro_batch_size):
    """No interleaving is supported."""

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size,
                                           micro_batch_size)
    if last_chunk > 0:
        num_micro_batches += 1

    # Preallocate memory for output logits.
    value = None
    if parallel_state.is_pipeline_last_stage():
        value = torch.empty(
            (batch_size, sequence_length),
            dtype=torch.float32, device=torch.cuda.current_device())

    # Preallocate recv buffer.
    recv_buffer = None

    for micro_batch_index in range(num_micro_batches):
        # Slice among the batch dimenion.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        tokens2use = tokens[start:end, ...] if tokens is not None else None
        attention_mask2use = attention_mask[start:end, ...]
        position_ids2use = None

        # Run a simple forward pass.
        if this_micro_batch_size != micro_batch_size:
            recv_buffer = None
        output = _forward_step_helper(model, tokens2use, position_ids2use,
                                      attention_mask2use, this_micro_batch_size, sequence_length, recv_buffer=recv_buffer)

        # Copy value.
        if parallel_state.is_pipeline_last_stage():
            value[start:end, ...] = output

    return value
