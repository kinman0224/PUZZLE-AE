import torch
import torch.distributed as dist
import torch.nn.functional as F

import copy

from megatron import get_args
from megatron.core import tensor_parallel, parallel_state
from megatron.text_generation.sampling import sample
from megatron.core.utils import get_attr_wrapped_model

from .forward_step import InferenceParams
from puzzle.pipeline.utils.communication import copy_from_last_to_first_pipeline_stage
from puzzle.pipeline.generation import p2p_communication
from puzzle.core.utils import apply_model_mpu

def get_model_config(model):
    if type(model) is list:
        model = model[0]
    return get_attr_wrapped_model(model, 'config')

def dis_bubble_generate(model, tokens, attention_mask, max_length=256,
             return_output_log_probs=False,
             top_k=0, top_p=0.0, top_p_decay=0.0, top_p_bound=0.0,
             temperature=1.0):
    args = get_args()
    config = get_model_config(model)

    # apply_model_mpu(model)

    if type(model) is list:
        model = model[0]

    batch_size = tokens.size(0)
    min_prompt_length = tokens.size(1)
    max_sequence_length = tokens.size(1) + max_length

    # Pad tokens to max_sequence_length
    tokens = torch.cat([tokens, torch.ones(batch_size, max_length, dtype=tokens.dtype).to(torch.cuda.current_device())], dim=-1)
    attention_mask = torch.cat([attention_mask, torch.ones(batch_size, max_length, dtype=attention_mask.dtype).to(torch.cuda.current_device())], dim=-1)

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1)
    if parallel_state.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = torch.zeros(output_log_probs_size,
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())

    inference_params = InferenceParams(batch_size, max_sequence_length)

    def forward_step_helper(input_tensor, tokens, attention_mask, inference_params=None):
        set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
        set_input_tensor(input_tensor)
        output_tensor = model(tokens, None, attention_mask,
                          inference_params=inference_params)
        return output_tensor

    pf_stage_micro_batch_size = args.pf_stage_mbs
    ar_stage_micro_batch_size = args.ar_stage_mbs

    assert pf_stage_micro_batch_size <= ar_stage_micro_batch_size
    assert ar_stage_micro_batch_size <= batch_size

    pf_stage_num = batch_size // pf_stage_micro_batch_size
    ar_stage_num = batch_size // ar_stage_micro_batch_size

    first_stage_counter = 0
    first_stage = 0
    pf_buffer = [None] * pf_stage_num
    ar_buffer = [None] * ar_stage_num

    with torch.no_grad():
        prev_context_length = 0
        input_tensors = pf_buffer

        for context_length in range(min_prompt_length, max_sequence_length):
            input_shape = (batch_size, context_length-prev_context_length)
            attention_mask2use = attention_mask[:, :context_length]
            tokens2use = None

            if context_length == min_prompt_length:
                micro_batch_size = pf_stage_micro_batch_size
            else:
                micro_batch_size = ar_stage_micro_batch_size
            num_micro_batches, last_chunk = divmod(batch_size, micro_batch_size)

            # define recv tensor shapes
            recv_tensor_shapes = None
            if parallel_state.is_pipeline_first_stage():
                recv_dtype = torch.int64
            else:
                recv_dtype = torch.float16
                if context_length == min_prompt_length:
                    recv_tensor_shapes = (min_prompt_length, pf_stage_micro_batch_size, config.hidden_size)
                else:
                    recv_tensor_shapes = (1, ar_stage_micro_batch_size, config.hidden_size)

            # For non first stage
            input_tensor = p2p_communication.recv_forward(recv_tensor_shapes, dtype=recv_dtype)

            for k in range(0, num_micro_batches):
                # For first stage to receive tokens from last stage
                if parallel_state.is_pipeline_first_stage():
                    if not (context_length == min_prompt_length and k < parallel_state.get_pipeline_model_parallel_world_size()):
                        if first_stage == 0:
                            recv_tensor_shapes = (pf_stage_micro_batch_size, 1)
                        else:
                            recv_tensor_shapes = (ar_stage_micro_batch_size, 1)
                        input_tensors[first_stage_counter] = p2p_communication.send_forward_recv_forward(None, True, recv_tensor_shapes, dtype=recv_dtype)
                        first_stage_counter += 1
                        if context_length == min_prompt_length + 1 and \
                                pf_stage_num - max(pf_stage_num - parallel_state.get_pipeline_model_parallel_world_size(), 0) > 0:
                            for _ in range((pf_stage_num - max(pf_stage_num - parallel_state.get_pipeline_model_parallel_world_size(), 0)) // ar_stage_num - 1):
                                input_tensors[first_stage_counter] = p2p_communication.send_forward_recv_forward(None, True, recv_tensor_shapes, dtype=recv_dtype)
                                first_stage_counter += 1

                        # switch stage and buffer
                        if first_stage == 0 and first_stage_counter == pf_stage_num:
                            first_stage = 1
                            first_stage_counter = 0
                            input_tensors = ar_buffer
                        elif first_stage == 1 and first_stage_counter == ar_stage_num:
                            first_stage_counter = 0

                start = k * micro_batch_size
                end = min(start + micro_batch_size, batch_size)
                this_micro_batch_size = end - start

                if parallel_state.is_pipeline_first_stage():
                    if context_length == min_prompt_length:
                        tokens2use = tokens[start:end, :context_length]
                    elif context_length == min_prompt_length + 1:
                        offset = ar_stage_micro_batch_size // pf_stage_micro_batch_size
                        tokens2use = torch.cat(pf_buffer[offset * k:offset * (k+1)], dim=0)
                        tokens[k*ar_stage_micro_batch_size : \
                               (k+1)*ar_stage_micro_batch_size, context_length] = tokens2use.squeeze(-1)
                    else:
                        tokens2use = ar_buffer[k]
                        tokens[k*ar_stage_micro_batch_size : \
                               (k+1)*ar_stage_micro_batch_size, context_length] = tokens2use.squeeze(-1)

                output_tensor = forward_step_helper(input_tensor,
                                    tokens2use,
                                    attention_mask2use[start:end,...] if attention_mask2use is not None else None,
                                    inference_params)

                # sample
                if parallel_state.is_pipeline_last_stage():
                    output_tensor = output_tensor.contiguous()
                    last_token_logits = output_tensor[:, -1, :]
                    new_sample = sample(last_token_logits,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            vocab_size=args.padded_vocab_size)
                    if top_p > 0.0 and top_p_decay > 0.0:
                        top_p = top_p * top_p_decay
                        if top_p_bound > 0.0:
                            top_p = max(top_p, top_p_bound)

                    # Update the tokens.
                    tokens[k*micro_batch_size:(k+1)*micro_batch_size, context_length] = new_sample

                    if return_output_log_probs:
                        log_probs = F.log_softmax(output_tensor, dim=2)
                        # Pick the tokens that we need to get the log
                        # probabilities for. Note that next input token is
                        # the token which we selected in the current logits,
                        # so shift by 1.
                        indices = torch.unsqueeze(
                            tokens[
                                k*micro_batch_size:(k+1)*micro_batch_size,
                                (prev_context_length + 1):(context_length + 1)],
                            2)
                        output_log_probs[k*micro_batch_size:(k+1)*micro_batch_size,
                                            prev_context_length:context_length] = \
                            torch.gather(log_probs, 2, indices).squeeze(2)

                    output_tensor = new_sample.unsqueeze(-1)

                recv_prev = k < num_micro_batches - 1 and not parallel_state.is_pipeline_first_stage()
                # send forward
                input_tensor = p2p_communication.send_forward_recv_forward(output_tensor, recv_prev, recv_tensor_shapes, dtype=recv_dtype)

                if inference_params:
                    inference_params.batch_size_offset += this_micro_batch_size

            if inference_params:
                # Once we are done with all the micro-batches, we can
                # adjust the sequence length offset.
                inference_params.sequence_len_offset += (context_length-prev_context_length)
                # and reset the batch size offset
                inference_params.batch_size_offset = 0

            # Update the context length for the next token generation.
            prev_context_length = context_length

        # the remain tokens
        if parallel_state.is_pipeline_first_stage():
            for k in range(min(parallel_state.get_pipeline_model_parallel_world_size(), ar_stage_num)):
                input_tensors[first_stage_counter] = p2p_communication.send_forward_recv_forward(None, True, recv_tensor_shapes, dtype=recv_dtype)
                first_stage_counter += 1
                if first_stage == 1 and first_stage_counter == ar_stage_num:
                    first_stage_counter = 0
                tokens[k*ar_stage_micro_batch_size : \
                               (k+1)*ar_stage_micro_batch_size, context_length] = \
                                input_tensors[first_stage_counter].squeeze(-1)

    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================
    tokens = tokens[:, :(context_length + 1)]

    # ===================================================
    # Copy log_probs to the first pipeline stage.
    # ===================================================
    if return_output_log_probs:
        output_log_probs = copy_from_last_to_first_pipeline_stage(output_log_probs_size, torch.float32, output_log_probs)

    return tokens, output_log_probs