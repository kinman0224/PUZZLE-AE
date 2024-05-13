from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron.core import tensor_parallel, parallel_state
from megatron.text_generation.sampling import sample

from puzzle.pipeline.utils.communication import copy_from_last_to_first_pipeline_stage
from .forward_step import ForwardStep

def generate(model, tokens, attention_mask, max_length=256,
             return_output_log_probs=False,
             top_k=0, top_p=0.0, top_p_decay=0.0, top_p_bound=0.0,
             temperature=1.0):
    args = get_args()

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
            output_log_probs = torch.empty(output_log_probs_size,
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())

    forward_step = ForwardStep(model, batch_size, max_sequence_length)

    with torch.no_grad():
        prev_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):

            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[:, :context_length]

            logits = forward_step(tokens2use, None, attention_mask2use)

            if parallel_state.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert logits is not None

                # Sample.
                last_token_logits = logits[:, -1, :]
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
                tokens[:, context_length] = new_sample

                # Calculate the log probabilities.
                if return_output_log_probs:
                    log_probs = F.log_softmax(logits, dim=2)
                    # Pick the tokens that we need to get the log
                    # probabilities for. Note that next input token is
                    # the token which we selected in the current logits,
                    # so shift by 1.
                    indices = torch.unsqueeze(
                        tokens[
                            :,
                            (prev_context_length + 1):(context_length + 1)],
                        2)
                    output_log_probs[:,
                                        prev_context_length:context_length] = \
                        torch.gather(log_probs, 2, indices).squeeze(2)

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                   tokens[:, context_length] if tokens is not None else None)

            # Update the context length for the next token generation.
            prev_context_length = context_length

    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================
    tokens = tokens[:, :(context_length + 1)]

    # ===================================================
    # Copy log_probs to the first pipeline stage.
    # ===================================================
    if return_output_log_probs:
        output_log_probs = copy_from_last_to_first_pipeline_stage(output_log_probs_size, torch.float32, output_log_probs)

    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    return tokens, output_log_probs