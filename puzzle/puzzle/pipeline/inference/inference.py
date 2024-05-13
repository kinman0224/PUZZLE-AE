import torch

from megatron import get_args
from megatron.core import parallel_state

from puzzle.utils.utils import gather_log_probs
from puzzle.pipeline.inference.forward_step import ForwardStep
from puzzle.pipeline.inference.forward_value_step import ForwardValueStep
from puzzle.pipeline.utils.communication import (
    copy_from_last_to_first_pipeline_stage,
    copy_from_first_to_last_pipeline_stage,
    broadcast_from_first_pipeline_stage
)

def forward(model, input_ids=None, attention_mask=None, batch_size=None, seq_length=None, return_output_log_probs=True):
    args = get_args()

    if type(model) is list:
        model = model[0]

    input_shape = (batch_size, seq_length)
    attention_mask = broadcast_from_first_pipeline_stage(input_shape, torch.int64, tensor=attention_mask)

    # forward step.
    forward_step = ForwardStep(model, batch_size, seq_length)

    # =============
    # Run infernece
    # =============
    with torch.no_grad():
        # logits will be meanigful only in the last pipeline stage.
        logits = forward_step(input_ids, None, attention_mask)

    # ======================================
    # Copy to the first pipeline stage.
    # ======================================
    ret = None
    if return_output_log_probs is True:
        input_ids = copy_from_first_to_last_pipeline_stage(input_shape, torch.int64, tensor=input_ids)
        output_log_probs = None
        if parallel_state.is_pipeline_last_stage():
            output_log_probs = gather_log_probs(logits[:, :-1, :], input_ids[:, 1:])
        log_probs_shape = (batch_size, seq_length-1)
        output_log_probs = copy_from_last_to_first_pipeline_stage(log_probs_shape, torch.float32, output_log_probs)
        ret = output_log_probs

    del logits
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    return ret

# Copied from DeepSpeedExamples.applications.DeepSpeed-Chat.training.utils.model.reward_model.py
def forward_value(model, input_ids=None, attention_mask=None, batch_size=None, prompt_length=None, seq_length=None, return_value_only=False, PAD_ID=None):
    args = get_args()

    if type(model) is list:
        model = model[0]

    input_shape = (batch_size, seq_length)
    attention_mask = broadcast_from_first_pipeline_stage(input_shape, torch.int64, tensor=attention_mask)

    # forward step.
    forward_value_step = ForwardValueStep(model, batch_size, seq_length)

    # ===================
    # Pre-allocate memory
    # ===================

    values_size = (batch_size, seq_length)

    # =============
    # Run infernece
    # =============
    with torch.no_grad():
        hidden_states = forward_value_step(input_ids, None, attention_mask)

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================
    values = copy_from_last_to_first_pipeline_stage(
        values_size, torch.float32, hidden_states)

    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    if return_value_only:
        return values
    else:
        # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
        chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
        if parallel_state.is_pipeline_first_stage():
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            chosen_end_scores = torch.stack(chosen_end_scores)

        return {
            "values": values,
            "chosen_end_scores": chosen_end_scores,
        }