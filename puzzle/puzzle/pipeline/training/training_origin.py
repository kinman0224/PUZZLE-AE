import torch

from megatron import get_args, get_timers
from megatron import get_num_microbatches
from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model
from megatron.core.pipeline_parallel import get_forward_backward_func

def train_step(forward_step_func, model, optimizer, opt_param_scheduler, data, print_grad=False):
    """Single training step."""
    args = get_args()
    config = get_attr_wrapped_model(model[0], 'config')
    timers = get_timers()

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in model:
            partition.zero_grad_buffer()
    optimizer.zero_grad()

    forward_backward_func = get_forward_backward_func()

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data,
        model=model,
        num_microbatches=get_num_microbatches(),
        dtype=args.params_dtype,
        tensor_shape=(args.seq_length, args.micro_batch_size, config.hidden_size),
    )

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Reduce gradients.
    optimizer.reduce_model_grads(args, timers)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    timers('optimizer').stop()

    # Gather params.
    if update_successful:
        optimizer.gather_model_params(args, timers)

    # # Vision momentum.
    # if args.vision_pretraining and args.vision_pretraining_type == "dino":
    #     unwrapped_model = unwrap_model(model[0],
    #                                    (torchDDP, LocalDDP, Float16Module))
    #     unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

     # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad