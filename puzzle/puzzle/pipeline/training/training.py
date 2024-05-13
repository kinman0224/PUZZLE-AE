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

def train_step_dual(forward_step_func_1, forward_step_func_2, model_1, model_2, optimizer_1, optimizer_2, opt_param_scheduler_1, opt_param_scheduler_2, data_1, data_2, print_grad=False):
    """Single training step. A simple implementation for pipestream(fake)"""
    args = get_args()
    config_1 = get_attr_wrapped_model(model_1[0], 'config')
    config_2 = get_attr_wrapped_model(model_2[0], 'config')
    timers = get_timers()

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in model_1:
            partition.zero_grad_buffer()
        for partition in model_2:
            partition.zero_grad_buffer()
    optimizer_1.zero_grad()
    optimizer_2.zero_grad()

    forward_backward_func = get_forward_backward_func(dual=True, bulk=False)

    losses_reduced_1, losses_reduced_2 = forward_backward_func(
        forward_step_func_1=forward_step_func_1,
        forward_step_func_2=forward_step_func_2,
        data_iterator_1=data_1,
        data_iterator_2=data_2,
        model_1=model_1,
        model_2=model_2,
        num_microbatches=get_num_microbatches(),
        dtype_1=args.params_dtype,
        dtype_2=args.params_dtype,
        tensor_shape_1=(args.seq_length, args.micro_batch_size, config_1.hidden_size),
        tensor_shape_2=(args.seq_length, args.micro_batch_size, config_2.hidden_size),
    )

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Reduce gradients.
    optimizer_1.reduce_model_grads(args, timers)
    optimizer_2.reduce_model_grads(args, timers)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful_1, grad_norm_1, num_zeros_in_grad_1 = optimizer_1.step(args, timers)
    update_successful_2, grad_norm_2, num_zeros_in_grad_2 = optimizer_2.step(args, timers)
    timers('optimizer').stop()

    # Gather params.
    if update_successful_1:
        optimizer_1.gather_model_params(args, timers)
    if update_successful_2:
        optimizer_2.gather_model_params(args, timers)

    # # Vision momentum.
    # if args.vision_pretraining and args.vision_pretraining_type == "dino":
    #     unwrapped_model = unwrap_model(model[0],
    #                                    (torchDDP, LocalDDP, Float16Module))
    #     unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful_1:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler_1.step(increment=increment)
        skipped_iter_1 = 0
    else:
        skipped_iter_1 = 1

    if update_successful_2:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler_2.step(increment=increment)
        skipped_iter_2 = 0
    else:
        skipped_iter_2 = 1

     # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced_1 = {}
        for key in losses_reduced_1[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced_1]
            loss_reduced_1[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        loss_reduced_2 = {}
        for key in losses_reduced_2[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced_2]
            loss_reduced_2[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced_1, loss_reduced_2, skipped_iter_1, skipped_iter_2, grad_norm_1, num_zeros_in_grad_1, grad_norm_2, num_zeros_in_grad_2
    return {}, {}, skipped_iter_1, skipped_iter_2, grad_norm_1, num_zeros_in_grad_1, grad_norm_2, num_zeros_in_grad_2

def train_step_dual_bulk(forward_step_func_1, forward_step_func_2, model_1, model_2, optimizer_1, optimizer_2, opt_param_scheduler_1, opt_param_scheduler_2, data_1, data_2, print_grad=False):
    """Single training step. A simple implementation for pipestream"""

    args = get_args()
    config_1 = get_attr_wrapped_model(model_1[0], 'config')
    config_2 = get_attr_wrapped_model(model_2[0], 'config')
    timers = get_timers()

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in model_1:
            partition.zero_grad_buffer()
        for partition in model_2:
            partition.zero_grad_buffer()
    optimizer_1.zero_grad()
    optimizer_2.zero_grad()

    forward_backward_func = get_forward_backward_func(dual=True, bulk=True)

    losses_reduced_1, losses_reduced_2 = forward_backward_func(
        forward_step_func_1=forward_step_func_1,
        forward_step_func_2=forward_step_func_2,
        data_iterator_1=data_1,
        data_iterator_2=data_2,
        model_1=model_1,
        model_2=model_2,
        optimizer_1=optimizer_1,
        optimizer_2=optimizer_2,
        opt_param_scheduler_1=opt_param_scheduler_1,
        opt_param_scheduler_2=opt_param_scheduler_2,
        num_microbatches=get_num_microbatches(),
        dtype_1=args.params_dtype,
        dtype_2=args.params_dtype,
        tensor_shape_1=(args.seq_length, args.micro_batch_size, config_1.hidden_size),
        tensor_shape_2=(args.seq_length, args.micro_batch_size, config_2.hidden_size),
        args=args,
        timers=timers,
    )

    # # Reduce gradients.
    # optimizer.reduce_model_grads(args, timers)
    # update_successful, _, _ = optimizer.step(args, timers)

    # # Gather params.
    # if update_successful:
    #     increment = get_num_microbatches() * \
    #                 args.micro_batch_size * \
    #                 args.data_parallel_size
    #     opt_param_scheduler_1.step(increment=increment)

     # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced_1 = {}
        for key in losses_reduced_1[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced_1]
            loss_reduced_1[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        loss_reduced_2 = {}
        for key in losses_reduced_2[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced_2]
            loss_reduced_2[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced_1, loss_reduced_2, None, None, None, None, None, None
    return {}, {}, None, None, None, None, None, None
