import torch

from megatron.core import parallel_state

def _is_cuda(tensor):
    """Check if a tensor is not none and is cuda."""
    assert tensor is not None
    assert tensor.is_cuda

def _is_cuda_contiguous(tensor):
    """Check if a tensor is not none, is cuda, and is contiguous."""
    _is_cuda(tensor)
    assert tensor.is_contiguous()

def copy_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
    """Broadcast tensor values from last stage into the first stage."""

    is_last_stage = parallel_state.is_pipeline_last_stage()
    is_first_stage = parallel_state.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if is_first_stage and is_last_stage:
        return tensor
    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        if is_last_stage:
            _is_cuda(tensor)
            is_contiguous = tensor.is_contiguous()
            if not is_contiguous:
                tensor = tensor.contiguous()
        else:
            tensor = torch.empty(size,
                                 dtype=dtype,
                                 device=torch.cuda.current_device())
        src = parallel_state.get_pipeline_model_parallel_last_rank()
        group = parallel_state.get_embedding_group()
        # Broadcast from first stage into the last stage.
        torch.distributed.broadcast(tensor, src, group)
    else:
        tensor = None

    return tensor

def copy_from_first_to_last_pipeline_stage(size, dtype, tensor=None):
    """Broadcast tensor values from first stage into the last stage."""

    is_last_stage = parallel_state.is_pipeline_last_stage()
    is_first_stage = parallel_state.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if is_first_stage and is_last_stage:
        return tensor
    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        if is_first_stage:
            _is_cuda_contiguous(tensor)
        else:
            tensor = torch.empty(size,
                                 dtype=dtype,
                                 device=torch.cuda.current_device())
        src = parallel_state.get_pipeline_model_parallel_first_rank()
        group = parallel_state.get_embedding_group()
        # Broadcast from first stage into the last stage.
        torch.distributed.broadcast(tensor, src, group)
    else:
        tensor = None

    return tensor

def broadcast_from_first_pipeline_stage(size, dtype, tensor=None):
    """Broadcast tensor values from last stage into the first stage."""

    is_first_stage = parallel_state.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if parallel_state.is_pipeline_last_stage() and is_first_stage:
        return tensor

    if is_first_stage:
        _is_cuda_contiguous(tensor)
    else:
        tensor = torch.empty(size,
                             dtype=dtype,
                             device=torch.cuda.current_device())
    # Get the group and corresponding source rank.
    src = parallel_state.get_pipeline_model_parallel_first_rank()
    group = parallel_state.get_pipeline_model_parallel_group()
    torch.distributed.broadcast(tensor, src, group)
    return tensor
