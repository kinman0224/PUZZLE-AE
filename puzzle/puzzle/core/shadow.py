import re
import types
from enum import Enum
import time

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model

from puzzle.core.utils import set_model_mpu, get_model_mpu_index

# ============================================== #
# Experimential
# ============================================== #

class ShadowModuleStatus(Enum):
    RELEASE = 0
    GATHER = 1


class ShadowParamStatus(Enum):
    NOT_AVAILABLE = 0
    INFLIGHT = 1
    AVAILABLE = 2


class ShadowModule(torch.nn.Module):

    def __init__(self, module, src_module):
        super().__init__()
        # assert isinstance(module, torch.nn.Module)
        # assert isinstance(src_module, torch.nn.Module)
        module = module[0]

        self.module = module
        self.src_module = src_module
        self._status = ShadowModuleStatus.RELEASE

        # ====
        src_mpu_index = get_model_mpu_index(src_module)
        dst_mpu_index = get_model_mpu_index(module)

        src_config = get_attr_wrapped_model(src_module[0], "config")

        parallel_state.switch_mpu_by_index(src_mpu_index)
        src_dp_size = parallel_state.get_data_parallel_world_size()
        src_tp_size = parallel_state.get_tensor_model_parallel_world_size()
        src_pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        src_is_first_stage = parallel_state.is_pipeline_first_stage()
        src_is_last_stage = parallel_state.is_pipeline_last_stage()
        src_num_layers = src_config.num_layers // src_pp_size
        src_layers_offset = parallel_state.get_pipeline_model_parallel_rank() * src_num_layers

        parallel_state.switch_mpu_by_index(dst_mpu_index)
        dst_dp_size = parallel_state.get_data_parallel_world_size()
        dst_tp_size = parallel_state.get_tensor_model_parallel_world_size()
        dst_pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        dst_is_first_stage = parallel_state.is_pipeline_first_stage()
        dst_is_last_stage = parallel_state.is_pipeline_last_stage()
        dst_num_layers = src_config.num_layers // dst_pp_size
        dst_layers_offset = parallel_state.get_pipeline_model_parallel_rank() * dst_num_layers

        switch_ranks, switch_gp = None, None
        _ranks_list = torch.LongTensor(range(dist.get_world_size()))
        _ranks_list = _ranks_list.view(src_pp_size, src_dp_size, src_tp_size)
        _ranks_list = _ranks_list.transpose(0, 2).contiguous()
        print(src_pp_size, dst_pp_size)
        _ranks_list = _ranks_list.view(-1, src_dp_size, src_pp_size//dst_pp_size)

        # TODO: create gather group
        for _dp_ranks in _ranks_list:
            for _ranks in _dp_ranks:
                _ranks = _ranks.tolist()
                gp = dist.new_group(ranks=_ranks)
                if dist.get_rank() in _ranks:
                    switch_ranks = _ranks
                    switch_gp = gp
        switch_gp_world_size = dist.get_world_size(group=switch_gp)

        self.switch_ranks = switch_ranks
        self.switch_gp = switch_gp

        # must be first pre_process layer, and last post_process layer
        _pre_process_layers = ['embedding.word_embeddings']
        _post_process_layers = ['lm_head.lm_head', 'model.norm']
        _row_parallel_weights = ['attention.dense.weight', 'mlp.down_proj.weight']
        _col_parallel_weights = ['attention.query_key_value.weight', 'attention.query_key_value.bias',
                                    'mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                    'word_embeddings.weight',
                                    'lm_head.lm_head']

        # ====

        def _register_hooks_recursively(name, module):

            for child_name, child in module.named_children():
                _register_hooks_recursively(".".join([name, child_name]), child)

            def _pre_forward_module_hook(module, *args):
                if not self._status == ShadowModuleStatus.GATHER:
                    return

                # gather params
                for param in module.parameters():
                    if param._shadow_status == ShadowParamStatus.NOT_AVAILABLE:
                        param._shadow_status = ShadowParamStatus.INFLIGHT
                        param.func1()
                        param._shadow_status = ShadowParamStatus.AVAILABLE

            def _post_forward_module_hook(module, *args):
                if not self._status == ShadowModuleStatus.RELEASE:
                    return

                # release params
                for param in module.parameters():
                    if param._shadow_status == ShadowParamStatus.AVAILABLE:
                        param._shadow_status = ShadowParamStatus.INFLIGHT
                        param.func2()
                        param._shadow_status = ShadowParamStatus.NOT_AVAILABLE


            if len(module._modules) == 0 and any(module.named_parameters()):
                # is leaf node and contains parameters
                _convert_to_shadow_parameters(module.parameters())

                module.register_forward_pre_hook(_pre_forward_module_hook)
                module.register_forward_hook(_post_forward_module_hook)

                def func1(self):
                    self.data = torch.zeros(self.shadow_shape, dtype=self.dtype, device=self.device)

                def func2(self):
                    self.data = torch.empty(0, dtype=self.dtype, device=self.device)

                # set each params' fn
                for param in module.parameters():

                    if 'layers' in name:
                        # e.g., module.language_model.encoder.layers.0.XXX
                        layer_re = re.compile(r"[a-z0-9_.]+.layers\.(\d+)\.[a-z0-9_.]+")
                        m = layer_re.match(name)
                        dst_layer_id = int(m.group(1))
                        layer_id = dst_layer_id + dst_layers_offset
                        layer_in_src = layer_id in range(src_layers_offset, src_layers_offset + src_num_layers)
                        name = name.replace(f".{dst_layer_id}.", f".{layer_id % src_num_layers}.")
                        src = switch_ranks[(layer_id-dst_layers_offset)//src_num_layers]

                    param.func1 = types.MethodType(func1, param)
                    param.func2 = types.MethodType(func2, param)

        _register_hooks_recursively("", module)

    def set_gather_params(self):
        assert self._status is ShadowModuleStatus.RELEASE
        self._status = ShadowModuleStatus.GATHER

    def set_release_params(self):
        assert self._status is ShadowModuleStatus.GATHER
        self._status = ShadowModuleStatus.RELEASE

    def forward(self, *inputs, **kwargs):
        outputs = self.module(*inputs, **kwargs)
        return outputs

    def set_model_mpu(self, mpu):
        self.module.set_model_mpu(mpu)


# ============================================== #
# ============================================== #

def _get_layer_by_name(model, name):
    for n, m in model.named_modules():
        if n.replace("module.", "") == name:
            return m
    return None

def _convert_to_shadow_parameters(param_list):
    for param in param_list:
        param.shadow_numel = param.numel()
        param.shadow_shape = param.shape
        param._shadow_status = ShadowParamStatus.NOT_AVAILABLE
        param.data = torch.empty(0, dtype=param.dtype, device=param.device)

def init_shadow_model_(model):
    if type(model) is not list:
        model = [model]
    for module in model:
        if module is not None:
            assert isinstance(module, torch.nn.Module)
            _convert_to_shadow_parameters(module.parameters(recurse=True))
    torch.cuda.empty_cache()

def init_shadow_model(model, mpu, shadow_model_provider):
    parallel_state.switch_mpu_by_index(mpu.get_index())
    shadow_model = shadow_model_provider()
    set_model_mpu(shadow_model, mpu)
    init_shadow_model_(shadow_model)
    model[0].shadow = shadow_model
    parallel_state.switch_mpu_by_index(get_model_mpu_index(model))

def apply_shadow_model(model):
    if type(model) is not list:
        model = [model]

    mpu_index = get_model_mpu_index(model)
    parallel_state.switch_mpu_by_index(mpu_index)
    # if parallel_state.get_data_parallel_world_size() == 1:
    #     _switch_parallelism_single_dp(model)
    # else:
    _apply_shadow_model(model)

def _apply_shadow_model_(model):
    src_model = model
    dst_model = model[0].shadow

    if type(src_model) is not list:
        src_model = [src_model]
    if type(dst_model) is not list:
        dst_model = [dst_model]

    src_mpu_index = get_model_mpu_index(src_model)
    dst_mpu_index = get_model_mpu_index(dst_model)

    src_config = get_attr_wrapped_model(src_model[0], "config")

    parallel_state.switch_mpu_by_index(src_mpu_index)
    src_dp_size = parallel_state.get_data_parallel_world_size()
    src_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    src_pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    src_is_first_stage = parallel_state.is_pipeline_first_stage()
    src_is_last_stage = parallel_state.is_pipeline_last_stage()
    src_num_layers = src_config.num_layers // src_pp_size
    src_layers_offset = parallel_state.get_pipeline_model_parallel_rank() * src_num_layers

    parallel_state.switch_mpu_by_index(dst_mpu_index)
    dst_dp_size = parallel_state.get_data_parallel_world_size()
    dst_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dst_pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    dst_is_first_stage = parallel_state.is_pipeline_first_stage()
    dst_is_last_stage = parallel_state.is_pipeline_last_stage()
    dst_num_layers = src_config.num_layers // dst_pp_size
    dst_layers_offset = parallel_state.get_pipeline_model_parallel_rank() * dst_num_layers

    switch_ranks, switch_gp = None, None
    _ranks_list = torch.LongTensor(range(dist.get_world_size()))
    _ranks_list = _ranks_list.view(src_pp_size, src_dp_size, src_tp_size)
    _ranks_list = _ranks_list.transpose(0, 2).contiguous()
    _ranks_list = _ranks_list.view(-1, src_dp_size, src_pp_size//dst_pp_size)

    for _dp_ranks in _ranks_list:
        for _ranks in _dp_ranks:
            _ranks = _ranks.tolist()
            gp = dist.new_group(ranks=_ranks)
            if dist.get_rank() in _ranks:
                switch_ranks = _ranks
                switch_gp = gp
    switch_gp_world_size = dist.get_world_size(group=switch_gp)

    # must be first pre_process layer, and last post_process layer
    _pre_process_layers = ['embedding.word_embeddings']
    _post_process_layers = ['lm_head.lm_head', 'model.norm']
    _row_parallel_weights = ['attention.dense.weight', 'mlp.down_proj.weight']
    _col_parallel_weights = ['attention.query_key_value.weight', 'attention.query_key_value.bias',
                                'mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                'word_embeddings.weight',
                                'lm_head.lm_head']

    for module in dst_model:
        for name, param in module.named_parameters(recurse=True):
            layer_in_src = False

            if 'layers' in name:
                # e.g., module.language_model.encoder.layers.0.XXX
                layer_re = re.compile(r"[a-z0-9_.]+.layers\.(\d+)\.[a-z0-9_.]+")
                m = layer_re.match(name)
                dst_layer_id = int(m.group(1))
                layer_id = dst_layer_id + dst_layers_offset
                layer_in_src = layer_id in range(src_layers_offset, src_layers_offset + src_num_layers)
                name = name.replace(f".{dst_layer_id}.", f".{layer_id % src_num_layers}.")
                src = switch_ranks[(layer_id-dst_layers_offset)//src_num_layers]
            else:
                _found = False
                for p in _pre_process_layers:
                    if p in name:
                        layer_in_src = True if src_is_first_stage else False
                        src = switch_ranks[0]
                        _found = True
                        break
                if not _found:
                    for p in _post_process_layers:
                        if p in name:
                            layer_in_src = True if src_is_last_stage else False
                            src = switch_ranks[-1]
                            _found = True
                            break
            name = name.replace("module.", "")

            scatter_list = None
            is_col_or_row_parallel_param = False

            # allocate memory
            param.data = torch.zeros(param.shadow_shape, dtype=param.dtype, device=param.device)

            for p in _row_parallel_weights:
                if p in name:
                    if layer_in_src:
                        src_data = src_model[0].state_dict()[name]
                        shard_size = src_data.shape[1] // switch_gp_world_size
                        scatter_list = [src_data[..., shard_size*i:shard_size*(i+1)].contiguous() for i in range(switch_gp_world_size)]
                    is_col_or_row_parallel_param = True
                    break

            for p in _col_parallel_weights:
                if p in name:
                    if layer_in_src:
                        src_data = src_model[0].state_dict()[name]
                        shard_size = src_data.shape[0] // switch_gp_world_size
                        scatter_list = [src_data[shard_size*i:shard_size*(i+1), ...] for i in range(switch_gp_world_size)]
                    is_col_or_row_parallel_param = True
                    break

            if is_col_or_row_parallel_param:
                dist.scatter(param.data, scatter_list, src=src, group=switch_gp, async_op=False)

                # HARD CODING here: update word_embeddings's vocab range
                if 'word_embeddings.weight' in name:
                    vocab_range_scatter_list = None
                    if layer_in_src:
                        word_embeddings = _get_layer_by_name(src_model[0], name.replace(".weight", ""))
                        vocab_start_index = word_embeddings.vocab_start_index
                        vocab_shard_size = word_embeddings.num_embeddings_per_partition // switch_gp_world_size
                        vocab_range_scatter_list = [torch.tensor([vocab_start_index+i*vocab_shard_size, vocab_start_index+(i+1)*vocab_shard_size],
                                                                    dtype=torch.int64,
                                                                    device=param.device)
                                                                    for i in range(switch_gp_world_size)]
                    vocab_range = torch.tensor([0, 0], dtype=torch.int64, device=param.device)
                    dist.scatter(vocab_range, vocab_range_scatter_list, src=src, group=switch_gp, async_op=False)
                    dst_word_embeddings = _get_layer_by_name(dst_model[0], name.replace(".weight", ""))
                    dst_word_embeddings.vocab_start_index = vocab_range[0].item()
                    dst_word_embeddings.vocab_end_index = vocab_range[1].item()
                continue

            # other param
            if layer_in_src:
                try:
                    src_data = src_model[0].state_dict()[name]
                except:
                    raise f"  > [RANK{dist.get_rank()}] !!!! Parameter not found: {name}"
                param.data.copy_(src_data)
            dist.broadcast(tensor=param.data, src=src, group=switch_gp, async_op=False)

def _switch_parallelism_single_dp(model):
    src_model = model
    dst_model = model[0].shadow

    if type(src_model) is not list:
        src_model = [src_model]
    if type(dst_model) is not list:
        dst_model = [dst_model]

    src_mpu_index = get_model_mpu_index(src_model)
    dst_mpu_index = get_model_mpu_index(dst_model)

    src_config = get_attr_wrapped_model(src_model[0], "config")

    parallel_state.switch_mpu_by_index(src_mpu_index)
    src_dp_size = parallel_state.get_data_parallel_world_size()
    src_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    src_pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    src_is_first_stage = parallel_state.is_pipeline_first_stage()
    src_is_last_stage = parallel_state.is_pipeline_last_stage()
    src_num_layers = src_config.num_layers // src_pp_size
    src_layers_offset = parallel_state.get_pipeline_model_parallel_rank() * src_num_layers

    parallel_state.switch_mpu_by_index(dst_mpu_index)
    dst_dp_size = parallel_state.get_data_parallel_world_size()
    dst_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dst_pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    dst_is_first_stage = parallel_state.is_pipeline_first_stage()
    dst_is_last_stage = parallel_state.is_pipeline_last_stage()
    dst_num_layers = src_config.num_layers // dst_pp_size
    dst_layers_offset = parallel_state.get_pipeline_model_parallel_rank() * dst_num_layers

    _ranks = torch.LongTensor(range(dist.get_world_size()))
    _ranks = _ranks.view(src_pp_size, src_dp_size, src_tp_size)
    _ranks = _ranks.transpose(0, 2).contiguous()
    _ranks = _ranks.view(-1, src_dp_size, src_pp_size//dst_pp_size)

    switch_ranks, switch_gp = None, None
    for _dp_ranks in _ranks:
        for _ranks in _dp_ranks:
            _ranks = _ranks.tolist()
            gp = dist.new_group(ranks=_ranks)
            if dist.get_rank() in _ranks:
                switch_ranks = _ranks
                switch_gp = gp
    switch_gp_world_size = dist.get_world_size(group=switch_gp)
    tp_ratio = dst_tp_size // src_tp_size
    dp_ratio = dst_dp_size // src_dp_size

    # must be first pre_process layer, and last post_process layer
    _pre_process_layers = ['embedding.word_embeddings']
    _post_process_layers = ['lm_head.lm_head', 'model.norm']
    _row_parallel_weights = ['attention.dense.weight', 'mlp.down_proj.weight']
    _col_parallel_weights = ['attention.query_key_value.weight', 'attention.query_key_value.bias',
                                'mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                'word_embeddings.weight',
                                'lm_head.lm_head']

    for module in dst_model:
        for name, param in module.named_parameters(recurse=True):
            layer_in_src = False

            if 'layers' in name:
                # e.g., module.language_model.encoder.layers.0.XXX
                layer_re = re.compile(r"[a-z0-9_.]+.layers\.(\d+)\.[a-z0-9_.]+")
                m = layer_re.match(name)
                dst_layer_id = int(m.group(1))
                layer_id = dst_layer_id + dst_layers_offset
                layer_in_src = layer_id in range(src_layers_offset, src_layers_offset + src_num_layers)
                name = name.replace(f".{dst_layer_id}.", f".{layer_id % src_num_layers}.")
                src = switch_ranks[(layer_id-dst_layers_offset)//src_num_layers]
            else:
                _found = False
                for p in _pre_process_layers:
                    if p in name:
                        layer_in_src = True if src_is_first_stage else False
                        src = switch_ranks[0]
                        _found = True
                        break
                if not _found:
                    for p in _post_process_layers:
                        if p in name:
                            layer_in_src = True if src_is_last_stage else False
                            src = switch_ranks[-1]
                            _found = True
                            break
            name = name.replace("module.", "")

            scatter_list = None
            is_col_or_row_parallel_param = False

            # allocate memory
            param.data = torch.zeros(param.shadow_shape, dtype=param.dtype, device=param.device)

            for p in _row_parallel_weights:
                if p in name:
                    if layer_in_src:
                        src_data = src_model[0].state_dict()[name]
                        shard_size = src_data.shape[1] // tp_ratio
                        scatter_list = [src_data[..., shard_size*i:shard_size*(i+1)].contiguous() for i in range(tp_ratio)] * dp_ratio
                    is_col_or_row_parallel_param = True
                    break

            for p in _col_parallel_weights:
                if p in name:
                    if layer_in_src:
                        src_data = src_model[0].state_dict()[name]
                        shard_size = src_data.shape[0] // tp_ratio
                        scatter_list = [src_data[shard_size*i:shard_size*(i+1), ...] for i in range(tp_ratio)] * dp_ratio
                    is_col_or_row_parallel_param = True
                    break

            if is_col_or_row_parallel_param:
                dist.scatter(param.data, scatter_list, src=src, group=switch_gp, async_op=False)

                # HARD CODING here: update word_embeddings's vocab range
                if 'word_embeddings.weight' in name:
                    vocab_range_scatter_list = None
                    if layer_in_src:
                        word_embeddings = _get_layer_by_name(src_model[0], name.replace(".weight", ""))
                        vocab_start_index = word_embeddings.vocab_start_index
                        vocab_shard_size = word_embeddings.num_embeddings_per_partition // tp_ratio
                        vocab_range_scatter_list = [torch.tensor([vocab_start_index+i*vocab_shard_size, vocab_start_index+(i+1)*vocab_shard_size],
                                                                    dtype=torch.int64,
                                                                    device=param.device)
                                                                    for i in range(tp_ratio)] * dp_ratio
                    vocab_range = torch.tensor([0, 0], dtype=torch.int64, device=param.device)
                    dist.scatter(vocab_range, vocab_range_scatter_list, src=src, group=switch_gp, async_op=False)
                    dst_word_embeddings = _get_layer_by_name(dst_model[0], name.replace(".weight", ""))
                    dst_word_embeddings.vocab_start_index = vocab_range[0].item()
                    dst_word_embeddings.vocab_end_index = vocab_range[1].item()
                continue

            # other param
            if layer_in_src:
                try:
                    src_data = src_model[0].state_dict()[name]
                except:
                    raise f"  > [RANK{dist.get_rank()}] !!!! Parameter not found: {name}"
                param.data.copy_(src_data)
            dist.broadcast(tensor=param.data, src=src, group=switch_gp, async_op=False)

def _apply_shadow_model(model):
    src_model = model
    dst_model = model[0].shadow

    if type(src_model) is not list:
        src_model = [src_model]
    if type(dst_model) is not list:
        dst_model = [dst_model]

    src_mpu_index = get_model_mpu_index(src_model)
    dst_mpu_index = get_model_mpu_index(dst_model)

    src_config = get_attr_wrapped_model(src_model[0], "config")

    parallel_state.switch_mpu_by_index(src_mpu_index)
    src_dp_size = parallel_state.get_data_parallel_world_size()
    src_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    src_pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    src_is_first_stage = parallel_state.is_pipeline_first_stage()
    src_is_last_stage = parallel_state.is_pipeline_last_stage()
    src_dp_rank = parallel_state.get_data_parallel_rank()
    src_tp_rank = parallel_state.get_tensor_model_parallel_rank()
    src_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    src_num_layers = src_config.num_layers // src_pp_size

    parallel_state.switch_mpu_by_index(dst_mpu_index)
    dst_dp_size = parallel_state.get_data_parallel_world_size()
    dst_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dst_pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    dst_is_first_stage = parallel_state.is_pipeline_first_stage()
    dst_is_last_stage = parallel_state.is_pipeline_last_stage()
    dst_dp_rank = parallel_state.get_data_parallel_rank()
    dst_tp_rank = parallel_state.get_tensor_model_parallel_rank()
    dst_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    dst_num_layers = src_config.num_layers // dst_pp_size

    parallel_state.switch_mpu_by_index(src_mpu_index)
    src_layers_offset = parallel_state.get_pipeline_model_parallel_rank() * src_num_layers

    parallel_state.switch_mpu_by_index(dst_mpu_index)
    dst_layers_offset = parallel_state.get_pipeline_model_parallel_rank() * dst_num_layers

    dp_ratio = dst_dp_size // src_dp_size

    src_replicas_map = [[] for _ in range(src_dp_size)]
    dst_replicas_map = [[] for _ in range(dst_dp_size)]

    ws = dist.get_world_size()
    rank = dist.get_rank()

    for i in range(ws):
        dp_rank = i // src_tp_size % src_dp_size
        src_replicas_map[dp_rank].append(i)

    for i in range(ws):
        dp_rank = i // dst_tp_size % dst_dp_size
        dst_replicas_map[dp_rank].append(i)

    src_replicas_map = torch.LongTensor(src_replicas_map)
    dst_replicas_map = torch.LongTensor(dst_replicas_map)

    src_replicas_map = src_replicas_map.view(-1, src_pp_size, src_tp_size)
    dst_replicas_map = dst_replicas_map.view(-1, dst_pp_size, dst_tp_size)

    src_replicas_map = src_replicas_map.reshape(list(src_replicas_map.shape[0:1]) + [dst_pp_size, -1] + list(src_replicas_map.shape[-1:]))
    src_replicas_map = src_replicas_map.repeat(1, dp_ratio, 1, 1)
    src_replicas_map = src_replicas_map.view(-1, *src_replicas_map.shape[-2:])
    src_list = src_replicas_map.tolist()

    dst_replicas_map = dst_replicas_map.reshape(list(dst_replicas_map.shape[:-1]) + [src_tp_size, -1])
    dst_replicas_map = dst_replicas_map.view(-1, *dst_replicas_map.shape[-2:])
    dst_list = dst_replicas_map.tolist()

    peers_map = {}
    send_peers = []
    recv_peers = []
    for src_sub_list, dst_sub_lit in zip(src_list, dst_list):
        for sub_sub_list in src_sub_list:
            for k, v in zip(sub_sub_list, dst_sub_lit):
                peers_map[k] = v
                if rank == k:
                    send_peers.append(v)
                if rank in v:
                    recv_peers.append(k)

    # if dist.get_rank() == 0:
    #     print("send_peers:", send_peers)
    #     print("recv_peers:", recv_peers)

    _pre_process_layers = ['embedding.word_embeddings']
    _post_process_layers = ['lm_head.lm_head', 'final_layernorm']
    _row_parallel_weights = ['attention.dense.weight', 'mlp.down_proj.weight']
    _col_parallel_weights = ['attention.query_key_value.weight', 'attention.query_key_value.bias',
                                'mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                'word_embeddings.weight',
                                'lm_head.lm_head']

    for module in dst_model:
        for name, param in module.named_parameters(recurse=True):
            contain_layer = False
            if 'layers' in name:
                # e.g., module.language_model.encoder.layers.0.XXX
                layer_re = re.compile(r"[a-z0-9_.]+.layers\.(\d+)\.[a-z0-9_.]+")
                m = layer_re.match(name)
                dst_layer_id = int(m.group(1))
                layer_id = dst_layer_id + dst_layers_offset
                contain_layer = layer_id in range(src_layers_offset, src_layers_offset + src_num_layers)
                name = name.replace(f".{dst_layer_id}.", f".{layer_id % src_num_layers}.")
                src = recv_peers[(layer_id-dst_layers_offset)//src_num_layers]
            else:
                _found = False
                for p in _pre_process_layers:
                    if p in name:
                        contain_layer = True if src_is_first_stage else False
                        # src = broadcast_ranks[0]
                        src = recv_peers[0]
                        _found = True
                        break
                if not _found:
                    for p in _post_process_layers:
                        if p in name:
                            contain_layer = True if src_is_last_stage else False
                            # src = broadcast_ranks[-1]
                            src = recv_peers[-1]
                            _found = True
                            break
            name = name.replace("module.", "")

            # allocate memory
            param.data = torch.zeros(param.shadow_shape, dtype=param.dtype, device=param.device)

            is_col_or_row_parallel_param = False
            for p in _row_parallel_weights:
                if p in name:
                    is_col_or_row_parallel_param = True
                    ops = []
                    ops.append(torch.distributed.P2POp(dist.irecv, param.data, src))
                    if contain_layer:
                        src_data = src_model[0].state_dict()[name]
                        shard_size = src_data.shape[1] // len(send_peers[0])
                        for peers in send_peers:
                            for i, peer in enumerate(peers):
                                data = src_data[..., shard_size*i : shard_size*(i+1)].contiguous()
                                ops.append(torch.distributed.P2POp(dist.isend, data, peer))

                    if len(ops) > 0:
                        reqs = torch.distributed.batch_isend_irecv(ops)
                        for req in reqs:
                            req.wait()
                    break

            if is_col_or_row_parallel_param:
                continue

            for p in _col_parallel_weights:
                if p in name:
                    is_col_or_row_parallel_param = True
                    ops = []
                    ops.append(torch.distributed.P2POp(dist.irecv, param.data, src))
                    if contain_layer:
                        src_data = src_model[0].state_dict()[name]
                        shard_size = src_data.shape[0] // len(send_peers[0])
                        for peers in send_peers:
                            for i, peer in enumerate(peers):
                                data = src_data[shard_size*i : shard_size*(i+1), ...].contiguous()
                                ops.append(torch.distributed.P2POp(dist.isend, data, peer))

                    if len(ops) > 0:
                        reqs = torch.distributed.batch_isend_irecv(ops)
                        for req in reqs:
                            req.wait()
                    break

            if is_col_or_row_parallel_param:
                continue

            # other param
            ops = []
            ops.append(torch.distributed.P2POp(dist.irecv, param.data, src))
            if contain_layer:
                try:
                    src_data = src_model[0].state_dict()[name]
                except:
                    raise f"  > [RANK{dist.get_rank()}] !!!! Parameter not found: {name}"
                # print(name, src_data.shape, param.shape, src, send_peers)
                for peers in send_peers:
                    for i, peer in enumerate(peers):
                        data = src_data.contiguous()
                        ops.append(torch.distributed.P2POp(dist.isend, data, peer))

            if len(ops) > 0:
                try:
                    reqs = torch.distributed.batch_isend_irecv(ops)
                except Exception as e:
                    print(e, name, rank, send_peers, src, contain_layer)
                    exit(-1)
                for req in reqs:
                    req.wait()

    torch.cuda.synchronize()
    # print("done")

def release_shadow_model(model):
    for module in model:
        for _, param in module.named_parameters(recurse=True):
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)

def exchange_input_data_(model, input):
    src_mpu_index = get_model_mpu_index(model)
    dst_mpu_index = get_model_mpu_index(model[0].shadow)

    parallel_state.switch_mpu_by_index(src_mpu_index)
    src_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    src_dp_size = parallel_state.get_data_parallel_world_size()

    parallel_state.switch_mpu_by_index(dst_mpu_index)
    dst_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dst_dp_size = parallel_state.get_data_parallel_world_size()

    ws = dist.get_world_size()
    dp_ratio = dst_dp_size // src_dp_size

    src_replicas_map = [[] for _ in range(src_dp_size)]
    dst_replicas_map = [[] for _ in range(dst_dp_size)]

    # each replica map contains all rank of this replica, the partition is following Megatron-LM's partition
    for i in range(ws):
        dp_rank = i // src_tp_size % src_dp_size
        src_replicas_map[dp_rank].append(i)

    for i in range(ws):
        dp_rank = i // dst_tp_size % dst_dp_size
        dst_replicas_map[dp_rank].append(i)

    def split_each_group_by_factor(groups, factor):
        factor = int(factor)
        new_group = []
        for group in groups:
            shard_len = len(group) // factor
            for i in range(0, len(group), shard_len):
                new_group.append(group[i:i+shard_len])
        return new_group

    # by handling the case where src_dp_size < dst_dp_size (dp_ratio = dst_dp_size // src_dp_size),
    # we can make sure that each replica in dst_dp_size have the same size of corresponding peer in src_dp_size
    src_replicas_map = split_each_group_by_factor(src_replicas_map, dp_ratio)

    recv_from = -1
    send_to = -1
    data_shard_id = -1
    for i, (s_replica, d_replica) in enumerate(zip(src_replicas_map, dst_replicas_map)):
        for s, d in zip(s_replica, d_replica):
            if dist.get_rank() == s:
                send_to = d
                data_shard_id = i % dp_ratio
            if dist.get_rank() == d:
                recv_from = s

    if recv_from == dist.get_rank() and send_to == dist.get_rank():
        for k in input.keys():
            shard_size = input[k].shape[0] // dp_ratio
            input[k] = input[k][shard_size*data_shard_id : shard_size*(data_shard_id+1), ...]
        return input

    # init _input
    shape_dict = {k: list(v.shape) for k, v in input.items()}
    # if dp_ratio > 1:
    for v in shape_dict.values():
        v[0] = v[0] // dp_ratio
    _input = {k: torch.zeros(v, dtype=input[k].dtype, device=input[k].device) for k, v in shape_dict.items()}

    ops = []
    for k, v in input.items():
        shard_size = v.shape[0] // dp_ratio
        v = v[shard_size*data_shard_id : shard_size*(data_shard_id+1), ...]
        send_op = dist.P2POp(dist.isend, v, send_to)
        ops.append(send_op)

    for k, v in _input.items():
        recv_op = dist.P2POp(dist.irecv, v, recv_from)
        ops.append(recv_op)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return _input

def exchange_input_data(model, input):
    src_mpu_index = get_model_mpu_index(model)
    dst_mpu_index = get_model_mpu_index(model[0].shadow)

    parallel_state.switch_mpu_by_index(src_mpu_index)
    src_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    src_dp_size = parallel_state.get_data_parallel_world_size()

    parallel_state.switch_mpu_by_index(dst_mpu_index)
    dst_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dst_dp_size = parallel_state.get_data_parallel_world_size()

    ws = dist.get_world_size()

    src_replicas_map = [[] for _ in range(src_dp_size)]
    dst_replicas_map = [[] for _ in range(dst_dp_size)]

    for i in range(ws):
        dp_rank = i // src_tp_size % src_dp_size
        src_replicas_map[dp_rank].append(i)

    for i in range(ws):
        dp_rank = i // dst_tp_size % dst_dp_size
        dst_replicas_map[dp_rank].append(i)

    src = -1
    dst = -1
    for s_replica, d_replica in zip(src_replicas_map, dst_replicas_map):
        for s, d in zip(s_replica, d_replica):
            if dist.get_rank() == s:
                dst = d
            if dist.get_rank() == d:
                src = s

    if src == dist.get_rank() and dst == dist.get_rank():
        return input

    _input = {k: torch.zeros_like(v) for k, v in input.items()}

    ops = []
    for k, v in input.items():
        send_op = dist.P2POp(dist.isend, v, dst)
        ops.append(send_op)
    for k, v in _input.items():
        recv_op = dist.P2POp(dist.irecv, v, src)
        ops.append(recv_op)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return _input

def exchange_output_data_(model, output):
    src_mpu_index = get_model_mpu_index(model)
    dst_mpu_index = get_model_mpu_index(model[0].shadow)

    parallel_state.switch_mpu_by_index(src_mpu_index)
    src_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    src_dp_size = parallel_state.get_data_parallel_world_size()

    parallel_state.switch_mpu_by_index(dst_mpu_index)
    dst_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dst_dp_size = parallel_state.get_data_parallel_world_size()

    ws = dist.get_world_size()
    dp_ratio = dst_dp_size // src_dp_size

    src_replicas_map = [[] for _ in range(src_dp_size)]
    dst_replicas_map = [[] for _ in range(dst_dp_size)]

    # each replica map contains all rank of this replica, the partition is following Megatron-LM's partition
    for i in range(ws):
        dp_rank = i // src_tp_size % src_dp_size
        src_replicas_map[dp_rank].append(i)

    for i in range(ws):
        dp_rank = i // dst_tp_size % dst_dp_size
        dst_replicas_map[dp_rank].append(i)

    # peers_map:
    # e.g., {0: [0, 2], 1: [4, 6]} means that 0 have to received output data from 0 and 2, each contain portion of batch data
    peers_map = {}
    for src_dp_rank, src_replica in enumerate(src_replicas_map):
        for src_tp_rank, src_id in enumerate(src_replica):
            if src_tp_rank >= src_tp_size:
                break
            for dst_replica in dst_replicas_map[dp_ratio*src_dp_rank : dp_ratio*(src_dp_rank+1)]:
                if src_id not in peers_map:
                    peers_map[src_id] = []
                peers_map[src_id].append(dst_replica[src_tp_rank])

    nothing_to_do = True
    for recv_peer, send_peers in peers_map.items():
        if dist.get_rank() in send_peers or dist.get_rank() == recv_peer:
            nothing_to_do = False
            break

    if nothing_to_do:
        return {k: None for k in output.keys()}

    # init _output
    _output = {}
    if dist.get_rank() in peers_map:
        for k, v in output.items():
            shard_size = v.shape[0]
            v_shape = list(v.shape)
            v_shape[0] = v_shape[0] * dp_ratio
            _output[k] = torch.zeros(v_shape, dtype=v.dtype, device=v.device)
    else:
        for k, v in output.items():
            _output[k] = None

    # ready to send & recv
    ops = []
    if dist.get_rank() in peers_map.keys():
        for shard_id, send_peer in enumerate(peers_map[dist.get_rank()]):
            for k, v in _output.items():
                recv_op = dist.P2POp(dist.irecv, v[shard_size*shard_id : shard_size*(shard_id+1), ...], send_peer)
                ops.append(recv_op)
                # print(f"  > [RANK{dist.get_rank()}] recv shard {shard_id} from {send_peer}")
    for recv_peer, send_peers in peers_map.items():
        if dist.get_rank() in send_peers:
            for k, v in output.items():
                send_op = dist.P2POp(dist.isend, v, recv_peer)
                ops.append(send_op)
                # print(f"  > [RANK{dist.get_rank()}] send to {recv_peer}")
            break

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return _output

def exchange_output_data(model, output):
    src_mpu_index = get_model_mpu_index(model)
    dst_mpu_index = get_model_mpu_index(model[0].shadow)

    parallel_state.switch_mpu_by_index(src_mpu_index)
    src_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    src_dp_size = parallel_state.get_data_parallel_world_size()
    is_pipeline_first_stage = parallel_state.is_pipeline_first_stage()

    parallel_state.switch_mpu_by_index(dst_mpu_index)
    dst_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dst_dp_size = parallel_state.get_data_parallel_world_size()

    ws = dist.get_world_size()

    src_replicas_map = [[] for _ in range(src_dp_size)]
    dst_replicas_map = [[] for _ in range(dst_dp_size)]

    for i in range(ws):
        dp_rank = i // src_tp_size % src_dp_size
        src_replicas_map[dp_rank].append(i)

    for i in range(ws):
        dp_rank = i // dst_tp_size % dst_dp_size
        dst_replicas_map[dp_rank].append(i)

    src = -1
    dst = -1
    is_src_pipeline_first_stage = False
    for s_replica, d_replica in zip(src_replicas_map, dst_replicas_map):
        for i, (s, d) in enumerate(zip(s_replica, d_replica)):
            if dist.get_rank() == s:
                dst = d
            if dist.get_rank() == d:
                if i < src_tp_size:
                    is_src_pipeline_first_stage = True
                src = s

    if src == dist.get_rank() and dst == dist.get_rank():
        return output

    if not (is_src_pipeline_first_stage or is_pipeline_first_stage):
        return {k: None for k in output.keys()}

    _output = {k: torch.zeros_like(v) for k, v in output.items()}
    ops = []
    for k, v in _output.items():
        if dst == dist.get_rank():
            send_op = dist.P2POp(dist.isend, output[k], src)
            ops.append(send_op)
        elif src == dist.get_rank():
            recv_op = dist.P2POp(dist.irecv, v, dst)
            ops.append(recv_op)
    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return _output


class ApplyShadowModel:

    def __init__(self, model, *args, **kwargs):
        self.model = model

    def __enter__(self):
        assert hasattr(self.model[0], "shadow"), "Shadow model is not initialized"
        torch.cuda.synchronize()
        st = time.time()
        apply_shadow_model(self.model)
        torch.cuda.synchronize()
        if torch.distributed.get_rank() == 0:
            print(f"apply shadow model time: {time.time()-st}")
        for module in self.model[0].shadow:
            module.eval()
        from puzzle.core.utils import apply_model_mpu
        apply_model_mpu(self.model[0].shadow)
        return self.model[0].shadow

    def __exit__(self, *exc):
        # torch.cuda.synchronize()
        # st = time.time()
        release_shadow_model(self.model[0].shadow)
        from puzzle.core.utils import apply_model_mpu
        apply_model_mpu(self.model)
        # torch.cuda.synchronize()
        # print(f"release shadow model time: {time.time()-st}")
