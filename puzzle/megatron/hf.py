import os
from typing import List, Tuple
import glob
import torch
from safetensors.torch import safe_open
import re

import torch.distributed

from megatron import get_args
from megatron import core
from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model

# NOTE:
# This file is convert LLaMA-2 7B or 13B model to Megatron-LM model
#

def _rename_hf2megatron(name):
    name = name.replace('model.', '')
    if name == 'embed_tokens.weight':
        return 'embedding.word_embeddings.weight'
    if name == 'norm.weight':
        return 'transformer.final_layernorm.weight'
    if name == 'lm_head.weight':
        return 'lm_head.' + name
    name = name.replace('self_attn.W_pack', 'attention.query_key_value')
    name = name.replace('self_attn.o_proj', 'attention.dense')
    return 'transformer.' + name


def _is_emb_mp(name):
    if name.find('word_embeddings') > -1:
        return True
    # if name.find('lm_head') != -1:
    #     return True
    return False


def _is_row_mp(name):
    if name.find('attention.query_key_value.weight') != -1:
        return True
    if name.find('mlp.up_proj') != -1:
        return True
    if name.find('mlp.gate_proj') != -1:
        return True
    if name.find('lm_head') != -1:
        return True
    return False


def _is_col_mp(name):
    if name.find('attention.dense') != -1:
        return True
    if name.find('mlp.down_proj') != -1:
        return True
    return False


def _parallel_state_slice(x, dim):
    slice_size = core.utils.divide(x.shape[dim], parallel_state.get_tensor_model_parallel_world_size())
    rank = parallel_state.get_tensor_model_parallel_rank()
    start = slice_size * rank
    end = start + slice_size
    if dim == 0:
        y = x[start:end, :]
    if dim == 1:
        y = x[:, start:end]
    return y


def _prepare_hf_weights(
    quantized_model_dir: str,
    load_format: str = "auto",
    fall_back_to_pt: bool = True,
) -> Tuple[str, List[str], bool]:
    if not os.path.isdir(quantized_model_dir):
        raise FileNotFoundError(
            f"The quantized model directory `{quantized_model_dir}` "
            "does not exist.")
    use_safetensors = False
    # Some quantized models use .pt files for storing the weights.
    if load_format == "auto":
        allow_patterns = ["*.safetensors", "*.bin"]
    elif load_format == "safetensors":
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == "pt":
        allow_patterns = ["*.pt"]
    elif load_format == "npz":
        allow_patterns = ["*.npz"]
    else:
        raise ValueError(f"Unknown load_format: {load_format}")
    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(
            os.path.join(quantized_model_dir, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break

    if not use_safetensors:
        # Exclude files that are not needed for inference.
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        hf_weights_files = [
            f for f in hf_weights_files
            if not any(f.endswith(x) for x in blacklist)
        ]

    if len(hf_weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any model weights with `{quantized_model_dir}`")

    return hf_weights_files, use_safetensors


def get_layer_by_name(model, name):
    for n, m in model.named_modules():
        if name in n.replace('module.', ''):
            return m
    return None


# Adopted from https://github.com/epfLLM/Megatron-LLM/blob/main/weights_conversion/hf_to_megatron.py
def permute_qkv(qkv_w: torch.Tensor, dim: int, n_heads: int,
                n_heads_kv: int, revert: bool = False) -> torch.Tensor:

    def permute(x):
        if revert:
            return x.view(head_dim//2, 2, dim).transpose(0, 1).reshape(head_dim, dim)
        return x.view(2, head_dim//2, dim).transpose(0, 1).reshape(head_dim, dim)

    head_dim = dim//n_heads
    n_qs_per_kv = n_heads//n_heads_kv
    n_groups = qkv_w.size(0)//head_dim//(n_qs_per_kv + 2)
    groups = torch.chunk(qkv_w, n_groups, dim=0)
    new = []
    for group in groups:
        *qs, k, v = torch.split(group, head_dim, dim=0)
        assert len(qs) == n_qs_per_kv, f"{len(qs)}, {n_qs_per_kv}"
        new += list(map(permute, qs)) + [permute(k), v]
    return torch.cat(new, dim=0)


def load_hf_ckpt(model):

    def permute(qkv_w):
        return permute_qkv(qkv_w, hidden_size, num_attention_heads, n_kv_heads)

    def rearrange_qkv(wq, wk, wv):
        wq = torch.split(wq, n_hidden_per_head, dim=0)
        wk = torch.split(wk, n_hidden_per_head, dim=0)
        wv = torch.split(wv, n_hidden_per_head, dim=0)
        assert len(wq) == num_attention_heads
        assert len(wk) == n_kv_heads
        assert len(wv) == n_kv_heads
        n_qs_per_kv = num_attention_heads//n_kv_heads
        w_qkv = []
        for i in range(n_kv_heads):
            w_qkv += [wq[i*n_qs_per_kv + j] for j in range(n_qs_per_kv)]
            w_qkv += [wk[i], wv[i]]
        return permute(torch.concat(w_qkv))

    args = get_args()
    params = dict()
    for n, p in model[0].named_parameters():
        n = n.replace('module.', '').replace('model.', '')
        params[n] = p

    loaded_params = dict()
    for n, p in model[0].named_parameters():
        n = n.replace('module.', '').replace('model.', '')
        loaded_params[n] = False

    config = get_attr_wrapped_model(model[0], "config")
    total_num_layers = config.num_layers
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    n_hidden_per_head = hidden_size // num_attention_heads
    n_kv_heads = num_attention_heads     # now only consider <= 13B

    num_layers = total_num_layers // parallel_state.get_pipeline_model_parallel_world_size()
    layers_offset = num_layers * parallel_state.get_pipeline_model_parallel_rank()
    layer_range = range(layers_offset, layers_offset + num_layers)

    with torch.no_grad():
        # filenames, _ = _prepare_hf_weights(args.load_hf)
        filenames, use_safetensors = _prepare_hf_weights("/home/dataset/llama-2-hf-all/Llama-2-7b-hf")
        assert use_safetensors is True, "Only support safetensors format for now."
        for filename in filenames:
            if len(filename) < 1:
                continue
            p = safe_open(filename, framework="pt")
            for n in p.keys():
                name_megatron = _rename_hf2megatron(n)
                if 'layers' in name_megatron:
                    layer_re = re.compile(r"[a-z0-9_.]+.layers\.(\d+)\.[a-z0-9_.]+")
                    m = layer_re.match(name_megatron)
                    layer_id = int(m.group(1))
                    if layer_id not in layer_range:
                        continue
                    update_layer_id = layer_id - layers_offset
                    # replace layer_id to update_layer_id
                    name_megatron = name_megatron.replace(f'.layers.{layer_id}.', f'.layers.{update_layer_id}.')

                if name_megatron.find('self_attn.q_proj') != -1:
                    hf_prefix = n.split('.self_attn.q_proj')[0]
                    np = rearrange_qkv(
                        p.get_tensor(f"{hf_prefix}.self_attn.q_proj.weight"),
                        p.get_tensor(f"{hf_prefix}.self_attn.k_proj.weight"),
                        p.get_tensor(f"{hf_prefix}.self_attn.v_proj.weight")
                    )
                    name_megatron = name_megatron.replace('self_attn.q_proj', \
                                                          'attention.query_key_value')
                    params[name_megatron].copy_(np)

                    # loaded
                    loaded_params[name_megatron] = True
                    continue

                if name_megatron not in params:
                    continue

                np = p.get_tensor(n)
                if _is_emb_mp(name_megatron):
                    # start = model[0].module.module.model.embedding.word_embeddings.vocab_start_index
                    # end = model[0].module.module.model.embedding.word_embeddings.vocab_end_index
                    start = get_layer_by_name(model[0], 'embedding.word_embeddings').vocab_start_index
                    end = get_layer_by_name(model[0], 'embedding.word_embeddings').vocab_end_index
                    np = np[start:end, :]
                    vocab_size = params[name_megatron].shape[0]
                    if vocab_size > np.shape[0]:
                        z = torch.zeros(vocab_size - np.shape[0], np.shape[1],
                                        dtype=np.dtype, device=np.device)
                        np = torch.cat([np, z], dim=0)
                if _is_row_mp(name_megatron):
                    np = _parallel_state_slice(np, 0)
                if _is_col_mp(name_megatron):
                    np = _parallel_state_slice(np, 1)
                mp = params[name_megatron]
                if mp.shape != np.shape:
                    print(f'{name_megatron} shape mismatch: {mp.shape} <- {np.shape}')
                    assert(False)
                else:
                    if torch.distributed.get_rank() == 0:
                        print(f'load {name_megatron} <- {n}')
                mp.copy_(np)

                # loaded
                loaded_params[name_megatron] = True

    for k, v in loaded_params.items():
        assert v is True, f'Not loaded: {k}'