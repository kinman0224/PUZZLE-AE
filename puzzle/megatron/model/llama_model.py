# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Llama model.
Following implementation from huggingface, https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""

from typing import List, Optional
import math

import torch
import torch.distributed
import torch.nn.functional as F

from megatron import get_args, core
from megatron.core import parallel_state, tensor_parallel
from megatron.model.module import MegatronModule, float16_to_fp32, fp32_to_float16
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.utils import get_linear_layer, init_method_normal, scaled_init_method_normal, attention_mask_func, \
    openai_gelu, erf_gelu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.language_model import Pooler


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), 1, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = ~expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), 1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset: q.shape[-2] + offset, :]
    sin = sin[..., offset: q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# TODO not able to build apex cpp extention for Fused cuda kernel RMSNorm
# Steps performed, 1. copy https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm.py, https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda.cpp, https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu to ./megatron/model/fused_layer_norm.py, ./megatron/fused_kernels/layer_norm_cuda.cpp, ./megatron/fused_kernels/layer_norm_cuda_kernel.cu, and update ./megatron/fused_kernels/__init__.py accordingly 2. use below line to import MixedFusedRMSNorm
# torch.nn.LayerNorm is slower than apex.FusedLayerNorm for shapes typical in NLP models. For example: (512, 16, 1024) with normalization over the last dimension is slower using torch.nn.LayerNorm
# from megatron.model.fused_layer_norm import MixedFusedRMSNorm as RMSNorm # for cuda
class RMSNorm(torch.nn.Module):  # for cpu
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = self.weight * hidden_states

        return hidden_states


class LlamaLMHead(MegatronModule):
    """Causal LM head for Llama

    Arguments:
        vocab_size: size of vocabulary.
        hidden_size: hidden size
        gather_output: wether output logits being gathered or not.
        init_method: init method for weight initialization
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 init_method,
                 parallel_output=True):
        super(LlamaLMHead, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.parallel_output = parallel_output

        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=self.hidden_size,
                                                output_size=vocab_size,
                                                bias=False,
                                                gather_output=not self.parallel_output,
                                                skip_bias_add=True,
                                                init_method=self.init_method, )

    def forward(self, inputs):
        logits, _ = self.lm_head(inputs)
        return logits


class LlamaEmbedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        init_method: weight initialization method
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 init_method):
        super(LlamaEmbedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(vocab_size, self.hidden_size,
                                                          init_method=self.init_method)

    def forward(self, input_ids):
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        return embeddings


class LlamaParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to intermediate
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, init_method, output_layer_init_method):
        super(LlamaParallelMLP, self).__init__()
        self.config = config

        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        # Project to intermediate.
        self.gate_proj = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=False,
            gather_output=False,
            init_method=self.init_method,
            skip_bias_add=True,
        )

        self.up_proj = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=False,
            gather_output=False,
            init_method=self.init_method,
            skip_bias_add=True,
        )

        self.activation_func = F.silu

        # Project back to h.
        self.down_proj = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True,
        )

    def forward(self, hidden_states):
        intermediate_parallel = self.gate_proj(hidden_states)[0] * self.up_proj(hidden_states)[0]

        intermediate_parallel = self.activation_func(intermediate_parallel)

        output, _ = self.down_proj(intermediate_parallel)
        return output


class CoreAttention(MegatronModule):

    def __init__(self, config, layer_number,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        self.config = config

        self.fp16 = config.fp16
        self.bf16 = config.bf16

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = config.sequence_parallel

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = core.utils.divide(projection_size,
                                                           world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            config.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            config.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # # Dropout. Note that for a single iteration, this layer will generate
        # # different outputs on different number of parallel partitions but
        # # on average it should not be partition dependent.
        # self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer,
                value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=key_layer.device)

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0 / self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class LlamaParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, config,
                 init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.causal):
        super(LlamaParallelAttention, self).__init__()

        assert attention_type == AttnType.self_attn
        assert attn_mask_type == AttnMaskType.causal

        self.config = config

        self.fp16 = config.fp16
        self.bf16 = config.bf16

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        self.num_attention_heads = config.num_attention_heads
        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = core.utils.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            config.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                3 * projection_size,
                bias=False,
                gather_output=False,
                init_method=self.init_method)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            config.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        ## Rotary Position Embedding
        self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head)

        self.core_attention = CoreAttention(self.config,
                                            self.layer_number,
                                            self.attn_mask_type)
        self.checkpoint_core_attention = config.recompute_granularity == 'selective'

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True)

    def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask):
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer,
                                          value_layer, attention_mask)
            return output_

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask, inference_params=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================

        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)

        # ==================================
        # Rotary Position Embedding
        # ==================================

        kv_seq_len = new_tensor_shape[0]
        if inference_params:
            kv_seq_len += inference_params.sequence_len_offset

        # [sq, b, np, hn] --> [b, np, sq, hn] TODO optimize the permute of dimension back and forth
        query_layer = query_layer.permute(1, 2, 0, 3)
        key_layer = key_layer.permute(1, 2, 0, 3)
        value_layer = value_layer.permute(1, 2, 0, 3)

        cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, offset=0)

        # [b, np, sq, hn] --> [sq, b, np, hn] TODO optimize the permute of dimension back and forth
        query_layer = query_layer.permute(2, 0, 1, 3).contiguous()
        key_layer = key_layer.permute(2, 0, 1, 3).contiguous()
        value_layer = value_layer.permute(2, 0, 1, 3).contiguous()

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
                                 batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
                                   batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[
                :sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                :sequence_end, batch_start:batch_end, ...]

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            context_layer = self._checkpointed_attention_forward(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output, _ = self.dense(context_layer)

        return output


class LlamaParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, config,
                 init_method, output_layer_init_method,
                 layer_number,
                 self_attn_mask_type=AttnMaskType.causal):
        self.config = config

        super(LlamaParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        assert self_attn_mask_type == AttnMaskType.causal

        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        # Layernorm on the input data.
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon)

        # Self attention.
        self.attention = LlamaParallelAttention(
            self.config,
            self.init_method,
            self.output_layer_init_method,
            layer_number,
            attn_mask_type=self_attn_mask_type)

        # Layernorm on the attention output
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon)

        # MLP
        self.mlp = LlamaParallelMLP(self.config, self.init_method, self.output_layer_init_method)

    def forward(self, hidden_states, attention_mask=None, inference_params=None):
        # hidden_states: [b, s, h]
        residual = hidden_states
        # Layer norm at the beginning of the transformer layer.
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention.
        hidden_states = self.attention(hidden_states,
                                       attention_mask,
                                       inference_params=inference_params)

        # Residual connection.
        hidden_states = hidden_states + residual
        residual = hidden_states

        # Layer norm post the self attention.
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP.
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class LlamaParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, config,
                 init_method, output_layer_init_method,
                 self_attn_mask_type=AttnMaskType.causal,
                 pre_process=True, post_process=True):

        super(LlamaParallelTransformer, self).__init__()
        self.config = config
        assert self_attn_mask_type == AttnMaskType.causal

        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        # Store activation checkpoiting flag.
        self.recompute_granularity = config.recompute_granularity
        self.recompute_method = config.recompute_method
        self.recompute_num_layers = config.recompute_num_layers
        self.distribute_saved_activations = \
            config.distribute_saved_activations and not config.sequence_parallel

        # Number of layers.
        # assert config.num_layers % parallel_state.get_pipeline_model_parallel_world_size() == 0, \
        #     'num_layers must be divisible by pipeline_model_parallel_size'
        # self.num_layers = config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
        self.num_layers = config.num_layers // parallel_state.get_pipeline_model_parallel_world_size() + \
                            int(parallel_state.get_pipeline_model_parallel_rank() < config.num_layers % parallel_state.get_pipeline_model_parallel_world_size())

        # Transformer layers.
        def build_layer(layer_number):
            return LlamaParallelTransformerLayer(
                self.config,
                self.init_method,
                self.output_layer_init_method,
                layer_number)

        if config.virtual_pipeline_model_parallel_size is not None:
            assert config.num_layers % config.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // config.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = parallel_state.get_virtual_pipeline_model_parallel_rank() * (
                    config.num_layers // config.virtual_pipeline_model_parallel_size) + \
                     (parallel_state.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            # offset = parallel_state.get_pipeline_model_parallel_rank() * self.num_layers
            offset = parallel_state.get_pipeline_model_parallel_rank() * (config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()) \
                      + min(parallel_state.get_pipeline_model_parallel_rank(), config.num_layers % parallel_state.get_pipeline_model_parallel_world_size())

        self.layers = []
        # Build the layers
        for i in range(self.num_layers):
            layer_num = i + 1 + offset
            self.layers.append(build_layer(layer_num))

        self.layers = torch.nn.ModuleList(self.layers)

        if self.post_process:
            # Final layer norm before output.
            self.final_layernorm = RMSNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_

            return custom_forward

        l = 0
        while l < self.num_layers:
            hidden_states = tensor_parallel.checkpoint(
                custom(l, l + self.recompute_num_layers),
                self.distribute_saved_activations,
                hidden_states, attention_mask)

            l += self.recompute_num_layers

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask, inference_params=None):

        if self.pre_process:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            # If the input flag for fp32 residual connection is set, convert for float.
            if self.fp32_residual_connection:
                hidden_states = hidden_states.transpose(0, 1).contiguous().float()
            # Otherwise, leave it as is.
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        batch_size, seq_length = hidden_states.size(1), hidden_states.size(0)
        past_key_values_length = 0
        if inference_params is not None:
            past_key_values_length = inference_params.sequence_len_offset
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), past_key_values_length)

        # Forward pass.
        if self.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(hidden_states, attention_mask)
        else:
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                hidden_states = layer(hidden_states,
                                      attention_mask=attention_mask,
                                      inference_params=inference_params)

        # Final layer norm.
        if self.post_process:
            # Reverting data format change [s b h] --> [b s h].
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states)
        else:
            output = hidden_states

        return output

    # Adopted from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.bool, # inputs_ids.dtype,
                device=torch.cuda.current_device(),
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # expanded_attn_mask = _expand_mask(attention_mask, inputs_ids.dtype, tgt_len=input_shape[-1]).to(
            expanded_attn_mask = _expand_mask(attention_mask, torch.bool, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else torch.max(expanded_attn_mask, combined_attention_mask)
            )
        return combined_attention_mask


def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    losses = tensor_parallel.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


class LlamaModel(MegatronModule):
    """llama Language model."""

    def __init__(self, config, pre_process, post_process, parallel_output=True):
        super(LlamaModel, self).__init__()
        args = get_args()
        self.config = config

        self.fp16_lm_cross_entropy = config.fp16_lm_cross_entropy
        self.hidden_size = config.hidden_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output

        self.init_method = init_method_normal(config.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(config.init_method_std, config.num_layers)
        self.self_attn_mask_type = AttnMaskType.causal
        self.padded_vocab_size = args.padded_vocab_size

        if self.pre_process:
            self.embedding = LlamaEmbedding(hidden_size=config.hidden_size,
                                            init_method=self.init_method,
                                            vocab_size=self.padded_vocab_size)

        # Transformer.
        self.transformer = LlamaParallelTransformer(
            self.config,
            self.init_method,
            self.output_layer_init_method,
            self_attn_mask_type=self.self_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        self._mpu = None

    def set_mpu(self, mpu):
        self._mpu = mpu

    def get_mpu(self):
        return self._mpu

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.transformer.set_input_tensor(input_tensor[0])

    def word_embeddings_weight(self):
        return self.embedding.word_embeddings.weight

    def forward(self, input_ids, position_ids, attention_mask, labels=None, inference_params=None):
        if self.pre_process:
            hidden_states = self.embedding(input_ids)
        else:
            hidden_states = input_ids

        # decoder
        hidden_states = self.transformer(hidden_states, attention_mask, inference_params=inference_params)

        return hidden_states


class LlamaForCausalLM(MegatronModule):

    def __init__(self, config, pre_process, post_process, parallel_output=True):
        super(LlamaForCausalLM, self).__init__(share_word_embeddings=False)
        args = get_args()
        self.config = config

        self.hidden_size = config.hidden_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output

        self.init_method = init_method_normal(config.init_method_std)
        self.padded_vocab_size = args.padded_vocab_size

        self.model = LlamaModel(config, pre_process, post_process, parallel_output=parallel_output)

        if self.post_process:
            self.lm_head = LlamaLMHead(hidden_size=config.hidden_size,
                                       vocab_size=self.padded_vocab_size,
                                       init_method=self.init_method,
                                       parallel_output=self.parallel_output)
        self._mpu = None

    def set_mpu(self, mpu):
        self._mpu = mpu

    def get_mpu(self):
        return self._mpu

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask, labels=None, inference_params=None):
        hidden_states = self.model(input_ids, position_ids, attention_mask, labels=labels, inference_params=inference_params)

        if self.post_process:
            hidden_states = self.lm_head(hidden_states)

            if labels is None:
                return hidden_states

            else:
                if self.fp16_lm_cross_entropy:
                    assert hidden_states.dtype == torch.half
                    loss = tensor_parallel.vocab_parallel_cross_entropy(hidden_states, labels)
                else:
                    loss = tensor_parallel.vocab_parallel_cross_entropy(hidden_states.float(), labels)
                return loss

        return hidden_states