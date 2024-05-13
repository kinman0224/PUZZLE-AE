import dataclasses
import torch
import torch.nn.functional as F

from megatron import get_args

from .transformer_config import TransformerConfig

def core_transformer_config_from_args(args):

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(TransformerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype

    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_gelu_fusion'] = False
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_

    return TransformerConfig(**kw_args)

def core_transformer_config_from_hf_config(hf_config):
    args = get_args()

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(TransformerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)

    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype

    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_gelu_fusion'] = False
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_

    # update attributes from hf_config
    assert hf_config is not None
    kw_args['hidden_size'] = hf_config.hidden_size
    kw_args['num_attention_heads'] = hf_config.num_attention_heads
    kw_args['num_layers'] = hf_config.num_hidden_layers
    kw_args['ffn_hidden_size'] = hf_config.intermediate_size

    return TransformerConfig(**kw_args)

__all__ = [
    'core_transformer_config_from_args',
    'core_transformer_config_from_hf_config'
]