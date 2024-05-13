from typing import List, Optional, Tuple, Union

import torch

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(1, device=device), device=device)
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

# Adopted from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length):
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
        expanded_attn_mask = _expand_mask(attention_mask, torch.bool, tgt_len=input_shape[-1]).to(
            torch.cuda.current_device()
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else torch.max(expanded_attn_mask, combined_attention_mask)
        )
    return combined_attention_mask