"""
Qwen2/Qwen3 SGLang state_dict to WeightBridge format conversion.
Mirrors wbridge/utils/megatron_utils/qwen2.py structure; converts SGLang's
TP-sharded state_dict to WeightData with HF-style names and shard metadata.

SGLang partitioning (from sglang/srt/models/qwen2.py, qwen3.py, layers/linear.py):
- Column parallel (shard dim 0): qkv_proj, gate_up_proj, embed_tokens, lm_head
- Row parallel (shard dim 1): o_proj, down_proj
- Replicated: input_layernorm, post_attention_layernorm, norm, q_norm, k_norm
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from wbridge.utils.data import WeightData

import logging
logger = logging.getLogger(__name__)


@dataclass
class Qwen2Config:
    """Model config for Qwen2/Qwen3. Matches HuggingFace config fields."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    tp_size: int
    tp_rank: int
    attn_tp_rank: int
    attn_tp_size: int
    head_dim: Optional[int] = None  # default: hidden_size // num_attention_heads

    @classmethod
    def from_model_config(
        cls, 
        model_config: Any, 
        tp_rank: int, 
        tp_size: int, 
        attn_tp_rank: Optional[int] = None, 
        attn_tp_size: Optional[int] = None,
    ) -> "Qwen2Config":
        """Build from SGLang ModelConfig (has hf_config, hidden_size, etc.)."""
        hf = getattr(model_config, "hf_text_config", model_config.hf_config)
        return cls(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=getattr(
                model_config, "num_key_value_heads", model_config.num_attention_heads
            ),
            intermediate_size=getattr(hf, "intermediate_size", model_config.hidden_size * 4),
            vocab_size=model_config.vocab_size,
            num_hidden_layers=model_config.num_hidden_layers,
            head_dim=getattr(model_config, "head_dim", None),
            tp_size=tp_size,
            tp_rank=tp_rank,
            attn_tp_rank=attn_tp_rank if attn_tp_rank is not None else tp_rank,
            attn_tp_size=attn_tp_size if attn_tp_size is not None else tp_size,
        )


def convert_split_qwen2_to_hf(config: Qwen2Config, name: str, param: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    if name == "model.embed_tokens.weight":
        return [("model.embed_tokens.weight", param, 0)]
    if name == "lm_head.weight":
        return [("lm_head.weight", param, 0)]
    if name == "model.norm.weight":
        return [("model.norm.weight", param, None)]
    
    head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_qheads_per_group = num_heads // num_kv_heads
    local_num_groups = num_kv_heads // config.attn_tp_size
    
    
    layer_match = re.match(r"model\.layers\.(\d+)\.(.+)", name)
    if layer_match:
        layer_idx, rest = layer_match.groups()
        layer_prefix = f"model.layers.{layer_idx}"
        
        if rest == "self_attn.qkv_proj.weight":
            param = param.view(local_num_groups, -1, head_dim, config.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[num_qheads_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, config.hidden_size)
            k_param = k_param.reshape(-1, config.hidden_size)
            v_param = v_param.reshape(-1, config.hidden_size)
            return [
                (f"{layer_prefix}.self_attn.q_proj.weight", q_param, 0),
                (f"{layer_prefix}.self_attn.k_proj.weight", k_param, 0),
                (f"{layer_prefix}.self_attn.v_proj.weight", v_param, 0),
            ]
        elif rest == "self_attn.qkv_proj.bias":
            param = param.view(local_num_groups, -1)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[num_qheads_per_group, 1, 1], dim=1)
            q_param = q_param.contiguous().flatten()
            k_param = k_param.contiguous().flatten()
            v_param = v_param.contiguous().flatten()
            return [
                (f"{layer_prefix}.self_attn.q_proj.bias", q_param, 0),
                (f"{layer_prefix}.self_attn.k_proj.bias", k_param, 0),
                (f"{layer_prefix}.self_attn.v_proj.bias", v_param, 0),
            ]
        elif rest == "mlp.gate_up_proj.weight":
            gate_param, up_param = param.chunk(2, dim=0)
            return [
                (f"{layer_prefix}.mlp.gate_proj.weight", gate_param, 0),
                (f"{layer_prefix}.mlp.up_proj.weight", up_param, 0),
            ]
        elif rest == "mlp.down_proj.weight" or rest == "self_attn.o_proj.weight":
            return [(name, param, 1)]
        else:
            return [(name, param, None)]
            
            
def convert_sglang_qwen2_to_wb(
    config: Qwen2Config,
    state_dict: Dict[str, torch.Tensor],
) -> WeightData:
    """
    Convert SGLang Qwen2/Qwen3 state_dict to WeightData.
    """
    out_meta_dict: Dict[str, Dict] = {}

    for sgl_name, sgl_param in state_dict.items():
        if not isinstance(sgl_param, torch.Tensor):
            continue

        # Skip non-weight tensors
        if "inv_freq" in sgl_name or "cos_cached" in sgl_name or "sin_cached" in sgl_name:
            continue
        if "projector" in sgl_name or "vision_tower" in sgl_name:
            continue
        
        converted = convert_split_qwen2_to_hf(config, sgl_name, sgl_param)
        for name, param, partition_dim in converted:
            tp_rank, tp_size = config.tp_rank, config.tp_size
            if any(attn_proj in name for attn_proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                tp_rank, tp_size = config.attn_tp_rank, config.attn_tp_size

            # Create shard metadata
            if partition_dim is None:
                shard = [(0, d, d) for d in param.shape]
            else:
                shard = []
                for i, d in enumerate(param.shape):
                    if i != partition_dim:
                        shard.append((0, d, d))
                    else:
                        shard_size = d  # local shard size
                        total = d * tp_size
                        l = tp_rank * shard_size
                        r = (tp_rank + 1) * shard_size
                        shard.append((l, r, total))
            
            # Handle vocab size truncation
            if config.vocab_size is not None and ("embed_tokens" in name or "lm_head" in name):
                l, r, w = shard[0]
                if l >= config.vocab_size:
                    continue
                r = min(config.vocab_size, r)
                w = min(config.vocab_size, w)
                shard = [(l, r, w)] + shard[1:]

            out_meta_dict[name] = {"shard": shard, "dtype": param.dtype}

    return WeightData(out_meta_dict)
