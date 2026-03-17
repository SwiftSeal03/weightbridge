"""
SGLang to WeightBridge format conversion.
Converts SGLang TP-sharded state_dict to WeightData for weight bridging.
"""

from typing import Any, Dict, Optional

import torch

from wbridge.utils.data import WeightData

from .qwen2 import Qwen2Config, convert_sglang_qwen2_to_wb


def convert_to_wb(
    model_config: Any,
    state_dict: Dict[str, torch.Tensor],
    tp_rank: int,
    tp_size: int,
    attn_tp_rank: Optional[int] = None,
    attn_tp_size: Optional[int] = None,
) -> WeightData:
    """
    Route SGLang state_dict to the appropriate converter based on model type.

    Args:
        model_config: SGLang ModelConfig (has hf_config.model_type, hidden_size, etc.).
        state_dict: Model state_dict from model.state_dict().
        tp_rank: Tensor parallel rank.
        tp_size: Tensor parallel world size.
        attn_tp_rank: Attention TP rank (for Qwen3). Defaults to tp_rank.
        attn_tp_size: Attention TP size. Defaults to tp_size.

    Returns:
        WeightData with HF-style names and shard metadata.
    """
    model_type = getattr(
        getattr(model_config, "hf_config", None),
        "model_type",
        None,
    )
    if model_type is None:
        raise ValueError("model_config has no hf_config.model_type")

    model_type = str(model_type).lower()
    if model_type in ("qwen2", "qwen3"):
        config = Qwen2Config.from_model_config(model_config, tp_rank, tp_size, attn_tp_rank, attn_tp_size)
        return convert_sglang_qwen2_to_wb(
            config=config,
            state_dict=state_dict,
        )
    raise ValueError(f"Unsupported model_type for SGLang->WeightBridge: {model_type}")


__all__ = [
    "Qwen2Config",
    "convert_sglang_qwen2_to_wb",
    "convert_to_wb",
]
