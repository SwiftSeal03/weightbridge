"""
Megatron to WeightBridge (WB) format conversion.
Mirrors slime megatron_to_hf structure; outputs format suitable for WeightData.
"""

import torch

from .qwen2 import convert_qwen2_to_wb


def convert_to_wb(args, model_name: str, named_tensors: list[tuple[str, torch.nn.Parameter]], quantization_config=None):
    """
    Convert a Megatron parameter to WeightBridge format.
    Returns list of (name, param) tuples (HF-style names, possibly split params).
    """
    weight_data = _convert_to_wb_core(args, model_name, named_tensors)

    # TODO: support quantization_config when needed
    return weight_data


def _convert_to_wb_core(args, model_name: str, named_tensors: list[tuple[str, torch.nn.Parameter]]):
    if "qwen2" in model_name or "qwen3" in model_name:
        return convert_qwen2_to_wb(args, named_tensors)
    raise ValueError(f"Unsupported model for convert_to_wb: {model_name}")
