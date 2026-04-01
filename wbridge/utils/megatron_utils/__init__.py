"""
Megatron to WeightBridge (WB) format conversion.
Mirrors slime megatron_to_hf structure; outputs format suitable for :class:`~wbridge.utils.data.ShardSpec`.
"""

import torch

from .qwen2 import convert_qwen2_to_wb


def convert_to_wb(args, model_name: str, named_tensors: list[tuple[str, torch.nn.Parameter]], quantization_config=None):
    """
    Convert Megatron parameters to WeightBridge format.

    Returns:
        ``(ShardSpec, dict[str, Tensor])`` — shard metadata and local tensor shards for send.
    """
    # TODO: support quantization_config when needed
    return _convert_to_wb_core(args, model_name, named_tensors)


def _convert_to_wb_core(args, model_name: str, named_tensors: list[tuple[str, torch.nn.Parameter]]):
    if "qwen2" in model_name or "qwen3" in model_name:
        return convert_qwen2_to_wb(args, named_tensors)
    raise ValueError(f"Unsupported model for convert_to_wb: {model_name}")
