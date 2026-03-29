"""
Qwen2/Qwen3 Megatron to WeightBridge format conversion.
Mirrors slime megatron_to_hf/qwen2.py structure.
"""

import re

import torch

from megatron.core import mpu
from wbridge.utils.data import WeightData, dtype_to_str

def convert_split_qwen2_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups
    local_n_query_groups = args.num_query_groups // mpu.get_tensor_model_parallel_world_size()

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":

            param = param.view(local_n_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(local_n_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


def convert_qwen2_to_wb_and_tensors(
    args, named_tensors: list[tuple[str, torch.nn.Parameter]]
) -> tuple[WeightData, dict[str, torch.Tensor]]:
    """HF-style metadata (``WeightData``) and matching local tensor shards for send."""
    meta_dict: dict[str, dict] = {}
    tensors: dict[str, torch.Tensor] = {}
    tprk = mpu.get_tensor_model_parallel_rank()
    tpws = mpu.get_tensor_model_parallel_world_size()
    vocab_size = args.vocab_size
    for name, param in named_tensors:
        converted = convert_split_qwen2_to_hf(args, name, param.data)
        part_dim = param.partition_dim
        for hf_name, hf_param in converted:
            t = hf_param.data
            shard = [
                (0, d, d) if i != part_dim else
                (tprk * d, (tprk + 1) * d, d * tpws)
                for i, d in enumerate(hf_param.shape)
            ]

            if "embed_token" in hf_name or "lm_head" in hf_name:
                l, r, w = shard[0]
                if l >= vocab_size:
                    continue
                r = min(vocab_size, r)
                w = min(vocab_size, w)
                shard[0] = (l, r, w)
                t = t[:r - l]

            meta_dict[hf_name] = {"shard": shard, "dtype": dtype_to_str(hf_param.dtype)}
            tensors[hf_name] = t

    return WeightData(meta_dict), tensors
