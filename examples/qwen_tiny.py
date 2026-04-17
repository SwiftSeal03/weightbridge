"""
Single-layer Qwen2-style HF checkpoint and loaders (no ``model.layers.N`` prefix).

HF tensors use split Q/K/V and gate/up (HuggingFace names). **Actor** ``wksd`` uses Megatron-Core
names/layout from ``slime/backends/megatron_utils/megatron_to_hf/qwen2.py`` (inverse of
``convert_qwen2_to_hf``):

* ``self_attention.linear_qkv.weight`` is **not** ``torch.cat([q,k,v], dim=0)``. It is packed as
  ``view(num_query_groups, -1, head_dim, hidden)`` with dim=1 split
  ``[value_num_per_group, 1, 1]`` where ``value_num_per_group = num_attention_heads // num_query_groups``
  (``num_query_groups == num_kv_heads`` for GQA).
* ``mlp.linear_fc1.weight`` is gate and up **row-stacked** (``chunk(2, dim=0)`` in the converter).

**Rollout** ``wksd`` keeps a runtime-friendly merged layout (``qkv_proj`` = Q||K||V along rows).

Wire :class:`~wbridge.ShardSpec` layouts live in ``workers.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QwenTinyConfig:
    vocab_size: int = 64
    hidden_size: int = 32
    intermediate_size: int = 48
    num_attn_heads: int = 4
    num_kv_heads: int = 2

    def __post_init__(self) -> None:
        if self.num_attn_heads % self.num_kv_heads != 0:
            raise ValueError("num_attn_heads must be divisible by num_kv_heads (GQA)")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attn_heads

    @property
    def q_out(self) -> int:
        return self.num_attn_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def num_query_groups(self) -> int:
        """Megatron ``num_query_groups`` (KV head groups)."""
        return self.num_kv_heads

    @property
    def value_num_per_group(self) -> int:
        """Queries per group: ``num_attention_heads // num_query_groups`` (see ``convert_qwen2_to_hf``)."""
        return self.num_attn_heads // self.num_query_groups

    @property
    def megatron_linear_qkv_rows(self) -> int:
        """Total output rows of ``self_attention.linear_qkv.weight`` (= ``q_out + 2 * kv_out``)."""
        return self.num_query_groups * (self.value_num_per_group + 2) * self.head_dim


DEFAULT_QWEN_TINY_CONFIG = QwenTinyConfig()


def hf_weight_keys() -> list[str]:
    return [
        "model.embed_tokens.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]


def build_qwen_tiny_hf_checkpoint(
    cfg: QwenTinyConfig,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Full in-memory HF checkpoint (CPU)."""
    g = torch.Generator(device=device).manual_seed(seed)
    h, iq, ikv = cfg.hidden_size, cfg.intermediate_size, cfg.kv_out
    qo = cfg.q_out
    v = cfg.vocab_size

    def R(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=dtype, device=device, generator=g)

    return {
        "model.embed_tokens.weight": R(v, h),
        "self_attn.q_proj.weight": R(qo, h),
        "self_attn.k_proj.weight": R(ikv, h),
        "self_attn.v_proj.weight": R(ikv, h),
        "self_attn.o_proj.weight": R(h, qo),
        "mlp.gate_proj.weight": R(iq, h),
        "mlp.up_proj.weight": R(iq, h),
        "mlp.down_proj.weight": R(h, iq),
        "input_layernorm.weight": R(h),
        "post_attention_layernorm.weight": R(h),
    }


def hf_qkv_weights_to_megatron_linear_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: QwenTinyConfig,
) -> torch.Tensor:
    """Inverse of ``convert_qwen2_to_hf`` for ``self_attention.linear_qkv.weight`` (2D weight)."""
    h = cfg.hidden_size
    n_g = cfg.num_query_groups
    vpg = cfg.value_num_per_group
    d = cfg.head_dim
    q_grp = q.view(n_g, vpg, d, h)
    k_grp = k.view(n_g, 1, d, h)
    v_grp = v.view(n_g, 1, d, h)
    stacked = torch.cat([q_grp, k_grp, v_grp], dim=1)
    return stacked.reshape(cfg.megatron_linear_qkv_rows, h)


def megatron_linear_qkv_to_hf_qkv(
    param: torch.Tensor,
    cfg: QwenTinyConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same split as ``convert_qwen2_to_hf`` (for tests / debugging)."""
    h = cfg.hidden_size
    d = cfg.head_dim
    n_g = cfg.num_query_groups
    vpg = cfg.value_num_per_group
    p = param.view(n_g, vpg + 2, d, h)
    q_param, k_param, v_param = torch.split(p, split_size_or_sections=[vpg, 1, 1], dim=1)
    q_param = q_param.reshape(-1, h)
    k_param = k_param.reshape(-1, h)
    v_param = v_param.reshape(-1, h)
    return q_param, k_param, v_param


def actor_param_shapes(cfg: QwenTinyConfig, tp_size: int) -> dict[str, tuple[int, ...]]:
    """Per-rank actor TP shard shapes (Megatron parameter names)."""
    assert cfg.vocab_size % tp_size == 0
    assert cfg.megatron_linear_qkv_rows % tp_size == 0
    assert cfg.q_out % tp_size == 0
    assert cfg.intermediate_size % tp_size == 0
    assert (2 * cfg.intermediate_size) % tp_size == 0
    v0 = cfg.vocab_size // tp_size
    qkv0 = cfg.megatron_linear_qkv_rows // tp_size
    q0 = cfg.q_out // tp_size
    i0 = cfg.intermediate_size // tp_size
    h = cfg.hidden_size
    iq2 = 2 * cfg.intermediate_size
    return {
        "embedding.word_embeddings.weight": (v0, h),
        "self_attention.linear_qkv.weight": (qkv0, h),
        "self_attention.linear_proj.weight": (h, q0),
        "mlp.linear_fc1.weight": (iq2 // tp_size, h),
        "mlp.linear_fc2.weight": (h, i0),
        "self_attention.linear_qkv.layer_norm_weight": (h,),
        "mlp.linear_fc1.layer_norm_weight": (h,),
    }


def build_actor_wksd(
    cfg: QwenTinyConfig,
    tp_rank: int,
    tp_size: int,
    *,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    del tp_rank  # symmetric shards; rank unused for allocation
    shapes = actor_param_shapes(cfg, tp_size)
    return {k: torch.empty(shapes[k], dtype=dtype, device=device) for k in shapes}


def actor_load_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    wksd: dict[str, torch.Tensor],
    cfg: QwenTinyConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    tp_rank: int,
    tp_size: int,
) -> None:
    wdict = dict(weights)
    for t in wksd.values():
        t.zero_()

    v_part = cfg.vocab_size // tp_size
    emb_rows = wdict["model.embed_tokens.weight"][tp_rank * v_part : (tp_rank + 1) * v_part]
    wksd["embedding.word_embeddings.weight"].copy_(emb_rows.to(device=device, dtype=dtype))

    qkv_full = hf_qkv_weights_to_megatron_linear_qkv(
        wdict["self_attn.q_proj.weight"],
        wdict["self_attn.k_proj.weight"],
        wdict["self_attn.v_proj.weight"],
        cfg,
    )
    wksd["self_attention.linear_qkv.weight"].copy_(
        torch.chunk(qkv_full, tp_size, dim=0)[tp_rank].to(device=device, dtype=dtype)
    )

    o = wdict["self_attn.o_proj.weight"]
    wksd["self_attention.linear_proj.weight"].copy_(
        torch.chunk(o, tp_size, dim=1)[tp_rank].to(device=device, dtype=dtype)
    )

    gate = wdict["mlp.gate_proj.weight"]
    up = wdict["mlp.up_proj.weight"]
    fc1 = torch.cat([gate, up], dim=0)
    wksd["mlp.linear_fc1.weight"].copy_(torch.chunk(fc1, tp_size, dim=0)[tp_rank].to(device=device, dtype=dtype))

    d = wdict["mlp.down_proj.weight"]
    wksd["mlp.linear_fc2.weight"].copy_(torch.chunk(d, tp_size, dim=1)[tp_rank].to(device=device, dtype=dtype))

    wksd["self_attention.linear_qkv.layer_norm_weight"].copy_(
        wdict["input_layernorm.weight"].to(device=device, dtype=dtype)
    )
    wksd["mlp.linear_fc1.layer_norm_weight"].copy_(
        wdict["post_attention_layernorm.weight"].to(device=device, dtype=dtype)
    )


def make_actor_load_weights(
    wksd: dict[str, torch.Tensor],
    cfg: QwenTinyConfig,
    *,
    device: str,
    dtype: torch.dtype,
    tp_rank: int,
    tp_size: int,
) -> Callable[[Iterable[tuple[str, torch.Tensor]]], None]:
    dev = torch.device(device)

    def lw(weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        actor_load_weights(weights, wksd, cfg, device=dev, dtype=dtype, tp_rank=tp_rank, tp_size=tp_size)

    return lw


def rollout_wksd_shapes(cfg: QwenTinyConfig) -> dict[str, tuple[int, ...]]:
    h, iq = cfg.hidden_size, cfg.intermediate_size
    qkv_r = cfg.megatron_linear_qkv_rows
    return {
        "model.embed_tokens.weight": (cfg.vocab_size, h),
        "self_attn.qkv_proj.weight": (qkv_r, h),
        "self_attn.o_proj.weight": (h, cfg.q_out),
        "mlp.gate_up_proj.weight": (2 * iq, h),
        "mlp.down_proj.weight": (h, iq),
        "input_layernorm.weight": (h,),
        "post_attention_layernorm.weight": (h,),
    }


def build_rollout_wksd(
    cfg: QwenTinyConfig,
    *,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    shapes = rollout_wksd_shapes(cfg)
    return {k: torch.empty(v, dtype=dtype, device=device) for k, v in shapes.items()}


def rollout_load_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    wksd: dict[str, torch.Tensor],
    cfg: QwenTinyConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    wdict = dict(weights)
    for t in wksd.values():
        t.zero_()

    wksd["model.embed_tokens.weight"].copy_(wdict["model.embed_tokens.weight"].to(device=device, dtype=dtype))

    q_w = wdict["self_attn.q_proj.weight"]
    k_w = wdict["self_attn.k_proj.weight"]
    v_w = wdict["self_attn.v_proj.weight"]
    packed = hf_qkv_weights_to_megatron_linear_qkv(q_w, k_w, v_w, cfg)
    wksd["self_attn.qkv_proj.weight"].copy_(packed.to(device=device, dtype=dtype))

    g_w = wdict["mlp.gate_proj.weight"]
    u_w = wdict["mlp.up_proj.weight"]
    wksd["mlp.gate_up_proj.weight"].copy_(torch.cat([g_w, u_w], dim=0).to(device=device, dtype=dtype))

    wksd["self_attn.o_proj.weight"].copy_(wdict["self_attn.o_proj.weight"].to(device=device, dtype=dtype))
    wksd["mlp.down_proj.weight"].copy_(wdict["mlp.down_proj.weight"].to(device=device, dtype=dtype))
    wksd["input_layernorm.weight"].copy_(wdict["input_layernorm.weight"].to(device=device, dtype=dtype))
    wksd["post_attention_layernorm.weight"].copy_(wdict["post_attention_layernorm.weight"].to(device=device, dtype=dtype))


def make_rollout_load_weights(
    wksd: dict[str, torch.Tensor],
    cfg: QwenTinyConfig,
    *,
    device: str,
    dtype: torch.dtype,
) -> Callable[[Iterable[tuple[str, torch.Tensor]]], None]:
    dev = torch.device(device)

    def lw(weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        rollout_load_weights(weights, wksd, cfg, device=dev, dtype=dtype)

    return lw


def actor_load_spec_path(load_spec_dir: str, rank: int) -> str:
    from pathlib import Path

    return str(Path(load_spec_dir) / f"actor_tp_rank{rank}.json")


def rollout_load_spec_path(load_spec_dir: str, rank: int) -> str:
    from pathlib import Path

    return str(Path(load_spec_dir) / f"rollout_rank{rank}.json")
