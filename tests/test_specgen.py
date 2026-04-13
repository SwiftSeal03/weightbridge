"""Tests for :func:`wbridge.utils.specgen.infer_load_spec` with a composite ``lw``."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest
import torch

from wbridge.utils.data import Shards
from wbridge.utils.specgen import (
    DEFAULT_MAX_HF_BYTES,
    infer_load_spec,
    verify_load_spec,
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("infer_load_spec requires CUDA worker tensors")
    return torch.device("cuda", torch.cuda.current_device())


def _complex_sglang_like_lw(
    wksd: dict[str, torch.Tensor],
    *,
    V: int,
    H: int,
    INTER: int,
    tp_rank: int,
    q_dim: int,
    kv_dim: int,
) -> Callable[[Iterable[tuple[str, torch.Tensor]]], None]:
    """Mimics several SGLang / vLLM-style load paths (PP skip, padding, QKV merge, TP, transpose, tie).

    * **PP:** names prefixed with ``pp_skip.`` are ignored (weights not on this pipeline stage).
    * **Padded vocab:** ``embed_tokens`` / ``lm_head`` HF tensors are wider than the runtime vocab;
      only ``[:V, :]`` is copied (extra rows mirror padded ``lm_head`` in large checkpoints).
    * **QKV merge:** three HF matrices are written into non-overlapping row blocks of ``qkv_proj``
      (like ``stacked_params_mapping`` merging ``q_proj`` / ``k_proj`` / ``v_proj`` into ``qkv_proj``).
    * **TP column-parallel:** ``gate_proj`` shards along dim 0 (output features).
    * **TP row-parallel:** ``o_proj`` shards along dim 1 (input features split across ranks).
    * **Tied embeddings:** after the first full pass, iterate weights again and copy
      ``model.embed_tokens.weight`` into ``lm_head.weight`` (``qwen2``-style second scan).
    """

    half = H // 2
    assert tp_rank in (0, 1)
    qkv_rows = q_dim + 2 * kv_dim

    def _dispatch(name: str, t: torch.Tensor) -> None:
        if name.startswith("pp_skip."):
            return
        if name == "model.embed_tokens.weight":
            wksd["embed_tokens.weight"].copy_(t[:V, :])
            wksd["lm_head.weight"].copy_(t[:V, :])
            return
        if name.endswith("self_attn.q_proj.weight"):
            wksd["layers.1.self_attn.qkv_proj.weight"][:q_dim, :].copy_(t)
            return
        if name.endswith("self_attn.k_proj.weight"):
            wksd["layers.1.self_attn.qkv_proj.weight"][q_dim : q_dim + kv_dim, :].copy_(t)
            return
        if name.endswith("self_attn.v_proj.weight"):
            wksd["layers.1.self_attn.qkv_proj.weight"][q_dim + kv_dim : qkv_rows, :].copy_(t)
            return
        if name.endswith("self_attn.o_proj.weight"):
            wksd["layers.1.self_attn.o_proj.weight"].copy_(
                t[:, tp_rank * half : (tp_rank + 1) * half]
            )
            return
        if name.endswith("mlp.gate_proj.weight"):
            wksd["layers.1.mlp.gate_proj.weight"].copy_(
                t[tp_rank * half : (tp_rank + 1) * half, :]
            )
            return
        # if name == "model.triton_linear.weight":
        #     wksd["triton_linear.weight"].copy_(t.T.contiguous())
        #     return

    def lw(weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, t in weights:
            _dispatch(name, t)
        for name, t in weights:
            if name == "model.embed_tokens.weight":
                wksd["lm_head.weight"].copy_(t[:V, :])
                break

    return lw


def test_infer_shard_spec_complex_lw(device: torch.device) -> None:
    # V, V_PAD, H, INTER = 256, 32, 64, 128
    V, V_PAD, H, INTER = 8, 1, 4, 8
    tp_rank = 1
    q_dim, kv_dim = 64, 16
    qkv_rows = q_dim + 2 * kv_dim
    half = H // 2

    hfsd = {
        "pp_skip.model.layers.0.mlp.fc.weight": torch.randn(3, 3),
        "model.embed_tokens.weight": torch.randn(V + V_PAD, H),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(q_dim, H),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(kv_dim, H),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(kv_dim, H),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(H, H),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(H, INTER),
        # "model.triton_linear.weight": torch.randn(H // 2, H),
    }
    for t in hfsd.values():
        assert t.device.type == "cpu"

    wksd = {
        "embed_tokens.weight": torch.zeros(V, H, device=device),
        "lm_head.weight": torch.zeros(V, H, device=device),
        "layers.1.self_attn.qkv_proj.weight": torch.zeros(qkv_rows, H, device=device),
        "layers.1.self_attn.o_proj.weight": torch.zeros(H, half, device=device),
        "layers.1.mlp.gate_proj.weight": torch.zeros(half, INTER, device=device),
        # "triton_linear.weight": torch.zeros(H, H // 2, device=device),
    }

    lw = _complex_sglang_like_lw(
        wksd,
        V=V,
        H=H,
        INTER=INTER,
        tp_rank=tp_rank,
        q_dim=q_dim,
        kv_dim=kv_dim,
    )

    load_spec = infer_load_spec(hfsd.items(), wksd, lw)
    spec = load_spec.src_spec()

    assert "pp_skip.model.layers.0.mlp.fc.weight" not in spec.entries

    def _shard(name: str) -> Shards:
        return spec[name]

    assert _shard("model.embed_tokens.weight") == [[(0, V, V + V_PAD), (0, H, H)]]
    assert _shard("model.layers.1.self_attn.q_proj.weight") == [[(0, q_dim, q_dim), (0, H, H)]]
    assert _shard("model.layers.1.self_attn.k_proj.weight") == [[(0, kv_dim, kv_dim), (0, H, H)]]
    assert _shard("model.layers.1.self_attn.v_proj.weight") == [[(0, kv_dim, kv_dim), (0, H, H)]]
    assert _shard("model.layers.1.self_attn.o_proj.weight") == [
        [(0, H, H), (tp_rank * half, (tp_rank + 1) * half, H)]
    ]
    assert _shard("model.layers.1.mlp.gate_proj.weight") == [
        [(tp_rank * half, (tp_rank + 1) * half, H), (0, INTER, INTER)]
    ]
    # assert _shard("model.triton_linear.weight") == [[(0, H // 2, H // 2), (0, H, H)]]


def test_infer_load_spec_batched_merge_matches_single(device: torch.device) -> None:
    """Small max_hf_bytes forces multiple batches; merged spec matches one-shot."""
    V, H = 8, 4
    hfsd = {
        "a.weight": torch.randn(H, H),
        "b.weight": torch.randn(H, H),
        "c.weight": torch.randn(H, H),
    }
    wksd = {
        "a.weight": torch.zeros(H, H, device=device),
        "b.weight": torch.zeros(H, H, device=device),
        "c.weight": torch.zeros(H, H, device=device),
    }

    def lw(weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, t in weights:
            short = name.replace("model.", "") if name.startswith("model.") else name
            if short in wksd:
                wksd[short].copy_(t)

    # Rename keys to match wksd for trivial lw
    hfsd_named = {
        "a.weight": hfsd["a.weight"],
        "b.weight": hfsd["b.weight"],
        "c.weight": hfsd["c.weight"],
    }

    cap = 2 * hfsd_named["a.weight"].numel() * hfsd_named["a.weight"].element_size()
    assert cap < DEFAULT_MAX_HF_BYTES

    full = infer_load_spec(hfsd_named.items(), wksd, lw, max_hf_bytes=DEFAULT_MAX_HF_BYTES).src_spec()
    batched = infer_load_spec(hfsd_named.items(), wksd, lw, max_hf_bytes=cap).src_spec()

    assert set(full.entries.keys()) == set(batched.entries.keys())
    for name in full.entries:
        assert full[name] == batched[name]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="needs CUDA with bfloat16",
)
def test_infer_load_spec_hf_worker_dtype_mismatch(device: torch.device) -> None:
    """HF tensor bf16, worker fp32 (like SGLang e_score_correction_bias); infer + verify."""
    hfsd = {
        "model.layers.0.mlp.gate.e_score_correction_bias": torch.randn(
            8, dtype=torch.bfloat16, device="cpu"
        ),
    }
    wksd = {
        "layers.0.mlp.gate.e_score_correction_bias": torch.zeros(8, dtype=torch.float32, device=device),
    }

    def lw(weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, t in weights:
            if name.endswith("e_score_correction_bias"):
                wksd["layers.0.mlp.gate.e_score_correction_bias"].copy_(t)

    load_spec = infer_load_spec(hfsd.items(), wksd, lw)
    spec = load_spec.src_spec()
    name = "model.layers.0.mlp.gate.e_score_correction_bias"
    assert spec[name] == [[(0, 8, 8)]]

    hfsd_verify = {k: v.clone() for k, v in hfsd.items()}
    lw(hfsd_verify.items())
    verify_load_spec(hfsd_verify.items(), wksd, load_spec)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="needs CUDA with bfloat16",
)
def test_infer_load_spec_bfloat16(device: torch.device) -> None:
    """``infer_load_spec`` with bf16 HF + worker tensors at production-like sizes."""
    dt = torch.bfloat16

    # Large square weight (e.g. attention projection): 1024×1024, row-parallel TP2 on dim 1.
    H = 1024
    half_w = H // 2
    tp_rank_o = 0
    hfsd_o = {
        "model.layers.0.self_attn.o_proj.weight": torch.randn(H, H, dtype=dt, device="cpu"),
    }
    wksd_o = {
        "layers.0.self_attn.o_proj.weight": torch.zeros(H, half_w, dtype=dt, device=device),
    }

    def lw_o(weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, t in weights:
            if name.endswith("o_proj.weight"):
                wksd_o["layers.0.self_attn.o_proj.weight"].copy_(
                    t[:, tp_rank_o * half_w : (tp_rank_o + 1) * half_w]
                )

    spec_o = infer_load_spec(hfsd_o.items(), wksd_o, lw_o).src_spec()
    assert spec_o["model.layers.0.self_attn.o_proj.weight"] == [
        [(0, H, H), (tp_rank_o * half_w, (tp_rank_o + 1) * half_w, H)]
    ]

    # Wide matrix 16×151936 (vocab-scale last dim): column-parallel TP2 on dim 1.
    rows, cols = 16, 151_936
    half_c = cols // 2
    tp_rank_e = 1
    hfsd_e = {
        "model.embed_tokens.weight": torch.randn(rows, cols, dtype=dt, device="cpu"),
    }
    wksd_e = {
        "model.embed_tokens.weight": torch.zeros(rows, half_c, dtype=dt, device=device),
    }

    def lw_e(weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, t in weights:
            if name.endswith("embed_tokens.weight"):
                wksd_e["model.embed_tokens.weight"].copy_(
                    t[:, tp_rank_e * half_c : (tp_rank_e + 1) * half_c]
                )

    spec_e = infer_load_spec(hfsd_e.items(), wksd_e, lw_e).src_spec()
    assert spec_e["model.embed_tokens.weight"] == [
        [(0, rows, rows), (tp_rank_e * half_c, (tp_rank_e + 1) * half_c, cols)]
    ]


def test_verify_load_spec_1to1(device: torch.device) -> None:
    """After load, :func:`verify_load_spec` matches HF slices to worker slices from the LoadSpec."""
    H = 4
    hfsd = {
        "a.weight": torch.randn(H, H),
        "b.weight": torch.randn(H, H),
    }
    wksd = {
        "a.weight": torch.zeros(H, H, device=device),
        "b.weight": torch.zeros(H, H, device=device),
    }
    
    hfsd_verify = {k: v.clone() for k, v in hfsd.items()}

    def lw(weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, t in weights:
            wksd[name].copy_(t)

    lw(hfsd.items())
    load_spec = infer_load_spec(hfsd.items(), wksd, lw)
    spec = load_spec.src_spec()
    verify_load_spec(hfsd_verify.items(), wksd, load_spec)
