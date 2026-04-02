"""Tests for :func:`wbridge.utils.specgen.infer_shard_spec` with a composite ``lw``."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest
import torch

from wbridge.utils.specgen import _LazyHfToDeviceWeights, infer_shard_spec


@pytest.fixture(scope="module")
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


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
    * **Transpose:** one linear uses ``.T.contiguous()`` (Triton / layout quirks).
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
            return
        if name == "lm_head.weight":
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
        if name == "model.triton_linear.weight":
            wksd["triton_linear.weight"].copy_(t.T.contiguous())
            return

    def lw(weights: _LazyHfToDeviceWeights | Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, t in weights:
            _dispatch(name, t)
        for name, t in weights:
            if name == "model.embed_tokens.weight":
                wksd["lm_head.weight"].copy_(t[:V, :])
                break

    return lw


def test_lazy_iterator_no_batch_gpu_transfer(device: torch.device, monkeypatch) -> None:
    """HF tensors stay on CPU until yielded; each tensor calls ``.to(device)`` only when consumed."""
    to_calls: list[tuple[int, torch.Size]] = []
    real_to = torch.Tensor.to

    def spy_to(self, *args, **kwargs):
        if args and args[0] == device:
            to_calls.append((id(self), self.shape))
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    hfsd = {
        "a": torch.randn(2, 3),
        "b": torch.randn(4, 5),
        "c": torch.randn(1, 1),
    }
    lazy = _LazyHfToDeviceWeights(hfsd, ["a", "b", "c"], device)
    assert to_calls == []

    first = list(lazy)
    assert len(to_calls) == 3
    assert len(first) == 3
    for _n, t in first:
        assert t.device == device
    assert all(v.device.type == "cpu" for v in hfsd.values())

    second = list(lazy)
    assert len(to_calls) == 6
    assert len(second) == 3


def test_infer_shard_spec_complex_lw(device: torch.device) -> None:
    V, V_PAD, H, INTER = 256, 32, 64, 128
    tp_rank = 1
    q_dim, kv_dim = 64, 16
    qkv_rows = q_dim + 2 * kv_dim
    half = H // 2

    hfsd = {
        "pp_skip.model.layers.0.mlp.fc.weight": torch.randn(3, 3),
        "model.embed_tokens.weight": torch.randn(V + V_PAD, H),
        "lm_head.weight": torch.randn(V + V_PAD, H),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(q_dim, H),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(kv_dim, H),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(kv_dim, H),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(H, H),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(H, INTER),
        "model.triton_linear.weight": torch.randn(H // 2, H),
    }
    for t in hfsd.values():
        assert t.device.type == "cpu"

    wksd = {
        "embed_tokens.weight": torch.zeros(V, H, device=device),
        "lm_head.weight": torch.zeros(V, H, device=device),
        "layers.1.self_attn.qkv_proj.weight": torch.zeros(qkv_rows, H, device=device),
        "layers.1.self_attn.o_proj.weight": torch.zeros(H, half, device=device),
        "layers.1.mlp.gate_proj.weight": torch.zeros(half, INTER, device=device),
        "triton_linear.weight": torch.zeros(H, H // 2, device=device),
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

    spec = infer_shard_spec(hfsd, wksd, lw, device=device)

    assert "pp_skip.model.layers.0.mlp.fc.weight" not in spec.entries

    def _shard(name: str) -> list:
        return spec[name]["shard"]

    # Tied pass writes the same HF tensor into ``lm_head`` as well → two worker targets.
    embed_shard = [(0, V, V + V_PAD), (0, H, H)]
    assert _shard("model.embed_tokens.weight") == [embed_shard, embed_shard]
    assert _shard("lm_head.weight") == [[(0, V, V + V_PAD), (0, H, H)]]
    assert _shard("model.layers.1.self_attn.q_proj.weight") == [[(0, q_dim, q_dim), (0, H, H)]]
    assert _shard("model.layers.1.self_attn.k_proj.weight") == [[(0, kv_dim, kv_dim), (0, H, H)]]
    assert _shard("model.layers.1.self_attn.v_proj.weight") == [[(0, kv_dim, kv_dim), (0, H, H)]]
    assert _shard("model.layers.1.self_attn.o_proj.weight") == [
        [(0, H, H), (tp_rank * half, (tp_rank + 1) * half, H)]
    ]
    assert _shard("model.layers.1.mlp.gate_proj.weight") == [
        [(tp_rank * half, (tp_rank + 1) * half, H), (0, INTER, INTER)]
    ]
    assert _shard("model.triton_linear.weight") == [[(0, H // 2, H // 2), (0, H, H)]]

    for _name, shards, dtype in spec:
        assert dtype == torch.float32
