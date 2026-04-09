"""
Infer a :class:`~wbridge.utils.data.ShardSpec` by probing HF ``(name, tensor)``
streams through ``lw`` (same contract as SGLang ``model.load_weights``).

Stage 1 maps each HF name to worker keys; stage 2 derives axis-aligned HF shard
boxes per (HF name, worker key). :func:`sharded_hf_weights_iter` masks HF tensors
to those boxes (or drops unknown names). :func:`verify_load_spec` re-applies
*hfsd* through :class:`~wbridge.utils.data.LoadSpec` into fresh zero tensors
shaped like *wksd*, then compares them to *wksd* (after ``lw``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
import logging

from wbridge.utils.data import (
    Shard,
    Shards,
    ShardSpec,
    LoadSpec,
    _normalize_shards,
)

WeightsIterable = Iterable[tuple[str, torch.Tensor]]
LoadWeightsFn = Callable[[WeightsIterable], Any]

# Default cap on total CPU storage for HF placeholder tensors per infer batch.
DEFAULT_MAX_HF_BYTES = 20 * 1024**3

logger = logging.getLogger(__name__)

def verify_load_spec(
    hf_iterator: WeightsIterable,
    wksd: dict[str, torch.Tensor],
    load_spec: LoadSpec,
) -> None:
    """Assume ``lw`` was already run so *wksd* holds the loaded weights.

    For each worker key, build a zero tensor matching *wksd*[*key*], copy *hfsd*
    slices into it according to *load_spec*, then ``torch.equal`` against *wksd*.
    """
    assert wksd, "wksd must be non-empty"
    expected = {k: torch.zeros_like(v, device="cpu") for k, v in wksd.items()}

    for sname, hf_tensor in hf_iterator:
        for dname, (sshard, dshard) in load_spec.entries[sname].items():
            assert len(sshard) == len(dshard)
            assert dname in expected, f"verify_load_spec: destination {dname!r} not in wksd"
            src_slices = tuple(slice(l, r) for l, r, _ in sshard)
            dst_slices = tuple(slice(l, r) for l, r, _ in dshard)
            expected[dname][dst_slices].copy_(hf_tensor[src_slices])

    for k, v in wksd.items():
        assert torch.equal(expected[k].to(v.device), v), f"verify_load_spec: mismatch on worker key {k!r}"
    logger.info("LoadSpec verification succeeded")


def _sd_subset_iterator(
    hfsd: dict[str, torch.Tensor],
    names: list[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    for name in names:
        yield name, hfsd[name]


def _worker_nonzero(t: torch.Tensor) -> bool:
    ret = bool(torch.any(t != 0).item())
    if ret:
        t.zero_()
    return ret


def _match_hf_to_worker_names(
    hfsd: dict[str, torch.Tensor],
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
) -> dict[str, list[str]]:
    """HF name → worker names that receive nonzeros when those HF keys are loaded (divide & conquer)."""
    hf_keys = list(hfsd.keys())
    wk_keys = list(wksd.keys())
    mapping: dict[str, list[str]] = defaultdict(list)

    for hf in hf_keys:
        hfsd[hf].fill_(1.0)
    for wk in wk_keys:
        wksd[wk].zero_()
        
    lw(_sd_subset_iterator(hfsd, hf_keys))
    wk_candidates = {wk for wk in wk_keys if _worker_nonzero(wksd[wk])}

    def recurse(h_subset: list[str], wk_candidates: list[str]) -> None:
        if not h_subset or not wk_candidates:
            return
        if len(h_subset) == 1:
            mapping[h_subset[0]] = wk_candidates
            return

        mid = len(h_subset) // 2
        left, right = h_subset[:mid], h_subset[mid:]

        lw(_sd_subset_iterator(hfsd, left))
        w_left = {wk for wk in wk_candidates if _worker_nonzero(wksd[wk])}

        lw(_sd_subset_iterator(hfsd, right))
        w_right = {wk for wk in wk_candidates if _worker_nonzero(wksd[wk])}

        recurse(left, w_left)
        recurse(right, w_right)

    recurse(hf_keys, wk_candidates)
    return dict(mapping)


def _extract_shard(
    hf_name: str,
    wk_name: str,
    hfsd: dict[str, torch.Tensor],
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
) -> Shard:
    """Bounding box in HF tensor index space for one (HF name, worker param) pair."""
    hf_src = hfsd[hf_name]
    wk_param = wksd[wk_name]
    wk_dtype = wk_param.dtype

    # HF checkpoints may store bf16 while the worker keeps fp32 (e.g. MoE correction bias).
    # Int-bit probing must use the worker element width; lw sees a feed tensor in wk_dtype.
    if hf_src.dtype != wk_dtype:
        feed = hf_src.detach().clone().to(wk_dtype).contiguous()
    else:
        feed = hf_src

    # View the feed and worker tensors as integers
    ele_bits = wk_dtype.itemsize * 8
    int_dtype = getattr(torch, f"int{ele_bits}")
    feed_v = feed.view(dtype=int_dtype)
    wk_v = wk_param.view(dtype=int_dtype)

    # Get mask of worker tensors that are affected by the HF tensor
    wk_param.zero_()
    lw(iter([(hf_name, feed)]))
    wk_mask = wk_param != 0
    coords = torch.nonzero(wk_mask).transpose(0, 1)
    wk_shard = [
        (coords[d].min().item(), coords[d].max().item() + 1, w)
        for d, w in enumerate(wk_param.shape)
    ]

    hf_numel = feed_v.numel()
    wk_numel = wk_mask.sum()
    assert wk_numel > 0, "no worker tensors affected by HF tensor"

    # Create auxiliary index tensors
    idx_bits = max(ele_bits, (hf_numel - 1).bit_length() + 1)
    idx_dtype: torch.dtype = getattr(torch, f"int{1 << (idx_bits - 1).bit_length()}")
    hf_indices = torch.arange(hf_numel, dtype=idx_dtype, device=feed.device).view(
        int_dtype
    ).reshape(*feed.shape, -1)
    wk_indices = torch.zeros(wk_numel, dtype=idx_dtype, device=wk_param.device).view(
        int_dtype
    ).reshape(wk_numel, -1)

    # Know source of each affected element in the worker tensor
    for k in range(hf_indices.shape[-1]):
        feed_v.copy_(hf_indices[..., k])
        lw(iter([(hf_name, feed)]))
        wk_indices[..., k] = wk_v[wk_mask]

    wk_indices = wk_indices.view(idx_dtype).view(-1)
    coords = torch.unravel_index(wk_indices, feed.shape) 

    hf_shard = [
        (coords[d].min().item(), coords[d].max().item() + 1, w)
        for d, w in enumerate(feed.shape)
    ]
    
    return hf_shard, wk_shard


def _infer_shard_spec_from_hfsd(
    hfsd: dict[str, torch.Tensor],
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
) -> dict[str, dict[str, tuple[Shard, Shard]]]:
    """Run stage 1+2 on one in-memory HF batch; returns spec dict (not wrapped)."""
    import time
    start_time = time.time()
    name_map = _match_hf_to_worker_names(hfsd, wksd, lw)
    end_time = time.time()
    logging.info(f"Time taken to match HF to worker names: {end_time - start_time} seconds")

    start_time = end_time
    entries = {
        hf_name: { 
            wk_name: _extract_shard(hf_name, wk_name, hfsd, wksd, lw)
            for wk_name in wk_names
        } for hf_name, wk_names in name_map.items()
    }
    return entries


def infer_load_spec(
    hf_iterator: WeightsIterable,
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
    *,
    max_hf_bytes: int = DEFAULT_MAX_HF_BYTES,
) -> LoadSpec:
    """Infer merged :class:`~wbridge.utils.data.LoadSpec`; chunk HF CPU placeholders by *max_hf_bytes*.

    Overwrites *wksd* during probing, then restores it from a CPU snapshot.
    Call :func:`verify_load_spec` after a full ``lw`` pass if you need to check the mapping.
    """
    assert wksd, "wksd must be non-empty"
    assert all(v.is_cuda for v in wksd.values()), "wksd tensors must be on CUDA"

    backup = {k: v.detach().cpu() for k, v in wksd.items()}
    try:
        merged: dict[str, dict[str, tuple[Shard, Shard]]] = {}
        batch: dict[str, torch.Tensor] = {}
        batch_bytes = 0

        for name, tensor in hf_iterator:
            assert tensor.is_cpu, "tensor must be on CPU"
            if name in batch or name in merged:
                raise ValueError(f"duplicate HF tensor name in iterator: {name!r}")
            batch[name] = tensor
            batch_bytes += tensor.nbytes
            if batch and batch_bytes + tensor.nbytes > max_hf_bytes:
                merged |= _infer_shard_spec_from_hfsd(batch, wksd, lw)
                batch, batch_bytes = {}, 0

        if batch:
            merged |= _infer_shard_spec_from_hfsd(batch, wksd, lw)

        result = LoadSpec(merged)
    finally:
        for k, v in backup.items():
            wksd[k].copy_(v, non_blocking=True)

    return result


__all__ = [
    "DEFAULT_MAX_HF_BYTES",
    "LoadWeightsFn",
    "WeightsIterable",
    "infer_load_spec",
    "sharded_hf_weights_iter",
    "verify_load_spec",
]
