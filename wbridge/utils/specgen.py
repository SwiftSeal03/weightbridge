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
import torch.nn.functional as F
import logging

from wbridge.utils.data import (
    Shard,
    ShardMapping,
    LoadSpec,
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
        if sname not in load_spec.entries:
            # logger.warning(f"verify_load_spec: source {sname!r} not in load_spec")
            continue
        for dname, mappings in load_spec.entries[sname].items():
            assert dname in expected, f"verify_load_spec: destination {dname!r} not in wksd"
            for s_shard, d_shard in mappings:
                src_slices = tuple(slice(l, r) for l, r, _ in s_shard)
                dst_slices = tuple(slice(l, r) for l, r, _ in d_shard)
                expected[dname][dst_slices].copy_(hf_tensor[src_slices])

    for k, v in wksd.items():
        assert torch.equal(expected[k].to(v.device), v), f"verify_load_spec: mismatch on worker key {k}"
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


def _extract_shard_mapping(
    hf_name: str,
    wk_name: str,
    hfsd: dict[str, torch.Tensor],
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
) -> list[tuple[Shard, Shard]]:
    """Bounding box in HF tensor index space for one (HF name, worker param) pair."""
    # View the feed and worker tensors as integers
    hf_tensor = hfsd[hf_name]
    wk_tensor = wksd[wk_name]
    wk_dtype = wk_tensor.dtype
    ele_bits = wk_dtype.itemsize * 8
    int_dtype = getattr(torch, f"int{ele_bits}")
    n_dim = wk_tensor.dim()

    # Get mask of worker tensors that are affected by the HF tensor
    wk_tensor.zero_()
    lw(iter([(hf_name, hf_tensor)]))    
    wk_mask = wk_tensor != 0
    
    # Create auxiliary index tensors
    hf_numel = hf_tensor.numel()
    idx_bits = max(ele_bits, (hf_numel - 1).bit_length() + 1)
    idx_dtype: torch.dtype = getattr(torch, f"int{1 << (idx_bits - 1).bit_length()}")
    hf_indices = torch.arange(hf_numel, dtype=idx_dtype, device=hf_tensor.device).view(
        int_dtype
    ).reshape(*hf_tensor.shape, -1)
    wk_indices = torch.zeros_like(wk_tensor, dtype=idx_dtype, device=wk_tensor.device).view(
        int_dtype
    ).reshape(*wk_tensor.shape, -1)

    # Know source of each affected element in the worker tensor
    hf_int_tensor = torch.zeros_like(hf_tensor, dtype=int_dtype, device=hf_tensor.device)
    wk_int_tensor = wk_tensor.view(dtype=int_dtype)
    for k in range(hf_indices.shape[-1]):
        hf_int_tensor.copy_(hf_indices[..., k])
        lw(iter([(hf_name, hf_int_tensor.view(wk_dtype))]))
        wk_indices[..., k] = wk_int_tensor
    wk_indices = wk_indices.view(idx_dtype).squeeze(-1)
    
    # NOTE for high-dimensional tensors, we still use the simple way
    if n_dim > 2:
        def coords_to_shard(coords: torch.Tensor) -> Shard:
            return [
                (coords[d].min().item(), coords[d].max().item() + 1, w)
                for d, w in enumerate(coords.shape)
            ]
        wk_coords = torch.nonzero(wk_mask).transpose(0, 1)
        wk_shard = coords_to_shard(wk_coords)
        hf_coords = torch.unravel_index(wk_indices, hf_tensor.shape) 
        hf_shard = coords_to_shard(hf_coords)
        return [(hf_shard, wk_shard)]
    
    # Special handling for 1D tensor
    if n_dim == 1:
        # Wether each element's source succeeds it's left neighbor's source
        succ_left = wk_indices == F.pad(wk_indices, (1, 0), value=-2)[:-1] + 1
        
        # NOTE: Here we assume no segment has a length of 1
        # -1 indicates beginning of 1D shard, 0 indicates middle of 1D shard, 1 indicates end of 1D shard
        delta_succ_left = succ_left ^ F.pad(succ_left, (0, 1), value=0)[1:]
        wk_shard_ends = delta_succ_left.nonzero().view(-1, 2).tolist()
                            
        shard_mappings = []
        hf_ids = wk_indices[tuple(wk_shard_ends)].tolist()
        for i, (l, r) in enumerate(wk_shard_ends):
            shard_mappings.append((
                [(hf_ids[i * 2], hf_ids[i * 2 + 1] + 1, hf_tensor.shape[0])],
                [(l, r + 1, wk_tensor.shape[0])]
            ))
        return shard_mappings
    
    # Special handling for 2D tensor
    if n_dim == 2:
        # Wether each element's source succeeds it's up/left neighbor's source
        hf_h, hf_w = hf_tensor.shape
        wk_rows_ids = wk_indices // hf_w
        wk_cols_ids = wk_indices % hf_w
        succ_up = wk_rows_ids == F.pad(wk_rows_ids, (0, 0, 1, 0), value=-2)[:-1] + 1
        succ_left = wk_cols_ids == F.pad(wk_cols_ids, (1, 0, 0, 0), value=-2)[:, :-1] + 1
        succ_up_left = succ_up & succ_left
        
        # NOTE: Here we assume no segment has a width or height of 1
        # delta values of 1 indicates 2 corners of a 2D shard
        padded = F.pad(succ_up_left, (0, 1, 0, 1), value=0)
        delta = succ_up_left ^ padded[1:, :-1] ^ padded[:-1, 1:] ^ padded[1:, 1:]
        corners = delta.nonzero(as_tuple=True)
        values = delta[corners].tolist()
        hf_ids = wk_indices[corners]
        corners = torch.stack(corners, dim=1).tolist()
        coord_2_id = { (x, y): i for i, (x, y) in enumerate(corners) }

        shard_mappings = []
        for i, (x0, y0) in enumerate(corners): # top left corner
            if not values[i]:
                continue
            hf_id = hf_ids[i].item()
            hf_x0, hf_y0 = divmod(hf_id, hf_w)
            x1, y1 = None, None
            for j in range(i + 1, len(corners)): # top right corner
                if corners[j][0] == x0:
                    values[j] = False
                    y1 = corners[j][1]
                    break
            for j in range(i + 1, len(corners)): # bottom left corner
                if corners[j][1] == y0:
                    values[j] = False
                    x1 = corners[j][0]
                    break
            assert x1 is not None and y1 is not None, f"2D shard corner not found for top left corner {x0}, {y0}"
            assert (x1, y1) in coord_2_id, f"2D shard not rectangular, ({x0}, {y0}) -> ({x1}, {y1})"
            k = coord_2_id[(x1, y1)]
            hf_x1, hf_y1 = divmod(hf_ids[k].item(), hf_w)
            values[k] = False
            shard_mappings.append((
                [(hf_x0, hf_x1 + 1, hf_h), (hf_y0, hf_y1 + 1, hf_w)],
                [(x0, x1 + 1, wk_tensor.shape[0]), (y0, y1 + 1, wk_tensor.shape[1])]
            ))
        return shard_mappings
            


def _infer_shard_spec_from_hfsd(
    hfsd: dict[str, torch.Tensor],
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
) -> dict[str, dict[str, list[ShardMapping]]]:
    """Run stage 1+2 on one in-memory HF batch; returns spec dict (not wrapped)."""
    import time
    start_time = time.time()
    name_map = _match_hf_to_worker_names(hfsd, wksd, lw)
    end_time = time.time()
    logging.info(f"Time taken to match HF to worker names: {end_time - start_time} seconds")
    logging.info(f"name_map: {name_map}")

    start_time = end_time
    entries = {
        hf_name: { 
            wk_name: _extract_shard_mapping(hf_name, wk_name, hfsd, wksd, lw)
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
        merged: dict[str, dict[str, list[ShardMapping]]] = {}
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
