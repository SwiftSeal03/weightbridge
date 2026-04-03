"""
Infer a :class:`~wbridge.utils.data.ShardSpec` by probing how HuggingFace-named
weights map into a worker state dict through a load function ``lw``.

``lw`` must follow the same contract as SGLang ``model.load_weights``:

* **Input:** an iterable of ``(name, tensor)`` pairs. Implementations may
  iterate it multiple times (e.g. tied embeddings in ``qwen2.py``); this module
  passes a **re-iterable** lazy iterable that moves **one HF tensor at a time**
  from CPU to the target device so not all weights are staged on GPU at once.
  (If ``lw`` does ``list(weights)``, it will still hold every tensor it
  collects.)
* **Behavior:** for each pair, load the tensor into the corresponding worker
  parameter(s) (e.g. via ``default_weight_loader`` / TP-aware loaders in
  ``sglang.srt.model_loader.weight_utils``).

** dtypes **

* Stage 1 fills every ``hfsd`` tensor with ones and zeros ``wksd``, then
  divides HF names to discover which worker tensors receive each HF key.
* Stage 2 first loads ones to obtain a **mask** of affected worker elements,
  then transfers the flat HF index in several **chunks** whose numeric range
  fits the worker dtype (e.g. mantissa-sized chunks for FP16), reassembles an
  ``int64`` linear index per masked worker slot, and derives shard bounding
  boxes from those HF indices.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch

from wbridge.utils.data import Shard, ShardSpec

LoadWeightsFn = Callable[[Iterable[tuple[str, torch.Tensor]]], Any]

def _sd_subset_iterator(
    hfsd: dict[str, torch.Tensor],
    names: list[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    for name in names:
        yield name, hfsd[name]


def _worker_nonzero(t: torch.Tensor) -> bool:
    ret =  bool(torch.any(t != 0).item())
    if ret:
        t.zero_()
    return ret


def _match_hf_to_worker_names(
    hfsd: dict[str, torch.Tensor],
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
) -> dict[str, list[str]]:
    """
    Stage 1: divide-and-conquer mapping from HF tensor name -> worker tensor
    names that receive data when those HF keys are loaded.
    """
    hf_keys = list(hfsd.keys())
    wk_keys = list(wksd.keys())
    mapping: dict[str, list[str]] = defaultdict(list)

    for hf in hf_keys:
        hfsd[hf].fill_(1.0)
    for wk in wk_keys:
        wksd[wk].zero_()

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

    recurse(hf_keys, wk_keys)
    return dict(mapping)


def _extract_shard(
    hf_name: str,
    wk_name: str,
    hfsd: dict[str, torch.Tensor],
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
) -> Shard:
    hf_tensor = hfsd[hf_name]
    wk_tensor = wksd[wk_name]
    
    hf_tensor.fill_(1.0)
    wk_tensor.zero_()
    lw(_sd_subset_iterator(hfsd, [hf_name]))
    wk_mask = wk_tensor != 0
    
    hf_numel = hf_tensor.numel()
    wk_numel = wk_mask.sum()
    ele_bits = wk_tensor.dtype.itemsize * 8
    idx_bits = max(ele_bits, (hf_numel - 1).bit_length() + 1)
    idx_dtype : torch.dtype = getattr(torch, f"int{1 << (idx_bits.bit_length() - 1)}")
    int_dtype : torch.dtype = getattr(torch, f"int{ele_bits}")

    hf_indices = torch.arange(hf_numel, dtype=idx_dtype, device=hf_tensor.device)\
        .view(int_dtype).reshape(*hf_tensor.shape, -1)
    wk_indices = torch.zeros(wk_numel, dtype=idx_dtype, device=wk_tensor.device)\
        .view(int_dtype).reshape(wk_numel, -1)
        
    for k in range(hf_indices.shape[-1]):
        hf_tensor.copy_(hf_indices[..., k])
        lw(_sd_subset_iterator(hfsd, [hf_name]))
        wk_indices[..., k] = wk_tensor[wk_mask]
        
    wk_indices = wk_indices.view(idx_dtype).view(-1)
    coords = torch.unravel_index(wk_indices, hf_tensor.shape)
    
    shard = [
        (coords[d].min().item(), coords[d].max().item() + 1, w) 
        for d, w in enumerate(hf_tensor.shape)
    ]
    return shard
    
    
def infer_shard_spec(
    hfsd: dict[str, torch.Tensor],
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
) -> ShardSpec:
    """
    Run the two-stage name / data matching procedure and build a
    :class:`~wbridge.utils.data.ShardSpec` keyed by HuggingFace tensor names.

    Modifies ``hfsd`` in place (stage 1 randomizes all HF tensors). ``wksd`` is
    left all-zero after stage 1; each stage-2 pair restores the touched worker
    tensor from a backup. Pass copies if you need original ``hfsd`` values
    preserved.

    Args:
        hfsd: HuggingFace-style state dict (CPU tensors); each tensor is moved to
            ``device`` only when ``lw`` iterates over that name.
        wksd: Worker tensors that ``lw`` writes into (typically GPU), same keys
            the loader targets.
        lw: Same role as ``model.load_weights`` in SGLang (iterable of
            ``(str, Tensor)``).
        device: Device for tensors passed into ``lw``. Defaults to the device
            of the first tensor in ``wksd``.

    Returns:
        ``ShardSpec`` describing, for each HF name with a detected mapping, the
        shard rectangles and dtype (from ``hfsd``).
    """
    assert hfsd and wksd, "hfsd and wksd must be non-empty"
    assert all(v.is_cpu for v in hfsd.values()), "hfsd must be on the CPU"
    assert all(v.is_cuda for v in wksd.values()), "wksd must be on the GPU"

    wk_dtypes = {k: v.dtype for k, v in wksd.items()}

    name_map = _match_hf_to_worker_names(hfsd, wksd, lw)

    entries: dict[str, dict] = {}
    for hf_name, wk_names in name_map.items():
        dtype = hfsd[hf_name].dtype
        assert all(dtype == wk_dtypes[wk_name] for wk_name in wk_names), \
            f"inconsistent dtypes for HF key {hf_name!r}"
        shards = list({
            tuple(_extract_shard(hf_name, wk_name, hfsd, wksd, lw))
            for wk_name in sorted(wk_names)
        })
        entries[hf_name] = {"shard": shards, "dtype": dtype}

    return ShardSpec(entries)


__all__ = ["LoadWeightsFn", "infer_shard_spec"]
