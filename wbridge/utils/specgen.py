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

* Stage 1 uses random values in ``hfsd`` and zeros in ``wksd`` (same dtypes as
  provided).
* Stage 2 encodes flat indices into the HF tensor and relies on ``lw`` to copy
  them into the worker tensor. Integer worker tensors use ``torch.int64``
  indices and a ``-1`` sentinel; floating-point worker tensors use
  ``torch.float64`` indices and a ``-1.0`` sentinel (indices are exact in
  ``float64`` for ``numel <= 2**53``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import torch

from wbridge.utils.data import Shard, ShardSpec

LoadWeightsFn = Callable[[Iterable[tuple[str, torch.Tensor]]], Any]


class _LazyHfToDeviceWeights(Iterable[tuple[str, torch.Tensor]]):
    """Re-iterable (name, tensor) stream: each tensor is ``.to(device)`` on demand.

    HF tensors are expected on CPU; each iteration pass transfers one tensor at a
    time as it is yielded (no up-front list of all GPU tensors).
    """

    def __init__(
        self,
        hfsd: dict[str, torch.Tensor],
        names: Sequence[str],
        device: torch.device,
        *,
        overrides: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self._hfsd = hfsd
        self._names = tuple(names)
        self._device = device
        self._overrides = overrides or {}

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        d = self._device
        non_blocking = d.type == "cuda"
        for name in self._names:
            t = self._overrides[name] if name in self._overrides else self._hfsd[name]
            yield name, t.to(d, non_blocking=non_blocking)


def _infer_device(wksd: dict[str, torch.Tensor]) -> torch.device:
    if not wksd:
        raise ValueError("wksd must be non-empty to infer device")
    return next(iter(wksd.values())).device


def _zero_keys(sd: dict[str, torch.Tensor], keys: Iterable[str]) -> None:
    for k in keys:
        sd[k].zero_()


def _randomize_keys(sd: dict[str, torch.Tensor], keys: Iterable[str]) -> None:
    """Fill selected tensors with uniform (0, 1) values (avoids exact zeros)."""
    for k in keys:
        t = sd[k]
        t.uniform_(0.0, 1.0)


def _worker_nonzero(t: torch.Tensor) -> bool:
    return bool(torch.any(t != 0).item())


def _iter_hf_weights_to_device(
    hfsd: dict[str, torch.Tensor],
    names: Sequence[str],
    device: torch.device,
    *,
    overrides: dict[str, torch.Tensor] | None = None,
) -> _LazyHfToDeviceWeights:
    """Lazy iterable of ``(name, tensor_on_device)`` for ``lw`` (see :class:`_LazyHfToDeviceWeights`)."""
    return _LazyHfToDeviceWeights(hfsd, names, device, overrides=overrides)


def _match_hf_to_worker_names(
    hfsd: dict[str, torch.Tensor],
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
    device: torch.device,
) -> dict[str, set[str]]:
    """
    Stage 1: divide-and-conquer mapping from HF tensor name -> worker tensor
    names that receive data when those HF keys are loaded.
    """
    hf_keys = list(hfsd.keys())
    wk_keys = list(wksd.keys())
    mapping: dict[str, set[str]] = defaultdict(set)

    _randomize_keys(hfsd, hf_keys)
    _zero_keys(wksd, wk_keys)

    def recurse(h_subset: list[str], wk_candidates: set[str]) -> None:
        if not h_subset or not wk_candidates:
            return
        if len(h_subset) == 1:
            hf = h_subset[0]
            _zero_keys(wksd, wk_candidates)
            lw(_iter_hf_weights_to_device(hfsd, [hf], device))
            for wk in wk_candidates:
                if _worker_nonzero(wksd[wk]):
                    mapping[hf].add(wk)
            return

        mid = len(h_subset) // 2
        left, right = h_subset[:mid], h_subset[mid:]

        _zero_keys(wksd, wk_candidates)
        lw(_iter_hf_weights_to_device(hfsd, left, device))
        w_left = {wk for wk in wk_candidates if _worker_nonzero(wksd[wk])}

        _zero_keys(wksd, wk_candidates)
        lw(_iter_hf_weights_to_device(hfsd, right, device))
        w_right = {wk for wk in wk_candidates if _worker_nonzero(wksd[wk])}

        recurse(left, w_left)
        recurse(right, w_right)

    recurse(hf_keys, set(wk_keys))
    return dict(mapping)


def _linear_indices_to_shard(lin: torch.Tensor, hf_shape: tuple[int, ...]) -> Shard:
    """Bounding box (per-dimension min..max+1) for flattened HF indices."""
    if lin.dtype.is_floating_point:
        lin = lin[lin >= 0].double().flatten().cpu().round().long()
    else:
        lin = lin[lin >= 0].long().flatten().cpu()
    if lin.numel() == 0:
        raise ValueError("no non-negative indices recovered after lw probe")
    u = lin.unique()
    coords = torch.stack(torch.unravel_index(u, hf_shape))
    shard: Shard = []
    for d in range(len(hf_shape)):
        w = int(hf_shape[d])
        c = coords[d]
        l = int(c.min().item())
        r = int(c.max().item()) + 1
        shard.append((l, r, w))
    return shard


def _stage2_shard_for_pair(
    hf_name: str,
    hf_shape: tuple[int, ...],
    wk_name: str,
    wksd: dict[str, torch.Tensor],
    hfsd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
    device: torch.device,
) -> Shard:
    numel = int(torch.tensor(hf_shape, dtype=torch.int64).prod().item())
    wk = wksd[wk_name]
    if wk.dtype.is_floating_point:
        idx_cpu = torch.arange(numel, dtype=torch.float64).reshape(hf_shape)
    else:
        idx_cpu = torch.arange(numel, dtype=torch.int64).reshape(hf_shape)

    wk_backup = wk.clone()

    try:
        if wk.dtype.is_floating_point:
            wk.fill_(-1.0)
        else:
            wk.fill_(-1)
        lw(
            _iter_hf_weights_to_device(
                hfsd, [hf_name], device, overrides={hf_name: idx_cpu}
            )
        )
        marked = wk.detach().flatten().cpu()
        return _linear_indices_to_shard(marked, hf_shape)
    finally:
        wk.copy_(wk_backup)


def infer_shard_spec(
    hfsd: dict[str, torch.Tensor],
    wksd: dict[str, torch.Tensor],
    lw: LoadWeightsFn,
    *,
    device: torch.device | str | None = None,
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
        shard rectangles (bounding box of loaded indices) and the worker
        parameter dtype.
    """
    if not hfsd or not wksd:
        return ShardSpec({})
    if device is None:
        device = _infer_device(wksd)
    else:
        device = torch.device(device)

    hf_shapes = {k: tuple(v.shape) for k, v in hfsd.items()}
    wk_dtypes = {k: v.dtype for k, v in wksd.items()}
    wk_keys = list(wksd.keys())

    name_map = _match_hf_to_worker_names(hfsd, wksd, lw, device)
    _zero_keys(wksd, wk_keys)

    entries: dict[str, dict] = {}
    for hf_name, wk_names in name_map.items():
        if not wk_names:
            continue
        hf_shape = hf_shapes[hf_name]
        shards_list: list[Shard] = []
        dtype_set: set[torch.dtype] = set()
        for wk_name in sorted(wk_names):
            shard = _stage2_shard_for_pair(
                hf_name, hf_shape, wk_name, wksd, hfsd, lw, device
            )
            shards_list.append(shard)
            dtype_set.add(wk_dtypes[wk_name])
        if len(dtype_set) != 1:
            raise ValueError(
                f"inconsistent dtypes for HF key {hf_name!r} across workers: {dtype_set}"
            )
        dtype = next(iter(dtype_set))
        entries[hf_name] = {"shard": shards_list, "dtype": dtype}

    return ShardSpec(entries)


__all__ = ["LoadWeightsFn", "infer_shard_spec"]
