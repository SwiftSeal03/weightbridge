from __future__ import annotations

import json
import math
from typing import Iterator, TypeAlias
import torch

Triple: TypeAlias = tuple[int, int, int]
Shard: TypeAlias = list[tuple[int, int, int]]
Shards: TypeAlias = list[Shard]

def dtype_str_to_torch(s: str) -> torch.dtype:
    """Map a dtype string (e.g. ``\"float32\"``, ``\"torch.float32\"``) to ``torch.dtype``."""
    s = s.strip()
    if s.startswith("torch."):
        s = s.removeprefix("torch.")
    return getattr(torch, s)


def dtype_to_str(dtype: torch.dtype) -> str:
    """Canonical short string for a ``torch.dtype`` (e.g. ``\"float32\"``)."""
    return str(dtype).split(".")[-1]


def _normalize_shards(
    shard: Shard | Shards,
) -> Shards:
    """Normalize a shard spec to the multi-shard form ``[[dim, ...], ...]``.

    Accepts both single-shard ``[(l, r, w), ...]`` and multi-shard
    ``[[(l, r, w), ...], ...]`` inputs.
    """
    if not shard:
        return []
    if isinstance(shard[0][0], (int, float)):
        return [shard]
    return shard

def _shard_to_numel(shard: Shard) -> int:
    return math.prod(r - l for l, r, _ in shard)


def _entry_total_numel(entry: dict[str, ...]) -> int:
    return sum(_shard_to_numel(s) for s in entry["shard"])


def _shards_iterator(shards: Shards, offset: int = 0, item_size: int = 1) -> Iterator[tuple[int, int, Shard]]:
    for shard in shards:
        length = _shard_to_numel(shard) * item_size
        yield offset, offset + length, shard
        offset += length

class WeightData:
    """
    Shard metadata for weight transfer (no tensor storage).

    Format::

        {
            "name": {
                "shard": [[(l, r, w), ...], ...],  # always multi-shard form after init
                "dtype": str,  # e.g. ``\"float32\"``
            },
            ...
        }

    At construction, each ``shard`` value is normalized to the multi-shard
    form ``[[dim, ...], ...]`` (see :func:`_normalize_shards`). Callers may
    pass either single-shard ``[(l, r, w), ...]`` or multi-shard input.

    Tensor values are passed separately as ``dict[str, torch.Tensor]`` to
    :meth:`__call__` (returns a :class:`WeightTensorBridge`) or
    """

    def __init__(self, meta_dict: dict[str, dict[str, ...]]):
        self.meta_dict = {}
        for name, value in meta_dict.items():
            entry = dict(value)
            entry["shard"] = _normalize_shards(value["shard"])
            self.meta_dict[name] = entry
        self.sanity_check(self.meta_dict)

    def __getitem__(self, key: str) -> dict[str, ...]:
        return self.meta_dict[key]

    def __call__(self, tensors: dict[str, torch.Tensor]) -> WeightTensorBridge:
        """Bind *tensors* to this metadata for overlap packing / unpacking.

        Returns a :class:`WeightTensorBridge` ``f`` such that ``v = f[c]`` packs
        into ``len(c)`` flat ``uint8`` chunks (one per overlap), and
        ``f[c] = v`` copies wire-format chunks into *tensors* (see
        :class:`WeightTensorBridge`).
        """
        return WeightTensorBridge(self, tensors)

    def __str__(self) -> str:
        return json.dumps(self.meta_dict, default=str, indent=4)
    
    def __bool__(self) -> bool:
        return bool(self.meta_dict)

    def total_nbytes(self) -> int:
        """Total byte size of the data described by all shard entries."""
        total = 0
        for entry in self.meta_dict.values():
            dtype = dtype_str_to_torch(entry["dtype"])
            total += _entry_total_numel(entry) * dtype.itemsize
        return total

    def sanity_check(self, meta_dict: dict[str, dict[str, ...]]) -> None:
        for name, value in meta_dict.items():
            assert isinstance(value["dtype"], str), (
                f"{name}: dtype must be str, got {type(value['dtype'])}"
            )
            dtype_str_to_torch(value["dtype"])  # validate
            shards = value["shard"]
            assert len(shards) > 0, f"Empty shard list for {name}"
            for shard in shards:
                assert len(shard) > 0, f"Empty shard in list for {name}"
                numel = 1
                for l, r, w in shard:
                    assert 0 <= l < r <= w, f"Invalid shard: {l, r, w} for {name}"
                    numel *= r - l

    @staticmethod
    def compute_overlap(sender: "WeightData", receiver: "WeightData") -> "WeightData":
        """Return a new WeightData whose entries describe the shard regions
        where *sender* and *receiver* overlap (metadata only, no tensor data).

        Sender and receiver metadata must be :class:`WeightData` (shards
        normalized at construction). Every sender shard is paired against every
        receiver shard and all non-empty overlaps are collected.
        """
        result: dict[str, dict] = {}
        for name, s_entry in sender.meta_dict.items():
            if name not in receiver.meta_dict:
                continue

            r_entry = receiver[name]
            dtype = s_entry["dtype"]
            assert dtype == r_entry["dtype"], f"Dtype mismatch for {name}: {dtype} vs {r_entry['dtype']}"

            s_shards = s_entry["shard"]
            r_shards = r_entry["shard"]

            overlap_shards: list[list[tuple[int, int, int]]] = []
            for s_shard in s_shards:
                for r_shard in r_shards:
                    alignment = _check_shard_compatibility(s_shard, r_shard)
                    if alignment is None:
                        raise ValueError(f"Shard compatibility check failed for {name}")

                    overlap_dims = [
                        (max(ls, lr), min(rs, rr), w)
                        for ls, rs, lr, rr, w in alignment
                    ]
                    
                    if all(lo < hi for lo, hi, _ in overlap_dims):
                        overlap_shards.append(overlap_dims)

            if overlap_shards:
                result[name] = {"shard": overlap_shards, "dtype": dtype}

        return WeightData(result)

    def entries_with_intv(self) -> Iterator[tuple[int, int, str, dict]]:
        offset = 0
        for name, entry in self.meta_dict.items():
            length = _entry_total_numel(entry) * dtype_str_to_torch(entry["dtype"]).itemsize
            yield offset, offset + length, name, entry
            offset += length


class WeightTensorBridge:
    """``f = metadata(tensors)``: overlap-indexed pack/unpack into *tensors*.

    * ``v = f[c]`` — *c* is a :class:`WeightData` or ``list[WeightData]`` (overlap
      specs). Returns a ``list`` of ``len(c)`` one-dimensional ``uint8`` tensors
      (wire layout), each the packed overlap for that entry.
    * ``f[c] = v`` — *v* is a matching ``list`` of ``uint8`` flat tensors (or a
      single tensor when ``len(c) == 1``), copied into *tensors* at the overlap
      regions (receiver layout; *metadata* must describe *tensors*).

    :meth:`pack_for` performs one overlap's pack or unpack (see below).
    """

    __slots__ = ("_metadata", "_tensors")

    def __init__(self, metadata: WeightData, tensors: dict[str, torch.Tensor]) -> None:
        self._metadata = metadata
        self._tensors = tensors
        self.sanity_check_and_flatten()

    def sanity_check_and_flatten(self) -> None:
        for name, entry in self._metadata.items():
            assert name in self._tensors, f"Missing tensor {name} for overlap entry"
            assert self._tensors[name].dtype == dtype_str_to_torch(entry["dtype"]), (
                f"Tensor dtype mismatch for {name}: {self._tensors[name].dtype} vs {dtype_str_to_torch(entry['dtype'])}"
            )
            assert self._tensors[name].is_contiguous(), f"Tensor {name} is not contiguous"
            self._tensors[name] = self._tensors[name].flatten().view(torch.uint8)
            
        for name in self._tensors.keys():
            if name not in self._metadata:
                del self._tensors[name]

    @staticmethod
    def slice_copy(
        large: WeightTensorBridge, 
        small: WeightTensorBridge,
        l2s: bool = True
    ) -> None:
        """
        Copy data from one flattened buffer to another.
        The direction of the copy is determined by the `l2s` parameter.
        "small" should represent a subset of the data in "large".
        """
        for name, s_entry in small._metadata.items():
            assert name in large._metadata, f"Missing tensor {name} for large entry"
            l_entry = large._metadata[name]
            
            s_tensor = small._tensors[name]
            l_tensor = large._tensors[name]
            dtype = dtype_str_to_torch(s_entry["dtype"])
            assert l_tensor.dtype == dtype, f"Tensor dtype mismatch for {name}: {l_tensor.dtype} vs {dtype}"
            
            s_shards = s_entry["shard"]
            l_shards = l_entry["shard"]

            for s_byte_start, s_byte_end, s_shard in _shards_iterator(s_shards):
                for l_byte_start, l_byte_end, l_shard in _shards_iterator(l_shards):
                    alignment = _check_shard_compatibility(l_shard, s_shard)
                    if alignment is None:
                        continue
                    if not all(
                        ll <= ls and rs <= rl
                        for ll, rl, ls, rs, _ in alignment
                    ):
                        continue
                    
                    s_byte_start = s_byte_end
                    s_byte_end += _shard_to_numel(s_shard) * dtype.itemsize
                    
                    s_shape = [rs - ls for ls, rs, _ in s_shard]
                    l_shape = [rl - ll for ll, rl, _ in l_shard]
                    slices = tuple(
                        slice(ls - ll, rs - ll)
                        for ll, _, ls, rs, _ in alignment
                    )
                    l_typed = l_tensor[l_byte_start:l_byte_end].view(dtype).view(l_shape)
                    s_typed = s_tensor[s_byte_start:s_byte_end].view(dtype).view(s_shape)
                    
                    if l2s:
                        s_typed = l_typed[slices]
                    else:
                        l_typed[slices] = s_typed
                    break

    def __getitem__(
        self, dst_metas: dict[int, WeightData]
    ) -> dict[int, torch.Tensor]:
        """
        Pack for each dst_meta entry, extract the data from self._tensors and pack it into a single tensor.
        """
        
        dst_tensors = {}
        for rank, dst_meta in dst_metas.items():
            dst_tensor = torch.empty(dst_meta.total_nbytes(), dtype=torch.uint8)
            state_dict = {
                name: dst_tensor[start:end]
                for start, end, name, _ in dst_meta.entries_with_intv()
            }
            WeightTensorBridge.slice_copy(self, dst_meta(state_dict), l2s=True)
            dst_tensors[rank] = dst_tensor
        return dst_tensors

    def __setitem__(
        self,
        src_metas: dict[int, WeightData],
        src_tensors: dict[int, torch.Tensor],
    ) -> None:
        for rank, src_meta in src_metas.items():
            src_tensor = src_tensors[rank]
            state_dict = {
                name: src_tensor[start:end]
                for start, end, name, _ in src_meta.entries_with_intv()
            }
            WeightTensorBridge.slice_copy(self, src_meta(state_dict), l2s=False)


_SENTINEL = object()


def _check_shard_compatibility(
    s_shard: list[tuple[int, int, int]],
    r_shard: list[tuple[int, int, int]],
) -> list[tuple[int, int, int, int, int]] | None:
    """Check whether two shard specs can be aligned to a common dimensionality.

    A dimension ``(l*a, r*a, w*a)`` is equivalent to
    ``[(l, r, w), (0, a, a)]`` — trailing full dimensions can be split off
    or merged.  More generally, a contiguous range ``[l, r)`` within a
    single "row" of an outer dimension can also be split.

    Returns a list of ``(l1, r1, l2, r2, w)`` pairs: sender interval
    ``[l1, r1)``, receiver interval ``[l2, r2)``, shared width ``w`` per
    aligned dimension, or ``None`` if no valid alignment exists.
    """
    s_cur = next(iter(s_shard), _SENTINEL)
    r_cur = next(iter(r_shard), _SENTINEL)
    s_iter = iter(s_shard[1:])
    r_iter = iter(r_shard[1:])
    aligned: list[tuple[int, int, int, int, int]] = []

    while s_cur is not _SENTINEL and r_cur is not _SENTINEL:
        sl, sr, sw = s_cur
        rl, rr, rw = r_cur

        if sw == rw:
            aligned.append((sl, sr, rl, rr, sw))
            s_cur = next(s_iter, _SENTINEL)
            r_cur = next(r_iter, _SENTINEL)

        elif sw > rw:
            if sw % rw != 0:
                return None
            tail = sw // rw
            if sl % tail == 0 and sr % tail == 0:
                aligned.append((sl // tail, sr // tail, rl, rr, rw))
                s_cur = (0, tail, tail)
            elif sl // tail == (sr - 1) // tail:
                row = sl // tail
                aligned.append((row, row + 1, rl, rr, rw))
                s_cur = (sl - row * tail, sr - row * tail, tail)
            else:
                return None
            r_cur = next(r_iter, _SENTINEL)

        else:  # sw < rw
            if rw % sw != 0:
                return None
            tail = rw // sw
            if rl % tail == 0 and rr % tail == 0:
                aligned.append((sl, sr, rl // tail, rr // tail, sw))
                r_cur = (0, tail, tail)
            elif rl // tail == (rr - 1) // tail:
                row = rl // tail
                aligned.append((sl, sr, row, row + 1, sw))
                r_cur = (rl - row * tail, rr - row * tail, tail)
            else:
                return None
            s_cur = next(s_iter, _SENTINEL)

    if s_cur is not _SENTINEL or r_cur is not _SENTINEL:
        return None
    return aligned