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


def _normalize_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Normalize ``dtype`` to :class:`torch.dtype` (accepts short strings, ``\"torch.*\"``, or dtype)."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return dtype_str_to_torch(dtype)
    raise TypeError(f"dtype must be torch.dtype or str, got {type(dtype)}")


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


def original_total_numel(shards: Shards) -> int:
    return _shard_to_numel(shards[0])


def shards_iterator(shards: Shards, offset: int = 0, item_size: int = 1) -> Iterator[tuple[int, int, Shard]]:
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
                "dtype": torch.dtype,  # stored as ``torch.dtype`` (str accepted at construction)
            },
            ...
        }

    At construction, each ``shard`` value is normalized to the multi-shard
    form ``[[dim, ...], ...]`` (see :func:`_normalize_shards`). Callers may
    pass either single-shard ``[(l, r, w), ...]`` or multi-shard input.
    ``dtype`` may be :class:`torch.dtype` or a string (for JSON / legacy).

    Tensor values are passed separately as ``dict[str, torch.Tensor]`` to
    :meth:`__call__` (returns a :class:`WeightTensorBridge`) or
    """

    def __init__(self, meta_dict: dict[str, dict[str, ...]] | bytes):
        if isinstance(meta_dict, bytes):
            meta_dict = json.loads(meta_dict.decode("utf-8"))
        self.meta_dict = {
            name: {
                "shard": _normalize_shards(value["shard"]),
                "dtype": _normalize_dtype(value["dtype"]),
            } for name, value in meta_dict.items()
        }

        # Sanity check
        for name, shards, dtype in self:
            assert isinstance(dtype, torch.dtype), f"Invalid dtype: {dtype}"
            numel = original_total_numel(shards)
            assert len(shards) > 0, f"Empty shard list for {name}"
            for shard in shards:
                assert len(shard) > 0, f"Empty shard in list for {name}"
                for l, r, w in shard:
                    assert 0 <= l < r <= w, f"Invalid shard: {l, r, w} for {name}"
                assert original_total_numel([shard]) == numel, \
                    f"Shard {shard} does not match original total numel: {numel}"

    def __call__(self, tensors: dict[str, torch.Tensor]) -> WeightTensorBridge:
        """Bind *tensors* to this metadata for overlap packing / unpacking.

        Returns a :class:`WeightTensorBridge` for ``f[overlaps]`` /
        ``f[overlaps] = chunks`` (see :class:`WeightTensorBridge`).
        """
        return WeightTensorBridge(self, tensors)
    
    def to_jsonable(self) -> dict[str, dict]:
        return {
            name: {
                "shard": shards,
                "dtype": dtype_to_str(dtype)
            } for name, shards, dtype in self
        }

    def __str__(self) -> str:
        return json.dumps(self.to_jsonable(), default=str)
    
    def __bytes__(self) -> bytes:
        return str(self).encode("utf-8")
    
    def __bool__(self) -> bool:
        return bool(self.meta_dict)
    
    def __iter__(self) -> Iterator[tuple[str, Shards, torch.dtype]]:
        for name, entry in self.meta_dict.items():
            yield name, entry["shard"], entry["dtype"]
    
    def __len__(self) -> int:
        return len(self.meta_dict)
    
    def __contains__(self, key: str) -> bool:
        return key in self.meta_dict
    
    def __getitem__(self, key: str) -> dict[str, ...]:
        return self.meta_dict[key]
    
    def __setitem__(self, key: str, value: dict[str, ...]) -> None:
        v = dict(value)
        v["dtype"] = _normalize_dtype(v["dtype"])
        v["shard"] = _normalize_shards(v["shard"])
        self.meta_dict[key] = v
    
    def iter_with_intv(self) -> Iterator[tuple[int, int, str, Shards, torch.dtype]]:
        offset = 0
        for name, shards, dtype in self:
            length = original_total_numel(shards) * dtype.itemsize
            yield offset, offset + length, name, shards, dtype
            offset += length

    def total_nbytes(self) -> int:
        """Total byte size of the data described by all shard entries."""
        total = 0
        for _, shards, dtype in self:
            total += original_total_numel(shards) * dtype.itemsize
        return total
        
    @staticmethod
    def compute_overlap(sender: "WeightData", receiver: "WeightData") -> "WeightData":
        """Return a new WeightData whose entries describe the shard regions
        where *sender* and *receiver* overlap (metadata only, no tensor data).

        Sender and receiver metadata must be :class:`WeightData` (shards
        normalized at construction). Every sender shard is paired against every
        receiver shard and all non-empty overlaps are collected.
        """
        result: dict[str, dict] = {}
        for name, s_shards, dtype in sender:
            if name not in receiver:
                continue
            r_entry = receiver[name]
            r_shards, r_dtype = r_entry["shard"], r_entry["dtype"]
            assert dtype == r_dtype, f"Dtype mismatch for {name}: {dtype} vs {r_dtype}"

            overlap_shards: Shards = []
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

        # Sanity check and flatten
        for name, shards, dtype in self._metadata:
            assert name in self._tensors, f"Missing tensor {name} for overlap entry"
            tensor = self._tensors[name]
            assert tensor.dtype == dtype, (
                f"Tensor dtype mismatch for {name}: {tensor.dtype} vs {dtype}"
            )
            assert tensor.is_contiguous(), f"Tensor {name} is not contiguous"
            assert original_total_numel(shards) == tensor.numel(), \
                f"Tensor {name} does not match original total numel: {original_total_numel(shards)} vs {tensor.numel()}"
            self._tensors[name] = tensor.flatten().view(torch.uint8)

        # Remove tensors that are not in the metadata
        for name in list(self._tensors):
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
        l_offsets = {}
        for l_byte_start, l_byte_end, name, _, _ in large._metadata.iter_with_intv():
            l_offsets[name] = l_byte_start
            
        
        for s_offset, _, name, s_shards, dtype in small._metadata.iter_with_intv():
            s_tensor = small._tensors[name]

            assert name in large._metadata, f"Missing tensor {name} for large entry"
            l_offset = l_offsets[name]
            l_shards = large._metadata[name]["shard"]
            l_tensor = large._tensors[name]
            assert l_tensor.dtype == dtype, f"Tensor dtype mismatch for {name}: {l_tensor.dtype} vs {dtype}"
            
            for s_byte_start, s_byte_end, s_shard in shards_iterator(
                s_shards, offset=s_offset, item_size=dtype.itemsize
            ):
                for l_byte_start, l_byte_end, l_shard in shards_iterator(
                    l_shards, offset=l_offset, item_size=dtype.itemsize
                ):
                    alignment = _check_shard_compatibility(l_shard, s_shard)
                    if alignment is None:
                        continue
                    if not all(
                        ll <= ls and rs <= rl
                        for ll, rl, ls, rs, _ in alignment
                    ):
                        continue

                    s_shape = [rs - ls for ls, rs, _ in s_shard]
                    l_shape = [rl - ll for ll, rl, _ in l_shard]
                    slices = tuple(
                        slice(ls - ll, rs - ll)
                        for ll, _, ls, rs, _ in alignment
                    )
                    l_typed = l_tensor[l_byte_start:l_byte_end].view(dtype).view(l_shape)
                    s_typed = s_tensor[s_byte_start:s_byte_end].view(dtype).view(s_shape)

                    if l2s:
                        s_typed.copy_(l_typed[slices])
                    else:
                        l_typed[slices] = s_typed
                    break

    def __getitem__(
        self, dst_metas: dict[int, WeightData]
    ) -> dict[int, torch.Tensor]:        
        dst_tensors = {}
        for rank, dst_meta in dst_metas.items():
            dst_tensor = torch.empty(dst_meta.total_nbytes(), dtype=torch.uint8)
            state_dict = {
                name: dst_tensor[start:end]
                for start, end, name, _, _ in dst_meta.iter_with_intv()
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
                for start, end, name, _, _ in src_meta.iter_with_intv()
            }
            WeightTensorBridge.slice_copy(self, src_meta(state_dict), l2s=False)


def _check_shard_compatibility(
    s_shard: Shard, r_shard: Shard
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
    s_cur = next(iter(s_shard), None)
    r_cur = next(iter(r_shard), None)
    s_iter = iter(s_shard[1:])
    r_iter = iter(r_shard[1:])
    aligned: list[tuple[int, int, int, int, int]] = []

    while s_cur is not None and r_cur is not None:
        sl, sr, sw = s_cur
        rl, rr, rw = r_cur

        if sw == rw:
            aligned.append((sl, sr, rl, rr, sw))
            s_cur = next(s_iter, None)
            r_cur = next(r_iter, None)

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
            r_cur = next(r_iter, None)

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
            s_cur = next(s_iter, None)

    if s_cur is not None or r_cur is not None:
        return None
    return aligned
