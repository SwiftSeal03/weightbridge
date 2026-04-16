from __future__ import annotations

import json
import math
from typing import Iterator, TypeAlias
import torch

Shard: TypeAlias = list[tuple[int, int, int]]
Shards: TypeAlias = list[Shard]
# One HF↔worker region pair: source box and destination box (same flattened numel).
ShardMapping: TypeAlias = tuple[Shard, Shard]

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


def shard_shape(shard: Shard) -> tuple[int, ...]:
    return tuple(r - l for l, r, _ in shard)


def shard_numel(shard: Shard) -> int:
    return math.prod(shard_shape(shard))


def shards_numel(shards: Shards) -> int:
    return sum(shard_numel(shard) for shard in shards)
        

def _sanity_check(name: str, shards: Shards) -> None:
    numel = None
    assert len(shards) > 0, f"Empty shard list for {name}"
    for shard in shards:
        assert len(shard) > 0, f"Empty shard in list for {name}"
        for l, r, w in shard:
            assert 0 <= l < r <= w, f"Invalid shard: {l, r, w} for {name}"
        cur_numel = math.prod(w for _, _, w in shard)
        if numel is None:
            numel = cur_numel
        else:
            assert numel == cur_numel, f"Shard {shard} does not match original total numel: {numel} != {cur_numel}!"

def shards_iterator(shards: Shards, tensor: torch.Tensor) -> Iterator[tuple[int, int, Shard]]:
    """Iterate over shards and yield the corresponding tensor slices."""
    offset = 0
    for shard in shards:
        length = shard_numel(shard)
        yield shard, tensor[offset:offset+length].view(shard_shape(shard))
        offset += length

class ShardSpec:
    """
    Per-tensor shard layout and dtype for weight transfer (no tensor storage).

    Format::

        {
            "name": [[(l, r, w), ...], ...],  # always multi-shard form after init
            ...
        }

    At construction, each ``shard`` value is normalized to the multi-shard
    form ``[[dim, ...], ...]`` (see :func:`_normalize_shards`). Callers may
    pass either single-shard ``[(l, r, w), ...]`` or multi-shard input.
    ``dtype`` may be :class:`torch.dtype` or a string (for JSON / legacy).

    Tensor values are passed separately as ``dict[str, torch.Tensor]`` to
    :meth:`__call__` (returns a :class:`BoundShardSpec`) or
    """

    def __init__(self, entries: dict[str, Shards]):
        self.entries = {
            name: _normalize_shards(shards) for name, shards in entries.items()
        }
        
        for name, shards in self:
            _sanity_check(name, shards)

    def __call__(self, tensors: dict[str, torch.Tensor]) -> "BoundShardSpec":
        """Bind *tensors* to this shard spec for overlap packing / unpacking.

        Returns a :class:`BoundShardSpec` for ``f[overlaps]`` /
        ``f[overlaps] = chunks`` (see :class:`BoundShardSpec`).
        """
        return BoundShardSpec(self, tensors)
    
    def __bool__(self) -> bool:
        return bool(self.entries)
    
    def __iter__(self) -> Iterator[tuple[str, Shards]]:
        return iter(self.entries.items())
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __contains__(self, key: str) -> bool:
        return key in self.entries
    
    def __delitem__(self, key: str) -> None:
        del self.entries[key]
    
    def __getitem__(self, key: str) -> Shards:
        return self.entries[key]
    
    def __setitem__(self, key: str, value: Shards) -> None:
        shards = _normalize_shards(value)
        _sanity_check(key, shards)
        self.entries[key] = shards
        
    def nbytes(self, tensors: dict[str, torch.Tensor]) -> int:
        return sum(shards_numel(shards) * tensors[name].element_size() for name, shards in self)
    
    def iter_with_intv(self, tensors: dict[str, torch.Tensor]) -> Iterator[tuple[int, int, str]]:
        offset = 0
        for name, shards in self:
            length = shards_numel(shards) * tensors[name].element_size()
            yield offset, offset + length, name
            offset += length
        
    @staticmethod
    def compute_overlap(sender: "ShardSpec", receiver: "ShardSpec") -> "ShardSpec":
        """Return a new :class:`ShardSpec` whose entries describe the shard regions
        where *sender* and *receiver* overlap (spec only, no tensor data).

        Sender and receiver specs must be :class:`ShardSpec` (shards
        normalized at construction). Every sender shard is paired against every
        receiver shard and all non-empty overlaps are collected.
        """
        result: dict[str, dict] = {}
        for name, s_shards in sender:
            if name not in receiver:
                continue
            r_shards = receiver[name]

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
                result[name] = overlap_shards

        return ShardSpec(result)


class BoundShardSpec:
    """``f = spec(tensors)``: overlap-indexed pack/unpack into *tensors*.

    * ``v = f[c]`` — *c* maps ranks to overlap :class:`ShardSpec` entries.
      Returns a ``dict`` of one-dimensional ``uint8`` tensors (wire layout).
    * ``f[c] = v`` — *v* maps ranks to matching ``uint8`` flat tensors, copied
      into *tensors* at the overlap regions (receiver layout; *shard_spec* must
      describe *tensors*).
    """

    def __init__(self, shard_spec: ShardSpec, tensors: dict[str, torch.Tensor]) -> None:
        self._shard_spec = shard_spec
        self._tensors = dict(tensors)
        self.device = tensors[list(tensors.keys())[0]].device

        # Sanity check and flatten
        for name, shards in self._shard_spec:
            assert name in self._tensors, f"Missing tensor {name} for overlap entry"
            tensor = self._tensors[name]
            assert tensor.is_contiguous(), f"Tensor {name} is not contiguous"
            assert tensor.dim() == 1, f"Tensor {name} must be 1D"
            assert shards_numel(shards) == tensor.numel(), \
                f"spec and tensor numel mismatch for {name}, {shards_numel(shards)} vs {tensor.numel()}"
            self._tensors[name] = tensor.flatten().view(torch.uint8)
            assert tensor.device == self.device, f"Tensor {name} is not on the same device as other tensors"
            
        # Drop tensors not listed in shard_spec
        for name in list(self._tensors):
            if name not in self._shard_spec:
                del self._tensors[name]

    @staticmethod
    def slice_copy(
        large: "BoundShardSpec",
        small: "BoundShardSpec",
        big2small: bool = True
    ) -> None:
        """
        Copy data from one flattened buffer to another.
        The direction of the copy is determined by the `l2s` parameter.
        "small" should represent a subset of the data in "large".
        """        
        for name, _ in small._shard_spec:
            s_tensor = small._tensors[name]
            assert name in large._shard_spec, f"Missing tensor {name} for large entry"
            b_tensor = large._tensors[name]
            
            assert b_tensor.dtype == s_tensor.dtype, \
                f"Tensor dtype mismatch for {name}: {b_tensor.dtype} vs {s_tensor.dtype}"
            
            for s_shard, s_tensor in shards_iterator(small._shard_spec[name], small._tensors[name]):
                for b_shard, b_tensor in shards_iterator(large._shard_spec[name], large._tensors[name]):
                    alignment = _check_shard_compatibility(b_shard, s_shard)
                    
                    if not alignment or not all(
                        bl <= sl and sr <= br
                        for bl, br, sl, sr, _ in alignment
                    ):
                        continue

                    slices = tuple(
                        slice(sl - bl, sr - bl)
                        for bl, _, sl, sr, _ in alignment
                    )
                    if big2small:
                        s_tensor.copy_(b_tensor[slices])
                    else:
                        b_tensor[slices] = s_tensor
                    break

    def __getitem__(
        self, dst_specs: dict[int, ShardSpec]
    ) -> dict[int, torch.Tensor]:        
        dst_tensors = {}
        for rank, dst_spec in dst_specs.items():
            dst_tensor = torch.empty(dst_spec.nbytes(self._tensors), dtype=torch.uint8, device=self.device)
            state_dict = {
                name: dst_tensor[start:end]
                for start, end, name in dst_spec.iter_with_intv(self._tensors)
            }
            BoundShardSpec.slice_copy(self, dst_spec(state_dict), big2small=True)
            dst_tensors[rank] = dst_tensor
        return dst_tensors

    def __setitem__(
        self,
        src_specs: dict[int, ShardSpec],
        src_tensors: dict[int, torch.Tensor],
    ) -> None:
        for rank, src_spec in src_specs.items():
            src_tensor = src_tensors[rank]
            state_dict = {
                name: src_tensor[start:end]
                for start, end, name in src_spec.iter_with_intv(self._tensors)
            }
            BoundShardSpec.slice_copy(self, src_spec(state_dict), big2small=False)


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


def _shard_contained_in(inner: Shard, outer: Shard) -> bool:
    """True if *inner*'s box lies inside *outer* (same dimensionality and width metadata)."""
    if len(inner) != len(outer):
        return False
    for (li, ri, wi), (lo, ro, wo) in zip(inner, outer):
        if wi != wo:
            return False
        if not (lo <= li < ri <= ro):
            return False
    return True


def _try_merge_two_shards(a: Shard, b: Shard) -> Shard | None:
    """If *a* and *b* match on all but one axis, and on that axis the intervals touch or overlap, return the union."""
    if len(a) != len(b):
        return None
    diff_dim: int | None = None
    for d, ((la, ra, wa), (lb, rb, wb)) in enumerate(zip(a, b)):
        if wa != wb:
            return None
        if (la, ra) != (lb, rb):
            if diff_dim is not None:
                return None
            diff_dim = d
    if diff_dim is None:
        return None
    la, ra, w = a[diff_dim]
    lb, rb, _ = b[diff_dim]
    if ra < lb or rb < la:
        return None
    ml, mr = min(la, lb), max(ra, rb)
    if ml >= mr:
        return None
    out = list(a)
    out[diff_dim] = (ml, mr, w)
    return out


def _merge_shards_for_src_spec(shards: Shards) -> Shards:
    """Collapse duplicate / contained shards and merge axis-aligned neighbors that differ on one axis only."""
    keep = [True] * len(shards)
    i = 0
    while i < len(keep):
        if not keep[i]:
            i += 1
            continue
        cur_shard = shards[i]
        for j in range(0, i):
            if keep[j]:
                shard = shards[j]
                if _shard_contained_in(shard, cur_shard):
                    keep[j] = False
                elif _shard_contained_in(cur_shard, shard):
                    keep[i] = False
                    break
                elif c:=_try_merge_two_shards(shard, cur_shard):
                    shards.append(c)
                    keep.append(True)
                    keep[i] = False
                    keep[j] = False
                    break
        i += 1
    return [shard for i, shard in enumerate(shards) if keep[i]]


class LoadSpec:
    """
    A transfer spec describes the correspondence between 2 sets of tensors.

    The format is::

        {
            "src_name": {
                "dst_name": [
                    (src_shard, dst_shard),  # each is a :class:`Shard`
                    ...
                ],
                ...
            },
            ...
        }

    Each ``(src_shard, dst_shard)`` pair maps one axis-aligned box in the source
    tensor to one box in the destination; ``shard_numel`` must match on both
    sides. Multiple pairs per ``(src_name, dst_name)`` are allowed (e.g. split
    QKV blocks).
    """

    def __init__(self, entries: dict[str, dict[str, list[ShardMapping]]]) -> None:
        self.entries = entries
        for sname, dname, s_shard, d_shard in self:
            _sanity_check(sname, [s_shard])
            _sanity_check(dname, [d_shard])
            assert shard_numel(s_shard) == shard_numel(d_shard), (
                f"numel mismatch for {sname}->{dname}: "
                f"{shard_numel(s_shard)} vs {shard_numel(d_shard)}"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def src_spec(self) -> ShardSpec:
        def _unique_shards_for_src(entry: dict[str, list[ShardMapping]]) -> Shards:
            seen: set[tuple[tuple[int, int, int], ...]] = set()
            out: Shards = []
            for mappings in entry.values():
                for s_shard, _ in mappings:
                    key = tuple(s_shard)
                    if key not in seen:
                        seen.add(key)
                        out.append(list(s_shard))
            return _merge_shards_for_src_spec(out)

        return ShardSpec({src_name: _unique_shards_for_src(entry) for src_name, entry in self.entries.items()})

    def __iter__(self) -> Iterator[tuple[str, str, Shard, Shard]]:
        for sname, entry in self.entries.items():
            for dname, mappings in entry.items():
                for s_shard, d_shard in mappings:
                    yield sname, dname, s_shard, d_shard
        
    def from_jsonable(self, jsonable: dict[str, dict[str, list[list[list[int]]]]]) -> "LoadSpec":
        return LoadSpec({
            sname: {
                dname: [
                    (
                        [tuple(t) for t in s_shard],
                        [tuple(t) for t in d_shard]
                    ) for s_shard, d_shard in mappings
                ] for dname, mappings in entry.items()
            } for sname, entry in jsonable.items()
        })
        
    def load_from_full(
        self, 
        src_iter: Iterator[tuple[str, torch.Tensor]],
        dst_tensors: dict[str, torch.Tensor],
    ) -> None:
        for sname, src_tensor in src_iter:
            if sname not in self.entries:
                continue
            for dname, mappings in self.entries[sname].items():
                assert dname in dst_tensors, f"Missing tensor {dname} for load"
                dst_tensor = dst_tensors[dname]
                for s_shard, d_shard in mappings:
                    src_slices = tuple(slice(l, r) for l, r, _ in s_shard)
                    dst_slices = tuple(slice(l, r) for l, r, _ in d_shard)
                    dst_tensor[dst_slices].copy_(src_tensor[src_slices])
                    
    def copy_fromto_sharded(
        self,
        shard_spec: ShardSpec,
        src_tensors: dict[str, torch.Tensor], # Flattened and concatenated source tensors
        dst_tensors: dict[str, torch.Tensor], # Destination tensors
        *,
        src_to_dst: bool
    ) -> None:
        """
        Load weights from sharded, flattened and concatenated source tensors to destination tensors.
        """
        for sname, dname, s_shard, d_shard in self:
            assert dname in dst_tensors, f"Missing tensor {dname} for load"
            dst_tensor = dst_tensors[dname]
            for shard, src_tensor in shards_iterator(shard_spec[sname], src_tensors[sname]):
                if _shard_contained_in(s_shard, shard):
                    src_slices = tuple(slice(l - l0, r - l0) for (l, r, _), (l0, _, _) in zip(s_shard, shard))
                    dst_slices = tuple(slice(l, r) for l, r, _ in d_shard)
                    if src_to_dst:
                        dst_tensor[dst_slices].copy_(src_tensor[src_slices])
                    else:
                        src_tensor[src_slices].copy_(dst_tensor[dst_slices])
                    break