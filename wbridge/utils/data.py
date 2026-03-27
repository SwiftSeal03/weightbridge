import json

import torch


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
    shard: list[tuple[int, int, int]] | list[list[tuple[int, int, int]]],
) -> list[list[tuple[int, int, int]]]:
    """Normalize a shard spec to the multi-shard form ``[[dim, ...], ...]``.

    Accepts both single-shard ``[(l, r, w), ...]`` and multi-shard
    ``[[(l, r, w), ...], ...]`` inputs.
    """
    if not shard:
        return []
    if isinstance(shard[0][0], (int, float)):
        return [shard]
    return shard


class WeightData:
    """
    Shard metadata for weight transfer (no tensor storage).

    Format::

        {
            "name": {
                "shard": [(l, r, w), ...] | [[...], ...],
                "dtype": str,  # e.g. ``\"float32\"``
            },
            ...
        }

    Tensor values are passed separately as ``dict[str, torch.Tensor]`` to
    :meth:`pack_for` when sending.
    """

    def __init__(self, meta_dict: dict[str, dict[str, ...]]):
        self.meta_dict = meta_dict
        self.sanity_check(self.meta_dict)

    def __getitem__(self, key: str) -> dict[str, ...]:
        return self.meta_dict[key]

    def __str__(self) -> str:
        return json.dumps(self.meta_dict, default=str, indent=4)

    def total_nbytes(self) -> int:
        """Total byte size of the data described by all shard entries."""
        total = 0
        for entry in self.meta_dict.values():
            dtype = dtype_str_to_torch(entry["dtype"])
            for shard in _normalize_shards(entry["shard"]):
                numel = 1
                for l, r, w in shard:
                    numel *= r - l
                total += numel * dtype.itemsize
        return total

    def sanity_check(self, meta_dict: dict[str, dict[str, ...]]) -> None:
        for name, value in meta_dict.items():
            assert isinstance(value["dtype"], str), (
                f"{name}: dtype must be str, got {type(value['dtype'])}"
            )
            dtype_str_to_torch(value["dtype"])  # validate
            shards = _normalize_shards(value["shard"])
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

        Both sides may use the single-shard or multi-shard format; every
        sender shard is paired against every receiver shard and all non-empty
        overlaps are collected.
        """
        result: dict[str, dict] = {}

        for name, s_entry in sender.meta_dict.items():
            if name not in receiver.meta_dict:
                continue

            r_entry = receiver[name]
            dtype = s_entry["dtype"]
            assert dtype == r_entry["dtype"], f"Dtype mismatch for {name}: {dtype} vs {r_entry['dtype']}"

            s_shards = _normalize_shards(s_entry["shard"])
            r_shards = _normalize_shards(r_entry["shard"])

            overlap_shards: list[list[tuple[int, int, int]]] = []

            for s_shard in s_shards:
                for r_shard in r_shards:
                    alignment = _check_shard_compatibility(s_shard, r_shard)
                    if alignment is None:
                        raise ValueError(
                            f"Shard compatibility check failed for {name}"
                        )
                    aligned_s, aligned_r = alignment

                    overlap_dims: list[tuple[int, int, int]] = []
                    has_overlap = True

                    for (ls, rs, ws), (lr, rr, _) in zip(aligned_s, aligned_r):
                        lo = max(ls, lr)
                        hi = min(rs, rr)
                        if lo >= hi:
                            has_overlap = False
                            break
                        overlap_dims.append((lo, hi, ws))

                    if not has_overlap:
                        continue

                    overlap_shards.append(overlap_dims)

            if not overlap_shards:
                continue

            out_shard = (
                overlap_shards[0] if len(overlap_shards) == 1 else overlap_shards
            )

            result[name] = {"shard": out_shard, "dtype": dtype}

        return WeightData(result)

    @staticmethod
    def pack_for(
        sender_metadata: "WeightData",
        tensors: dict[str, torch.Tensor],
        overlap: "WeightData",
    ) -> torch.Tensor:
        """Pack local *tensors* (matching *sender_metadata* shards) into overlap order.

        Iterates over overlap entries in meta_dict order.  For each one,
        locates the containing sender sub-shard (handling dimensional
        alignment via ``_check_shard_compatibility``), slices the overlap
        region, and appends bytes.  Returns a 1-D ``uint8`` tensor.
        """
        chunks: list[torch.Tensor] = []
        for name, o_entry in overlap.meta_dict.items():
            assert name in tensors, f"Missing tensor {name} for overlap entry"
            s_entry = sender_metadata.meta_dict[name]
            tensor = tensors[name]
            torch_dtype = dtype_str_to_torch(s_entry["dtype"])
            assert tensor.dtype == torch_dtype, f"Tensor dtype mismatch for {name}: {tensor.dtype} vs {torch_dtype}"
            s_data = tensor.contiguous().reshape(-1).view(torch.uint8)
            s_shards = _normalize_shards(s_entry["shard"])
            o_shards = _normalize_shards(o_entry["shard"])

            byte_end = 0
            s_byte_ranges: list[tuple[int, int]] = []
            for s_shard in s_shards:
                numel = 1
                for l, r, w in s_shard:
                    numel *= r - l
                start = byte_end
                byte_end += numel * torch_dtype.itemsize
                s_byte_ranges.append((start, byte_end))

            for o_shard in o_shards:
                found = False
                for s_idx, s_shard in enumerate(s_shards):
                    alignment = _check_shard_compatibility(s_shard, o_shard)
                    if alignment is None:
                        continue
                    aligned_s, aligned_o = alignment
                    if not all(
                        ls <= lo and ro <= rs
                        for (ls, rs, _), (lo, ro, _) in zip(aligned_s, aligned_o)
                    ):
                        continue

                    beg, end = s_byte_ranges[s_idx]
                    aligned_shape = [rs - ls for ls, rs, _ in aligned_s]
                    typed = s_data[beg:end].view(torch_dtype).reshape(aligned_shape)
                    slices = tuple(
                        slice(lo - ls, ro - ls)
                        for (ls, rs, _), (lo, ro, _) in zip(aligned_s, aligned_o)
                    )
                    chunks.append(
                        typed[slices].contiguous().reshape(-1).view(torch.uint8)
                    )
                    found = True
                    break

                if not found:
                    raise ValueError(
                        f"Cannot extract overlap for '{name}': "
                        f"no compatible sender shard"
                    )

        if not chunks:
            return torch.empty(0, dtype=torch.uint8)
        return torch.cat(chunks)

    def tensors_from_flat(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Rebuild tensors from flat ``uint8`` in the same chunk order as ``pack_for`` produces.

        *overlap* metadata must match the packed layout (same shard iteration order).
        """
        offset = 0
        out: dict[str, torch.Tensor] = {}
        for name, o_entry in self.meta_dict.items():
            torch_dtype = dtype_str_to_torch(o_entry["dtype"])
            o_shards = _normalize_shards(o_entry["shard"])
            parts: list[torch.Tensor] = []
            for o_shard in o_shards:
                numel = 1
                shape_dims: list[int] = []
                for l, r, w in o_shard:
                    dim = r - l
                    numel *= dim
                    shape_dims.append(dim)
                nbytes = numel * torch_dtype.itemsize
                end = offset + nbytes
                raw = flat[offset:end]
                chunk = raw.view(torch_dtype).reshape(shape_dims)
                parts.append(chunk)
                offset = end
            if len(parts) == 1:
                out[name] = parts[0]
            else:
                out[name] = torch.cat(parts, dim=0)
        assert offset == flat.numel(), (
            f"flat size mismatch: consumed {offset}, tensor has {flat.numel()}"
        )
        return out


_SENTINEL = object()


def _check_shard_compatibility(
    s_shard: list[tuple[int, int, int]],
    r_shard: list[tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    """Check whether two shard specs can be aligned to a common dimensionality.

    A dimension ``(l*a, r*a, w*a)`` is equivalent to
    ``[(l, r, w), (0, a, a)]`` — trailing full dimensions can be split off
    or merged.  More generally, a contiguous range ``[l, r)`` within a
    single "row" of an outer dimension can also be split.

    Returns ``(aligned_s, aligned_r)`` — both lists have the same length
    with matching widths per dimension — or ``None`` if no valid alignment
    exists.
    """
    s_cur = next(iter(s_shard), _SENTINEL)
    r_cur = next(iter(r_shard), _SENTINEL)
    s_iter = iter(s_shard[1:])
    r_iter = iter(r_shard[1:])
    aligned_s: list[tuple[int, int, int]] = []
    aligned_r: list[tuple[int, int, int]] = []

    while s_cur is not _SENTINEL and r_cur is not _SENTINEL:
        sl, sr, sw = s_cur
        rl, rr, rw = r_cur

        if sw == rw:
            aligned_s.append((sl, sr, sw))
            aligned_r.append((rl, rr, rw))
            s_cur = next(s_iter, _SENTINEL)
            r_cur = next(r_iter, _SENTINEL)

        elif sw > rw:
            if sw % rw != 0:
                return None
            tail = sw // rw
            if sl % tail == 0 and sr % tail == 0:
                aligned_s.append((sl // tail, sr // tail, rw))
                aligned_r.append((rl, rr, rw))
                s_cur = (0, tail, tail)
            elif sl // tail == (sr - 1) // tail:
                row = sl // tail
                aligned_s.append((row, row + 1, rw))
                aligned_r.append((rl, rr, rw))
                s_cur = (sl - row * tail, sr - row * tail, tail)
            else:
                return None
            r_cur = next(r_iter, _SENTINEL)

        else:  # sw < rw
            if rw % sw != 0:
                return None
            tail = rw // sw
            if rl % tail == 0 and rr % tail == 0:
                aligned_s.append((sl, sr, sw))
                aligned_r.append((rl // tail, rr // tail, sw))
                r_cur = (0, tail, tail)
            elif rl // tail == (rr - 1) // tail:
                row = rl // tail
                aligned_s.append((sl, sr, sw))
                aligned_r.append((row, row + 1, sw))
                r_cur = (rl - row * tail, rr - row * tail, tail)
            else:
                return None
            s_cur = next(s_iter, _SENTINEL)

    if s_cur is not _SENTINEL or r_cur is not _SENTINEL:
        return None
    return aligned_s, aligned_r