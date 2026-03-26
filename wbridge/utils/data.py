import json

import torch


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
    A unified representation of weight metadata for receivers.
    The format is:
    {
        "name": {
            "metadata": {
                "shard": [(l, r, w), ...] | [[(l, r, w), ...], [(l, r, w), ...], ...],
                "dtype": torch.dtype,
            },
            "data": torch.Tensor | None,  # optional, receivers only need metadata
        },
        ...
    }

    where [l, r) is the range of the local shard index on the dimension, w is the total width.
    Receivers only need metadata (shard + dtype); data is optional for senders.
    """

    def __init__(self, state_dict: dict[str, dict[str, ...]]):
        self.state_dict = state_dict
        self.sanity_check(self.state_dict)

    def __getitem__(self, key: str) -> dict[str, ...]:
        return self.state_dict[key]

    def to_metadata_dict(self) -> dict[str, dict]:
        """JSON-serializable metadata only (shard + dtype string)."""
        return {
            k: {"shard": v["metadata"]["shard"], "dtype": str(v["metadata"]["dtype"])}
            for k, v in self.state_dict.items()
        }

    @classmethod
    def from_metadata_dict(cls, metadata_dict: dict[str, dict]) -> "WeightData":
        """Reconstruct a metadata-only WeightData from the JSON-serializable
        form produced by ``to_metadata_dict``."""
        state_dict = {}
        for name, meta in metadata_dict.items():
            dtype_str = meta["dtype"]
            dtype = getattr(torch, dtype_str.removeprefix("torch."))
            state_dict[name] = {
                "metadata": {"shard": meta["shard"], "dtype": dtype},
            }
        return cls(state_dict)

    def __str__(self) -> str:
        return json.dumps(self.to_metadata_dict(), indent=4)

    def sanity_check(self, state_dict: dict[str, dict[str, ...]]) -> None:
        for name, value in state_dict.items():
            meta = value["metadata"]
            dtype = meta["dtype"]
            shards = _normalize_shards(meta["shard"])
            assert len(shards) > 0, f"Empty shard list for {name}"
            total_numel = 0
            for shard in shards:
                assert len(shard) > 0, f"Empty shard in list for {name}"
                numel = 1
                for l, r, w in shard:
                    assert 0 <= l < r <= w, f"Invalid shard: {l, r, w} for {name}"
                    numel *= r - l
                total_numel += numel
            if (t := value.get("data")) is not None:
                nbytes = total_numel * dtype.itemsize
                assert t.dtype == torch.uint8, f"Invalid dtype: {t.dtype} for {name}"
                assert t.nbytes == nbytes, f"Invalid nbytes: {t.nbytes} for {name}, expected {nbytes}"
                assert len(t.shape) == 1, f"Invalid shape: {t.shape} for {name}"

    @staticmethod
    def compute_overlap(sender: "WeightData", receiver: "WeightData") -> "WeightData":
        """Return a new WeightData whose entries describe the shard regions
        where *sender* and *receiver* overlap (metadata only, no tensor data).

        Both sides may use the single-shard or multi-shard format; every
        sender shard is paired against every receiver shard and all non-empty
        overlaps are collected.
        """
        result: dict[str, dict] = {}

        for name, s_entry in sender.state_dict.items():
            if name not in receiver.state_dict:
                continue

            r_entry = receiver[name]
            dtype = s_entry["metadata"]["dtype"]
            assert dtype == r_entry["metadata"]["dtype"], (
                f"Dtype mismatch for {name}: {dtype} vs {r_entry['metadata']['dtype']}"
            )

            s_shards = _normalize_shards(s_entry["metadata"]["shard"])
            r_shards = _normalize_shards(r_entry["metadata"]["shard"])

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

            result[name] = {
                "metadata": {"shard": out_shard, "dtype": dtype},
            }

        return WeightData(result)


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