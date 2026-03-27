"""Tests for _check_shard_compatibility, WeightData.compute_overlap, and pack_for."""

import torch

from wbridge.utils.data import WeightData, _check_shard_compatibility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(shard, dtype: str = "float32"):
    """Single-entry metadata-only WeightData."""
    return WeightData({
        "weight": {"shard": shard, "dtype": dtype},
    })


def _tensor_for_shard(shard, dtype=torch.float32):  # torch dtype for tensors
    numel = 1
    for l, r, w in shard:
        numel *= r - l
    return torch.arange(numel, dtype=dtype).reshape(
        tuple(r - l for l, r, w in shard)
    )


# ---------------------------------------------------------------------------
# _check_shard_compatibility tests
# ---------------------------------------------------------------------------

class TestCheckShardCompatibility:
    """Unit tests for _check_shard_compatibility."""

    def test_same_dims(self):
        """Identical dimensionality and widths — returned as-is."""
        s = [(0, 4, 4), (0, 6, 6)]
        r = [(2, 3, 4), (1, 5, 6)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert aligned_s == [(0, 4, 4), (0, 6, 6)]
        assert aligned_r == [(2, 3, 4), (1, 5, 6)]

    def test_sender_flat_receiver_split(self):
        """Sender has 1 dim, receiver has 2 — sender gets split."""
        s = [(0, 24, 24)]
        r = [(0, 4, 4), (0, 6, 6)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [4, 6]
        assert [w for _, _, w in aligned_r] == [4, 6]
        assert aligned_s == [(0, 4, 4), (0, 6, 6)]
        assert aligned_r == [(0, 4, 4), (0, 6, 6)]

    def test_receiver_flat_sender_split(self):
        """Receiver has 1 dim, sender has 2 — receiver gets split."""
        s = [(0, 4, 4), (0, 6, 6)]
        r = [(0, 24, 24)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert aligned_s == [(0, 4, 4), (0, 6, 6)]
        assert aligned_r == [(0, 4, 4), (0, 6, 6)]

    def test_sender_flat_partial_full_inner(self):
        """Sender 1D range spans full rows — splits cleanly."""
        s = [(6, 12, 24)]
        r = [(0, 4, 4), (0, 6, 6)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert aligned_s == [(1, 2, 4), (0, 6, 6)]
        assert aligned_r == [(0, 4, 4), (0, 6, 6)]

    def test_sender_flat_single_row(self):
        """Sender 1D range fits within one row — single-row split."""
        s = [(1, 5, 24)]
        r = [(0, 4, 4), (0, 6, 6)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert aligned_s == [(0, 1, 4), (1, 5, 6)]
        assert aligned_r == [(0, 4, 4), (0, 6, 6)]

    def test_receiver_flat_single_row(self):
        """Receiver 1D range fits within one row — single-row split."""
        s = [(0, 4, 4), (0, 6, 6)]
        r = [(7, 11, 24)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert aligned_s == [(0, 4, 4), (0, 6, 6)]
        assert aligned_r == [(1, 2, 4), (1, 5, 6)]

    def test_multi_level_split(self):
        """Sender 1D, receiver 3D — requires cascaded splitting."""
        s = [(4, 8, 24)]
        r = [(0, 2, 2), (0, 3, 3), (0, 4, 4)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [2, 3, 4]
        assert aligned_s == [(0, 1, 2), (1, 2, 3), (0, 4, 4)]
        assert aligned_r == [(0, 2, 2), (0, 3, 3), (0, 4, 4)]

    def test_incompatible_spans_multiple_rows(self):
        """Range spans multiple non-complete rows — not a rectangle."""
        s = [(5, 13, 24)]
        r = [(0, 4, 4), (0, 6, 6)]
        assert _check_shard_compatibility(s, r) is None

    def test_incompatible_not_divisible(self):
        """Width not divisible — no valid split."""
        s = [(0, 10, 12)]
        r = [(0, 3, 3), (0, 4, 4)]
        assert _check_shard_compatibility(s, r) is None

    def test_both_partial(self):
        """Both sides have partial ranges, same dims."""
        s = [(1, 3, 4), (2, 5, 6)]
        r = [(2, 4, 4), (0, 4, 6)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert aligned_s == [(1, 3, 4), (2, 5, 6)]
        assert aligned_r == [(2, 4, 4), (0, 4, 6)]

    def test_sender_3d_receiver_1d(self):
        """Sender 3D, receiver 1D — receiver gets fully split."""
        s = [(0, 2, 2), (0, 3, 3), (0, 4, 4)]
        r = [(0, 24, 24)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert aligned_s == [(0, 2, 2), (0, 3, 3), (0, 4, 4)]
        assert aligned_r == [(0, 2, 2), (0, 3, 3), (0, 4, 4)]

    def test_single_dim_identical(self):
        """Single dimension on both sides."""
        s = [(2, 7, 10)]
        r = [(4, 9, 10)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert aligned_s == [(2, 7, 10)]
        assert aligned_r == [(4, 9, 10)]

    # --- alignment failures ---

    def test_fail_transposed_dims(self):
        """4×3 vs 3×4 — coprime first widths, incompatible."""
        s = [(0, 4, 4), (0, 3, 3)]
        r = [(0, 3, 3), (0, 4, 4)]
        assert _check_shard_compatibility(s, r) is None

    def test_fail_coprime_first_dims(self):
        """3×8 vs 4×6 — first widths 3 and 4 share no factor."""
        s = [(0, 3, 3), (0, 8, 8)]
        r = [(0, 4, 4), (0, 6, 6)]
        assert _check_shard_compatibility(s, r) is None

    def test_fail_receiver_spans_multiple_rows(self):
        """Receiver 1D range crosses row boundaries non-rectangularly."""
        s = [(0, 3, 3), (0, 4, 4)]
        r = [(1, 11, 12)]
        assert _check_shard_compatibility(s, r) is None

    def test_fail_inner_range_not_rectangular(self):
        """Outer split succeeds but the inner remainder can't align."""
        s = [(2, 6, 8), (0, 3, 3)]
        r = [(0, 4, 4), (1, 5, 6)]
        assert _check_shard_compatibility(s, r) is None

    def test_fail_sender_nonrect_in_3d(self):
        """1D range [10,20) can't form a rectangle in a 2×3×4 grid."""
        s = [(10, 20, 24)]
        r = [(0, 2, 2), (0, 3, 3), (0, 4, 4)]
        assert _check_shard_compatibility(s, r) is None

    def test_fail_width_not_divisible_deep(self):
        """Sender 10×3 vs receiver 6×5 — 10%6≠0."""
        s = [(0, 10, 10), (0, 3, 3)]
        r = [(0, 6, 6), (0, 5, 5)]
        assert _check_shard_compatibility(s, r) is None

    def test_fail_partial_receiver_in_complex_split(self):
        """3×2×4 vs 6×4 — receiver partial range [2,5) on dim0."""
        s = [(1, 3, 3), (0, 2, 2), (1, 3, 4)]
        r = [(2, 5, 6), (0, 4, 4)]
        assert _check_shard_compatibility(s, r) is None

    # --- complex multi-dimensional alignment ---

    def test_4x6_vs_2x4x3(self):
        """4×6 and 2×4×3 — common refinement is 2×2×2×3."""
        s = [(0, 4, 4), (0, 6, 6)]
        r = [(0, 2, 2), (0, 4, 4), (0, 3, 3)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [2, 2, 2, 3]
        assert [w for _, _, w in aligned_r] == [2, 2, 2, 3]
        assert aligned_s == [(0, 2, 2), (0, 2, 2), (0, 2, 2), (0, 3, 3)]
        assert aligned_r == [(0, 2, 2), (0, 2, 2), (0, 2, 2), (0, 3, 3)]

    def test_4x6_vs_2x12(self):
        """4×6 and 2×12 — common refinement is 2×2×6."""
        s = [(0, 4, 4), (0, 6, 6)]
        r = [(0, 2, 2), (0, 12, 12)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [2, 2, 6]
        assert aligned_s == [(0, 2, 2), (0, 2, 2), (0, 6, 6)]
        assert aligned_r == [(0, 2, 2), (0, 2, 2), (0, 6, 6)]

    def test_3x2x4_vs_6x4(self):
        """3×2×4 and 6×4 — receiver dim0 (6) splits into 3×2."""
        s = [(0, 3, 3), (0, 2, 2), (0, 4, 4)]
        r = [(0, 6, 6), (0, 4, 4)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [3, 2, 4]
        assert aligned_s == [(0, 3, 3), (0, 2, 2), (0, 4, 4)]
        assert aligned_r == [(0, 3, 3), (0, 2, 2), (0, 4, 4)]

    def test_3x2x4_vs_6x4_partial(self):
        """3×2×4 vs 6×4 with partial ranges — aligned to 3×2×4."""
        s = [(1, 3, 3), (0, 2, 2), (1, 3, 4)]
        r = [(2, 4, 6), (0, 4, 4)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [3, 2, 4]
        assert aligned_s == [(1, 3, 3), (0, 2, 2), (1, 3, 4)]
        assert aligned_r == [(1, 2, 3), (0, 2, 2), (0, 4, 4)]

    def test_8x3_vs_4x6_full_range(self):
        """8×3 vs 4×6 with full receiver — common refinement 4×2×3."""
        s = [(2, 6, 8), (0, 3, 3)]
        r = [(0, 4, 4), (0, 6, 6)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [4, 2, 3]
        assert aligned_s == [(1, 3, 4), (0, 2, 2), (0, 3, 3)]
        assert aligned_r == [(0, 4, 4), (0, 2, 2), (0, 3, 3)]

    def test_2x2x6_vs_4x3x2(self):
        """2×2×6 vs 4×3×2 — common refinement 2×2×3×2."""
        s = [(0, 2, 2), (0, 2, 2), (0, 6, 6)]
        r = [(0, 4, 4), (0, 3, 3), (0, 2, 2)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [2, 2, 3, 2]
        assert aligned_s == [(0, 2, 2), (0, 2, 2), (0, 3, 3), (0, 2, 2)]
        assert aligned_r == [(0, 2, 2), (0, 2, 2), (0, 3, 3), (0, 2, 2)]

    def test_1d_vs_4d_cascade(self):
        """1D (48) vs 2×3×2×4 — deep cascading split."""
        s = [(0, 48, 48)]
        r = [(0, 2, 2), (0, 3, 3), (0, 2, 2), (0, 4, 4)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [2, 3, 2, 4]
        assert aligned_s == [(0, 2, 2), (0, 3, 3), (0, 2, 2), (0, 4, 4)]

    def test_1d_partial_vs_4d(self):
        """Partial 1D range in 48 vs 2×3×2×4."""
        s = [(0, 24, 48)]
        r = [(0, 2, 2), (0, 3, 3), (0, 2, 2), (0, 4, 4)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [2, 3, 2, 4]
        assert aligned_s == [(0, 1, 2), (0, 3, 3), (0, 2, 2), (0, 4, 4)]

    def test_4x6_vs_2x4x3_partial_receiver(self):
        """4×6 vs 2×4×3 with partial range in receiver dim1."""
        s = [(0, 4, 4), (0, 6, 6)]
        r = [(0, 2, 2), (2, 4, 4), (0, 3, 3)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [2, 2, 2, 3]
        assert aligned_s == [(0, 2, 2), (0, 2, 2), (0, 2, 2), (0, 3, 3)]
        assert aligned_r == [(0, 2, 2), (1, 2, 2), (0, 2, 2), (0, 3, 3)]

    def test_mixed_partial_and_full(self):
        """Sender [0,4)×[0,6) vs receiver [0,2)×[6,12)."""
        s = [(0, 4, 4), (0, 6, 6)]
        r = [(0, 2, 2), (6, 12, 12)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [2, 2, 6]
        assert aligned_s == [(0, 2, 2), (0, 2, 2), (0, 6, 6)]
        assert aligned_r == [(0, 2, 2), (1, 2, 2), (0, 6, 6)]


# ---------------------------------------------------------------------------
# compute_overlap (metadata only)
# ---------------------------------------------------------------------------

class TestComputeOverlap:
    def test_same_dims_partial_overlap(self):
        dtype = torch.float32
        s_shard = [(1, 3, 4), (0, 6, 6)]
        r_shard = [(2, 4, 4), (2, 5, 6)]
        sender = _make_meta(s_shard, dtype)
        receiver = _make_meta(r_shard, dtype)
        overlap = WeightData.compute_overlap(sender, receiver)
        assert "weight" in overlap.meta_dict
        o_shard = overlap["weight"]["shard"]
        assert o_shard == [(2, 3, 4), (2, 5, 6)]

    def test_no_overlap(self):
        """Disjoint shards produce an empty result."""
        dtype = torch.float32
        s_shard = [(0, 2, 4), (0, 6, 6)]
        r_shard = [(2, 4, 4), (0, 6, 6)]
        sender = _make_meta(s_shard, dtype)
        receiver = _make_meta(r_shard, dtype)
        overlap = WeightData.compute_overlap(sender, receiver)
        assert len(overlap.meta_dict) == 0


# ---------------------------------------------------------------------------
# pack_for / tensors_from_flat
# ---------------------------------------------------------------------------

class TestPackFor:
    def test_full_shard_roundtrip(self):
        dtype = torch.float32
        shard = [(0, 4, 4), (0, 6, 6)]
        sender = _make_meta(shard, dtype)
        receiver = _make_meta(shard, dtype)
        overlap = WeightData.compute_overlap(sender, receiver)
        t = _tensor_for_shard(shard, dtype)
        packed = WeightData.pack_for(sender, {"weight": t}, overlap)
        out = overlap.tensors_from_flat(packed)
        assert torch.equal(out["weight"], t)
