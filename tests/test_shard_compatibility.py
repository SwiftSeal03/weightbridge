"""Tests for _check_shard_compatibility and WeightData.compute_overlap."""

import torch

from wbridge.utils.data import WeightData, _check_shard_compatibility, _normalize_shards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weight_data(shard, dtype=torch.float32, data_tensor=None):
    """Build a single-entry WeightData for testing."""
    numel = 1
    for l, r, w in shard:
        numel *= r - l
    if data_tensor is None:
        data_tensor = torch.arange(numel, dtype=dtype).contiguous().view(torch.uint8).flatten()
    return WeightData({
        "weight": {
            "metadata": {"shard": shard, "dtype": dtype},
            "data": data_tensor,
        }
    })


def _make_receiver(shard, dtype=torch.float32):
    """Build a metadata-only WeightData (no data field)."""
    return WeightData({
        "weight": {
            "metadata": {"shard": shard, "dtype": dtype},
        }
    })


def _make_multi_sender(shards, dtype=torch.float32, data_tensor=None):
    """Build a multi-shard WeightData with data.

    *shards* is ``[shard0, shard1, ...]`` where each shard is
    ``[(l, r, w), ...]``.  Data is concatenated in shard order.
    """
    if data_tensor is None:
        parts = []
        for shard in shards:
            numel = 1
            for l, r, w in shard:
                numel *= r - l
            parts.append(
                torch.arange(numel, dtype=dtype)
                .contiguous()
                .view(torch.uint8)
                .flatten()
            )
        data_tensor = torch.cat(parts)
    return WeightData({
        "weight": {
            "metadata": {"shard": shards, "dtype": dtype},
            "data": data_tensor,
        }
    })


def _make_multi_receiver(shards, dtype=torch.float32):
    """Build a multi-shard metadata-only WeightData."""
    return WeightData({
        "weight": {
            "metadata": {"shard": shards, "dtype": dtype},
        }
    })


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
        # [1,11) in a 3×4 grid: (0,1)..(2,2) — not a rectangle
        assert _check_shard_compatibility(s, r) is None

    def test_fail_inner_range_not_rectangular(self):
        """Outer split succeeds but the inner remainder can't align.

        8×3 vs 4×6 with receiver partial range [1,5) on dim1.
        After splitting 8→4×2, the remaining sender dim has width 2 vs
        receiver inner width 6. tail=3, but receiver [1,5) doesn't
        align to 3-element boundaries or fit in a single row.
        """
        s = [(2, 6, 8), (0, 3, 3)]
        r = [(0, 4, 4), (1, 5, 6)]
        assert _check_shard_compatibility(s, r) is None

    def test_fail_sender_nonrect_in_3d(self):
        """1D range [10,20) can't form a rectangle in a 2×3×4 grid.

        sw=24>rw=2, tail=12.  10%12≠0 and 10//12=0 ≠ 19//12=1,
        so neither full-inner nor single-row applies.
        """
        s = [(10, 20, 24)]
        r = [(0, 2, 2), (0, 3, 3), (0, 4, 4)]
        assert _check_shard_compatibility(s, r) is None

    def test_fail_width_not_divisible_deep(self):
        """Widths match on the first dim but the remainder can't split.

        6×5 vs 2×3×5 — first split gives 6→2×3, but then inner width 5
        vs receiver width 3: 5%3≠0 and 3%5≠0.
        Wait, let me use a case where the first dims split fine but the
        second level fails.

        Actually: sender 6×5=30 vs receiver 2×15=30.
        sw=6>rw=2, tail=3. 0%3=0, 6%3=0 → split ok, s_cur=(0,3,3).
        Next: sw=3 < rw=15. 15%3=0, tail=5. rl=0,rr=15, 0%5=0, 15%5=0 → split ok.
        r_cur=(0,5,5), s_cur=next=(0,5,5). sw==rw → done.  This works!

        Use: sender 10×3=30 vs receiver 6×5=30.
        sw=10>rw=6.  10%6≠0.  Fails.
        """
        s = [(0, 10, 10), (0, 3, 3)]
        r = [(0, 6, 6), (0, 5, 5)]
        assert _check_shard_compatibility(s, r) is None

    def test_fail_partial_receiver_in_complex_split(self):
        """3×2×4 vs 6×4 — receiver partial range [2,5) on dim0.

        sw=3<rw=6, tail=2.  rl=2, rr=5.  2%2=0 but 5%2=1≠0.
        Single-row: 2//2=1, (5-1)//2=2.  1≠2.  Fails.
        """
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
        """Partial 1D range [12,24) in 48 vs 2×3×2×4.

        12//24=0, tail=24, 12%24=12≠0 → single row?
        12//24=0, (24-1)//24=0 → yes, single row, row=0.
        s_cur = (12, 24, 24).  Then 24 vs 3: sw=24>rw=3, tail=8.
        12%8=4≠0.  12//8=1, (24-1)//8=23//8=2.  1≠2 → None.

        Let me use [(0, 24, 48)] instead:
        sw=48>rw=2, tail=24. 0%24=0, 24%24=0. Full inner → (0,1,2).
        s_cur=(0,24,24), r_cur=next=(0,3,3).
        sw=24>rw=3, tail=8. 0%8=0, 24%8=0. Full inner → (0,3,3).
        s_cur=(0,8,8), r_cur=next=(0,2,2).
        sw=8>rw=2, tail=4. 0%4=0, 8%4=0. Full inner → (0,2,2).
        s_cur=(0,4,4), r_cur=next=(0,4,4). Match.
        """
        s = [(0, 24, 48)]
        r = [(0, 2, 2), (0, 3, 3), (0, 2, 2), (0, 4, 4)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [2, 3, 2, 4]
        assert aligned_s == [(0, 1, 2), (0, 3, 3), (0, 2, 2), (0, 4, 4)]

    def test_4x6_vs_2x4x3_partial_receiver(self):
        """4×6 vs 2×4×3 with partial range in receiver dim1.

        Receiver (0,2,2)×(2,4,4)×(0,3,3): only the second half of the
        4-wide middle dim.  Common refinement is 2×2×2×3, with the
        receiver's partial range landing in aligned dim1.
        """
        s = [(0, 4, 4), (0, 6, 6)]
        r = [(0, 2, 2), (2, 4, 4), (0, 3, 3)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        assert [w for _, _, w in aligned_s] == [2, 2, 2, 3]
        assert aligned_s == [(0, 2, 2), (0, 2, 2), (0, 2, 2), (0, 3, 3)]
        assert aligned_r == [(0, 2, 2), (1, 2, 2), (0, 2, 2), (0, 3, 3)]

    def test_mixed_partial_and_full(self):
        """Some dims partial, some full, across different granularity.

        Sender 4×6 with partial [1,3)×[0,6), receiver 2×12 with [0,2)×[2,8).
        4→2×2: sl=1,sr=3, tail=2. 1%2=1≠0. Single row: 1//2=0, 2//2=1. 0≠1. Fails.

        Use aligned ranges: sender [0,4)×[0,6) vs receiver [0,2)×[6,12).
        """
        s = [(0, 4, 4), (0, 6, 6)]
        r = [(0, 2, 2), (6, 12, 12)]
        result = _check_shard_compatibility(s, r)
        assert result is not None
        aligned_s, aligned_r = result
        # 4→2×2: 0%2=0,4%2=0 → full inner → (0,2,2)+(0,2,2)
        # then (0,2,2) vs (6,12,12): sw=2<rw=12, tail=6. 6%6=0,12%6=0 → (0,2,2)+(0,6,6)
        # then (0,6,6) vs s_cur=next=(0,6,6): match
        assert [w for _, _, w in aligned_s] == [2, 2, 6]
        assert aligned_s == [(0, 2, 2), (0, 2, 2), (0, 6, 6)]
        assert aligned_r == [(0, 2, 2), (1, 2, 2), (0, 6, 6)]


# ---------------------------------------------------------------------------
# compute_overlap integration tests (validates data slicing too)
# ---------------------------------------------------------------------------

class TestComputeOverlap:
    """Integration tests for WeightData.compute_overlap."""

    def _flat_indices(self, shard, total_shape):
        """Compute the set of flat (row-major) indices covered by a shard."""
        import itertools
        ranges = [range(l, r) for l, r, _ in shard]
        widths = [w for _, _, w in shard]
        strides = []
        for i in range(len(widths)):
            stride = 1
            for j in range(i + 1, len(widths)):
                stride *= widths[j]
            strides.append(stride)
        indices = set()
        for combo in itertools.product(*ranges):
            idx = sum(c * s for c, s in zip(combo, strides))
            indices.add(idx)
        return indices

    def test_same_dims_partial_overlap(self):
        """2D shards with partial overlap in both dimensions."""
        dtype = torch.float32
        s_shard = [(1, 3, 4), (0, 6, 6)]
        r_shard = [(2, 4, 4), (2, 5, 6)]

        sender = _make_weight_data(s_shard, dtype)
        receiver = _make_receiver(r_shard, dtype)
        overlap = WeightData.compute_overlap(sender, receiver)

        assert "weight" in overlap.state_dict
        o_shard = overlap["weight"]["metadata"]["shard"]
        assert o_shard == [(2, 3, 4), (2, 5, 6)]

        numel = 1
        for l, r, _ in o_shard:
            numel *= r - l
        assert numel == 1 * 3
        o_data = overlap["weight"]["data"].view(dtype)
        assert o_data.numel() == numel

    def test_no_overlap(self):
        """Disjoint shards produce an empty result."""
        dtype = torch.float32
        s_shard = [(0, 2, 4), (0, 6, 6)]
        r_shard = [(2, 4, 4), (0, 6, 6)]

        sender = _make_weight_data(s_shard, dtype)
        receiver = _make_receiver(r_shard, dtype)
        overlap = WeightData.compute_overlap(sender, receiver)
        assert len(overlap.state_dict) == 0

    def test_cross_dim_overlap_flat_sender(self):
        """Sender is 1D, receiver is 2D — overlap computed after alignment."""
        dtype = torch.float32
        s_shard = [(6, 18, 24)]
        r_shard = [(0, 4, 4), (2, 5, 6)]

        sender = _make_weight_data(s_shard, dtype)
        receiver = _make_receiver(r_shard, dtype)
        overlap = WeightData.compute_overlap(sender, receiver)

        assert "weight" in overlap.state_dict
        o_shard = overlap["weight"]["metadata"]["shard"]
        assert o_shard == [(1, 3, 4), (2, 5, 6)]

    def test_cross_dim_overlap_flat_receiver(self):
        """Sender is 2D, receiver is 1D — overlap computed after alignment."""
        dtype = torch.float32
        s_shard = [(0, 4, 4), (0, 6, 6)]
        r_shard = [(6, 12, 24)]

        sender = _make_weight_data(s_shard, dtype)
        receiver = _make_receiver(r_shard, dtype)
        overlap = WeightData.compute_overlap(sender, receiver)

        o_shard = overlap["weight"]["metadata"]["shard"]
        assert o_shard == [(1, 2, 4), (0, 6, 6)]

    def test_data_values_correct(self):
        """Verify sliced data contains the right elements."""
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype)
        s_shard = [(0, 24, 24)]
        data_bytes = full.contiguous().view(torch.uint8).flatten()
        sender = WeightData({
            "weight": {
                "metadata": {"shard": s_shard, "dtype": dtype},
                "data": data_bytes,
            }
        })
        r_shard = [(0, 4, 4), (2, 5, 6)]
        receiver = _make_receiver(r_shard, dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o_data = overlap["weight"]["data"].view(dtype)

        expected = full.reshape(4, 6)[:4, 2:5].contiguous().flatten()
        assert torch.equal(o_data, expected), f"{o_data} != {expected}"

    def test_single_row_data_values(self):
        """Verify data for a single-row split case."""
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype)
        s_shard = [(1, 5, 24)]
        data_bytes = full[1:5].contiguous().view(torch.uint8).flatten()
        sender = WeightData({
            "weight": {
                "metadata": {"shard": s_shard, "dtype": dtype},
                "data": data_bytes,
            }
        })
        r_shard = [(0, 4, 4), (0, 6, 6)]
        receiver = _make_receiver(r_shard, dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o_shard = overlap["weight"]["metadata"]["shard"]
        assert o_shard == [(0, 1, 4), (1, 5, 6)]

        o_data = overlap["weight"]["data"].view(dtype)
        expected = full[1:5]
        assert torch.equal(o_data, expected), f"{o_data} != {expected}"

    def test_4x6_vs_2x4x3_data_values(self):
        """4×6 sender vs 2×4×3 receiver with partial receiver dim1.

        Aligned to 2×2×2×3.  Receiver covers dim1=[1,2), so only the
        second half of each 2-wide chunk overlaps.
        """
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype).reshape(4, 6)
        s_shard = [(0, 4, 4), (0, 6, 6)]
        sender = _make_weight_data(s_shard, dtype,
            full.contiguous().view(torch.uint8).flatten())

        r_shard = [(0, 2, 2), (2, 4, 4), (0, 3, 3)]
        receiver = _make_receiver(r_shard, dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o_shard = overlap["weight"]["metadata"]["shard"]
        assert o_shard == [(0, 2, 2), (1, 2, 2), (0, 2, 2), (0, 3, 3)]

        o_data = overlap["weight"]["data"].view(dtype)
        # aligned shape is 2×2×2×3; overlap keeps dim0=[0:2), dim1=[1:2), dim2=[0:2), dim3=[0:3)
        # in the original 4×6 grid, the aligned 2×2×2×3 means:
        #   dim0 (width 2): rows 0-1, rows 2-3
        #   dim1 (width 2): within each row-pair, first or second row
        #   dim2 (width 2): within each row's 6 cols, first or second group-of-3
        #   dim3 (width 3): column within the group
        # dim1=[1,2) means "second row of each pair" → rows 1,3
        # so overlap is full[1::2, :] = rows 1 and 3, all 6 columns = 12 elements
        expected = full.reshape(2, 2, 2, 3)[:, 1:2, :, :].contiguous().flatten()
        assert torch.equal(o_data, expected), f"{o_data}\n!=\n{expected}"

    def test_8x3_vs_4x6_data_values(self):
        """Sender 8×3 partial [2,6)×[0,3) vs receiver 4×6 full.

        Aligned to 4×2×3.  Sender becomes [1,3)×[0,2)×[0,3).
        """
        dtype = torch.float32
        full_8x3 = torch.arange(24, dtype=dtype).reshape(8, 3)
        s_shard = [(2, 6, 8), (0, 3, 3)]
        sender = _make_weight_data(s_shard, dtype,
            full_8x3[2:6].contiguous().view(torch.uint8).flatten())

        r_shard = [(0, 4, 4), (0, 6, 6)]
        receiver = _make_receiver(r_shard, dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o_shard = overlap["weight"]["metadata"]["shard"]
        assert o_shard == [(1, 3, 4), (0, 2, 2), (0, 3, 3)]

        o_data = overlap["weight"]["data"].view(dtype)
        # sender data is rows 2-5 of the 8×3 matrix = 12 elements
        # aligned shape = (2, 2, 3)  (sender covers rows 1-2 of 4, full 2×3)
        # overlap is the full sender range, so all 12 elements
        expected = full_8x3[2:6].contiguous().flatten()
        assert torch.equal(o_data, expected), f"{o_data}\n!=\n{expected}"

    def test_3x2x4_vs_6x4_data_values(self):
        """Sender 3×2×4 partial vs receiver 6×4 partial — aligned to 3×2×4."""
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype).reshape(3, 2, 4)
        s_shard = [(0, 3, 3), (0, 2, 2), (1, 3, 4)]
        sender = _make_weight_data(s_shard, dtype,
            full[:, :, 1:3].contiguous().view(torch.uint8).flatten())

        r_shard = [(2, 4, 6), (0, 4, 4)]
        receiver = _make_receiver(r_shard, dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o_shard = overlap["weight"]["metadata"]["shard"]
        # aligned_r = [(1,2,3), (0,2,2), (0,4,4)]
        # overlap dim0: max(0,1)=1, min(3,2)=2 → (1,2,3)
        # overlap dim1: (0,2,2)
        # overlap dim2: max(1,0)=1, min(3,4)=3 → (1,3,4)
        assert o_shard == [(1, 2, 3), (0, 2, 2), (1, 3, 4)]

        o_data = overlap["weight"]["data"].view(dtype)
        # sender has full[:, :, 1:3] shape (3, 2, 2), reshaped in aligned space
        # overlap takes dim0=[1:2) of sender's [0:3) → slice [1:2]
        # dim1 full, dim2 full (sender [1:3) ∩ receiver [0:4) = [1:3))
        expected = full[1:2, :, 1:3].contiguous().flatten()
        assert torch.equal(o_data, expected), f"{o_data}\n!=\n{expected}"

    def test_mixed_alignment_with_no_overlap(self):
        """Complex alignment succeeds but the actual ranges are disjoint."""
        dtype = torch.float32
        s_shard = [(0, 2, 4), (0, 6, 6)]
        r_shard = [(2, 4, 4), (0, 6, 6)]
        # Same dims but disjoint on dim0
        sender = _make_weight_data(s_shard, dtype)
        receiver = _make_receiver(r_shard, dtype)
        overlap = WeightData.compute_overlap(sender, receiver)
        assert len(overlap.state_dict) == 0

        # Now with different granularity: sender 1D covers first half,
        # receiver 2D covers second half
        s_shard2 = [(0, 12, 24)]
        r_shard2 = [(2, 4, 4), (0, 6, 6)]
        sender2 = _make_weight_data(s_shard2, dtype)
        receiver2 = _make_receiver(r_shard2, dtype)
        overlap2 = WeightData.compute_overlap(sender2, receiver2)
        assert len(overlap2.state_dict) == 0


# ---------------------------------------------------------------------------
# Multi-shard compute_overlap tests
# ---------------------------------------------------------------------------

class TestComputeOverlapMultiShard:
    """Tests for compute_overlap with multi-shard sender / receiver."""

    def test_multi_sender_single_receiver_full(self):
        """Sender split into 2 row-halves, receiver covers full tensor.

        Each sender shard overlaps the single receiver shard, producing
        2 output shards whose concatenated data equals the full tensor.
        """
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype).reshape(4, 6)
        s_shards = [[(0, 2, 4), (0, 6, 6)], [(2, 4, 4), (0, 6, 6)]]
        s_data = torch.cat([
            full[0:2].contiguous().view(torch.uint8).flatten(),
            full[2:4].contiguous().view(torch.uint8).flatten(),
        ])
        sender = _make_multi_sender(s_shards, dtype, s_data)

        receiver = _make_receiver([(0, 4, 4), (0, 6, 6)], dtype)
        overlap = WeightData.compute_overlap(sender, receiver)

        o = overlap["weight"]
        o_shards = _normalize_shards(o["metadata"]["shard"])
        assert len(o_shards) == 2
        assert o_shards[0] == [(0, 2, 4), (0, 6, 6)]
        assert o_shards[1] == [(2, 4, 4), (0, 6, 6)]

        o_data = o["data"].view(dtype)
        assert torch.equal(o_data, full.flatten())

    def test_single_sender_multi_receiver(self):
        """Sender covers full tensor, receiver split into 2 column-halves.

        The single sender shard overlaps both receiver shards, producing
        2 output shards: left columns then right columns.
        """
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype).reshape(4, 6)
        sender = _make_weight_data(
            [(0, 4, 4), (0, 6, 6)], dtype,
            full.contiguous().view(torch.uint8).flatten(),
        )
        receiver = _make_multi_receiver(
            [[(0, 4, 4), (0, 3, 6)], [(0, 4, 4), (3, 6, 6)]], dtype,
        )

        overlap = WeightData.compute_overlap(sender, receiver)
        o = overlap["weight"]
        o_shards = _normalize_shards(o["metadata"]["shard"])
        assert len(o_shards) == 2
        assert o_shards[0] == [(0, 4, 4), (0, 3, 6)]
        assert o_shards[1] == [(0, 4, 4), (3, 6, 6)]

        left = full[:, :3].contiguous().view(torch.uint8).flatten()
        right = full[:, 3:].contiguous().view(torch.uint8).flatten()
        assert torch.equal(o["data"], torch.cat([left, right]))

    def test_multi_sender_multi_receiver_cross(self):
        """Sender row-halves × receiver column-halves → 4 overlap shards.

        All 4 quadrants of a 4×6 matrix are produced.
        """
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype).reshape(4, 6)
        s_shards = [[(0, 2, 4), (0, 6, 6)], [(2, 4, 4), (0, 6, 6)]]
        s_data = torch.cat([
            full[0:2].contiguous().view(torch.uint8).flatten(),
            full[2:4].contiguous().view(torch.uint8).flatten(),
        ])
        sender = _make_multi_sender(s_shards, dtype, s_data)

        r_shards = [[(0, 4, 4), (0, 3, 6)], [(0, 4, 4), (3, 6, 6)]]
        receiver = _make_multi_receiver(r_shards, dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o = overlap["weight"]
        o_shards = _normalize_shards(o["metadata"]["shard"])
        assert len(o_shards) == 4
        assert o_shards[0] == [(0, 2, 4), (0, 3, 6)]
        assert o_shards[1] == [(0, 2, 4), (3, 6, 6)]
        assert o_shards[2] == [(2, 4, 4), (0, 3, 6)]
        assert o_shards[3] == [(2, 4, 4), (3, 6, 6)]

        expected = torch.cat([
            full[0:2, 0:3].contiguous().view(torch.uint8).flatten(),
            full[0:2, 3:6].contiguous().view(torch.uint8).flatten(),
            full[2:4, 0:3].contiguous().view(torch.uint8).flatten(),
            full[2:4, 3:6].contiguous().view(torch.uint8).flatten(),
        ])
        assert torch.equal(o["data"], expected)

    def test_multi_shard_some_pairs_disjoint(self):
        """Sender covers top rows; receiver has top-left + bottom-right.

        Only the top-left pair overlaps; the bottom-right pair is disjoint
        with the sender, so the result is a single shard.
        """
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype).reshape(4, 6)
        sender = _make_weight_data(
            [(0, 2, 4), (0, 6, 6)], dtype,
            full[0:2].contiguous().view(torch.uint8).flatten(),
        )
        r_shards = [[(0, 2, 4), (0, 3, 6)], [(2, 4, 4), (3, 6, 6)]]
        receiver = _make_multi_receiver(r_shards, dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o = overlap["weight"]
        o_shards = _normalize_shards(o["metadata"]["shard"])
        assert len(o_shards) == 1
        assert o_shards[0] == [(0, 2, 4), (0, 3, 6)]

        expected = full[0:2, 0:3].contiguous().view(torch.uint8).flatten()
        assert torch.equal(o["data"], expected)

    def test_multi_shard_all_disjoint(self):
        """All sender×receiver pairs are spatially disjoint → empty result."""
        dtype = torch.float32
        sender = _make_weight_data([(0, 2, 4), (0, 3, 6)], dtype)
        receiver = _make_multi_receiver(
            [[(2, 4, 4), (3, 6, 6)], [(3, 4, 4), (0, 3, 6)]], dtype,
        )
        overlap = WeightData.compute_overlap(sender, receiver)
        assert len(overlap.state_dict) == 0

    def test_multi_sender_cross_dim_alignment(self):
        """Sender is 1D split into two halves, receiver is 2D.

        Demonstrates multi-shard combined with cross-dimension alignment.
        """
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype)

        s_shards = [[(0, 12, 24)], [(12, 24, 24)]]
        s_data = full.contiguous().view(torch.uint8).flatten()
        sender = _make_multi_sender(s_shards, dtype, s_data)

        receiver = _make_receiver([(1, 3, 4), (0, 6, 6)], dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o = overlap["weight"]
        o_shards = _normalize_shards(o["metadata"]["shard"])
        assert len(o_shards) == 2
        # shard 0 [0,12) aligns to rows [0,2); overlap with [1,3) = row 1
        assert o_shards[0] == [(1, 2, 4), (0, 6, 6)]
        # shard 1 [12,24) aligns to rows [2,4); overlap with [1,3) = row 2
        assert o_shards[1] == [(2, 3, 4), (0, 6, 6)]

        o_data = o["data"].view(dtype)
        expected = torch.cat([full[6:12], full[12:18]])
        assert torch.equal(o_data, expected)

    def test_multi_receiver_cross_dim_alignment(self):
        """Sender is 2D, receiver is 1D split into two halves.

        Receiver shards are 1D ranges that get aligned to sender's 2D grid.
        """
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype).reshape(4, 6)
        sender = _make_weight_data(
            [(0, 4, 4), (0, 6, 6)], dtype,
            full.contiguous().view(torch.uint8).flatten(),
        )
        # Receiver 1D halves: [0,12)=rows 0-1, [12,24)=rows 2-3
        r_shards = [[(0, 12, 24)], [(12, 24, 24)]]
        receiver = _make_multi_receiver(r_shards, dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o = overlap["weight"]
        o_shards = _normalize_shards(o["metadata"]["shard"])
        assert len(o_shards) == 2
        assert o_shards[0] == [(0, 2, 4), (0, 6, 6)]
        assert o_shards[1] == [(2, 4, 4), (0, 6, 6)]

        o_data = o["data"].view(dtype)
        assert torch.equal(o_data, full.flatten())

    def test_multi_sender_3_shards_partial_overlap(self):
        """Sender has 3 row-shards; receiver covers the middle two rows.

        Only shards that spatially intersect contribute to the result.
        """
        dtype = torch.float32
        full = torch.arange(36, dtype=dtype).reshape(6, 6)
        s_shards = [
            [(0, 2, 6), (0, 6, 6)],
            [(2, 4, 6), (0, 6, 6)],
            [(4, 6, 6), (0, 6, 6)],
        ]
        s_data = full.contiguous().view(torch.uint8).flatten()
        sender = _make_multi_sender(s_shards, dtype, s_data)

        receiver = _make_receiver([(1, 5, 6), (0, 6, 6)], dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o = overlap["weight"]
        o_shards = _normalize_shards(o["metadata"]["shard"])
        # s0 [0,2) ∩ [1,5) = [1,2)   → overlap
        # s1 [2,4) ∩ [1,5) = [2,4)   → overlap
        # s2 [4,6) ∩ [1,5) = [4,5)   → overlap
        assert len(o_shards) == 3
        assert o_shards[0] == [(1, 2, 6), (0, 6, 6)]
        assert o_shards[1] == [(2, 4, 6), (0, 6, 6)]
        assert o_shards[2] == [(4, 5, 6), (0, 6, 6)]

        o_data = o["data"].view(dtype)
        expected = torch.cat([
            full[1:2].contiguous().flatten(),
            full[2:4].contiguous().flatten(),
            full[4:5].contiguous().flatten(),
        ])
        assert torch.equal(o_data, expected)

    def test_output_single_shard_form_when_one_overlap(self):
        """When only one overlap is produced, result uses single-shard form."""
        dtype = torch.float32
        sender = _make_weight_data([(0, 4, 4), (0, 6, 6)], dtype)
        receiver = _make_receiver([(0, 4, 4), (0, 6, 6)], dtype)

        overlap = WeightData.compute_overlap(sender, receiver)
        o_shard = overlap["weight"]["metadata"]["shard"]
        # Single-shard form: first element is a tuple, not a list-of-tuples
        assert isinstance(o_shard[0][0], (int, float)), (
            "Expected single-shard form when only one overlap"
        )

    def test_output_multi_shard_form_when_multiple_overlaps(self):
        """When >1 overlaps are produced, result uses multi-shard form."""
        dtype = torch.float32
        full = torch.arange(24, dtype=dtype).reshape(4, 6)
        s_shards = [[(0, 2, 4), (0, 6, 6)], [(2, 4, 4), (0, 6, 6)]]
        s_data = full.contiguous().view(torch.uint8).flatten()
        sender = _make_multi_sender(s_shards, dtype, s_data)

        receiver = _make_receiver([(0, 4, 4), (0, 6, 6)], dtype)
        overlap = WeightData.compute_overlap(sender, receiver)
        o_shard = overlap["weight"]["metadata"]["shard"]
        # Multi-shard form: first element is itself a list
        assert isinstance(o_shard[0][0], (list, tuple)) and not isinstance(
            o_shard[0][0], (int, float)
        ), "Expected multi-shard form when multiple overlaps"

    def test_sanity_check_accepts_multi_shard(self):
        """WeightData construction with multi-shard format passes sanity_check."""
        dtype = torch.float32
        shards = [[(0, 2, 4), (0, 6, 6)], [(2, 4, 4), (0, 6, 6)]]
        numel = 12 + 12
        data = torch.zeros(numel * dtype.itemsize, dtype=torch.uint8)
        wd = WeightData({
            "w": {"metadata": {"shard": shards, "dtype": dtype}, "data": data}
        })
        assert "w" in wd.state_dict

    def test_sanity_check_accepts_single_shard(self):
        """Original single-shard format still passes sanity_check."""
        dtype = torch.float32
        shard = [(0, 4, 4), (0, 6, 6)]
        data = torch.zeros(24 * dtype.itemsize, dtype=torch.uint8)
        wd = WeightData({
            "w": {"metadata": {"shard": shard, "dtype": dtype}, "data": data}
        })
        assert "w" in wd.state_dict


if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0
    for cls in [TestCheckShardCompatibility, TestComputeOverlap,
                TestComputeOverlapMultiShard]:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name}: {e}")
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
