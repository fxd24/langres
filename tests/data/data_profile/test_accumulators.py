"""Tests for the streaming accumulators (``OnlineHistogram`` / ``RunningStats``).

Every claim is checked against a direct ``numpy`` reference over the same finite
values, so the accumulators are pinned to exact behaviour (not just internal
consistency). Non-finite handling, under/overflow bins, and mergeability are
covered explicitly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from langres.data.data_profile.accumulators import OnlineHistogram, RunningStats


class TestOnlineHistogramConstruction:
    def test_rejects_non_positive_bins(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            OnlineHistogram(0.0, 1.0, 0)

    def test_rejects_degenerate_range(self) -> None:
        with pytest.raises(ValueError, match="lo < hi"):
            OnlineHistogram(1.0, 1.0, 4)
        with pytest.raises(ValueError, match="lo < hi"):
            OnlineHistogram(2.0, 1.0, 4)

    def test_rejects_non_finite_bounds(self) -> None:
        with pytest.raises(ValueError, match="lo < hi"):
            OnlineHistogram(0.0, float("inf"), 4)

    def test_edges_are_linspace(self) -> None:
        hist = OnlineHistogram(0.0, 1.0, 4)
        np.testing.assert_allclose(hist.edges, np.linspace(0.0, 1.0, 5))

    def test_starts_empty(self) -> None:
        hist = OnlineHistogram(0.0, 1.0, 4)
        assert hist.counts.tolist() == [0] * 6  # 4 bins + under + over
        assert hist.bin_counts.tolist() == [0, 0, 0, 0]
        assert hist.underflow == 0 and hist.overflow == 0


class TestOnlineHistogramUpdate:
    def test_in_range_matches_numpy(self) -> None:
        values = [0.05, 0.15, 0.25, 0.25, 0.95]
        hist = OnlineHistogram(0.0, 1.0, 10).update(values)
        expected, _ = np.histogram(values, bins=hist.edges)
        assert hist.bin_counts.tolist() == expected.tolist()
        assert hist.underflow == 0 and hist.overflow == 0

    def test_value_equal_hi_lands_in_last_bin(self) -> None:
        hist = OnlineHistogram(0.0, 1.0, 4).update([1.0])
        assert hist.bin_counts.tolist() == [0, 0, 0, 1]
        assert hist.overflow == 0

    def test_underflow_and_overflow_counted(self) -> None:
        hist = OnlineHistogram(0.0, 1.0, 4).update([-3.0, -0.1, 0.5, 1.5, 9.0])
        assert hist.underflow == 2
        assert hist.overflow == 2
        assert int(hist.bin_counts.sum()) == 1

    def test_non_finite_dropped(self) -> None:
        hist = OnlineHistogram(0.0, 1.0, 4).update([0.5, float("nan"), float("inf"), float("-inf")])
        assert int(hist.counts.sum()) == 1

    def test_accepts_ndarray_and_accumulates_across_batches(self) -> None:
        hist = OnlineHistogram(0.0, 1.0, 2)
        hist.update(np.array([0.1, 0.2])).update([0.9])
        assert hist.bin_counts.tolist() == [2, 1]

    def test_empty_batch_is_noop(self) -> None:
        hist = OnlineHistogram(0.0, 1.0, 2).update([])
        assert int(hist.counts.sum()) == 0

    def test_update_returns_self(self) -> None:
        hist = OnlineHistogram(0.0, 1.0, 2)
        assert hist.update([0.5]) is hist

    def test_counts_property_is_a_copy(self) -> None:
        hist = OnlineHistogram(0.0, 1.0, 2).update([0.5])
        snapshot = hist.counts
        snapshot[1] = 999
        assert hist.counts.tolist() != snapshot.tolist()


class TestOnlineHistogramMerge:
    def test_merge_sums_without_mutating_operands(self) -> None:
        a = OnlineHistogram(0.0, 1.0, 4).update([0.1, 0.9])
        b = OnlineHistogram(0.0, 1.0, 4).update([0.1, -1.0, 2.0])
        before_a = a.counts.tolist()
        before_b = b.counts.tolist()
        merged = a.merge(b)
        assert merged.counts.tolist() == (a.counts + b.counts).tolist()
        # Operands untouched.
        assert a.counts.tolist() == before_a
        assert b.counts.tolist() == before_b

    def test_merge_equivalent_to_single_pass(self) -> None:
        batch1 = [0.1, 0.2, 0.9]
        batch2 = [0.15, 0.95, 1.0]
        a = OnlineHistogram(0.0, 1.0, 5).update(batch1)
        b = OnlineHistogram(0.0, 1.0, 5).update(batch2)
        single = OnlineHistogram(0.0, 1.0, 5).update(batch1 + batch2)
        assert a.merge(b).counts.tolist() == single.counts.tolist()

    def test_merge_rejects_mismatched_binning(self) -> None:
        a = OnlineHistogram(0.0, 1.0, 4)
        with pytest.raises(ValueError, match="different binning"):
            a.merge(OnlineHistogram(0.0, 2.0, 4))
        with pytest.raises(ValueError, match="different binning"):
            a.merge(OnlineHistogram(0.0, 1.0, 5))


class TestRunningStats:
    def test_empty_is_nan(self) -> None:
        stats = RunningStats()
        assert stats.count == 0
        assert math.isnan(stats.mean)
        assert math.isnan(stats.variance)
        assert math.isnan(stats.std)

    def test_matches_numpy_population_moments(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 10.0]
        stats = RunningStats().update(values)
        assert stats.count == 5
        assert stats.mean == pytest.approx(float(np.mean(values)))
        assert stats.variance == pytest.approx(float(np.var(values)))  # ddof=0
        assert stats.std == pytest.approx(float(np.std(values)))  # ddof=0

    def test_non_finite_dropped(self) -> None:
        values = [1.0, float("nan"), 2.0, float("inf"), 3.0]
        finite = [1.0, 2.0, 3.0]
        stats = RunningStats().update(values)
        assert stats.count == 3
        assert stats.mean == pytest.approx(float(np.mean(finite)))
        assert stats.std == pytest.approx(float(np.std(finite)))

    def test_accumulates_across_batches(self) -> None:
        stats = RunningStats().update([1.0, 2.0]).update([3.0, 4.0])
        assert stats.count == 4
        assert stats.mean == pytest.approx(2.5)

    def test_update_returns_self(self) -> None:
        stats = RunningStats()
        assert stats.update([1.0]) is stats

    def test_empty_batch_is_noop(self) -> None:
        # An empty (or all-non-finite) batch leaves the accumulator untouched:
        # exercises the ``if arr.size`` false branch of update().
        stats = RunningStats().update([]).update([float("nan")])
        assert stats.count == 0
        assert math.isnan(stats.mean)

    def test_ndarray_batch(self) -> None:
        stats = RunningStats().update(np.array([2.0, 4.0, 6.0]))
        assert stats.mean == pytest.approx(4.0)

    def test_merge_equivalent_to_single_pass(self) -> None:
        batch1 = [1.0, 2.0, 3.0]
        batch2 = [10.0, 20.0]
        a = RunningStats().update(batch1)
        b = RunningStats().update(batch2)
        merged = a.merge(b)
        single = RunningStats().update(batch1 + batch2)
        assert merged.count == single.count
        assert merged.mean == pytest.approx(single.mean)
        assert merged.variance == pytest.approx(single.variance)

    def test_merge_does_not_mutate_operands(self) -> None:
        a = RunningStats().update([1.0, 2.0])
        b = RunningStats().update([3.0])
        a.merge(b)
        assert a.count == 2 and b.count == 1

    def test_variance_non_negative_on_constant_input(self) -> None:
        # E[x^2] - E[x]^2 can go slightly negative from float error; must clamp.
        stats = RunningStats().update([1e8, 1e8, 1e8])
        assert stats.variance >= 0.0
        assert stats.variance == pytest.approx(0.0)
