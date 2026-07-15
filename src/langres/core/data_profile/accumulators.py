"""Streaming accumulators for profiling a corpus/embedding matrix in batches.

Two minimal, mergeable accumulators so a profiler can stream a large corpus (or
an embedding matrix that does not fit comfortably in memory) in batches and still
report exact fixed-bin histograms and running mean/std -- then combine partial
accumulators computed over shards.

Deliberately minimal (the plan's middle-path decision): running **sums**, not
Welford's online variance and not reservoir sampling. For the profile report's
purpose -- headline mean/std and a fixed-edge histogram -- summing ``n``,
``sum(x)``, ``sum(x^2)`` is exact enough and trivially mergeable, and a
fixed-edge histogram with under/overflow bins captures the shape without
retaining samples. Both drop non-finite values before accumulating (a ``NaN``
norm or a ``+inf`` field length is noise, not a data point).

numpy only (a langres core dependency); no heavy/optional imports.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def _finite(batch: Iterable[float]) -> NDArray[np.float64]:
    """Flatten ``batch`` to a 1-D float array with every non-finite value dropped.

    Accepts an ``ndarray`` (any shape/dtype) or any other iterable of numbers
    (list, tuple, generator); the latter is materialised so a one-shot generator
    is handled correctly.
    """
    source = batch if isinstance(batch, np.ndarray) else list(batch)
    arr = np.asarray(source, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


class OnlineHistogram:
    """A fixed-edge histogram with under/overflow bins, updatable in batches.

    ``n_bins`` equal-width in-range bins span ``[lo, hi]`` (the last bin is
    closed on the right, so a value exactly ``hi`` lands in it, matching
    ``numpy.histogram``). Two extra bins catch everything outside the range: an
    underflow bin (``x < lo``) and an overflow bin (``x > hi``). This makes the
    histogram robust to a bad ``[lo, hi]`` guess -- out-of-range mass is counted,
    never silently dropped.

    Attributes:
        lo: Lower edge of the in-range span.
        hi: Upper edge of the in-range span.
        n_bins: Number of equal-width in-range bins.
        edges: The ``n_bins + 1`` in-range bin edges (``numpy`` array).
    """

    def __init__(self, lo: float, hi: float, n_bins: int) -> None:
        """Create an empty histogram over ``[lo, hi]`` with ``n_bins`` in-range bins.

        Raises:
            ValueError: If ``n_bins < 1`` or ``hi <= lo`` (a degenerate range
                has no well-defined bins).
        """
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError(f"require finite lo < hi, got lo={lo!r} hi={hi!r}")
        self.lo = float(lo)
        self.hi = float(hi)
        self.n_bins = int(n_bins)
        # Explicit float64 dtype: numpy's ``linspace`` return type is
        # ``floating[Any]``; pinning it here keeps ``self.edges`` typed as
        # ``NDArray[np.float64]`` for both mypy and Pyright.
        self.edges: NDArray[np.float64] = np.linspace(
            self.lo, self.hi, self.n_bins + 1, dtype=np.float64
        )
        # Layout: [underflow, bin_0, ..., bin_{n-1}, overflow] -> length n_bins + 2.
        self._counts: NDArray[np.int64] = np.zeros(self.n_bins + 2, dtype=np.int64)

    def update(self, batch: Iterable[float]) -> OnlineHistogram:
        """Add a batch of values (non-finite dropped). Returns ``self`` (fluent)."""
        arr = _finite(batch)
        if arr.size:
            under = int(np.count_nonzero(arr < self.lo))
            over = int(np.count_nonzero(arr > self.hi))
            in_range = arr[(arr >= self.lo) & (arr <= self.hi)]
            hist, _ = np.histogram(in_range, bins=self.edges)
            self._counts[0] += under
            self._counts[1:-1] += hist.astype(np.int64)
            self._counts[-1] += over
        return self

    def merge(self, other: OnlineHistogram) -> OnlineHistogram:
        """Return a NEW histogram summing ``self`` and ``other`` (neither mutated).

        Raises:
            ValueError: If the two histograms do not share ``(lo, hi, n_bins)``.
        """
        if (other.lo, other.hi, other.n_bins) != (self.lo, self.hi, self.n_bins):
            raise ValueError("cannot merge histograms with different binning")
        merged = OnlineHistogram(self.lo, self.hi, self.n_bins)
        merged._counts = self._counts + other._counts
        return merged

    @property
    def counts(self) -> NDArray[np.int64]:
        """Full count vector ``[underflow, bin_0..bin_{n-1}, overflow]`` (length ``n_bins + 2``)."""
        return self._counts.copy()

    @property
    def bin_counts(self) -> NDArray[np.int64]:
        """Just the ``n_bins`` in-range counts (underflow/overflow excluded)."""
        return self._counts[1:-1].copy()

    @property
    def underflow(self) -> int:
        """Count of values below ``lo``."""
        return int(self._counts[0])

    @property
    def overflow(self) -> int:
        """Count of values above ``hi``."""
        return int(self._counts[-1])


class RunningStats:
    """Running mean/std/variance over batches via summed moments (not Welford).

    Accumulates ``n``, ``sum(x)`` and ``sum(x^2)``; mean and (population)
    variance derive from those. Exact and trivially mergeable -- the intended
    tradeoff for a profile headline (the plan's middle-path decision). Non-finite
    values are dropped before accumulating.
    """

    def __init__(self) -> None:
        """Create an empty accumulator (no samples yet)."""
        self._n = 0
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, batch: Iterable[float]) -> RunningStats:
        """Add a batch of values (non-finite dropped). Returns ``self`` (fluent)."""
        arr = _finite(batch)
        if arr.size:
            self._n += int(arr.size)
            self._sum += float(arr.sum())
            self._sum_sq += float(np.square(arr).sum())
        return self

    def merge(self, other: RunningStats) -> RunningStats:
        """Return a NEW accumulator summing ``self`` and ``other`` (neither mutated)."""
        merged = RunningStats()
        merged._n = self._n + other._n
        merged._sum = self._sum + other._sum
        merged._sum_sq = self._sum_sq + other._sum_sq
        return merged

    @property
    def count(self) -> int:
        """Number of finite values accumulated."""
        return self._n

    @property
    def mean(self) -> float:
        """Arithmetic mean, or ``nan`` when no values have been seen."""
        return self._sum / self._n if self._n else float("nan")

    @property
    def variance(self) -> float:
        """Population variance (ddof=0), or ``nan`` when empty.

        Clamped at ``0.0`` so floating-point cancellation in
        ``E[x^2] - E[x]^2`` can never surface a tiny negative variance.
        """
        if self._n == 0:
            return float("nan")
        mean = self._sum / self._n
        return max(self._sum_sq / self._n - mean * mean, 0.0)

    @property
    def std(self) -> float:
        """Population standard deviation (ddof=0), or ``nan`` when empty."""
        var = self.variance
        return math.sqrt(var) if math.isfinite(var) else float("nan")
