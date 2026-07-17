"""Candidate miners for cold-start gold-set bootstrapping (M1).

A miner decides *which* candidate pairs are worth paying a teacher to label.
:class:`HardNegativeMiner` does this by stratified sampling over the blocker's
``similarity_score``, deliberately spending the labeling budget across three
strata rather than only the ambiguous middle.
"""

import logging
import math
import random
from typing import Any, cast

from langres.curation._pairs import canonical_pair_key
from langres.curation.base import Miner
from langres.core.models import ERCandidate

logger = logging.getLogger(__name__)

_STRATA = ("high", "mid", "low")


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches ``numpy.percentile`` default).

    Args:
        sorted_vals: Values sorted ascending (non-empty).
        pct: Percentile in ``[0, 100]``.

    Returns:
        The interpolated value at ``pct``.
    """
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (rank - lo)


class HardNegativeMiner(Miner):
    """Stratified candidate sampler over blocker ``similarity_score``.

    Rather than mining only the ambiguous middle band, this sampler spends the
    labeling budget across three strata so the resulting gold set measures the
    teacher along all three axes (design-review W4):

    - **high** (``>= high_pct`` percentile): easy positives — measure teacher
      *recall* (does it catch the obvious matches?).
    - **mid** (``[mid_pct, high_pct)`` percentile): the hard, ambiguous core —
      weighted heaviest because this is where the teacher's judgment matters.
    - **low** (``< mid_pct`` percentile): easy negatives — measure teacher
      *specificity* (does it reject the obvious non-matches?).

    Percentile thresholds are computed over the *input* candidates' similarity
    scores, so the strata adapt to each blocker's score distribution.

    Contract:
        The Wave-4 orchestrator performs cross-source filtering (e.g. dropping
        same-source pairs for record-linkage) *before* calling :meth:`mine`.
        This miner sees an already-filtered candidate pool and only bands and
        samples it — it does no filtering of its own.

    Determinism:
        Sampling uses a seeded :class:`random.Random`, so the same input and
        ``seed`` always yield the same pairs.
    """

    def __init__(
        self,
        *,
        high_pct: float = 85.0,
        mid_pct: float = 40.0,
        high_proportion: float = 0.25,
        mid_proportion: float = 0.50,
        low_proportion: float = 0.25,
        seed: int = 0,
    ) -> None:
        """Initialize the miner.

        Args:
            high_pct: Lower percentile boundary of the *high* stratum.
            mid_pct: Lower percentile boundary of the *mid* stratum (and upper
                boundary of *low*).
            high_proportion: Target share of capped output drawn from *high*.
            mid_proportion: Target share drawn from *mid* (weighted heaviest).
            low_proportion: Target share drawn from *low*.
            seed: RNG seed for deterministic sampling.

        Raises:
            ValueError: If percentile boundaries are not ``0 <= mid_pct <
                high_pct <= 100``, or if any proportion is negative, or if the
                proportions sum to zero.
        """
        if not 0.0 <= mid_pct < high_pct <= 100.0:
            raise ValueError("require 0 <= mid_pct < high_pct <= 100")
        proportions = (high_proportion, mid_proportion, low_proportion)
        if any(p < 0.0 for p in proportions):
            raise ValueError("proportions must be non-negative")
        if sum(proportions) <= 0.0:
            raise ValueError("proportions must not all be zero")

        self.high_pct = high_pct
        self.mid_pct = mid_pct
        self.high_proportion = high_proportion
        self.mid_proportion = mid_proportion
        self.low_proportion = low_proportion
        self.seed = seed

    def mine(
        self,
        candidates: list[ERCandidate[Any]],
        *,
        max_pairs: int | None = None,
    ) -> list[ERCandidate[Any]]:
        """Deduplicate, stratify, and (optionally) cap the candidate pool.

        Args:
            candidates: Candidate pairs to mine. Every candidate must carry a
                non-``None`` ``similarity_score`` (the miner bands on it).
            max_pairs: Optional cap on returned pairs. When ``None`` (or ``>=``
                the number of unique candidates) every unique candidate is
                returned. When set, the cap is split across the three strata in
                the configured proportions (redistributing any stratum's
                shortfall to the others) and sampled deterministically.

        Returns:
            The selected candidate pairs, in the input's (deduplicated) order.

        Raises:
            ValueError: If any candidate has ``similarity_score is None``, or if
                ``max_pairs`` is negative.
        """
        if max_pairs is not None and max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        deduped = self._dedup(candidates)
        if not deduped:
            return []
        if max_pairs is None or max_pairs >= len(deduped):
            return deduped

        strata = self._stratify(deduped)
        sizes = {name: len(strata[name]) for name in _STRATA}
        alloc = self._allocate(sizes, max_pairs)

        rng = random.Random(self.seed)
        selected: set[tuple[str, str]] = set()
        for name in _STRATA:
            count = alloc[name]
            if count:
                for cand in rng.sample(strata[name], count):
                    selected.add(self._key(cand))

        return [cand for cand in deduped if self._key(cand) in selected]

    @staticmethod
    def _key(candidate: ERCandidate[Any]) -> tuple[str, str]:
        """Order-independent identity of a pair (handles (a,b) == (b,a))."""
        return canonical_pair_key(candidate.left.id, candidate.right.id)

    def _dedup(self, candidates: list[ERCandidate[Any]]) -> list[ERCandidate[Any]]:
        """Drop duplicate pairs (first occurrence wins); validate scores present."""
        seen: set[tuple[str, str]] = set()
        unique: list[ERCandidate[Any]] = []
        for cand in candidates:
            if cand.similarity_score is None:
                raise ValueError(
                    "HardNegativeMiner requires similarity_score on every candidate; "
                    f"pair {self._key(cand)} has none"
                )
            key = self._key(cand)
            if key not in seen:
                seen.add(key)
                unique.append(cand)
        return unique

    def _stratify(self, candidates: list[ERCandidate[Any]]) -> dict[str, list[ERCandidate[Any]]]:
        """Bucket candidates into high/mid/low strata by similarity percentile.

        ``_dedup`` has already guaranteed every ``similarity_score`` is non-None,
        so the :func:`cast` here is safe (it narrows ``float | None`` to ``float``
        without a runtime check).
        """
        scores = sorted(cast(float, c.similarity_score) for c in candidates)
        t_low = _percentile(scores, self.mid_pct)
        t_high = _percentile(scores, self.high_pct)

        strata: dict[str, list[ERCandidate[Any]]] = {name: [] for name in _STRATA}
        for cand in candidates:
            score = cast(float, cand.similarity_score)
            if score >= t_high:
                strata["high"].append(cand)
            elif score >= t_low:
                strata["mid"].append(cand)
            else:
                strata["low"].append(cand)
        return strata

    def _allocate(self, sizes: dict[str, int], total: int) -> dict[str, int]:
        """Split ``total`` across strata by proportion, capping at availability.

        Uses floor-then-largest-remainder allocation, then redistributes any
        leftover (from rounding or from a stratum smaller than its target) to
        strata that still have spare capacity. ``total`` is assumed ``<``
        ``sum(sizes)`` (the uncapped fast path handles the rest), so the
        leftover loop always terminates with ``sum(alloc) == total``.
        """
        weights = {
            "high": self.high_proportion,
            "mid": self.mid_proportion,
            "low": self.low_proportion,
        }
        wsum = sum(weights.values())
        ideal = {name: total * weights[name] / wsum for name in _STRATA}
        alloc = {name: min(sizes[name], math.floor(ideal[name])) for name in _STRATA}

        while sum(alloc.values()) < total and any(alloc[name] < sizes[name] for name in _STRATA):
            with_capacity = [name for name in _STRATA if alloc[name] < sizes[name]]
            # Largest fractional remainder first. On an exact frac tie, the
            # ``-_STRATA.index`` term favors ``high`` then ``mid`` then ``low``;
            # this only affects the 1-2 leftover units, since the mid stratum is
            # already weighted heaviest by its proportion (and thus its frac).
            best = max(
                with_capacity,
                key=lambda name: (ideal[name] - math.floor(ideal[name]), -_STRATA.index(name)),
            )
            alloc[best] += 1
        return alloc
