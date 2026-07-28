"""Paired cluster/entity bootstrap and separate split-instability summaries."""

from __future__ import annotations

import math
import random
import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from langres.experiments.protocol import FrozenDict, freeze_mapping


class PairedScore(BaseModel):
    """Two architecture scores on one fixed-test-set entity."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    entity_id: str
    baseline: float | None
    candidate: float | None
    cluster_id: str | None = None


class BootstrapInterval(BaseModel):
    """Paired candidate-minus-baseline uncertainty over cluster/entity units."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    observed_difference: float
    lower: float | None
    upper: float | None
    confidence_level: float
    standard_error: float | None
    n_entities: int
    n_clusters: int
    samples: int
    unit: str = "cluster"
    status: Literal["available", "insufficient"]
    reason: str | None = None

    #: Two-sided achieved significance level, from the **same replicates** as the
    #: interval. ``p_value <= a`` if and only if the ``1 - a`` percentile interval
    #: over this draw excludes zero, so a caller may compare it against a level
    #: other than ``confidence_level`` -- which is what a multiplicity correction
    #: needs and what an interval alone cannot supply. ``None`` when ``status`` is
    #: ``"insufficient"``. See :func:`paired_entity_bootstrap` for the estimator.
    p_value: float | None = None

    #: Monte-Carlo standard error of :attr:`p_value` -- ``sqrt(p (1 - p) / samples)``.
    #: The p-value is itself an estimate from a finite draw, so a decision taken
    #: within a few of these of a threshold is resolution-limited rather than
    #: settled, and raising ``samples`` is what resolves it. ``None`` alongside a
    #: ``None`` p-value.
    p_value_standard_error: float | None = None


class SplitInstability(BaseModel):
    """Sensitivity across split seeds, intentionally not a population CI."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    values: dict[str, float]
    mean: float
    standard_deviation: float | None
    minimum: float
    maximum: float
    range: float | None
    status: Literal["available", "insufficient"]
    reason: str | None = None

    @field_validator("values", mode="after")
    @classmethod
    def _freeze_values(cls, value: dict[str, float]) -> FrozenDict:
        return freeze_mapping(value)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _achieved_significance_level(bootstrap_differences: list[float]) -> float:
    """Two-sided p-value obtained by *inverting* the percentile interval.

    The percentile interval at level ``1 - a`` excludes zero exactly when fewer
    than ``a / 2`` of the replicates fall on zero's far side. Reading that
    backwards gives the smallest ``a`` at which the interval would exclude zero:

        ``p = 2 * min( P(diff* <= 0), P(diff* >= 0) )``

    which is what this returns, with the usual ``(count + 1) / (samples + 1)``
    correction so a p-value is never reported as exactly zero -- with ``B``
    replicates nothing smaller than ``2 / (B + 1)`` is observable, and claiming
    ``0`` would assert a precision the draw does not have.

    **Why this and not a normal tail.** A caller correcting for multiplicity
    needs the tail probability at levels *other* than the one the interval was
    cut at (Holm tests at ``a/m`` … ``a``). Recovering those by turning an
    interval endpoint into a standard error and assuming normality is an
    extrapolation: it is pinned to agree at the published level and is
    uncalibrated everywhere else, which is precisely where the correction reads
    it. This estimator instead comes from the replicate distribution itself, so
    ``p_value <= t`` is equivalent to "the ``1 - t`` percentile interval over
    this same draw excludes zero" at *every* ``t``, by construction.

    It inherits the percentile method's limits -- no bias or acceleration
    correction (BCa would give both) -- but it is coherent with the intervals
    reported alongside it, which the normal shortcut is not.
    """
    samples = len(bootstrap_differences)
    at_or_below = sum(1 for difference in bootstrap_differences if difference <= 0.0)
    at_or_above = sum(1 for difference in bootstrap_differences if difference >= 0.0)
    tail = min(at_or_below, at_or_above)
    return min(1.0, 2.0 * (tail + 1) / (samples + 1))


def paired_entity_bootstrap(
    observations: tuple[PairedScore, ...],
    *,
    samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap paired entity scores by cluster, never by dependent pair rows."""
    if samples < 100:
        raise ValueError("paired bootstrap requires at least 100 samples")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if not observations:
        raise ValueError("paired bootstrap requires at least one observation")

    entity_ids = [observation.entity_id for observation in observations]
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("entity_id values must be unique")
    if any(
        observation.baseline is None or observation.candidate is None
        for observation in observations
    ):
        raise ValueError("paired bootstrap requires paired baseline and candidate values")

    by_cluster: dict[str, list[float]] = {}
    all_differences: list[float] = []
    for observation in observations:
        assert observation.baseline is not None
        assert observation.candidate is not None
        difference = observation.candidate - observation.baseline
        cluster_id = (
            f"cluster:{observation.cluster_id}"
            if observation.cluster_id is not None
            else f"entity:{observation.entity_id}"
        )
        by_cluster.setdefault(cluster_id, []).append(difference)
        all_differences.append(difference)

    for differences in by_cluster.values():
        differences.sort()
    all_differences.sort()
    cluster_ids = tuple(sorted(by_cluster))
    observed_difference = statistics.fmean(all_differences)
    if len(cluster_ids) < 2:
        return BootstrapInterval(
            observed_difference=observed_difference,
            lower=None,
            upper=None,
            confidence_level=confidence_level,
            standard_error=None,
            n_entities=len(observations),
            n_clusters=len(cluster_ids),
            samples=samples,
            status="insufficient",
            reason="paired bootstrap requires at least two independent cluster/entity units",
        )

    rng = random.Random(seed)
    bootstrap_differences: list[float] = []
    for _ in range(samples):
        sampled: list[float] = []
        for _ in cluster_ids:
            sampled.extend(by_cluster[rng.choice(cluster_ids)])
        bootstrap_differences.append(statistics.fmean(sampled))

    alpha = (1.0 - confidence_level) / 2.0
    standard_error = (
        statistics.stdev(bootstrap_differences) if len(bootstrap_differences) > 1 else 0.0
    )
    p_value = _achieved_significance_level(bootstrap_differences)
    return BootstrapInterval(
        observed_difference=observed_difference,
        lower=_percentile(bootstrap_differences, alpha),
        upper=_percentile(bootstrap_differences, 1.0 - alpha),
        confidence_level=confidence_level,
        standard_error=standard_error,
        n_entities=len(observations),
        n_clusters=len(cluster_ids),
        samples=samples,
        status="available",
        p_value=p_value,
        p_value_standard_error=math.sqrt(p_value * (1.0 - p_value) / len(bootstrap_differences)),
    )


def split_instability(values: dict[str, float]) -> SplitInstability:
    """Describe metric spread across split seeds without calling it a CI."""
    if not values:
        raise ValueError("split instability requires at least one split value")
    observed = list(values.values())
    minimum = min(observed)
    maximum = max(observed)
    sufficient = len(observed) > 1
    return SplitInstability(
        values=dict(values),
        mean=statistics.fmean(observed),
        standard_deviation=statistics.stdev(observed) if sufficient else None,
        minimum=minimum,
        maximum=maximum,
        range=maximum - minimum if sufficient else None,
        status="available" if sufficient else "insufficient",
        reason=None if sufficient else "split instability requires at least two split seeds",
    )
