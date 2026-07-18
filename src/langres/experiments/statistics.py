"""Paired cluster/entity bootstrap and separate split-instability summaries."""

from __future__ import annotations

import math
import random
import statistics

from pydantic import BaseModel, ConfigDict, Field


class PairedScore(BaseModel):
    """Two architecture scores on one fixed-test-set entity."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    baseline: float | None
    candidate: float | None
    cluster_id: str | None = None


class BootstrapInterval(BaseModel):
    """Paired candidate-minus-baseline uncertainty over cluster/entity units."""

    model_config = ConfigDict(frozen=True)

    observed_difference: float
    lower: float
    upper: float
    confidence_level: float
    standard_error: float
    n_entities: int
    n_clusters: int
    samples: int
    unit: str = "cluster"


class SplitInstability(BaseModel):
    """Sensitivity across split seeds, intentionally not a population CI."""

    model_config = ConfigDict(frozen=True)

    values: dict[str, float]
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    range: float


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


def paired_entity_bootstrap(
    observations: tuple[PairedScore, ...],
    *,
    samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap paired entity scores by cluster, never by dependent pair rows."""
    if samples < 1:
        raise ValueError("samples must be positive")
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
        cluster_id = observation.cluster_id or observation.entity_id
        by_cluster.setdefault(cluster_id, []).append(difference)
        all_differences.append(difference)

    cluster_ids = tuple(by_cluster)
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
    return BootstrapInterval(
        observed_difference=statistics.fmean(all_differences),
        lower=_percentile(bootstrap_differences, alpha),
        upper=_percentile(bootstrap_differences, 1.0 - alpha),
        confidence_level=confidence_level,
        standard_error=standard_error,
        n_entities=len(observations),
        n_clusters=len(cluster_ids),
        samples=samples,
    )


def split_instability(values: dict[str, float]) -> SplitInstability:
    """Describe metric spread across split seeds without calling it a CI."""
    if not values:
        raise ValueError("split instability requires at least one split value")
    observed = list(values.values())
    minimum = min(observed)
    maximum = max(observed)
    return SplitInstability(
        values=dict(values),
        mean=statistics.fmean(observed),
        standard_deviation=statistics.stdev(observed) if len(observed) > 1 else 0.0,
        minimum=minimum,
        maximum=maximum,
        range=maximum - minimum,
    )
