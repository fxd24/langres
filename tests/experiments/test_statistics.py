from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from langres.experiments import PairedScore, paired_entity_bootstrap, split_instability


def test_paired_entity_bootstrap_preserves_cluster_units() -> None:
    observations = (
        PairedScore(entity_id="a", cluster_id="c1", baseline=0.2, candidate=0.3),
        PairedScore(entity_id="b", cluster_id="c1", baseline=0.4, candidate=0.5),
        PairedScore(entity_id="c", cluster_id="c2", baseline=0.7, candidate=0.8),
    )

    result = paired_entity_bootstrap(observations, samples=200, seed=5)

    assert result.observed_difference == pytest.approx(0.1)
    assert result.lower == pytest.approx(0.1)
    assert result.upper == pytest.approx(0.1)
    assert result.n_entities == 3
    assert result.n_clusters == 2
    assert result.unit == "cluster"
    assert result.status == "available"


def test_paired_entity_bootstrap_rejects_duplicate_or_missing_pair_members() -> None:
    with pytest.raises(ValueError, match="unique"):
        paired_entity_bootstrap(
            (
                PairedScore(entity_id="a", baseline=0.1, candidate=0.2),
                PairedScore(entity_id="a", baseline=0.2, candidate=0.3),
            )
        )

    with pytest.raises(ValueError, match="paired"):
        paired_entity_bootstrap((PairedScore(entity_id="a", baseline=0.1, candidate=None),))


def test_split_instability_is_reported_separately_from_bootstrap_uncertainty() -> None:
    result = split_instability({"seed-1": 0.70, "seed-2": 0.80, "seed-3": 0.75})

    assert result.mean == pytest.approx(0.75)
    assert result.standard_deviation == pytest.approx(0.05)
    assert result.range == pytest.approx(0.10)
    assert result.values == {"seed-1": 0.70, "seed-2": 0.80, "seed-3": 0.75}


def test_one_resampling_unit_is_explicitly_insufficient() -> None:
    result = paired_entity_bootstrap(
        (PairedScore(entity_id="a", cluster_id="only", baseline=0.2, candidate=0.3),)
    )

    assert result.status == "insufficient"
    assert result.lower is None
    assert result.upper is None
    assert result.standard_error is None
    assert "at least two" in (result.reason or "")


def test_explicit_cluster_ids_cannot_collide_with_entity_fallback_ids() -> None:
    result = paired_entity_bootstrap(
        (
            PairedScore(
                entity_id="entity-in-cluster", cluster_id="same", baseline=0.1, candidate=0.2
            ),
            PairedScore(entity_id="same", baseline=0.2, candidate=0.4),
        ),
        samples=100,
    )

    assert result.n_clusters == 2
    assert result.status == "available"


def test_seeded_bootstrap_is_invariant_to_observation_input_order() -> None:
    observations = (
        PairedScore(entity_id="a", cluster_id="c2", baseline=0.1, candidate=0.5),
        PairedScore(entity_id="b", cluster_id="c1", baseline=0.2, candidate=0.3),
        PairedScore(entity_id="c", cluster_id="c2", baseline=0.4, candidate=0.2),
        PairedScore(entity_id="d", cluster_id="c3", baseline=0.3, candidate=0.9),
    )

    forward = paired_entity_bootstrap(observations, samples=200, seed=17)
    reversed_order = paired_entity_bootstrap(tuple(reversed(observations)), samples=200, seed=17)

    assert forward == reversed_order


def test_non_finite_scores_are_rejected() -> None:
    with pytest.raises(ValidationError, match="finite"):
        PairedScore(entity_id="a", baseline=math.nan, candidate=0.2)
    with pytest.raises(ValidationError, match="finite"):
        split_instability({"seed": math.inf})


def test_single_split_is_explicitly_insufficient_not_zero_instability() -> None:
    result = split_instability({"seed-1": 0.7})
    assert result.status == "insufficient"
    assert result.standard_deviation is None
    assert result.range is None
