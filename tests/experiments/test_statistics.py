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


def _shifted(shift: float, count: int = 60) -> tuple[PairedScore, ...]:
    """Paired scores whose candidate-minus-baseline difference centres on ``shift``."""
    return tuple(
        PairedScore(
            entity_id=f"e{index}",
            cluster_id=f"c{index}",
            baseline=0.5,
            candidate=0.5 + shift + (0.02 if index % 2 else -0.02),
        )
        for index in range(count)
    )


def test_p_value_agrees_with_its_own_interval_at_every_level() -> None:
    """The property the estimator exists for, checked at levels away from 0.95.

    ``p_value <= a`` must mean "the ``1 - a`` percentile interval over this draw
    excludes zero" for *any* ``a`` -- that is what a multiplicity correction reads,
    testing at ``a/m`` … ``a`` rather than at the level the interval was cut at.
    Recovering the tail from an interval endpoint under a normality assumption
    satisfies this at 0.95 only, which is the one level Holm mostly does not use.
    """
    observations = _shifted(0.01)
    reference = paired_entity_bootstrap(observations, samples=4000, seed=3)
    assert reference.p_value is not None

    for level in (0.5, 0.8, 0.9, 0.95, 0.99, 0.995):
        interval = paired_entity_bootstrap(
            observations, samples=4000, seed=3, confidence_level=level
        )
        assert interval.lower is not None and interval.upper is not None
        excludes_zero = not (interval.lower <= 0.0 <= interval.upper)
        assert excludes_zero == (reference.p_value <= 1.0 - level), level


def test_p_value_is_two_sided_and_symmetric_under_sign_flip() -> None:
    positive = paired_entity_bootstrap(_shifted(0.05), samples=1000, seed=11)
    negative = paired_entity_bootstrap(_shifted(-0.05), samples=1000, seed=11)
    assert positive.p_value == negative.p_value
    assert positive.observed_difference == pytest.approx(-negative.observed_difference)


def test_a_null_effect_is_not_significant_and_a_large_one_hits_the_resolution_floor() -> None:
    """Both ends: no effect reads ~1, and an unmissable effect reads the floor 2/(B+1)."""
    null = paired_entity_bootstrap(_shifted(0.0), samples=1000, seed=7)
    assert null.p_value == pytest.approx(1.0)

    huge = paired_entity_bootstrap(_shifted(5.0), samples=1000, seed=7)
    assert huge.p_value == pytest.approx(2 / 1001)
    # Never exactly zero: B replicates cannot evidence a p below 2/(B+1).
    assert huge.p_value > 0.0


def test_the_resolution_floor_falls_as_samples_rise() -> None:
    """A censored p-value is a property of the draw size, and must say so."""
    small = paired_entity_bootstrap(_shifted(5.0), samples=1000, seed=7)
    large = paired_entity_bootstrap(_shifted(5.0), samples=8000, seed=7)
    assert large.p_value is not None and small.p_value is not None
    assert large.p_value < small.p_value
    assert large.p_value == pytest.approx(2 / 8001)


def test_the_monte_carlo_error_of_the_p_value_is_reported() -> None:
    """A p-value near a threshold is resolution-limited; the caller needs to see that."""
    result = paired_entity_bootstrap(_shifted(0.019), samples=2000, seed=13)
    assert result.p_value is not None and result.p_value_standard_error is not None
    expected = math.sqrt(result.p_value * (1 - result.p_value) / 2000)
    assert result.p_value_standard_error == pytest.approx(expected)
    # More replicates, tighter estimate -- the lever a caller actually has.
    bigger = paired_entity_bootstrap(_shifted(0.019), samples=8000, seed=13)
    assert bigger.p_value_standard_error is not None
    assert bigger.p_value_standard_error < result.p_value_standard_error


def test_an_insufficient_bootstrap_reports_no_p_value_rather_than_a_default() -> None:
    """One cluster cannot be resampled, so there is no p-value -- not p=1."""
    result = paired_entity_bootstrap(
        (PairedScore(entity_id="a", cluster_id="c1", baseline=0.2, candidate=0.9),),
        samples=200,
        seed=1,
    )
    assert result.status == "insufficient"
    assert result.p_value is None
    assert result.p_value_standard_error is None
