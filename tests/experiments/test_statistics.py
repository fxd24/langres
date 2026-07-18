from __future__ import annotations

import pytest

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
