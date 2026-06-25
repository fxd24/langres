"""Config-registry plumbing tests for Clusterer (Wave 2b).

The Clusterer's only state is its ``threshold``, so serialization is trivial:
``config`` emits ``{"threshold": ...}`` and ``from_config`` rebuilds it.
"""

from langres.core.clusterer import Clusterer
from langres.core.models import PairwiseJudgement
from langres.core.registry import get_component


def _judgement(left: str, right: str, score: float) -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id=left,
        right_id=right,
        score=score,
        score_type="heuristic",
        decision_step="test",
        provenance={},
    )


def test_registered_under_type_name() -> None:
    """Clusterer is registered under 'clusterer'."""
    assert get_component("clusterer") is Clusterer


def test_config_shape() -> None:
    """config exposes the threshold only."""
    clusterer = Clusterer(threshold=0.7)

    assert clusterer.config == {"threshold": 0.7}


def test_from_config_reconstructs_threshold() -> None:
    """from_config rebuilds a Clusterer with the same threshold."""
    rebuilt = Clusterer.from_config({"threshold": 0.42})

    assert rebuilt.threshold == 0.42


def test_config_roundtrip_reproduces_clusters() -> None:
    """config -> from_config preserves clustering behavior."""
    clusterer = Clusterer(threshold=0.7)
    rebuilt = Clusterer.from_config(clusterer.config)

    judgements = [
        _judgement("a", "b", 0.9),  # >= threshold -> merge
        _judgement("b", "c", 0.6),  # < threshold -> no edge
        _judgement("d", "e", 0.8),  # >= threshold -> merge
    ]

    before = sorted(sorted(c) for c in clusterer.cluster(judgements))
    after = sorted(sorted(c) for c in rebuilt.cluster(judgements))

    assert before == after
    assert after == [["a", "b"], ["d", "e"]]
