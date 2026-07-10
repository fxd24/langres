"""Tests for CorrelationClusterer (C6, W1.3): merge-resistant clustering.

The default ``Clusterer`` builds a graph from edges >= threshold and takes
connected components -- FULL transitive closure, so a chain of edges (A-B,
B-C) with no direct A-C edge merges A, B, and C into one cluster even though
A and C were never directly compared. This is the documented M3 over-merge
failure mode (-0.63 BCubed).

``CorrelationClusterer`` implements the classic *pivot algorithm* for
correlation clustering (Ailon, Charikar & Newman, "Aggregating Inconsistent
Information: Ranking and Clustering", JACM 2008): process nodes in a
deterministic, highest-confidence-first order; each pivot's cluster is itself
plus only its DIRECT neighbours >= threshold. A node with no direct edge to a
cluster's pivot is never pulled in by transitivity alone -- this is what makes
it merge-resistant relative to the base ``Clusterer``.
"""

import pytest

from langres.core.clusterer import Clusterer
from langres.core.clusterers.correlation import CorrelationClusterer
from langres.core.models import PairwiseJudgement
from langres.core.registry import get_component


def _j(left: str, right: str, score: float) -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id=left,
        right_id=right,
        score=score,
        score_type="heuristic",
        decision_step="test",
        provenance={},
    )


# ---------------------------------------------------------------------------
# The headline merge-resistance property
# ---------------------------------------------------------------------------


def test_correlation_clusterer_resists_chain_over_merge() -> None:
    """A-B and B-C edges, NO direct A-C edge -> A and C do NOT end up together.

    The base (transitive-closure) Clusterer merges all three into one cluster
    on this exact input -- this is the documented over-merge failure mode C6
    fixes.
    """
    judgements = [_j("A", "B", 0.9), _j("B", "C", 0.9)]

    base_clusters = Clusterer(threshold=0.8).cluster(judgements)
    assert base_clusters == [{"A", "B", "C"}]  # transitive closure over-merges

    clusterer = CorrelationClusterer(threshold=0.8)
    clusters = clusterer.cluster(judgements)

    assert clusters == [{"A", "B"}, {"C"}]


def test_correlation_clusterer_still_merges_a_fully_connected_triangle() -> None:
    """A direct triangle (every pair connected) merges fully, same as the base."""
    judgements = [_j("A", "B", 0.9), _j("B", "C", 0.9), _j("A", "C", 0.9)]

    clusterer = CorrelationClusterer(threshold=0.8)
    clusters = clusterer.cluster(judgements)

    assert clusters == [{"A", "B", "C"}]


def test_correlation_clusterer_longer_chain_stays_broken_up() -> None:
    """A 4-node chain (A-B, B-C, C-D) fragments rather than one giant cluster."""
    judgements = [_j("A", "B", 0.9), _j("B", "C", 0.9), _j("C", "D", 0.9)]

    base_clusters = Clusterer(threshold=0.8).cluster(judgements)
    assert base_clusters == [{"A", "B", "C", "D"}]

    clusters = CorrelationClusterer(threshold=0.8).cluster(judgements)
    total_clustered = sum(len(c) for c in clusters)

    assert total_clustered == 4  # every node accounted for
    assert len(clusters) >= 2  # NOT collapsed into one giant cluster


# ---------------------------------------------------------------------------
# Threshold semantics (mirrors base Clusterer)
# ---------------------------------------------------------------------------


def test_correlation_clusterer_threshold_is_inclusive() -> None:
    """score == threshold counts as a match (mirrors base Clusterer's >=)."""
    judgements = [_j("A", "B", 0.5)]
    clusterer = CorrelationClusterer(threshold=0.5)

    assert clusterer.cluster(judgements) == [{"A", "B"}]


def test_correlation_clusterer_below_threshold_excluded() -> None:
    """Edges below threshold produce no cluster (nodes simply absent)."""
    judgements = [_j("A", "B", 0.4)]
    clusterer = CorrelationClusterer(threshold=0.5)

    assert clusterer.cluster(judgements) == []


def test_correlation_clusterer_rejects_invalid_threshold() -> None:
    """Threshold validation mirrors the base Clusterer."""
    with pytest.raises(ValueError, match="threshold"):
        CorrelationClusterer(threshold=1.5)


# ---------------------------------------------------------------------------
# Duplicate judgements, self-pairs, determinism
# ---------------------------------------------------------------------------


def test_correlation_clusterer_keeps_max_score_for_duplicate_pair_judgements() -> None:
    """If the same pair is judged twice, the stronger edge wins (no double counting)."""
    judgements = [_j("A", "B", 0.3), _j("A", "B", 0.9)]
    clusterer = CorrelationClusterer(threshold=0.8)

    assert clusterer.cluster(judgements) == [{"A", "B"}]


def test_correlation_clusterer_a_later_weaker_duplicate_does_not_downgrade_the_edge() -> None:
    """A weaker (but still >= threshold) duplicate seen AFTER the strong one is a no-op."""
    judgements = [_j("A", "B", 0.9), _j("A", "B", 0.6)]
    clusterer = CorrelationClusterer(threshold=0.5)

    assert clusterer.cluster(judgements) == [{"A", "B"}]


def test_correlation_clusterer_ignores_self_pairs() -> None:
    """A left_id == right_id judgement contributes no edge."""
    judgements = [_j("A", "A", 0.99)]
    clusterer = CorrelationClusterer(threshold=0.5)

    assert clusterer.cluster(judgements) == []


def test_correlation_clusterer_empty_input() -> None:
    """No judgements -> no clusters."""
    assert CorrelationClusterer(threshold=0.5).cluster([]) == []


def test_correlation_clusterer_is_deterministic_across_runs() -> None:
    """Repeated calls on the same judgements produce the identical result."""
    judgements = [
        _j("A", "B", 0.9),
        _j("B", "C", 0.85),
        _j("D", "E", 0.95),
        _j("E", "F", 0.7),
    ]
    clusterer = CorrelationClusterer(threshold=0.6)

    first = clusterer.cluster(judgements)
    second = clusterer.cluster(list(reversed(judgements)))

    assert first == second


def test_correlation_clusterer_accepts_an_iterator() -> None:
    """cluster() accepts an iterator, not just a list (matches base Clusterer)."""
    judgements = iter([_j("A", "B", 0.9)])
    clusterer = CorrelationClusterer(threshold=0.5)

    assert clusterer.cluster(judgements) == [{"A", "B"}]


# ---------------------------------------------------------------------------
# Inherits Clusterer's generic evaluate()/inspect_clusters() (no override)
# ---------------------------------------------------------------------------


def test_correlation_clusterer_is_a_clusterer_subclass() -> None:
    """CorrelationClusterer IS-A Clusterer -- drop-in for Resolver's clusterer slot."""
    clusterer = CorrelationClusterer(threshold=0.7)
    assert isinstance(clusterer, Clusterer)


def test_correlation_clusterer_evaluate_works_via_inheritance() -> None:
    """evaluate() (BCubed/pairwise) is inherited unchanged and works on our output."""
    judgements = [_j("A", "B", 0.9)]
    clusterer = CorrelationClusterer(threshold=0.8)
    predicted = clusterer.cluster(judgements)

    metrics = clusterer.evaluate(predicted, gold_clusters=[{"A", "B"}])

    assert metrics["bcubed"]["f1"] == 1.0


# ---------------------------------------------------------------------------
# Registry / config-registry serialization plumbing
# ---------------------------------------------------------------------------


def test_correlation_clusterer_registered_under_type_name() -> None:
    """CorrelationClusterer is registered under 'correlation_clusterer'."""
    assert get_component("correlation_clusterer") is CorrelationClusterer


def test_correlation_clusterer_config_shape() -> None:
    """config exposes the threshold only (inherited from Clusterer)."""
    clusterer = CorrelationClusterer(threshold=0.65)
    assert clusterer.config == {"threshold": 0.65}


def test_correlation_clusterer_from_config_round_trips() -> None:
    """from_config rebuilds a CorrelationClusterer (not a base Clusterer)."""
    rebuilt = CorrelationClusterer.from_config({"threshold": 0.42})

    assert isinstance(rebuilt, CorrelationClusterer)
    assert rebuilt.threshold == 0.42


# ---------------------------------------------------------------------------
# Edge weight for score-less (decider) and abstaining judgements
# (the judgement-contract wave: score is now float | None, and score doubles
# as the edge weight -- a binary "yes" must not collapse to a zero-weight edge)
# ---------------------------------------------------------------------------


def _decision_j(
    left: str,
    right: str,
    *,
    decision: bool | None = None,
    score: float | None = None,
    confidence: float | None = None,
) -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id=left,
        right_id=right,
        decision=decision,
        score=score,
        confidence=confidence,
        score_type="prob_llm",
        decision_step="test",
        provenance={},
    )


def test_score_less_decider_edge_uses_unit_weight_not_zero() -> None:
    """A binary "yes" with no score gets a full-strength 1.0 edge, never a silent 0.0.

    Without the fallback, ``edges[key] = judgement.score`` would write ``None``
    (or, coerced, ``0.0``) and the merge would be silently lost.
    """
    judgement = _decision_j("A", "B", decision=True)
    adjacency = CorrelationClusterer(threshold=0.5)._build_adjacency([judgement])
    assert adjacency["A"]["B"] == 1.0


def test_edge_weight_falls_back_to_confidence_when_no_score() -> None:
    judgement = _decision_j("A", "B", decision=True, confidence=0.8)
    adjacency = CorrelationClusterer(threshold=0.5)._build_adjacency([judgement])
    assert adjacency["A"]["B"] == 0.8


def test_score_is_the_edge_weight_when_present() -> None:
    """Score wins over confidence for the weight (score is the confidence-ordered value)."""
    judgement = _decision_j("A", "B", decision=True, score=0.9, confidence=0.3)
    adjacency = CorrelationClusterer(threshold=0.5)._build_adjacency([judgement])
    assert adjacency["A"]["B"] == 0.9


def test_negative_decision_and_abstain_are_excluded_from_edges() -> None:
    """A "no" (decision=False) and an abstention (neither set) form no edge."""
    judgements = [
        _decision_j("A", "B", decision=False),  # explicit no
        _decision_j("C", "D"),  # abstain: no decision, no score
    ]
    adjacency = CorrelationClusterer(threshold=0.5)._build_adjacency(judgements)
    assert adjacency == {}
