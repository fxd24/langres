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

import random

import pytest

from langres.core.clusterer import Clusterer
from langres.core.clusterers.correlation import CorrelationClusterer
from langres.core.models import PairwiseJudgement
from langres.core.registry import get_component
from langres.data.benchmark import complete_partition
from langres.metrics.metrics import evaluate_clustering


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

    # C is stranded, so it is absent -- NOT emitted as a {"C"} singleton. Same
    # output shape as the base Clusterer; see the singleton-contract tests below.
    assert clusters == [{"A", "B"}]


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


# ---------------------------------------------------------------------------
# The singleton contract: a drop-in must not change the output SHAPE
#
# The pivot loop can form a one-node {node} cluster -- a record WITH a
# qualifying edge whose neighbours an earlier pivot already claimed. The base
# Clusterer cannot produce that (an unmerged record never enters its graph) and
# the documented dedupe() contract returns multi-record clusters only. So the
# pivot clusterer drops them, and these tests pin that: the class docstring
# claimed "no singleton clusters" long before the code actually did it.
# ---------------------------------------------------------------------------


class _SingletonEmittingCorrelationClusterer(CorrelationClusterer):
    """The PRE-FIX pivot loop, kept verbatim as the "before" side of a comparison.

    Identical to :meth:`CorrelationClusterer.cluster` except that it appends every
    cluster, including the one-node ones. Without this, the metric-neutrality
    tests below would compare the new behaviour against itself and could never
    fail -- the shape of gate this repo has been bitten by before.
    """

    def cluster(self, judgements):  # type: ignore[no-untyped-def]
        adjacency = self._build_adjacency(judgements)
        remaining = set(adjacency)
        clusters: list[set[str]] = []
        for node in sorted(adjacency, key=lambda n: self._pivot_priority(n, adjacency)):
            if node not in remaining:
                continue
            cluster = {node} | (set(adjacency[node]) & remaining)
            remaining -= cluster
            clusters.append(cluster)
        return clusters


#: The reproduction that proved the old docstring wrong. A is the first pivot
#: (ties break by node id) and claims B and D; C's only neighbour B is gone, so
#: the pre-fix loop emitted {"C"}.
_STRANDING_JUDGEMENTS = [_j("A", "B", 0.9), _j("B", "C", 0.9), _j("A", "D", 0.6)]
_ALL_IDS = ["A", "B", "C", "D"]


def test_the_pre_fix_loop_really_did_emit_a_singleton() -> None:
    """Pins the bug itself: without the drop, C comes back as a {"C"} cluster.

    If this ever stops emitting the singleton, the comparison tests below have
    silently become no-ops and must be re-derived rather than trusted.
    """
    before = _SingletonEmittingCorrelationClusterer(threshold=0.5)
    assert before.cluster(_STRANDING_JUDGEMENTS) == [{"A", "B", "D"}, {"C"}]


def test_stranded_record_is_absent_not_a_singleton() -> None:
    """The shipped clusterer drops it -- same shape the base Clusterer produces."""
    clusters = CorrelationClusterer(threshold=0.5).cluster(_STRANDING_JUDGEMENTS)

    assert clusters == [{"A", "B", "D"}]
    assert all(len(cluster) > 1 for cluster in clusters)


def test_base_clusterer_never_emits_a_size_one_cluster_on_this_input() -> None:
    """The shape the pivot clusterer is being held to, asserted on the base."""
    clusters = Clusterer(threshold=0.5).cluster(_STRANDING_JUDGEMENTS)

    assert all(len(cluster) > 1 for cluster in clusters)


def _random_judgements(rng: random.Random, n_ids: int, n_edges: int) -> list[PairwiseJudgement]:
    ids = [f"r{i}" for i in range(n_ids)]
    out = []
    for _ in range(n_edges):
        left, right = rng.sample(ids, 2)
        out.append(_j(left, right, rng.random()))
    return out


def test_dropping_singletons_does_not_move_any_measured_number() -> None:
    """The completed partitions are IDENTICAL, so every metric over them is too.

    This is why the portfolio numbers quoted in ``CorrelationClusterer``'s
    docstring (``docs/research/20260727_closure_diagnostic.md``: better BCubed F1
    at 36 of 45 scored grid points, tied 9, worse 0) survive this change rather
    than needing a re-run. The harness scores
    ``complete_partition(clusters, all_ids)``, which restores every id not in a
    predicted cluster as its own singleton -- so a singleton the clusterer drops
    is put straight back before anything is measured.

    Checked over a randomized battery, not one hand-picked graph, and compared
    against the pre-fix loop above so the assertion has a way to fail.
    """
    rng = random.Random(20260727)
    before = _SingletonEmittingCorrelationClusterer(threshold=0.5)
    after = CorrelationClusterer(threshold=0.5)
    saw_a_dropped_singleton = False

    for _ in range(200):
        n_ids = rng.randint(2, 12)
        judgements = _random_judgements(rng, n_ids, rng.randint(1, 18))
        all_ids = [f"r{i}" for i in range(n_ids)]

        old = before.cluster(judgements)
        new = after.cluster(judgements)
        if len(old) != len(new):
            saw_a_dropped_singleton = True

        # Set-of-frozensets: the completed partitions differ at most in ORDER
        # (a restored singleton is appended last), and every metric here is
        # order-independent over the cluster list.
        assert {frozenset(c) for c in complete_partition(old, all_ids)} == {
            frozenset(c) for c in complete_partition(new, all_ids)
        }

    assert saw_a_dropped_singleton, "battery never hit the case under test"


def test_bcubed_and_pairwise_are_unchanged_on_the_stranding_case() -> None:
    """The metric-level statement of the same fact, on the known-stranding input."""
    gold = [{"A", "B"}, {"C", "D"}]
    before = _SingletonEmittingCorrelationClusterer(threshold=0.5).cluster(_STRANDING_JUDGEMENTS)
    after = CorrelationClusterer(threshold=0.5).cluster(_STRANDING_JUDGEMENTS)

    assert before != after  # the inputs to the metric really do differ

    assert evaluate_clustering(complete_partition(before, _ALL_IDS), gold) == evaluate_clustering(
        complete_partition(after, _ALL_IDS), gold
    )


def test_rejected_pairs_inside_clusters_is_unchanged_by_the_drop() -> None:
    """The other diagnostic column: a one-node cluster can hold no pair at all.

    ``rejected-inside`` counts pairs whose two ids share an output cluster, so a
    size-1 cluster contributes zero by construction and dropping it cannot move
    the count. Asserted rather than argued.
    """
    rng = random.Random(4242)
    before = _SingletonEmittingCorrelationClusterer(threshold=0.5)
    after = CorrelationClusterer(threshold=0.5)

    def in_cluster_pairs(clusters: list[set[str]]) -> set[frozenset[str]]:
        return {
            frozenset((left, right))
            for cluster in clusters
            for left in cluster
            for right in cluster
            if left != right
        }

    for _ in range(100):
        n_ids = rng.randint(2, 12)
        judgements = _random_judgements(rng, n_ids, rng.randint(1, 18))
        assert in_cluster_pairs(before.cluster(judgements)) == in_cluster_pairs(
            after.cluster(judgements)
        )
