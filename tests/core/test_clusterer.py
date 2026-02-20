"""
Comprehensive tests for langres.core.Clusterer.

The Clusterer is responsible for converting pairwise match judgements into
entity clusters using graph-based algorithms. These tests validate:
1. Threshold validation in __init__
2. Empty input handling
3. Single pair clustering
4. Transitive closure (A=B, B=C -> {A,B,C})
5. Threshold filtering (scores below threshold are ignored)
6. Multiple disjoint clusters
7. Complex graphs with mixed scores
8. Works with both Iterator and list inputs
"""

from collections.abc import Iterator

import pytest

from langres.core import PairwiseJudgement
from langres.core.clusterer import Clusterer


class TestClustererInitialization:
    """Test Clusterer initialization and parameter validation."""

    def test_can_initialize_with_default_threshold(self):
        """Clusterer can be initialized with default threshold of 0.5."""
        clusterer = Clusterer()
        assert clusterer.threshold == 0.5

    def test_can_initialize_with_custom_threshold(self):
        """Clusterer can be initialized with a custom threshold."""
        clusterer = Clusterer(threshold=0.7)
        assert clusterer.threshold == 0.7

    def test_threshold_must_be_between_0_and_1(self):
        """Threshold must be in range [0.0, 1.0]."""
        # Valid edge cases
        Clusterer(threshold=0.0)
        Clusterer(threshold=1.0)

        # Invalid cases
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            Clusterer(threshold=-0.1)

        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            Clusterer(threshold=1.1)

        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            Clusterer(threshold=2.0)


class TestClustererEmptyInputs:
    """Test Clusterer handles empty inputs gracefully."""

    def test_empty_list_returns_empty_clusters(self):
        """Empty judgements list returns empty cluster list."""
        clusterer = Clusterer(threshold=0.5)
        clusters = clusterer.cluster([])
        assert clusters == []

    def test_empty_iterator_returns_empty_clusters(self):
        """Empty judgements iterator returns empty cluster list."""
        clusterer = Clusterer(threshold=0.5)
        empty_iter = iter([])
        clusters = clusterer.cluster(empty_iter)
        assert clusters == []


class TestClustererSinglePair:
    """Test Clusterer with a single pairwise judgement."""

    def test_single_pair_above_threshold_creates_one_cluster(self):
        """Single judgement with score >= threshold creates one cluster of size 2."""
        clusterer = Clusterer(threshold=0.5)

        judgements = [
            PairwiseJudgement(
                left_id="id1",
                right_id="id2",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )
        ]

        clusters = clusterer.cluster(judgements)

        assert len(clusters) == 1
        assert clusters[0] == {"id1", "id2"}

    def test_single_pair_below_threshold_creates_no_clusters(self):
        """Single judgement with score < threshold is ignored."""
        clusterer = Clusterer(threshold=0.7)

        judgements = [
            PairwiseJudgement(
                left_id="id1",
                right_id="id2",
                score=0.5,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )
        ]

        clusters = clusterer.cluster(judgements)

        # No edges added, so no clusters (graph is empty)
        assert clusters == []

    def test_single_pair_exactly_at_threshold_creates_cluster(self):
        """Judgement with score exactly equal to threshold is included."""
        clusterer = Clusterer(threshold=0.5)

        judgements = [
            PairwiseJudgement(
                left_id="id1",
                right_id="id2",
                score=0.5,  # Exactly at threshold
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )
        ]

        clusters = clusterer.cluster(judgements)

        assert len(clusters) == 1
        assert clusters[0] == {"id1", "id2"}


class TestClustererTransitiveClosure:
    """Test that Clusterer correctly implements transitive closure."""

    def test_simple_transitive_closure_three_entities(self):
        """A=B and B=C should result in cluster {A, B, C}."""
        clusterer = Clusterer(threshold=0.7)

        judgements = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="B",
                right_id="C",
                score=0.8,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        assert len(clusters) == 1
        assert clusters[0] == {"A", "B", "C"}

    def test_transitive_closure_four_entities_chain(self):
        """A=B, B=C, C=D should result in cluster {A, B, C, D}."""
        clusterer = Clusterer(threshold=0.5)

        judgements = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="B",
                right_id="C",
                score=0.8,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="C",
                right_id="D",
                score=0.7,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        assert len(clusters) == 1
        assert clusters[0] == {"A", "B", "C", "D"}

    def test_transitive_closure_with_redundant_edges(self):
        """A=B, B=C, A=C should still result in single cluster {A, B, C}."""
        clusterer = Clusterer(threshold=0.5)

        judgements = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="B",
                right_id="C",
                score=0.8,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="A",
                right_id="C",
                score=0.85,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        assert len(clusters) == 1
        assert clusters[0] == {"A", "B", "C"}


class TestClustererThresholdFiltering:
    """Test that Clusterer correctly filters judgements by threshold."""

    def test_only_scores_above_threshold_are_used(self):
        """Only judgements with score >= threshold create edges."""
        clusterer = Clusterer(threshold=0.7)

        judgements = [
            # This should create an edge
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            # This should be ignored (below threshold)
            PairwiseJudgement(
                left_id="B",
                right_id="C",
                score=0.5,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        # Only A-B edge exists, so only one cluster
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}

    def test_threshold_filtering_prevents_transitive_closure(self):
        """Low score in chain prevents full transitive closure."""
        clusterer = Clusterer(threshold=0.7)

        judgements = [
            # A-B: above threshold
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            # B-C: below threshold (this breaks the chain)
            PairwiseJudgement(
                left_id="B",
                right_id="C",
                score=0.6,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            # C-D: above threshold
            PairwiseJudgement(
                left_id="C",
                right_id="D",
                score=0.8,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        # Should create two separate clusters: {A, B} and {C, D}
        assert len(clusters) == 2
        cluster_sets = sorted([sorted(list(c)) for c in clusters])
        assert cluster_sets == [["A", "B"], ["C", "D"]]


class TestClustererMultipleClusters:
    """Test Clusterer with disjoint graph components."""

    def test_two_disjoint_pairs_create_two_clusters(self):
        """Two independent pairs create two separate clusters."""
        clusterer = Clusterer(threshold=0.5)

        judgements = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="C",
                right_id="D",
                score=0.8,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        assert len(clusters) == 2
        cluster_sets = sorted([sorted(list(c)) for c in clusters])
        assert cluster_sets == [["A", "B"], ["C", "D"]]

    def test_multiple_clusters_different_sizes(self):
        """Multiple clusters can have different sizes."""
        clusterer = Clusterer(threshold=0.5)

        judgements = [
            # Cluster 1: {A, B, C} (size 3)
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="B",
                right_id="C",
                score=0.8,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            # Cluster 2: {D, E} (size 2)
            PairwiseJudgement(
                left_id="D",
                right_id="E",
                score=0.7,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            # Cluster 3: {F, G, H, I} (size 4)
            PairwiseJudgement(
                left_id="F",
                right_id="G",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="G",
                right_id="H",
                score=0.85,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="H",
                right_id="I",
                score=0.8,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        assert len(clusters) == 3

        # Sort clusters by size for easier assertion
        clusters_by_size = sorted(clusters, key=len)
        assert len(clusters_by_size[0]) == 2  # {D, E}
        assert len(clusters_by_size[1]) == 3  # {A, B, C}
        assert len(clusters_by_size[2]) == 4  # {F, G, H, I}

        assert clusters_by_size[0] == {"D", "E"}
        assert clusters_by_size[1] == {"A", "B", "C"}
        assert clusters_by_size[2] == {"F", "G", "H", "I"}


class TestClustererComplexGraphs:
    """Test Clusterer with complex graph structures."""

    def test_complex_graph_with_mixed_scores(self):
        """Complex graph with high and low scores produces correct clusters."""
        clusterer = Clusterer(threshold=0.6)

        judgements = [
            # High confidence cluster
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.95,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="B",
                right_id="C",
                score=0.90,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            # Low score that gets filtered out
            PairwiseJudgement(
                left_id="C",
                right_id="D",
                score=0.4,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            # Another high confidence cluster
            PairwiseJudgement(
                left_id="D",
                right_id="E",
                score=0.85,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="E",
                right_id="F",
                score=0.80,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        # Should have two clusters: {A, B, C} and {D, E, F}
        # The C-D edge is filtered out due to low score
        assert len(clusters) == 2

        cluster_sets = sorted([sorted(list(c)) for c in clusters])
        assert cluster_sets == [["A", "B", "C"], ["D", "E", "F"]]

    def test_fully_connected_cluster(self):
        """Fully connected graph (all pairs match) creates single cluster."""
        clusterer = Clusterer(threshold=0.7)

        # Create all pairs for nodes {A, B, C, D}
        judgements = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="A",
                right_id="C",
                score=0.85,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="A",
                right_id="D",
                score=0.8,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="B",
                right_id="C",
                score=0.95,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="B",
                right_id="D",
                score=0.88,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="C",
                right_id="D",
                score=0.92,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        assert len(clusters) == 1
        assert clusters[0] == {"A", "B", "C", "D"}


class TestClustererInputTypes:
    """Test that Clusterer accepts both Iterator and list inputs."""

    def test_accepts_list_input(self):
        """Clusterer.cluster() accepts a list of judgements."""
        clusterer = Clusterer(threshold=0.5)

        judgements_list = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )
        ]

        clusters = clusterer.cluster(judgements_list)

        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}

    def test_accepts_iterator_input(self):
        """Clusterer.cluster() accepts an Iterator of judgements."""
        clusterer = Clusterer(threshold=0.5)

        judgements_list = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )
        ]

        judgements_iter: Iterator[PairwiseJudgement] = iter(judgements_list)

        clusters = clusterer.cluster(judgements_iter)

        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}

    def test_consumes_iterator_fully(self):
        """Clusterer consumes the entire iterator."""
        clusterer = Clusterer(threshold=0.5)

        judgements = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="C",
                right_id="D",
                score=0.8,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(iter(judgements))

        # Should have processed both judgements
        assert len(clusters) == 2


class TestClustererEdgeCases:
    """Test Clusterer with edge cases and unusual inputs."""

    def test_duplicate_judgements_are_idempotent(self):
        """Duplicate judgements (same pair) don't affect clustering."""
        clusterer = Clusterer(threshold=0.5)

        judgements = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            # Exact duplicate
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        # Should still be just one cluster of size 2
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}

    def test_reversed_pairs_are_treated_as_same_edge(self):
        """(A, B) and (B, A) represent the same undirected edge."""
        clusterer = Clusterer(threshold=0.5)

        judgements = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            # Reversed pair
            PairwiseJudgement(
                left_id="B",
                right_id="A",
                score=0.85,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        # Should still be one cluster
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}

    def test_self_loop_is_ignored(self):
        """Judgement where left_id == right_id is handled gracefully."""
        clusterer = Clusterer(threshold=0.5)

        judgements = [
            # Self-loop (entity compared to itself)
            PairwiseJudgement(
                left_id="A",
                right_id="A",
                score=1.0,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            # Normal pair
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        clusters = clusterer.cluster(judgements)

        # Should create one cluster {A, B}
        # networkx handles self-loops gracefully
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}

    def test_different_threshold_values_produce_different_clusters(self):
        """Different threshold values affect clustering results."""
        judgements = [
            PairwiseJudgement(
                left_id="A",
                right_id="B",
                score=0.9,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="B",
                right_id="C",
                score=0.6,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            ),
        ]

        # Low threshold: B-C edge is included
        clusterer_low = Clusterer(threshold=0.5)
        clusters_low = clusterer_low.cluster(judgements)
        assert len(clusters_low) == 1
        assert clusters_low[0] == {"A", "B", "C"}

        # High threshold: B-C edge is filtered out
        clusterer_high = Clusterer(threshold=0.7)
        clusters_high = clusterer_high.cluster(judgements)
        assert len(clusters_high) == 1
        assert clusters_high[0] == {"A", "B"}
