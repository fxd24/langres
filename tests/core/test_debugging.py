"""
Test suite for PipelineDebugger utility.

This module tests the debugging capabilities for entity resolution pipelines,
including candidate generation analysis, scoring analysis, clustering analysis,
and report generation.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from langres.core.debugging import (
    CandidateStats,
    ClusterStats,
    ErrorExample,
    PipelineDebugger,
    ScoreStats,
)
from langres.core.models import ERCandidate, PairwiseJudgement


# Test entity schema
class TestEntity(BaseModel):
    """Simple test entity schema with id field."""

    id: str
    name: str


# Fixtures
@pytest.fixture
def ground_truth_clusters() -> list[set[str]]:
    """Ground truth clusters for testing."""
    return [
        {"e1", "e2", "e3"},  # Cluster 0: 3 entities
        {"e4", "e5"},  # Cluster 1: 2 entities
        {"e6"},  # Cluster 2: singleton
    ]


@pytest.fixture
def test_entities() -> list[TestEntity]:
    """Test entities matching ground truth."""
    return [
        TestEntity(id="e1", name="Company A"),
        TestEntity(id="e2", name="Company A Inc"),
        TestEntity(id="e3", name="Company A LLC"),
        TestEntity(id="e4", name="Company B"),
        TestEntity(id="e5", name="Company B Corp"),
        TestEntity(id="e6", name="Company C"),
    ]


@pytest.fixture
def perfect_candidates(test_entities: list[TestEntity]) -> list[ERCandidate[TestEntity]]:
    """Perfect candidate generation: all and only true matches."""
    return [
        # Cluster 0 pairs (e1, e2, e3)
        ERCandidate(left=test_entities[0], right=test_entities[1], blocker_name="test"),
        ERCandidate(left=test_entities[0], right=test_entities[2], blocker_name="test"),
        ERCandidate(left=test_entities[1], right=test_entities[2], blocker_name="test"),
        # Cluster 1 pairs (e4, e5)
        ERCandidate(left=test_entities[3], right=test_entities[4], blocker_name="test"),
    ]


@pytest.fixture
def imperfect_candidates(test_entities: list[TestEntity]) -> list[ERCandidate[TestEntity]]:
    """Imperfect candidates: missing one match, including one false positive."""
    return [
        # Cluster 0: missing e2-e3 pair
        ERCandidate(left=test_entities[0], right=test_entities[1], blocker_name="test"),
        ERCandidate(left=test_entities[0], right=test_entities[2], blocker_name="test"),
        # Cluster 1: all pairs present
        ERCandidate(left=test_entities[3], right=test_entities[4], blocker_name="test"),
        # False positive: e1-e4 (different clusters)
        ERCandidate(left=test_entities[0], right=test_entities[3], blocker_name="test"),
    ]


@pytest.fixture
def perfect_judgements() -> list[PairwiseJudgement]:
    """Perfect scores: high for matches, low for non-matches."""
    return [
        # True matches with high scores
        PairwiseJudgement(
            left_id="e1",
            right_id="e2",
            score=0.95,
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Same company name",
            provenance={"model": "gpt-4"},
        ),
        PairwiseJudgement(
            left_id="e1",
            right_id="e3",
            score=0.92,
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Same company name",
            provenance={"model": "gpt-4"},
        ),
        PairwiseJudgement(
            left_id="e2",
            right_id="e3",
            score=0.90,
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Same company name",
            provenance={"model": "gpt-4"},
        ),
        PairwiseJudgement(
            left_id="e4",
            right_id="e5",
            score=0.88,
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Same company name",
            provenance={"model": "gpt-4"},
        ),
    ]


@pytest.fixture
def imperfect_judgements() -> list[PairwiseJudgement]:
    """Imperfect scores: some misaligned with ground truth."""
    return [
        # True match with good score
        PairwiseJudgement(
            left_id="e1",
            right_id="e2",
            score=0.95,
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Same company",
            provenance={},
        ),
        # True match with LOW score (should be flagged)
        PairwiseJudgement(
            left_id="e1",
            right_id="e3",
            score=0.25,
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Unclear",
            provenance={},
        ),
        # Non-match with HIGH score (false positive, should be flagged)
        PairwiseJudgement(
            left_id="e1",
            right_id="e4",
            score=0.85,
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Similar names",
            provenance={},
        ),
        # True match with good score
        PairwiseJudgement(
            left_id="e4",
            right_id="e5",
            score=0.90,
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Same company",
            provenance={},
        ),
    ]


# =============================
# Initialization Tests
# =============================


def test_initialization_builds_entity_to_cluster_mapping(
    ground_truth_clusters: list[set[str]],
) -> None:
    """Test that initialization correctly maps entity IDs to cluster IDs."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    # Check entity_to_cluster mapping
    assert debugger.entity_to_cluster["e1"] == 0
    assert debugger.entity_to_cluster["e2"] == 0
    assert debugger.entity_to_cluster["e3"] == 0
    assert debugger.entity_to_cluster["e4"] == 1
    assert debugger.entity_to_cluster["e5"] == 1
    assert debugger.entity_to_cluster["e6"] == 2


def test_initialization_builds_true_matches_set(
    ground_truth_clusters: list[set[str]],
) -> None:
    """Test that initialization correctly identifies all true match pairs."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    # Check true_matches set (should be sorted tuples)
    expected_matches = {
        ("e1", "e2"),
        ("e1", "e3"),
        ("e2", "e3"),
        ("e4", "e5"),
    }
    assert debugger.true_matches == expected_matches


def test_initialization_handles_empty_clusters() -> None:
    """Test that initialization handles empty cluster list gracefully."""
    debugger = PipelineDebugger(ground_truth_clusters=[])

    assert debugger.entity_to_cluster == {}
    assert debugger.true_matches == set()


def test_initialization_with_sample_size() -> None:
    """Test that sample_size parameter is stored correctly."""
    debugger = PipelineDebugger(ground_truth_clusters=[{"e1", "e2"}], sample_size=5)

    assert debugger.sample_size == 5


# =============================
# analyze_candidates Tests
# =============================


def test_analyze_candidates_perfect_recall_precision(
    ground_truth_clusters: list[set[str]],
    perfect_candidates: list[ERCandidate[TestEntity]],
    test_entities: list[TestEntity],
) -> None:
    """Test candidate analysis with perfect blocker (100% recall and precision)."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    stats = debugger.analyze_candidates(perfect_candidates, test_entities)

    # Perfect recall: all 4 true matches found
    assert stats.total_candidates == 4
    assert stats.candidate_recall == 1.0
    # Perfect precision: all 4 candidates are true matches
    assert stats.candidate_precision == 1.0
    assert stats.missed_matches_count == 0
    assert stats.false_positive_candidates_count == 0
    # Average candidates per entity: 4 pairs / 6 entities (counting each entity once per pair)
    # Actually: (e1:3, e2:2, e3:2, e4:1, e5:1, e6:0) / 6 = 9/6 = 1.5
    assert stats.avg_candidates_per_entity == pytest.approx(1.5, abs=0.01)


def test_analyze_candidates_imperfect_blocker(
    ground_truth_clusters: list[set[str]],
    imperfect_candidates: list[ERCandidate[TestEntity]],
    test_entities: list[TestEntity],
) -> None:
    """Test candidate analysis with imperfect blocker (missed match and false positive)."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    stats = debugger.analyze_candidates(imperfect_candidates, test_entities)

    # 4 total candidates, but 1 is false positive
    assert stats.total_candidates == 4
    # Recall: 3 out of 4 true matches found (missing e2-e3)
    assert stats.candidate_recall == pytest.approx(0.75, abs=0.01)
    # Precision: 3 out of 4 candidates are true matches
    assert stats.candidate_precision == pytest.approx(0.75, abs=0.01)
    assert stats.missed_matches_count == 1
    assert stats.false_positive_candidates_count == 1


def test_analyze_candidates_generates_error_examples(
    ground_truth_clusters: list[set[str]],
    imperfect_candidates: list[ERCandidate[TestEntity]],
    test_entities: list[TestEntity],
) -> None:
    """Test that analyze_candidates generates ErrorExample objects for issues."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters, sample_size=10)
    stats = debugger.analyze_candidates(imperfect_candidates, test_entities)

    # Should have stored error examples
    assert len(debugger.error_examples) > 0

    # Should have missed_match error for e2-e3
    missed_errors = [e for e in debugger.error_examples if e.error_type == "missed_match"]
    assert len(missed_errors) == 1
    assert set(missed_errors[0].entity_ids) == {"e2", "e3"}


def test_analyze_candidates_respects_sample_size(
    ground_truth_clusters: list[set[str]],
    test_entities: list[TestEntity],
) -> None:
    """Test that error examples are limited by sample_size."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters, sample_size=2)

    # Create many missed matches
    candidates: list[ERCandidate[TestEntity]] = []  # Empty - all matches missed
    stats = debugger.analyze_candidates(candidates, test_entities)

    # Should have 4 missed matches, but only sample 2
    assert stats.missed_matches_count == 4
    missed_errors = [e for e in debugger.error_examples if e.error_type == "missed_match"]
    assert len(missed_errors) == 2


def test_analyze_candidates_with_no_candidates(
    ground_truth_clusters: list[set[str]],
    test_entities: list[TestEntity],
) -> None:
    """Test candidate analysis with empty candidate list."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    stats = debugger.analyze_candidates([], test_entities)

    assert stats.total_candidates == 0
    assert stats.candidate_recall == 0.0
    assert stats.candidate_precision == 0.0  # 0/0 edge case
    assert stats.missed_matches_count == 4  # All true matches missed
    assert stats.false_positive_candidates_count == 0


def test_analyze_candidates_with_no_true_matches() -> None:
    """Test candidate analysis when ground truth has no matches (all singletons)."""
    debugger = PipelineDebugger(ground_truth_clusters=[{"e1"}, {"e2"}, {"e3"}])
    entities = [
        TestEntity(id="e1", name="A"),
        TestEntity(id="e2", name="B"),
        TestEntity(id="e3", name="C"),
    ]
    # Create a false positive candidate
    candidates = [ERCandidate(left=entities[0], right=entities[1], blocker_name="test")]

    stats = debugger.analyze_candidates(candidates, entities)

    assert stats.total_candidates == 1
    assert stats.candidate_recall == 1.0  # No true matches to miss
    assert stats.candidate_precision == 0.0  # All candidates are false positives
    assert stats.missed_matches_count == 0
    assert stats.false_positive_candidates_count == 1


# =============================
# analyze_scores Tests
# =============================


def test_analyze_scores_perfect_separation(
    ground_truth_clusters: list[set[str]],
    perfect_judgements: list[PairwiseJudgement],
) -> None:
    """Test score analysis with perfect separation between matches and non-matches."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    stats = debugger.analyze_scores(perfect_judgements)

    # All scores should be in 0.88-0.95 range
    assert stats.mean_score == pytest.approx(0.9125, abs=0.01)  # (0.95+0.92+0.90+0.88)/4
    assert stats.median_score == pytest.approx(0.91, abs=0.01)  # (0.90+0.92)/2
    assert stats.std_score > 0.0

    # All judgements are true matches
    assert stats.true_match_mean == pytest.approx(0.9125, abs=0.01)
    assert stats.non_match_mean == 0.0  # No non-matches
    assert stats.separation == pytest.approx(0.9125, abs=0.01)


def test_analyze_scores_imperfect_calibration(
    ground_truth_clusters: list[set[str]],
    imperfect_judgements: list[PairwiseJudgement],
) -> None:
    """Test score analysis with imperfect calibration (some misaligned scores)."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    stats = debugger.analyze_scores(imperfect_judgements)

    # Calculate expected values
    # Scores: [0.95, 0.25, 0.85, 0.90]
    # True matches: e1-e2 (0.95), e1-e3 (0.25), e4-e5 (0.90) -> mean = 0.70
    # Non-matches: e1-e4 (0.85) -> mean = 0.85
    assert stats.true_match_mean == pytest.approx(0.70, abs=0.01)
    assert stats.non_match_mean == pytest.approx(0.85, abs=0.01)
    # Negative separation indicates poor calibration
    assert stats.separation == pytest.approx(-0.15, abs=0.01)


def test_analyze_scores_generates_error_examples(
    ground_truth_clusters: list[set[str]],
    imperfect_judgements: list[PairwiseJudgement],
) -> None:
    """Test that score analysis generates error examples for misaligned scores."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters, sample_size=10)
    stats = debugger.analyze_scores(imperfect_judgements)

    # Should flag low-scoring match (e1-e3 with score 0.25)
    low_match_errors = [e for e in debugger.error_examples if e.error_type == "low_scoring_match"]
    assert len(low_match_errors) == 1
    assert set(low_match_errors[0].entity_ids) == {"e1", "e3"}

    # Should flag high-scoring non-match (e1-e4 with score 0.85)
    high_nonmatch_errors = [
        e for e in debugger.error_examples if e.error_type == "high_scoring_nonmatch"
    ]
    assert len(high_nonmatch_errors) == 1
    assert set(high_nonmatch_errors[0].entity_ids) == {"e1", "e4"}


def test_analyze_scores_calculates_percentiles(
    ground_truth_clusters: list[set[str]],
    perfect_judgements: list[PairwiseJudgement],
) -> None:
    """Test that score analysis calculates percentiles correctly."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    stats = debugger.analyze_scores(perfect_judgements)

    # Scores: [0.88, 0.90, 0.92, 0.95]
    assert stats.p25 == pytest.approx(0.89, abs=0.02)
    assert stats.p75 == pytest.approx(0.9275, abs=0.02)
    assert stats.p95 == pytest.approx(0.946, abs=0.02)


def test_analyze_scores_with_empty_judgements(ground_truth_clusters: list[set[str]]) -> None:
    """Test score analysis with no judgements."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    stats = debugger.analyze_scores([])

    # All stats should be 0.0 or NaN for empty data
    assert stats.mean_score == 0.0
    assert stats.median_score == 0.0
    assert stats.true_match_mean == 0.0
    assert stats.non_match_mean == 0.0
    assert stats.separation == 0.0


def test_analyze_scores_metadata_includes_reasoning(
    ground_truth_clusters: list[set[str]],
    imperfect_judgements: list[PairwiseJudgement],
) -> None:
    """Test that error examples include LLM reasoning in metadata."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters, sample_size=10)
    stats = debugger.analyze_scores(imperfect_judgements)

    # Find the low-scoring match error
    low_match_errors = [e for e in debugger.error_examples if e.error_type == "low_scoring_match"]
    assert len(low_match_errors) == 1

    # Should include score and reasoning in metadata
    error = low_match_errors[0]
    assert "score" in error.metadata
    assert "reasoning" in error.metadata
    assert error.metadata["reasoning"] == "Unclear"


# =============================
# analyze_clusters Tests
# =============================


def test_analyze_clusters_perfect_clustering(ground_truth_clusters: list[set[str]]) -> None:
    """Test cluster analysis with perfect predicted clusters."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    # Predict exactly the ground truth
    predicted = [{"e1", "e2", "e3"}, {"e4", "e5"}, {"e6"}]
    stats = debugger.analyze_clusters(predicted)

    assert stats.num_predicted_clusters == 3
    assert stats.num_gold_clusters == 3
    assert stats.avg_cluster_size == pytest.approx(2.0, abs=0.01)  # (3+2+1)/3
    assert stats.num_singletons == 1
    assert stats.largest_cluster_size == 3
    assert stats.num_false_merges == 0
    assert stats.num_false_splits == 0


def test_analyze_clusters_false_merge(ground_truth_clusters: list[set[str]]) -> None:
    """Test cluster analysis with false merge (entities from different clusters grouped)."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    # Merge cluster 0 and cluster 1 incorrectly
    predicted = [{"e1", "e2", "e3", "e4", "e5"}, {"e6"}]
    stats = debugger.analyze_clusters(predicted)

    assert stats.num_predicted_clusters == 2
    # One false merge: predicted cluster contains entities from clusters 0 and 1
    assert stats.num_false_merges == 1
    # No splits in this case
    assert stats.num_false_splits == 0


def test_analyze_clusters_false_split(ground_truth_clusters: list[set[str]]) -> None:
    """Test cluster analysis with false split (gold cluster split across predictions)."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    # Split cluster 0 incorrectly
    predicted = [{"e1"}, {"e2", "e3"}, {"e4", "e5"}, {"e6"}]
    stats = debugger.analyze_clusters(predicted)

    assert stats.num_predicted_clusters == 4
    # Gold cluster 0 is split across 2 predicted clusters
    assert stats.num_false_splits == 1
    # No merges in this case
    assert stats.num_false_merges == 0


def test_analyze_clusters_generates_error_examples(
    ground_truth_clusters: list[set[str]],
) -> None:
    """Test that cluster analysis generates error examples."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters, sample_size=10)
    # Create a false merge
    predicted = [{"e1", "e2", "e3", "e4"}, {"e5"}, {"e6"}]
    stats = debugger.analyze_clusters(predicted)

    # Should have false merge error
    merge_errors = [e for e in debugger.error_examples if e.error_type == "false_merge"]
    assert len(merge_errors) >= 1

    # Check error contains relevant entity IDs
    merge_error = merge_errors[0]
    assert len(merge_error.entity_ids) > 0


def test_analyze_clusters_calculates_singleton_count(
    ground_truth_clusters: list[set[str]],
) -> None:
    """Test that singleton count is calculated correctly."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    # All entities as singletons
    predicted = [{"e1"}, {"e2"}, {"e3"}, {"e4"}, {"e5"}, {"e6"}]
    stats = debugger.analyze_clusters(predicted)

    assert stats.num_singletons == 6
    assert stats.avg_cluster_size == 1.0


def test_analyze_clusters_with_empty_predictions(
    ground_truth_clusters: list[set[str]],
) -> None:
    """Test cluster analysis with no predicted clusters."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    stats = debugger.analyze_clusters([])

    assert stats.num_predicted_clusters == 0
    assert stats.num_gold_clusters == 3
    assert stats.avg_cluster_size == 0.0
    assert stats.num_singletons == 0


# =============================
# Report Generation Tests
# =============================


def test_to_dict_returns_complete_structure(
    ground_truth_clusters: list[set[str]],
    perfect_candidates: list[ERCandidate[TestEntity]],
    perfect_judgements: list[PairwiseJudgement],
    test_entities: list[TestEntity],
) -> None:
    """Test that to_dict returns complete dictionary with all stats."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    # Run all analyses
    debugger.analyze_candidates(perfect_candidates, test_entities)
    debugger.analyze_scores(perfect_judgements)
    debugger.analyze_clusters([{"e1", "e2", "e3"}, {"e4", "e5"}, {"e6"}])

    result = debugger.to_dict()

    # Check top-level keys
    assert "candidate_stats" in result
    assert "score_stats" in result
    assert "cluster_stats" in result
    assert "error_examples" in result
    assert "recommendations" in result

    # Check nested structure
    assert isinstance(result["candidate_stats"], dict)
    assert isinstance(result["score_stats"], dict)
    assert isinstance(result["cluster_stats"], dict)
    assert isinstance(result["error_examples"], list)
    assert isinstance(result["recommendations"], list)


def test_to_markdown_generates_valid_markdown(
    ground_truth_clusters: list[set[str]],
    perfect_candidates: list[ERCandidate[TestEntity]],
    perfect_judgements: list[PairwiseJudgement],
    test_entities: list[TestEntity],
) -> None:
    """Test that to_markdown generates valid markdown with all sections."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    # Run all analyses
    debugger.analyze_candidates(perfect_candidates, test_entities)
    debugger.analyze_scores(perfect_judgements)
    debugger.analyze_clusters([{"e1", "e2", "e3"}, {"e4", "e5"}, {"e6"}])

    markdown = debugger.to_markdown()

    # Check for required sections
    assert "# Pipeline Debug Report" in markdown
    assert "## Candidate Generation" in markdown
    assert "## Score Distribution" in markdown
    assert "## Clustering Results" in markdown
    assert "## Error Examples" in markdown
    assert "## Recommendations" in markdown

    # Check for table formatting
    assert "|" in markdown  # Tables should be present
    assert "---" in markdown  # Table separators


def test_generate_recommendations_low_candidate_recall(
    ground_truth_clusters: list[set[str]],
    test_entities: list[TestEntity],
) -> None:
    """Test that recommendations suggest increasing k_neighbors for low recall."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    # Create candidates with low recall (only 1 of 4 true matches)
    candidates = [
        ERCandidate(
            left=test_entities[0], right=test_entities[1], blocker_name="test"
        ),  # Only e1-e2
    ]
    debugger.analyze_candidates(candidates, test_entities)

    recommendations = debugger._generate_recommendations()

    # Should recommend increasing k_neighbors
    assert any("k_neighbors" in rec.lower() or "recall" in rec.lower() for rec in recommendations)


def test_generate_recommendations_low_candidate_precision(
    ground_truth_clusters: list[set[str]],
    test_entities: list[TestEntity],
) -> None:
    """Test that recommendations suggest decreasing k_neighbors for low precision."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    # Create many false positive candidates (low precision)
    candidates = [
        ERCandidate(left=test_entities[i], right=test_entities[j], blocker_name="test")
        for i in range(6)
        for j in range(i + 1, 6)
    ]  # All pairs (15 total, only 4 are true matches)
    debugger.analyze_candidates(candidates, test_entities)

    recommendations = debugger._generate_recommendations()

    # Should recommend decreasing k_neighbors or improving blocker
    assert any(
        "precision" in rec.lower() or "false positive" in rec.lower() for rec in recommendations
    )


def test_generate_recommendations_poor_score_separation(
    ground_truth_clusters: list[set[str]],
    imperfect_judgements: list[PairwiseJudgement],
) -> None:
    """Test that recommendations suggest improving LLM prompt for poor separation."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    debugger.analyze_scores(imperfect_judgements)

    recommendations = debugger._generate_recommendations()

    # Should recommend improving prompt or score calibration
    assert any(
        "prompt" in rec.lower() or "separation" in rec.lower() or "calibration" in rec.lower()
        for rec in recommendations
    )


def test_generate_recommendations_many_false_merges(
    ground_truth_clusters: list[set[str]],
) -> None:
    """Test that recommendations suggest increasing threshold for false merges."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    # Create clustering with false merge
    predicted = [{"e1", "e2", "e3", "e4", "e5"}, {"e6"}]  # Merged clusters 0 and 1
    debugger.analyze_clusters(predicted)

    recommendations = debugger._generate_recommendations()

    # Should recommend increasing threshold
    assert any("threshold" in rec.lower() and "increas" in rec.lower() for rec in recommendations)


def test_generate_recommendations_many_false_splits(
    ground_truth_clusters: list[set[str]],
) -> None:
    """Test that recommendations suggest decreasing threshold for false splits."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    # Create clustering with false splits
    predicted = [{"e1"}, {"e2"}, {"e3"}, {"e4"}, {"e5"}, {"e6"}]  # All singletons
    debugger.analyze_clusters(predicted)

    recommendations = debugger._generate_recommendations()

    # Should recommend decreasing threshold
    assert any("threshold" in rec.lower() and "decreas" in rec.lower() for rec in recommendations)


def test_generate_recommendations_many_singletons(
    ground_truth_clusters: list[set[str]],
) -> None:
    """Test that recommendations suggest checking blocker for many singletons."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    # Create clustering with many singletons
    predicted = [{"e1"}, {"e2"}, {"e3"}, {"e4"}, {"e5"}, {"e6"}]
    debugger.analyze_clusters(predicted)

    recommendations = debugger._generate_recommendations()

    # Should recommend checking blocker recall
    assert any("singleton" in rec.lower() or "recall" in rec.lower() for rec in recommendations)


# =============================
# File I/O Tests
# =============================


def test_save_report_json(
    ground_truth_clusters: list[set[str]],
    perfect_candidates: list[ERCandidate[TestEntity]],
    test_entities: list[TestEntity],
    tmp_path: Path,
) -> None:
    """Test saving report as JSON file."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    debugger.analyze_candidates(perfect_candidates, test_entities)

    output_file = tmp_path / "debug_report.json"
    debugger.save_report(output_file, format="json")

    # Check file was created
    assert output_file.exists()

    # Check content is valid JSON
    with open(output_file) as f:
        data = json.load(f)

    assert "candidate_stats" in data
    assert "recommendations" in data


def test_save_report_markdown(
    ground_truth_clusters: list[set[str]],
    perfect_candidates: list[ERCandidate[TestEntity]],
    test_entities: list[TestEntity],
    tmp_path: Path,
) -> None:
    """Test saving report as markdown file."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    debugger.analyze_candidates(perfect_candidates, test_entities)

    output_file = tmp_path / "debug_report.md"
    debugger.save_report(output_file, format="markdown")

    # Check file was created
    assert output_file.exists()

    # Check content is markdown
    content = output_file.read_text()
    assert "# Pipeline Debug Report" in content
    assert "##" in content  # Section headers


def test_save_report_creates_parent_directories(
    ground_truth_clusters: list[set[str]],
    perfect_candidates: list[ERCandidate[TestEntity]],
    test_entities: list[TestEntity],
    tmp_path: Path,
) -> None:
    """Test that save_report creates parent directories if needed."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)
    debugger.analyze_candidates(perfect_candidates, test_entities)

    # Use nested path that doesn't exist
    output_file = tmp_path / "nested" / "dir" / "report.json"
    debugger.save_report(output_file, format="json")

    assert output_file.exists()


# =============================
# Integration Test
# =============================


def test_full_pipeline_integration(
    ground_truth_clusters: list[set[str]],
    imperfect_candidates: list[ERCandidate[TestEntity]],
    imperfect_judgements: list[PairwiseJudgement],
    test_entities: list[TestEntity],
    tmp_path: Path,
) -> None:
    """Test full pipeline: all analyses + report generation."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters, sample_size=5)

    # Run all analyses
    candidate_stats = debugger.analyze_candidates(imperfect_candidates, test_entities)
    score_stats = debugger.analyze_scores(imperfect_judgements)
    cluster_stats = debugger.analyze_clusters([{"e1", "e2", "e3", "e4"}, {"e5"}, {"e6"}])

    # Verify stats objects are populated
    assert candidate_stats.candidate_recall < 1.0
    assert score_stats.separation < 0.0  # Poor calibration
    assert cluster_stats.num_false_merges > 0

    # Generate reports
    json_file = tmp_path / "report.json"
    md_file = tmp_path / "report.md"
    debugger.save_report(json_file, format="json")
    debugger.save_report(md_file, format="markdown")

    # Verify both files created
    assert json_file.exists()
    assert md_file.exists()

    # Verify error examples collected across all stages
    assert len(debugger.error_examples) > 0
    error_types = {e.error_type for e in debugger.error_examples}
    # Should have errors from multiple stages
    assert len(error_types) >= 2

    # Verify recommendations are context-appropriate
    recommendations = debugger._generate_recommendations()
    assert len(recommendations) > 0
    # Should address multiple issues
    assert any("recall" in rec.lower() or "precision" in rec.lower() for rec in recommendations)
    assert any(
        "separation" in rec.lower() or "calibration" in rec.lower() for rec in recommendations
    )


# =============================
# Edge Cases and Error Handling
# =============================


def test_handles_duplicate_entity_ids_in_clusters() -> None:
    """Test that initialization handles duplicate entity IDs across clusters (invalid input)."""
    # This is technically invalid input, but we should handle gracefully
    # The later occurrence should overwrite
    debugger = PipelineDebugger(ground_truth_clusters=[{"e1", "e2"}, {"e2", "e3"}])

    # e2 appears in both clusters - should map to the last one (cluster 1)
    assert debugger.entity_to_cluster["e2"] == 1


def test_analyze_candidates_with_unknown_entity_ids(
    ground_truth_clusters: list[set[str]],
    test_entities: list[TestEntity],
) -> None:
    """Test candidate analysis when candidate contains entity not in ground truth."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    # Create candidate with unknown entity ID
    unknown_entity = TestEntity(id="unknown", name="Unknown Corp")
    candidates = [
        ERCandidate(left=test_entities[0], right=unknown_entity, blocker_name="test"),
    ]

    # Should handle gracefully (treat as non-match since not in ground truth)
    stats = debugger.analyze_candidates(candidates, test_entities)

    assert stats.total_candidates == 1
    # This pair is not a true match (unknown entity)
    assert stats.false_positive_candidates_count == 1


def test_to_dict_before_any_analysis(ground_truth_clusters: list[set[str]]) -> None:
    """Test that to_dict works even when no analyses have been run."""
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters)

    result = debugger.to_dict()

    # Should return structure with None or empty values
    assert "candidate_stats" in result
    assert "score_stats" in result
    assert "cluster_stats" in result
    assert result["error_examples"] == []


def test_error_example_dataclass_creation() -> None:
    """Test that ErrorExample dataclass can be created correctly."""
    error = ErrorExample(
        error_type="test_error",
        entity_ids=["e1", "e2"],
        entity_texts=["Company A", "Company B"],
        explanation="Test explanation",
        metadata={"key": "value"},
    )

    assert error.error_type == "test_error"
    assert error.entity_ids == ["e1", "e2"]
    assert error.metadata["key"] == "value"


def test_stats_dataclasses_creation() -> None:
    """Test that stats dataclasses can be created correctly."""
    candidate_stats = CandidateStats(
        total_candidates=10,
        avg_candidates_per_entity=2.5,
        candidate_recall=0.95,
        candidate_precision=0.85,
        missed_matches_count=2,
        false_positive_candidates_count=3,
    )
    assert candidate_stats.total_candidates == 10

    score_stats = ScoreStats(
        mean_score=0.75,
        median_score=0.78,
        std_score=0.15,
        p25=0.65,
        p75=0.85,
        p95=0.95,
        true_match_mean=0.90,
        non_match_mean=0.30,
        separation=0.60,
    )
    assert score_stats.separation == 0.60

    cluster_stats = ClusterStats(
        num_predicted_clusters=5,
        num_gold_clusters=6,
        avg_cluster_size=3.2,
        num_singletons=2,
        largest_cluster_size=8,
        num_false_merges=1,
        num_false_splits=2,
    )
    assert cluster_stats.num_false_merges == 1
