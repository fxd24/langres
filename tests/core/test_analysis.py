"""Tests for blocker analysis functions."""

import pytest
import numpy as np

from langres.core.models import ERCandidate
from langres.core.analysis import (
    _compute_score_metrics,
    _compute_rank_metrics,
    _compute_recall_curve,
    evaluate_blocker_detailed,
)
from langres.core.reports import (
    ScoreMetrics,
    RankMetrics,
    RecallCurveStats,
    BlockerEvaluationReport,
)


# Fixtures


@pytest.fixture
def sample_candidates_with_scores():
    """Sample candidates with similarity scores for testing."""
    return [
        # Entity "1" candidates
        ERCandidate(left_id="1", right_id="2", similarity_score=0.95),  # True match
        ERCandidate(left_id="1", right_id="3", similarity_score=0.30),  # False
        ERCandidate(left_id="1", right_id="4", similarity_score=0.40),  # False
        # Entity "2" candidates
        ERCandidate(left_id="2", right_id="1", similarity_score=0.95),  # True match (duplicate)
        ERCandidate(left_id="2", right_id="3", similarity_score=0.90),  # True match
        ERCandidate(left_id="2", right_id="4", similarity_score=0.25),  # False
        # Entity "3" candidates
        ERCandidate(left_id="3", right_id="2", similarity_score=0.90),  # True match (duplicate)
        ERCandidate(left_id="3", right_id="1", similarity_score=0.30),  # False (duplicate)
        ERCandidate(left_id="3", right_id="4", similarity_score=0.35),  # False
    ]


@pytest.fixture
def sample_gold_clusters():
    """Sample ground truth clusters."""
    return [
        {"1", "2"},  # Cluster 1
        {"3", "2"},  # Cluster 2 (overlaps with cluster 1 via "2")
        {"4"},  # Cluster 3
    ]


@pytest.fixture
def clear_separation_candidates():
    """Candidates with clear separation between true and false scores."""
    return [
        # True matches: high scores (0.85-0.95)
        ERCandidate(left_id="1", right_id="2", similarity_score=0.95),
        ERCandidate(left_id="1", right_id="3", similarity_score=0.90),
        ERCandidate(left_id="2", right_id="3", similarity_score=0.85),
        # False candidates: low scores (0.15-0.35)
        ERCandidate(left_id="1", right_id="4", similarity_score=0.15),
        ERCandidate(left_id="2", right_id="4", similarity_score=0.25),
        ERCandidate(left_id="3", right_id="4", similarity_score=0.35),
    ]


@pytest.fixture
def clear_separation_clusters():
    """Gold clusters for clear separation scenario."""
    return [
        {"1", "2", "3"},  # One cluster
        {"4"},  # Separate entity
    ]


@pytest.fixture
def perfect_blocker_candidates():
    """Perfect blocker: all true matches score 1.0."""
    return [
        ERCandidate(left_id="1", right_id="2", similarity_score=1.0),
        ERCandidate(left_id="2", right_id="3", similarity_score=1.0),
        ERCandidate(left_id="1", right_id="3", similarity_score=1.0),
    ]


@pytest.fixture
def perfect_blocker_clusters():
    """Gold clusters for perfect blocker."""
    return [{"1", "2", "3"}]


@pytest.fixture
def terrible_blocker_candidates():
    """Terrible blocker: true scores lower than false scores."""
    return [
        # True matches: low scores
        ERCandidate(left_id="1", right_id="2", similarity_score=0.20),
        ERCandidate(left_id="2", right_id="3", similarity_score=0.25),
        # False candidates: high scores
        ERCandidate(left_id="1", right_id="4", similarity_score=0.80),
        ERCandidate(left_id="2", right_id="4", similarity_score=0.90),
    ]


@pytest.fixture
def terrible_blocker_clusters():
    """Gold clusters for terrible blocker."""
    return [
        {"1", "2", "3"},
        {"4"},
    ]


# Tests for _compute_score_metrics()


def test_compute_score_metrics_clear_separation(
    clear_separation_candidates, clear_separation_clusters
):
    """Test score metrics with clear separation between true and false scores."""
    metrics = _compute_score_metrics(clear_separation_candidates, clear_separation_clusters)

    # True scores should be high (mean ~0.90)
    assert metrics.true_mean > 0.85
    assert metrics.true_median >= 0.90
    assert metrics.true_std < 0.1  # Low variance

    # False scores should be low (mean ~0.25)
    assert metrics.false_mean < 0.30
    assert metrics.false_median <= 0.25
    assert metrics.false_std < 0.15

    # Separation should be positive and large
    assert metrics.separation > 0.5
    assert metrics.separation == pytest.approx(metrics.true_median - metrics.false_median, abs=0.01)

    # Overlap should be minimal
    assert metrics.overlap_fraction < 0.2

    # Histograms should exist
    assert isinstance(metrics.true_histogram, dict)
    assert isinstance(metrics.false_histogram, dict)
    assert len(metrics.true_histogram) > 0
    assert len(metrics.false_histogram) > 0


def test_compute_score_metrics_overlapping_distributions(
    sample_candidates_with_scores, sample_gold_clusters
):
    """Test score metrics with overlapping score distributions."""
    metrics = _compute_score_metrics(sample_candidates_with_scores, sample_gold_clusters)

    # Should have both true and false scores
    assert metrics.true_mean > 0
    assert metrics.false_mean > 0

    # Separation exists but may be smaller
    assert metrics.separation > 0  # True median should still be higher

    # Overlap fraction should be significant
    assert 0 < metrics.overlap_fraction < 1.0

    # Check histogram structure
    for bin_center, count in metrics.true_histogram.items():
        assert isinstance(bin_center, float)
        assert isinstance(count, int)
        assert count >= 0


def test_compute_score_metrics_perfect_blocker(
    perfect_blocker_candidates, perfect_blocker_clusters
):
    """Test score metrics with perfect blocker (all true scores = 1.0)."""
    metrics = _compute_score_metrics(perfect_blocker_candidates, perfect_blocker_clusters)

    # All true scores are 1.0
    assert metrics.true_mean == 1.0
    assert metrics.true_median == 1.0
    assert metrics.true_std == 0.0

    # No false candidates in this scenario
    assert metrics.false_mean == 0.0  # Default when no false scores
    assert metrics.false_median == 0.0
    assert metrics.false_std == 0.0

    # Maximum separation
    assert metrics.separation == 1.0

    # No overlap
    assert metrics.overlap_fraction == 0.0


def test_compute_score_metrics_terrible_blocker(
    terrible_blocker_candidates, terrible_blocker_clusters
):
    """Test score metrics with terrible blocker (true scores < false scores)."""
    metrics = _compute_score_metrics(terrible_blocker_candidates, terrible_blocker_clusters)

    # True scores should be low
    assert metrics.true_mean < 0.30

    # False scores should be high
    assert metrics.false_mean > 0.75

    # Negative separation (inverted blocker)
    assert metrics.separation < 0

    # High overlap
    assert metrics.overlap_fraction > 0.5


def test_compute_score_metrics_histogram_structure(
    clear_separation_candidates, clear_separation_clusters
):
    """Test histogram structure is correct."""
    metrics = _compute_score_metrics(clear_separation_candidates, clear_separation_clusters)

    # Check true histogram
    assert len(metrics.true_histogram) <= 50  # Max 50 bins
    total_true_count = sum(metrics.true_histogram.values())
    assert total_true_count == 3  # 3 true matches in fixture

    # Check false histogram
    assert len(metrics.false_histogram) <= 50
    total_false_count = sum(metrics.false_histogram.values())
    assert total_false_count == 3  # 3 false candidates in fixture

    # Bin centers should be floats, counts ints
    for hist in [metrics.true_histogram, metrics.false_histogram]:
        for bin_center, count in hist.items():
            assert isinstance(bin_center, float)
            assert isinstance(count, int)
            assert 0 <= bin_center <= 1.0  # Scores in [0, 1]


# Tests for _compute_rank_metrics()


def test_compute_rank_metrics_basic_ranking():
    """Test rank metrics with true matches at various positions."""
    candidates = [
        # Entity "1" candidates (sorted by score desc)
        ERCandidate(left_id="1", right_id="2", similarity_score=0.95),  # Rank 1: True
        ERCandidate(left_id="1", right_id="5", similarity_score=0.80),  # Rank 2: False
        ERCandidate(left_id="1", right_id="3", similarity_score=0.70),  # Rank 3: True
        ERCandidate(left_id="1", right_id="4", similarity_score=0.30),  # Rank 4: False
        # Entity "2" candidates
        ERCandidate(left_id="2", right_id="1", similarity_score=0.90),  # Rank 1: True
        ERCandidate(left_id="2", right_id="3", similarity_score=0.85),  # Rank 2: True
    ]
    gold_clusters = [{"1", "2", "3"}, {"4"}, {"5"}]

    metrics = _compute_rank_metrics(candidates, gold_clusters)

    # True match ranks: [1, 3] for entity 1, [1, 2] for entity 2
    # All ranks: [1, 3, 1, 2]
    assert metrics.median_rank <= 2.0
    assert metrics.percentile_95 <= 3.0

    # Should have 100% in top-5
    assert metrics.percent_in_top_5 == 1.0

    # Should have 100% in top-10
    assert metrics.percent_in_top_10 == 1.0

    # Rank histogram should show distribution
    assert isinstance(metrics.rank_histogram, dict)
    assert sum(metrics.rank_histogram.values()) == 4  # 4 true matches total


def test_compute_rank_metrics_perfect_ranking():
    """Test rank metrics when all true matches rank #1."""
    candidates = [
        # All true matches score highest
        ERCandidate(left_id="1", right_id="2", similarity_score=0.95),
        ERCandidate(left_id="1", right_id="3", similarity_score=0.40),  # False, lower
        ERCandidate(left_id="2", right_id="1", similarity_score=0.90),
        ERCandidate(left_id="2", right_id="3", similarity_score=0.35),  # False, lower
    ]
    gold_clusters = [{"1", "2"}, {"3"}]

    metrics = _compute_rank_metrics(candidates, gold_clusters)

    # All true matches at rank 1
    assert metrics.median_rank == 1.0
    assert metrics.percentile_95 == 1.0
    assert metrics.percent_in_top_5 == 1.0
    assert metrics.percent_in_top_10 == 1.0
    assert metrics.percent_in_top_20 == 1.0

    # Rank histogram should only have rank 1
    assert metrics.rank_histogram == {1: 2}  # 2 true matches, both rank 1


def test_compute_rank_metrics_poor_ranking():
    """Test rank metrics when true matches rank low."""
    candidates = [
        # True matches score low
        ERCandidate(left_id="1", right_id="5", similarity_score=0.90),  # Rank 1: False
        ERCandidate(left_id="1", right_id="6", similarity_score=0.85),  # Rank 2: False
        ERCandidate(left_id="1", right_id="7", similarity_score=0.80),  # Rank 3: False
        ERCandidate(left_id="1", right_id="2", similarity_score=0.40),  # Rank 4: True
    ]
    gold_clusters = [{"1", "2"}, {"5"}, {"6"}, {"7"}]

    metrics = _compute_rank_metrics(candidates, gold_clusters)

    # True match at rank 4
    assert metrics.median_rank == 4.0
    assert metrics.percentile_95 >= 4.0

    # Should NOT be in top-3
    assert metrics.percent_in_top_5 == 1.0  # In top-5
    assert metrics.percent_in_top_10 == 1.0
    assert metrics.percent_in_top_20 == 1.0

    # Histogram
    assert metrics.rank_histogram.get(4) == 1


def test_compute_rank_metrics_percentile_computation():
    """Test percentile computation is correct."""
    # Create scenario with known rank distribution
    candidates = []
    for i in range(1, 101):  # 100 entities
        # Each entity has one true match at rank i
        for rank in range(1, i + 1):
            is_true = rank == i
            score = 1.0 - (rank / 100.0)  # Decreasing scores
            other_id = "match" if is_true else f"false_{rank}"
            candidates.append(
                ERCandidate(left_id=f"entity_{i}", right_id=other_id, similarity_score=score)
            )

    gold_clusters = [{f"entity_{i}", "match"} for i in range(1, 101)]

    metrics = _compute_rank_metrics(candidates, gold_clusters)

    # Median rank should be around 50
    assert 45 <= metrics.median_rank <= 55

    # 95th percentile should be around 95
    assert metrics.percentile_95 >= 90


def test_compute_rank_metrics_percent_in_top_k():
    """Test percent_in_top_k calculations are correct."""
    candidates = [
        # Entity with true match at rank 3
        ERCandidate(left_id="1", right_id="x", similarity_score=0.95),  # Rank 1: False
        ERCandidate(left_id="1", right_id="y", similarity_score=0.90),  # Rank 2: False
        ERCandidate(left_id="1", right_id="2", similarity_score=0.85),  # Rank 3: True
        # Entity with true match at rank 1
        ERCandidate(left_id="3", right_id="4", similarity_score=0.90),  # Rank 1: True
    ]
    gold_clusters = [{"1", "2"}, {"3", "4"}, {"x"}, {"y"}]

    metrics = _compute_rank_metrics(candidates, gold_clusters)

    # 2 true matches total: one at rank 3, one at rank 1
    # percent_in_top_5: 2/2 = 1.0
    assert metrics.percent_in_top_5 == 1.0

    # percent_in_top_10: 2/2 = 1.0
    assert metrics.percent_in_top_10 == 1.0

    # percent_in_top_20: 2/2 = 1.0
    assert metrics.percent_in_top_20 == 1.0


# Tests for _compute_recall_curve()


def test_compute_recall_curve_basic():
    """Test recall curve computation with basic data."""
    candidates = [
        # Entity "1": 3 candidates, 2 true matches at rank 1 and 3
        ERCandidate(left_id="1", right_id="2", similarity_score=0.95),  # True
        ERCandidate(left_id="1", right_id="5", similarity_score=0.80),  # False
        ERCandidate(left_id="1", right_id="3", similarity_score=0.70),  # True
        # Entity "2": 2 candidates, 1 true match at rank 1
        ERCandidate(left_id="2", right_id="1", similarity_score=0.90),  # True
        ERCandidate(left_id="2", right_id="5", similarity_score=0.60),  # False
    ]
    gold_clusters = [{"1", "2", "3"}, {"5"}]
    k_values = [1, 2, 3]

    recall_curve = _compute_recall_curve(candidates, gold_clusters, k_values)

    # At k=1: entity 1 finds 1/2, entity 2 finds 1/1 → total 2/3
    assert recall_curve.recall_at_k[1] == pytest.approx(2 / 3, abs=0.01)

    # At k=2: entity 1 finds 1/2, entity 2 finds 1/1 → total 2/3
    assert recall_curve.recall_at_k[2] == pytest.approx(2 / 3, abs=0.01)

    # At k=3: entity 1 finds 2/2, entity 2 finds 1/1 → total 3/3 = 1.0
    assert recall_curve.recall_at_k[3] == 1.0

    # Check avg_pairs increases with k
    assert recall_curve.avg_pairs_at_k[1] == pytest.approx(1.0, abs=0.01)
    assert recall_curve.avg_pairs_at_k[2] == pytest.approx(2.0, abs=0.01)
    assert recall_curve.avg_pairs_at_k[3] == pytest.approx(2.5, abs=0.01)  # (3 + 2) / 2


def test_compute_recall_curve_recall_increases():
    """Test that recall@k increases monotonically with k."""
    candidates = [
        ERCandidate(left_id="1", right_id="2", similarity_score=0.9),  # True, rank 1
        ERCandidate(left_id="1", right_id="3", similarity_score=0.8),  # True, rank 2
        ERCandidate(left_id="1", right_id="4", similarity_score=0.7),  # False, rank 3
        ERCandidate(left_id="1", right_id="5", similarity_score=0.6),  # True, rank 4
    ]
    gold_clusters = [{"1", "2", "3", "5"}, {"4"}]
    k_values = [1, 2, 3, 4]

    recall_curve = _compute_recall_curve(candidates, gold_clusters, k_values)

    # Recall should increase
    prev_recall = 0.0
    for k in k_values:
        assert recall_curve.recall_at_k[k] >= prev_recall
        prev_recall = recall_curve.recall_at_k[k]


def test_compute_recall_curve_cost_increases():
    """Test that avg_pairs@k increases with k (cost proxy)."""
    candidates = [
        ERCandidate(left_id="1", right_id="2", similarity_score=0.9),
        ERCandidate(left_id="1", right_id="3", similarity_score=0.8),
        ERCandidate(left_id="1", right_id="4", similarity_score=0.7),
    ]
    gold_clusters = [{"1", "2"}, {"3"}, {"4"}]
    k_values = [1, 2, 3]

    recall_curve = _compute_recall_curve(candidates, gold_clusters, k_values)

    # avg_pairs should increase
    assert recall_curve.avg_pairs_at_k[1] < recall_curve.avg_pairs_at_k[2]
    assert recall_curve.avg_pairs_at_k[2] < recall_curve.avg_pairs_at_k[3]


def test_compute_recall_curve_plateau():
    """Test recall plateaus when all true matches are found."""
    candidates = [
        ERCandidate(left_id="1", right_id="2", similarity_score=0.9),  # True, rank 1
        ERCandidate(left_id="1", right_id="3", similarity_score=0.8),  # False
        ERCandidate(left_id="1", right_id="4", similarity_score=0.7),  # False
    ]
    gold_clusters = [{"1", "2"}, {"3"}, {"4"}]
    k_values = [1, 2, 3]

    recall_curve = _compute_recall_curve(candidates, gold_clusters, k_values)

    # Recall should be 1.0 at all k values (only 1 true match at rank 1)
    assert recall_curve.recall_at_k[1] == 1.0
    assert recall_curve.recall_at_k[2] == 1.0
    assert recall_curve.recall_at_k[3] == 1.0


# Tests for evaluate_blocker_detailed()


def test_evaluate_blocker_detailed_complete_pipeline(
    clear_separation_candidates, clear_separation_clusters
):
    """Test complete evaluation pipeline with all metrics."""
    report = evaluate_blocker_detailed(clear_separation_candidates, clear_separation_clusters)

    # Check all sections are populated
    assert isinstance(report.candidates, type(report.candidates))
    assert isinstance(report.ranking, type(report.ranking))
    assert isinstance(report.scores, ScoreMetrics)
    assert isinstance(report.ranks, RankMetrics)
    assert isinstance(report.recall_curve, RecallCurveStats)

    # Candidate metrics
    assert report.candidates.recall > 0
    assert report.candidates.total == 6

    # Ranking metrics
    assert 0 <= report.ranking.map <= 1.0
    assert 0 <= report.ranking.mrr <= 1.0

    # Score metrics
    assert report.scores.true_mean > report.scores.false_mean
    assert report.scores.separation > 0

    # Rank metrics
    assert report.ranks.median_rank >= 1

    # Recall curve (default k_values)
    assert 1 in report.recall_curve.recall_at_k
    assert 5 in report.recall_curve.recall_at_k
    assert 10 in report.recall_curve.recall_at_k


def test_evaluate_blocker_detailed_custom_k_values():
    """Test evaluate_blocker_detailed with custom k_values."""
    candidates = [
        ERCandidate(left_id="1", right_id="2", similarity_score=0.9),
    ]
    gold_clusters = [{"1", "2"}]
    custom_k = [1, 3, 7]

    report = evaluate_blocker_detailed(candidates, gold_clusters, k_values=custom_k)

    # Check custom k_values used
    assert set(report.recall_curve.recall_at_k.keys()) == set(custom_k)
    assert set(report.recall_curve.avg_pairs_at_k.keys()) == set(custom_k)


def test_evaluate_blocker_detailed_all_metric_categories():
    """Test all metric categories are correctly populated."""
    candidates = [
        ERCandidate(left_id="1", right_id="2", similarity_score=0.95),
        ERCandidate(left_id="1", right_id="3", similarity_score=0.40),
        ERCandidate(left_id="2", right_id="3", similarity_score=0.85),
    ]
    gold_clusters = [{"1", "2"}, {"3"}]

    report = evaluate_blocker_detailed(candidates, gold_clusters)

    # Verify BlockerEvaluationReport structure
    assert hasattr(report, "candidates")
    assert hasattr(report, "ranking")
    assert hasattr(report, "scores")
    assert hasattr(report, "ranks")
    assert hasattr(report, "recall_curve")

    # Candidate metrics
    assert hasattr(report.candidates, "recall")
    assert hasattr(report.candidates, "precision")
    assert hasattr(report.candidates, "total")
    assert hasattr(report.candidates, "avg_per_entity")

    # Ranking metrics
    assert hasattr(report.ranking, "map")
    assert hasattr(report.ranking, "mrr")
    assert hasattr(report.ranking, "ndcg_at_10")

    # Score metrics
    assert hasattr(report.scores, "true_mean")
    assert hasattr(report.scores, "false_mean")
    assert hasattr(report.scores, "separation")

    # Rank metrics
    assert hasattr(report.ranks, "median_rank")
    assert hasattr(report.ranks, "percentile_95")

    # Recall curve
    assert hasattr(report.recall_curve, "recall_at_k")
    assert hasattr(report.recall_curve, "avg_pairs_at_k")


def test_evaluate_blocker_detailed_realistic_data():
    """Test with realistic blocker evaluation scenario."""
    # Simulate a decent blocker: high-scoring true matches, some noise
    candidates = []

    # Entity 1: has true matches with entities 2, 3
    candidates.extend(
        [
            ERCandidate(left_id="1", right_id="2", similarity_score=0.92),  # True, rank 1
            ERCandidate(left_id="1", right_id="3", similarity_score=0.88),  # True, rank 2
            ERCandidate(left_id="1", right_id="4", similarity_score=0.65),  # False, rank 3
            ERCandidate(left_id="1", right_id="5", similarity_score=0.45),  # False, rank 4
        ]
    )

    # Entity 2: has true matches with 1, 3
    candidates.extend(
        [
            ERCandidate(left_id="2", right_id="3", similarity_score=0.90),  # True, rank 1
            ERCandidate(left_id="2", right_id="1", similarity_score=0.85),  # True, rank 2
            ERCandidate(left_id="2", right_id="4", similarity_score=0.50),  # False, rank 3
        ]
    )

    gold_clusters = [{"1", "2", "3"}, {"4"}, {"5"}]

    report = evaluate_blocker_detailed(candidates, gold_clusters)

    # High recall (all true matches found)
    assert report.candidates.recall == 1.0

    # Good ranking (true matches rank high)
    assert report.ranking.map > 0.8
    assert report.ranks.median_rank <= 2

    # Clear score separation
    assert report.scores.separation > 0.2

    # Recall curve
    assert report.recall_curve.recall_at_k[1] > 0.5  # At least 50% at k=1
    assert report.recall_curve.recall_at_k[5] == 1.0  # All found by k=5
