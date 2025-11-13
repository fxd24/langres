"""Tests for ranking evaluation metrics.

This test suite validates ranking metrics (MAP, MRR, NDCG@K, Recall@K, Precision@K)
for evaluating blocking stage performance when similarity scores are available.
"""

import logging

import pytest

from langres.core.metrics import evaluate_blocking_with_ranking
from langres.core.models import CompanySchema, ERCandidate

logger = logging.getLogger(__name__)


def test_map_calculation_known_example() -> None:
    """Test Mean Average Precision with hand-calculable example.

    MAP measures how well true matches are ranked across all queries.
    For each entity, we compute Average Precision (area under precision-recall curve),
    then average across all entities.

    Example:
        Entity e1 has candidates: [(e2, 0.9), (e3, 0.7), (e4, 0.5)]
        Gold pairs: {(e1, e2), (e1, e4)}  # e2 and e4 are true matches

        Precision@1 = 1/1 = 1.0  (e2 is true match)
        Precision@2 = 1/2 = 0.5  (e3 is false match)
        Precision@3 = 2/3 = 0.67 (e4 is true match)

        AP for e1 = (1.0 + 0.67) / 2 = 0.835 (average of precisions at true match ranks)
        MAP = 0.835 (only one query entity in this example)
    """
    # Create candidates with known scores
    left1 = CompanySchema(id="e1", name="Apple Inc")
    right1 = CompanySchema(id="e2", name="Apple Incorporated")
    right2 = CompanySchema(id="e3", name="Microsoft Corp")
    right3 = CompanySchema(id="e4", name="Apple LLC")

    candidates = [
        ERCandidate(left=left1, right=right1, blocker_name="test", similarity_score=0.9),
        ERCandidate(left=left1, right=right2, blocker_name="test", similarity_score=0.7),
        ERCandidate(left=left1, right=right3, blocker_name="test", similarity_score=0.5),
    ]

    # Gold clusters: e1, e2, e4 are same entity; e3 is different
    gold_clusters = [{"e1", "e2", "e4"}, {"e3"}]

    # Calculate metrics
    metrics = evaluate_blocking_with_ranking(candidates, gold_clusters)

    # Verify MAP
    # AP for e1: true matches at ranks 1 and 3
    # Precision@1 = 1/1 = 1.0
    # Precision@3 = 2/3 = 0.667
    # AP = (1.0 + 0.667) / 2 = 0.8335
    expected_map = 0.8335
    assert abs(metrics["map"] - expected_map) < 0.01, (
        f"MAP should be ~{expected_map}, got {metrics['map']}"
    )

    logger.info(f"MAP = {metrics['map']:.4f} (expected ~{expected_map})")


def test_mrr_calculation_known_example() -> None:
    """Test Mean Reciprocal Rank with hand-calculable example.

    MRR measures the average rank of the FIRST true match for each query.
    RR = 1 / rank_of_first_true_match

    Example:
        Entity e1: first true match at rank 1 -> RR = 1/1 = 1.0
        Entity e2: first true match at rank 3 -> RR = 1/3 = 0.333
        MRR = (1.0 + 0.333) / 2 = 0.667
    """
    # Entity e1 has first true match at rank 1
    left1 = CompanySchema(id="e1", name="Apple Inc")
    right1 = CompanySchema(id="e2", name="Apple Incorporated")  # True match
    right2 = CompanySchema(id="e3", name="Microsoft Corp")

    # Entity e4 has first true match at rank 3
    left2 = CompanySchema(id="e4", name="Google LLC")
    right3 = CompanySchema(id="e5", name="Microsoft Inc")
    right4 = CompanySchema(id="e6", name="Oracle Corp")
    right5 = CompanySchema(id="e7", name="Google Incorporated")  # True match

    candidates = [
        # e1's candidates: true match at rank 1
        ERCandidate(left=left1, right=right1, blocker_name="test", similarity_score=0.9),
        ERCandidate(left=left1, right=right2, blocker_name="test", similarity_score=0.7),
        # e4's candidates: true match at rank 3
        ERCandidate(left=left2, right=right3, blocker_name="test", similarity_score=0.8),
        ERCandidate(left=left2, right=right4, blocker_name="test", similarity_score=0.6),
        ERCandidate(left=left2, right=right5, blocker_name="test", similarity_score=0.4),
    ]

    # Gold clusters
    gold_clusters = [
        {"e1", "e2"},  # e1 and e2 are same
        {"e3"},
        {"e4", "e7"},  # e4 and e7 are same
        {"e5"},
        {"e6"},
    ]

    metrics = evaluate_blocking_with_ranking(candidates, gold_clusters)

    # MRR = (1/1 + 1/3) / 2 = (1.0 + 0.333) / 2 = 0.667
    expected_mrr = 0.667
    assert abs(metrics["mrr"] - expected_mrr) < 0.01, (
        f"MRR should be ~{expected_mrr}, got {metrics['mrr']}"
    )

    logger.info(f"MRR = {metrics['mrr']:.4f} (expected ~{expected_mrr})")


def test_ndcg_at_20_calculation() -> None:
    """Test NDCG@20 calculation using ranx library.

    NDCG (Normalized Discounted Cumulative Gain) measures ranking quality
    with position-based discounting. Better ranks get more weight.
    """
    # Create candidates with varying scores
    left = CompanySchema(id="e1", name="Apple Inc")
    candidates = [
        ERCandidate(
            left=left,
            right=CompanySchema(id=f"e{i}", name=f"Company {i}"),
            blocker_name="test",
            similarity_score=1.0 - (i * 0.05),
        )
        for i in range(2, 22)  # 20 candidates
    ]

    # First 5 are true matches, rest are not
    gold_clusters = [{"e1", "e2", "e3", "e4", "e5", "e6"}]

    metrics = evaluate_blocking_with_ranking(candidates, gold_clusters)

    # NDCG@20 should be present and in [0, 1]
    assert "ndcg_at_20" in metrics
    assert 0.0 <= metrics["ndcg_at_20"] <= 1.0

    # With 5 true matches at top ranks, NDCG should be reasonably high
    assert metrics["ndcg_at_20"] > 0.5, (
        f"NDCG@20 should be > 0.5 with top-ranked true matches, got {metrics['ndcg_at_20']}"
    )

    logger.info(f"NDCG@20 = {metrics['ndcg_at_20']:.4f}")


def test_recall_at_20_calculation() -> None:
    """Test Recall@20 calculation.

    Recall@K = (# true matches in top-K) / (# total true matches)

    Example: Entity has 3 true matches total, top-20 contains 2 of them
    -> Recall@20 = 2/3 = 0.667
    """
    # Create entity with 3 true matches
    left = CompanySchema(id="e1", name="Apple Inc")

    # True matches: e2 (rank 1), e5 (rank 10), e25 (rank 24, outside top-20)
    candidates = []
    for i in range(2, 30):
        candidates.append(
            ERCandidate(
                left=left,
                right=CompanySchema(id=f"e{i}", name=f"Company {i}"),
                blocker_name="test",
                similarity_score=1.0 - (i * 0.01),
            )
        )

    # Gold: e1, e2, e5, e25 are same entity
    gold_clusters = [{"e1", "e2", "e5", "e25"}]

    metrics = evaluate_blocking_with_ranking(candidates, gold_clusters, k_values=[20])

    # Top-20 contains e2 (rank 1) and e5 (rank 4), but not e25 (rank 24)
    # Total true matches for e1: 3 (e2, e5, e25)
    # Recall@20 = 2/3 = 0.667
    expected_recall = 2.0 / 3.0
    assert abs(metrics["recall_at_20"] - expected_recall) < 0.01, (
        f"Recall@20 should be ~{expected_recall}, got {metrics['recall_at_20']}"
    )

    logger.info(f"Recall@20 = {metrics['recall_at_20']:.4f} (expected ~{expected_recall})")


def test_precision_at_20_calculation() -> None:
    """Test Precision@20 calculation.

    Precision@K = (# true matches in top-K) / K

    Example: Top-20 contains 5 true matches
    -> Precision@20 = 5/20 = 0.25
    """
    # Create entity with many candidates
    left = CompanySchema(id="e1", name="Apple Inc")

    candidates = []
    for i in range(2, 30):
        candidates.append(
            ERCandidate(
                left=left,
                right=CompanySchema(id=f"e{i}", name=f"Company {i}"),
                blocker_name="test",
                similarity_score=1.0 - (i * 0.01),
            )
        )

    # True matches: e2, e3, e4, e5, e6 (first 5)
    # Top-20 will contain all 5 true matches
    gold_clusters = [{"e1", "e2", "e3", "e4", "e5", "e6"}]

    metrics = evaluate_blocking_with_ranking(candidates, gold_clusters, k_values=[20])

    # Precision@20 = 5/20 = 0.25
    expected_precision = 5.0 / 20.0
    assert abs(metrics["precision_at_20"] - expected_precision) < 0.01, (
        f"Precision@20 should be ~{expected_precision}, got {metrics['precision_at_20']}"
    )

    logger.info(f"Precision@20 = {metrics['precision_at_20']:.4f} (expected ~{expected_precision})")


def test_ranking_metrics_raises_without_scores() -> None:
    """Test that evaluate_blocking_with_ranking raises ValueError when scores are missing.

    Ranking metrics require similarity scores. If any candidate is missing
    similarity_score, the function should raise a clear error.
    """
    left = CompanySchema(id="e1", name="Apple Inc")
    right = CompanySchema(id="e2", name="Apple Incorporated")

    # Create candidate WITHOUT similarity_score
    candidates = [
        ERCandidate(
            left=left,
            right=right,
            blocker_name="test",
            # similarity_score=None (default)
        )
    ]

    gold_clusters = [{"e1", "e2"}]

    with pytest.raises(ValueError) as exc_info:
        evaluate_blocking_with_ranking(candidates, gold_clusters)

    assert "similarity_score" in str(exc_info.value).lower()
    assert "missing" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()

    logger.info(f"Correctly raised ValueError: {exc_info.value}")


def test_ranking_metrics_edge_case_no_matches() -> None:
    """Test ranking metrics when no gold pairs exist.

    Edge case: empty gold clusters or no true matches.
    Metrics should handle gracefully (return 0.0 or N/A).
    """
    left = CompanySchema(id="e1", name="Apple Inc")
    candidates = [
        ERCandidate(
            left=left,
            right=CompanySchema(id="e2", name="Microsoft Corp"),
            blocker_name="test",
            similarity_score=0.9,
        )
    ]

    # No gold pairs (all singletons)
    gold_clusters = [{"e1"}, {"e2"}]

    metrics = evaluate_blocking_with_ranking(candidates, gold_clusters)

    # With no true matches, metrics should be 0 or handle gracefully
    assert metrics["map"] == 0.0 or metrics["map"] is None
    assert metrics["mrr"] == 0.0 or metrics["mrr"] is None

    logger.info("No matches case: metrics handled gracefully")


def test_ranking_metrics_edge_case_all_matches() -> None:
    """Test ranking metrics when all candidates are true matches.

    Edge case: perfect blocking where every candidate is a true match.
    MAP and MRR should be 1.0, Recall@K should be 1.0.
    """
    left = CompanySchema(id="e1", name="Apple Inc")
    candidates = [
        ERCandidate(
            left=left,
            right=CompanySchema(id="e2", name="Apple Incorporated"),
            blocker_name="test",
            similarity_score=0.95,
        ),
        ERCandidate(
            left=left,
            right=CompanySchema(id="e3", name="Apple LLC"),
            blocker_name="test",
            similarity_score=0.85,
        ),
    ]

    # All candidates are true matches
    gold_clusters = [{"e1", "e2", "e3"}]

    metrics = evaluate_blocking_with_ranking(candidates, gold_clusters)

    # Perfect ranking: all metrics should be 1.0
    assert abs(metrics["map"] - 1.0) < 0.01, f"MAP should be 1.0, got {metrics['map']}"
    assert abs(metrics["mrr"] - 1.0) < 0.01, f"MRR should be 1.0, got {metrics['mrr']}"
    assert abs(metrics["recall_at_20"] - 1.0) < 0.01, (
        f"Recall@20 should be 1.0, got {metrics['recall_at_20']}"
    )

    logger.info("All matches case: MAP, MRR, Recall@20 all ~1.0 (perfect)")


def test_ranking_metrics_multiple_k_values() -> None:
    """Test ranking metrics with multiple K values.

    Should compute Recall@K and Precision@K for each K value.
    """
    left = CompanySchema(id="e1", name="Apple Inc")
    candidates = []
    for i in range(2, 52):  # 50 candidates
        candidates.append(
            ERCandidate(
                left=left,
                right=CompanySchema(id=f"e{i}", name=f"Company {i}"),
                blocker_name="test",
                similarity_score=1.0 - (i * 0.01),
            )
        )

    # First 10 are true matches
    gold_entities = {"e1"} | {f"e{i}" for i in range(2, 12)}
    gold_clusters = [gold_entities]

    metrics = evaluate_blocking_with_ranking(candidates, gold_clusters, k_values=[5, 10, 20])

    # Should have metrics for all K values
    assert "recall_at_5" in metrics
    assert "recall_at_10" in metrics
    assert "recall_at_20" in metrics
    assert "precision_at_5" in metrics
    assert "precision_at_10" in metrics
    assert "precision_at_20" in metrics

    # Recall should increase with K (more candidates = more true matches captured)
    assert metrics["recall_at_5"] <= metrics["recall_at_10"]
    assert metrics["recall_at_10"] <= metrics["recall_at_20"]

    # Precision should decrease with K (dilution effect)
    assert metrics["precision_at_20"] <= metrics["precision_at_10"]
    assert metrics["precision_at_10"] <= metrics["precision_at_5"]

    logger.info(
        f"Multiple K: Recall@5={metrics['recall_at_5']:.2f}, "
        f"Recall@10={metrics['recall_at_10']:.2f}, Recall@20={metrics['recall_at_20']:.2f}"
    )
