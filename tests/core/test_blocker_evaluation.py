"""Tests for Blocker.evaluate() integration with BlockerEvaluationReport."""

import pytest

from langres.core.blocker import Blocker
from langres.core.models import CompanySchema, ERCandidate
from langres.core.reports import (
    BlockerEvaluationReport,
    CandidateInspectionReport,
    CandidateMetrics,
    RankingMetrics,
    RankMetrics,
    RecallCurveStats,
    ScoreMetrics,
)


class SimpleTestBlocker(Blocker[CompanySchema]):
    """Test blocker for evaluation tests."""

    def stream(self, data):
        """Not used in evaluation tests."""
        return iter([])

    def inspect_candidates(
        self,
        candidates: list[ERCandidate[CompanySchema]],
        entities: list[CompanySchema],
        sample_size: int = 10,
    ) -> CandidateInspectionReport:
        """Minimal test implementation."""
        return CandidateInspectionReport(
            total_candidates=len(candidates),
            avg_candidates_per_entity=len(candidates) / len(entities) if entities else 0.0,
            candidate_distribution={},
            examples=[],
            recommendations=[],
        )


@pytest.fixture
def test_blocker():
    """Create test blocker instance."""
    return SimpleTestBlocker()


@pytest.fixture
def sample_candidates():
    """Create sample candidates with scores for testing."""
    candidates = [
        ERCandidate(
            left=CompanySchema(id="c1", name="Acme Corp"),
            right=CompanySchema(id="c2", name="Acme Inc"),
            blocker_name="test_blocker",
            similarity_score=0.95,
        ),
        ERCandidate(
            left=CompanySchema(id="c1", name="Acme Corp"),
            right=CompanySchema(id="c3", name="Beta Corp"),
            blocker_name="test_blocker",
            similarity_score=0.25,
        ),
        ERCandidate(
            left=CompanySchema(id="c2", name="Acme Inc"),
            right=CompanySchema(id="c3", name="Beta Corp"),
            blocker_name="test_blocker",
            similarity_score=0.30,
        ),
    ]
    return candidates


@pytest.fixture
def sample_gold_clusters():
    """Create sample gold clusters for testing."""
    # c1 and c2 are duplicates (Acme), c3 is different (Beta)
    return [{"c1", "c2"}, {"c3"}]


def test_blocker_evaluate_returns_blocker_evaluation_report(
    test_blocker, sample_candidates, sample_gold_clusters
):
    """Test blocker.evaluate() returns BlockerEvaluationReport."""
    report = test_blocker.evaluate(sample_candidates, sample_gold_clusters)

    assert isinstance(report, BlockerEvaluationReport)


def test_blocker_evaluate_has_all_metric_categories(
    test_blocker, sample_candidates, sample_gold_clusters
):
    """Test report has candidates, ranking, scores, ranks, recall_curve."""
    report = test_blocker.evaluate(sample_candidates, sample_gold_clusters)

    assert isinstance(report.candidates, CandidateMetrics)
    assert isinstance(report.ranking, RankingMetrics)
    assert isinstance(report.scores, ScoreMetrics)
    assert isinstance(report.ranks, RankMetrics)
    assert isinstance(report.recall_curve, RecallCurveStats)


def test_blocker_evaluate_metrics_are_accessible(
    test_blocker, sample_candidates, sample_gold_clusters
):
    """Test semantic access pattern works."""
    report = test_blocker.evaluate(sample_candidates, sample_gold_clusters)

    # Candidate metrics
    assert isinstance(report.candidates.recall, float)
    assert 0.0 <= report.candidates.recall <= 1.0
    assert isinstance(report.candidates.precision, float)
    assert 0.0 <= report.candidates.precision <= 1.0

    # Ranking metrics
    assert isinstance(report.ranking.map, float)
    assert 0.0 <= report.ranking.map <= 1.0
    assert isinstance(report.ranking.mrr, float)
    assert 0.0 <= report.ranking.mrr <= 1.0

    # Score metrics
    assert isinstance(report.scores.separation, float)
    assert isinstance(report.scores.true_median, float)
    assert isinstance(report.scores.false_median, float)

    # Rank metrics
    assert isinstance(report.ranks.median, float)
    assert report.ranks.median >= 1.0
    assert isinstance(report.ranks.percentile_95, float)
    assert report.ranks.percentile_95 >= 1.0

    # Recall curve
    assert isinstance(report.recall_curve.k_values, list)
    assert isinstance(report.recall_curve.recall_values, list)
    assert len(report.recall_curve.k_values) == len(report.recall_curve.recall_values)


def test_blocker_evaluate_custom_k_values(test_blocker, sample_candidates, sample_gold_clusters):
    """Test k_values parameter works."""
    custom_k_values = [1, 10, 50]
    report = test_blocker.evaluate(
        sample_candidates, sample_gold_clusters, k_values=custom_k_values
    )

    assert report.recall_curve.k_values == custom_k_values
    assert len(report.recall_curve.recall_values) == len(custom_k_values)


def test_blocker_evaluate_default_k_values(test_blocker, sample_candidates, sample_gold_clusters):
    """Test default k_values are [1, 5, 10, 20, 50]."""
    report = test_blocker.evaluate(sample_candidates, sample_gold_clusters)

    # Default from evaluate_blocker_detailed
    assert report.recall_curve.k_values == [1, 5, 10, 20, 50]


def test_blocker_evaluate_empty_candidates(test_blocker, sample_gold_clusters):
    """Test evaluate handles empty candidate list."""
    report = test_blocker.evaluate([], sample_gold_clusters)

    assert isinstance(report, BlockerEvaluationReport)
    assert report.candidates.recall == 0.0
    assert report.candidates.total == 0


def test_blocker_evaluate_computes_correct_recall(
    test_blocker, sample_candidates, sample_gold_clusters
):
    """Test recall computation is correct."""
    report = test_blocker.evaluate(sample_candidates, sample_gold_clusters)

    # We have 1 true match (c1, c2) in candidates, and 1 true match in gold
    # So recall should be 1.0 (100%)
    assert report.candidates.recall == 1.0


def test_blocker_evaluate_computes_correct_precision(
    test_blocker, sample_candidates, sample_gold_clusters
):
    """Test precision computation is correct."""
    report = test_blocker.evaluate(sample_candidates, sample_gold_clusters)

    # We have 3 candidates total, 1 is a true match
    # Precision = 1/3 = 0.333...
    assert pytest.approx(report.candidates.precision, rel=0.01) == 1.0 / 3.0


def test_blocker_evaluate_score_separation(test_blocker, sample_candidates, sample_gold_clusters):
    """Test score separation is computed."""
    report = test_blocker.evaluate(sample_candidates, sample_gold_clusters)

    # True match (c1, c2) has score 0.95
    # False matches have scores 0.25, 0.30
    # Separation should be positive (true_median > false_median)
    assert report.scores.separation > 0.0
    assert report.scores.true_median > report.scores.false_median


def test_blocker_evaluate_rank_metrics(test_blocker, sample_candidates, sample_gold_clusters):
    """Test rank metrics are computed."""
    report = test_blocker.evaluate(sample_candidates, sample_gold_clusters)

    # Ranks should be >= 1
    assert report.ranks.median >= 1.0
    assert report.ranks.percentile_95 >= 1.0
    assert 0.0 <= report.ranks.percent_in_top_5 <= 100.0
    assert 0.0 <= report.ranks.percent_in_top_10 <= 100.0
    assert 0.0 <= report.ranks.percent_in_top_20 <= 100.0


def test_blocker_evaluate_recall_curve_values(
    test_blocker, sample_candidates, sample_gold_clusters
):
    """Test recall curve has valid values."""
    report = test_blocker.evaluate(sample_candidates, sample_gold_clusters)

    # All recall values should be in [0.0, 1.0]
    for recall in report.recall_curve.recall_values:
        assert 0.0 <= recall <= 1.0

    # Avg pairs should be >= 0
    for avg_pairs in report.recall_curve.avg_pairs_values:
        assert avg_pairs >= 0.0


def test_blocker_evaluate_report_is_frozen(test_blocker, sample_candidates, sample_gold_clusters):
    """Test that BlockerEvaluationReport is immutable (frozen)."""
    report = test_blocker.evaluate(sample_candidates, sample_gold_clusters)

    # Try to modify - should raise ValidationError
    with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen models
        report.candidates = CandidateMetrics(
            recall=0.5,
            precision=0.5,
            total=10,
            avg_per_entity=5.0,
            missed_matches=5,
            false_positives=5,
        )


def test_blocker_evaluate_with_no_true_matches(test_blocker):
    """Test evaluation when no true matches exist in candidates."""
    # Create candidates with no overlapping pairs with gold clusters
    candidates = [
        ERCandidate(
            left=CompanySchema(id="c1", name="A"),
            right=CompanySchema(id="c2", name="B"),
            blocker_name="test",
            similarity_score=0.8,
        )
    ]
    gold_clusters = [{"c3", "c4"}]  # Different entities

    report = test_blocker.evaluate(candidates, gold_clusters)

    # Recall should be 0 (no true matches found)
    assert report.candidates.recall == 0.0
    # Precision should be 0 (no true matches in candidates)
    assert report.candidates.precision == 0.0


def test_blocker_evaluate_with_all_true_matches(test_blocker):
    """Test evaluation when all candidates are true matches."""
    candidates = [
        ERCandidate(
            left=CompanySchema(id="c1", name="A"),
            right=CompanySchema(id="c2", name="A"),
            blocker_name="test",
            similarity_score=0.9,
        )
    ]
    gold_clusters = [{"c1", "c2"}]

    report = test_blocker.evaluate(candidates, gold_clusters)

    # Both recall and precision should be 1.0
    assert report.candidates.recall == 1.0
    assert report.candidates.precision == 1.0


@pytest.mark.integration
def test_blocker_evaluation_with_messy_production_data():
    """Test evaluation handles messy real-world data gracefully.

    This test ensures robustness against common data quality issues:
    - Entities with no candidates (blocker missed them completely)
    - Singleton clusters (entities with no duplicates)
    - Very high candidate counts per entity (stress test)
    - Wide score distribution including edge values (0.0, 1.0)
    - Large clusters and small clusters mixed
    """
    from langres.core.analysis import evaluate_blocker_detailed

    # Create messy dataset with 100 entities
    entities = []
    for i in range(100):
        entities.append(CompanySchema(id=str(i), name=f"Company {i}"))

    candidates = []

    # Entity 0: Tons of candidates (stress test - 49 candidates)
    for i in range(1, 50):
        candidates.append(
            ERCandidate(
                left=entities[0],
                right=entities[i],
                similarity_score=0.5 + 0.01 * i,  # Wide range 0.51 to 0.99
                blocker_name="messy_test",
            )
        )

    # Entity 1-10: Normal candidates (true matches between consecutive entities)
    for i in range(1, 10):
        for j in range(i + 1, min(i + 5, 10)):
            candidates.append(
                ERCandidate(
                    left=entities[i],
                    right=entities[j],
                    similarity_score=0.6 if i == j - 1 else 0.3,
                    blocker_name="messy_test",
                )
            )

    # Entity 15: No candidates at all (missed by blocker)
    # Just don't add any candidates for entity 15

    # Entity 20: Edge case scores (0.0 and 1.0)
    candidates.extend(
        [
            ERCandidate(
                left=entities[20],
                right=entities[21],
                similarity_score=0.0,
                blocker_name="messy_test",
            ),  # Min score
            ERCandidate(
                left=entities[20],
                right=entities[22],
                similarity_score=1.0,
                blocker_name="messy_test",
            ),  # Max score
        ]
    )

    # Gold clusters with various patterns
    gold_clusters = [
        {str(i) for i in range(1, 10)},  # Large cluster (9 entities)
        {str(i) for i in range(50, 55)},  # Medium cluster with no candidates
        {"20", "22"},  # Small cluster with edge scores
        {"99"},  # Singleton (no duplicates - should not affect metrics)
        {"15"},  # Entity with no candidates
    ]

    # Should not crash despite messy data
    report = evaluate_blocker_detailed(candidates, gold_clusters)

    # Basic sanity checks
    assert 0.0 <= report.candidates.recall <= 1.0
    assert 0.0 <= report.candidates.precision <= 1.0
    assert report.candidates.total == len(candidates)
    assert report.ranks.median >= 1.0

    # Should handle entities with no candidates
    assert report.candidates.missed_matches > 0  # Entity 15 + cluster 50-55 have no candidates

    # Should handle wide score distributions
    assert len(report.scores.histogram["true"]) > 0
    assert len(report.scores.histogram["false"]) > 0

    # Should handle extreme candidate counts
    # Entity 0 has 49 candidates, which is way more than average
    assert report.candidates.total > 50


@pytest.mark.integration
def test_blocker_evaluation_with_all_singletons():
    """Test evaluation when all clusters are singletons (no duplicates).

    This is a degenerate but valid case where no entities are duplicates.
    All candidates should be false positives.
    """
    from langres.core.analysis import evaluate_blocker_detailed

    entities = [CompanySchema(id=str(i), name=f"Company {i}") for i in range(10)]

    # Generate some candidates, but none are true matches
    candidates = [
        ERCandidate(left=entities[i], right=entities[j], similarity_score=0.4, blocker_name="test")
        for i in range(5)
        for j in range(i + 1, i + 3)
        if j < 10
    ]

    # All clusters are singletons (no duplicates exist)
    gold_clusters = [{str(i)} for i in range(10)]

    # Should handle gracefully
    report = evaluate_blocker_detailed(candidates, gold_clusters)

    # No true matches exist - all candidates are false positives
    true_positives = report.candidates.total - report.candidates.false_positives
    assert true_positives == 0
    # When no true matches exist in gold data (tp + fn = 0), recall is 0.0 (undefined/N/A)
    # This is the correct behavior: you can't have perfect recall if no matches exist
    assert report.candidates.recall == 0.0
    assert report.candidates.precision == 0.0  # All candidates are false positives
    assert report.candidates.total == len(candidates)
    assert report.candidates.missed_matches == 0  # No matches to miss

    # Ranking metrics should handle no true matches
    assert report.ranking.map == 0.0
    assert report.ranking.mrr == 0.0


@pytest.mark.integration
def test_blocker_evaluation_with_perfect_blocking():
    """Test evaluation when blocker achieves perfect recall and precision.

    This is the ideal case: blocker finds exactly all true matches with
    perfect scores, and no false positives.
    """
    from langres.core.analysis import evaluate_blocker_detailed

    entities = [CompanySchema(id=str(i), name=f"Company {i}") for i in range(10)]

    # Perfect candidates: only true matches, all with score 1.0, ranked at position 1
    candidates = [
        ERCandidate(
            left=entities[0], right=entities[1], similarity_score=1.0, blocker_name="perfect"
        ),
        ERCandidate(
            left=entities[2], right=entities[3], similarity_score=1.0, blocker_name="perfect"
        ),
    ]

    gold_clusters = [
        {"0", "1"},  # Duplicates
        {"2", "3"},  # Duplicates
        {"4"},  # Singleton
        {"5"},  # Singleton
    ]

    report = evaluate_blocker_detailed(candidates, gold_clusters)

    # Should get perfect scores across the board
    assert report.candidates.recall == 1.0
    assert report.candidates.precision == 1.0
    assert report.ranking.map == 1.0
    assert report.ranking.mrr == 1.0
    assert report.ranks.median == 1.0
    assert report.ranks.percent_in_top_5 == 100.0

    # All true scores should be 1.0
    assert report.scores.true_mean == 1.0
    assert report.scores.true_median == 1.0

    # Should have maximal separation (no false candidates)
    assert report.scores.separation > 0.9  # true median (1.0) - false median (undefined or 0)
