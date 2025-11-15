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
