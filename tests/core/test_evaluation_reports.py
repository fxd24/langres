"""Tests for blocker evaluation report Pydantic models.

This module tests the new comprehensive evaluation report models:
- CandidateMetrics
- RankingMetrics
- ScoreMetrics
- RankMetrics
- RecallCurveStats
- BlockerEvaluationReport

These tests verify Pydantic validation, immutability, serialization,
and the plotting delegation interface.
"""

import pytest
from pydantic import ValidationError

from langres.core.reports import (
    BlockerEvaluationReport,
    CandidateMetrics,
    RankingMetrics,
    RankMetrics,
    RecallCurveStats,
    ScoreMetrics,
)


class TestCandidateMetrics:
    """Test CandidateMetrics Pydantic model."""

    def test_candidate_metrics_valid_data(self):
        """Test CandidateMetrics with valid data."""
        metrics = CandidateMetrics(
            recall=0.95,
            precision=0.80,
            total=1000,
            avg_per_entity=10.5,
            missed_matches=50,
            false_positives=200,
        )

        assert metrics.recall == 0.95
        assert metrics.precision == 0.80
        assert metrics.total == 1000
        assert metrics.avg_per_entity == 10.5
        assert metrics.missed_matches == 50
        assert metrics.false_positives == 200

    def test_candidate_metrics_recall_bounds(self):
        """Test CandidateMetrics validates recall in [0, 1]."""
        # Valid: recall = 0.0
        metrics = CandidateMetrics(
            recall=0.0,
            precision=0.5,
            total=100,
            avg_per_entity=5.0,
            missed_matches=0,
            false_positives=0,
        )
        assert metrics.recall == 0.0

        # Valid: recall = 1.0
        metrics = CandidateMetrics(
            recall=1.0,
            precision=0.5,
            total=100,
            avg_per_entity=5.0,
            missed_matches=0,
            false_positives=0,
        )
        assert metrics.recall == 1.0

        # Invalid: recall > 1.0
        with pytest.raises(ValidationError) as exc_info:
            CandidateMetrics(
                recall=1.5,
                precision=0.5,
                total=100,
                avg_per_entity=5.0,
                missed_matches=0,
                false_positives=0,
            )
        assert "recall" in str(exc_info.value).lower()

        # Invalid: recall < 0.0
        with pytest.raises(ValidationError) as exc_info:
            CandidateMetrics(
                recall=-0.1,
                precision=0.5,
                total=100,
                avg_per_entity=5.0,
                missed_matches=0,
                false_positives=0,
            )
        assert "recall" in str(exc_info.value).lower()

    def test_candidate_metrics_precision_bounds(self):
        """Test CandidateMetrics validates precision in [0, 1]."""
        # Invalid: precision > 1.0
        with pytest.raises(ValidationError) as exc_info:
            CandidateMetrics(
                recall=0.9,
                precision=1.2,
                total=100,
                avg_per_entity=5.0,
                missed_matches=0,
                false_positives=0,
            )
        assert "precision" in str(exc_info.value).lower()

        # Invalid: precision < 0.0
        with pytest.raises(ValidationError) as exc_info:
            CandidateMetrics(
                recall=0.9,
                precision=-0.1,
                total=100,
                avg_per_entity=5.0,
                missed_matches=0,
                false_positives=0,
            )
        assert "precision" in str(exc_info.value).lower()

    def test_candidate_metrics_non_negative_counts(self):
        """Test CandidateMetrics validates counts >= 0."""
        # Invalid: total < 0
        with pytest.raises(ValidationError):
            CandidateMetrics(
                recall=0.9,
                precision=0.8,
                total=-1,
                avg_per_entity=5.0,
                missed_matches=0,
                false_positives=0,
            )

        # Invalid: missed_matches < 0
        with pytest.raises(ValidationError):
            CandidateMetrics(
                recall=0.9,
                precision=0.8,
                total=100,
                avg_per_entity=5.0,
                missed_matches=-5,
                false_positives=0,
            )

        # Invalid: false_positives < 0
        with pytest.raises(ValidationError):
            CandidateMetrics(
                recall=0.9,
                precision=0.8,
                total=100,
                avg_per_entity=5.0,
                missed_matches=0,
                false_positives=-10,
            )

    def test_candidate_metrics_immutable(self):
        """Test CandidateMetrics is frozen (immutable)."""
        metrics = CandidateMetrics(
            recall=0.95,
            precision=0.80,
            total=1000,
            avg_per_entity=10.5,
            missed_matches=50,
            false_positives=200,
        )

        with pytest.raises(ValidationError) as exc_info:
            metrics.recall = 0.99  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower()

    def test_candidate_metrics_serialization(self):
        """Test CandidateMetrics serialization."""
        metrics = CandidateMetrics(
            recall=0.95,
            precision=0.80,
            total=1000,
            avg_per_entity=10.5,
            missed_matches=50,
            false_positives=200,
        )

        # Test model_dump()
        data = metrics.model_dump()
        assert data == {
            "recall": 0.95,
            "precision": 0.80,
            "total": 1000,
            "avg_per_entity": 10.5,
            "missed_matches": 50,
            "false_positives": 200,
        }

        # Test round-trip
        restored = CandidateMetrics(**data)
        assert restored == metrics


class TestRankingMetrics:
    """Test RankingMetrics Pydantic model."""

    def test_ranking_metrics_valid_data(self):
        """Test RankingMetrics with valid data."""
        metrics = RankingMetrics(
            map=0.85,
            mrr=0.90,
            ndcg_at_10=0.88,
            ndcg_at_20=0.89,
            recall_at_5=0.75,
            recall_at_10=0.85,
            recall_at_20=0.92,
        )

        assert metrics.map == 0.85
        assert metrics.mrr == 0.90
        assert metrics.ndcg_at_10 == 0.88
        assert metrics.ndcg_at_20 == 0.89
        assert metrics.recall_at_5 == 0.75
        assert metrics.recall_at_10 == 0.85
        assert metrics.recall_at_20 == 0.92

    def test_ranking_metrics_bounds(self):
        """Test RankingMetrics validates all metrics in [0, 1]."""
        # Invalid: map > 1.0
        with pytest.raises(ValidationError):
            RankingMetrics(
                map=1.5,
                mrr=0.90,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            )

        # Invalid: mrr < 0.0
        with pytest.raises(ValidationError):
            RankingMetrics(
                map=0.85,
                mrr=-0.1,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            )

    def test_ranking_metrics_immutable(self):
        """Test RankingMetrics is frozen."""
        metrics = RankingMetrics(
            map=0.85,
            mrr=0.90,
            ndcg_at_10=0.88,
            ndcg_at_20=0.89,
            recall_at_5=0.75,
            recall_at_10=0.85,
            recall_at_20=0.92,
        )

        with pytest.raises(ValidationError) as exc_info:
            metrics.map = 0.99  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower()


class TestScoreMetrics:
    """Test ScoreMetrics Pydantic model."""

    def test_score_metrics_valid_data(self):
        """Test ScoreMetrics with valid data."""
        metrics = ScoreMetrics(
            separation=0.45,
            true_median=0.85,
            true_mean=0.82,
            true_std=0.12,
            false_median=0.40,
            false_mean=0.38,
            false_std=0.15,
            overlap_fraction=0.20,
            histogram={
                "true": {0.5: 10, 0.7: 20, 0.9: 30},
                "false": {0.2: 15, 0.4: 25, 0.6: 10},
            },
        )

        assert metrics.separation == 0.45
        assert metrics.true_median == 0.85
        assert metrics.overlap_fraction == 0.20
        assert "true" in metrics.histogram
        assert "false" in metrics.histogram

    def test_score_metrics_overlap_bounds(self):
        """Test ScoreMetrics validates overlap_fraction in [0, 1]."""
        # Invalid: overlap_fraction > 1.0
        with pytest.raises(ValidationError):
            ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=1.5,
                histogram={"true": {}, "false": {}},
            )

        # Invalid: overlap_fraction < 0.0
        with pytest.raises(ValidationError):
            ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=-0.1,
                histogram={"true": {}, "false": {}},
            )

    def test_score_metrics_negative_separation_allowed(self):
        """Test ScoreMetrics allows negative separation (poor blocker case)."""
        metrics = ScoreMetrics(
            separation=-0.10,  # False scores higher than true (bad!)
            true_median=0.40,
            true_mean=0.42,
            true_std=0.15,
            false_median=0.50,
            false_mean=0.52,
            false_std=0.12,
            overlap_fraction=0.80,
            histogram={"true": {}, "false": {}},
        )

        assert metrics.separation == -0.10

    def test_score_metrics_immutable(self):
        """Test ScoreMetrics is frozen."""
        metrics = ScoreMetrics(
            separation=0.45,
            true_median=0.85,
            true_mean=0.82,
            true_std=0.12,
            false_median=0.40,
            false_mean=0.38,
            false_std=0.15,
            overlap_fraction=0.20,
            histogram={"true": {}, "false": {}},
        )

        with pytest.raises(ValidationError) as exc_info:
            metrics.separation = 0.50  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower()


class TestRankMetrics:
    """Test RankMetrics Pydantic model."""

    def test_rank_metrics_valid_data(self):
        """Test RankMetrics with valid data."""
        metrics = RankMetrics(
            median=5.0,
            percentile_95=18.0,
            percent_in_top_5=60.0,
            percent_in_top_10=80.0,
            percent_in_top_20=95.0,
            rank_counts={1: 10, 2: 15, 3: 8, 5: 12, 10: 5},
        )

        assert metrics.median == 5.0
        assert metrics.percentile_95 == 18.0
        assert metrics.percent_in_top_5 == 60.0
        assert metrics.rank_counts[1] == 10

    def test_rank_metrics_minimum_rank(self):
        """Test RankMetrics validates rank values >= 1.0."""
        # Invalid: median < 1.0
        with pytest.raises(ValidationError):
            RankMetrics(
                median=0.5,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            )

        # Invalid: percentile_95 < 1.0
        with pytest.raises(ValidationError):
            RankMetrics(
                median=5.0,
                percentile_95=0.8,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            )

    def test_rank_metrics_percent_bounds(self):
        """Test RankMetrics validates percentages in [0, 100]."""
        # Invalid: percent_in_top_5 > 100
        with pytest.raises(ValidationError):
            RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=150.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            )

        # Invalid: percent_in_top_10 < 0
        with pytest.raises(ValidationError):
            RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=-5.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            )

    def test_rank_metrics_immutable(self):
        """Test RankMetrics is frozen."""
        metrics = RankMetrics(
            median=5.0,
            percentile_95=18.0,
            percent_in_top_5=60.0,
            percent_in_top_10=80.0,
            percent_in_top_20=95.0,
            rank_counts={1: 10},
        )

        with pytest.raises(ValidationError) as exc_info:
            metrics.median = 10.0  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower()


class TestRecallCurveStats:
    """Test RecallCurveStats Pydantic model."""

    def test_recall_curve_stats_valid_data(self):
        """Test RecallCurveStats with valid data."""
        stats = RecallCurveStats(
            k_values=[1, 5, 10, 20, 50],
            recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
            avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
        )

        assert stats.k_values == [1, 5, 10, 20, 50]
        assert stats.recall_values == [0.10, 0.60, 0.85, 0.95, 0.99]
        assert stats.avg_pairs_values == [1.0, 5.0, 10.0, 20.0, 50.0]

    def test_recall_curve_stats_optimal_k(self):
        """Test RecallCurveStats.optimal_k() finds smallest k for target recall."""
        stats = RecallCurveStats(
            k_values=[1, 5, 10, 20, 50],
            recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
            avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
        )

        # Find k for 95% recall
        optimal = stats.optimal_k(target_recall=0.95)
        assert optimal == 20  # First k where recall >= 0.95

        # Find k for 85% recall
        optimal = stats.optimal_k(target_recall=0.85)
        assert optimal == 10

        # Find k for 60% recall
        optimal = stats.optimal_k(target_recall=0.60)
        assert optimal == 5

    def test_recall_curve_stats_optimal_k_unreachable(self):
        """Test RecallCurveStats.optimal_k() when target unreachable."""
        stats = RecallCurveStats(
            k_values=[1, 5, 10, 20, 50],
            recall_values=[0.10, 0.60, 0.85, 0.90, 0.92],
            avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
        )

        # Target 95% recall not reached - should return largest k
        optimal = stats.optimal_k(target_recall=0.95)
        assert optimal == 50

    def test_recall_curve_stats_optimal_k_default_target(self):
        """Test RecallCurveStats.optimal_k() default target is 0.95."""
        stats = RecallCurveStats(
            k_values=[1, 5, 10, 20, 50],
            recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
            avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
        )

        optimal = stats.optimal_k()  # No argument
        assert optimal == 20  # First k where recall >= 0.95 (default)

    def test_recall_curve_stats_immutable(self):
        """Test RecallCurveStats is frozen."""
        stats = RecallCurveStats(
            k_values=[1, 5, 10],
            recall_values=[0.10, 0.60, 0.85],
            avg_pairs_values=[1.0, 5.0, 10.0],
        )

        with pytest.raises(ValidationError) as exc_info:
            stats.k_values = [1, 2, 3]  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower()


class TestBlockerEvaluationReport:
    """Test BlockerEvaluationReport Pydantic model."""

    def test_blocker_evaluation_report_valid_data(self):
        """Test BlockerEvaluationReport with valid data."""
        report = BlockerEvaluationReport(
            candidates=CandidateMetrics(
                recall=0.95,
                precision=0.80,
                total=1000,
                avg_per_entity=10.5,
                missed_matches=50,
                false_positives=200,
            ),
            ranking=RankingMetrics(
                map=0.85,
                mrr=0.90,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            ),
            scores=ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=0.20,
                histogram={"true": {}, "false": {}},
            ),
            ranks=RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            ),
            recall_curve=RecallCurveStats(
                k_values=[1, 5, 10, 20, 50],
                recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
                avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
            ),
        )

        # Test progressive disclosure
        assert report.candidates.recall == 0.95
        assert report.ranking.map == 0.85
        assert report.scores.separation == 0.45
        assert report.ranks.median == 5.0
        assert report.recall_curve.k_values == [1, 5, 10, 20, 50]

    def test_blocker_evaluation_report_semantic_access(self):
        """Test BlockerEvaluationReport semantic category access."""
        report = BlockerEvaluationReport(
            candidates=CandidateMetrics(
                recall=0.95,
                precision=0.80,
                total=1000,
                avg_per_entity=10.5,
                missed_matches=50,
                false_positives=200,
            ),
            ranking=RankingMetrics(
                map=0.85,
                mrr=0.90,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            ),
            scores=ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=0.20,
                histogram={"true": {}, "false": {}},
            ),
            ranks=RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            ),
            recall_curve=RecallCurveStats(
                k_values=[1, 5, 10, 20, 50],
                recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
                avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
            ),
        )

        # Test that categories are properly typed
        assert isinstance(report.candidates, CandidateMetrics)
        assert isinstance(report.ranking, RankingMetrics)
        assert isinstance(report.scores, ScoreMetrics)
        assert isinstance(report.ranks, RankMetrics)
        assert isinstance(report.recall_curve, RecallCurveStats)

    def test_blocker_evaluation_report_immutable(self):
        """Test BlockerEvaluationReport is frozen."""
        report = BlockerEvaluationReport(
            candidates=CandidateMetrics(
                recall=0.95,
                precision=0.80,
                total=1000,
                avg_per_entity=10.5,
                missed_matches=50,
                false_positives=200,
            ),
            ranking=RankingMetrics(
                map=0.85,
                mrr=0.90,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            ),
            scores=ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=0.20,
                histogram={"true": {}, "false": {}},
            ),
            ranks=RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            ),
            recall_curve=RecallCurveStats(
                k_values=[1, 5, 10, 20, 50],
                recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
                avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
            ),
        )

        with pytest.raises(ValidationError) as exc_info:
            report.candidates = CandidateMetrics(  # type: ignore[misc]
                recall=0.99,
                precision=0.90,
                total=500,
                avg_per_entity=5.0,
                missed_matches=10,
                false_positives=50,
            )
        assert "frozen" in str(exc_info.value).lower()

    def test_blocker_evaluation_report_serialization(self):
        """Test BlockerEvaluationReport serialization."""
        report = BlockerEvaluationReport(
            candidates=CandidateMetrics(
                recall=0.95,
                precision=0.80,
                total=1000,
                avg_per_entity=10.5,
                missed_matches=50,
                false_positives=200,
            ),
            ranking=RankingMetrics(
                map=0.85,
                mrr=0.90,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            ),
            scores=ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=0.20,
                histogram={"true": {0.8: 10}, "false": {0.4: 20}},
            ),
            ranks=RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10, 2: 8},
            ),
            recall_curve=RecallCurveStats(
                k_values=[1, 5, 10],
                recall_values=[0.10, 0.60, 0.85],
                avg_pairs_values=[1.0, 5.0, 10.0],
            ),
        )

        # Test model_dump()
        data = report.model_dump()
        assert "candidates" in data
        assert "ranking" in data
        assert "scores" in data
        assert "ranks" in data
        assert "recall_curve" in data

        # Test round-trip
        restored = BlockerEvaluationReport(**data)
        assert restored.candidates.recall == report.candidates.recall
        assert restored.ranking.map == report.ranking.map

    def test_blocker_evaluation_report_to_markdown(self):
        """Test BlockerEvaluationReport.to_markdown() exists and returns string."""
        report = BlockerEvaluationReport(
            candidates=CandidateMetrics(
                recall=0.95,
                precision=0.80,
                total=1000,
                avg_per_entity=10.5,
                missed_matches=50,
                false_positives=200,
            ),
            ranking=RankingMetrics(
                map=0.85,
                mrr=0.90,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            ),
            scores=ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=0.20,
                histogram={"true": {}, "false": {}},
            ),
            ranks=RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            ),
            recall_curve=RecallCurveStats(
                k_values=[1, 5, 10, 20, 50],
                recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
                avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
            ),
        )

        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert len(markdown) > 0
        # Should contain metric categories
        assert "Candidate" in markdown or "candidate" in markdown
        assert "Ranking" in markdown or "ranking" in markdown


class TestBlockerEvaluationReportPlotting:
    """Test BlockerEvaluationReport plotting methods (delegation)."""

    def test_plot_score_distribution_delegates_to_plotting_module(self):
        """Test plot_score_distribution() delegates to langres.plotting.blockers."""
        report = BlockerEvaluationReport(
            candidates=CandidateMetrics(
                recall=0.95,
                precision=0.80,
                total=1000,
                avg_per_entity=10.5,
                missed_matches=50,
                false_positives=200,
            ),
            ranking=RankingMetrics(
                map=0.85,
                mrr=0.90,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            ),
            scores=ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=0.20,
                histogram={"true": {}, "false": {}},
            ),
            ranks=RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            ),
            recall_curve=RecallCurveStats(
                k_values=[1, 5, 10, 20, 50],
                recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
                avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
            ),
        )

        # Should raise ImportError when matplotlib not available
        # (We test the delegation interface, not actual plotting)
        with pytest.raises(ImportError) as exc_info:
            report.plot_score_distribution()
        assert "matplotlib" in str(exc_info.value).lower()
        assert "langres[viz]" in str(exc_info.value) or "pip install" in str(exc_info.value)

    def test_plot_rank_distribution_delegates(self):
        """Test plot_rank_distribution() delegates to plotting module."""
        report = BlockerEvaluationReport(
            candidates=CandidateMetrics(
                recall=0.95,
                precision=0.80,
                total=1000,
                avg_per_entity=10.5,
                missed_matches=50,
                false_positives=200,
            ),
            ranking=RankingMetrics(
                map=0.85,
                mrr=0.90,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            ),
            scores=ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=0.20,
                histogram={"true": {}, "false": {}},
            ),
            ranks=RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            ),
            recall_curve=RecallCurveStats(
                k_values=[1, 5, 10, 20, 50],
                recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
                avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
            ),
        )

        with pytest.raises(ImportError) as exc_info:
            report.plot_rank_distribution()
        assert "matplotlib" in str(exc_info.value).lower()

    def test_plot_recall_curve_delegates(self):
        """Test plot_recall_curve() delegates to plotting module."""
        report = BlockerEvaluationReport(
            candidates=CandidateMetrics(
                recall=0.95,
                precision=0.80,
                total=1000,
                avg_per_entity=10.5,
                missed_matches=50,
                false_positives=200,
            ),
            ranking=RankingMetrics(
                map=0.85,
                mrr=0.90,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            ),
            scores=ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=0.20,
                histogram={"true": {}, "false": {}},
            ),
            ranks=RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            ),
            recall_curve=RecallCurveStats(
                k_values=[1, 5, 10, 20, 50],
                recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
                avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
            ),
        )

        with pytest.raises(ImportError) as exc_info:
            report.plot_recall_curve()
        assert "matplotlib" in str(exc_info.value).lower()

    def test_plot_all_delegates(self):
        """Test plot_all() delegates to plotting module."""
        report = BlockerEvaluationReport(
            candidates=CandidateMetrics(
                recall=0.95,
                precision=0.80,
                total=1000,
                avg_per_entity=10.5,
                missed_matches=50,
                false_positives=200,
            ),
            ranking=RankingMetrics(
                map=0.85,
                mrr=0.90,
                ndcg_at_10=0.88,
                ndcg_at_20=0.89,
                recall_at_5=0.75,
                recall_at_10=0.85,
                recall_at_20=0.92,
            ),
            scores=ScoreMetrics(
                separation=0.45,
                true_median=0.85,
                true_mean=0.82,
                true_std=0.12,
                false_median=0.40,
                false_mean=0.38,
                false_std=0.15,
                overlap_fraction=0.20,
                histogram={"true": {}, "false": {}},
            ),
            ranks=RankMetrics(
                median=5.0,
                percentile_95=18.0,
                percent_in_top_5=60.0,
                percent_in_top_10=80.0,
                percent_in_top_20=95.0,
                rank_counts={1: 10},
            ),
            recall_curve=RecallCurveStats(
                k_values=[1, 5, 10, 20, 50],
                recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
                avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
            ),
        )

        with pytest.raises(ImportError) as exc_info:
            report.plot_all()
        assert "matplotlib" in str(exc_info.value).lower()
