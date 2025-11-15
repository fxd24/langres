"""Tests for inspection report models.

This module tests the three report models used for component inspection:
- CandidateInspectionReport: for blocker output inspection
- ScoreInspectionReport: for module output inspection
- ClusterInspectionReport: for clusterer output inspection
"""

import pytest

from langres.core.reports import (
    CandidateInspectionReport,
    ClusterInspectionReport,
    ScoreInspectionReport,
)


class TestCandidateInspectionReport:
    """Tests for CandidateInspectionReport model."""

    def test_valid_report_creation(self) -> None:
        """Test creating a valid CandidateInspectionReport."""
        report = CandidateInspectionReport(
            total_candidates=100,
            avg_candidates_per_entity=5.0,
            candidate_distribution={"1-3": 60, "4-6": 30, "7+": 10},
            examples=[
                {
                    "left_id": "c1",
                    "right_id": "c2",
                    "left_text": "Acme Corp",
                    "right_text": "ACME Corporation",
                }
            ],
            recommendations=["Increase k_neighbors for better recall"],
        )

        assert report.total_candidates == 100
        assert report.avg_candidates_per_entity == 5.0
        assert report.candidate_distribution == {"1-3": 60, "4-6": 30, "7+": 10}
        assert len(report.examples) == 1
        assert len(report.recommendations) == 1

    def test_empty_report_creation(self) -> None:
        """Test creating a report with zero candidates."""
        report = CandidateInspectionReport(
            total_candidates=0,
            avg_candidates_per_entity=0.0,
            candidate_distribution={},
            examples=[],
            recommendations=["No candidates generated - check data and parameters"],
        )

        assert report.total_candidates == 0
        assert report.avg_candidates_per_entity == 0.0
        assert report.candidate_distribution == {}
        assert report.examples == []
        assert len(report.recommendations) == 1

    def test_stats_property(self) -> None:
        """Test stats property returns numerical metrics only."""
        report = CandidateInspectionReport(
            total_candidates=100,
            avg_candidates_per_entity=5.0,
            candidate_distribution={"1-3": 60, "4-6": 30, "7+": 10},
            examples=[{"left_id": "c1", "right_id": "c2"}],
            recommendations=["Test recommendation"],
        )

        stats = report.stats
        assert stats["total_candidates"] == 100
        assert stats["avg_candidates_per_entity"] == 5.0
        assert "examples" not in stats
        assert "recommendations" not in stats

    def test_to_dict(self) -> None:
        """Test to_dict returns JSON-serializable dictionary."""
        report = CandidateInspectionReport(
            total_candidates=100,
            avg_candidates_per_entity=5.0,
            candidate_distribution={"1-3": 60, "4-6": 30, "7+": 10},
            examples=[{"left_id": "c1", "right_id": "c2"}],
            recommendations=["Test recommendation"],
        )

        result = report.to_dict()
        assert isinstance(result, dict)
        assert result["total_candidates"] == 100
        assert result["avg_candidates_per_entity"] == 5.0
        assert "candidate_distribution" in result
        assert "examples" in result
        assert "recommendations" in result

    def test_to_markdown(self) -> None:
        """Test to_markdown generates readable markdown."""
        report = CandidateInspectionReport(
            total_candidates=100,
            avg_candidates_per_entity=5.0,
            candidate_distribution={"1-3": 60, "4-6": 30, "7+": 10},
            examples=[
                {
                    "left_id": "c1",
                    "right_id": "c2",
                    "left_text": "Acme Corp",
                    "right_text": "ACME Corporation",
                }
            ],
            recommendations=["Increase k_neighbors for better recall"],
        )

        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "# Candidate Inspection Report" in markdown
        assert "100" in markdown  # total_candidates
        assert "5.0" in markdown  # avg_candidates_per_entity
        assert "Increase k_neighbors" in markdown  # recommendation

    def test_to_markdown_with_empty_fields(self) -> None:
        """Test to_markdown handles empty distribution, examples, and recommendations."""
        report = CandidateInspectionReport(
            total_candidates=0,
            avg_candidates_per_entity=0.0,
            candidate_distribution={},  # Empty dict
            examples=[],  # Empty list
            recommendations=[],  # Empty list
        )

        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "# Candidate Inspection Report" in markdown
        assert "0" in markdown  # total_candidates
        # Should not include sections for empty fields
        assert "## Candidate Distribution" not in markdown
        assert "## Sample Candidates" not in markdown
        assert "## Recommendations" not in markdown


class TestScoreInspectionReport:
    """Tests for ScoreInspectionReport model."""

    def test_valid_report_creation(self) -> None:
        """Test creating a valid ScoreInspectionReport."""
        report = ScoreInspectionReport(
            total_judgements=100,
            score_distribution={
                "mean": 0.5,
                "median": 0.48,
                "std": 0.2,
                "p25": 0.3,
                "p50": 0.48,
                "p75": 0.7,
                "p90": 0.85,
                "p95": 0.92,
            },
            high_scoring_examples=[
                {
                    "left_id": "c1",
                    "right_id": "c2",
                    "score": 0.95,
                    "reasoning": "Strong match",
                }
            ],
            low_scoring_examples=[
                {"left_id": "c3", "right_id": "c4", "score": 0.05, "reasoning": "No match"}
            ],
            recommendations=["Use threshold around 0.7 for clustering"],
        )

        assert report.total_judgements == 100
        assert report.score_distribution["mean"] == 0.5
        assert len(report.high_scoring_examples) == 1
        assert len(report.low_scoring_examples) == 1
        assert len(report.recommendations) == 1

    def test_empty_report_creation(self) -> None:
        """Test creating a report with zero judgements."""
        report = ScoreInspectionReport(
            total_judgements=0,
            score_distribution={
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
            },
            high_scoring_examples=[],
            low_scoring_examples=[],
            recommendations=["No judgements to analyze"],
        )

        assert report.total_judgements == 0
        assert report.score_distribution["mean"] == 0.0
        assert report.high_scoring_examples == []
        assert report.low_scoring_examples == []

    def test_stats_property(self) -> None:
        """Test stats property returns numerical metrics only."""
        report = ScoreInspectionReport(
            total_judgements=100,
            score_distribution={
                "mean": 0.5,
                "median": 0.48,
                "std": 0.2,
                "p25": 0.3,
                "p50": 0.48,
                "p75": 0.7,
                "p90": 0.85,
                "p95": 0.92,
            },
            high_scoring_examples=[{"left_id": "c1", "right_id": "c2", "score": 0.95}],
            low_scoring_examples=[{"left_id": "c3", "right_id": "c4", "score": 0.05}],
            recommendations=["Test recommendation"],
        )

        stats = report.stats
        assert stats["total_judgements"] == 100
        assert "score_distribution" in stats
        assert "high_scoring_examples" not in stats
        assert "recommendations" not in stats

    def test_to_dict(self) -> None:
        """Test to_dict returns JSON-serializable dictionary."""
        report = ScoreInspectionReport(
            total_judgements=100,
            score_distribution={
                "mean": 0.5,
                "median": 0.48,
                "std": 0.2,
                "p25": 0.3,
                "p50": 0.48,
                "p75": 0.7,
                "p90": 0.85,
                "p95": 0.92,
            },
            high_scoring_examples=[{"left_id": "c1", "right_id": "c2", "score": 0.95}],
            low_scoring_examples=[{"left_id": "c3", "right_id": "c4", "score": 0.05}],
            recommendations=["Test recommendation"],
        )

        result = report.to_dict()
        assert isinstance(result, dict)
        assert result["total_judgements"] == 100
        assert "score_distribution" in result
        assert "high_scoring_examples" in result
        assert "low_scoring_examples" in result
        assert "recommendations" in result

    def test_to_markdown(self) -> None:
        """Test to_markdown generates readable markdown."""
        report = ScoreInspectionReport(
            total_judgements=100,
            score_distribution={
                "mean": 0.5,
                "median": 0.48,
                "std": 0.2,
                "p25": 0.3,
                "p50": 0.48,
                "p75": 0.7,
                "p90": 0.85,
                "p95": 0.92,
            },
            high_scoring_examples=[
                {"left_id": "c1", "right_id": "c2", "score": 0.95, "reasoning": "Strong"}
            ],
            low_scoring_examples=[
                {"left_id": "c3", "right_id": "c4", "score": 0.05, "reasoning": "Weak"}
            ],
            recommendations=["Use threshold around 0.7"],
        )

        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "# Score Inspection Report" in markdown
        assert "100" in markdown  # total_judgements
        assert "0.5" in markdown  # mean

    def test_to_markdown_with_empty_fields(self) -> None:
        """Test to_markdown handles empty distribution, examples, and recommendations."""
        report = ScoreInspectionReport(
            total_judgements=0,
            score_distribution={},  # Empty dict
            high_scoring_examples=[],  # Empty list
            low_scoring_examples=[],  # Empty list
            recommendations=[],  # Empty list
        )

        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "# Score Inspection Report" in markdown
        assert "0" in markdown  # total_judgements
        # Should not include sections for empty fields
        assert "## Score Distribution" not in markdown
        assert "## High Scoring Examples" not in markdown
        assert "## Low Scoring Examples" not in markdown
        assert "## Recommendations" not in markdown


class TestClusterInspectionReport:
    """Tests for ClusterInspectionReport model."""

    def test_valid_report_creation(self) -> None:
        """Test creating a valid ClusterInspectionReport."""
        report = ClusterInspectionReport(
            total_clusters=50,
            singleton_rate=0.6,
            cluster_size_distribution={"1": 30, "2-3": 15, "4-6": 4, "7+": 1},
            largest_clusters=[
                {
                    "cluster_id": 0,
                    "size": 10,
                    "entity_ids": ["c1", "c2", "c3"],
                    "sample_text": ["Acme Corp", "ACME Corporation"],
                }
            ],
            recommendations=["High singleton rate - consider lowering threshold"],
        )

        assert report.total_clusters == 50
        assert report.singleton_rate == 0.6
        assert report.cluster_size_distribution == {"1": 30, "2-3": 15, "4-6": 4, "7+": 1}
        assert len(report.largest_clusters) == 1
        assert len(report.recommendations) == 1

    def test_empty_report_creation(self) -> None:
        """Test creating a report with zero clusters."""
        report = ClusterInspectionReport(
            total_clusters=0,
            singleton_rate=0.0,
            cluster_size_distribution={},
            largest_clusters=[],
            recommendations=["No clusters formed - check input data"],
        )

        assert report.total_clusters == 0
        assert report.singleton_rate == 0.0
        assert report.cluster_size_distribution == {}
        assert report.largest_clusters == []

    def test_stats_property(self) -> None:
        """Test stats property returns numerical metrics only."""
        report = ClusterInspectionReport(
            total_clusters=50,
            singleton_rate=0.6,
            cluster_size_distribution={"1": 30, "2-3": 15, "4-6": 4, "7+": 1},
            largest_clusters=[{"cluster_id": 0, "size": 10}],
            recommendations=["Test recommendation"],
        )

        stats = report.stats
        assert stats["total_clusters"] == 50
        assert stats["singleton_rate"] == 0.6
        assert "largest_clusters" not in stats
        assert "recommendations" not in stats

    def test_to_dict(self) -> None:
        """Test to_dict returns JSON-serializable dictionary."""
        report = ClusterInspectionReport(
            total_clusters=50,
            singleton_rate=0.6,
            cluster_size_distribution={"1": 30, "2-3": 15, "4-6": 4, "7+": 1},
            largest_clusters=[{"cluster_id": 0, "size": 10}],
            recommendations=["Test recommendation"],
        )

        result = report.to_dict()
        assert isinstance(result, dict)
        assert result["total_clusters"] == 50
        assert result["singleton_rate"] == 0.6
        assert "cluster_size_distribution" in result
        assert "largest_clusters" in result
        assert "recommendations" in result

    def test_to_markdown(self) -> None:
        """Test to_markdown generates readable markdown."""
        report = ClusterInspectionReport(
            total_clusters=50,
            singleton_rate=0.6,
            cluster_size_distribution={"1": 30, "2-3": 15, "4-6": 4, "7+": 1},
            largest_clusters=[
                {
                    "cluster_id": 0,
                    "size": 10,
                    "entity_ids": ["c1", "c2"],
                    "sample_text": ["Acme", "ACME"],
                }
            ],
            recommendations=["High singleton rate"],
        )

        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "# Cluster Inspection Report" in markdown
        assert "50" in markdown  # total_clusters
        assert "0.6" in markdown or "60" in markdown  # singleton_rate

    def test_to_markdown_with_empty_fields(self) -> None:
        """Test to_markdown handles empty distribution, clusters, and recommendations."""
        report = ClusterInspectionReport(
            total_clusters=0,
            singleton_rate=0.0,
            cluster_size_distribution={},  # Empty dict
            largest_clusters=[],  # Empty list
            recommendations=[],  # Empty list
        )

        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "# Cluster Inspection Report" in markdown
        assert "0" in markdown  # total_clusters
        # Should not include sections for empty fields
        assert "## Cluster Size Distribution" not in markdown
        assert "## Largest Clusters" not in markdown
        assert "## Recommendations" not in markdown


class TestRecallCurveStats:
    """Tests for RecallCurveStats validation."""

    def test_recall_curve_stats_validates_equal_list_lengths(self) -> None:
        """Test that RecallCurveStats requires all lists to have same length."""
        from langres.core.reports import RecallCurveStats

        # Should succeed - all same length
        valid_stats = RecallCurveStats(
            k_values=[1, 5, 10],
            recall_values=[0.1, 0.6, 0.9],
            avg_pairs_values=[1.0, 5.0, 10.0],
        )
        assert len(valid_stats.k_values) == 3

        # Should fail - recall_values too short
        with pytest.raises(ValueError, match="same length"):
            RecallCurveStats(
                k_values=[1, 5, 10],
                recall_values=[0.1, 0.6],  # Too short
                avg_pairs_values=[1.0, 5.0, 10.0],
            )

        # Should fail - recall_values too long
        with pytest.raises(ValueError, match="same length"):
            RecallCurveStats(
                k_values=[1, 5],
                recall_values=[0.1, 0.6, 0.9],  # Too long
                avg_pairs_values=[1.0, 5.0],
            )

    def test_recall_curve_stats_rejects_empty_lists(self) -> None:
        """Test that empty lists are rejected."""
        from langres.core.reports import RecallCurveStats

        with pytest.raises(ValueError, match="at least one"):
            RecallCurveStats(
                k_values=[],
                recall_values=[],
                avg_pairs_values=[],
            )

    def test_recall_curve_stats_allows_single_element(self) -> None:
        """Test that single-element lists are valid."""
        from langres.core.reports import RecallCurveStats

        stats = RecallCurveStats(
            k_values=[5],
            recall_values=[0.85],
            avg_pairs_values=[5.0],
        )
        assert len(stats.k_values) == 1
        assert stats.k_values[0] == 5
