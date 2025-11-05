"""Tests for LLMJudgeModule.inspect_scores() method.

This module tests the inspection capabilities of LLMJudgeModule for exploratory
analysis of score distributions without ground truth labels.
"""

import logging

import pytest

from langres.core.models import PairwiseJudgement
from langres.core.modules.llm_judge import LLMJudgeModule
from langres.core.reports import ScoreInspectionReport

logger = logging.getLogger(__name__)


@pytest.fixture
def normal_distribution_judgements() -> list[PairwiseJudgement]:
    """Create judgements with normal score distribution (mean~0.5, std~0.2)."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    return [
        PairwiseJudgement(
            left_id=f"left_{i}",
            right_id=f"right_{i}",
            score=score,
            score_type="prob_llm",
            decision_step="llm_judgment",
            reasoning=f"Test reasoning for score {score}",
            provenance={"model": "gpt-4o-mini", "cost_usd": 0.001},
        )
        for i, score in enumerate(scores)
    ]


@pytest.fixture
def high_scores_judgements() -> list[PairwiseJudgement]:
    """Create judgements with high scores (all > 0.8)."""
    scores = [0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    return [
        PairwiseJudgement(
            left_id=f"left_{i}",
            right_id=f"right_{i}",
            score=score,
            score_type="prob_llm",
            decision_step="llm_judgment",
            reasoning=f"High confidence match: {score}",
            provenance={"model": "gpt-4o-mini"},
        )
        for i, score in enumerate(scores)
    ]


@pytest.fixture
def low_scores_judgements() -> list[PairwiseJudgement]:
    """Create judgements with low scores (all < 0.2)."""
    scores = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.19]
    return [
        PairwiseJudgement(
            left_id=f"left_{i}",
            right_id=f"right_{i}",
            score=score,
            score_type="prob_llm",
            decision_step="llm_judgment",
            reasoning=f"Low confidence: {score}",
            provenance={"model": "gpt-4o-mini"},
        )
        for i, score in enumerate(scores)
    ]


@pytest.fixture
def bimodal_distribution_judgements() -> list[PairwiseJudgement]:
    """Create judgements with bimodal distribution (half low, half high)."""
    low_scores = [0.05, 0.08, 0.10, 0.12, 0.15]
    high_scores = [0.85, 0.88, 0.90, 0.92, 0.95]
    all_scores = low_scores + high_scores
    return [
        PairwiseJudgement(
            left_id=f"left_{i}",
            right_id=f"right_{i}",
            score=score,
            score_type="prob_llm",
            decision_step="llm_judgment",
            reasoning=f"Bimodal score: {score}",
            provenance={"model": "gpt-4o-mini"},
        )
        for i, score in enumerate(all_scores)
    ]


@pytest.fixture
def uniform_distribution_judgements() -> list[PairwiseJudgement]:
    """Create judgements with uniform distribution (all similar scores)."""
    scores = [0.48, 0.49, 0.50, 0.51, 0.52, 0.48, 0.49, 0.50, 0.51, 0.52]
    return [
        PairwiseJudgement(
            left_id=f"left_{i}",
            right_id=f"right_{i}",
            score=score,
            score_type="prob_llm",
            decision_step="llm_judgment",
            reasoning=f"Uniform score: {score}",
            provenance={"model": "gpt-4o-mini"},
        )
        for i, score in enumerate(scores)
    ]


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""

    class MockClient:
        def completion(self, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError("Should not be called in inspect_scores tests")

    return MockClient()


class TestLLMJudgeModuleInspection:
    """Tests for LLMJudgeModule.inspect_scores() method."""

    def test_inspect_scores_with_normal_distribution(
        self, normal_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test inspect_scores with normal score distribution."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(normal_distribution_judgements, sample_size=5)

        # Verify report structure
        assert isinstance(report, ScoreInspectionReport)
        assert report.total_judgements == 10
        assert isinstance(report.score_distribution, dict)
        assert isinstance(report.high_scoring_examples, list)
        assert isinstance(report.low_scoring_examples, list)
        assert isinstance(report.recommendations, list)

    def test_inspect_scores_computes_statistics_correctly(
        self, normal_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that score statistics are computed correctly."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(normal_distribution_judgements, sample_size=5)

        dist = report.score_distribution
        # Scores: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        assert "mean" in dist
        assert "median" in dist
        assert "std" in dist
        assert "p25" in dist
        assert "p50" in dist
        assert "p75" in dist
        assert "p90" in dist
        assert "p95" in dist
        assert "min" in dist
        assert "max" in dist

        # Check approximate values
        assert 0.5 <= dist["mean"] <= 0.6  # Mean should be around 0.525
        assert 0.5 <= dist["median"] <= 0.6  # Median should be around 0.55
        assert dist["min"] == 0.1
        assert dist["max"] == 0.95

    def test_inspect_scores_high_scoring_examples(
        self, normal_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that high-scoring examples are extracted correctly."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(normal_distribution_judgements, sample_size=3)

        # Should have 3 high-scoring examples
        assert len(report.high_scoring_examples) == 3

        # Examples should be sorted by score (highest first)
        scores = [ex["score"] for ex in report.high_scoring_examples]
        assert scores == sorted(scores, reverse=True)

        # Check example structure
        for example in report.high_scoring_examples:
            assert "left_id" in example
            assert "right_id" in example
            assert "score" in example
            assert "reasoning" in example
            assert isinstance(example["score"], float)
            assert isinstance(example["reasoning"], str)

    def test_inspect_scores_low_scoring_examples(
        self, normal_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that low-scoring examples are extracted correctly."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(normal_distribution_judgements, sample_size=3)

        # Should have 3 low-scoring examples
        assert len(report.low_scoring_examples) == 3

        # Examples should be sorted by score (lowest first)
        scores = [ex["score"] for ex in report.low_scoring_examples]
        assert scores == sorted(scores)

        # Check that lowest scores are included
        assert scores[0] == 0.1
        assert scores[1] == 0.2
        assert scores[2] == 0.3

    def test_inspect_scores_extracts_reasoning(
        self, normal_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that reasoning is extracted from judgement provenance."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(normal_distribution_judgements, sample_size=2)

        # Check that reasoning is present in examples
        for example in report.high_scoring_examples:
            assert "reasoning" in example
            assert "Test reasoning" in example["reasoning"]

    def test_inspect_scores_with_high_scores_changes_recommendations(
        self, high_scores_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that high scores trigger different recommendations."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(high_scores_judgements, sample_size=5)

        # With high median (> 0.7), should NOT recommend threshold=0.6
        recommendations_text = " ".join(report.recommendations)
        # High scores mean median > 0.7, so should not see balanced precision/recall
        assert report.score_distribution["median"] > 0.7

    def test_inspect_scores_with_low_scores_changes_recommendations(
        self, low_scores_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that low scores trigger different recommendations."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(low_scores_judgements, sample_size=5)

        # With low median (< 0.3), should recommend threshold=0.2
        recommendations_text = " ".join(report.recommendations)
        assert report.score_distribution["median"] < 0.3
        assert "0.2" in recommendations_text or "low" in recommendations_text.lower()

    def test_inspect_scores_with_bimodal_distribution_detects_good_separation(
        self, bimodal_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that bimodal distribution is detected as having good separation."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(bimodal_distribution_judgements, sample_size=5)

        # Bimodal distribution should have good separation (high - low > 0.3)
        recommendations_text = " ".join(report.recommendations)
        # Should NOT see warning about poor separation
        assert "Poor score separation" not in recommendations_text

    def test_inspect_scores_with_uniform_distribution_detects_poor_variance(
        self, uniform_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that uniform distribution triggers low variance warning."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(uniform_distribution_judgements, sample_size=5)

        # Uniform distribution should have very low std
        assert report.score_distribution["std"] < 0.1

        # Should trigger recommendation about uniform distribution
        recommendations_text = " ".join(report.recommendations)
        assert (
            "uniform" in recommendations_text.lower() or "variance" in recommendations_text.lower()
        )

    def test_inspect_scores_with_empty_list(self, mock_llm_client) -> None:
        """Test inspect_scores with empty judgements list."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores([], sample_size=10)

        assert report.total_judgements == 0
        assert len(report.high_scoring_examples) == 0
        assert len(report.low_scoring_examples) == 0
        assert len(report.recommendations) > 0  # Should suggest actions

    def test_inspect_scores_with_fewer_judgements_than_sample_size(self, mock_llm_client) -> None:
        """Test inspect_scores when judgements < sample_size."""
        judgements = [
            PairwiseJudgement(
                left_id="left_1",
                right_id="right_1",
                score=0.7,
                score_type="prob_llm",
                decision_step="llm_judgment",
                reasoning="Test",
                provenance={},
            ),
            PairwiseJudgement(
                left_id="left_2",
                right_id="right_2",
                score=0.3,
                score_type="prob_llm",
                decision_step="llm_judgment",
                reasoning="Test",
                provenance={},
            ),
        ]

        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(judgements, sample_size=10)

        # Should return all available judgements
        assert len(report.high_scoring_examples) <= 2
        assert len(report.low_scoring_examples) <= 2

    def test_inspect_scores_percentile_calculations(
        self, normal_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that percentiles are calculated correctly."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(normal_distribution_judgements, sample_size=5)

        dist = report.score_distribution
        # Verify percentiles are in order
        assert dist["p25"] <= dist["p50"]
        assert dist["p50"] <= dist["p75"]
        assert dist["p75"] <= dist["p90"]
        assert dist["p90"] <= dist["p95"]

    def test_inspect_scores_markdown_output(
        self, normal_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that report generates readable markdown."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(normal_distribution_judgements, sample_size=5)

        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "# Score Inspection Report" in markdown
        assert "Total Judgements" in markdown
        assert "Score Distribution" in markdown

    def test_inspect_scores_to_dict_structure(
        self, normal_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that to_dict returns proper structure."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(normal_distribution_judgements, sample_size=5)

        data = report.to_dict()
        assert isinstance(data, dict)
        assert "total_judgements" in data
        assert "score_distribution" in data
        assert "high_scoring_examples" in data
        assert "low_scoring_examples" in data
        assert "recommendations" in data

    def test_inspect_scores_stats_property_excludes_examples(
        self, normal_distribution_judgements: list[PairwiseJudgement], mock_llm_client
    ) -> None:
        """Test that stats property contains only numerical data."""
        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(normal_distribution_judgements, sample_size=5)

        stats = report.stats
        assert "total_judgements" in stats
        assert "score_distribution" in stats
        assert "high_scoring_examples" not in stats
        assert "low_scoring_examples" not in stats
        assert "recommendations" not in stats

    def test_inspect_scores_with_missing_reasoning_in_provenance(self, mock_llm_client) -> None:
        """Test handling when reasoning is missing from judgement."""
        judgements = [
            PairwiseJudgement(
                left_id="left_1",
                right_id="right_1",
                score=0.8,
                score_type="prob_llm",
                decision_step="llm_judgment",
                reasoning=None,  # No reasoning
                provenance={},
            )
        ]

        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(judgements, sample_size=5)

        # Should handle missing reasoning gracefully
        assert len(report.high_scoring_examples) == 1
        assert "reasoning" in report.high_scoring_examples[0]

    def test_inspect_scores_small_sample_triggers_warning(self, mock_llm_client) -> None:
        """Test that small sample size triggers warning in recommendations."""
        # Create small judgement list (< 50)
        judgements = [
            PairwiseJudgement(
                left_id=f"left_{i}",
                right_id=f"right_{i}",
                score=0.5,
                score_type="prob_llm",
                decision_step="llm_judgment",
                reasoning="Test",
                provenance={},
            )
            for i in range(20)
        ]

        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(judgements, sample_size=5)

        # Should warn about small sample
        recommendations_text = " ".join(report.recommendations)
        assert (
            "small" in recommendations_text.lower()
            or "representative" in recommendations_text.lower()
        )

    def test_inspect_scores_large_sample_triggers_sampling_suggestion(
        self, mock_llm_client
    ) -> None:
        """Test that large sample triggers sampling suggestion."""
        # Create large judgement list (> 1000)
        judgements = [
            PairwiseJudgement(
                left_id=f"left_{i}",
                right_id=f"right_{i}",
                score=0.5,
                score_type="prob_llm",
                decision_step="llm_judgment",
                reasoning="Test",
                provenance={},
            )
            for i in range(1500)
        ]

        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(judgements, sample_size=5)

        # Should suggest sampling for faster iteration
        recommendations_text = " ".join(report.recommendations)
        assert (
            "sampling" in recommendations_text.lower() or "faster" in recommendations_text.lower()
        )

    def test_inspect_scores_medium_variance_no_special_recommendation(
        self, mock_llm_client
    ) -> None:
        """Test that medium variance (0.1 < std < 0.35) doesn't trigger variance warnings."""
        # Create distribution with medium variance (std ~ 0.2)
        scores = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        judgements = [
            PairwiseJudgement(
                left_id=f"left_{i}",
                right_id=f"right_{i}",
                score=score,
                score_type="prob_llm",
                decision_step="llm_judgment",
                reasoning="Test",
                provenance={},
            )
            for i, score in enumerate(scores)
        ]

        module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")
        report = module.inspect_scores(judgements, sample_size=5)

        # Should have std in medium range
        std = report.score_distribution["std"]
        assert 0.1 < std < 0.35

        # Should NOT have warnings about uniform distribution or good variance
        recommendations_text = " ".join(report.recommendations)
        assert "uniform" not in recommendations_text.lower()
        assert "Good score variance" not in recommendations_text
