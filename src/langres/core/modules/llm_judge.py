"""LLMJudgeModule implementation for LLM-based entity matching.

This module uses OpenAI API (or compatible) for match judgments with natural
language reasoning and calibrated probability scores.

Supports both direct OpenAI client and LiteLLM for enhanced observability.
"""

import logging
import re
from collections.abc import Iterator
from typing import Any

import litellm
import numpy as np
from openai import OpenAI

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, SchemaT
from langres.core.reports import ScoreInspectionReport

logger = logging.getLogger(__name__)

# Default prompt template for LLM judgment
DEFAULT_PROMPT = """You are an expert at entity resolution. Determine if these two company records refer to the same real-world company.

Company A:
{left}

Company B:
{right}

Respond in exactly this format:
MATCH or NO_MATCH
Score: <probability between 0.0 and 1.0>
Reasoning: <brief explanation>

The score should be your confidence that these are the same company (1.0 = definitely same, 0.0 = definitely different)."""


class LLMJudgeModule(Module[SchemaT]):
    """Schema-agnostic LLM-based matching module using LiteLLM.

    This module uses an LLM (like GPT-4) to make match judgments with
    natural language reasoning. It provides calibrated probability scores
    and tracks API costs for observability.

    The module accepts a pre-configured LiteLLM client, enabling:
    - Automatic Langfuse tracing for observability
    - Support for multiple LLM providers (OpenAI, Azure, etc.)
    - Proper separation of concerns (client configuration vs. matching logic)

    Example:
        from langres.clients import create_llm_client
        from langres.clients.settings import Settings

        settings = Settings()
        llm_client = create_llm_client(settings)

        module = LLMJudgeModule(
            client=llm_client,
            model="gpt-4o-mini",
            temperature=0.0,
        )

        for judgement in module.forward(candidates):
            print(f"{judgement.left_id} vs {judgement.right_id}: {judgement.score}")
            print(f"Reasoning: {judgement.reasoning}")
            print(f"Cost: ${judgement.provenance['cost_usd']}")

    Note:
        The module uses gpt-4o-mini by default for cost efficiency. For higher
        quality, use gpt-4 (but costs ~30x more).

    Note:
        Cost tracking uses approximate pricing:
        - gpt-4o-mini: $0.150/1M input tokens, $0.600/1M output tokens
        - gpt-4: $30/1M input tokens, $60/1M output tokens
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        prompt_template: str | None = None,
    ):
        """Initialize LLMJudgeModule.

        Args:
            client: Pre-configured LLM client (LiteLLM or OpenAI client).
                   Use langres.clients.create_llm_client() to create a client
                   with Langfuse tracing enabled.
            model: Model name (e.g., "gpt-4o-mini", "azure/gpt-5-mini")
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = random)
            prompt_template: Custom prompt template (uses DEFAULT_PROMPT if None)

        Raises:
            ValueError: If temperature out of range

        Example:
            # Create LiteLLM client with tracing
            from langres.clients import create_llm_client
            from langres.clients.settings import Settings

            settings = Settings()
            llm_client = create_llm_client(settings)

            # Initialize module with client
            module = LLMJudgeModule(
                client=llm_client,
                model="azure/gpt-5-mini",
                temperature=0.0
            )

        Note:
            The client handles all authentication and tracing configuration.
            Use langres.clients.create_llm_client() to create a properly
            configured client with Langfuse observability.
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        self.client = client
        self.model = model
        self.temperature = temperature
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PROMPT

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Compare entity pairs using LLM judgment.

        Args:
            candidates: Stream of normalized entity pairs

        Yields:
            PairwiseJudgement objects with LLM scores and reasoning

        Note:
            Each API call is made synchronously. For production use with high
            volume, consider batching or async processing.
        """
        for candidate in candidates:
            # Format entities as strings
            left_str = self._format_entity(candidate.left)
            right_str = self._format_entity(candidate.right)

            # Create prompt
            prompt = self.prompt_template.format(left=left_str, right=right_str)

            # Call LLM API
            logger.debug(
                "Calling LLM API for pair: %s vs %s",
                candidate.left.id,  # type: ignore[attr-defined]
                candidate.right.id,  # type: ignore[attr-defined]
            )

            # Call client (works for both LiteLLM and OpenAI)
            response = self.client.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )

            # Extract score and reasoning from response
            content = response.choices[0].message.content or ""
            score = self._extract_score(content)
            reasoning = self._extract_reasoning(content)

            # Calculate cost
            cost_usd = self._calculate_cost(response)

            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=score,
                score_type="prob_llm",
                decision_step="llm_judgment",
                reasoning=reasoning,
                provenance={
                    "model": self.model,
                    "cost_usd": cost_usd,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": (
                        response.usage.completion_tokens if response.usage else 0
                    ),
                },
            )

    def _format_entity(self, entity: SchemaT) -> str:
        """Format entity as string for LLM prompt.

        Args:
            entity: Pydantic entity schema

        Returns:
            String representation of entity
        """
        return entity.model_dump_json(indent=2)

    def _extract_score(self, content: str) -> float:
        """Extract probability score from LLM response.

        Args:
            content: LLM response text

        Returns:
            Probability score in range [0.0, 1.0]

        Note:
            Looks for "Score: 0.XX" pattern. Defaults to 0.5 if not found.
        """
        # Look for "Score: 0.XX" pattern
        match = re.search(r"Score:\s*(\d+\.?\d*)", content, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            # Clamp to [0, 1] range
            return max(0.0, min(1.0, score))

        logger.warning("Could not extract score from LLM response, defaulting to 0.5")
        return 0.5

    def _extract_reasoning(self, content: str) -> str:
        """Extract reasoning from LLM response.

        Args:
            content: LLM response text

        Returns:
            Reasoning text

        Note:
            Looks for "Reasoning:" followed by text. Returns full content if not found.
        """
        # Look for "Reasoning:" followed by text
        match = re.search(r"Reasoning:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: return full content
        return content

    def _calculate_cost(self, response) -> float:  # type: ignore[no-untyped-def]
        """Calculate API call cost in USD.

        Args:
            response: OpenAI API response

        Returns:
            Cost in USD

        Note:
            Uses approximate pricing as of 2024:
            - gpt-4o-mini: $0.150/1M input, $0.600/1M output
            - gpt-4: $30/1M input, $60/1M output
        """
        if not response.usage:
            return 0.0

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        # Pricing per 1M tokens (approximate)
        if "gpt-4o-mini" in self.model:
            input_price = 0.150
            output_price = 0.600
        elif "gpt-4" in self.model:
            input_price = 30.0
            output_price = 60.0
        else:
            # Default to gpt-4o-mini pricing
            input_price = 0.150
            output_price = 0.600

        cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
        return float(cost)

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground truth labels.

        Use this method to understand scoring output before labeling:
        - Score distribution statistics
        - High and low scoring examples with reasoning
        - Threshold recommendations based on distribution

        For quality evaluation with ground truth labels, use
        PipelineDebugger.analyze_scores() instead.

        Args:
            judgements: List of PairwiseJudgement objects to analyze
            sample_size: Number of examples to include (default: 10)

        Returns:
            ScoreInspectionReport with statistics, examples, and recommendations
        """
        return _inspect_scores_impl(judgements, sample_size)


def _inspect_scores_impl(
    judgements: list[PairwiseJudgement], sample_size: int = 10
) -> ScoreInspectionReport:
    """Shared implementation of inspect_scores for all Module types.

    This function provides common score inspection logic that works for all
    Module implementations since they all return PairwiseJudgement objects.

    Args:
        judgements: List of PairwiseJudgement objects to analyze
        sample_size: Number of examples to include (default: 10)

    Returns:
        ScoreInspectionReport with statistics, examples, and recommendations
    """
    # Handle empty judgements
    if not judgements:
        return ScoreInspectionReport(
            total_judgements=0,
            score_distribution={},
            high_scoring_examples=[],
            low_scoring_examples=[],
            recommendations=[
                "No judgements to analyze - generate some predictions first",
                "Run Module.forward() on candidates to produce judgements",
            ],
        )

    # Extract scores
    scores = [j.score for j in judgements]
    total = len(judgements)

    # Compute statistics
    score_distribution = {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }

    # Extract high-scoring examples (top sample_size)
    sorted_by_score_desc = sorted(judgements, key=lambda j: j.score, reverse=True)
    high_scoring_examples = [
        {
            "left_id": j.left_id,
            "right_id": j.right_id,
            "score": j.score,
            "reasoning": j.reasoning if j.reasoning else "",
        }
        for j in sorted_by_score_desc[:sample_size]
    ]

    # Extract low-scoring examples (bottom sample_size)
    sorted_by_score_asc = sorted(judgements, key=lambda j: j.score)
    low_scoring_examples = [
        {
            "left_id": j.left_id,
            "right_id": j.right_id,
            "score": j.score,
            "reasoning": j.reasoning if j.reasoning else "",
        }
        for j in sorted_by_score_asc[:sample_size]
    ]

    # Generate recommendations
    recommendations = _generate_recommendations_impl(
        total_judgements=total,
        score_distribution=score_distribution,
        scores=scores,
    )

    return ScoreInspectionReport(
        total_judgements=total,
        score_distribution=score_distribution,
        high_scoring_examples=high_scoring_examples,
        low_scoring_examples=low_scoring_examples,
        recommendations=recommendations,
    )


def _generate_recommendations_impl(
    total_judgements: int,
    score_distribution: dict[str, float],
    scores: list[float],
) -> list[str]:
    """Generate rule-based recommendations based on score distribution.

    Args:
        total_judgements: Total number of judgements
        score_distribution: Statistics dictionary
        scores: List of all scores

    Returns:
        List of actionable recommendations
    """
    recommendations = []

    # 1. Threshold suggestion based on median
    median = score_distribution["median"]
    if median > 0.7:
        recommendations.append(
            "High median score (>0.7) suggests most pairs are matches. "
            "Consider threshold=0.6 for balanced precision/recall."
        )
    elif median < 0.3:
        recommendations.append(
            "Low median score (<0.3) suggests most pairs are non-matches. "
            "Consider threshold=0.2 as starting point."
        )
    else:
        recommendations.append(
            f"Median score is {median:.2f}. Consider threshold={median:.1f} as starting point."
        )

    # 2. Score separation analysis
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    high_scores = sorted_scores[int(0.75 * n) :]  # Top 25%
    low_scores = sorted_scores[: int(0.25 * n)]  # Bottom 25%

    if high_scores and low_scores:
        separation = abs(float(np.mean(high_scores)) - float(np.mean(low_scores)))
        if separation < 0.3:
            recommendations.append(
                "⚠️ Poor score separation (<0.3) - scores are not well-calibrated. "
                "Consider tuning the prompt or using a different model."
            )

    # 3. Variance analysis
    std = score_distribution["std"]
    if std < 0.1:
        recommendations.append(
            "⚠️ Very uniform distribution (std<0.1) - scores lack discriminative power. "
            "Consider more diverse scoring or feature engineering."
        )
    elif std > 0.35:
        recommendations.append(
            "✅ Good score variance (std>0.35) - indicates discriminative scoring."
        )

    # 4. Sample size guidance
    if total_judgements < 50:
        recommendations.append(
            "Small sample (<50 judgements) - results may not be representative. "
            "Generate more predictions for reliable analysis."
        )
    elif total_judgements > 1000:
        recommendations.append(
            "Large sample (>1000 judgements) - consider sampling a subset for faster iteration "
            "during parameter tuning."
        )

    return recommendations
