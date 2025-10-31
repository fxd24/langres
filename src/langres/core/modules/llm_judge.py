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
from openai import OpenAI

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, SchemaT

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
    """Schema-agnostic LLM-based matching module using OpenAI API.

    This module uses an LLM (like GPT-4) to make match judgments with
    natural language reasoning. It provides calibrated probability scores
    and tracks API costs for observability.

    Example:
        module = LLMJudgeModule(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
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
        model: str = "gpt-4o-mini",
        api_key: str = "",
        temperature: float = 0.0,
        prompt_template: str | None = None,
        use_litellm: bool = True,
        litellm_client: Any | None = None,
    ):
        """Initialize LLMJudgeModule.

        Args:
            model: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4")
            api_key: OpenAI API key
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = random)
            prompt_template: Custom prompt template (uses DEFAULT_PROMPT if None)
            use_litellm: If True, use LiteLLM for enhanced observability (Langfuse tracing).
                        If False, use OpenAI client directly. Default: True.
            litellm_client: Optional pre-configured LiteLLM client. If None and
                           use_litellm=True, uses default litellm module.

        Raises:
            ValueError: If api_key is empty or temperature out of range

        Example:
            # With LiteLLM (recommended for Langfuse tracing)
            module = LLMJudgeModule(
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                use_litellm=True
            )

            # With direct OpenAI client (legacy)
            module = LLMJudgeModule(
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                use_litellm=False
            )

        Note:
            When use_litellm=True, LLM calls will be automatically traced in
            Langfuse if you've configured LiteLLM callbacks via
            langres.clients.create_llm_client().
        """
        if not api_key:
            raise ValueError("API key is required")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PROMPT
        self.use_litellm = use_litellm

        if use_litellm:
            self._litellm = litellm_client if litellm_client is not None else litellm
        else:
            self._client = OpenAI(api_key=api_key)

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

            if self.use_litellm:
                # Use LiteLLM (supports Langfuse tracing)
                response = self._litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
            else:
                # Use OpenAI client directly
                response = self._client.chat.completions.create(
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
