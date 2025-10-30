"""CascadeModule implementation for hybrid embeddings + LLM with early exit.

This module implements the cascade pattern that optimizes cost while maintaining
quality by using cheap embedding similarity checks for obvious cases and expensive
LLM judgment only for uncertain cases.
"""

import logging
import re
from collections.abc import Iterator

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

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


class CascadeModule(Module[SchemaT]):
    """Schema-agnostic cascade module: embeddings + LLM with early exit.

    This module implements a multi-stage cascade pattern to optimize cost:
    1. Stage 1: Cheap embedding similarity check
    2. Early exit if score < low_threshold (definitely not a match)
    3. Early exit if score > high_threshold (definitely a match)
    4. Stage 2: Expensive LLM judgment for uncertain cases

    This approach reduces LLM API costs while maintaining high quality by only
    calling the LLM for the hardest cases.

    Example:
        module = CascadeModule(
            embedding_model_name="all-MiniLM-L6-v2",
            llm_model="gpt-4o-mini",
            llm_api_key=os.getenv("OPENAI_API_KEY"),
            low_threshold=0.3,  # Below this = not a match
            high_threshold=0.9,  # Above this = match
        )

        for judgement in module.forward(candidates):
            print(f"Decision: {judgement.decision_step}")
            print(f"Score: {judgement.score}")
            if judgement.decision_step == "llm_judgment":
                print(f"LLM reasoning: {judgement.reasoning}")
                print(f"Cost: ${judgement.provenance['llm_cost_usd']}")

    Note:
        The thresholds should be tuned based on your precision/recall requirements
        and cost constraints. Lower low_threshold = more LLM calls = higher cost
        but better recall. Higher high_threshold = more LLM calls = higher cost
        but better precision.

    Note:
        Typical threshold ranges:
        - low_threshold: 0.2-0.4 (below this is definitely not a match)
        - high_threshold: 0.85-0.95 (above this is definitely a match)
        - The gap between them determines LLM usage (larger gap = fewer LLM calls)
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str = "",
        low_threshold: float = 0.3,
        high_threshold: float = 0.9,
        llm_temperature: float = 0.0,
        llm_prompt_template: str | None = None,
    ):
        """Initialize CascadeModule.

        Args:
            embedding_model_name: Name of sentence-transformers model for embeddings
            llm_model: OpenAI model name for LLM judgment (e.g., "gpt-4o-mini")
            llm_api_key: OpenAI API key
            low_threshold: Embedding similarity threshold for early exit (no match)
            high_threshold: Embedding similarity threshold for early exit (match)
            llm_temperature: LLM sampling temperature (0.0 = deterministic)
            llm_prompt_template: Custom LLM prompt (uses DEFAULT_PROMPT if None)

        Raises:
            ValueError: If thresholds are invalid or API key is missing
        """
        if not llm_api_key:
            raise ValueError("LLM API key is required")

        if not 0.0 <= low_threshold <= 1.0 or not 0.0 <= high_threshold <= 1.0:
            raise ValueError("Thresholds must be between 0.0 and 1.0")

        if low_threshold >= high_threshold:
            raise ValueError("low_threshold must be < high_threshold")

        self.embedding_model_name = embedding_model_name
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.llm_temperature = llm_temperature
        self.llm_prompt_template = llm_prompt_template or DEFAULT_PROMPT

        # Lazy-load models
        self._embedding_model: SentenceTransformer | None = None
        self._llm_client: OpenAI | None = None

    def _get_embedding_model(self) -> SentenceTransformer:
        """Get or load the sentence-transformers model.

        Returns:
            Loaded SentenceTransformer model.
        """
        if self._embedding_model is None:
            logger.info("Loading embedding model: %s", self.embedding_model_name)
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _get_llm_client(self) -> OpenAI:
        """Get or create the OpenAI client.

        Returns:
            OpenAI client instance.
        """
        if self._llm_client is None:
            self._llm_client = OpenAI(api_key=self.llm_api_key)
        return self._llm_client

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Compare entity pairs using cascade pattern.

        Args:
            candidates: Stream of normalized entity pairs

        Yields:
            PairwiseJudgement objects with cascaded decisions

        Note:
            Each candidate goes through:
            1. Embedding similarity calculation
            2. If score < low_threshold: early exit (no match)
            3. If score > high_threshold: early exit (match)
            4. Otherwise: LLM judgment (uncertain case)
        """
        model = self._get_embedding_model()

        for candidate in candidates:
            # Stage 1: Compute embedding similarity
            left_text = self._extract_text(candidate.left)
            right_text = self._extract_text(candidate.right)

            # Encode texts to embeddings
            embeddings = model.encode([left_text, right_text], convert_to_numpy=True)
            left_emb = embeddings[0]
            right_emb = embeddings[1]

            # Calculate cosine similarity
            embed_score = self._cosine_similarity(left_emb, right_emb)

            # Decision: Early exit low similarity
            if embed_score < self.low_threshold:
                yield PairwiseJudgement(
                    left_id=candidate.left.id,  # type: ignore[attr-defined]
                    right_id=candidate.right.id,  # type: ignore[attr-defined]
                    score=embed_score,
                    score_type="sim_cos",
                    decision_step="early_exit_low_similarity",
                    reasoning="",
                    provenance={
                        "embed_score": float(embed_score),
                        "embedding_model": self.embedding_model_name,
                    },
                )
                continue

            # Decision: Early exit high similarity
            if embed_score > self.high_threshold:
                yield PairwiseJudgement(
                    left_id=candidate.left.id,  # type: ignore[attr-defined]
                    right_id=candidate.right.id,  # type: ignore[attr-defined]
                    score=embed_score,
                    score_type="sim_cos",
                    decision_step="early_exit_high_similarity",
                    reasoning="",
                    provenance={
                        "embed_score": float(embed_score),
                        "embedding_model": self.embedding_model_name,
                    },
                )
                continue

            # Stage 2: LLM judgment for uncertain case
            logger.debug(
                "Uncertain case (embed_score=%.3f), calling LLM for %s vs %s",
                embed_score,
                candidate.left.id,  # type: ignore[attr-defined]
                candidate.right.id,  # type: ignore[attr-defined]
            )

            llm_score, llm_reasoning, llm_cost = self._llm_judgment(candidate)

            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=llm_score,
                score_type="prob_llm",
                decision_step="llm_judgment",
                reasoning=llm_reasoning,
                provenance={
                    "embed_score": float(embed_score),
                    "embedding_model": self.embedding_model_name,
                    "llm_cost_usd": llm_cost,
                    "model": self.llm_model,
                },
            )

    def _extract_text(self, entity: SchemaT) -> str:
        """Extract text from entity for embedding.

        Args:
            entity: Pydantic entity schema

        Returns:
            Text representation of entity
        """
        # Default: use all fields as JSON
        return entity.model_dump_json(indent=2)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Cosine similarity in range [-1, 1] (typically [0, 1] for text)
        """
        # Cosine similarity = dot(a, b) / (||a|| * ||b||)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        return float(similarity)

    def _llm_judgment(self, candidate: ERCandidate[SchemaT]) -> tuple[float, str, float]:
        """Get LLM judgment for a candidate pair.

        Args:
            candidate: Entity pair to judge

        Returns:
            Tuple of (score, reasoning, cost_usd)
        """
        client = self._get_llm_client()

        # Format entities as strings
        left_str = candidate.left.model_dump_json(indent=2)
        right_str = candidate.right.model_dump_json(indent=2)

        # Create prompt
        prompt = self.llm_prompt_template.format(left=left_str, right=right_str)

        # Call LLM API
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.llm_temperature,
        )

        # Extract score and reasoning
        content = response.choices[0].message.content or ""
        score = self._extract_score(content)
        reasoning = self._extract_reasoning(content)

        # Calculate cost
        cost = self._calculate_cost(response)

        return score, reasoning, cost

    def _extract_score(self, content: str) -> float:
        """Extract probability score from LLM response.

        Args:
            content: LLM response text

        Returns:
            Probability score in range [0.0, 1.0]
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
        """
        if not response.usage:
            return 0.0

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        # Pricing per 1M tokens (approximate)
        if "gpt-4o-mini" in self.llm_model:
            input_price = 0.150
            output_price = 0.600
        elif "gpt-4" in self.llm_model:
            input_price = 30.0
            output_price = 60.0
        else:
            # Default to gpt-4o-mini pricing
            input_price = 0.150
            output_price = 0.600

        cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
        return float(cost)
