"""LLMJudge: serializable LLM-based entity-matching Module.

This module uses OpenAI API (or compatible) for match judgments with natural
language reasoning and calibrated probability scores.

Supports both direct OpenAI client and LiteLLM for enhanced observability.
"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, ClassVar

import litellm
from openai import OpenAI

# Type checking for litellm exceptions
try:
    from litellm import RateLimitError
except ImportError:
    RateLimitError = Exception

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, SchemaT
from langres.core.registry import register
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl

logger = logging.getLogger(__name__)

# Default prompt template for LLM judgment.
#
# Single source of truth for the neutral prompt (Cascade imports this too). It is
# domain-neutral: ``{entity_noun}`` is woven in at render time so the same
# template serves companies, products, people, etc. ``{left}`` / ``{right}`` stay
# unfilled here and are formatted with the two records at judgement time.
DEFAULT_PROMPT = """You are an expert at entity resolution. Determine if these two {entity_noun} records refer to the same real-world {entity_noun}.

Record A:
{left}

Record B:
{right}

Respond in exactly this format:
MATCH or NO_MATCH
Score: <probability between 0.0 and 1.0>
Reasoning: <brief explanation>

The score should be your confidence that these are the same {entity_noun} (1.0 = definitely same, 0.0 = definitely different)."""


def render_default_prompt(entity_noun: str = "entity") -> str:
    """Render :data:`DEFAULT_PROMPT` for ``entity_noun``, leaving ``{left}``/``{right}``.

    Substitutes only the ``{entity_noun}`` placeholder so the result is still a
    ``str.format`` template expecting ``left`` and ``right`` at judgement time.
    """
    return DEFAULT_PROMPT.replace("{entity_noun}", entity_noun)


class _RateLimiter:
    """Token-aware rate limiter for LLM API calls.

    Tracks both requests-per-minute (RPM) and tokens-per-minute (TPM)
    in a sliding window to prevent exceeding API rate limits.

    This is a lightweight, single-process rate limiter. For distributed
    systems, use LiteLLM Router with Redis instead.
    """

    def __init__(self, rpm_limit: int, tpm_limit: int):
        """Initialize rate limiter.

        Args:
            rpm_limit: Maximum requests per minute
            tpm_limit: Maximum tokens per minute
        """
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit

        # Track requests and tokens in 1-minute sliding windows
        self._request_times: list[float] = []
        self._token_usage: list[tuple[float, int]] = []  # (timestamp, token_count)
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """Wait until request can be made without exceeding limits.

        Args:
            estimated_tokens: Estimated tokens for this request (default: 1000)
        """
        async with self._lock:
            now = time.time()
            one_minute_ago = now - 60.0

            # Remove old entries outside the 1-minute window
            self._request_times = [t for t in self._request_times if t > one_minute_ago]
            self._token_usage = [(t, c) for t, c in self._token_usage if t > one_minute_ago]

            # Wait if we're at RPM limit
            while len(self._request_times) >= self.rpm_limit:
                oldest_request = self._request_times[0]
                sleep_time = 60.0 - (now - oldest_request) + 0.1  # Add 100ms buffer
                logger.debug("RPM limit reached, sleeping for %.2fs", sleep_time)
                await asyncio.sleep(sleep_time)

                # Refresh window
                now = time.time()
                one_minute_ago = now - 60.0
                self._request_times = [t for t in self._request_times if t > one_minute_ago]

            # Wait if we're at TPM limit
            current_tokens = sum(count for _, count in self._token_usage)
            while current_tokens + estimated_tokens > self.tpm_limit:
                oldest_token_time = self._token_usage[0][0]
                sleep_time = 60.0 - (now - oldest_token_time) + 0.1  # Add 100ms buffer
                logger.debug(
                    "TPM limit reached (%d tokens), sleeping for %.2fs", current_tokens, sleep_time
                )
                await asyncio.sleep(sleep_time)

                # Refresh window
                now = time.time()
                one_minute_ago = now - 60.0
                self._token_usage = [(t, c) for t, c in self._token_usage if t > one_minute_ago]
                current_tokens = sum(count for _, count in self._token_usage)

            # Record this request
            self._request_times.append(now)

    async def record_usage_async(self, token_count: int) -> None:
        """Record actual token usage after API call (async, thread-safe).

        Args:
            token_count: Actual tokens used in the request

        Note:
            This method is async and uses the internal lock to prevent
            race conditions when multiple tasks record usage concurrently.
        """
        async with self._lock:
            self._token_usage.append((time.time(), token_count))


@register("llm_judge")
class LLMJudge(Module[SchemaT]):
    """Schema-agnostic LLM-based matching module using LiteLLM.

    This module uses an LLM to make match judgments with natural language
    reasoning. It provides calibrated probability scores and tracks API costs
    (via LiteLLM's own pricing) for observability.

    It is a first-class, serializable Resolver component: it carries a registry
    ``type_name`` and a pure :attr:`config` (model, temperature, prompt,
    entity_noun) so a Resolver with an LLM judge in the ``module`` slot can
    ``save`` / ``load``. The LLM ``client`` is **never** serialized — it is
    reconstructed from environment at load time via the lazy-client path.

    The client is optional. When omitted, it is lazily built from the
    environment with ``create_llm_client(Settings())`` on first use, enabling:
    - Automatic Langfuse tracing for observability
    - Support for multiple LLM providers (OpenAI, Azure, etc.)
    - Serialization without persisting any secret

    Example:
        # Happy path: build the client from environment.
        module = LLMJudge.from_env(model="gpt-5-mini")

        for judgement in module.forward(candidates):
            print(f"{judgement.left_id} vs {judgement.right_id}: {judgement.score}")
            print(f"Reasoning: {judgement.reasoning}")
            print(f"Cost: ${judgement.provenance['cost_usd']}")

    Example:
        # Escape hatch: inject a pre-configured client (e.g. in tests).
        from langres.clients import create_llm_client, Settings

        module = LLMJudge(client=create_llm_client(Settings()), model="gpt-5-mini")

    Note:
        Defaults to ``gpt-5-mini`` at ``temperature=1.0``. Cost tracking is
        delegated to ``litellm.completion_cost`` so pricing stays honest for
        whatever model is actually used (no hardcoded table).
    """

    # Registry key, mirrored as a class attribute so the Resolver's uniform
    # serialization helper can discover the type name (see resolver.py).
    type_name: ClassVar[str] = "llm_judge"

    def __init__(
        self,
        client: Any = None,
        model: str = "gpt-5-mini",
        temperature: float = 1.0,
        prompt_template: str | None = None,
        entity_noun: str = "entity",
    ):
        """Initialize LLMJudge.

        Args:
            client: Optional pre-configured LLM client (LiteLLM or OpenAI
                client). When ``None`` (the default), the client is lazily built
                from the environment via ``create_llm_client(Settings())`` on
                first use. Inject a client only as an escape hatch (e.g. tests
                or a custom client); use :meth:`from_env` for the happy path.
            model: Model name (e.g., "gpt-5-mini", "azure/gpt-5-mini")
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = random)
            prompt_template: Custom prompt template (uses the neutral
                :data:`DEFAULT_PROMPT`, rendered for ``entity_noun``, if None)
            entity_noun: Domain noun woven into the default prompt (e.g.
                "company", "product"). Ignored when ``prompt_template`` is given.

        Raises:
            ValueError: If temperature out of range
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        self.client = client
        self.model = model
        self.temperature = temperature
        self.entity_noun = entity_noun
        self.prompt_template = (
            prompt_template if prompt_template else render_default_prompt(entity_noun)
        )

    @classmethod
    def from_env(
        cls,
        model: str = "gpt-5-mini",
        **kwargs: Any,
    ) -> "LLMJudge[SchemaT]":
        """Build an LLMJudge with a client constructed from the environment.

        The documented happy path: reads provider/tracing config from env via
        ``create_llm_client(Settings())``. ``kwargs`` are forwarded to
        ``__init__`` (``temperature``, ``prompt_template``, ``entity_noun``).
        """
        from langres.clients import Settings, create_llm_client

        return cls(client=create_llm_client(Settings()), model=model, **kwargs)

    def _get_client(self) -> Any:
        """Return the client, lazily building one from env on first use."""
        if self.client is None:
            from langres.clients import Settings, create_llm_client

            self.client = create_llm_client(Settings())
        return self.client

    @property
    def config(self) -> dict[str, object]:
        """Pure, serializable construction config (never the client or secrets)."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "prompt_template": self.prompt_template,
            "entity_noun": self.entity_noun,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "LLMJudge[SchemaT]":
        """Rebuild from :attr:`config` via the lazy-client path (client from env)."""
        return cls(
            client=None,
            model=str(config["model"]),
            temperature=float(config["temperature"]),  # type: ignore[arg-type]
            prompt_template=str(config["prompt_template"]),
            entity_noun=str(config["entity_noun"]),
        )

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
            response = self._get_client().completion(
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

    async def forward_async(
        self,
        candidates: list[ERCandidate[SchemaT]],
        max_concurrent: int = 50,
        rpm_limit: int = 250,
        tpm_limit: int = 250000,
        max_retries: int = 3,
    ) -> list[PairwiseJudgement]:
        """Compare entity pairs using async batch processing with rate limiting.

        This method provides significant speedup over forward() by processing
        multiple candidates concurrently while respecting API rate limits.

        Args:
            candidates: List of normalized entity pairs (materialized, not streaming)
            max_concurrent: Maximum parallel API calls (default: 50)
            rpm_limit: Requests per minute limit (default: 250)
            tpm_limit: Tokens per minute limit (default: 250,000)
            max_retries: Maximum retry attempts for rate limit errors (default: 3)

        Returns:
            List of PairwiseJudgement objects in same order as input candidates

        Example:
            import asyncio
            from langres.clients import create_llm_client

            client = create_llm_client()
            module = LLMJudgeModule(client=client, model="gpt-4o-mini")

            # Process 100 candidates with async batching
            candidates = list(blocker.forward(data))
            judgements = asyncio.run(module.forward_async(
                candidates,
                max_concurrent=50,  # 50 concurrent requests
                rpm_limit=250,      # Stay under API limits
                tpm_limit=250000
            ))

        Note:
            - Uses exponential backoff retry for rate limit (429) errors
            - Tracks token usage to prevent exceeding TPM limits
            - Maintains same PairwiseJudgement output format as forward()
            - Results are returned in the same order as input candidates

        Performance:
            - Sequential forward(): ~4 requests/second (250 RPM / 60s)
            - Async forward_async(): ~50 requests/second with max_concurrent=50
            - Speedup: ~12.5x for typical workloads

        Warning:
            This method materializes all results in memory. For very large
            candidate sets (>10,000 pairs), consider processing in batches:

            batch_size = 1000
            for i in range(0, len(all_candidates), batch_size):
                batch = all_candidates[i:i+batch_size]
                results = await module.forward_async(batch)
                # Process results...
        """
        # Initialize rate limiter
        rate_limiter = _RateLimiter(rpm_limit=rpm_limit, tpm_limit=tpm_limit)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        # Process all candidates concurrently
        tasks = [
            self._process_candidate_async(
                candidate=candidate,
                semaphore=semaphore,
                rate_limiter=rate_limiter,
                max_retries=max_retries,
            )
            for candidate in candidates
        ]

        # Gather results (maintains order)
        judgements = await asyncio.gather(*tasks)
        return list(judgements)

    async def _process_candidate_async(
        self,
        candidate: ERCandidate[SchemaT],
        semaphore: asyncio.Semaphore,
        rate_limiter: _RateLimiter,
        max_retries: int,
    ) -> PairwiseJudgement:
        """Process a single candidate with rate limiting and retry logic.

        Args:
            candidate: Entity pair to judge
            semaphore: Concurrency control semaphore
            rate_limiter: Token-aware rate limiter
            max_retries: Maximum retry attempts

        Returns:
            PairwiseJudgement for this candidate
        """
        async with semaphore:
            # Format entities as strings
            left_str = self._format_entity(candidate.left)
            right_str = self._format_entity(candidate.right)

            # Create prompt
            prompt = self.prompt_template.format(left=left_str, right=right_str)

            # Estimate token usage (rough approximation: 4 chars = 1 token)
            estimated_tokens = len(prompt) // 4 + 200  # Add buffer for response

            # Wait for rate limit clearance
            await rate_limiter.acquire(estimated_tokens=estimated_tokens)

            logger.debug(
                "Calling async LLM API for pair: %s vs %s",
                candidate.left.id,  # type: ignore[attr-defined]
                candidate.right.id,  # type: ignore[attr-defined]
            )

            # Call LLM API with retry logic
            response = await self._call_llm_with_retry(
                prompt=prompt,
                max_retries=max_retries,
            )

            # Record actual token usage (thread-safe)
            if response.usage:
                actual_tokens = response.usage.prompt_tokens + response.usage.completion_tokens
                await rate_limiter.record_usage_async(actual_tokens)

            # Extract score and reasoning from response
            content = response.choices[0].message.content or ""
            score = self._extract_score(content)
            reasoning = self._extract_reasoning(content)

            # Calculate cost
            cost_usd = self._calculate_cost(response)

            return PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=score,
                score_type="prob_llm",
                decision_step="llm_judgment_async",
                reasoning=reasoning,
                provenance={
                    "model": self.model,
                    "cost_usd": cost_usd,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": (
                        response.usage.completion_tokens if response.usage else 0
                    ),
                    "method": "async_batch",
                },
            )

    async def _call_llm_with_retry(
        self,
        prompt: str,
        max_retries: int,
    ) -> Any:
        """Call LLM API with exponential backoff retry for rate limits.

        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum retry attempts

        Returns:
            LLM API response

        Raises:
            RateLimitError: If max retries exceeded

        Note:
            Implements exponential backoff: 1s, 2s, 4s, 8s, ... up to 60s max
        """
        client = self._get_client()
        for attempt in range(max_retries):
            try:
                response = await client.acompletion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                return response
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds, max 60s
                    wait_time = min(2**attempt, 60)
                    logger.warning(
                        "Rate limit error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        wait_time,
                        e,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max retries (%d) exceeded for rate limit error", max_retries)
                    raise

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
        """Calculate API call cost in USD via LiteLLM's own pricing.

        Delegates to ``litellm.completion_cost`` so pricing stays honest for
        whatever model is actually used (no hardcoded table). Returns ``0.0`` if
        the model is unknown to LiteLLM or usage is missing — cost tracking is
        observability, so it must never raise or flake.

        Args:
            response: LLM API response (LiteLLM/OpenAI shape)

        Returns:
            Cost in USD (``0.0`` when unavailable)
        """
        try:
            return float(litellm.completion_cost(completion_response=response))
        except Exception:  # unknown model / missing usage — never raise/flake
            logger.warning("completion_cost unavailable for model %s; reporting 0.0", self.model)
            return 0.0

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


# Backward-compatible public alias. ``LLMJudge`` is the public name; existing
# imports of ``LLMJudgeModule`` keep working.
LLMJudgeModule = LLMJudge
