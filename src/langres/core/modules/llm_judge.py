"""LLMJudge: serializable LLM-based entity-matching Module.

This module uses OpenAI API (or compatible) for match judgments with natural
language reasoning and calibrated probability scores.

Supports both direct OpenAI client and LiteLLM for enhanced observability.
"""

import asyncio
import logging
import math
import re
import string
import time
from collections.abc import Callable, Iterator
from typing import Any, ClassVar, Literal

import litellm
from pydantic import BaseModel

# Type checking for litellm exceptions
try:
    from litellm import RateLimitError
except ImportError:
    RateLimitError = Exception

from langres.clients.openrouter import parse_openrouter_billing
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, SchemaT
from langres.core.registry import register
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl
from langres.core.runs import current_run
from langres.core.usage import LLMUsage

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


# Matches only the two record placeholders, so a single `re.sub` pass over the
# template can never rescan (and corrupt) an already-substituted record.
_PLACEHOLDER_RE = re.compile(r"\{(left|right)\}")


def render_default_prompt(entity_noun: str = "entity") -> str:
    """Render :data:`DEFAULT_PROMPT` for ``entity_noun``, leaving ``{left}``/``{right}``.

    Substitutes only the ``{entity_noun}`` placeholder so the result is still a
    template with ``{left}`` / ``{right}`` filled at judgement time.
    """
    return DEFAULT_PROMPT.replace("{entity_noun}", entity_noun)


class LLMParseError(ValueError):
    """Raised by :class:`LLMJudge` when ``on_parse_error='raise'`` and the
    configured ``response_parser`` could not parse a score from a response."""


class ParsedVerdict(BaseModel):
    """The output of an :class:`LLMJudge` ``response_parser``.

    A parser returns EITHER a ``decision`` (the binary yes/no family) XOR a
    ``score`` (the rating family) — never both:

    - ``decision`` is an explicit match verdict (``True``/``False``), for a
      binary judge that decides directly and does not rank. ``score`` stays
      ``None`` — a binary judge has no meaningful score, so a fabricated
      ``0.0``/``1.0`` would lie.
    - ``score`` is the match confidence in ``[0, 1]`` for a ranking judge;
      ``decision`` stays ``None``.

    **Both ``None`` signals a parse failure / abstain** — the parser could read
    neither a decision nor a score from the response. That ``None``/``None`` case
    is the load-bearing distinction the old silent-0.5 fallback erased: the judge
    routes it through its ``on_parse_error`` policy rather than emitting a real
    mid-confidence verdict. ``reasoning`` is the optional free-text justification
    the parser extracted (``None`` when absent).
    """

    decision: bool | None = None
    score: float | None = None
    reasoning: str | None = None


def parse_score_response(content: str) -> ParsedVerdict:
    """Default parser: read a ``Score: 0.XX`` line (the neutral :data:`DEFAULT_PROMPT`).

    On a match the score is clamped to ``[0, 1]`` and the reasoning is taken from
    a ``Reasoning:`` line (falling back to the full content). When there is no
    ``Score:`` line at all the verdict's ``score`` is ``None`` — a parse failure,
    NOT ``0.5`` — so a naive replication of a "Yes"/"No" paper prompt (which has
    no score line) is caught instead of silently scored as noise.
    """
    match = re.search(r"Score:\s*(\d+\.?\d*)", content, re.IGNORECASE)
    if match is None:
        return ParsedVerdict(score=None)
    score = max(0.0, min(1.0, float(match.group(1))))
    reasoning_match = re.search(r"Reasoning:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else content
    return ParsedVerdict(score=score, reasoning=reasoning)


def parse_binary_yes_no(content: str) -> ParsedVerdict:
    """Canonical parser for the published yes/no ER-prompt family.

    Yields a **decision** (``True``/``False``), not a score: a binary judge
    decides directly, so ``score`` stays ``None`` and downstream code
    (``predicted_match`` / ``classify_pairs`` / ``select_for_review``) sees a real
    verdict instead of a fabricated ``0.0``/``1.0``.

    This is the single source of truth for the yes/no contract: it deliberately
    mirrors the reference ``check_for_prediction`` from Peeters, Steiner & Bizer
    (*Entity Matching using Large Language Models*, arXiv 2310.11244) **exactly**
    — ``strip()`` → **delete** ``string.punctuation`` → ``lower()`` →
    ``"yes" in text`` maps to ``True`` (MATCH) else ``False``. ``langres.data.
    peeters.parse_binary_answer`` is a thin ``int`` adapter over this function;
    there is only one code path so the ``$0`` offline replay validates the exact
    parser the paid ``LLMJudge(response_parser=...)`` run will use.

    Fidelity to the reference beats cleverness, so the crudeness is preserved on
    purpose:

    - Punctuation is **deleted**, not replaced with a space, so intra-word
      punctuation collapses: ``"ye-s"``, ``"Y.E.S."``, ``"ye_s"`` all → ``"yes"``
      → MATCH. (``_`` is in ``string.punctuation``, so it is deleted too.)
    - It is a bare substring test with no negation handling: ``"Not yes"``
      contains ``"yes"`` and therefore MATCHES. Reproducing the paper's reported
      F1 requires the paper's parser, warts included.

    It is deliberately **total**: ``decision`` is always ``True``/``False`` and
    never ``None`` — absence of "yes" is a confident non-match (``decision=False``),
    never a parse failure/abstain. So ``on_parse_error`` never fires for this
    family and a long *paid* run cannot abort on one flaky response. Pass a custom
    parser that returns an all-``None`` :class:`ParsedVerdict` if you want strict
    abstention instead.
    """
    cleaned = content.strip().translate(str.maketrans("", "", string.punctuation)).lower()
    decision = "yes" in cleaned
    return ParsedVerdict(decision=decision, score=None, reasoning=content.strip() or None)


#: Below this combined yes+no probability mass, a first-token credence is refused
#: (``p_yes=None``) rather than manufactured from noise — see
#: :meth:`LLMJudge._confidence_from_response`.
_CONFIDENCE_MASS_FLOOR = 1e-9


def _normalize_answer_token(token: str) -> str:
    """Normalise a logprob token for yes/no matching (mirrors :func:`parse_binary_yes_no`).

    ``strip()`` → delete ``string.punctuation`` → ``lower()``. So ``" Yes"``,
    ``"YES"``, ``"Yes."`` all collapse to ``"yes"``. Unlike ``parse_binary_yes_no``
    (a substring test) the caller compares the result for *exact* equality to
    ``"yes"``/``"no"``, so ``"Yesterday"`` (→ ``"yesterday"``) is NOT counted.
    """
    return token.strip().translate(str.maketrans("", "", string.punctuation)).lower()


def default_record_serializer(entity: Any) -> str:
    """Default ``record_serializer``: today's ``entity.model_dump_json(indent=2)``.

    Kept as a standalone, importable callable so a caller can wrap or compose it;
    the behavior is unchanged for existing users (the full record, including
    ``id`` / ``source`` / ``embed_text``, is rendered as indented JSON).
    """
    return str(entity.model_dump_json(indent=2))


#: Named ``response_parser`` registry: these names are accepted anywhere a
#: parser is (``LLMJudge(response_parser=...)``, the verbs' / ``from_schema``'s
#: ``judge="prompt_llm"`` seam) and -- unlike a bare callable -- **serialize**
#: in :attr:`LLMJudge.config`, so a saved paper-replication judge reloads with
#: its parser intact (the model-identity design note's round-trip fix).
#: Adding an entry makes that name resolvable and serializable in-process.
RESPONSE_PARSERS: dict[str, Callable[[str], ParsedVerdict]] = {
    "score": parse_score_response,
    "binary_yes_no": parse_binary_yes_no,
}

#: Named ``record_serializer`` registry -- same contract as
#: :data:`RESPONSE_PARSERS` (names serialize in :attr:`LLMJudge.config`;
#: custom callables do not).
RECORD_SERIALIZERS: dict[str, Callable[[Any], str]] = {
    "json": default_record_serializer,
}


def _resolve_named(
    value: Callable[..., Any] | str | None,
    registry: dict[str, Callable[..., Any]],
    *,
    kind: str,
    default_name: str,
) -> tuple[Callable[..., Any], str | None]:
    """Resolve a parser/serializer given by name, callable, or ``None``.

    Returns ``(callable, name)`` where ``name`` is the registered name to
    serialize in :attr:`LLMJudge.config` -- ``None`` for a custom callable that
    is not in ``registry`` (documented as non-serializable: it reverts to the
    default on load).

    Raises:
        ValueError: For an unknown name, listing the registered ones.
    """
    if value is None:
        return registry[default_name], default_name
    if isinstance(value, str):
        resolved = registry.get(value)
        if resolved is None:
            raise ValueError(
                f"unknown {kind} name {value!r}; registered names: "
                f"{', '.join(sorted(registry))}. Pass a callable for a custom "
                f"{kind} (it will not serialize in config)."
            )
        return resolved, value
    name = next((n for n, fn in registry.items() if fn is value), None)
    return value, name


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

        Note:
            ``asyncio.sleep`` is **never** awaited while holding ``self._lock``.
            Each iteration locks only long enough to prune the 1-minute sliding
            windows and check the RPM/TPM limits; if the request fits it is
            recorded and we return, otherwise we compute the required wait,
            release the lock, sleep outside the critical section, and retry.
            Holding the lock across the sleep would block every other coroutine
            calling :meth:`acquire`/:meth:`record_usage_async`, stalling the
            whole pipeline on the first rate-limit hit.
        """
        while True:
            async with self._lock:
                now = time.time()
                one_minute_ago = now - 60.0

                # Remove old entries outside the 1-minute window
                self._request_times = [t for t in self._request_times if t > one_minute_ago]
                self._token_usage = [(t, c) for t, c in self._token_usage if t > one_minute_ago]

                # Compute how long we must wait (0.0 means "clear to proceed").
                sleep_time = 0.0

                # RPM sliding window
                if len(self._request_times) >= self.rpm_limit:
                    oldest_request = self._request_times[0]
                    rpm_sleep = 60.0 - (now - oldest_request) + 0.1  # Add 100ms buffer
                    logger.debug("RPM limit reached, sleeping for %.2fs", rpm_sleep)
                    sleep_time = max(sleep_time, rpm_sleep)

                # TPM sliding window
                current_tokens = sum(count for _, count in self._token_usage)
                if current_tokens + estimated_tokens > self.tpm_limit:
                    oldest_token_time = self._token_usage[0][0]
                    tpm_sleep = 60.0 - (now - oldest_token_time) + 0.1  # Add 100ms buffer
                    logger.debug(
                        "TPM limit reached (%d tokens), sleeping for %.2fs",
                        current_tokens,
                        tpm_sleep,
                    )
                    sleep_time = max(sleep_time, tpm_sleep)

                # Within both limits → reserve this request and return.
                if sleep_time == 0.0:
                    self._request_times.append(now)
                    return

            # Lock released: sleep outside the critical section, then retry.
            await asyncio.sleep(sleep_time)

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
    - Support for multiple LLM providers (OpenAI, Azure, etc.)
    - Serialization without persisting any secret
    - Optional Langfuse tracing for observability (opt-in: pass
      ``client=create_llm_client(Settings(), enable_langfuse=True)`` -- off by
      default since langfuse is a dev-only dependency, not part of the
      ``[llm]`` extra)

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
        Defaults to ``gpt-5-mini`` at ``temperature=0.0`` (deterministic, the ER
        convention; see :meth:`__init__`). Cost tracking prefers
        OpenRouter's *actual* billed cost (via usage accounting, parsed by
        :func:`~langres.clients.openrouter.parse_openrouter_billing`) and falls
        back to ``litellm.completion_cost`` (the pinned/table estimate) when the
        response carries no real cost. Provenance records the serving provider
        and whether the recorded cost was real (``cost_is_real``).

    Note:
        On an ``openrouter/...`` model, pass ``provider`` to pin which upstream
        provider serves the request (reproducible benchmark cost), e.g.
        ``provider={"order": ["DeepInfra"], "allow_fallbacks": False}`` or
        ``provider={"only": ["Together"]}``. It is sent as OpenRouter's
        ``extra_body["provider"]`` routing block. Off OpenRouter it is ignored.
    """

    # Registry key, mirrored as a class attribute so the Resolver's uniform
    # serialization helper can discover the type name (see resolver.py).
    type_name: ClassVar[str] = "llm_judge"

    def __init__(
        self,
        client: Any = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        prompt_template: str | None = None,
        entity_noun: str = "entity",
        provider: dict[str, Any] | None = None,
        *,
        system_prompt: str | None = None,
        response_parser: Callable[[str], ParsedVerdict] | str | None = None,
        record_serializer: Callable[[Any], str] | str | None = None,
        on_parse_error: Literal["abstain", "raise"] = "abstain",
        confidence: Literal["none", "logprob"] = "none",
    ):
        """Initialize LLMJudge.

        Args:
            client: Optional pre-configured LLM client (LiteLLM or OpenAI
                client). When ``None`` (the default), the client is lazily built
                from the environment via ``create_llm_client(Settings())`` on
                first use. Inject a client only as an escape hatch (e.g. tests
                or a custom client); use :meth:`from_env` for the happy path.
            model: Model name (e.g., "gpt-5-mini", "azure/gpt-5-mini")
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = random).
                Defaults to ``0.0`` — ER papers score at temperature 0 for
                reproducibility, and the sibling ``DSPyJudge`` already defaults
                to 0.0 (this makes 0.0 the house default for the judge family).
            prompt_template: Custom prompt template. Must contain both ``{left}``
                and ``{right}`` placeholders (the two records are substituted in
                at judgement time). Any other braces are preserved verbatim, so a
                paper's prompt carrying a literal JSON schema (e.g. ``{"match":
                true}``) works unchanged. Uses the neutral :data:`DEFAULT_PROMPT`
                (rendered for ``entity_noun``) when ``None``.
            entity_noun: Domain noun woven into the default prompt (e.g.
                "company", "product"). Ignored when ``prompt_template`` is given.
            provider: Optional OpenRouter provider-routing block, sent as
                ``extra_body["provider"]`` to pin which upstream provider serves
                the request (reproducible cost on a benchmark), e.g.
                ``{"order": ["DeepInfra"], "allow_fallbacks": False}`` or
                ``{"only": ["Together"]}``. ``None`` (the default) keeps
                OpenRouter's own routing. Ignored for non-``openrouter/`` models.
            system_prompt: Optional system message. When set, the request sends
                two messages (``system`` then ``user``); when ``None`` (default)
                a single ``user`` message is sent (byte-identical to before).
            response_parser: A registered parser *name* (see
                :data:`RESPONSE_PARSERS`: ``"score"`` -- the default
                ``Score:``-line parser -- or ``"binary_yes_no"`` for the
                published yes/no ER-prompt family) or a callable mapping the
                raw response text to a :class:`ParsedVerdict`. A verdict whose
                ``score`` is ``None`` is a parse failure and is routed through
                ``on_parse_error``. A registered name (or one of the registered
                callables) serializes in :attr:`config` and round-trips through
                ``Resolver.save``/``load``; an unregistered custom callable does
                NOT (it reverts to the default on load -- see :attr:`config`).
            record_serializer: A registered serializer *name* (see
                :data:`RECORD_SERIALIZERS`: ``"json"``, the default) or a
                callable ``(entity) -> str`` rendering each record into the
                prompt. Override to control exactly what the LLM sees (e.g.
                drop ``id``/``source``). Same serialization contract as
                ``response_parser``.
            on_parse_error: What to do on a parse failure. ``"abstain"`` (the
                default) emits a judgement flagged ``provenance["parse_error"] =
                True`` with ``score=0.0`` — the evaluator surfaces and warns on
                the count; ``"raise"`` raises :class:`LLMParseError` immediately.
                The default abstains because aborting a long paid run on a single
                flaky response is worse than a surfaced, counted abstention.
            confidence: First-token credence probe. ``"none"`` (default) is a
                no-op — the request and judgement are byte-identical to before,
                except a decision judge is tagged ``confidence_source=
                "unrequested"`` (it *could* expose logprobs; you did not ask).
                ``"logprob"`` requests ``logprobs`` + ``top_logprobs`` on every
                completion and records the first-token P(Yes) credence in each
                judgement's ``provenance`` (keys ``p_yes``,
                ``confidence_leaked_mass``, ``p_yes_is_bound`` — see
                :meth:`_confidence_from_response`) AND, when a usable ``p_yes`` was
                produced, promotes it onto the judgement itself: ``score = p_yes``
                (an honest continuous ranking signal), ``confidence = max(p_yes,
                1 - p_yes)``, ``confidence_source = "logprob"`` (see
                :meth:`_map_verdict`). Only meaningful for a binary yes/no protocol
                (e.g. ``response_parser=parse_binary_yes_no``). Serialized in
                :attr:`config` so a saved logprob judge reloads as one.

        Raises:
            ValueError: If temperature out of range, ``on_parse_error`` is not
                ``"abstain"``/``"raise"``, ``confidence`` is not
                ``"none"``/``"logprob"``, ``prompt_template`` lacks
                ``{left}``/``{right}``, or a ``response_parser``/
                ``record_serializer`` name is not registered.
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if on_parse_error not in ("abstain", "raise"):
            raise ValueError("on_parse_error must be 'abstain' or 'raise'")
        if confidence not in ("none", "logprob"):
            raise ValueError("confidence must be 'none' or 'logprob'")

        self.client = client
        self.model = model
        self.temperature = temperature
        self.entity_noun = entity_noun
        self.provider = provider
        self.system_prompt = system_prompt
        self.on_parse_error = on_parse_error
        self.confidence = confidence
        self._parse, self._parser_name = _resolve_named(
            response_parser, RESPONSE_PARSERS, kind="response_parser", default_name="score"
        )
        self._serialize, self._serializer_name = _resolve_named(
            record_serializer, RECORD_SERIALIZERS, kind="record_serializer", default_name="json"
        )
        self.prompt_template = (
            prompt_template if prompt_template else render_default_prompt(entity_noun)
        )
        if "{left}" not in self.prompt_template or "{right}" not in self.prompt_template:
            raise ValueError(
                "prompt_template must contain both {left} and {right} placeholders "
                "(the two records are substituted at judgement time)"
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
        """Pure, serializable construction config (never the client or secrets).

        Carries the ``system_prompt``, ``on_parse_error`` policy and the
        ``confidence`` credence mode so a saved paper-replication / logprob judge
        reloads with them — without ``confidence`` a ``save``/``load`` would
        silently revert a logprob judge to ``confidence="none"`` (PR #105 review).
        ``response_parser`` / ``record_serializer`` are serialized **by
        registered name** (see :data:`RESPONSE_PARSERS` /
        :data:`RECORD_SERIALIZERS`), so a judge built with ``"binary_yes_no"``
        round-trips. An *unregistered* custom callable serializes as ``None``
        (there is no no-pickle way to persist an arbitrary callable) — like the
        ``client``, it reverts to the default on
        :meth:`from_config`/``Resolver.load``; re-inject it after load if you
        need it.
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "prompt_template": self.prompt_template,
            "entity_noun": self.entity_noun,
            "provider": self.provider,
            "system_prompt": self.system_prompt,
            "on_parse_error": self.on_parse_error,
            "confidence": self.confidence,
            "response_parser": self._parser_name,
            "record_serializer": self._serializer_name,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "LLMJudge[SchemaT]":
        """Rebuild from :attr:`config` via the lazy-client path (client from env).

        Older artifacts without ``system_prompt`` / ``on_parse_error`` /
        ``confidence`` / ``response_parser`` / ``record_serializer`` fall back
        to the constructor defaults (``None`` / ``"abstain"`` / ``"none"`` /
        the ``"score"`` parser / the ``"json"`` serializer).
        """
        provider = config.get("provider")
        return cls(
            client=None,
            model=str(config["model"]),
            temperature=float(config["temperature"]),  # type: ignore[arg-type]
            prompt_template=str(config["prompt_template"]),
            entity_noun=str(config["entity_noun"]),
            provider=provider,  # type: ignore[arg-type]
            system_prompt=config.get("system_prompt"),  # type: ignore[arg-type]
            on_parse_error=config.get("on_parse_error", "abstain"),  # type: ignore[arg-type]
            confidence=config.get("confidence", "none"),  # type: ignore[arg-type]
            # Serialized by registered name; None (a custom callable, or a
            # pre-registry artifact) resolves to the default.
            response_parser=config.get("response_parser"),  # type: ignore[arg-type]
            record_serializer=config.get("record_serializer"),  # type: ignore[arg-type]
        )

    def _completion_kwargs(self) -> dict[str, Any]:
        """Per-call OpenRouter extras: usage accounting + optional provider pin.

        On an ``openrouter/...`` model, returns
        ``{"extra_body": {"usage": {"include": True}, "provider": ...}}`` so the
        response carries the real billed cost (and, when :attr:`provider` is set,
        pins routing for a reproducible run). Off OpenRouter it returns ``{}`` —
        usage accounting and provider routing are OpenRouter features and would
        400 on OpenAI/Azure — so those calls are made exactly as before.
        """
        if not self.model.startswith("openrouter/"):
            return {}
        extra_body: dict[str, Any] = {"usage": {"include": True}}
        if self.provider is not None:
            extra_body["provider"] = self.provider
        return {"extra_body": extra_body}

    def _logprobs_kwargs(self) -> dict[str, Any]:
        """Top-level ``logprobs``/``top_logprobs`` request when the credence probe is on.

        Returns ``{"logprobs": True, "top_logprobs": 20}`` when
        ``confidence == "logprob"``, else ``{}``. These are **standard** OpenAI
        chat-completion params, so this is merged at the completion call sites
        directly and deliberately kept **out** of :meth:`_completion_kwargs` —
        that method early-returns ``{}`` for any non-``openrouter/`` model, so
        folding logprobs into it would silently never request them on plain
        OpenAI/Azure. ``top_logprobs=20`` is the API maximum, widening the
        two-way yes/no subspace we can attribute mass to (less leaked mass).
        """
        if self.confidence != "logprob":
            return {}
        return {"logprobs": True, "top_logprobs": 20}

    def _confidence_from_response(self, response: Any) -> dict[str, Any] | None:
        """First-token P(Yes) credence from logprobs, or ``None`` when unavailable.

        Off (``confidence != "logprob"``) or when the response carries no
        logprobs, returns ``None`` — no confidence, no crash, provenance
        unchanged. Otherwise reads ``choices[0].logprobs.content``, takes the
        first non-whitespace generated token, and over ITS ``top_logprobs`` sums
        the probability mass (``exp(logprob)``) on tokens normalising to ``"yes"``
        vs ``"no"`` (:func:`_normalize_answer_token`). Returns a provenance
        fragment:

        - ``p_yes``: ``yes_mass / (yes_mass + no_mass)`` — renormalised over the
          yes/no **two-way subspace only**. ``None`` when the combined mass is
          below :data:`_CONFIDENCE_MASS_FLOOR` (don't manufacture credence from
          noise).
        - ``confidence_leaked_mass``: ``1 - (yes_mass + no_mass)`` — the mass that
          went to NEITHER yes nor no (other tokens in the top-k, plus everything
          below the top-k cutoff). **Never** normalised away; it is the honest
          record of how much of the first-token distribution this two-way credence
          ignores.
        - ``p_yes_is_bound``: ``True`` when exactly one side has zero mass, so
          ``p_yes`` (``0.0`` or ``1.0``) is a one-sided **bound**, not a point —
          the other side's true mass is merely below the top-k cutoff, not zero.
        """
        if self.confidence != "logprob":
            return None
        logprobs = getattr(response.choices[0], "logprobs", None)
        content = getattr(logprobs, "content", None) if logprobs is not None else None
        if not content:
            return None
        first = next((tok for tok in content if tok.token.strip()), None)
        if first is None:
            return None
        yes_mass = 0.0
        no_mass = 0.0
        for alt in getattr(first, "top_logprobs", None) or []:
            norm = _normalize_answer_token(alt.token)
            if norm == "yes":
                yes_mass += math.exp(alt.logprob)
            elif norm == "no":
                no_mass += math.exp(alt.logprob)
        two_way = yes_mass + no_mass
        leaked = 1.0 - two_way
        # NOTE: this ``confidence_leaked_mass`` provenance key is persisted by the
        # Peeters harness's ``_row_from_judgement`` under the column name
        # ``leaked_mass`` (renamed there) -- a grep for either name finds the other.
        if two_way < _CONFIDENCE_MASS_FLOOR:
            return {"p_yes": None, "confidence_leaked_mass": leaked, "p_yes_is_bound": False}
        return {
            "p_yes": yes_mass / two_way,
            "confidence_leaked_mass": leaked,
            "p_yes_is_bound": yes_mass == 0.0 or no_mass == 0.0,
        }

    def _billing(self, response: Any) -> tuple[float, str | None, bool]:
        """Return ``(cost_usd, serving_provider, cost_is_real)`` for a response.

        Prefers OpenRouter's actual billed cost (usage accounting); falls back to
        the pinned per-token estimate via :meth:`_calculate_cost` when the
        response carries no real cost (offline, non-OpenRouter, or accounting
        off). ``cost_is_real`` records which source was used.
        """
        real_cost, provider = parse_openrouter_billing(response)
        if real_cost is not None:
            return real_cost, provider, True
        return self._calculate_cost(response), provider, False

    def _run_correlation_metadata(
        self, client: Any, left_id: Any, right_id: Any, decision_step: str
    ) -> dict[str, Any] | None:
        """Litellm ``metadata`` correlating this call to the active run, or ``None``.

        S5 run correlation: when a ``capture_run`` is open (``current_run`` set)
        AND the client is the litellm module (identity check), return the
        ``metadata`` payload -- the run's ``attempt_id``, the pair ids, and the
        ``decision_step`` -- so a Langfuse/OTel trace joins the ``RunRecord`` and
        ``JudgementLog`` on ``langres_attempt_id``. Returns ``None`` otherwise
        (no open run, or a user-supplied direct client that would 400 on an
        unknown ``metadata`` kwarg), so the completion call stays byte-identical
        to before when no run is active. Shared by :meth:`forward` (sync) and
        :meth:`_call_llm_with_retry` (async) so the two paths cannot drift.
        """
        attempt_id = current_run.get()
        if attempt_id is not None and client is litellm:
            return {
                "langres_attempt_id": attempt_id,
                "left_id": left_id,
                "right_id": right_id,
                "decision_step": decision_step,
            }
        return None

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
            # Render each record via the injectable serializer and substitute the
            # two into the template (literal braces preserved -- see _render_prompt).
            left_str = self._serialize(candidate.left)
            right_str = self._serialize(candidate.right)
            prompt = self._render_prompt(left_str, right_str)

            # Call LLM API
            logger.debug(
                "Calling LLM API for pair: %s vs %s",
                candidate.left.id,  # type: ignore[attr-defined]
                candidate.right.id,  # type: ignore[attr-defined]
            )

            # Call client (works for both LiteLLM and OpenAI)
            client = self._get_client()
            completion_kwargs = self._completion_kwargs()
            # Standard top-level logprobs request (credence probe). Merged here,
            # NOT inside _completion_kwargs (which returns {} off openrouter/ and
            # would drop logprobs on plain OpenAI). {} when confidence is off.
            completion_kwargs.update(self._logprobs_kwargs())
            # Run correlation (S5): stamp the active tracking run + pair identity
            # into litellm's ``metadata`` param (shared with the async path via
            # ``_run_correlation_metadata``). ``None`` off a run keeps the call
            # byte-identical -- no ``metadata`` key is added.
            metadata = self._run_correlation_metadata(
                client,
                candidate.left.id,  # type: ignore[attr-defined]
                candidate.right.id,  # type: ignore[attr-defined]
                "llm_judgment",
            )
            if metadata is not None:
                completion_kwargs["metadata"] = metadata
            response = client.completion(
                model=self.model,
                messages=self._messages(prompt),
                temperature=self.temperature,
                **completion_kwargs,
            )

            # Parse the verdict + fold in the optional first-token logprob
            # credence; an abstain routes through ``on_parse_error`` (never a
            # fabricated verdict). See :meth:`_map_verdict`.
            content = response.choices[0].message.content or ""
            confidence_fragment = self._confidence_from_response(response)
            decision, score, confidence, confidence_source, reasoning, parse_error = (
                self._map_verdict(content, confidence_fragment)
            )

            # Cost: OpenRouter's real billed cost when available, else the estimate.
            cost_usd, provider, cost_is_real = self._billing(response)
            usage = LLMUsage.from_response(response, model=self.model, provider=provider)

            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                decision=decision,
                score=score,
                score_type="prob_llm",
                confidence=confidence,
                confidence_source=confidence_source,
                decision_step="llm_judgment",
                reasoning=reasoning,
                provenance=self._build_provenance(
                    cost_usd,
                    cost_is_real,
                    provider,
                    usage,
                    parse_error=parse_error,
                    confidence=confidence_fragment,
                ),
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
            # Render each record via the injectable serializer and substitute.
            left_str = self._serialize(candidate.left)
            right_str = self._serialize(candidate.right)

            # Create prompt (literal braces preserved -- see _render_prompt)
            prompt = self._render_prompt(left_str, right_str)

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
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
            )

            # Record actual token usage (thread-safe)
            if response.usage:
                actual_tokens = response.usage.prompt_tokens + response.usage.completion_tokens
                await rate_limiter.record_usage_async(actual_tokens)

            # Parse the verdict + fold in the optional first-token logprob
            # credence; an abstain routes through ``on_parse_error`` (never a
            # fabricated verdict). See :meth:`_map_verdict`.
            content = response.choices[0].message.content or ""
            confidence_fragment = self._confidence_from_response(response)
            decision, score, confidence, confidence_source, reasoning, parse_error = (
                self._map_verdict(content, confidence_fragment)
            )

            # Cost: OpenRouter's real billed cost when available, else the estimate.
            cost_usd, provider, cost_is_real = self._billing(response)
            usage = LLMUsage.from_response(response, model=self.model, provider=provider)

            return PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                decision=decision,
                score=score,
                score_type="prob_llm",
                confidence=confidence,
                confidence_source=confidence_source,
                decision_step="llm_judgment_async",
                reasoning=reasoning,
                provenance=self._build_provenance(
                    cost_usd,
                    cost_is_real,
                    provider,
                    usage,
                    parse_error=parse_error,
                    method="async_batch",
                    confidence=confidence_fragment,
                ),
            )

    async def _call_llm_with_retry(
        self,
        prompt: str,
        max_retries: int,
        left_id: Any,
        right_id: Any,
    ) -> Any:
        """Call LLM API with exponential backoff retry for rate limits.

        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum retry attempts
            left_id: Left record id, for the run-correlation ``metadata``
            right_id: Right record id, for the run-correlation ``metadata``

        Returns:
            LLM API response

        Raises:
            RateLimitError: If max retries exceeded

        Note:
            Implements exponential backoff: 1s, 2s, 4s, 8s, ... up to 60s max
        """
        client = self._get_client()
        completion_kwargs = self._completion_kwargs()
        # Standard top-level logprobs request (credence probe) — mirror the sync
        # merge. Kept out of _completion_kwargs so it also reaches plain OpenAI.
        completion_kwargs.update(self._logprobs_kwargs())
        # Run correlation (S5): mirror forward()'s litellm ``metadata`` injection
        # on the async path -- async judging inside ``capture_run`` would otherwise
        # lose trace correlation. ``None`` off a run keeps the call byte-identical.
        metadata = self._run_correlation_metadata(client, left_id, right_id, "llm_judgment_async")
        if metadata is not None:
            completion_kwargs["metadata"] = metadata
        for attempt in range(max_retries):
            try:
                response = await client.acompletion(
                    model=self.model,
                    messages=self._messages(prompt),
                    temperature=self.temperature,
                    **completion_kwargs,
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

    def _render_prompt(self, left_str: str, right_str: str) -> str:
        """Substitute the two records into the template in a single pass.

        Scans the *template* once for ``{left}`` / ``{right}`` rather than using
        ``str.format`` so that any *other* braces in the template — a paper's
        JSON output schema, for instance — are preserved verbatim instead of
        raising ``KeyError`` / ``IndexError``. Both placeholders are guaranteed
        present (validated at construction).

        Single-pass matters: chained ``str.replace`` calls would rescan the
        already-inserted left record, so a record whose text happens to contain
        the literal ``{right}`` would have that token overwritten with the right
        record. One pass substitutes template placeholders only, never data.
        """
        values = {"left": left_str, "right": right_str}
        return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], self.prompt_template)

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        """Build the chat messages, prepending ``system_prompt`` when configured."""
        if self.system_prompt is not None:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        return [{"role": "user", "content": prompt}]

    def _map_verdict(
        self, content: str, confidence_fragment: dict[str, Any] | None
    ) -> tuple[
        bool | None,
        float | None,
        float | None,
        Literal["none", "unrequested", "logprob"],
        str | None,
        bool,
    ]:
        """Map a parsed response (+ optional logprob credence) to judgement fields.

        Returns ``(decision, score, confidence, confidence_source, reasoning,
        parse_error)`` for the :class:`PairwiseJudgement`:

        - **Decision family** (:func:`parse_binary_yes_no`): ``decision`` is the
          verdict and ``score`` is ``None`` — a binary judge does not rank, so a
          fabricated ``0.0``/``1.0`` would lie (the whole point of this change).
        - **Ranking family** (:func:`parse_score_response`): ``score`` is the
          float and ``decision`` is ``None``.
        - **Logprob credence on** (``confidence="logprob"`` with usable first-token
          yes/no mass): ``score`` becomes the honest continuous ``p_yes`` ranking
          signal, ``confidence`` is ``max(p_yes, 1 - p_yes)`` (the model's credence
          in its OWN answer — what the roc_auc-0.95 probe gated on) and
          ``confidence_source="logprob"``.
        - **Abstain** (parser read neither a decision nor a score): routed through
          ``on_parse_error`` — ``"raise"`` raises :class:`LLMParseError`; the
          default ``"abstain"`` returns all-``None`` (``is_abstain``) flagged
          ``parse_error=True``, never a fabricated verdict.

        ``confidence_source`` keeps three distinct meanings: ``"logprob"`` = an
        earned first-token credence; ``"unrequested"`` = a decision judge that
        *could* expose logprobs but was not asked (``confidence="none"``);
        ``"none"`` = no confidence notion here (the ranking-score parser, or a
        logprob run whose first token carried no usable yes/no mass).
        """
        parsed = self._parse(content)
        if parsed.decision is None and parsed.score is None:
            # Abstain: the parser read no verdict at all -> on_parse_error policy.
            if self.on_parse_error == "raise":
                raise LLMParseError(
                    f"response_parser could not parse a verdict from response: {content!r}"
                )
            logger.warning(
                "response_parser could not parse a verdict; abstaining "
                "(decision=None, score=None, parse_error flagged). Raw response: %r",
                content,
            )
            return None, None, None, "none", parsed.reasoning, True

        p_yes = confidence_fragment.get("p_yes") if confidence_fragment is not None else None
        if p_yes is not None and parsed.decision is not None:
            # Earned credence, only for a binary DECISION judge: p_yes is an honest
            # continuous ranking signal and the credence in the model's OWN answer
            # is max(p_yes, 1 - p_yes). A rating parser (decision=None, score=<float>)
            # must NOT have its parsed rating clobbered by first-token yes/no mass,
            # which is meaningless for a "rate 0-1" response.
            confidence = max(p_yes, 1.0 - p_yes)
            return parsed.decision, p_yes, confidence, "logprob", parsed.reasoning, False

        source: Literal["none", "unrequested"]
        if self.confidence == "none" and parsed.decision is not None:
            # A decision judge that could expose logprobs but was not asked.
            source = "unrequested"
        else:
            source = "none"
        return parsed.decision, parsed.score, None, source, parsed.reasoning, False

    def _build_provenance(
        self,
        cost_usd: float,
        cost_is_real: bool,
        provider: str | None,
        usage: LLMUsage,
        *,
        parse_error: bool,
        method: str | None = None,
        confidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the judgement provenance (shared by the sync + async paths).

        Keeps the legacy ``prompt_tokens`` / ``completion_tokens`` keys (readers
        such as ``JudgementLog``, ``bootstrap.labelers`` and
        ``openrouter.make_token_cost_track`` depend on them) alongside the new
        typed ``usage`` vector, and flags ``parse_error`` only when it fired.
        ``confidence`` (from :meth:`_confidence_from_response`) is merged in when
        the credence probe produced one — adding ``p_yes`` /
        ``confidence_leaked_mass`` / ``p_yes_is_bound`` — and is ``None`` (a no-op)
        when the probe is off or logprobs were absent.
        """
        provenance: dict[str, Any] = {
            "model": self.model,
            "cost_usd": cost_usd,
            "cost_is_real": cost_is_real,
            "provider": provider,
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "usage": usage.model_dump(),
        }
        if parse_error:
            provenance["parse_error"] = True
        if method is not None:
            provenance["method"] = method
        if confidence is not None:
            provenance.update(confidence)
        return provenance

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
