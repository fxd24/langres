"""``LLMUsage``: the typed token-usage vector for one LLM call (the fact layer).

Judges (``LLMJudge``, ``DSPyJudge``, ``SelectJudge``) previously recorded only
``prompt_tokens`` / ``completion_tokens`` into ``PairwiseJudgement.provenance``
and discarded everything else LiteLLM hands us. This module captures the full
vector, in the **OpenTelemetry GenAI** vocabulary flattened to snake_case, with
OTel's **SUBSET** semantics:

- ``input_tokens`` — TOTAL input, INCLUDING cache read + cache creation.
- ``output_tokens`` — TOTAL output, INCLUDING reasoning.
- ``cache_read_input_tokens`` — subset of ``input_tokens``.
- ``cache_creation_input_tokens`` — subset of ``input_tokens``.
- ``reasoning_tokens`` — subset of ``output_tokens``.

**The normalization trap (verified, pinned by tests):** OpenAI's
``prompt_tokens`` already *includes* ``cached_tokens``; Anthropic's raw
``input_tokens`` *excludes* cache. LiteLLM normalizes at its own boundary —
``AnthropicConfig.calculate_usage`` does ``prompt_tokens += cache_creation`` and
``+= cache_read`` (litellm ``llms/anthropic/chat/transformation.py``), so by the
time we see ``usage.prompt_tokens`` it is ALREADY the inclusive total for both
providers. We therefore read it straight through and must NOT add the cache
fields ourselves — doing so would double-count every cached call. (LiteLLM's own
``generic_cost_per_token`` confirms this: it *subtracts* the cache subsets from
``prompt_tokens`` to recover the full-rate text tokens.) The subsets are read
from ``prompt_tokens_details`` / ``completion_tokens_details``.

**Import-safety:** this module depends only on ``pydantic`` (no litellm / dspy /
torch), so a future ``PriceBook`` in the tracking layer can import ``LLMUsage``
to price a call without pulling any of core's heavy optional dependencies.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


def _read(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict OR an object, returning ``None`` if absent.

    Handles all three usage shapes we see: a LiteLLM ``Usage`` object (attribute
    access), a DSPy ``get_lm_usage()`` entry (plain dict), and a test ``Mock``
    (auto-attribute) — for the last, a missing detail resolves to a child Mock
    that :func:`_as_int` then coerces to ``0``.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _as_int(value: Any) -> int:
    """Coerce ``value`` to ``int``; anything non-numeric (``None``, a ``Mock``,
    a missing ``*_details`` field) becomes ``0`` rather than raising.

    Token capture is observability — it must never make a judge flake — so an
    absent or unexpected field is ``0``, not an error.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract(usage: Any) -> tuple[int, int, int, int, int]:
    """Pull the OTel subset vector from one usage-like object/dict.

    Returns ``(input_tokens, output_tokens, cache_read, cache_creation,
    reasoning)``. ``usage is None`` (older/other providers) yields all zeros.
    ``input_tokens`` / ``output_tokens`` are read as the inclusive totals
    LiteLLM already normalized (see module docstring); the subsets come from the
    ``*_details`` sub-objects.
    """
    if usage is None:
        return (0, 0, 0, 0, 0)
    input_tokens = _as_int(_read(usage, "prompt_tokens"))
    output_tokens = _as_int(_read(usage, "completion_tokens"))
    prompt_details = _read(usage, "prompt_tokens_details")
    cache_read = _as_int(_read(prompt_details, "cached_tokens"))
    cache_creation = _as_int(_read(prompt_details, "cache_creation_tokens"))
    completion_details = _read(usage, "completion_tokens_details")
    reasoning = _as_int(_read(completion_details, "reasoning_tokens"))
    return (input_tokens, output_tokens, cache_read, cache_creation, reasoning)


class LLMUsage(BaseModel):
    """Frozen OTel-GenAI token-usage vector for one LLM call, plus its serving id.

    Stored JSON-serialized (``model_dump()``) under ``provenance["usage"]`` by the
    judges. ``input_tokens`` / ``output_tokens`` are the inclusive totals (equal
    to the legacy ``prompt_tokens`` / ``completion_tokens`` provenance keys, which
    are kept alongside for the readers that still consume them); the remaining
    fields are the cache/reasoning subsets described in the module docstring.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    provider: str | None = None
    model: str = ""

    @classmethod
    def from_response(
        cls, response: Any, *, model: str, provider: str | None = None
    ) -> "LLMUsage":
        """Build the vector from a LiteLLM/OpenAI-shaped completion ``response``.

        Reads ``response.usage`` (``None``-safe). ``provider`` is the serving
        provider the caller already resolved (e.g. OpenRouter's ``provider``
        field via ``parse_openrouter_billing``); ``model`` is the routing id.
        """
        input_tokens, output_tokens, cache_read, cache_creation, reasoning = _extract(
            getattr(response, "usage", None)
        )
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            reasoning_tokens=reasoning,
            provider=provider,
            model=model,
        )

    @classmethod
    def from_lm_usage(
        cls, usage_by_lm: dict[str, Any] | None, *, model: str, provider: str | None = None
    ) -> "LLMUsage":
        """Build the vector from DSPy's ``prediction.get_lm_usage()`` mapping.

        DSPy returns ``{lm_name: usage_dict}`` where each ``usage_dict`` is the
        flattened LiteLLM usage (``prompt_tokens`` / ``completion_tokens`` plus,
        when present, ``prompt_tokens_details`` / ``completion_tokens_details`` as
        plain dicts). The vector is summed across every LM in the mapping; a
        ``{}`` (DummyLM records none) or ``None`` yields all zeros.
        """
        totals = [0, 0, 0, 0, 0]
        for entry in (usage_by_lm or {}).values():
            for index, value in enumerate(_extract(entry)):
                totals[index] += value
        return cls(
            input_tokens=totals[0],
            output_tokens=totals[1],
            cache_read_input_tokens=totals[2],
            cache_creation_input_tokens=totals[3],
            reasoning_tokens=totals[4],
            provider=provider,
            model=model,
        )
