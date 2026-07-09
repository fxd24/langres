"""Tests for :class:`langres.core.usage.LLMUsage` — the OTel-GenAI token vector.

The critical invariant these tests pin is OTel's **SUBSET** semantics against
LiteLLM's actual normalization: ``input_tokens`` is the TOTAL input (already
including cache read + creation), and ``cache_read_input_tokens`` /
``cache_creation_input_tokens`` are subsets of it. LiteLLM normalizes Anthropic's
raw ``input_tokens`` (which EXCLUDES cache) up to the inclusive total, so we must
NOT add the cache fields ourselves — that would double-count every cached call.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import pytest

from langres.core.usage import LLMUsage


class _Resp:
    """A minimal completion-response stub carrying only ``usage``."""

    def __init__(self, usage: Any) -> None:
        self.usage = usage


# ---------------------------------------------------------------------------
# The typed model
# ---------------------------------------------------------------------------


class TestLLMUsageModel:
    def test_is_frozen(self) -> None:
        """The vector is immutable — assignment raises."""
        usage = LLMUsage(input_tokens=100, output_tokens=50, model="gpt-5-mini")
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError on frozen
            usage.input_tokens = 200  # type: ignore[misc]

    def test_is_json_serializable(self) -> None:
        usage = LLMUsage(input_tokens=100, output_tokens=50, model="gpt-5-mini")
        dumped = usage.model_dump()
        json.dumps(dumped)  # must not raise
        assert dumped["input_tokens"] == 100
        assert dumped["output_tokens"] == 50

    def test_defaults_are_zero_and_none(self) -> None:
        """Absent counts default to 0; provider defaults to None; model to ''."""
        usage = LLMUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_input_tokens == 0
        assert usage.cache_creation_input_tokens == 0
        assert usage.reasoning_tokens == 0
        assert usage.provider is None
        assert usage.model == ""

    def test_carries_the_full_otel_vector(self) -> None:
        """All five token fields plus provider + model are captured."""
        usage = LLMUsage(
            input_tokens=150,
            output_tokens=60,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=20,
            reasoning_tokens=25,
            provider="DeepInfra",
            model="openrouter/z-ai/glm-5.2",
        )
        assert usage.model_dump() == {
            "input_tokens": 150,
            "output_tokens": 60,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 20,
            "reasoning_tokens": 25,
            "provider": "DeepInfra",
            "model": "openrouter/z-ai/glm-5.2",
        }


# ---------------------------------------------------------------------------
# from_response — reads a litellm/openai-shaped completion response
# ---------------------------------------------------------------------------


class TestFromResponse:
    def test_reads_totals_and_records_model_and_provider(self) -> None:
        usage_obj = Mock()
        usage_obj.prompt_tokens = 100
        usage_obj.completion_tokens = 50
        usage_obj.prompt_tokens_details = None
        usage_obj.completion_tokens_details = None
        result = LLMUsage.from_response(_Resp(usage_obj), model="gpt-5-mini", provider="Together")
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.model == "gpt-5-mini"
        assert result.provider == "Together"

    def test_none_usage_yields_all_zeros_without_exploding(self) -> None:
        """``response.usage is None`` (older/other providers) => zeros, not errors."""
        result = LLMUsage.from_response(_Resp(None), model="gpt-5-mini")
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cache_read_input_tokens == 0
        assert result.reasoning_tokens == 0
        assert result.model == "gpt-5-mini"

    def test_missing_details_objects_are_zero_not_errors(self) -> None:
        """A bare Mock usage (auto-attribute details) must not raise; subsets => 0."""
        # Mock auto-creates ``prompt_tokens_details`` as a child Mock; the extractor
        # must treat a non-int subset field as 0, never crash.
        usage_obj = Mock()
        usage_obj.prompt_tokens = 100
        usage_obj.completion_tokens = 50
        result = LLMUsage.from_response(_Resp(usage_obj), model="m")
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cache_read_input_tokens == 0
        assert result.cache_creation_input_tokens == 0
        assert result.reasoning_tokens == 0

    def test_reads_cache_and_reasoning_subsets(self) -> None:
        usage_obj = Mock()
        usage_obj.prompt_tokens = 150  # already inclusive
        usage_obj.completion_tokens = 60
        usage_obj.prompt_tokens_details = Mock(cached_tokens=30, cache_creation_tokens=20)
        usage_obj.completion_tokens_details = Mock(reasoning_tokens=25)
        result = LLMUsage.from_response(_Resp(usage_obj), model="m")
        assert result.input_tokens == 150
        assert result.cache_read_input_tokens == 30
        assert result.cache_creation_input_tokens == 20
        assert result.reasoning_tokens == 25


# ---------------------------------------------------------------------------
# PIN: LiteLLM's Anthropic normalization (guards the double-count trap)
# ---------------------------------------------------------------------------


class TestLiteLLMAnthropicNormalizationPin:
    def test_prompt_tokens_is_inclusive_for_anthropic(self) -> None:
        """LiteLLM's Anthropic transform makes ``prompt_tokens`` the INCLUSIVE total.

        Raw Anthropic ``input_tokens`` EXCLUDES cache; LiteLLM adds
        ``cache_read_input_tokens`` + ``cache_creation_input_tokens`` into
        ``prompt_tokens`` (transformation.py). So ``LLMUsage.input_tokens`` reads
        that inclusive total directly and does NOT re-add the subsets. If LiteLLM
        ever stops normalizing (or starts double-counting), this test breaks loudly.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        raw = {
            "input_tokens": 100,  # EXCLUDES cache (raw Anthropic)
            "output_tokens": 50,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 20,
        }
        usage_obj = AnthropicConfig().calculate_usage(usage_object=raw, reasoning_content=None)
        # Sanity: litellm itself made prompt_tokens inclusive.
        assert usage_obj.prompt_tokens == 150

        result = LLMUsage.from_response(_Resp(usage_obj), model="claude-x")
        assert result.input_tokens == 150  # inclusive total, NOT 100 and NOT 200
        assert result.cache_read_input_tokens == 30  # subset
        assert result.cache_creation_input_tokens == 20  # subset
        assert result.output_tokens == 50

    def test_no_cache_case_has_zero_subsets(self) -> None:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        raw = {"input_tokens": 100, "output_tokens": 50}
        usage_obj = AnthropicConfig().calculate_usage(usage_object=raw, reasoning_content=None)
        result = LLMUsage.from_response(_Resp(usage_obj), model="claude-x")
        assert result.input_tokens == 100
        assert result.cache_read_input_tokens == 0
        assert result.cache_creation_input_tokens == 0

    def test_openai_style_prompt_tokens_details_is_inclusive(self) -> None:
        """A LiteLLM ``Usage`` built OpenAI-style (cached_tokens already inside
        prompt_tokens) still reports the inclusive total and the cache subset."""
        from litellm.types.utils import Usage

        # OpenAI: prompt_tokens (120) already INCLUDES cached_tokens (40).
        usage_obj = Usage(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            prompt_tokens_details={"cached_tokens": 40},
        )
        result = LLMUsage.from_response(_Resp(usage_obj), model="gpt-5-mini")
        assert result.input_tokens == 120  # inclusive, not 120+40
        assert result.cache_read_input_tokens == 40


# ---------------------------------------------------------------------------
# from_lm_usage — the DSPy get_lm_usage() shape (dict per LM, summed)
# ---------------------------------------------------------------------------


class TestFromLMUsage:
    def test_sums_flat_token_counts_across_lms(self) -> None:
        usage_by_lm = {
            "openrouter/a": {"prompt_tokens": 100, "completion_tokens": 40},
            "openrouter/b": {"prompt_tokens": 200, "completion_tokens": 60},
        }
        result = LLMUsage.from_lm_usage(usage_by_lm, model="openrouter/a")
        assert result.input_tokens == 300
        assert result.output_tokens == 100
        assert result.model == "openrouter/a"

    def test_reads_nested_details_dicts(self) -> None:
        """DSPy flattens ``*_details`` to plain dicts — the subsets are read from them."""
        usage_by_lm = {
            "m": {
                "prompt_tokens": 150,
                "completion_tokens": 60,
                "prompt_tokens_details": {"cached_tokens": 30, "cache_creation_tokens": 20},
                "completion_tokens_details": {"reasoning_tokens": 25},
            }
        }
        result = LLMUsage.from_lm_usage(usage_by_lm, model="m")
        assert result.input_tokens == 150
        assert result.cache_read_input_tokens == 30
        assert result.cache_creation_input_tokens == 20
        assert result.reasoning_tokens == 25

    def test_empty_usage_dict_yields_zeros(self) -> None:
        """DummyLM records no usage => ``get_lm_usage()`` is ``{}`` => all zeros."""
        result = LLMUsage.from_lm_usage({}, model="m")
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_none_usage_dict_is_safe(self) -> None:
        result = LLMUsage.from_lm_usage(None, model="m")  # type: ignore[arg-type]
        assert result.input_tokens == 0
