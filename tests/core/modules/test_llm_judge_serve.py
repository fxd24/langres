"""PR-E serve-path tests for LLMMatcher: ``api_base`` threading, backend routing,
``model_ref`` round-trip, and the in-process backend's shared logprob→score step.

All $0: no API key, no network, no torch. ``api_base`` is asserted at BOTH litellm
call sites (sync ``completion`` + async ``acompletion``); the in-process backend is
exercised through an injected fake so the *shared* first-token-logprob→score
computation (identical to the served path) is verified without downloading a model.
"""

from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from langres.core.matchers.llm_judge import (
    LLMMatcher,
    _default_backend_kind,
    parse_binary_yes_no,
)
from langres.core.matchers.model_ref import ModelRef, normalize_model_ref
from langres.core.models import CompanySchema, ERCandidate


def _candidate() -> ERCandidate[CompanySchema]:
    return ERCandidate(
        left=CompanySchema(id="a1", name="Acme Corp"),
        right=CompanySchema(id="b1", name="Acme Corporation"),
        blocker_name="test",
    )


def _top(token: str, prob: float) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=math.log(prob), bytes=None)


def _logprob_response(
    first_token_alts: list[tuple[str, float]],
    *,
    message: str = "Yes",
) -> SimpleNamespace:
    """A litellm/OpenAI-shaped response with one generated token + its top_logprobs.

    Mirrors what the served path returns AND what ``TransformersBackend.complete``
    reproduces, so the identical ``_confidence_from_response`` step consumes both.
    """
    content = [
        SimpleNamespace(
            token=message,
            logprob=0.0,
            bytes=None,
            top_logprobs=[_top(t, p) for t, p in first_token_alts],
        )
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=message),
                logprobs=SimpleNamespace(content=content),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=42,
            completion_tokens=1,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        _hidden_params={},
    )


class _FakeClient:
    """Records the kwargs of the last ``completion`` call; returns a fixed response."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response
        self.last_kwargs: dict[str, Any] | None = None

    def completion(self, **kwargs: Any) -> SimpleNamespace:
        self.last_kwargs = kwargs
        return self._response


class _FakeBackend:
    """Stand-in for ``TransformersBackend``: records calls, returns a fixed response."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, messages: list[dict[str, str]], *, temperature: float, want_logprobs: bool
    ) -> SimpleNamespace:
        self.calls.append(
            {"messages": messages, "temperature": temperature, "want_logprobs": want_logprobs}
        )
        return self._response


# --------------------------------------------------------------------------- #
# api_base threading (served vLLM/Ollama/OpenAI-compatible endpoints).
# --------------------------------------------------------------------------- #


def test_api_base_reaches_the_sync_completion_call() -> None:
    client = _FakeClient(_logprob_response([("Yes", 0.9), ("No", 0.05)]))
    judge = LLMMatcher[CompanySchema](
        client=client,
        model="my-ft-model",
        api_base="http://localhost:8000/v1",
    )
    list(judge.forward(iter([_candidate()])))
    assert client.last_kwargs is not None
    assert client.last_kwargs["api_base"] == "http://localhost:8000/v1"
    # And the served model id is passed as-is.
    assert client.last_kwargs["model"] == "my-ft-model"


def test_api_base_absent_from_kwargs_when_unset() -> None:
    client = _FakeClient(_logprob_response([("Yes", 0.9), ("No", 0.05)]))
    judge = LLMMatcher[CompanySchema](client=client, model="gpt-4o-mini")
    list(judge.forward(iter([_candidate()])))
    assert client.last_kwargs is not None
    assert "api_base" not in client.last_kwargs


def test_api_base_reaches_the_async_acompletion_call() -> None:
    resp = _logprob_response([("Yes", 0.9), ("No", 0.05)])
    client = SimpleNamespace(acompletion=AsyncMock(return_value=resp))
    judge = LLMMatcher[CompanySchema](
        client=client,
        model="my-ft-model",
        api_base="http://localhost:8000/v1",
    )
    asyncio.run(judge.forward_async([_candidate()], max_concurrent=1))
    _args, kwargs = client.acompletion.call_args
    assert kwargs["api_base"] == "http://localhost:8000/v1"


def test_api_base_absent_from_async_kwargs_when_unset() -> None:
    resp = _logprob_response([("Yes", 0.9), ("No", 0.05)])
    client = SimpleNamespace(acompletion=AsyncMock(return_value=resp))
    judge = LLMMatcher[CompanySchema](client=client, model="gpt-4o-mini")
    asyncio.run(judge.forward_async([_candidate()], max_concurrent=1))
    _args, kwargs = client.acompletion.call_args
    assert "api_base" not in kwargs


# --------------------------------------------------------------------------- #
# Weightless config round-trip: api_base + model_ref.
# --------------------------------------------------------------------------- #


def test_api_base_round_trips_through_config() -> None:
    original = LLMMatcher[CompanySchema](
        client=object(),
        model="my-ft-model",
        api_base="http://localhost:8000/v1",
    )
    assert original.config["api_base"] == "http://localhost:8000/v1"
    rebuilt = LLMMatcher.from_config(original.config)
    assert rebuilt.api_base == "http://localhost:8000/v1"
    assert rebuilt.config == original.config


def test_plain_string_model_config_is_byte_identical() -> None:
    # A base-only model stays a plain string in config (old artifacts unchanged).
    judge = LLMMatcher[CompanySchema](client=object(), model="gpt-5-mini")
    assert judge.config["model"] == "gpt-5-mini"
    json.dumps(judge.config)  # weightless: reference strings only


def test_base_adapter_model_ref_round_trips_weightlessly() -> None:
    original = LLMMatcher[CompanySchema](
        client=object(),
        model={"base": "meta-llama/Llama-3.1-8B", "adapter": "your-org/lora"},
    )
    # self.model stays the base id string for every existing consumer.
    assert original.model == "meta-llama/Llama-3.1-8B"
    assert original.model_ref == ModelRef(base="meta-llama/Llama-3.1-8B", adapter="your-org/lora")
    # config["model"] widens to a dict of reference strings (no weights).
    assert original.config["model"] == {
        "base": "meta-llama/Llama-3.1-8B",
        "adapter": "your-org/lora",
    }
    json.dumps(original.config)

    rebuilt = LLMMatcher.from_config(original.config)
    assert rebuilt.model_ref == original.model_ref
    assert rebuilt.config == original.config


# --------------------------------------------------------------------------- #
# Backend routing decision (served API vs in-process transformers).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model", "api_base", "expected"),
    [
        ("gpt-5-mini", None, "litellm"),  # bare OpenAI-style name
        ("openrouter/openai/gpt-4o-mini", None, "litellm"),  # provider prefix
        ("azure/gpt-4", None, "litellm"),
        ("huggingface/org/model", None, "litellm"),  # litellm HF *inference* route
        ("your-org/your-ft-model", None, "transformers"),  # HF Hub id -> in-process
        ("my-ft-model", "http://localhost:8000/v1", "litellm"),  # served endpoint
        ("your-org/your-ft-model", "http://localhost:8000/v1", "litellm"),  # api_base wins
    ],
)
def test_default_backend_kind(model: str, api_base: str | None, expected: str) -> None:
    assert _default_backend_kind(normalize_model_ref(model), api_base) == expected


def test_local_directory_routes_in_process(tmp_path: Any) -> None:
    ref = normalize_model_ref(str(tmp_path))  # an existing dir
    assert _default_backend_kind(ref, None) == "transformers"


def test_base_adapter_always_routes_in_process_even_with_api_base() -> None:
    # An unmerged adapter can only be assembled locally -> in-process wins.
    ref = ModelRef(base="org/base", adapter="org/adapter")
    assert _default_backend_kind(ref, "http://localhost:8000/v1") == "transformers"


def test_matcher_resolves_backend_kind_at_construction() -> None:
    assert LLMMatcher(client=object(), model="gpt-5-mini")._backend_kind == "litellm"
    assert LLMMatcher(client=object(), model="your-org/ft")._backend_kind == "transformers"


# --------------------------------------------------------------------------- #
# In-process backend: the SAME logprob→score step as the served path.
# --------------------------------------------------------------------------- #


def test_inprocess_backend_score_is_the_shared_pyes_step() -> None:
    """An in-process response feeds the identical first-token yes/no credence step.

    Same assertion as the served-path logprob test: p_yes = 0.8/0.95 becomes the
    judgement score. This proves there is ONE logprob→score computation, reused --
    only the response *shape* is reproduced in-process.
    """
    p_yes = 0.8 / 0.95
    backend = _FakeBackend(_logprob_response([("Yes", 0.8), (" No", 0.15)], message="Yes"))
    judge = LLMMatcher[CompanySchema](
        client=object(),  # must NEVER be called on the in-process path
        model="your-org/your-ft-model",
        confidence="logprob",
        response_parser=parse_binary_yes_no,
    )
    assert judge._backend_kind == "transformers"
    judge._backend = backend  # inject the fake (no torch, no download)

    out = list(judge.forward(iter([_candidate()])))[0]

    assert out.decision is True  # from the "Yes" text
    assert out.score == pytest.approx(p_yes)  # honest continuous p_yes
    assert out.confidence == pytest.approx(max(p_yes, 1.0 - p_yes))
    assert out.confidence_source == "logprob"
    # The backend saw the credence request; the client was never touched.
    assert backend.calls and backend.calls[0]["want_logprobs"] is True


def test_inprocess_backend_not_the_silent_half_fallback() -> None:
    """A confident in-process p_yes is a real score, never the 0.5 parse-miss value."""
    backend = _FakeBackend(_logprob_response([("Yes", 0.97), ("No", 0.01)], message="Yes"))
    judge = LLMMatcher[CompanySchema](
        client=object(),
        model="your-org/your-ft-model",
        confidence="logprob",
        response_parser=parse_binary_yes_no,
    )
    judge._backend = backend
    out = list(judge.forward(iter([_candidate()])))[0]
    assert out.score is not None
    assert out.score != pytest.approx(0.5)
    assert 0.0 <= out.score <= 1.0
    assert out.provenance.get("parse_error") is None


def test_inprocess_backend_works_on_the_async_path() -> None:
    """The async path routes to the same in-process backend (parity with sync)."""
    p_yes = 0.8 / 0.95
    backend = _FakeBackend(_logprob_response([("Yes", 0.8), (" No", 0.15)], message="Yes"))
    judge = LLMMatcher[CompanySchema](
        client=object(),
        model="your-org/your-ft-model",
        confidence="logprob",
        response_parser=parse_binary_yes_no,
    )
    judge._backend = backend
    out = asyncio.run(judge.forward_async([_candidate()], max_concurrent=1))[0]
    assert out.score == pytest.approx(p_yes)
    assert backend.calls and backend.calls[0]["want_logprobs"] is True
