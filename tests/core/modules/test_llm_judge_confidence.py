"""$0 tests for the LLMMatcher first-token logprob credence probe.

Every test runs at **$0** with an injected fake client — no API key, no network,
no real model call. Covers the P(Yes) math (:meth:`LLMMatcher._confidence_from_response`),
the two-way-subspace renormalisation, the never-normalised leaked mass, the
one-sided *bound* flag, the below-floor abstention, and — the bug the plan calls
out — that ``logprobs``/``top_logprobs`` are requested at BOTH completion call
sites for a **non-openrouter** model (they must NOT ride inside
``_completion_kwargs``, which early-returns ``{}`` off ``openrouter/``).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from langres.core.models import CompanySchema, ERCandidate
from langres.core.matchers.llm_judge import (
    LLMMatcher,
    _normalize_answer_token,
    parse_binary_yes_no,
)


# --------------------------------------------------------------------------- #
# Fakes: responses carrying an OpenAI/litellm-shaped logprobs block.
# --------------------------------------------------------------------------- #


def _top(token: str, prob: float) -> SimpleNamespace:
    """One ``top_logprobs`` alternative: a token + its logprob (from a probability)."""
    return SimpleNamespace(token=token, logprob=math.log(prob), bytes=None)


def _response(
    content_tokens: list[tuple[str, list[tuple[str, float]]]] | None,
    *,
    message: str = "Yes",
    cost: float = 1e-5,
) -> SimpleNamespace:
    """A fake completion response.

    ``content_tokens`` is a list of ``(generated_token, [(alt_token, prob), ...])``
    per generated position, becoming ``choices[0].logprobs.content``. ``None``
    omits the logprobs block entirely (the "logprobs not returned" case).
    """
    logprobs: Any = None
    if content_tokens is not None:
        content = [
            SimpleNamespace(
                token=tok,
                logprob=0.0,
                bytes=None,
                top_logprobs=[_top(t, p) for t, p in alts],
            )
            for tok, alts in content_tokens
        ]
        logprobs = SimpleNamespace(content=content)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=message), logprobs=logprobs)],
        usage=SimpleNamespace(
            prompt_tokens=70,
            completion_tokens=1,
            cost=cost,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        _hidden_params={},
        provider="fake",
        model="fake",
    )


class _FakeClient:
    """Records the kwargs of the last ``completion`` call; returns a fixed response."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response
        self.last_kwargs: dict[str, Any] | None = None

    def completion(self, **kwargs: Any) -> SimpleNamespace:
        self.last_kwargs = kwargs
        return self._response


def _judge(model: str = "gpt-4o-mini", confidence: str = "logprob") -> LLMMatcher[CompanySchema]:
    # A sentinel client so the lazy-from-env path is NEVER taken (no real client).
    return LLMMatcher[CompanySchema](
        client=object(),
        model=model,
        confidence=confidence,  # type: ignore[arg-type]
        response_parser=parse_binary_yes_no,
    )


def _candidate() -> ERCandidate[CompanySchema]:
    return ERCandidate(
        left=CompanySchema(id="a1", name="Acme Corp"),
        right=CompanySchema(id="b1", name="Acme Corporation"),
        blocker_name="test",
    )


def test_p_yes_does_not_clobber_a_rating_parsers_score() -> None:
    """A rating parser (decision=None, score=<float>) + logprobs must keep its rating.

    p_yes from first-token yes/no mass is meaningless for a "rate 0-1" response;
    promoting it to `score` would silently discard the parsed rating. The promotion
    is gated on `parsed.decision is not None` (a binary decider), so a rating flows
    through untouched.
    """
    from langres.core.matchers.llm_judge import parse_score_response

    judge = LLMMatcher[CompanySchema](
        client=object(),  # sentinel; _map_verdict never touches the client
        model="gpt-4o-mini",
        confidence="logprob",
        response_parser=parse_score_response,
    )
    decision, score, confidence, source, _reasoning, parse_error = judge._map_verdict(
        "Score: 0.42", {"p_yes": 0.9}
    )
    assert decision is None  # a ranker, not a decider
    assert score == pytest.approx(0.42)  # the parsed rating, NOT p_yes=0.9
    assert confidence is None
    assert source == "none"
    assert parse_error is False


# --------------------------------------------------------------------------- #
# _normalize_answer_token — casing/whitespace/punctuation collapse.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("yes", "yes"),
        (" Yes", "yes"),
        ("YES", "yes"),
        ("Yes.", "yes"),
        (" No", "no"),
        ("NO,", "no"),
        ("\n No\n", "no"),
    ],
)
def test_normalize_answer_token_variants(raw: str, expected: str) -> None:
    assert _normalize_answer_token(raw) == expected


# --------------------------------------------------------------------------- #
# _confidence_from_response — the P(Yes) math.
# --------------------------------------------------------------------------- #


def test_pyes_sums_casing_and_whitespace_variants() -> None:
    """All yes-variants add into yes_mass, no-variants into no_mass; renorm 2-way."""
    j = _judge()
    resp = _response(
        [
            (
                "Yes",
                [("Yes", 0.5), (" Yes", 0.15), ("YES", 0.05), (" No", 0.20), ("maybe", 0.05)],
            )
        ]
    )
    conf = j._confidence_from_response(resp)
    assert conf is not None
    # yes_mass = 0.5+0.15+0.05 = 0.70 ; no_mass = 0.20 ; two_way = 0.90
    assert conf["p_yes"] == pytest.approx(0.70 / 0.90)
    assert conf["confidence_leaked_mass"] == pytest.approx(1.0 - 0.90)
    assert conf["p_yes_is_bound"] is False


def test_leaked_mass_is_recorded_and_never_normalised_away() -> None:
    """Mass on non-yes/no tokens stays in confidence_leaked_mass, not folded into p_yes."""
    j = _judge()
    # 0.4 yes, 0.1 no, 0.5 elsewhere. p_yes renormalises over 0.5 only; leaked stays 0.5.
    resp = _response([("Yes", [("Yes", 0.4), ("No", 0.1), ("Maybe", 0.5)])])
    conf = j._confidence_from_response(resp)
    assert conf is not None
    assert conf["p_yes"] == pytest.approx(0.4 / 0.5)
    assert conf["confidence_leaked_mass"] == pytest.approx(0.5)
    # p_yes was renormalised over the 2-way subspace, but the leaked 0.5 is preserved.


def test_one_sided_mass_is_flagged_as_a_bound() -> None:
    """Only-yes mass => p_yes=1.0 is a BOUND (the true no-mass is below the top-k cutoff)."""
    j = _judge()
    resp = _response([("Yes", [("Yes", 0.9), ("perhaps", 0.05)])])
    conf = j._confidence_from_response(resp)
    assert conf is not None
    assert conf["p_yes"] == 1.0
    assert conf["p_yes_is_bound"] is True
    assert conf["confidence_leaked_mass"] == pytest.approx(1.0 - 0.9)


def test_only_no_mass_is_also_a_bound() -> None:
    j = _judge()
    resp = _response([("No", [("No", 0.85), ("nah", 0.1)])])
    conf = j._confidence_from_response(resp)
    assert conf is not None
    assert conf["p_yes"] == 0.0
    assert conf["p_yes_is_bound"] is True


def test_below_floor_mass_yields_p_yes_none() -> None:
    """No yes/no mass at all => p_yes is None (don't manufacture credence from noise)."""
    j = _judge()
    resp = _response([("Maybe", [("Maybe", 0.6), ("Unsure", 0.3)])])
    conf = j._confidence_from_response(resp)
    assert conf is not None
    assert conf["p_yes"] is None
    assert conf["p_yes_is_bound"] is False
    assert conf["confidence_leaked_mass"] == pytest.approx(1.0)


def test_first_non_whitespace_token_is_used() -> None:
    """A leading whitespace-only generated token is skipped; the next one is scored."""
    j = _judge()
    resp = _response(
        [
            ("\n", [("\n", 0.99)]),  # pure-whitespace generated token -> skipped
            (" No", [(" No", 0.8), (" Yes", 0.15)]),
        ]
    )
    conf = j._confidence_from_response(resp)
    assert conf is not None
    assert conf["p_yes"] == pytest.approx(0.15 / 0.95)


def test_missing_logprobs_yields_no_confidence_no_crash() -> None:
    """logprobs block absent => None (no confidence), never an exception."""
    j = _judge()
    assert j._confidence_from_response(_response(None)) is None


def test_empty_content_yields_no_confidence() -> None:
    j = _judge()
    resp = SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[]))])
    assert j._confidence_from_response(resp) is None


def test_confidence_off_returns_none_even_with_logprobs() -> None:
    """confidence='none' is a no-op: no credence even when logprobs are present."""
    joff = _judge(confidence="none")
    resp = _response([("Yes", [("Yes", 0.9), ("No", 0.1)])])
    assert joff._confidence_from_response(resp) is None


# --------------------------------------------------------------------------- #
# _logprobs_kwargs — requested for a NON-openrouter model (the plan's bug).
# --------------------------------------------------------------------------- #


def test_logprobs_kwargs_on_and_off() -> None:
    assert _judge(confidence="logprob")._logprobs_kwargs() == {"logprobs": True, "top_logprobs": 20}
    assert _judge(confidence="none")._logprobs_kwargs() == {}


def test_logprobs_kwargs_present_for_non_openrouter_model() -> None:
    """The bug: logprobs must be requested on plain OpenAI too (NOT via _completion_kwargs).

    _completion_kwargs early-returns {} for any non-openrouter/ model, so logprobs
    would silently never be requested if they rode inside it.
    """
    j = _judge(model="gpt-4o-mini", confidence="logprob")
    assert j._completion_kwargs() == {}  # non-openrouter -> no extra_body
    assert j._logprobs_kwargs() == {"logprobs": True, "top_logprobs": 20}  # but logprobs still on


def test_confidence_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="confidence must be 'none' or 'logprob'"):
        LLMMatcher[CompanySchema](client=object(), model="gpt-4o-mini", confidence="always")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Merged at the SYNC call site — end-to-end forward().
# --------------------------------------------------------------------------- #


def test_sync_forward_requests_logprobs_on_non_openrouter_and_records_pyes() -> None:
    resp = _response([("Yes", [("Yes", 0.8), (" No", 0.15)])])
    client = _FakeClient(resp)
    j = LLMMatcher[CompanySchema](
        client=client,
        model="gpt-4o-mini",  # NON-openrouter
        confidence="logprob",
        response_parser=parse_binary_yes_no,
    )
    out = list(j.forward(iter([_candidate()])))
    assert client.last_kwargs is not None
    # logprobs reached the wire as TOP-LEVEL params (not extra_body).
    assert client.last_kwargs["logprobs"] is True
    assert client.last_kwargs["top_logprobs"] == 20
    assert "extra_body" not in client.last_kwargs  # non-openrouter -> none
    prov = out[0].provenance
    assert prov["p_yes"] == pytest.approx(0.8 / 0.95)
    assert prov["confidence_leaked_mass"] == pytest.approx(1.0 - 0.95)
    assert prov["p_yes_is_bound"] is False


def test_logprob_forward_promotes_pyes_onto_the_judgement() -> None:
    """A usable p_yes becomes the judgement's score + a max(p_yes,1-p_yes) confidence."""
    p_yes = 0.8 / 0.95
    resp = _response([("Yes", [("Yes", 0.8), (" No", 0.15)])], message="Yes")
    j = LLMMatcher[CompanySchema](
        client=_FakeClient(resp),
        model="gpt-4o-mini",
        confidence="logprob",
        response_parser=parse_binary_yes_no,
    )
    out = list(j.forward(iter([_candidate()])))[0]
    # decision from the "Yes" text; score is the honest continuous p_yes.
    assert out.decision is True
    assert out.score == pytest.approx(p_yes)
    # confidence = credence in its OWN answer (the roc_auc-0.95 probe quantity).
    assert out.confidence == pytest.approx(max(p_yes, 1.0 - p_yes))
    assert out.confidence_source == "logprob"


def test_logprob_run_with_no_usable_mass_falls_back_to_decision_only() -> None:
    """logprob requested but the first token carried no yes/no mass -> p_yes None.

    The judge still decides (from the text), score stays None (binary family, no
    p_yes to promote), and confidence_source is "none" (no earned credence).
    """
    resp = _response([("Maybe", [("Maybe", 0.6), ("Unsure", 0.3)])], message="Yes")
    j = LLMMatcher[CompanySchema](
        client=_FakeClient(resp),
        model="gpt-4o-mini",
        confidence="logprob",
        response_parser=parse_binary_yes_no,
    )
    out = list(j.forward(iter([_candidate()])))[0]
    assert out.decision is True
    assert out.score is None
    assert out.confidence is None
    assert out.confidence_source == "none"
    assert out.provenance["p_yes"] is None  # the fragment is still recorded


def test_sync_forward_openrouter_sends_both_extra_body_and_logprobs() -> None:
    resp = _response([("Yes", [("Yes", 0.9), ("No", 0.05)])])
    client = _FakeClient(resp)
    j = LLMMatcher[CompanySchema](
        client=client,
        model="openrouter/openai/gpt-4o-mini",
        confidence="logprob",
        response_parser=parse_binary_yes_no,
    )
    list(j.forward(iter([_candidate()])))
    assert client.last_kwargs is not None
    assert client.last_kwargs["logprobs"] is True  # top-level logprobs
    assert "extra_body" in client.last_kwargs  # AND openrouter usage accounting


def test_sync_forward_confidence_off_is_byte_identical() -> None:
    """confidence='none' adds no logprobs kwarg and no p_yes provenance key."""
    resp = _response([("Yes", [("Yes", 0.9), ("No", 0.05)])])
    client = _FakeClient(resp)
    j = LLMMatcher[CompanySchema](
        client=client,
        model="gpt-4o-mini",
        response_parser=parse_binary_yes_no,
    )
    out = list(j.forward(iter([_candidate()])))
    assert client.last_kwargs is not None
    assert "logprobs" not in client.last_kwargs
    assert "top_logprobs" not in client.last_kwargs
    assert "p_yes" not in out[0].provenance


# --------------------------------------------------------------------------- #
# Merged at the ASYNC call site — end-to-end forward_async().
# --------------------------------------------------------------------------- #


def test_async_forward_requests_logprobs_and_records_pyes() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    resp = _response([("No", [("No", 0.7), (" Yes", 0.2)])])
    client = SimpleNamespace(acompletion=AsyncMock(return_value=resp))
    j = LLMMatcher[CompanySchema](
        client=client,
        model="gpt-4o-mini",  # NON-openrouter
        confidence="logprob",
        response_parser=parse_binary_yes_no,
    )
    out = asyncio.run(j.forward_async([_candidate()], max_concurrent=1))
    # logprobs reached acompletion as top-level params.
    _args, kwargs = client.acompletion.call_args
    assert kwargs["logprobs"] is True and kwargs["top_logprobs"] == 20
    assert "extra_body" not in kwargs
    prov = out[0].provenance
    assert prov["p_yes"] == pytest.approx(0.2 / 0.9)
    assert prov["p_yes_is_bound"] is False
