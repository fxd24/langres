"""$0 smoke tests for the Peeters LLM-EM live/dry-run harness.

Every test runs at **$0** — no API key, no network, no real model call. A fake
client (returning canned "Yes"/"No" answers with a fake usage/cost) drives the
live path so the whole flow — build candidates, judge, charge the SpendMonitor,
aggregate usage, score pairwise F1 — is verified without spending. Proves:

1. The dry-run renders + counts tokens with zero API calls, priced from the table.
2. The live core runs end-to-end and reports F1 + the aggregated usage vector +
   the real billed cost.
3. The hard SpendMonitor cap FIRES (partial run) when cost crosses the budget.
4. The safety guards (priced-model assertion, model resolution) behave.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from examples.research.peeters_llm_em_replication import (
    PAID_MODELS,
    _aggregate_usage,
    _assert_priced,
    _resolve_paid_models,
    dry_run,
    run_live,
)
from langres.clients.openrouter import PRICES_PER_1M
from langres.data.peeters import get_peeters_replication, load_peeters_sample

_MODEL = "openrouter/openai/gpt-4o-mini-2024-07-18"


# --------------------------------------------------------------------------- #
# Fake client: canned answers + a fake usage/cost on every response.
# --------------------------------------------------------------------------- #


def _response(content: str, *, cost: float, in_tok: int = 80, out_tok: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            cost=cost,  # parse_openrouter_billing reads usage.cost -> cost_is_real
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        _hidden_params={},
        provider="fake-provider",
        model="fake",
    )


class _FakeClient:
    """A stand-in for litellm: hands back canned answers with a fixed per-call cost."""

    def __init__(self, answers: list[str], *, cost_per_call: float) -> None:
        self._answers = answers
        self._i = 0
        self._cost = cost_per_call

    def completion(self, **_kwargs: Any) -> SimpleNamespace:
        content = self._answers[self._i]
        self._i += 1
        return _response(content, cost=self._cost)


def _perfect_answers(spec: Any) -> list[str]:
    """ "Yes" for the gold positives, "No" for the rest — a perfect-F1 answer set."""
    return ["Yes" if label == 1 else "No" for _l, _r, label in load_peeters_sample(spec)]


# --------------------------------------------------------------------------- #
# Priced-model guards
# --------------------------------------------------------------------------- #


def test_paid_models_are_all_priced() -> None:
    """Every paid-run model must have a PRICES_PER_1M entry (else the cap is blind)."""
    for model in PAID_MODELS:
        assert model in PRICES_PER_1M


def test_assert_priced_rejects_unpriced_model() -> None:
    with pytest.raises(SystemExit, match="no PRICES_PER_1M entry"):
        _assert_priced(["openrouter/openai/not-a-real-model"])


def test_resolve_paid_models_defaults_to_both() -> None:
    assert _resolve_paid_models(None) == list(PAID_MODELS)
    assert _resolve_paid_models("both") == list(PAID_MODELS)
    assert _resolve_paid_models(_MODEL) == [_MODEL]


def test_resolve_paid_models_rejects_unknown() -> None:
    with pytest.raises(SystemExit, match="not a paid-run model"):
        _resolve_paid_models("gpt-4-0613")


# --------------------------------------------------------------------------- #
# Dry run ($0, injected counter — no litellm/tiktoken needed)
# --------------------------------------------------------------------------- #


def test_dry_run_counts_and_prices_with_injected_counter() -> None:
    spec = get_peeters_replication("abt-buy")
    report = dry_run(spec, _MODEL, count_tokens=lambda _prompt: 10)
    n = report["n_pairs"]
    assert n == 1206
    assert report["input_tokens"] == 10 * n
    assert report["output_tokens_est"] == 2 * n
    assert report["max_input_tokens"] == 10
    in_1m, out_1m = PRICES_PER_1M[_MODEL]
    expected = (10 * n) * in_1m / 1e6 + (2 * n) * out_1m / 1e6
    assert report["est_usd"] == pytest.approx(expected)


@pytest.mark.slow
def test_dry_run_real_token_total_matches_measurement() -> None:
    """The real tiktoken count over all 1206 rendered prompts is 100,256 input tokens.

    Pins prompt-rendering fidelity: a drift in the live template/serializer would
    move this away from the value measured with o200k_base before any paid run.
    """
    pytest.importorskip("litellm")
    spec = get_peeters_replication("abt-buy")
    report = dry_run(spec, _MODEL)
    assert report["input_tokens"] == 100256
    assert report["output_tokens_est"] == 2412


# --------------------------------------------------------------------------- #
# Live core ($0 via the fake client)
# --------------------------------------------------------------------------- #


def test_run_live_end_to_end_with_perfect_answers() -> None:
    spec = get_peeters_replication("abt-buy")
    answers = _perfect_answers(spec)
    client = _FakeClient(answers, cost_per_call=0.0)
    result = run_live(spec, _MODEL, budget_usd=1.0, client=client)

    assert result["n_judged"] == 1206
    assert result["budget_hit"] is False
    assert result["f1"] == pytest.approx(100.0)
    assert result["precision"] == pytest.approx(100.0)
    assert result["recall"] == pytest.approx(100.0)
    assert result["fp"] == 0 and result["fn"] == 0
    # Usage aggregated across all 1206 fake calls (80 in / 2 out each).
    assert result["usage"]["input_tokens"] == 1206 * 80
    assert result["usage"]["output_tokens"] == 1206 * 2
    assert result["published_f1"] == 90.95


def test_run_live_aggregates_real_billed_cost() -> None:
    spec = get_peeters_replication("abt-buy")
    answers = _perfect_answers(spec)
    client = _FakeClient(answers, cost_per_call=0.0001)
    result = run_live(spec, _MODEL, budget_usd=1.0, client=client)

    assert result["cost_is_real"] is True
    assert result["real_cost_usd"] == pytest.approx(1206 * 0.0001)
    assert result["usd_per_1k_pairs"] == pytest.approx(0.0001 * 1000.0)


def test_run_live_spend_cap_fires_and_returns_partial() -> None:
    """The hard cap stops the run: high per-call cost + tiny budget => partial."""
    spec = get_peeters_replication("abt-buy")
    answers = _perfect_answers(spec)
    client = _FakeClient(answers, cost_per_call=0.5)
    result = run_live(spec, _MODEL, budget_usd=1.0, client=client)

    assert result["budget_hit"] is True
    assert result["n_judged"] == 3  # 0.5*3 = 1.5 > 1.0 budget; stops on the 3rd
    assert result["n_pairs"] == 1206


def test_aggregate_usage_sums_vectors() -> None:
    judgements = [
        SimpleNamespace(provenance={"usage": {"input_tokens": 5, "output_tokens": 1}}),
        SimpleNamespace(provenance={"usage": {"input_tokens": 7, "output_tokens": 2}}),
        SimpleNamespace(provenance={}),  # a judgement with no usage vector
    ]
    usage = _aggregate_usage(judgements, _MODEL)
    assert usage.input_tokens == 12
    assert usage.output_tokens == 3
    assert usage.model == _MODEL
