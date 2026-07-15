"""Tests for langres.clients.openrouter (mock-based, $0 — no network, no spend)."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import litellm
import pytest

from langres.clients.openrouter import (
    PRICES_PER_1M,
    BudgetExceeded,
    SpendMonitor,
    make_token_cost_track,
    no_keepalive_http_client,
    parse_openrouter_billing,
    patch_litellm_prices,
    per_token_worst_price,
    register_runtime_model_price,
)
from langres.core.models import PairwiseJudgement
from langres.core.usage import LLMUsage

GLM = "openrouter/z-ai/glm-5.2"

# The hidden-param header LiteLLM's OpenRouter transform writes the real cost to.
_COST_HEADER = "llm_provider-x-litellm-response-cost"


def _resp(
    *,
    hidden_params: object = None,
    usage: object = None,
    provider: object = None,
    model_extra: object = None,
) -> SimpleNamespace:
    """Build a minimal completion-response stand-in for billing parsing.

    Only the attributes explicitly passed are set; ``getattr(..., default)`` in
    the parser handles the rest, matching how a real ``ModelResponse`` surfaces
    (or omits) these fields.
    """
    ns = SimpleNamespace()
    if hidden_params is not None:
        ns._hidden_params = hidden_params
    if usage is not None:
        ns.usage = usage
    if provider is not None:
        ns.provider = provider
    if model_extra is not None:
        ns.model_extra = model_extra
    return ns


def _judgement(prompt_tokens: object, completion_tokens: object) -> PairwiseJudgement:
    """A minimal LLM-style judgement carrying token counts in provenance."""
    return PairwiseJudgement(
        left_id="a",
        right_id="b",
        score=0.5,
        score_type="prob_llm",
        decision_step="llm_judgment",
        provenance={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )


def _judgement_with_usage(prompt_tokens: int, completion_tokens: int) -> PairwiseJudgement:
    """A judgement carrying both the legacy token keys and the nested usage vector.

    Mirrors what ``LLMMatcher``/``DSPyMatcher`` actually write (see
    ``llm_judge.py::_build_provenance``): the legacy ``prompt_tokens``/
    ``completion_tokens`` keys plus the full ``LLMUsage.model_dump()`` under
    ``"usage"``.
    """
    usage = LLMUsage(input_tokens=prompt_tokens, output_tokens=completion_tokens)
    return PairwiseJudgement(
        left_id="a",
        right_id="b",
        score=0.5,
        score_type="prob_llm",
        decision_step="llm_judgment",
        provenance={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usage": usage.model_dump(),
        },
    )


class TestPatchLitellmPrices:
    """patch_litellm_prices writes the pinned price under routing + bare keys."""

    def test_writes_routing_and_bare_keys(self) -> None:
        with patch.dict(litellm.model_cost, {}, clear=False):
            patch_litellm_prices(GLM)
            in_per_1m, out_per_1m = PRICES_PER_1M[GLM]
            for key in (GLM, "z-ai/glm-5.2"):
                entry = litellm.model_cost[key]
                assert entry["input_cost_per_token"] == in_per_1m / 1_000_000.0
                assert entry["output_cost_per_token"] == out_per_1m / 1_000_000.0
                assert entry["litellm_provider"] == "openrouter"
                assert entry["mode"] == "chat"

    def test_custom_prices_and_bare_model_without_slash(self) -> None:
        # A model id with no "/" exercises the ``bare = model`` branch, and a
        # caller-supplied table proves ``prices`` overrides the default.
        prices = {"mymodel": (1.0, 2.0)}
        with patch.dict(litellm.model_cost, {}, clear=False):
            patch_litellm_prices("mymodel", prices)
            assert litellm.model_cost["mymodel"]["input_cost_per_token"] == 1.0 / 1_000_000.0
            assert litellm.model_cost["mymodel"]["output_cost_per_token"] == 2.0 / 1_000_000.0


class TestRegisterRuntimeModelPrice:
    """register_runtime_model_price pins the dated runtime id from a probe."""

    def test_pins_dated_id(self) -> None:
        resp = MagicMock()
        resp.model = "z-ai/glm-5.2-20260616"
        with (
            patch.object(litellm, "completion", return_value=resp) as mock_completion,
            patch.dict(litellm.model_cost, {}, clear=False),
        ):
            dated = register_runtime_model_price(GLM)
            assert dated == "z-ai/glm-5.2-20260616"
            in_per_1m, out_per_1m = PRICES_PER_1M[GLM]
            entry = litellm.model_cost["z-ai/glm-5.2-20260616"]
            assert entry["input_cost_per_token"] == in_per_1m / 1_000_000.0
            assert entry["output_cost_per_token"] == out_per_1m / 1_000_000.0
            mock_completion.assert_called_once()

    def test_unknown_model_returns_none_without_probing(self) -> None:
        with patch.object(litellm, "completion") as mock_completion:
            assert register_runtime_model_price("openrouter/unknown/model") is None
            mock_completion.assert_not_called()

    def test_probe_failure_returns_none(self) -> None:
        with (
            patch.object(litellm, "completion", side_effect=RuntimeError("boom")),
            patch.dict(litellm.model_cost, {}, clear=False),
        ):
            assert register_runtime_model_price(GLM) is None


class TestPerTokenWorstPrice:
    """per_token_worst_price returns the dearer of input/output, per token."""

    def test_uses_dearer_side(self) -> None:
        in_per_1m, out_per_1m = PRICES_PER_1M[GLM]
        expected = max(in_per_1m, out_per_1m) / 1_000_000.0
        assert per_token_worst_price(GLM) == expected

    def test_custom_prices(self) -> None:
        assert per_token_worst_price("m", {"m": (5.0, 2.0)}) == 5.0 / 1_000_000.0


class TestUnknownModelError:
    """Price lookups raise a descriptive KeyError (unknown id + the known ids)."""

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda: patch_litellm_prices("nope/model"), id="patch_litellm_prices"),
            pytest.param(lambda: per_token_worst_price("nope/model"), id="per_token_worst_price"),
            pytest.param(lambda: make_token_cost_track("nope/model"), id="make_token_cost_track"),
        ],
    )
    def test_descriptive_keyerror(self, call: object) -> None:
        with pytest.raises(KeyError, match=r"unknown model .*nope/model.*known"):
            call()  # type: ignore[operator]


class TestMakeTokenCostTrack:
    """make_token_cost_track prices judgements from their provenance token counts."""

    def test_prices_from_token_counts(self) -> None:
        in_per_1m, out_per_1m = PRICES_PER_1M[GLM]
        in_tok, out_tok = in_per_1m / 1_000_000.0, out_per_1m / 1_000_000.0
        track = make_token_cost_track(GLM)
        judgements = [_judgement(1000, 500), _judgement(200, 100)]
        expected = (1000 + 200) * in_tok + (500 + 100) * out_tok

        result = track(judgements)

        assert result.usd_total == pytest.approx(expected)
        per_pair = expected / 2
        assert result.usd_per_1k_pairs == pytest.approx(per_pair * 1_000.0)
        assert result.est_usd_per_100k == pytest.approx(per_pair * 100_000.0)

    def test_empty_judgements_is_zero(self) -> None:
        result = make_token_cost_track(GLM)([])
        assert result.usd_total == 0.0
        assert result.usd_per_1k_pairs == 0.0
        assert result.est_usd_per_100k == 0.0

    def test_missing_or_none_token_counts_count_as_zero(self) -> None:
        # ``None`` (and absent keys) must coerce to 0 via the ``or 0`` guard.
        result = make_token_cost_track(GLM)([_judgement(None, None)])
        assert result.usd_total == 0.0


class TestMakeTokenCostTrackHonesty:
    """cost_basis/usage: this tracker prices from a pinned table, never real billing.

    Regression coverage for the bug this PR fixes: ``make_token_cost_track``
    priced a real dollar amount but left ``cost_basis``/``usage`` at their
    all-``"none"``/all-zero defaults, so a costed run reported ``usd_total >
    0`` alongside ``cost_basis == "none"`` and ``cost_is_real == False`` --
    the exact dishonest cell ``CostTrack.cost_basis`` exists to eliminate.
    """

    def test_real_token_counts_are_estimated_never_real(self) -> None:
        # This tracker multiplies token counts by PRICES_PER_1M -- it never
        # reads a provider-billed amount off a response -- so a nonzero total
        # must be "estimated", and cost_is_real (== cost_basis == "real")
        # must be False.
        track = make_token_cost_track(GLM)
        result = track([_judgement(1000, 500), _judgement(200, 100)])

        assert result.usd_total > 0.0
        assert result.cost_basis == "estimated"
        assert result.cost_is_real is False

    def test_usage_vector_sums_token_counts_from_provenance(self) -> None:
        track = make_token_cost_track(GLM)
        judgements = [_judgement_with_usage(1000, 500), _judgement_with_usage(200, 100)]

        result = track(judgements)

        assert result.usage.input_tokens == 1000 + 200
        assert result.usage.output_tokens == 500 + 100

    def test_malformed_usage_dict_degrades_to_zero_vector(self) -> None:
        # A corrupt/foreign "usage" payload must not crash pricing -- usage
        # capture is observability, never a hard failure (mirrors
        # benchmark.py::_judgement_usage's own degrade-to-zero contract).
        judgement = PairwiseJudgement(
            left_id="a",
            right_id="b",
            score=0.5,
            score_type="prob_llm",
            decision_step="llm_judgment",
            provenance={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "usage": {"input_tokens": "not-a-number"},
            },
        )

        result = make_token_cost_track(GLM)([judgement])

        assert result.usd_total > 0.0  # pricing still runs off the legacy keys
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_empty_judgements_has_none_cost_basis(self) -> None:
        result = make_token_cost_track(GLM)([])
        assert result.usd_total == 0.0
        assert result.cost_basis == "none"

    def test_zero_tokens_has_none_cost_basis_no_divide_by_zero(self) -> None:
        result = make_token_cost_track(GLM)([_judgement(None, None)])
        assert result.usd_total == 0.0
        assert result.cost_basis == "none"

    def test_nonzero_cost_implies_non_none_basis(self) -> None:
        # The invariant that matters: a run that reports real dollars can
        # never carry cost_basis == "none" -- that combination is exactly the
        # dishonest cell this PR eliminates.
        result = make_token_cost_track(GLM)([_judgement(1000, 500)])
        assert result.usd_total > 0.0
        assert result.cost_basis != "none"


class TestNoKeepaliveHttpClient:
    """no_keepalive_http_client builds a stall-proof httpx client."""

    def test_builds_client_with_timeout(self) -> None:
        client = no_keepalive_http_client(5.0)
        try:
            assert isinstance(client, httpx.Client)
            assert client.timeout.read == 5.0
        finally:
            client.close()

    def test_default_timeout(self) -> None:
        client = no_keepalive_http_client()
        try:
            assert client.timeout.read == 60.0
        finally:
            client.close()


class TestSpendMonitor:
    """SpendMonitor accumulates, warns at the threshold, and raises past budget."""

    def test_accumulates_and_reports_remaining(self) -> None:
        monitor = SpendMonitor(budget_usd=10.0)
        monitor.add(1.0)
        monitor.add(2.5)
        assert monitor.spent == pytest.approx(3.5)
        assert monitor.remaining == pytest.approx(6.5)

    def test_check_below_threshold_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        monitor = SpendMonitor(budget_usd=10.0, warn_frac=0.8)
        monitor.add(5.0)
        with caplog.at_level(logging.WARNING, logger="langres.clients.openrouter"):
            monitor.check()
        assert caplog.records == []

    def test_check_warns_past_warn_fraction(self, caplog: pytest.LogCaptureFixture) -> None:
        monitor = SpendMonitor(budget_usd=10.0, warn_frac=0.8)
        monitor.add(8.0)
        with caplog.at_level(logging.WARNING, logger="langres.clients.openrouter"):
            monitor.check()
        assert any("budget" in r.message for r in caplog.records)

    def test_check_at_budget_warns_but_does_not_raise(self) -> None:
        monitor = SpendMonitor(budget_usd=10.0, warn_frac=0.8)
        monitor.add(10.0)
        monitor.check()  # spent == budget is not > budget, so no raise
        assert monitor.remaining == pytest.approx(0.0)

    def test_check_raises_past_budget(self) -> None:
        monitor = SpendMonitor(budget_usd=5.0)
        monitor.add(6.0)
        with pytest.raises(BudgetExceeded, match="exceeds budget"):
            monitor.check()
        assert monitor.remaining == pytest.approx(-1.0)

    def test_defaults(self) -> None:
        monitor = SpendMonitor()
        assert monitor.remaining == pytest.approx(5.0)

    def test_budget_usd_exposes_configured_budget(self) -> None:
        monitor = SpendMonitor(budget_usd=7.5)
        monitor.add(2.0)
        assert monitor.budget_usd == pytest.approx(7.5)  # constant across spend
        assert SpendMonitor().budget_usd == pytest.approx(5.0)  # default budget


class TestParseOpenRouterBilling:
    """parse_openrouter_billing reads the real billed cost + serving provider."""

    def test_real_cost_from_litellm_hidden_param_header(self) -> None:
        # LiteLLM's OpenRouter transform stashes the provider-billed cost here.
        resp = _resp(
            hidden_params={"additional_headers": {_COST_HEADER: 0.00042}},
            provider="DeepInfra",
        )
        cost, provider = parse_openrouter_billing(resp)
        assert cost == pytest.approx(0.00042)
        assert provider == "DeepInfra"

    def test_real_cost_from_usage_cost_and_provider_from_model_extra(self) -> None:
        # No hidden-param header: fall back to a raw usage.cost, provider via model_extra.
        resp = _resp(
            usage=SimpleNamespace(cost=0.0009),
            model_extra={"provider": "Together"},
        )
        cost, provider = parse_openrouter_billing(resp)
        assert cost == pytest.approx(0.0009)
        assert provider == "Together"

    def test_hidden_param_header_wins_over_usage_cost(self) -> None:
        # The header is OpenRouter's authoritative figure; it takes precedence.
        resp = _resp(
            hidden_params={"additional_headers": {_COST_HEADER: 0.005}},
            usage=SimpleNamespace(cost=0.009),
        )
        cost, _ = parse_openrouter_billing(resp)
        assert cost == pytest.approx(0.005)

    def test_absent_cost_and_provider_return_none(self) -> None:
        # A bare response (no usage accounting, non-OpenRouter) yields no real cost.
        cost, provider = parse_openrouter_billing(_resp(usage=SimpleNamespace()))
        assert cost is None
        assert provider is None

    def test_malformed_header_and_usage_cost_are_ignored(self) -> None:
        # Non-numeric values must not crash — they degrade to "no real cost".
        resp = _resp(
            hidden_params={"additional_headers": {_COST_HEADER: "not-a-number"}},
            usage=SimpleNamespace(cost="also-bad"),
        )
        assert parse_openrouter_billing(resp) == (None, None)

    def test_hidden_params_without_additional_headers_falls_through(self) -> None:
        # _hidden_params present but no additional_headers → use usage.cost.
        resp = _resp(hidden_params={}, usage=SimpleNamespace(cost=0.001))
        cost, _ = parse_openrouter_billing(resp)
        assert cost == pytest.approx(0.001)

    def test_empty_provider_string_falls_back_to_model_extra(self) -> None:
        # An empty provider attr is treated as absent; model_extra fills in.
        resp = _resp(provider="", model_extra={"provider": "Fireworks"})
        _, provider = parse_openrouter_billing(resp)
        assert provider == "Fireworks"

    def test_non_string_model_extra_provider_is_none(self) -> None:
        resp = _resp(model_extra={"provider": 123})
        _, provider = parse_openrouter_billing(resp)
        assert provider is None


class TestPriceTableRefresh:
    """The 2026-07-07 price refresh added/updated the current OpenRouter ids."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("openrouter/z-ai/glm-5.2", (0.90, 2.86)),
            ("openrouter/deepseek/deepseek-v4-flash", (0.09, 0.18)),
            ("openrouter/deepseek/deepseek-v4-pro", (0.435, 0.87)),
        ],
    )
    def test_refreshed_prices_present(self, model: str, expected: tuple[float, float]) -> None:
        assert PRICES_PER_1M[model] == expected

    def test_existing_entries_kept(self) -> None:
        # The refresh must not drop pre-existing pinned models.
        for model in (
            "openrouter/z-ai/glm-4.6",
            "openrouter/openai/gpt-4o",
            "openrouter/openai/gpt-4o-mini",
            "openai/gpt-5-mini",
        ):
            assert model in PRICES_PER_1M
