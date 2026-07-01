"""Tests for langres.clients.openrouter (mock-based, $0 — no network, no spend)."""

import logging
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
    patch_litellm_prices,
    per_token_worst_price,
    register_runtime_model_price,
)
from langres.core.models import PairwiseJudgement

GLM = "openrouter/z-ai/glm-5.2"


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
