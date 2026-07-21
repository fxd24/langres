from __future__ import annotations

import pytest
from pydantic import ValidationError

from langres.core.usage import LLMUsage
from langres.experiments import (
    EmbeddingFacts,
    FunnelFacts,
    PriceSnapshot,
    RuntimeFacts,
    StageMeasurement,
    TokenUsage,
)


def test_token_usage_preserves_unknown_as_none_and_measured_zero_as_zero() -> None:
    unknown = TokenUsage()
    measured = TokenUsage(input_tokens=0, output_tokens=0)

    assert unknown.input_tokens is None
    assert unknown.output_tokens is None
    assert measured.input_tokens == 0
    assert measured.output_tokens == 0


def test_llm_usage_migrates_losslessly() -> None:
    old = LLMUsage(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=10,
        reasoning_tokens=5,
        provider="openrouter",
        model="provider/model",
    )

    usage = TokenUsage.from_llm_usage(old)

    assert usage.input_tokens == 100
    assert usage.reasoning_output_tokens == 5
    assert usage.provider_usage == {
        "langres": {"provider": "openrouter", "model": "provider/model"}
    }


def test_token_subsets_cannot_exceed_inclusive_totals() -> None:
    with pytest.raises(ValidationError, match="input_tokens"):
        TokenUsage(input_tokens=10, cache_read_input_tokens=11)

    with pytest.raises(ValidationError, match="output_tokens"):
        TokenUsage(output_tokens=2, reasoning_output_tokens=3)


def test_price_snapshot_reprices_without_inference() -> None:
    price = PriceSnapshot(
        provider="provider",
        model="model",
        captured_at="2026-07-18T12:00:00Z",
        input_usd_per_token=0.001,
        output_usd_per_token=0.002,
        cache_read_input_usd_per_token=0.0001,
        request_usd=0.01,
        source="user",
    )
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=0,
    )

    estimate = price.reprice(usage, requests=2)

    assert estimate.complete is True
    # input_tokens is inclusive of cache reads, so the cache rate replaces
    # (rather than adds to) the base input rate for those 30 tokens.
    assert estimate.amount == pytest.approx(0.133)
    assert estimate.currency == "USD"


def test_price_snapshot_rejects_non_usd_currency_for_usd_named_rates() -> None:
    with pytest.raises(ValidationError, match="USD"):
        PriceSnapshot(
            provider="provider",
            model="model",
            currency="EUR",
            captured_at="2026-07-18T12:00:00Z",
            source="user",
        )


def test_repricing_preserves_unknown_instead_of_manufacturing_zero() -> None:
    price = PriceSnapshot(
        provider="provider",
        model="model",
        captured_at="2026-07-18T12:00:00Z",
        input_usd_per_token=0.001,
        source="provider",
    )

    estimate = price.reprice(TokenUsage(), requests=None)

    assert estimate.amount is None
    assert estimate.complete is False
    assert "input_tokens" in estimate.missing


def test_cache_tokens_need_specialized_rate_unless_fallback_is_explicit() -> None:
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=0,
        cache_read_input_tokens=25,
        cache_creation_input_tokens=0,
    )
    strict = PriceSnapshot(
        provider="provider",
        model="model",
        captured_at="2026-07-18T12:00:00Z",
        input_usd_per_token=0.001,
        output_usd_per_token=0.002,
        source="provider",
    )
    fallback = strict.model_copy(update={"cache_rate_policy": "base_rate_fallback"})

    strict_estimate = strict.reprice(usage, requests=0)
    fallback_estimate = fallback.reprice(usage, requests=0)

    assert strict_estimate.amount is None
    assert "cache_read_input_usd_per_token" in strict_estimate.missing
    assert fallback_estimate.complete is True
    assert fallback_estimate.amount == pytest.approx(0.1)


def test_zero_inclusive_input_makes_unknown_cache_subsets_exactly_zero() -> None:
    price = PriceSnapshot(
        provider="provider",
        model="model",
        captured_at="2026-07-18T12:00:00Z",
        input_usd_per_token=0.001,
        output_usd_per_token=0.002,
        source="provider",
    )

    estimate = price.reprice(TokenUsage(input_tokens=0, output_tokens=0), requests=0)

    assert estimate.complete is True
    assert estimate.amount == 0.0
    assert estimate.missing == ()


def test_embedding_runtime_stage_and_funnel_facts_round_trip() -> None:
    embedding = EmbeddingFacts(
        dimensions=384,
        dtype="float32",
        vectors_produced=10,
        bytes_per_vector=1536,
        total_vector_bytes=15360,
    )
    runtime = RuntimeFacts(
        hardware_cohort="m2-cpu",
        device="cpu",
        dtype="float32",
        batch_size=16,
    )
    stage = StageMeasurement(
        stage_id="retrieve",
        operation_kind="score",
        wall_seconds=0.5,
        items_in=10,
        pairs_out=20,
        throughput_per_second=40.0,
        throughput_unit="pairs",
        embedding=embedding,
        runtime=runtime,
        cache_hit=False,
    )
    funnel = FunnelFacts(possible_pairs=45, retrieved_pairs=20, llm_pairs=0)

    assert StageMeasurement.model_validate_json(stage.model_dump_json()) == stage
    assert stage.throughput_unit == "pairs"
    assert funnel.llm_pairs == 0
    assert funnel.reranker_pairs is None
