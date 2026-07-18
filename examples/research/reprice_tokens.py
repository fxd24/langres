"""Reprice stored token facts without repeating an LLM call."""

from datetime import UTC, datetime

from langres.experiments import PriceSnapshot, TokenUsage


def main() -> None:
    usage = TokenUsage(
        input_tokens=12_000,
        output_tokens=800,
        cache_read_input_tokens=2_000,
        cache_creation_input_tokens=0,
    )
    current_rates = PriceSnapshot(
        provider="example-provider",
        model="example-model",
        captured_at=datetime(2026, 7, 18, tzinfo=UTC),
        input_usd_per_token=2.0 / 1_000_000,
        output_usd_per_token=8.0 / 1_000_000,
        cache_read_input_usd_per_token=0.2 / 1_000_000,
        cache_creation_input_usd_per_token=2.0 / 1_000_000,
        source="user",
        source_reference="provider price page captured by the researcher",
    )
    estimate = current_rates.reprice(usage, requests=1)
    print(estimate.model_dump())


if __name__ == "__main__":
    main()
