"""Tiered, provider-neutral measurements and reproducible repricing."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from langres.core.usage import LLMUsage
from langres.experiments.protocol import FrozenDict, freeze_mapping

JsonScalar = str | int | float | bool | None
ProviderUsage = dict[str, dict[str, JsonScalar]]


class TokenUsage(BaseModel):
    """Inclusive token totals plus optional subsets; unknown is ``None``."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_output_tokens: int | None = Field(default=None, ge=0)
    provider_usage: ProviderUsage = Field(default_factory=dict)

    @field_validator("provider_usage", mode="after")
    @classmethod
    def _freeze_provider_usage(cls, value: ProviderUsage) -> FrozenDict:
        return freeze_mapping(value)

    @model_validator(mode="after")
    def _validate_subsets(self) -> "TokenUsage":
        cache_read = self.cache_read_input_tokens
        cache_creation = self.cache_creation_input_tokens
        if self.input_tokens is not None and cache_read is not None and cache_creation is not None:
            cache_total = cache_read + cache_creation
            if cache_total > self.input_tokens:
                raise ValueError("cache input subsets cannot exceed inclusive input_tokens")
        elif self.input_tokens is not None:
            for subset in (self.cache_read_input_tokens, self.cache_creation_input_tokens):
                if subset is not None and subset > self.input_tokens:
                    raise ValueError("cache input subset cannot exceed inclusive input_tokens")
        if (
            self.output_tokens is not None
            and self.reasoning_output_tokens is not None
            and self.reasoning_output_tokens > self.output_tokens
        ):
            raise ValueError("reasoning_output_tokens cannot exceed inclusive output_tokens")
        return self

    @classmethod
    def from_llm_usage(cls, usage: LLMUsage) -> "TokenUsage":
        """Losslessly migrate the existing always-measured ``LLMUsage`` vector."""
        serving: dict[str, JsonScalar] = {"provider": usage.provider, "model": usage.model}
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            reasoning_output_tokens=usage.reasoning_tokens,
            provider_usage={"langres": serving},
        )


class EmbeddingFacts(BaseModel):
    """Optional size and representation facts exposed by an embedder."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    dimensions: int | None = Field(default=None, ge=1)
    dtype: str | None = None
    quantization: str | None = None
    vectors_produced: int | None = Field(default=None, ge=0)
    bytes_per_vector: int | None = Field(default=None, ge=0)
    total_vector_bytes: int | None = Field(default=None, ge=0)
    parameter_count: int | None = Field(default=None, ge=0)
    artifact_bytes: int | None = Field(default=None, ge=0)
    loaded_memory_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_vector_bytes(self) -> "EmbeddingFacts":
        if (
            self.vectors_produced is not None
            and self.bytes_per_vector is not None
            and self.total_vector_bytes is not None
            and self.vectors_produced * self.bytes_per_vector != self.total_vector_bytes
        ):
            raise ValueError("total_vector_bytes must equal vectors_produced * bytes_per_vector")
        return self


class RuntimeFacts(BaseModel):
    """Hardware/runtime cohort facts attached to performance measurements."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    hardware_cohort: str = Field(min_length=1)
    host: str | None = None
    operating_system: str | None = None
    python_version: str | None = None
    langres_version: str | None = None
    cpu: str | None = None
    ram_bytes: int | None = Field(default=None, ge=0)
    accelerator: str | None = None
    accelerator_count: int | None = Field(default=None, ge=0)
    library_versions: dict[str, str] = Field(default_factory=dict)
    device: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    worker_count: int | None = Field(default=None, ge=1)

    @field_validator("library_versions", mode="after")
    @classmethod
    def _freeze_library_versions(cls, value: dict[str, str]) -> FrozenDict:
        return freeze_mapping(value)


class PriceEstimate(BaseModel):
    """A cost derived from immutable usage and pricing facts."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    amount: float | None = Field(default=None, ge=0.0)
    currency: str
    complete: bool
    missing: tuple[str, ...] = ()


class PriceSnapshot(BaseModel):
    """Rates actually used, retained so tokens can be repriced later."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    provider: str
    model: str
    currency: Literal["USD"] = "USD"
    effective_at: datetime | None = None
    captured_at: datetime
    input_usd_per_token: float | None = Field(default=None, ge=0.0)
    output_usd_per_token: float | None = Field(default=None, ge=0.0)
    cache_read_input_usd_per_token: float | None = Field(default=None, ge=0.0)
    cache_creation_input_usd_per_token: float | None = Field(default=None, ge=0.0)
    request_usd: float | None = Field(default=None, ge=0.0)
    cache_rate_policy: Literal["specialized_required", "base_rate_fallback"] = (
        "specialized_required"
    )
    source: Literal["provider", "user", "catalog"]
    source_reference: str | None = None

    def reprice(self, usage: TokenUsage, *, requests: int | None) -> PriceEstimate:
        """Price inclusive totals, replacing base input rates for known cache subsets."""
        missing: list[str] = []
        amount = 0.0

        if usage.input_tokens is None:
            missing.append("input_tokens")
        elif self.input_usd_per_token is None:
            if usage.input_tokens:
                missing.append("input_usd_per_token")
        else:
            amount += usage.input_tokens * self.input_usd_per_token
            amount, cache_missing = self._adjust_cache_rate(
                amount,
                usage.cache_read_input_tokens,
                self.cache_read_input_usd_per_token,
                "cache_read_input_tokens",
            )
            missing.extend(cache_missing)
            amount, cache_missing = self._adjust_cache_rate(
                amount,
                usage.cache_creation_input_tokens,
                self.cache_creation_input_usd_per_token,
                "cache_creation_input_tokens",
            )
            missing.extend(cache_missing)

        if usage.output_tokens is None:
            missing.append("output_tokens")
        elif self.output_usd_per_token is None:
            if usage.output_tokens:
                missing.append("output_usd_per_token")
        else:
            amount += usage.output_tokens * self.output_usd_per_token

        if requests is None:
            if self.request_usd is not None:
                missing.append("requests")
        elif requests < 0:
            raise ValueError("requests must be non-negative")
        elif self.request_usd is not None:
            amount += requests * self.request_usd

        unique_missing = tuple(dict.fromkeys(missing))
        return PriceEstimate(
            amount=None if unique_missing else amount,
            currency=self.currency,
            complete=not unique_missing,
            missing=unique_missing,
        )

    def _adjust_cache_rate(
        self,
        amount: float,
        tokens: int | None,
        specialized_rate: float | None,
        token_field: str,
    ) -> tuple[float, list[str]]:
        if specialized_rate is None:
            if tokens in (None, 0):
                if tokens is None and self.cache_rate_policy == "specialized_required":
                    return amount, [token_field]
                return amount, []
            if self.cache_rate_policy == "base_rate_fallback":
                return amount, []
            return amount, [token_field.replace("_tokens", "_usd_per_token")]
        if tokens is None:
            return amount, [token_field]
        if self.input_usd_per_token is None:  # handled by caller
            return amount, []
        return amount + tokens * (specialized_rate - self.input_usd_per_token), []


class StageMeasurement(BaseModel):
    """One stage's Tier-0 facts plus optional capability extensions."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    version: Literal[1] = 1
    stage_id: str = Field(min_length=1)
    operation_kind: str = Field(min_length=1)
    wall_seconds: float = Field(ge=0.0)
    cpu_seconds: float | None = Field(default=None, ge=0.0)
    items_in: int | None = Field(default=None, ge=0)
    items_out: int | None = Field(default=None, ge=0)
    pairs_in: int | None = Field(default=None, ge=0)
    pairs_out: int | None = Field(default=None, ge=0)
    throughput_per_second: float | None = Field(default=None, ge=0.0)
    p50_item_latency_seconds: float | None = Field(default=None, ge=0.0)
    p95_item_latency_seconds: float | None = Field(default=None, ge=0.0)
    temperature: Literal["cold", "warm"] | None = None
    resource_slot: str | None = None
    resource_id: str | None = None
    usage: TokenUsage | None = None
    embedding: EmbeddingFacts | None = None
    runtime: RuntimeFacts | None = None
    price: PriceSnapshot | None = None
    observed_usd: float | None = Field(default=None, ge=0.0)
    derived_usd: float | None = Field(default=None, ge=0.0)
    external_calls: int | None = Field(default=None, ge=0)
    cache_hit: bool | None = None
    warnings: tuple[str, ...] = ()


class FunnelFacts(BaseModel):
    """Counts through the ER funnel; unsupported facts stay ``None``."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    possible_pairs: int | None = Field(default=None, ge=0)
    retrieved_pairs: int | None = Field(default=None, ge=0)
    pairs_after_select: tuple[int, ...] = ()
    reranker_pairs: int | None = Field(default=None, ge=0)
    llm_pairs: int | None = Field(default=None, ge=0)
    parsed_abstentions: int | None = Field(default=None, ge=0)
    selected_match_edges: int | None = Field(default=None, ge=0)
    clusters_produced: int | None = Field(default=None, ge=0)
