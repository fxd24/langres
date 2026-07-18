"""Versioned statistical protocol and deterministic proof-matrix expansion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Never, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AggregationRule = Literal["mean", "median"]
ConfidenceIntervalMethod = Literal["paired_entity_bootstrap", "none"]
PublicationProfile = Literal["exploratory", "official"]

OFFICIAL_TOPOLOGIES = (
    "Retrieve",
    "RetrieveRerank",
    "RetrieveLLM",
    "RetrieveRerankLLM",
    "CustomTopology",
)
STOCHASTIC_TOPOLOGIES = frozenset({"RetrieveLLM", "RetrieveRerankLLM"})
OFFICIAL_PAID_PROOF_BUDGET_USD = 20.0
MIN_BOOTSTRAP_SAMPLES = 100


class FrozenDict(dict[str, Any]):
    """A JSON-serializable dict whose complete nested value tree is immutable."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("experiment identity mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    setdefault = _immutable
    update = _immutable

    def popitem(self) -> tuple[str, Any]:
        self._raise_immutable()

    def __ior__(self, value: Any) -> Self:  # type: ignore[override,misc]
        del value
        self._raise_immutable()

    @staticmethod
    def _raise_immutable() -> Never:
        raise TypeError("experiment identity mappings are immutable")


def deep_freeze(value: Any) -> Any:
    """Recursively freeze JSON-shaped values while preserving stable serialization."""
    if isinstance(value, Mapping):
        return FrozenDict({str(key): deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise ValueError("sets are not valid JSON protocol values; use an ordered list")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("experiment identity values must be finite")
    return value


def freeze_mapping(value: Mapping[str, Any]) -> FrozenDict:
    """Pydantic field helper for deep immutable, JSON-stable mappings."""
    frozen = deep_freeze(value)
    assert isinstance(frozen, FrozenDict)
    return frozen


class EvaluationProtocol(BaseModel):
    """Everything that makes benchmark rows statistically comparable.

    The protocol owns split, repeat, threshold, metric, confidence-interval,
    and cohort policy. Runtime orchestration must not duplicate those knobs.
    Ordinary and zero-cost official runs may be uncapped. Official publication
    requires pinned data/test identity; the separate paid-proof policy requires
    its exact USD 20 cap before work starts.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False, validate_default=True)

    version: Literal[1] = 1
    benchmark_ids: tuple[str, ...] = Field(min_length=1)
    dataset_fingerprints: dict[str, str] = Field(default_factory=dict)
    dataset_revisions: dict[str, str] = Field(default_factory=dict)
    split_ids: tuple[str, ...] = Field(min_length=1)
    fixed_test_set_id: str | None = Field(default=None, min_length=1)
    test_set_identities: dict[str, str] = Field(default_factory=dict)
    split_seeds: tuple[int, ...] = Field(min_length=1)
    deterministic_resources: dict[str, Any] = Field(default_factory=dict)
    stochastic_repeats: int = Field(default=1, ge=1)
    aggregation: AggregationRule = "mean"
    threshold_split_id: str = Field(min_length=1)
    test_split_id: str = Field(min_length=1)
    threshold_grid: tuple[float, ...] = (0.5,)
    pair_metrics: tuple[str, ...] = ("pair_f1",)
    cluster_metrics: tuple[str, ...] = ("bcubed_f1",)
    confidence_interval_method: ConfidenceIntervalMethod = "paired_entity_bootstrap"
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap_samples: int = Field(default=1000, ge=1)
    hardware_cohort: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    publication_profile: PublicationProfile = "exploratory"
    paid_proof: bool = False
    budget_usd: float | None = Field(default=None, gt=0.0)

    @field_validator(
        "dataset_fingerprints",
        "dataset_revisions",
        "test_set_identities",
        "deterministic_resources",
        mode="after",
    )
    @classmethod
    def _freeze_mappings(cls, value: dict[str, Any]) -> FrozenDict:
        return freeze_mapping(value)

    @model_validator(mode="after")
    def _validate_protocol(self) -> "EvaluationProtocol":
        if self.threshold_split_id == self.test_split_id:
            raise ValueError(
                "threshold_split_id and test_split_id must differ so the test split stays untouched"
            )
        if not self.threshold_grid:
            raise ValueError("threshold_grid must contain at least one candidate")
        if len(set(self.benchmark_ids)) != len(self.benchmark_ids):
            raise ValueError("benchmark_ids must be unique")
        if len(set(self.split_ids)) != len(self.split_ids):
            raise ValueError("split_ids must be unique")
        if len(set(self.split_seeds)) != len(self.split_seeds):
            raise ValueError("split_seeds must be unique")
        if not self.pair_metrics and not self.cluster_metrics:
            raise ValueError("at least one pair or cluster metric is required")
        if (
            self.confidence_interval_method == "paired_entity_bootstrap"
            and self.bootstrap_samples < MIN_BOOTSTRAP_SAMPLES
        ):
            raise ValueError(
                "paired_entity_bootstrap requires at least "
                f"{MIN_BOOTSTRAP_SAMPLES} bootstrap samples"
            )
        if self.paid_proof:
            if self.publication_profile != "official":
                raise ValueError("paid_proof requires publication_profile='official'")
            if self.budget_usd != OFFICIAL_PAID_PROOF_BUDGET_USD:
                raise ValueError("official paid proof requires a budget cap of exactly USD 20")
        if self.publication_profile == "official":
            missing_data = [
                benchmark
                for benchmark in self.benchmark_ids
                if not self.dataset_fingerprints.get(benchmark)
                and not self.dataset_revisions.get(benchmark)
            ]
            if missing_data:
                raise ValueError(
                    "official publication requires a dataset fingerprint or revision for "
                    f"every benchmark; missing {missing_data}"
                )
            per_dataset_test_ids = all(
                self.test_set_identities.get(benchmark) for benchmark in self.benchmark_ids
            )
            if self.fixed_test_set_id is None and not per_dataset_test_ids:
                raise ValueError(
                    "official publication requires a composite fixed_test_set_id or a "
                    "test-set identity for every benchmark"
                )
        elif self.fixed_test_set_id is None and not all(
            self.test_set_identities.get(benchmark) for benchmark in self.benchmark_ids
        ):
            raise ValueError(
                "provide fixed_test_set_id or a non-empty test_set_identity for every benchmark"
            )
        return self

    @classmethod
    def smoke(cls, *, seed: int = 0) -> "EvaluationProtocol":
        """A deterministic, uncapped, zero-network CI protocol."""
        return cls(
            benchmark_ids=("tiny_fixture",),
            split_ids=("smoke",),
            fixed_test_set_id="tiny_fixture:test:v1",
            split_seeds=(seed,),
            stochastic_repeats=1,
            threshold_split_id="validation",
            test_split_id="test",
            threshold_grid=(0.5,),
            confidence_interval_method="none",
            bootstrap_samples=1,
            hardware_cohort="local-smoke",
            benchmark_version="1",
        )

    @classmethod
    def official_proof(
        cls,
        *,
        benchmark_ids: tuple[str, str],
        dataset_fingerprints: Mapping[str, str] | None = None,
        dataset_revisions: Mapping[str, str] | None = None,
        fixed_test_set_id: str | None = None,
        test_set_identities: Mapping[str, str] | None = None,
        split_seed: int = 0,
        budget_usd: float = OFFICIAL_PAID_PROOF_BUDGET_USD,
    ) -> "EvaluationProtocol":
        """The guarded two-dataset protocol underlying the exact 18-cell proof."""
        return cls(
            benchmark_ids=benchmark_ids,
            dataset_fingerprints=dict(dataset_fingerprints or {}),
            dataset_revisions=dict(dataset_revisions or {}),
            split_ids=("official",),
            fixed_test_set_id=fixed_test_set_id,
            test_set_identities=dict(test_set_identities or {}),
            split_seeds=(split_seed,),
            stochastic_repeats=3,
            threshold_split_id="validation",
            test_split_id="test",
            hardware_cohort="official-declared",
            benchmark_version="1",
            publication_profile="official",
            paid_proof=True,
            budget_usd=budget_usd,
        )

    def evaluation_payload(self) -> dict[str, Any]:
        """Canonical statistical question; execution-budget policy is excluded."""
        return self.model_dump(
            mode="json",
            exclude={"budget_usd", "publication_profile", "paid_proof"},
        )


class ProofCell(BaseModel):
    """One planned proof cell before retries."""

    model_config = ConfigDict(frozen=True)

    topology: str
    benchmark_id: str
    split_id: str
    split_seed: int
    repeat_index: int = Field(ge=0)
    retry_index: int = 0

    @property
    def cell_id(self) -> str:
        """Human-readable stable cell identity."""
        return (
            f"{self.topology}:{self.benchmark_id}:{self.split_id}:"
            f"{self.split_seed}:repeat-{self.repeat_index}"
        )


def expand_official_proof_matrix(protocol: EvaluationProtocol) -> tuple[ProofCell, ...]:
    """Expand five topologies x two datasets to exactly 18 pre-retry cells."""
    if protocol.publication_profile != "official":
        raise ValueError("the official proof matrix requires publication_profile='official'")
    if not protocol.paid_proof:
        raise ValueError("the official proof matrix requires paid_proof=True")
    if len(protocol.benchmark_ids) != 2:
        raise ValueError("the official proof matrix requires exactly two benchmark_ids")
    if len(protocol.split_ids) != 1 or len(protocol.split_seeds) != 1:
        raise ValueError("the official proof matrix requires one split_id and one split seed")
    if protocol.stochastic_repeats != 3:
        raise ValueError("the official proof matrix requires stochastic_repeats=3")
    missing_data = [
        benchmark_id
        for benchmark_id in protocol.benchmark_ids
        if not protocol.dataset_fingerprints.get(benchmark_id)
        and not protocol.dataset_revisions.get(benchmark_id)
    ]
    if missing_data:
        raise ValueError(
            "the official proof matrix requires non-empty dataset provenance for "
            f"every benchmark; missing {missing_data}"
        )
    if protocol.fixed_test_set_id is None and not all(
        protocol.test_set_identities.get(benchmark_id) for benchmark_id in protocol.benchmark_ids
    ):
        raise ValueError(
            "the official proof matrix requires a non-empty composite fixed_test_set_id "
            "or test-set identity for every benchmark"
        )

    cells: list[ProofCell] = []
    for topology in OFFICIAL_TOPOLOGIES:
        repeats = protocol.stochastic_repeats if topology in STOCHASTIC_TOPOLOGIES else 1
        for benchmark_id in protocol.benchmark_ids:
            for repeat_index in range(repeats):
                cells.append(
                    ProofCell(
                        topology=topology,
                        benchmark_id=benchmark_id,
                        split_id=protocol.split_ids[0],
                        split_seed=protocol.split_seeds[0],
                        repeat_index=repeat_index,
                    )
                )
    if len(cells) != 18:  # pragma: no cover - protects edits to the named matrix
        raise AssertionError(f"official proof matrix must contain 18 cells, got {len(cells)}")
    return tuple(cells)
