"""Versioned statistical protocol and deterministic proof-matrix expansion."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class EvaluationProtocol(BaseModel):
    """Everything that makes benchmark rows statistically comparable.

    The protocol owns split, repeat, threshold, metric, confidence-interval,
    and cohort policy. Runtime orchestration must not duplicate those knobs.
    Ordinary exploratory runs may be uncapped; the guarded ``official`` profile
    requires a positive spend cap before any work starts.
    """

    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    benchmark_ids: tuple[str, ...] = Field(min_length=1)
    dataset_fingerprints: dict[str, str] = Field(default_factory=dict)
    dataset_revisions: dict[str, str] = Field(default_factory=dict)
    split_ids: tuple[str, ...] = Field(min_length=1)
    fixed_test_set_id: str = Field(min_length=1)
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
    budget_usd: float | None = Field(default=None, gt=0.0)

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
        if self.publication_profile == "official" and self.budget_usd is None:
            raise ValueError("official publication_profile requires a positive budget_usd cap")
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
        split_seed: int = 0,
        budget_usd: float = 20.0,
    ) -> "EvaluationProtocol":
        """The guarded two-dataset protocol underlying the exact 18-cell proof."""
        return cls(
            benchmark_ids=benchmark_ids,
            split_ids=("official",),
            fixed_test_set_id="official-proof:test:v1",
            split_seeds=(split_seed,),
            stochastic_repeats=3,
            threshold_split_id="validation",
            test_split_id="test",
            hardware_cohort="official-declared",
            benchmark_version="1",
            publication_profile="official",
            budget_usd=budget_usd,
        )

    def evaluation_payload(self) -> dict[str, Any]:
        """Canonical statistical question; execution-budget policy is excluded."""
        return self.model_dump(
            mode="json",
            exclude={"budget_usd", "publication_profile"},
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
    if len(protocol.benchmark_ids) != 2:
        raise ValueError("the official proof matrix requires exactly two benchmark_ids")
    if len(protocol.split_ids) != 1 or len(protocol.split_seeds) != 1:
        raise ValueError("the official proof matrix requires one split_id and one split seed")
    if protocol.stochastic_repeats != 3:
        raise ValueError("the official proof matrix requires stochastic_repeats=3")

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
