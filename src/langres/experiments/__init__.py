"""Import-light contracts for reproducible entity-resolution experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langres.experiments.identity import (
    AttemptIdentity,
    CacheIdentity,
    CacheIdentityInput,
    EvaluationIdentity,
    RecipeIdentity,
    ResourceSlotIdentity,
    SourceState,
    compute_cache_identity,
    compute_evaluation_identity,
    compute_recipe_identity,
    detect_source_state,
)
from langres.experiments.measurements import (
    EmbeddingFacts,
    FunnelFacts,
    PriceEstimate,
    PriceSnapshot,
    RuntimeFacts,
    StageMeasurement,
    TokenUsage,
)
from langres.experiments.cache import (
    ScoreCacheError,
    StageArtifactManifest,
    StageArtifactStore,
    ordered_input_fingerprint,
)
from langres.experiments.protocol import (
    EvaluationProtocol,
    ProofCell,
    expand_official_proof_matrix,
)
from langres.experiments.report import (
    AggregateRow,
    CohortView,
    ExperimentReport,
    ExperimentRun,
    IncompatibleProtocolError,
    MetricConfidenceInterval,
    ParetoRow,
    ReportConstraints,
)
from langres.experiments.statistics import (
    BootstrapInterval,
    PairedScore,
    SplitInstability,
    paired_entity_bootstrap,
    split_instability,
)
if TYPE_CHECKING:
    from langres.experiments.runner import (
        ArchitectureFactory,
        Experiment,
        ExperimentConfigurationError,
    )

__all__ = [
    "AggregateRow",
    "ArchitectureFactory",
    "AttemptIdentity",
    "BootstrapInterval",
    "CacheIdentity",
    "CacheIdentityInput",
    "CohortView",
    "EmbeddingFacts",
    "Experiment",
    "ExperimentConfigurationError",
    "EvaluationIdentity",
    "EvaluationProtocol",
    "ExperimentReport",
    "ExperimentRun",
    "FunnelFacts",
    "IncompatibleProtocolError",
    "MetricConfidenceInterval",
    "PairedScore",
    "ParetoRow",
    "PriceEstimate",
    "PriceSnapshot",
    "ProofCell",
    "RecipeIdentity",
    "ResourceSlotIdentity",
    "ReportConstraints",
    "RuntimeFacts",
    "ScoreCacheError",
    "SourceState",
    "SplitInstability",
    "StageMeasurement",
    "StageArtifactManifest",
    "StageArtifactStore",
    "TokenUsage",
    "compute_cache_identity",
    "compute_evaluation_identity",
    "compute_recipe_identity",
    "detect_source_state",
    "expand_official_proof_matrix",
    "flatten_numeric",
    "ordered_input_fingerprint",
    "paired_entity_bootstrap",
    "split_instability",
]


def __getattr__(name: str) -> Any:
    """Resolve the benchmark-dependent runner only when explicitly requested."""
    if name in {
        "ArchitectureFactory",
        "Experiment",
        "ExperimentConfigurationError",
        "flatten_numeric",
    }:
        from langres.experiments import runner

        value = getattr(runner, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
