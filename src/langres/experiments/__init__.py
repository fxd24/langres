"""Import-light contracts for reproducible entity-resolution experiments."""

from langres.experiments.identity import (
    AttemptIdentity,
    CacheIdentity,
    CacheIdentityInput,
    EvaluationIdentity,
    RecipeIdentity,
    SourceState,
    compute_cache_identity,
    compute_evaluation_identity,
    compute_recipe_identity,
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
    ReportConstraints,
)
from langres.experiments.statistics import (
    BootstrapInterval,
    PairedScore,
    SplitInstability,
    paired_entity_bootstrap,
    split_instability,
)

__all__ = [
    "AggregateRow",
    "AttemptIdentity",
    "BootstrapInterval",
    "CacheIdentity",
    "CacheIdentityInput",
    "CohortView",
    "EmbeddingFacts",
    "EvaluationIdentity",
    "EvaluationProtocol",
    "ExperimentReport",
    "ExperimentRun",
    "FunnelFacts",
    "IncompatibleProtocolError",
    "PairedScore",
    "PriceEstimate",
    "PriceSnapshot",
    "ProofCell",
    "RecipeIdentity",
    "ReportConstraints",
    "RuntimeFacts",
    "SourceState",
    "SplitInstability",
    "StageMeasurement",
    "TokenUsage",
    "compute_cache_identity",
    "compute_evaluation_identity",
    "compute_recipe_identity",
    "expand_official_proof_matrix",
    "paired_entity_bootstrap",
    "split_instability",
]
