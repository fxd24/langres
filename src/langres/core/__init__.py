"""
langres.core: Low-level API for entity resolution.

This module provides the foundational primitives for building custom
entity resolution pipelines.
"""

from langres.core import benchmark, metrics, optimizers
from langres.core.adapters.glinker import GLinkerAdapter
from langres.core.blocker import Blocker
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator, StringComparator
from langres.core.debugging import (
    CandidateStats,
    ClusterStats,
    ErrorExample,
    PipelineDebugger,
    ScoreStats,
)
from langres.core.embeddings import (
    EmbeddingProvider,
    FakeEmbedder,
    FakeSparseEmbedder,
    FastEmbedSparseEmbedder,
    SentenceTransformerEmbedder,
    SparseEmbeddingProvider,
)
from langres.core.feature import ComparisonLevel, ComparisonVector, FeatureSpec
from langres.core.fit import SupervisedFitMixin, UnsupervisedFitMixin
from langres.core.groups import ERCandidateGroup, derive_groups_from_pairs
from langres.core.indexes import (
    FAISSIndex,
    FakeHybridVectorIndex,
    FakeVectorIndex,
    QdrantHybridIndex,
    VectorIndex,
)
from langres.core.judges.embedding_score import EmbeddingScoreJudge
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import (
    CompanySchema,
    EntityProtocol,
    ERCandidate,
    PairwiseJudgement,
)
from langres.core.module import GroupwiseModule, Module, stamp_group_cost

# Importing LLMJudge here ensures its ``@register("llm_judge")`` runs on plain
# ``import langres.core`` — so a fresh process doing ``Resolver.load(path)`` on
# an LLMJudge artifact finds the type in the registry (mirrors WeightedAverageJudge
# above). It also makes ``from langres.core import LLMJudge`` work.
from langres.core.modules.llm_judge import LLMJudge
from langres.core.registry import (
    SchemaNotRegistered,
    UnknownComponentType,
    get_component,
    get_schema,
    register,
    register_schema,
)
from langres.core.resolver import Resolver
from langres.core.serialization import (
    ARTIFACT_VERSION,
    ArtifactManifest,
    ComponentSpec,
    SerializableState,
)

__all__ = [
    "ARTIFACT_VERSION",
    "AllPairsBlocker",
    "ArtifactManifest",
    "benchmark",
    "Blocker",
    "CandidateStats",
    "ClusterStats",
    "Clusterer",
    "CompanySchema",
    "Comparator",
    "ComparisonLevel",
    "ComparisonVector",
    "ComponentSpec",
    "derive_groups_from_pairs",
    "EmbeddingProvider",
    "EmbeddingScoreJudge",
    "EntityProtocol",
    "ERCandidate",
    "ERCandidateGroup",
    "ErrorExample",
    "FAISSIndex",
    "FakeEmbedder",
    "FakeHybridVectorIndex",
    "FakeSparseEmbedder",
    "FakeVectorIndex",
    "FastEmbedSparseEmbedder",
    "FeatureSpec",
    "get_component",
    "get_schema",
    "GLinkerAdapter",
    "GroupwiseModule",
    "LLMJudge",
    "metrics",
    "Module",
    "optimizers",
    "PairwiseJudgement",
    "PipelineDebugger",
    "QdrantHybridIndex",
    "register",
    "register_schema",
    "Resolver",
    "ScoreStats",
    "SchemaNotRegistered",
    "SentenceTransformerEmbedder",
    "SerializableState",
    "SparseEmbeddingProvider",
    "stamp_group_cost",
    "StringComparator",
    "SupervisedFitMixin",
    "UnknownComponentType",
    "UnsupervisedFitMixin",
    "VectorBlocker",
    "VectorIndex",
    "WeightedAverageJudge",
]
