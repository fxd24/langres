"""
langres.core: Low-level API for entity resolution.

This module provides the foundational primitives for building custom
entity resolution pipelines.
"""

from langres.core import metrics, optimizers
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
from langres.core.indexes import (
    FAISSIndex,
    FakeHybridVectorIndex,
    FakeVectorIndex,
    QdrantHybridIndex,
    VectorIndex,
)
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import (
    CompanySchema,
    EntityProtocol,
    ERCandidate,
    PairwiseJudgement,
)
from langres.core.module import Module
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
    "Blocker",
    "CandidateStats",
    "ClusterStats",
    "Clusterer",
    "CompanySchema",
    "Comparator",
    "ComparisonLevel",
    "ComparisonVector",
    "ComponentSpec",
    "EmbeddingProvider",
    "EntityProtocol",
    "ERCandidate",
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
    "StringComparator",
    "UnknownComponentType",
    "VectorBlocker",
    "VectorIndex",
    "WeightedAverageJudge",
]
