"""
langres.core: Low-level API for entity resolution.

This module provides the foundational primitives for building custom
entity resolution pipelines.
"""

from langres.core import metrics, optimizers
from langres.core.blocker import Blocker
from langres.core.clusterer import Clusterer
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
from langres.core.indexes import (
    FAISSIndex,
    FakeHybridVectorIndex,
    FakeVectorIndex,
    QdrantHybridIndex,
    VectorIndex,
)
from langres.core.models import (
    CompanySchema,
    EntityProtocol,
    ERCandidate,
    PairwiseJudgement,
)
from langres.core.module import Module

__all__ = [
    "Blocker",
    "CandidateStats",
    "ClusterStats",
    "Clusterer",
    "CompanySchema",
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
    "metrics",
    "Module",
    "optimizers",
    "PairwiseJudgement",
    "PipelineDebugger",
    "QdrantHybridIndex",
    "ScoreStats",
    "SentenceTransformerEmbedder",
    "SparseEmbeddingProvider",
    "VectorIndex",
]
