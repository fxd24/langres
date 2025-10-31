"""
langres.core: Low-level API for entity resolution.

This module provides the foundational primitives for building custom
entity resolution pipelines.
"""

from langres.core import metrics
from langres.core.blocker import Blocker
from langres.core.clusterer import Clusterer
from langres.core.embeddings import (
    EmbeddingProvider,
    FakeEmbedder,
    SentenceTransformerEmbedder,
)
from langres.core.models import (
    CompanySchema,
    EntityProtocol,
    ERCandidate,
    PairwiseJudgement,
)
from langres.core.module import Module
from langres.core.vector_index import FAISSIndex, FakeVectorIndex, VectorIndex

__all__ = [
    "Blocker",
    "Clusterer",
    "CompanySchema",
    "EmbeddingProvider",
    "EntityProtocol",
    "ERCandidate",
    "FAISSIndex",
    "FakeEmbedder",
    "FakeVectorIndex",
    "metrics",
    "Module",
    "PairwiseJudgement",
    "SentenceTransformerEmbedder",
    "VectorIndex",
]
