"""
langres.core: Low-level API for entity resolution.

This module provides the foundational primitives for building custom
entity resolution pipelines.
"""

from langres.core import metrics
from langres.core.blocker import Blocker
from langres.core.clusterer import Clusterer
from langres.core.models import (
    CompanySchema,
    EntityProtocol,
    ERCandidate,
    PairwiseJudgement,
)
from langres.core.module import Module

__all__ = [
    "Blocker",
    "Clusterer",
    "CompanySchema",
    "EntityProtocol",
    "ERCandidate",
    "metrics",
    "Module",
    "PairwiseJudgement",
]
