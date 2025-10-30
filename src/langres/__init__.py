"""
langres: A composable entity resolution framework.

This package provides a two-layer API for entity resolution:
- langres.core: Low-level primitives for custom pipelines
- langres.tasks: High-level task runners for common use cases (coming soon)
"""

from langres.core import CompanySchema, ERCandidate, PairwiseJudgement

__all__ = ["CompanySchema", "ERCandidate", "PairwiseJudgement"]

__version__ = "0.1.0"
