"""
langres.core: Low-level API for entity resolution.

This module provides the foundational primitives for building custom
entity resolution pipelines.
"""

from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement

__all__ = ["CompanySchema", "ERCandidate", "PairwiseJudgement"]
