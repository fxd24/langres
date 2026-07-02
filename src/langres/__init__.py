"""
langres: A composable entity resolution framework.

This package provides:
- ``link`` / ``dedupe``: the three-verb DX layer (schema-optional, judge="auto"
  by default, spend-capped) -- see ``langres.verbs``.
- ``langres.core``: Low-level primitives for custom pipelines (``Resolver``,
  ``Blocker``, ``Module``, ``Clusterer``, ...).
"""

from langres.core import CompanySchema, ERCandidate, PairwiseJudgement, Resolver
from langres.verbs import LinkVerdict, dedupe, link

__all__ = [
    "CompanySchema",
    "ERCandidate",
    "LinkVerdict",
    "PairwiseJudgement",
    "Resolver",
    "dedupe",
    "link",
]

__version__ = "0.1.0"
