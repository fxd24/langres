"""
langres: A composable entity resolution framework.

This package provides:
- ``link`` / ``dedupe``: the two-verb DX layer (schema-optional, judge="auto"
  by default, spend-capped) -- see ``langres.verbs``.
- ``langres.core``: Low-level primitives for custom pipelines (``Resolver``,
  ``Blocker``, ``Module``, ``Clusterer``, ...).
- ``JudgementLog``: opt-in JSONL signal log for judge calls, wired via
  ``log=`` on ``link``/``dedupe`` -- the flywheel inlet (see
  ``langres.core.judgement_log``).
"""

from langres.core import CompanySchema, ERCandidate, JudgementLog, PairwiseJudgement, Resolver
from langres.verbs import LinkVerdict, dedupe, link

__all__ = [
    "CompanySchema",
    "ERCandidate",
    "JudgementLog",
    "LinkVerdict",
    "PairwiseJudgement",
    "Resolver",
    "dedupe",
    "link",
]

__version__ = "0.1.0"
