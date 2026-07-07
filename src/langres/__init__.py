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
- ``NoJudgeAvailableError`` / ``BudgetExceeded``: the two exceptions a
  front-door user must catch (fail-fast ``judge="auto"``; the spend cap).
"""

from langres.clients.openrouter import BudgetExceeded
from langres.core import (
    CompanySchema,
    ERCandidate,
    JudgementLog,
    PairwiseJudgement,
    Resolver,
    ReviewQueue,
    select_for_review,
)
from langres.core.presets import NoJudgeAvailableError
from langres.verbs import LinkVerdict, dedupe, link

__all__ = [
    "BudgetExceeded",
    "CompanySchema",
    "ERCandidate",
    "JudgementLog",
    "LinkVerdict",
    "NoJudgeAvailableError",
    "PairwiseJudgement",
    "Resolver",
    "ReviewQueue",
    "dedupe",
    "link",
    "select_for_review",
]

__version__ = "0.2.0"
