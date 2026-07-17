"""The flywheel loop: log every call -> pick the margin -> harvest the labels.

See ``langres.core._exports`` for the fragment contract.
"""

from langres.curation.harvest import (
    Correction,
    CorrectionLog,
    LabeledPair,
    derive_threshold_from_pairs,
    harvest_labeled_pairs,
)
from langres.core.judgement_log import JudgementLog, LoggingMatcher
from langres.curation.review import ReviewItem, ReviewQueue, select_for_review

__all__ = [
    "Correction",
    "CorrectionLog",
    "derive_threshold_from_pairs",
    "harvest_labeled_pairs",
    "JudgementLog",
    "LabeledPair",
    "LoggingMatcher",
    "ReviewItem",
    "ReviewQueue",
    "select_for_review",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
