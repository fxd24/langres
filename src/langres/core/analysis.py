# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: blocker-analysis functions moved to :mod:`langres.metrics.analysis`.

They analyse (score/diagnose) a blocker's output; they are not part of the
modelling contract, so they now live in the ``langres.metrics`` package. This
shim keeps the old ``langres.core.analysis`` import path working while the
refactor's final sweep repoints callers.
"""

from langres.metrics.analysis import (
    evaluate_blocker_detailed,
    extract_false_positives,
    extract_missed_matches,
)

__all__ = [
    "evaluate_blocker_detailed",
    "extract_false_positives",
    "extract_missed_matches",
]
