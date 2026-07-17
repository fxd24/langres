# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: the diagnostic example models moved to :mod:`langres.metrics.diagnostics`.

They describe a metric's error cases (false positives / missed matches); they
are not part of the modelling contract, so they now live in the
``langres.metrics`` package. This shim keeps the old ``langres.core.diagnostics``
import path working while the refactor's final sweep repoints callers.
"""

from langres.metrics.diagnostics import (
    DiagnosticExamples,
    FalsePositiveExample,
    MissedMatchExample,
)

__all__ = [
    "DiagnosticExamples",
    "FalsePositiveExample",
    "MissedMatchExample",
]
