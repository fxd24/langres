# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: the pipeline debugger moved to :mod:`langres.metrics.debugging`.

It is diagnostics (it *inspects* a run), not a modelling contract, so it now
lives in the ``langres.metrics`` package. This shim keeps the old
``langres.core.debugging`` import path working while the refactor's final sweep
repoints callers.
"""

from langres.metrics.debugging import (
    CandidateStats,
    ClusterStats,
    ErrorExample,
    PipelineDebugger,
    ScoreStats,
)

__all__ = [
    "CandidateStats",
    "ClusterStats",
    "ErrorExample",
    "PipelineDebugger",
    "ScoreStats",
]
