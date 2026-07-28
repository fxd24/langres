# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: the pipeline debugger moved to :mod:`langres.metrics.debugging`.

It is diagnostics (it *inspects* a run), not a modelling contract, so it now
lives in the ``langres.metrics`` package. This shim keeps the old
``langres.core.debugging`` import path working while the refactor's final sweep
repoints callers.
"""

# `# pragma: no cover`, as on every other W2 shim (see core/harvest.py): a
# re-export owns no contract. `langres.metrics.debugging` is itself inside the
# `langres.core` contract coverage gate, so measuring this redirect would book a
# miss against code already covered at its real home. Goes away with this file
# in the W2 sweep.
from langres.metrics.debugging import (  # pragma: no cover
    CandidateStats,
    ClusterStats,
    ErrorExample,
    PipelineDebugger,
    ScoreStats,
)

__all__ = [  # pragma: no cover
    "CandidateStats",
    "ClusterStats",
    "ErrorExample",
    "PipelineDebugger",
    "ScoreStats",
]
