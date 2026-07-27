# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: the diagnostic example models moved to :mod:`langres.metrics.diagnostics`.

They describe a metric's error cases (false positives / missed matches); they
are not part of the modelling contract, so they now live in the
``langres.metrics`` package. This shim keeps the old ``langres.core.diagnostics``
import path working while the refactor's final sweep repoints callers.
"""

# `# pragma: no cover`, as on every other W2 shim (see core/harvest.py): a
# re-export owns no contract. `langres.metrics.diagnostics` is itself inside the
# `langres.core` contract coverage gate, so measuring this redirect would book a
# miss against code already covered at its real home. Goes away with this file
# in the W2 sweep.
from langres.metrics.diagnostics import (  # pragma: no cover
    DiagnosticExamples,
    FalsePositiveExample,
    MissedMatchExample,
)

__all__ = [  # pragma: no cover
    "DiagnosticExamples",
    "FalsePositiveExample",
    "MissedMatchExample",
]
