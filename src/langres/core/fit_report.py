"""Back-compat shim: ``langres.core.fit_report`` moved to ``langres.training.fit_report``.

# TEMPORARY: deleted by the W2 sweep

The ``FitReport`` fit digest describes what *fitting* produced, not
entity-resolution modelling itself, so it now lives in ``langres.training``
beside ``core``. Import from ``langres.training.fit_report`` (or the ``langres``
/ ``langres.core`` facades, which still re-export ``FitReport``).
"""

from langres.training.fit_report import CalibrationDelta, FitReport

__all__ = ["CalibrationDelta", "FitReport"]
