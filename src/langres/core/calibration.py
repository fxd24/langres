"""Back-compat shim: ``langres.core.calibration`` moved to ``langres.training.calibration``.

# TEMPORARY: deleted by the W2 sweep

Score calibration is part of *fitting* a matcher, not entity-resolution
modelling itself, so ``derive_threshold`` + the Platt/isotonic ``Calibrator``
now live in ``langres.training`` beside ``core``. Import from
``langres.training.calibration`` (or the ``langres`` / ``langres.core`` facades,
which resolve ``derive_threshold`` / ``Calibrator`` lazily behind the
``[trained]`` extra).

Like the real module, this shim imports scikit-learn at module scope, so it is
never on the eager ``import langres`` path -- it is reached only when a caller
imports it explicitly (or the lazy facade resolves the [trained] symbols, which
it does straight from ``langres.training.calibration``, not through this shim).
"""

from langres.training.calibration import Calibrator, ThresholdMethod, derive_threshold

__all__ = ["Calibrator", "derive_threshold", "ThresholdMethod"]
