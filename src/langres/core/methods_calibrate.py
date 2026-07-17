"""Back-compat shim: ``langres.core.methods_calibrate`` moved to ``langres.training.methods_calibrate``.

# TEMPORARY: deleted by the W2 sweep

The score-calibration ``Method`` objects (``Platt`` / ``Isotonic``) configure
how a matcher is *fitted*, not entity-resolution modelling itself, so they now
live in ``langres.training`` beside ``core``. Import from
``langres.training.methods_calibrate`` (or the ``langres`` / ``langres.core``
facades, which still re-export ``Platt`` / ``Isotonic``).
"""

from langres.training.methods_calibrate import CalibrateMethod, Isotonic, Platt

__all__ = ["CalibrateMethod", "Isotonic", "Platt"]
