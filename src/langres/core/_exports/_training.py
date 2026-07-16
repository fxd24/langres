"""The training surface: fit mixins, the fit digest, and the ``Method`` objects.

The method objects passed to ``Resolver.fit(method=...)`` are import-light
config -- dspy/scikit-learn stay lazy inside their fit paths (and behind
``Calibrator`` below), so these names are safe to import eagerly.

See ``langres.core._exports`` for the fragment contract.
"""

from typing import TYPE_CHECKING

from langres.core.fit import SupervisedFitMixin, UnsupervisedFitMixin
from langres.core.fit_report import FitReport
from langres.core.harvest import align_pairs
from langres.core.methods_api import Method
from langres.core.methods_calibrate import Isotonic, Platt
from langres.core.methods_prompt import Bootstrap, GEPA, MIPRO

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy name visible to `mypy --strict`
    # without pulling scikit-learn into a bare `import langres`.
    from langres.core.calibration import Calibrator

__all__ = [
    "align_pairs",
    "Bootstrap",
    "FitReport",
    "GEPA",
    "Isotonic",
    "Method",
    "MIPRO",
    "Platt",
    "SupervisedFitMixin",
    "UnsupervisedFitMixin",
]

LAZY_SUBMODULES: tuple[str, ...] = ()

LAZY_SYMBOLS: dict[str, str] = {
    "Calibrator": "langres.core.calibration",
}

EXTRA_BY_SYMBOL: dict[str, str] = {
    "Calibrator": "trained",
}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SUBMODULES, *LAZY_SYMBOLS)
