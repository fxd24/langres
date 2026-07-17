"""Calibration :class:`~langres.core.methods_api.Method` s (``kind == "calibrate"``).

The concrete strategies :meth:`Resolver.fit(method=...) <langres.core.resolver.Resolver.fit>`
dispatches to when you want to turn a matcher's raw scores into real, comparable
probabilities: learn a score→probability map from labeled pairs. Each strategy
names one :class:`~langres.training.calibration.Calibrator` map:

- :class:`Platt` -> logistic (Platt) scaling -- ``sigmoid(a·score + b)``, a
  two-parameter smooth map for monotone over/under-confidence;
- :class:`Isotonic` -> non-parametric isotonic regression -- a flexible monotone
  step map that needs more data.

Import-light by construction (Pydantic only -- no scikit-learn): a ``Method`` is a
*value* the caller constructs at the fit call site, so constructing ``Platt()``
must never pull scikit-learn into ``sys.modules`` (locked by
``tests/test_import_budget.py``). The heavy scikit-learn import stays in
:mod:`langres.training.calibration`, reached only when
:meth:`~langres.core.resolver.Resolver.fit` actually fits the calibrator -- the
same split ``methods_prompt`` uses to keep ``dspy`` out of a bare import.
"""

from typing import ClassVar

from langres.core.methods_api import Method

__all__ = ["CalibrateMethod", "Isotonic", "Platt"]


class CalibrateMethod(Method):
    """Base for the score-calibration strategies (``kind == "calibrate"``).

    Fixes the contract :meth:`~langres.core.resolver.Resolver._fit_calibrate`
    reads: a :attr:`strategy` naming the
    :class:`~langres.training.calibration.Calibrator` map to fit. Like
    :attr:`~langres.core.methods_api.Method.kind`, :attr:`strategy` is a
    ``ClassVar`` -- strategy-type identity, not serialized config -- so the base
    declares it and each concrete subclass sets it. Mirrors
    :class:`~langres.training.methods_prompt.PromptMethod`'s ``optimizer``.
    """

    kind: ClassVar[str] = "calibrate"
    #: The :class:`~langres.training.calibration.Calibrator` map this strategy fits
    #: (``"platt"`` / ``"isotonic"``). Set by concrete subclasses; unset on the
    #: base (never instantiated).
    strategy: ClassVar[str]


class Platt(CalibrateMethod):
    """Calibrate by Platt scaling: a one-feature logistic map ``sigmoid(a·s + b)``.

    The two-parameter, smooth calibrator -- best when the matcher ranks well but
    is systematically over/under-confident. Fits from as few as a handful of
    labeled pairs (needs both a positive and a negative).
    """

    strategy: ClassVar[str] = "platt"

    def describe(self) -> str:
        """One-liner: ``"calibrate (Platt scaling)"``."""
        return "calibrate (Platt scaling)"


class Isotonic(CalibrateMethod):
    """Calibrate by isotonic regression: a non-parametric monotone step map.

    More flexible than :class:`Platt` (it can correct a non-monotone confidence
    curve) at the cost of needing more labeled data to avoid overfitting the step
    grid.
    """

    strategy: ClassVar[str] = "isotonic"

    def describe(self) -> str:
        """One-liner: ``"calibrate (isotonic regression)"``."""
        return "calibrate (isotonic regression)"
