"""Score→threshold and score→probability calibration for ER scorers.

Two calibration roles live here, both label-aware and both ``[trained]`` (they
need scikit-learn):

- :func:`derive_threshold` -- a pure function that reads a score distribution
  plus its gold labels and picks a *decision cut* from the data (Youden's J or a
  percentile), replacing the framework's hand-set ``0.5``/``0.3``/``0.9``
  thresholds. It maps a distribution to a single ``float``; it does not change
  what the scores *mean*.
- :class:`Calibrator` -- a fittable component (the concrete
  :class:`~langres.core.fit.CalibratorFitMixin`) that learns a score→probability
  map (Platt scaling or isotonic regression) so a raw matcher score becomes a
  real, comparable probability. It is a serializable Resolver slot: its learned
  parameters are a handful of plain floats, so ``Resolver.save``/``load``
  round-trips a *fitted* calibrator with no weight files and no pickle -- and the
  learned map is applied with pure NumPy at predict time (scikit-learn is touched
  only during ``fit_calibrator``, mirroring
  :class:`~langres.core.matchers.random_forest_judge.RandomForestMatcher`'s
  no-pickle forest state).

Calibration *quality* is characterized with :mod:`langres.core.metrics`
(``brier_score`` / ``expected_calibration_error``); :meth:`Resolver._fit_calibrate`
reports the Brier/ECE before-vs-after a :class:`Calibrator` fit.
"""

from collections.abc import Sequence
from typing import ClassVar, Literal, cast

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from langres.core.registry import register

ThresholdMethod = Literal["youden", "percentile"]
"""How to derive the threshold from the score distribution.

``"youden"`` maximizes Youden's J (``tpr - fpr``) on the ROC curve -- the
label-aware operating point that best trades sensitivity against specificity.
``"percentile"`` returns a label-agnostic percentile cut of the scores.
"""


def derive_threshold(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    method: ThresholdMethod = "youden",
    percentile: float | None = None,
) -> float:
    """Derive a decision threshold from a score distribution and its gold labels.

    A pair is predicted a match when ``score >= threshold``; this function picks
    that threshold from the data instead of hand-setting it.

    Args:
        scores: Per-pair scores (e.g. embedding cosine similarities). Any real
            range is accepted; the returned threshold is clamped to
            ``[min(scores), max(scores)]`` so it is always a usable cut -- except
            for the anti-correlated degenerate below, where ``"youden"`` returns a
            cut just above ``max(scores)`` ("classify nothing").
        labels: Gold match/non-match labels aligned with ``scores``.
        method: ``"youden"`` (default) maximizes Youden's J via
            :func:`sklearn.metrics.roc_curve`; ``"percentile"`` returns the
            ``percentile``-th percentile of ``scores``.
        percentile: Required (and only used) when ``method="percentile"``; the
            percentile in ``[0, 100]`` to cut at.

    Returns:
        The derived threshold as a plain ``float`` within the observed score
        range (for ``"youden"`` on an inversely-correlated scorer where no real
        cut has positive Youden's J, a value just above ``max(scores)`` so nothing
        is classified as a match).

    Raises:
        ValueError: If ``scores`` and ``labels`` differ in length or are empty;
            if ``method="youden"`` and ``labels`` are all one class (Youden's J
            is undefined without both a positive and a negative); if
            ``method="percentile"`` and ``percentile`` is missing or outside
            ``[0, 100]``; or if ``method`` is unrecognized.

    Example:
        >>> derive_threshold([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
        0.8
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels must have equal length, got {len(scores)} and {len(labels)}"
        )
    if not scores:
        raise ValueError("scores and labels must be non-empty")

    score_array = np.asarray(scores, dtype=float)
    low, high = float(score_array.min()), float(score_array.max())

    if method == "youden":
        label_array = np.asarray(labels, dtype=bool)
        if label_array.all() or not label_array.any():
            raise ValueError(
                "youden needs both classes present in labels (at least one match "
                "and one non-match); got a single class"
            )
        _fpr, tpr, thresholds = roc_curve(label_array, score_array)
        # roc_curve prepends a +inf sentinel threshold meaning "classify nothing
        # as a match". It must be excluded before argmax: for an inversely
        # correlated scorer no real cut has positive Youden's J, so argmax would
        # otherwise pick that sentinel and -- once clamped down to max(scores) --
        # silently turn "classify nothing" into "classify the top non-match as a
        # match".
        finite = np.isfinite(thresholds)
        youden = (tpr - _fpr)[finite]
        finite_thresholds = thresholds[finite]
        if youden.size == 0:  # pragma: no cover - defensive; both classes present
            # Degenerate: only the +inf sentinel survived (unreachable while both
            # classes are present, which is validated above) — abstain safely.
            return high
        best = int(np.argmax(youden))
        if youden[best] <= 0.0 and low < high:
            # No finite cut carries positive signal on a scorer whose scores span
            # a range: the honest operating point is "classify nothing" -- a
            # threshold strictly above every score, so the top non-match is never
            # called a match (rather than clamping the sentinel down onto it).
            return float(np.nextafter(high, np.inf))
        return float(np.clip(finite_thresholds[best], low, high))

    if method == "percentile":
        if percentile is None:
            raise ValueError("method='percentile' requires a percentile value in [0, 100]")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must lie in [0, 100], got {percentile}")
        return float(np.percentile(score_array, percentile))

    raise ValueError(f"method must be 'youden' or 'percentile', got {method!r}")


CalibrationMethod = Literal["platt", "isotonic"]
"""Which score→probability map a :class:`Calibrator` learns.

``"platt"`` fits a one-feature logistic regression (``sigmoid(a·score + b)``) --
a smooth, two-parameter map, best when miscalibration is a monotone
over/under-confidence. ``"isotonic"`` fits a non-parametric monotone step map
(``sklearn.isotonic.IsotonicRegression``) -- more flexible, needs more data, and
can correct a non-monotone confidence curve.
"""


@register("calibrator")
class Calibrator:
    """A fittable score→probability map (Platt or isotonic) -- the concrete calibrator.

    Implements :class:`~langres.core.fit.CalibratorFitMixin` structurally:
    :meth:`fit_calibrator` learns the map from ``(scores, labels)``, and
    :meth:`transform` applies it. A raw matcher score (a rank-ordered number that
    is *not* a true probability) becomes a calibrated probability in ``[0, 1]``,
    so a clustering threshold on it is meaningful and two judges' scores are
    comparable.

    **No pickle, no weight files.** The learned state is a handful of plain floats
    (Platt: two coefficients; isotonic: a small ``(x, y)`` interpolation grid),
    carried inline in :attr:`config` so ``Resolver.save``/``load`` round-trips a
    *fitted* calibrator through the JSON manifest alone. scikit-learn is used only
    inside :meth:`fit_calibrator` to *learn* the parameters; :meth:`transform`
    applies them with pure NumPy, so a reloaded calibrator scores identically to a
    freshly-fit one without ever reconstructing an sklearn estimator (mirrors
    :class:`~langres.core.matchers.random_forest_judge.RandomForestMatcher`).

    Attributes:
        method: The strategy declared at construction (``"platt"`` | ``"isotonic"``)
            -- visible up front so a trainable role is never hidden.
    """

    type_name: ClassVar[str] = "calibrator"

    def __init__(
        self, method: CalibrationMethod = "platt", *, params: dict[str, object] | None = None
    ) -> None:
        """Initialize a calibrator; unfitted unless ``params`` is supplied.

        Args:
            method: ``"platt"`` (logistic, default) or ``"isotonic"``.
            params: The learned parameters, for reconstructing a *fitted*
                calibrator from :attr:`config` (see :meth:`from_config`). Leave
                ``None`` to build an unfit calibrator that :meth:`fit_calibrator`
                then trains.

        Raises:
            ValueError: If ``method`` is not ``"platt"`` or ``"isotonic"``.
        """
        if method not in ("platt", "isotonic"):
            raise ValueError(f"method must be 'platt' or 'isotonic', got {method!r}")
        self.method: CalibrationMethod = method
        self._params = params

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit_calibrator` (or a fitted :meth:`from_config`) has run."""
        return self._params is not None

    def fit_calibrator(self, scores: Sequence[float], labels: Sequence[bool]) -> None:
        """Learn the score→probability map from scores + gold labels.

        Args:
            scores: Raw matcher scores to calibrate, aligned with ``labels``.
            labels: Gold match/non-match labels for each score.

        Raises:
            ValueError: If ``scores``/``labels`` differ in length or are empty, or
                if the labels are a single class (calibration needs both a positive
                and a negative to learn a map).
        """
        if len(scores) != len(labels):
            raise ValueError(
                f"scores and labels must have equal length, got {len(scores)} and {len(labels)}"
            )
        if not scores:
            raise ValueError("scores and labels must be non-empty")
        label_array = np.asarray(labels, dtype=bool)
        if label_array.all() or not label_array.any():
            raise ValueError(
                "calibration needs both classes present in labels (at least one "
                "match and one non-match); got a single class"
            )
        score_array = np.asarray(scores, dtype=float)
        if self.method == "platt":
            clf = LogisticRegression()
            clf.fit(score_array.reshape(-1, 1), label_array.astype(int))
            self._params = {
                "coef": float(clf.coef_[0][0]),
                "intercept": float(clf.intercept_[0]),
            }
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(score_array, label_array.astype(float))
            self._params = {
                "x": [float(v) for v in iso.X_thresholds_],
                "y": [float(v) for v in iso.y_thresholds_],
            }

    def transform(self, scores: Sequence[float]) -> list[float]:
        """Apply the learned map, returning calibrated probabilities in ``[0, 1]``.

        Pure NumPy -- never touches scikit-learn -- so a reloaded calibrator maps
        identically to a freshly-fit one. Platt evaluates ``sigmoid(a·s + b)``;
        isotonic linearly interpolates the fitted ``(x, y)`` grid, clipping to its
        endpoints (matching ``out_of_bounds="clip"``).

        Raises:
            RuntimeError: If called before :meth:`fit_calibrator`.
        """
        if self._params is None:
            raise RuntimeError(
                "Calibrator.transform() called before fit: fit_calibrator(scores, "
                "labels) first, or reconstruct a fitted calibrator via from_config()."
            )
        score_array = np.asarray(scores, dtype=float)
        if self.method == "platt":
            coef = cast("float", self._params["coef"])
            intercept = cast("float", self._params["intercept"])
            probs = 1.0 / (1.0 + np.exp(-(coef * score_array + intercept)))
        else:
            xs = np.asarray(cast("list[float]", self._params["x"]), dtype=float)
            ys = np.asarray(cast("list[float]", self._params["y"]), dtype=float)
            probs = np.interp(score_array, xs, ys)
        # Guard the [0, 1] contract against floating-point drift so a calibrated
        # score never trips PairwiseJudgement.score's ge=0/le=1 validation.
        return [float(p) for p in np.clip(probs, 0.0, 1.0)]

    __call__ = transform

    @property
    def config(self) -> dict[str, object]:
        """Full serializable config: strategy + inline learned params (or ``None``).

        The learned params ARE the config (Platt: 2 floats; isotonic: a small
        grid), so :meth:`from_config` reconstructs a *fitted* calibrator with no
        sidecar state -- the ``[trained]`` extra is needed only to import this
        module, never to apply a reloaded map.
        """
        return {"method": self.method, "params": self._params}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "Calibrator":
        """Reconstruct a calibrator (fitted iff ``config`` carries ``params``)."""
        method = cast("CalibrationMethod", config["method"])
        params = cast("dict[str, object] | None", config.get("params"))
        return cls(method=method, params=params)
