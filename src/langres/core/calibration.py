"""Data-driven threshold derivation for entity-resolution scorers.

Thresholds in the framework (module defaults, the cascade band) were hand-set --
``0.5`` here, ``0.3``/``0.9`` there -- not read off the actual score
distribution. This module is the minimal fix: one pure function that inspects a
score distribution plus its gold labels and derives a threshold *from the data*.

It is deliberately a single-responsibility, side-effect-free function module: no
serializable component, no Resolver slot (a runtime ``CalibratedModule`` is
deferred to M4.5). Experiments and factories consume the returned ``float``
directly. Calibration *quality* is characterized elsewhere with
:mod:`langres.core.metrics` (``brier_score`` / ``expected_calibration_error``);
this module only picks the cut.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from sklearn.metrics import roc_curve

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
