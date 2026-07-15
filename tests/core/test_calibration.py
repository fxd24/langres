"""Tests for the calibration module: ``derive_threshold`` + the ``Calibrator``.

All tests are deterministic and cost $0: they exercise pure functions / the
NumPy-applied calibrator on small synthetic score/label sets. Coverage focuses
on the two threshold-derivation methods (Youden's J and percentile) and their
degenerate inputs, plus the fittable Platt/isotonic :class:`Calibrator` -- the
honest before/after Brier/ECE proof, its fit guards, and the JSON config
round-trip that reconstructs a fitted, scikit-learn-free map.
"""

import json

import numpy as np
import pytest

from langres.core.calibration import Calibrator, derive_threshold
from langres.core.fit import CalibratorFitMixin
from langres.core.metrics import brier_score, expected_calibration_error


def test_youden_separates_clean_bimodal() -> None:
    """On a cleanly separated bimodal set, Youden picks a separating threshold."""
    neg = [0.05, 0.10, 0.12, 0.20]
    pos = [0.80, 0.85, 0.90, 0.95]
    scores = neg + pos
    labels = [False] * len(neg) + [True] * len(pos)

    thr = derive_threshold(scores, labels, method="youden")

    # A separating threshold: every negative is below it, every positive at/above.
    assert max(neg) < thr <= min(pos)
    # And it perfectly classifies the set (score >= thr => positive).
    preds = [s >= thr for s in scores]
    assert preds == labels


def test_youden_is_the_default_method() -> None:
    """Omitting ``method`` uses Youden's J (same result as passing it explicitly)."""
    neg = [0.1, 0.2]
    pos = [0.8, 0.9]
    scores = neg + pos
    labels = [False, False, True, True]

    assert derive_threshold(scores, labels) == derive_threshold(scores, labels, method="youden")


def test_youden_threshold_within_observed_range() -> None:
    """The derived threshold never falls outside the observed score range."""
    scores = [0.2, 0.3, 0.4, 0.9, 0.95]
    labels = [False, False, False, True, True]

    thr = derive_threshold(scores, labels, method="youden")

    assert min(scores) <= thr <= max(scores)


def test_youden_ties_all_equal_scores_stays_in_range() -> None:
    """Identical scores can't separate; the threshold still lands in range."""
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [False, True, False, True]

    thr = derive_threshold(scores, labels, method="youden")

    # roc_curve's leading +inf sentinel must be clamped down to the observed max.
    assert thr == 0.5


def test_youden_inversely_correlated_does_not_misclassify_top_non_match() -> None:
    """An anti-correlated scorer must not clamp the +inf sentinel onto the top non-match.

    With ``scores=[0.9, 0.1]`` and ``labels=[False, True]`` the highest score is a
    NON-match, so no ``score >= threshold`` cut carries positive Youden's J. The
    old code let ``argmax`` pick roc_curve's leading ``+inf`` sentinel and clamped
    it down to ``max(scores)=0.9`` — turning "classify nothing" into "classify the
    top non-match (0.9) as a match". The threshold must instead sit strictly above
    the top non-match so it is predicted a non-match.
    """
    scores = [0.9, 0.1]
    labels = [False, True]

    thr = derive_threshold(scores, labels, method="youden")

    top_non_match = 0.9  # highest-scoring negative
    assert top_non_match < thr, f"top non-match {top_non_match} must not be a match at {thr}"


def test_youden_inversely_correlated_multi_point_classifies_nothing() -> None:
    """A larger anti-correlated set also abstains (no negative predicted a match)."""
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [False, False, True, True]  # high score => non-match

    thr = derive_threshold(scores, labels, method="youden")

    # Every negative (the two high scores) sits below the threshold => non-match.
    assert all(s < thr for s, y in zip(scores, labels, strict=True) if not y)


def test_percentile_returns_the_requested_cut() -> None:
    """``percentile`` returns the numpy percentile of the score distribution."""
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = [False, False, True, True, True]

    thr = derive_threshold(scores, labels, method="percentile", percentile=50.0)

    assert thr == pytest.approx(float(np.percentile(scores, 50.0)))
    assert thr == pytest.approx(0.5)


def test_percentile_zero_and_hundred_are_range_endpoints() -> None:
    """The 0th/100th percentiles are the min/max of the observed scores."""
    scores = [0.1, 0.4, 0.9]
    labels = [False, True, True]

    assert derive_threshold(scores, labels, method="percentile", percentile=0.0) == pytest.approx(
        0.1
    )
    assert derive_threshold(scores, labels, method="percentile", percentile=100.0) == pytest.approx(
        0.9
    )


def test_percentile_single_point_returns_that_point() -> None:
    """A single score returns that score for any percentile (degenerate but valid)."""
    thr = derive_threshold([0.42], [True], method="percentile", percentile=90.0)

    assert thr == pytest.approx(0.42)


def test_percentile_requires_percentile_argument() -> None:
    """``method='percentile'`` without a ``percentile`` value is a clear error."""
    with pytest.raises(ValueError, match="percentile"):
        derive_threshold([0.1, 0.9], [False, True], method="percentile")


@pytest.mark.parametrize("bad", [-1.0, 100.1, 150.0])
def test_percentile_out_of_bounds_raises(bad: float) -> None:
    """A percentile outside [0, 100] is a clear error."""
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        derive_threshold([0.1, 0.9], [False, True], method="percentile", percentile=bad)


def test_youden_all_positive_labels_raises() -> None:
    """Youden needs both classes; all-positive labels raise a clear error."""
    with pytest.raises(ValueError, match="both classes"):
        derive_threshold([0.1, 0.5, 0.9], [True, True, True], method="youden")


def test_youden_all_negative_labels_raises() -> None:
    """Youden needs both classes; all-negative labels raise a clear error."""
    with pytest.raises(ValueError, match="both classes"):
        derive_threshold([0.1, 0.5, 0.9], [False, False, False], method="youden")


def test_mismatched_lengths_raises() -> None:
    """Scores and labels of different lengths are a clear error."""
    with pytest.raises(ValueError, match="equal length"):
        derive_threshold([0.1, 0.2, 0.3], [True, False], method="youden")


def test_empty_inputs_raise() -> None:
    """Empty scores/labels are a clear error."""
    with pytest.raises(ValueError, match="non-empty"):
        derive_threshold([], [], method="youden")


def test_unknown_method_raises() -> None:
    """An unrecognized method is a clear error (defensive, beyond the type hint)."""
    with pytest.raises(ValueError, match="method"):
        derive_threshold([0.1, 0.9], [False, True], method="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Calibrator: the fittable score->probability map (Platt / isotonic).
# ---------------------------------------------------------------------------


def _miscalibrated_set(n: int = 400, seed: int = 1) -> tuple[list[float], list[bool]]:
    """A deliberately miscalibrated set: correct ranking, over-confident scores.

    Outcomes are Bernoulli(raw), but the observed score squashes ``raw`` away from
    0.5 (``0.5 + sign·sqrt|raw-0.5|``) -- the ranking is preserved while the score
    systematically over-states its confidence, so a real calibrator must lower
    both Brier and ECE.
    """
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, n)
    labels = rng.uniform(0.0, 1.0, n) < raw
    scores = np.clip(0.5 + np.sign(raw - 0.5) * np.sqrt(np.abs(raw - 0.5)), 0.0, 1.0)
    return scores.tolist(), [bool(x) for x in labels]


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_lowers_brier_and_ece(method: str) -> None:
    """The honest proof: fitting a calibrator drives Brier AND ECE down.

    On a set that ranks correctly but is over-confident, both Platt and isotonic
    must produce a strictly better Brier score and a no-worse ECE than the raw,
    uncalibrated scores -- otherwise the map is not really calibrating.
    """
    scores, labels = _miscalibrated_set()
    brier_before = brier_score(scores, labels)
    ece_before = expected_calibration_error(scores, labels)

    cal = Calibrator(method)  # type: ignore[arg-type]
    cal.fit_calibrator(scores, labels)
    calibrated = cal.transform(scores)

    assert all(0.0 <= p <= 1.0 for p in calibrated)
    assert brier_score(calibrated, labels) < brier_before
    assert expected_calibration_error(calibrated, labels) <= ece_before


def test_calibrator_satisfies_the_fit_protocol() -> None:
    """A Calibrator IS a CalibratorFitMixin (structural: fit_calibrator + transform)."""
    assert isinstance(Calibrator("platt"), CalibratorFitMixin)


def test_platt_is_the_default_method() -> None:
    """Omitting ``method`` builds a Platt (logistic) calibrator."""
    assert Calibrator().method == "platt"


def test_unknown_method_rejected_at_construction() -> None:
    """An unsupported strategy is a clear error, not a late surprise at fit."""
    with pytest.raises(ValueError, match="platt.*isotonic"):
        Calibrator("sigmoidal")  # type: ignore[arg-type]


def test_transform_before_fit_raises() -> None:
    """Applying an unfit calibrator is a clear RuntimeError, not silent garbage."""
    cal = Calibrator("platt")
    assert not cal.fitted
    with pytest.raises(RuntimeError, match="before fit"):
        cal.transform([0.5])


def test_fit_needs_both_classes() -> None:
    """A single-class label set cannot define a calibration map."""
    cal = Calibrator("platt")
    with pytest.raises(ValueError, match="both classes"):
        cal.fit_calibrator([0.1, 0.5, 0.9], [True, True, True])


def test_fit_rejects_mismatched_and_empty() -> None:
    """Length-mismatched and empty inputs are clear errors."""
    cal = Calibrator("isotonic")
    with pytest.raises(ValueError, match="equal length"):
        cal.fit_calibrator([0.1, 0.2, 0.3], [True, False])
    with pytest.raises(ValueError, match="non-empty"):
        cal.fit_calibrator([], [])


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_from_config_round_trips_a_fitted_calibrator(method: str) -> None:
    """save/load fidelity: config -> JSON -> from_config maps identically.

    The learned params are plain floats carried inline in ``config`` (no weight
    files), and ``transform`` is pure NumPy -- so a reconstructed calibrator
    scores byte-identically to the freshly-fit one it came from.
    """
    scores, labels = _miscalibrated_set(n=120, seed=3)
    cal = Calibrator(method)  # type: ignore[arg-type]
    cal.fit_calibrator(scores, labels)

    config = json.loads(json.dumps(cal.config))  # prove the config is JSON-safe
    rebuilt = Calibrator.from_config(config)

    assert rebuilt.method == method
    assert rebuilt.fitted
    probe = [0.0, 0.15, 0.37, 0.5, 0.63, 0.88, 1.0]
    assert rebuilt.transform(probe) == cal.transform(probe)


def test_transform_clips_to_unit_interval() -> None:
    """Isotonic outputs stay in [0, 1] even for out-of-range inputs (clip)."""
    cal = Calibrator("isotonic")
    cal.fit_calibrator([0.2, 0.4, 0.6, 0.8], [False, False, True, True])
    out = cal.transform([-5.0, 0.5, 5.0])
    assert all(0.0 <= p <= 1.0 for p in out)


def test_call_is_transform() -> None:
    """``calibrator(scores)`` is sugar for ``calibrator.transform(scores)``."""
    cal = Calibrator("platt")
    cal.fit_calibrator([0.1, 0.4, 0.6, 0.9], [False, False, True, True])
    assert cal([0.3, 0.7]) == cal.transform([0.3, 0.7])
