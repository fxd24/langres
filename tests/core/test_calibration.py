"""Tests for the data-driven threshold helper ``derive_threshold``.

All tests are deterministic and cost $0: they exercise the pure function on
small synthetic score/label sets. Coverage focuses on the two derivation
methods (Youden's J and percentile), the in-range guarantee, and every
degenerate input (all-one-class, single point, ties, mismatched lengths,
empty, bad arguments).
"""

import numpy as np
import pytest

from langres.core.calibration import derive_threshold


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
