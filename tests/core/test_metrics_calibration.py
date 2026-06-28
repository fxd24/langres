"""Tests for agreement (kappa, MCC) and calibration (Brier, ECE, reliability) metrics.

All expected values are hand-computed. Coverage targets 100% of the new pure
functions in ``langres.core.metrics`` including every validation and edge branch.
"""

import math

import pytest

from langres.core.metrics import (
    _bin_indices,
    brier_score,
    cohens_kappa,
    expected_calibration_error,
    matthews_corrcoef,
    reliability_bins,
)

# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def test_cohens_kappa_hand_computed() -> None:
    # y_true has 2 positives, y_pred 1 positive, 3/4 agree.
    # p_o=0.75, p_e=0.5*0.25 + 0.5*0.75 = 0.5 -> (0.75-0.5)/0.5 = 0.5
    assert cohens_kappa([True, True, False, False], [True, False, False, False]) == 0.5


def test_cohens_kappa_perfect_agreement() -> None:
    # p_o=1, p_e=0.5 -> 1.0
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0


def test_cohens_kappa_prevalence_paradox() -> None:
    """High raw accuracy but kappa collapses under ~1% prevalence (W5)."""
    y_true = [True] + [False] * 99
    y_pred = [False] * 100
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / 100
    assert accuracy == 0.99
    # p_e = 0.01*0 + 0.99*1 = 0.99 == p_o -> kappa 0.0 despite 99% accuracy.
    assert cohens_kappa(y_true, y_pred) == 0.0


def test_cohens_kappa_no_variance_returns_zero() -> None:
    """Both raters all-positive -> p_e == 1 -> undefined, convention 0.0."""
    assert cohens_kappa([True, True], [True, True]) == 0.0


def test_cohens_kappa_negative() -> None:
    # Perfect disagreement: p_o=0, p_e=0.5 -> (0-0.5)/0.5 = -1.0
    assert cohens_kappa([True, False], [False, True]) == -1.0


def test_cohens_kappa_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        cohens_kappa([True], [True, False])


def test_cohens_kappa_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cohens_kappa([], [])


# ---------------------------------------------------------------------------
# Matthews correlation coefficient
# ---------------------------------------------------------------------------


def test_mcc_hand_computed() -> None:
    # TP=1, TN=2, FP=0, FN=1 -> (2-0)/sqrt(1*2*2*3) = 2/sqrt(12)
    expected = 2 / math.sqrt(12)
    assert matthews_corrcoef([True, True, False, False], [True, False, False, False]) == expected


def test_mcc_perfect() -> None:
    assert matthews_corrcoef([True, False, True, False], [True, False, True, False]) == 1.0


def test_mcc_perfect_inverse() -> None:
    assert matthews_corrcoef([True, False, True, False], [False, True, False, True]) == -1.0


def test_mcc_zero_denominator_returns_zero() -> None:
    """No predicted positives -> (TP+FP)=0 factor -> denominator 0 -> 0.0."""
    assert matthews_corrcoef([True, False], [False, False]) == 0.0


def test_mcc_prevalence_robust_vs_kappa() -> None:
    """Under skew, both kappa and MCC flag the useless all-negative predictor."""
    y_true = [True] + [False] * 99
    y_pred = [False] * 100
    assert matthews_corrcoef(y_true, y_pred) == 0.0


def test_mcc_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        matthews_corrcoef([True], [True, False])


def test_mcc_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        matthews_corrcoef([], [])


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def test_brier_hand_computed() -> None:
    # ((0.9-1)^2 + (0.1-0)^2)/2 = (0.01 + 0.01)/2 = 0.01
    assert brier_score([0.9, 0.1], [True, False]) == pytest.approx(0.01)


def test_brier_perfect_is_zero() -> None:
    assert brier_score([1.0, 0.0], [True, False]) == 0.0


def test_brier_worst_is_one() -> None:
    assert brier_score([0.0, 1.0], [True, False]) == 1.0


def test_brier_constant_half() -> None:
    assert brier_score([0.5, 0.5], [True, False]) == 0.25


def test_brier_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        brier_score([0.5], [True, False])


def test_brier_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        brier_score([], [])


def test_brier_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1.5], [True])


def test_brier_confidence_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([-0.1], [True])


# ---------------------------------------------------------------------------
# Expected calibration error
# ---------------------------------------------------------------------------


def test_ece_single_bin() -> None:
    # One bin: mean conf 0.8, accuracy 0.5 -> |0.8-0.5| = 0.3
    assert expected_calibration_error([0.8, 0.8], [True, False], n_bins=1) == pytest.approx(0.3)


def test_ece_quantile_two_bins_hand_computed() -> None:
    # confidences sorted [0.4,0.8,0.9]; equal-mass split -> bin0=[0.4,0.8], bin1=[0.9].
    # bin0: conf 0.6, acc 0.5 (one True one False) -> |0.5-0.6|=0.1, weight 2/3
    # bin1: conf 0.9, acc 1.0 -> 0.1, weight 1/3
    # ECE = 2/3*0.1 + 1/3*0.1 = 0.1
    conf = [0.9, 0.8, 0.4]
    out = [True, False, True]  # 0.9->True, 0.8->False, 0.4->True
    assert expected_calibration_error(conf, out, n_bins=2) == pytest.approx(0.1)


def test_ece_ties_stay_in_one_bin_and_are_order_independent() -> None:
    """Equal confidences must not be split across quantile bins (Codex P2).

    Four identical 0.5 predictions that are right half the time are perfectly
    calibrated -> ECE 0.0, regardless of row order.
    """
    conf = [0.5, 0.5, 0.5, 0.5]
    assert expected_calibration_error(conf, [True, True, False, False], n_bins=2) == 0.0
    assert expected_calibration_error(conf, [False, True, False, True], n_bins=2) == 0.0
    bins = reliability_bins(conf, [True, True, False, False], n_bins=2)
    assert len(bins) == 1
    assert bins[0].count == 4
    assert bins[0].mean_confidence == 0.5
    assert bins[0].observed_frequency == 0.5


def test_ece_partial_ties_extend_boundary() -> None:
    # Sorted confidences [0.3, 0.5, 0.5, 0.9], n_bins=2. The even cut at index 2
    # would split the tied 0.5s; the boundary extends so both 0.5s share a bin.
    bins = reliability_bins([0.9, 0.5, 0.5, 0.3], [True, True, False, False], n_bins=2)
    assert [b.count for b in bins] == [3, 1]  # {0.3, 0.5, 0.5} then {0.9}


def test_ece_perfectly_calibrated_is_zero() -> None:
    # Each bin's accuracy equals its confidence.
    assert expected_calibration_error([0.0, 1.0], [False, True], n_bins=2) == 0.0


def test_ece_uniform_strategy() -> None:
    # Uniform 2 bins over [0,1]: 0.2->bin0, 0.9->bin1.
    # bin0: conf 0.2 acc 0 -> 0.2 weight .5 ; bin1 conf 0.9 acc 1 -> 0.1 weight .5
    # ECE = 0.5*0.2 + 0.5*0.1 = 0.15
    val = expected_calibration_error([0.2, 0.9], [False, True], n_bins=2, strategy="uniform")
    assert val == pytest.approx(0.15)


def test_ece_uniform_confidence_one_clamps_to_last_bin() -> None:
    # p=1.0 with n_bins=4 must land in the top bin, not overflow.
    val = expected_calibration_error([1.0], [True], n_bins=4, strategy="uniform")
    assert val == 0.0


def test_ece_n_bins_below_one_raises() -> None:
    with pytest.raises(ValueError, match="n_bins must be >= 1"):
        expected_calibration_error([0.5], [True], n_bins=0)


def test_ece_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="quantile.*uniform"):
        expected_calibration_error([0.5], [True], strategy="bogus")  # type: ignore[arg-type]


def test_ece_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        expected_calibration_error([0.5], [True, False])


# ---------------------------------------------------------------------------
# Reliability bins
# ---------------------------------------------------------------------------


def test_reliability_bins_hand_computed() -> None:
    bins = reliability_bins([0.2, 0.8], [False, True], n_bins=2)
    assert [(b.mean_confidence, b.observed_frequency, b.count) for b in bins] == [
        (0.2, 0.0, 1),
        (0.8, 1.0, 1),
    ]


def test_reliability_bins_more_bins_than_points_drops_empty() -> None:
    # n_bins=5 but only 2 points -> empty bins are skipped (size==0 branch).
    bins = reliability_bins([0.3, 0.7], [False, True], n_bins=5)
    assert len(bins) == 2
    assert bins[0].count == 1
    assert bins[1].count == 1


def test_reliability_bins_single_bin_aggregates() -> None:
    bins = reliability_bins([0.4, 0.6], [True, False], n_bins=1)
    assert len(bins) == 1
    assert bins[0].mean_confidence == pytest.approx(0.5)
    assert bins[0].observed_frequency == 0.5
    assert bins[0].count == 2


def test_reliability_bins_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        reliability_bins([], [])


# ---------------------------------------------------------------------------
# _bin_indices internals (uniform empty-bin drop)
# ---------------------------------------------------------------------------


def test_bin_indices_uniform_drops_empty_bins() -> None:
    # Both points fall in the same uniform bin -> only one non-empty group.
    groups = _bin_indices([0.1, 0.15], n_bins=4, strategy="uniform")
    assert groups == [[0, 1]]
