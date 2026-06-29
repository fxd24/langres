"""Unit tests for pair-level (pre-clustering) metrics.

``classify_pairs`` / ``pair_pr_curve`` classify each candidate judgement against
the gold match pairs *before* clustering, on small hand-built inputs with known
TP/FP/FN. These isolate the scorer from the clusterer's transitive-closure
amplification (the M3 methodology fix).
"""

import pytest

from langres.core.metrics import PairMetrics, classify_pairs, pair_pr_curve
from langres.core.models import PairwiseJudgement


def _judgement(left: str, right: str, score: float) -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id=left,
        right_id=right,
        score=score,
        score_type="calibrated_prob",
        decision_step="test",
        provenance={},
    )


# Gold: a/b and d/e and f/g are true matches; f/g is never even judged (the
# blocker missed it), so it must still count as a false negative.
_GOLD = {frozenset({"a", "b"}), frozenset({"d", "e"}), frozenset({"f", "g"})}


def test_classify_pairs_counts_tp_fp_fn_at_threshold() -> None:
    judgements = [
        _judgement("a", "b", 0.9),  # gold + predicted -> TP
        _judgement("a", "c", 0.8),  # not gold + predicted -> FP
        _judgement("d", "e", 0.2),  # gold but below threshold -> FN (rejected)
    ]
    result = classify_pairs(judgements, _GOLD, threshold=0.5)

    assert result.tp == 1
    assert result.fp == 1
    # d/e rejected + f/g never judged = 2 missed gold pairs.
    assert result.fn == 2
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1 / 3)
    assert result.f1 == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))
    assert result.threshold == 0.5


def test_classify_pairs_is_order_independent() -> None:
    # Same pair, ids swapped, must classify identically (frozenset keying).
    a = classify_pairs([_judgement("a", "b", 0.9)], _GOLD, threshold=0.5)
    b = classify_pairs([_judgement("b", "a", 0.9)], _GOLD, threshold=0.5)
    assert a.tp == b.tp == 1


def test_classify_pairs_no_predictions_is_zero_precision() -> None:
    # All judgements below threshold -> nothing predicted, precision falls back to 0.
    result = classify_pairs([_judgement("a", "b", 0.1)], _GOLD, threshold=0.5)
    assert result.tp == 0
    assert result.fp == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_classify_pairs_no_gold_is_zero_recall() -> None:
    result = classify_pairs([_judgement("a", "b", 0.9)], set(), threshold=0.5)
    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 0
    assert result.recall == 0.0


def test_classify_pairs_threshold_is_inclusive() -> None:
    # score == threshold counts as a predicted match.
    result = classify_pairs([_judgement("a", "b", 0.5)], _GOLD, threshold=0.5)
    assert result.tp == 1


def test_pair_pr_curve_one_entry_per_threshold_in_order() -> None:
    judgements = [
        _judgement("a", "b", 0.9),
        _judgement("a", "c", 0.8),
        _judgement("d", "e", 0.2),
    ]
    grid = [0.5, 0.85]
    curve = pair_pr_curve(judgements, _GOLD, grid)

    assert [m.threshold for m in curve] == grid
    assert all(isinstance(m, PairMetrics) for m in curve)
    # At 0.85, only a/b (0.9) is predicted: 1 TP, 0 FP.
    assert curve[1].tp == 1
    assert curve[1].fp == 0
    assert curve[1].precision == pytest.approx(1.0)


def test_pair_pr_curve_empty_grid_raises() -> None:
    with pytest.raises(ValueError, match="grid is empty"):
        pair_pr_curve([_judgement("a", "b", 0.9)], _GOLD, [])
