"""Exit-criteria gate for the flywheel harvest (W2.4).

Drives the importable core of ``examples/flywheel_threshold_harvest.py`` against
the committed Fodors-Zagat fixtures and enforces the plan's HARD exit criterion:
>=25 human corrections must move a threshold in the CORRECT direction, proven on
a HELD-OUT gold split the threshold was never fit on. "Moved" is not enough --
the new threshold must score a strictly (and materially) higher F1 on gold.

The fixtures are committed and the harvest is pure Python, so this runs at $0 in
the default suite (no rapidfuzz, no split, no model at test time).
"""

from __future__ import annotations

from examples.flywheel_threshold_harvest import (
    evaluate_threshold,
    format_report,
    run_flywheel_harvest,
)

#: The plan's floor: at least this many corrections must drive the improvement.
_MIN_CORRECTIONS = 25

#: A margin comfortably below the observed gain (~+0.15) so the gate is decisive,
#: not "any wiggle counts", yet not brittle to fixture regeneration.
_MIN_F1_GAIN = 0.05


def test_corrections_move_threshold_correctly_on_held_out_gold() -> None:
    """>=25 corrections raise held-out gold F1 -- the flywheel's exit criterion."""
    result = run_flywheel_harvest()

    # The improvement is driven by a real batch of corrections, not one lucky flip.
    assert result.n_corrections >= _MIN_CORRECTIONS

    # The initial (weak-verdict) threshold is measurably suboptimal on gold, and
    # the correction-informed threshold scores strictly -- and materially -- better.
    assert result.after.f1 > result.before.f1, (
        f"corrections did not improve held-out F1 "
        f"(before={result.before.f1:.4f}, after={result.after.f1:.4f})"
    )
    assert result.f1_gain >= _MIN_F1_GAIN, (
        f"held-out F1 gain {result.f1_gain:+.4f} is below the {_MIN_F1_GAIN} floor; "
        "the corrections did not move the threshold decisively in the correct direction"
    )

    # The threshold actually changed (corrections were not a no-op).
    assert result.after.threshold != result.before.threshold


def test_before_threshold_recovers_the_weak_verdict_cut() -> None:
    """Deriving from verdicts alone just recovers the judge's bad cut (no free lunch).

    This pins the "self-training is circular" half of the exit rationale: without
    corrections, harvesting the judge's own verdicts and re-deriving lands right
    back on the deliberately-bad 0.55 verdict threshold -- so the AFTER gain must
    come from the human signal, not from re-reading the same weak labels.
    """
    result = run_flywheel_harvest()
    assert result.before.threshold == 0.55


def test_gain_comes_from_precision_not_a_recall_collapse() -> None:
    """The corrected threshold wins by cutting false positives, keeping recall high.

    Guards against a degenerate "better F1" that merely trades away recall: the
    before-threshold over-merges (low precision), and corrections should lift
    precision while holding recall near its original level.
    """
    result = run_flywheel_harvest()
    assert result.after.precision > result.before.precision
    assert result.after.recall >= result.before.recall - 0.1


def test_format_report_renders_before_and_after() -> None:
    """The demo's report renders both thresholds and the signed F1 gain."""
    report = format_report(run_flywheel_harvest())
    assert "BEFORE" in report
    assert "AFTER" in report
    assert "held-out F1" in report


def test_evaluate_threshold_extremes_are_degenerate() -> None:
    """A below-all threshold predicts every pair; an above-all one predicts none."""
    heldout = [(0.2, False), (0.8, True), (0.6, True)]
    predict_all = evaluate_threshold(heldout, 0.0)
    assert predict_all.fp == 1 and predict_all.fn == 0
    predict_none = evaluate_threshold(heldout, 1.0)
    assert predict_none.tp == 0 and predict_none.fn == 2
    assert predict_none.f1 == 0.0
