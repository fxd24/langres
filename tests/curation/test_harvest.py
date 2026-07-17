"""Tests for langres.curation.harvest (the flywheel's harvest half).

Covers the ``corrections.jsonl`` contract (:class:`Correction` /
:class:`CorrectionLog`), the verdicts+corrections -> labeled-pairs merge
(:func:`harvest_labeled_pairs`), and the wiring to ``derive_threshold``
(:func:`derive_threshold_from_pairs`). Everything here is zero-spend and
dependency-light: plain dicts and JSONL round-trips, no judge, no model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from langres.curation.harvest import (
    Correction,
    CorrectionLog,
    LabeledPair,
    derive_threshold_from_pairs,
    harvest_labeled_pairs,
)


def _row(left_id: str, right_id: str, score: float, verdict: bool) -> dict[str, Any]:
    """A minimal JudgementLog-format row (the keys harvest actually reads)."""
    return {
        "v": 1,
        "left_id": left_id,
        "right_id": right_id,
        "score": score,
        "verdict": verdict,
        "model": "test",
        "cost_usd": 0.0,
    }


# --------------------------------------------------------------------------- #
# Correction contract                                                         #
# --------------------------------------------------------------------------- #


def test_correction_minimal_defaults() -> None:
    """Only ids + label are required; version defaults to 1 and audit fields to None."""
    c = Correction(left_id="a", right_id="b", label=True)
    assert c.v == 1
    assert c.label is True
    assert c.original_score is None
    assert c.original_verdict is None
    assert c.reviewer is None
    assert c.timestamp is None


def test_correction_carries_optional_audit_context() -> None:
    """The optional audit fields round-trip when the review tool supplies them."""
    c = Correction(
        left_id="a",
        right_id="b",
        label=False,
        original_score=0.62,
        original_verdict=True,
        reviewer="alice",
        timestamp="2026-07-03T10:00:00+00:00",
    )
    assert c.original_score == 0.62
    assert c.original_verdict is True
    assert c.reviewer == "alice"
    assert c.timestamp == "2026-07-03T10:00:00+00:00"


# --------------------------------------------------------------------------- #
# CorrectionLog JSONL round-trip                                              #
# --------------------------------------------------------------------------- #


def test_correction_log_read_missing_file_is_empty(tmp_path: Path) -> None:
    """A never-written corrections file reads back as ``[]`` (not an error)."""
    log = CorrectionLog(tmp_path / "nope.jsonl")
    assert log.read() == []


def test_correction_log_append_creates_parent_dirs(tmp_path: Path) -> None:
    """Appending creates missing parent directories (mirrors JudgementLog)."""
    log = CorrectionLog(tmp_path / "nested" / "dir" / "corrections.jsonl")
    log.append(Correction(left_id="a", right_id="b", label=True))
    assert log.path.exists()


def test_correction_log_round_trip_preserves_order_and_fields(tmp_path: Path) -> None:
    """Every appended correction reloads in write order with all fields intact."""
    log = CorrectionLog(tmp_path / "corrections.jsonl")
    log.append(Correction(left_id="a", right_id="b", label=True, reviewer="alice"))
    log.append(Correction(left_id="c", right_id="d", label=False, original_score=0.4))

    reloaded = log.read()
    assert [c.left_id for c in reloaded] == ["a", "c"]
    assert reloaded[0].reviewer == "alice"
    assert reloaded[1].label is False
    assert reloaded[1].original_score == 0.4


def test_correction_log_read_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines in the file are ignored (trailing newlines, hand edits)."""
    path = tmp_path / "corrections.jsonl"
    path.write_text(
        '{"v": 1, "left_id": "a", "right_id": "b", "label": true}\n'
        "\n"
        "   \n"
        '{"v": 1, "left_id": "c", "right_id": "d", "label": false}\n',
        encoding="utf-8",
    )
    reloaded = CorrectionLog(path).read()
    assert len(reloaded) == 2
    assert reloaded[0].left_id == "a"
    assert reloaded[1].left_id == "d" or reloaded[1].right_id == "d"


# --------------------------------------------------------------------------- #
# harvest_labeled_pairs merge semantics                                       #
# --------------------------------------------------------------------------- #


def test_harvest_verdicts_only_passes_weak_labels_through() -> None:
    """With no corrections, each row's verdict becomes the weak label."""
    rows = [_row("a", "b", 0.9, True), _row("c", "d", 0.2, False)]
    pairs = harvest_labeled_pairs(rows, corrections=[])
    assert pairs == [
        LabeledPair(left_id="a", right_id="b", score=0.9, label=True, source="verdict"),
        LabeledPair(left_id="c", right_id="d", score=0.2, label=False, source="verdict"),
    ]


def test_harvest_correction_overrides_verdict() -> None:
    """A correction for a judged pair overrides the verdict and marks the source."""
    rows = [_row("a", "b", 0.62, True)]  # judge said match...
    corrections = [Correction(left_id="a", right_id="b", label=False)]  # ...human says no
    pairs = harvest_labeled_pairs(rows, corrections)
    assert pairs == [
        LabeledPair(left_id="a", right_id="b", score=0.62, label=False, source="correction")
    ]


def test_harvest_matches_corrections_order_independently() -> None:
    """A correction identifies its pair by set membership, not left/right order."""
    rows = [_row("a", "b", 0.7, False)]
    corrections = [Correction(left_id="b", right_id="a", label=True)]  # swapped order
    pairs = harvest_labeled_pairs(rows, corrections)
    assert pairs[0].label is True
    assert pairs[0].source == "correction"


def test_harvest_last_correction_wins_for_duplicate_pair() -> None:
    """Two corrections for the same pair: the later one takes effect."""
    rows = [_row("a", "b", 0.5, True)]
    corrections = [
        Correction(left_id="a", right_id="b", label=True),
        Correction(left_id="a", right_id="b", label=False),
    ]
    pairs = harvest_labeled_pairs(rows, corrections)
    assert pairs[0].label is False


def test_harvest_preserves_judgement_row_order_and_count() -> None:
    """One labeled pair per judgement row, in row order (duplicate scorings kept)."""
    rows = [_row("a", "b", 0.9, True), _row("a", "b", 0.4, False), _row("c", "d", 0.8, True)]
    pairs = harvest_labeled_pairs(rows, corrections=[])
    assert [(p.score, p.label) for p in pairs] == [(0.9, True), (0.4, False), (0.8, True)]


def test_harvest_skips_correction_for_unjudged_pair_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A correction referencing a pair absent from the log is skipped and warned about."""
    rows = [_row("a", "b", 0.9, True)]
    corrections = [
        Correction(left_id="a", right_id="b", label=True),  # matches a row
        Correction(left_id="x", right_id="y", label=True),  # no matching row
    ]
    with caplog.at_level(logging.WARNING):
        pairs = harvest_labeled_pairs(rows, corrections)
    assert len(pairs) == 1  # only the judged pair yields a labeled pair
    assert "1 correction(s)" in caplog.text


def test_harvest_coerces_row_field_types() -> None:
    """Non-str ids and non-bool verdicts in a row are coerced to the model types."""
    rows = [{"left_id": 1, "right_id": 2, "score": "0.75", "verdict": 1}]
    pairs = harvest_labeled_pairs(rows, corrections=[])
    assert pairs[0].left_id == "1"
    assert pairs[0].right_id == "2"
    assert pairs[0].score == 0.75
    assert pairs[0].label is True


def test_harvest_decision_only_row_carries_score_none() -> None:
    """A v3 decision-only row (``score: null``) yields a LabeledPair with
    ``score=None`` -- the label is still usable, but there is no score for
    calibration. The null is carried as-is, never coerced to 0.0."""
    rows: list[dict[str, Any]] = [
        {"v": 3, "left_id": "a", "right_id": "b", "score": None, "verdict": True}
    ]
    pairs = harvest_labeled_pairs(rows, corrections=[])
    assert pairs[0].score is None
    assert pairs[0].label is True


def test_harvest_abstention_row_is_skipped_not_labeled_false() -> None:
    """A v3 abstention row (verdict=None: the judge neither decided nor scored)
    carries NO usable label and must be omitted -- never coerced to a False
    non-match. Fabricating a "not a match" would seed silver labels with a
    verdict the judge never gave, the label-side twin of coercing a null score
    to 0.0. A real verdict row alongside it still passes through."""
    rows: list[dict[str, Any]] = [
        {"v": 3, "left_id": "a", "right_id": "b", "score": None, "verdict": None},
        {"v": 3, "left_id": "c", "right_id": "d", "score": 0.9, "verdict": True},
    ]
    pairs = harvest_labeled_pairs(rows, corrections=[])
    # The abstention is dropped; only the decided pair survives.
    assert [(p.left_id, p.right_id) for p in pairs] == [("c", "d")]
    assert pairs[0].label is True


def test_harvest_correction_rescues_an_abstention_row() -> None:
    """A human correction supplies the label an abstention lacked, so the pair
    IS harvested (from the correction), proving the skip is verdict-only, not a
    blanket drop of the pair."""
    rows: list[dict[str, Any]] = [
        {"v": 3, "left_id": "a", "right_id": "b", "score": None, "verdict": None},
    ]
    pairs = harvest_labeled_pairs(
        rows, corrections=[Correction(left_id="a", right_id="b", label=True)]
    )
    assert len(pairs) == 1
    assert pairs[0].label is True
    assert pairs[0].source == "correction"
    assert pairs[0].score is None


# --------------------------------------------------------------------------- #
# derive_threshold_from_pairs wiring                                          #
# --------------------------------------------------------------------------- #


def test_derive_threshold_from_pairs_youden() -> None:
    """The bridge feeds scores+labels to derive_threshold and returns its cut."""
    pairs = [
        LabeledPair(left_id="a", right_id="b", score=0.1, label=False, source="verdict"),
        LabeledPair(left_id="c", right_id="d", score=0.2, label=False, source="verdict"),
        LabeledPair(left_id="e", right_id="f", score=0.8, label=True, source="correction"),
        LabeledPair(left_id="g", right_id="h", score=0.9, label=True, source="correction"),
    ]
    assert derive_threshold_from_pairs(pairs) == pytest.approx(0.8)


def test_derive_threshold_from_pairs_percentile_passthrough() -> None:
    """method/percentile kwargs pass through to derive_threshold."""
    pairs = [
        LabeledPair(left_id="a", right_id="b", score=0.0, label=False, source="verdict"),
        LabeledPair(left_id="c", right_id="d", score=1.0, label=True, source="correction"),
    ]
    assert derive_threshold_from_pairs(
        pairs, method="percentile", percentile=50.0
    ) == pytest.approx(0.5)


def test_derive_threshold_from_pairs_raises_on_scoreless_pair() -> None:
    """A score-less pair (decision-only judge, score=None) makes a *score*
    threshold underivable. The guard raises a clear ValueError naming the
    offending pair and the cause -- rather than silently calibrating on the
    biased subset that happens to have scores."""
    pairs = [
        LabeledPair(left_id="a", right_id="b", score=0.2, label=False, source="verdict"),
        LabeledPair(left_id="c", right_id="d", score=None, label=True, source="correction"),
    ]
    with pytest.raises(ValueError, match="decision-only judge has no scores") as excinfo:
        derive_threshold_from_pairs(pairs)
    assert "c/d" in str(excinfo.value)  # the offending pair is named


def test_derive_threshold_from_pairs_propagates_single_class_error() -> None:
    """Youden on a single-class label set raises (propagated from derive_threshold)."""
    pairs = [
        LabeledPair(left_id="a", right_id="b", score=0.1, label=True, source="verdict"),
        LabeledPair(left_id="c", right_id="d", score=0.9, label=True, source="correction"),
    ]
    with pytest.raises(ValueError, match="both classes"):
        derive_threshold_from_pairs(pairs)


# --------------------------------------------------------------------------- #
# derive_threshold_from_pairs silver-only guardrail                            #
# --------------------------------------------------------------------------- #


def test_derive_threshold_from_pairs_warns_on_silver_only_input() -> None:
    """All-silver input (every source=='verdict') fires the circularity warning."""
    pairs = [
        LabeledPair(left_id="a", right_id="b", score=0.1, label=False, source="verdict"),
        LabeledPair(left_id="c", right_id="d", score=0.2, label=False, source="verdict"),
        LabeledPair(left_id="e", right_id="f", score=0.8, label=True, source="verdict"),
        LabeledPair(left_id="g", right_id="h", score=0.9, label=True, source="verdict"),
    ]
    with pytest.warns(UserWarning, match="silver-only calibration is circular"):
        threshold = derive_threshold_from_pairs(pairs)
    assert threshold == pytest.approx(0.8)  # the warning does not change the result


def test_derive_threshold_from_pairs_no_warning_when_correction_present(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """One human-corrected pair in the mix means gold is present: no warning."""
    pairs = [
        LabeledPair(left_id="a", right_id="b", score=0.1, label=False, source="verdict"),
        LabeledPair(left_id="c", right_id="d", score=0.2, label=False, source="verdict"),
        LabeledPair(left_id="e", right_id="f", score=0.8, label=True, source="verdict"),
        LabeledPair(left_id="g", right_id="h", score=0.9, label=True, source="correction"),
    ]
    threshold = derive_threshold_from_pairs(pairs)
    assert threshold == pytest.approx(0.8)
    assert len(recwarn) == 0


def test_derive_threshold_from_pairs_empty_input_raises_without_warning(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Empty input still hits the existing ValueError -- the guard must not warn first."""
    with pytest.raises(ValueError, match="non-empty"):
        derive_threshold_from_pairs([])
    assert len(recwarn) == 0
