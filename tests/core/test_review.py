"""Tests for langres.core.review (the flywheel's review selector).

Covers the ``review_queue.jsonl`` contract (:class:`ReviewItem` /
:class:`ReviewQueue` snapshot semantics) and the three selection strategies of
:func:`select_for_review` (uncertainty, disagreement, first-class audit), plus
the audit mix-in, correction exclusion, last-write-wins pair keying, the
malformed-row skip path, and the ``records=`` content join. Everything here is
zero-spend and dependency-light: plain dicts and JSONL round-trips, no judge,
no model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from langres.core.harvest import Correction
from langres.core.review import ReviewItem, ReviewQueue, select_for_review

_LOGGER_NAME = "langres.core.review"


def _row(left: str, right: str, score: float, verdict: bool, **overrides: Any) -> dict[str, Any]:
    """A minimal JudgementLog-format row (the keys the selector actually reads)."""
    row: dict[str, Any] = {
        "v": 1,
        "left_id": left,
        "right_id": right,
        "score": score,
        "verdict": verdict,
        "model": "test-model",
        "cost_usd": 0.0,
        "decision_step": "test_judge",
    }
    row.update(overrides)
    return row


def _pairs(items: list[ReviewItem]) -> list[tuple[str, str]]:
    return [(item.left_id, item.right_id) for item in items]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == _LOGGER_NAME and record.levelno == logging.WARNING
    ]


# --------------------------------------------------------------------------- #
# Validation errors                                                           #
# --------------------------------------------------------------------------- #


def test_uncertainty_without_threshold_raises() -> None:
    """The uncertainty strategy has no default cut -- threshold= is required."""
    with pytest.raises(ValueError, match="threshold"):
        select_for_review([_row("a", "b", 0.5, True)], strategy="uncertainty")


def test_disagreement_without_against_raises() -> None:
    """The disagreement strategy needs a second log to disagree with."""
    with pytest.raises(ValueError, match="against"):
        select_for_review([_row("a", "b", 0.5, True)], strategy="disagreement")


def test_unknown_strategy_raises() -> None:
    """A typo'd strategy fails loudly instead of silently selecting nothing."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        select_for_review([_row("a", "b", 0.5, True)], strategy=cast(Any, "certainty"))


# --------------------------------------------------------------------------- #
# Uncertainty strategy                                                        #
# --------------------------------------------------------------------------- #


def test_uncertainty_filters_by_margin_and_sorts_most_uncertain_first() -> None:
    """Only |score - threshold| <= margin qualifies, ordered by that distance."""
    rows = [
        _row("a", "b", 0.55, True),  # distance 0.05
        _row("c", "d", 0.42, False),  # distance 0.08
        _row("e", "f", 0.50, True),  # distance 0.00
        _row("g", "h", 0.90, True),  # outside the margin
        _row("i", "j", 0.39, False),  # distance 0.11 -- just outside
    ]
    items = select_for_review(
        rows, strategy="uncertainty", threshold=0.5, margin=0.1, audit_fraction=0.0
    )
    assert _pairs(items) == [("e", "f"), ("a", "b"), ("c", "d")]
    assert all(item.reason == "uncertainty" for item in items)


def test_uncertainty_item_carries_log_fields_and_details() -> None:
    """Score, verdict, decision_step, model and threshold/distance all survive."""
    rows = [_row("a", "b", 0.55, True, decision_step="cascade_student")]
    (item,) = select_for_review(rows, strategy="uncertainty", threshold=0.5, audit_fraction=0.0)
    assert item.v == 1
    assert item.score == 0.55
    assert item.verdict is True
    assert item.decision_step == "cascade_student"
    assert item.model == "test-model"
    assert item.details["threshold"] == 0.5
    assert item.details["distance"] == pytest.approx(0.05)
    # Privacy posture: records= was omitted, so the item is ids-only.
    assert item.left_record is None
    assert item.right_record is None


def test_uncertainty_none_model_stays_none() -> None:
    """JudgementLog writes model: null for offline judges -- must not become 'None'."""
    rows = [_row("a", "b", 0.5, True, model=None)]
    (item,) = select_for_review(rows, strategy="uncertainty", threshold=0.5, audit_fraction=0.0)
    assert item.model is None


def test_limit_truncates_primary_items() -> None:
    """At most limit items come back, keeping the most uncertain ones."""
    rows = [_row(f"l{i}", f"r{i}", 0.5 + i * 0.01, True) for i in range(5)]
    items = select_for_review(
        rows, strategy="uncertainty", threshold=0.5, limit=2, audit_fraction=0.0
    )
    assert _pairs(items) == [("l0", "r0"), ("l1", "r1")]


# --------------------------------------------------------------------------- #
# Pair keying: dedupe, last-write-wins, correction exclusion                  #
# --------------------------------------------------------------------------- #


def test_duplicate_pair_last_write_wins_even_when_ids_are_swapped() -> None:
    """(a,b) and (b,a) are the same pair; the later logged row wins."""
    rows = [
        _row("a", "b", 0.50, True),
        _row("b", "a", 0.55, False),  # re-judged later, reversed orientation
    ]
    (item,) = select_for_review(rows, strategy="uncertainty", threshold=0.5, audit_fraction=0.0)
    assert item.left_id == "b"
    assert item.right_id == "a"
    assert item.score == 0.55
    assert item.verdict is False


def test_corrected_pairs_are_never_reasked() -> None:
    """A pair answered in corrections= is excluded, orientation-independently."""
    rows = [
        _row("a", "b", 0.50, True),
        _row("c", "d", 0.52, False),
    ]
    corrections = [Correction(left_id="b", right_id="a", label=True)]
    items = select_for_review(
        rows,
        strategy="uncertainty",
        threshold=0.5,
        corrections=corrections,
        audit_fraction=0.0,
    )
    assert _pairs(items) == [("c", "d")]


# --------------------------------------------------------------------------- #
# Disagreement strategy                                                       #
# --------------------------------------------------------------------------- #


def test_disagreement_selects_differing_verdicts_sorted_by_score_gap() -> None:
    """Only verdict flips qualify, largest score gap (most disagreement) first."""
    rows = [
        _row("a", "b", 0.9, True),
        _row("c", "d", 0.6, True),
        _row("e", "f", 0.3, False),  # same verdict in both logs
        _row("g", "h", 0.7, True),  # absent from the second log
    ]
    against = [
        _row("a", "b", 0.2, False),  # gap 0.7
        _row("d", "c", 0.5, False),  # gap 0.1, reversed orientation still joins
        _row("e", "f", 0.25, False),
    ]
    items = select_for_review(rows, strategy="disagreement", against=against, audit_fraction=0.0)
    assert _pairs(items) == [("a", "b"), ("c", "d")]
    assert all(item.reason == "disagreement" for item in items)


def test_disagreement_details_carry_the_against_side() -> None:
    """The second log's score/verdict/model/decision_step land in details."""
    rows = [_row("a", "b", 0.9, True)]
    against = [_row("a", "b", 0.2, False, model="frontier", decision_step="teacher_judge")]
    (item,) = select_for_review(rows, strategy="disagreement", against=against, audit_fraction=0.0)
    assert item.score == 0.9  # primary log's side stays on the item itself
    assert item.details == {
        "against_score": 0.2,
        "against_verdict": False,
        "against_model": "frontier",
        "against_decision_step": "teacher_judge",
    }


# --------------------------------------------------------------------------- #
# Audit: first-class strategy, mix-in, determinism                            #
# --------------------------------------------------------------------------- #


def test_audit_strategy_is_a_seeded_sample_needing_no_threshold() -> None:
    """strategy='audit' works without threshold/against and honors limit."""
    rows = [_row(f"l{i}", f"r{i}", i / 30, i % 2 == 0) for i in range(30)]
    items = select_for_review(rows, strategy="audit", limit=5)
    assert len(items) == 5
    assert all(item.reason == "audit" for item in items)


def test_audit_strategy_is_deterministic_per_seed() -> None:
    """Same seed -> identical batch; a different seed reshuffles it."""
    rows = [_row(f"l{i}", f"r{i}", i / 30, True) for i in range(30)]
    first = select_for_review(rows, strategy="audit", limit=5, seed=0)
    second = select_for_review(rows, strategy="audit", limit=5, seed=0)
    other_seed = select_for_review(rows, strategy="audit", limit=5, seed=1)
    assert first == second
    assert first != other_seed


def test_audit_strategy_with_pool_smaller_than_limit_returns_pool() -> None:
    """A tiny log just returns everything judged (minus corrected pairs)."""
    rows = [_row("a", "b", 0.9, True), _row("c", "d", 0.1, False)]
    corrections = [Correction(left_id="a", right_id="b", label=True)]
    items = select_for_review(rows, strategy="audit", limit=10, corrections=corrections)
    assert _pairs(items) == [("c", "d")]


def test_audit_mixin_reserves_a_slice_of_the_limit() -> None:
    """limit=20, audit_fraction=0.1 -> int(20 * 0.1) = 2 audit + 18 primary slots."""
    in_band = [_row(f"l{i}", f"r{i}", 0.5, True) for i in range(25)]
    out_of_band = [_row(f"L{i}", f"R{i}", 0.95, True) for i in range(10)]
    items = select_for_review(
        in_band + out_of_band,
        strategy="uncertainty",
        threshold=0.5,
        limit=20,
        audit_fraction=0.1,
    )
    assert len(items) == 20
    reasons = [item.reason for item in items]
    assert reasons.count("uncertainty") == 18
    assert reasons.count("audit") == 2
    # Primary items come first; the audit slice is appended.
    assert reasons == ["uncertainty"] * 18 + ["audit"] * 2
    # The audit slice never duplicates a pair already selected as primary.
    assert len({frozenset(pair) for pair in _pairs(items)}) == 20


def test_audit_mixin_is_deterministic_per_seed() -> None:
    """The mixed-in audit slice flows through the same seeded RNG."""
    rows = [_row(f"l{i}", f"r{i}", i / 40, True) for i in range(40)]
    kwargs: dict[str, Any] = {
        "strategy": "uncertainty",
        "threshold": 0.5,
        "margin": 0.2,
        "limit": 10,
        "audit_fraction": 0.2,
    }
    assert select_for_review(rows, **kwargs, seed=7) == select_for_review(rows, **kwargs, seed=7)


def test_audit_fraction_zero_is_the_no_audit_escape_hatch() -> None:
    """audit_fraction=0.0 yields a pure-strategy batch."""
    rows = [_row(f"l{i}", f"r{i}", 0.5, True) for i in range(30)]
    items = select_for_review(
        rows, strategy="uncertainty", threshold=0.5, limit=10, audit_fraction=0.0
    )
    assert len(items) == 10
    assert all(item.reason == "uncertainty" for item in items)


def test_exhausted_primary_returns_empty_with_no_audit_padding() -> None:
    """Zero primary items is the stop signal -- [] even though pairs remain."""
    rows = [_row("a", "b", 0.95, True), _row("c", "d", 0.05, False)]
    items = select_for_review(
        rows,
        strategy="uncertainty",
        threshold=0.5,
        margin=0.1,
        audit_fraction=0.5,
    )
    assert items == []


def test_empty_log_returns_empty() -> None:
    """No judgements yet -> nothing to review, for any strategy."""
    assert select_for_review([], strategy="uncertainty", threshold=0.5) == []
    assert select_for_review([], strategy="audit") == []


# --------------------------------------------------------------------------- #
# Malformed judgement rows                                                    #
# --------------------------------------------------------------------------- #


def test_malformed_rows_are_skipped_with_one_summary_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every malformed shape is dropped; exactly ONE warning carries the count."""
    good = _row("a", "b", 0.5, True)
    malformed: list[dict[str, Any]] = [
        {k: v for k, v in _row("m1", "x", 0.5, True).items() if k != "left_id"},
        {k: v for k, v in _row("m2", "x", 0.5, True).items() if k != "right_id"},
        {k: v for k, v in _row("m3", "x", 0.5, True).items() if k != "score"},
        {k: v for k, v in _row("m4", "x", 0.5, True).items() if k != "verdict"},
        _row("m5", "x", float("nan"), True),
        _row("m6", "x", float("inf"), True),
        _row("m7", "x", 1.5, True),
        _row("m8", "x", -0.1, True),
        _row("m9", "x", cast(Any, "0.5"), True),  # non-numeric score
        _row("m10", "x", 0.5, cast(Any, "yes")),  # non-bool verdict
        _row("m11", "x", 0.5, cast(Any, 1)),  # int is not a verdict
        _row("m12", "x", cast(Any, True), True),  # bool is not a score
    ]
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        items = select_for_review(
            [good, *malformed],
            strategy="uncertainty",
            threshold=0.5,
            margin=1.0,
            audit_fraction=0.0,
        )
    assert _pairs(items) == [("a", "b")]
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "12" in warnings[0].getMessage()


def test_malformed_rows_in_the_against_log_are_also_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The second log gets the same sanitation (and its own summary warning)."""
    rows = [_row("a", "b", 0.9, True)]
    against = [
        _row("a", "b", float("nan"), False),  # malformed -- would have disagreed
        _row("c", "d", 0.5, True),
    ]
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        items = select_for_review(
            rows, strategy="disagreement", against=against, audit_fraction=0.0
        )
    assert items == []
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "against" in warnings[0].getMessage()


# --------------------------------------------------------------------------- #
# records= content join                                                       #
# --------------------------------------------------------------------------- #


class _RecordModel(BaseModel):
    id: str
    name: str


class _RecordObject:
    def __init__(self, id: str, name: str) -> None:  # noqa: A002
        self.id = id
        self.name = name


def test_records_join_accepts_mappings() -> None:
    rows = [_row("a", "b", 0.5, True)]
    records = [{"id": "a", "name": "ACME"}, {"id": "b", "name": "Acme Corp"}]
    (item,) = select_for_review(
        rows, strategy="uncertainty", threshold=0.5, records=records, audit_fraction=0.0
    )
    assert item.left_record == {"id": "a", "name": "ACME"}
    assert item.right_record == {"id": "b", "name": "Acme Corp"}


def test_records_join_accepts_pydantic_models_and_plain_objects() -> None:
    """Anything with an id attribute joins; content comes from its fields."""
    rows = [_row("a", "b", 0.5, True)]
    records = [_RecordModel(id="a", name="ACME"), _RecordObject(id="b", name="Acme Corp")]
    (item,) = select_for_review(
        rows, strategy="uncertainty", threshold=0.5, records=records, audit_fraction=0.0
    )
    assert item.left_record == {"id": "a", "name": "ACME"}
    assert item.right_record == {"id": "b", "name": "Acme Corp"}


def test_records_join_matches_non_string_ids_as_strings() -> None:
    """Record ids are compared as strings, matching the log's string ids."""
    rows = [_row("1", "2", 0.5, True)]
    records = [{"id": 1, "name": "left"}, {"id": 2, "name": "right"}]
    (item,) = select_for_review(
        rows, strategy="uncertainty", threshold=0.5, records=records, audit_fraction=0.0
    )
    assert item.left_record == {"id": 1, "name": "left"}
    assert item.right_record == {"id": 2, "name": "right"}


def test_records_without_ids_skip_the_join_with_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Positional-id logs (schema-less dedupe) hit this: ONE warning, ids-only items."""
    rows = [_row("0", "1", 0.5, True)]
    records = [{"name": "no id here"}, {"name": "me neither"}]
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        (item,) = select_for_review(
            rows,
            strategy="uncertainty",
            threshold=0.5,
            records=records,
            audit_fraction=0.0,
        )
    assert item.left_record is None
    assert item.right_record is None
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "2" in warnings[0].getMessage()


def test_logged_id_missing_from_records_leaves_that_side_ids_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A partial record list joins what it can and warns about the misses."""
    rows = [_row("a", "b", 0.5, True)]
    records = [{"id": "a", "name": "ACME"}]  # nothing for "b"
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        (item,) = select_for_review(
            rows,
            strategy="uncertainty",
            threshold=0.5,
            records=records,
            audit_fraction=0.0,
        )
    assert item.left_record == {"id": "a", "name": "ACME"}
    assert item.right_record is None
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "1" in warnings[0].getMessage()


# --------------------------------------------------------------------------- #
# ReviewQueue snapshot semantics                                              #
# --------------------------------------------------------------------------- #


def _items() -> list[ReviewItem]:
    return select_for_review(
        [_row("a", "b", 0.55, True), _row("c", "d", 0.48, False)],
        strategy="uncertainty",
        threshold=0.5,
        records=[{"id": "a", "name": "ACME"}, {"id": "b", "name": "Acme Corp"}],
        audit_fraction=0.0,
    )


def test_queue_read_missing_file_is_empty(tmp_path: Path) -> None:
    """A never-written queue reads back as [] (not an error)."""
    assert ReviewQueue(tmp_path / "nope.jsonl").read() == []


def test_queue_round_trip_preserves_items(tmp_path: Path) -> None:
    """Write then read returns the same items, order and fields intact."""
    queue = ReviewQueue(tmp_path / "queue.jsonl")
    items = _items()
    queue.write(items)
    assert queue.read() == items


def test_queue_write_creates_parent_dirs(tmp_path: Path) -> None:
    """Writing into a missing directory creates it (mirrors the two logs)."""
    queue = ReviewQueue(tmp_path / "nested" / "dir" / "queue.jsonl")
    queue.write(_items())
    assert queue.path.exists()


def test_queue_write_is_a_snapshot_that_truncates(tmp_path: Path) -> None:
    """A second write replaces the queue -- it never appends stale batches."""
    queue = ReviewQueue(tmp_path / "queue.jsonl")
    queue.write(_items())
    survivor = _items()[:1]
    queue.write(survivor)
    assert queue.read() == survivor


def test_queue_read_skips_blank_lines(tmp_path: Path) -> None:
    """Trailing newlines / blank lines don't break the reload."""
    queue = ReviewQueue(tmp_path / "queue.jsonl")
    queue.write(_items()[:1])
    with queue.path.open("a", encoding="utf-8") as fh:
        fh.write("\n   \n")
    assert len(queue.read()) == 1


def test_queue_corrupt_json_line_raises_with_line_number(tmp_path: Path) -> None:
    """Invalid JSON names the offending line and says to regenerate."""
    path = tmp_path / "queue.jsonl"
    queue = ReviewQueue(path)
    queue.write(_items()[:1])
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(ValueError, match="line 2") as excinfo:
        queue.read()
    assert "regenerate" in str(excinfo.value)


def test_queue_schema_invalid_line_raises_with_line_number(tmp_path: Path) -> None:
    """Valid JSON that is not a ReviewItem (hand edit) fails the same way."""
    path = tmp_path / "queue.jsonl"
    path.write_text('{"left_id": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        ReviewQueue(path).read()


# --------------------------------------------------------------------------- #
# ReviewItem contract                                                         #
# --------------------------------------------------------------------------- #


def test_review_item_minimal_defaults() -> None:
    """Only ids/score/verdict/reason are required; the rest defaults empty."""
    item = ReviewItem(left_id="a", right_id="b", score=0.5, verdict=True, reason="audit")
    assert item.v == 1
    assert item.decision_step is None
    assert item.model is None
    assert item.left_record is None
    assert item.right_record is None
    assert item.details == {}
