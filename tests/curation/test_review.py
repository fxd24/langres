"""Tests for langres.curation.review (the flywheel's review selector).

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

from langres.curation.harvest import Correction
from langres.core.judgement_log import JudgementLog, LoggingMatcher
from langres.core.models import CompanySchema, ERCandidate
from langres.curation.review import ReviewItem, ReviewQueue, _is_well_formed, select_for_review
from langres.testing import ScriptedJudge

_LOGGER_NAME = "langres.curation.review"


def _row(
    left: str, right: str, score: float | None, verdict: bool, **overrides: Any
) -> dict[str, Any]:
    """A minimal JudgementLog-format row (the keys the selector actually reads).

    ``score=None`` builds a decider row (a binary judge that emits a decision and
    no score); pass ``decision=``/``confidence=``/``confidence_source=`` via
    ``overrides`` to model the v3 columns.
    """
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


@pytest.mark.parametrize("bad", [1.5, -0.1, 10.0])
def test_audit_fraction_out_of_range_raises(bad: float) -> None:
    """audit_fraction outside [0, 1] is rejected -- a value > 1 would otherwise let
    the audit slice exceed limit; a negative one would trip rng.sample."""
    with pytest.raises(ValueError, match="audit_fraction must be in"):
        select_for_review(
            [_row("a", "b", 0.5, True)],
            strategy="uncertainty",
            threshold=0.5,
            audit_fraction=bad,
        )


def test_negative_limit_raises() -> None:
    """A negative limit would otherwise make rng.sample's count negative and
    raise a raw ValueError from random.Random.sample instead of a clear one."""
    with pytest.raises(ValueError, match="limit must be >= 0"):
        select_for_review([_row("a", "b", 0.5, True)], strategy="audit", limit=-1)


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
    """Only verdict flips qualify, largest score gap (most disagreement) first.

    Insertion order is deliberately the OPPOSITE of the expected output (the
    smaller-gap pair c-d is logged first), so this assertion fails if the sort
    key is dropped -- it is not accidentally satisfied by input ordering.
    """
    rows = [
        _row("c", "d", 0.6, True),  # gap 0.1 -- logged first
        _row("a", "b", 0.9, True),  # gap 0.7 -- logged second, but must sort first
        _row("e", "f", 0.3, False),  # same verdict in both logs -> skipped
        _row("g", "h", 0.7, True),  # absent from the second log -> skipped
    ]
    against = [
        _row("d", "c", 0.5, False),  # gap 0.1, reversed orientation still joins
        _row("a", "b", 0.2, False),  # gap 0.7
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
# Binary / decision judge: the flywheel must actually run (A3c)                #
# --------------------------------------------------------------------------- #


def test_binary_decision_row_is_well_formed_signal_less_row_is_not() -> None:
    """A binary decider (score None, bool decision/verdict) is usable; a pure
    ranker (finite score, no verdict) is usable; a row with no ids or no
    actionable signal is not. This is the relaxed ``_is_well_formed`` contract."""
    assert _is_well_formed({"left_id": "a", "right_id": "b", "score": None, "decision": True})
    assert _is_well_formed({"left_id": "a", "right_id": "b", "score": None, "verdict": False})
    assert _is_well_formed({"left_id": "a", "right_id": "b", "score": 0.6})  # ranker, no verdict
    assert not _is_well_formed({"right_id": "b", "decision": True})  # no left_id
    assert not _is_well_formed({"left_id": "a", "right_id": "b"})  # ids only, no signal
    assert not _is_well_formed(  # non-bool verdict AND unusable score
        {"left_id": "a", "right_id": "b", "score": float("nan"), "verdict": "yes"}
    )


def test_uncertainty_on_confidenceless_binary_log_raises_not_empty() -> None:
    """THE flagship fix. A binary decider log (every score None, no confidence)
    has nothing to rank by uncertainty. It must RAISE a ValueError naming the fix
    -- never return [], which is indistinguishable from a finished loop and let
    the bug hide forever behind ``[] == []``."""
    binary_log = [
        _row("a", "b", None, True, decision=True),
        _row("c", "d", None, False, decision=False),
        _row("e", "f", None, True, decision=True),
    ]
    with pytest.raises(ValueError) as excinfo:
        select_for_review(binary_log, strategy="uncertainty", threshold=0.5)
    message = str(excinfo.value)
    assert "disagreement" in message
    assert 'confidence="logprob"' in message


def test_uncertainty_on_logged_binary_judge_raises_end_to_end(tmp_path: Path) -> None:
    """The real flywheel path: a binary judge whose 0/1 scores are logged by
    LoggingMatcher produces a log the uncertainty selector cannot rank -- proving
    the no-op bug is caught on the actual logged shape, not just hand-written rows."""
    candidates = [
        ERCandidate(
            left=CompanySchema(id="a", name="Acme"),
            right=CompanySchema(id="b", name="Acme Inc"),
            blocker_name="t",
        ),
        ERCandidate(
            left=CompanySchema(id="c", name="Beta"),
            right=CompanySchema(id="d", name="Gamma"),
            blocker_name="t",
        ),
    ]
    log = JudgementLog(tmp_path / "judgements.jsonl")
    # A binary decider: every pair is scored exactly 1.0 or 0.0, never a middling
    # value -- so |score - threshold| is a constant and there is no gradient.
    judge: ScriptedJudge[CompanySchema] = ScriptedJudge(
        lambda cand: 1.0 if cand.left.id == "a" else 0.0
    )
    list(LoggingMatcher(judge, log=log, threshold=0.5).forward(iter(candidates)))

    rows = log.read()
    assert rows and all(row["score"] in (0.0, 1.0) for row in rows)  # genuinely binary
    with pytest.raises(ValueError, match="disagreement"):
        select_for_review(rows, strategy="uncertainty", threshold=0.5)


def test_uncertainty_ranks_by_confidence_least_confident_first() -> None:
    """With a logged confidence (the confidence='logprob' path) rank by
    |confidence - 0.5| -- least confident (closest to 0.5) first -- not by score.
    Deciders carry score None; the credence and its source surface on the item."""
    rows = [
        # insertion order deliberately NOT the output order, so the sort is exercised
        _row("c", "d", None, False, decision=False, confidence=0.95, confidence_source="logprob"),
        _row("a", "b", None, True, decision=True, confidence=0.52, confidence_source="logprob"),
        _row("e", "f", None, True, decision=True, confidence=0.60, confidence_source="logprob"),
    ]
    items = select_for_review(
        rows, strategy="uncertainty", threshold=0.5, margin=0.15, audit_fraction=0.0
    )
    assert _pairs(items) == [("a", "b"), ("e", "f")]  # 0.95 is outside the |.5|+/-.15 band
    assert all(item.reason == "uncertainty" for item in items)
    first = items[0]
    assert first.score is None  # a decider has no score
    assert first.confidence == 0.52
    assert first.confidence_source == "logprob"
    assert first.verdict is True
    assert first.details["distance"] == pytest.approx(0.02)


def test_uncertainty_mixed_log_does_not_drop_score_only_rows() -> None:
    """A MIXED log -- some rows carry a logprob confidence, some only a score
    (a CascadeMatcher: cheap-student score-only rows + escalated logprob rows) --
    must review BOTH bands. The score-only uncertain pair must not vanish just
    because a confidence-bearing row exists (the silent no-op, relocated). The
    credence-ranked row comes first, then the score-ranked one."""
    rows = [
        # escalated, carries a real credence near 0.5 (maximally uncertain)
        _row("a", "b", None, True, decision=True, confidence=0.52, confidence_source="logprob"),
        # cheap student, score-only, sits right on the threshold (uncertain)
        _row("c", "d", 0.51, True),
        # cheap student, score-only, far from threshold -> outside the band
        _row("e", "f", 0.99, True),
    ]
    items = select_for_review(
        rows, strategy="uncertainty", threshold=0.5, margin=0.1, audit_fraction=0.0
    )
    pairs = _pairs(items)
    assert ("a", "b") in pairs  # credence-bearing uncertain row kept
    assert ("c", "d") in pairs  # score-only uncertain row NOT dropped
    assert ("e", "f") not in pairs  # score-only but outside the band
    assert pairs == [("a", "b"), ("c", "d")]  # credence band first, then score band


def test_uncertainty_confident_about_everything_returns_empty_not_raise() -> None:
    """Confidence present but all far from 0.5 -> band empty -> [] (a genuinely
    finished loop). This is the *signal-exists* case: distinct from the no-signal
    RAISE. It must NOT raise -- nothing is uncertain enough to review."""
    rows = [
        _row("a", "b", None, True, decision=True, confidence=0.97, confidence_source="logprob"),
        _row("c", "d", None, False, decision=False, confidence=0.99, confidence_source="logprob"),
    ]
    items = select_for_review(
        rows, strategy="uncertainty", threshold=0.5, margin=0.1, audit_fraction=0.5
    )
    assert items == []


def test_disagreement_works_on_two_binary_logs() -> None:
    """The fallback the uncertainty raise points at must work on a binary/decision
    log (score None): compare decisions, skip agreements, and don't crash on the
    absent score gap."""
    student = [
        _row("a", "b", None, True, decision=True),  # match
        _row("c", "d", None, True, decision=True),  # both agree -> skipped
        _row("e", "f", None, False, decision=False),  # no-match
    ]
    teacher = [
        _row("a", "b", None, False, decision=False),  # disagrees with student
        _row("c", "d", None, True, decision=True),
        _row("e", "f", None, True, decision=True),  # disagrees with student
    ]
    items = select_for_review(student, strategy="disagreement", against=teacher, audit_fraction=0.0)
    assert {(item.left_id, item.right_id) for item in items} == {("a", "b"), ("e", "f")}
    assert all(item.reason == "disagreement" for item in items)
    ab = next(item for item in items if (item.left_id, item.right_id) == ("a", "b"))
    assert ab.verdict is True  # student's own decision, on the item
    assert ab.score is None  # a decider has no score
    assert ab.details["against_verdict"] is False  # teacher's decision
    assert ab.details["against_score"] is None


def test_item_surfaces_reasoning_and_defaults_missing_credence() -> None:
    """_build_item lifts reasoning onto the item (the reviewer sees *why*), and a
    v1/v2 row lacking the credence columns leaves them None -- not a crash. The
    score path itself is unchanged."""
    rows = [_row("a", "b", 0.55, True, reasoning="names match modulo the Corp suffix")]
    (item,) = select_for_review(rows, strategy="uncertainty", threshold=0.5, audit_fraction=0.0)
    assert item.reasoning == "names match modulo the Corp suffix"
    assert item.confidence is None
    assert item.confidence_source is None
    assert item.score == 0.55  # score path unchanged for a continuous judge


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
    """A row needs ids AND one actionable signal (a bool decision/verdict OR a
    finite score). Only rows with *neither* signal (or no ids) are dropped; ONE
    warning carries the count. A missing score alone (a decider) or a missing
    verdict alone (a ranker) is now usable, not malformed.
    """
    good = _row("a", "b", 0.5, True)
    malformed: list[dict[str, Any]] = [
        {k: v for k, v in _row("m1", "x", 0.5, True).items() if k != "left_id"},  # no left_id
        {k: v for k, v in _row("m2", "x", 0.5, True).items() if k != "right_id"},  # no right_id
        # ids present, but non-bool verdict AND an unusable score => no signal at all:
        _row("m3", "x", float("nan"), cast(Any, "yes")),  # nan score, str verdict
        _row("m4", "x", float("inf"), cast(Any, 1)),  # inf score, int (not bool) verdict
        _row("m5", "x", 1.5, cast(Any, "no")),  # out-of-range score, str verdict
        _row("m6", "x", -0.1, cast(Any, None)),  # out-of-range score, None verdict
        _row("m7", "x", cast(Any, "0.5"), cast(Any, 0)),  # non-numeric score, int verdict
        # neither score nor verdict, and no decision => nothing actionable:
        {k: v for k, v in _row("m8", "x", 0.5, True).items() if k not in ("score", "verdict")},
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
    assert "8" in warnings[0].getMessage()


def test_malformed_rows_in_the_against_log_are_also_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The second log gets the same sanitation (and its own summary warning)."""
    rows = [_row("a", "b", 0.9, True)]
    against = [
        # malformed: non-bool verdict AND unusable score => no signal (a bare
        # nan score with a valid verdict would now be usable, so make it truly
        # signal-less). Would have disagreed with rows if it had a real verdict.
        _row("a", "b", float("nan"), cast(Any, "no")),
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


def test_queue_out_of_range_score_raises_with_line_number(tmp_path: Path) -> None:
    """A hand-edited line with score outside [0, 1] fails the same corrupt-line way."""
    path = tmp_path / "queue.jsonl"
    path.write_text(
        '{"left_id": "a", "right_id": "b", "score": 1.5, "verdict": true, "reason": "audit"}\n',
        encoding="utf-8",
    )
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
