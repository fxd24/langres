"""Cross-slice integration for the judgement-contract flywheel.

Each Wave-2 slice of the judgement contract was built and tested in isolation
(``judgement_log``, ``review``, ``benchmark``, ``llm_judge``). These tests prove
the slices compose: a binary **decision** judge's output flows
judge -> ``JudgementLog`` (v3) -> ``read()`` -> ``select_for_review``, the loop
that *silently no-op'd* on any binary judge before this contract existed
(``select_for_review`` returned ``[]``, documented as "the loop is exhausted",
having never started).

Zero spend: hand-built :class:`PairwiseJudgement` objects stand in for a real
``LLMMatcher`` so no client is ever constructed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from langres.tracking.judgement_log import JudgementLog
from langres.core.models import PairwiseJudgement
from langres.core.review import select_for_review


def _decider(
    left_id: str,
    right_id: str,
    *,
    decision: bool,
    confidence: float | None = None,
) -> PairwiseJudgement:
    """A binary decider: a decision and no score (the shape a real LLMMatcher emits)."""
    return PairwiseJudgement(
        left_id=left_id,
        right_id=right_id,
        decision=decision,
        score=None,
        confidence=confidence,
        confidence_source="logprob" if confidence is not None else "unrequested",
        score_type="prob_llm",
        decision_step="binary_llm",
        provenance={},
    )


def test_decision_flows_judge_to_log_to_review(tmp_path: Path) -> None:
    """A decider's decision + confidence survives the log round-trip and drives review."""
    log = JudgementLog(tmp_path / "conf.jsonl")
    for lid, rid, decision, conf in [
        ("a", "b", True, 0.99),  # very sure
        ("c", "d", True, 0.55),  # unsure -> should surface first
        ("e", "f", False, 0.90),  # fairly sure
    ]:
        j = _decider(lid, rid, decision=decision, confidence=conf)
        log.append(j, verdict=decision)

    rows = log.read()
    # v3 persists the decision and the credence (not just a fabricated score).
    assert all(r["decision"] is not None for r in rows)
    assert rows[1]["decision"] is True
    assert rows[1]["confidence"] == pytest.approx(0.55)

    items = select_for_review(rows, strategy="uncertainty", threshold=0.5, margin=0.5)
    assert items, "the flywheel returned nothing on a confidence-carrying binary log"
    # Ranked by credence: the pair the judge was least sure of comes first.
    assert {items[0].left_id, items[0].right_id} == {"c", "d"}
    assert items[0].confidence == pytest.approx(0.55)


def test_confidence_less_binary_log_raises_instead_of_silent_noop(tmp_path: Path) -> None:
    """The pre-contract bug's tell was ``[]``; now it must RAISE, naming the fix.

    A binary log with no confidence has no uncertainty gradient (every
    ``|score - threshold|`` is identical), so uncertainty selection cannot work.
    Returning ``[]`` there is the silent no-op that reported "exhausted" having
    never started -- asserting the empty list would pass forever. Assert the
    raise and its message instead.
    """
    log = JudgementLog(tmp_path / "noconf.jsonl")
    for lid, rid, decision in [("a", "b", True), ("c", "d", False)]:
        log.append(_decider(lid, rid, decision=decision), verdict=decision)

    with pytest.raises(ValueError) as excinfo:
        select_for_review(log.read(), strategy="uncertainty", threshold=0.5)
    message = str(excinfo.value)
    assert "disagreement" in message
    assert 'confidence="logprob"' in message


def test_disagreement_is_the_working_fallback_on_two_binary_logs(tmp_path: Path) -> None:
    """The fallback the raise points at must actually run on binary logs."""
    log_a = JudgementLog(tmp_path / "a.jsonl")
    log_b = JudgementLog(tmp_path / "b.jsonl")
    # Same pair, opposite decisions across the two judges.
    log_a.append(_decider("a", "b", decision=True), verdict=True)
    log_b.append(_decider("a", "b", decision=False), verdict=False)

    items = select_for_review(log_a.read(), strategy="disagreement", against=log_b.read())
    assert {(i.left_id, i.right_id) for i in items} == {("a", "b")}


def test_cascade_shaped_cost_key_persists_through_the_log(tmp_path: Path) -> None:
    """A cascade writes ``llm_cost_usd``; the log must not silently persist 0.0."""
    log = JudgementLog(tmp_path / "cost.jsonl")
    j = PairwiseJudgement(
        left_id="a",
        right_id="b",
        decision=True,
        score=None,
        score_type="prob_llm",
        decision_step="cascade",
        provenance={"llm_cost_usd": 0.0042, "model": "glm"},
    )
    log.append(j, verdict=True)
    assert log.read()[0]["cost_usd"] == pytest.approx(0.0042)
