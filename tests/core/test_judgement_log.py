"""Tests for langres.core.judgement_log (JudgementLog / LoggingMatcher).

The opt-in signal-log seam (W0.2): a JSONL sink (``JudgementLog``) plus a
boundary-component ``Matcher`` wrapper (``LoggingMatcher``) that logs each
``PairwiseJudgement`` as it streams past, without breaking generator
laziness. Zero-spend throughout -- everything here uses plain fake Modules,
never a real judge.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pytest

from langres.clients.openrouter import BudgetExceeded
from langres.core.judgement_log import JudgementLog, LoggingMatcher
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.matcher import Matcher
from langres.core.reports import ScoreInspectionReport
from langres.core.runs import RunContext, capture_run


def _judgement(
    left_id: str = "a",
    right_id: str = "b",
    score: float | None = 0.9,
    *,
    decision: bool | None = None,
    confidence: float | None = None,
    confidence_source: Literal[
        "none", "unrequested", "logprob", "calibrated", "heuristic"
    ] = "none",
    reasoning: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id=left_id,
        right_id=right_id,
        decision=decision,
        score=score,
        score_type="heuristic",
        confidence=confidence,
        confidence_source=confidence_source,
        decision_step="test_step",
        reasoning=reasoning,
        provenance=provenance if provenance is not None else {},
    )


class _CountingModule(Matcher[object]):
    """Yields ``judgements`` one at a time, tracking how many were pulled.

    Used to prove ``LoggingMatcher.forward`` is lazy -- it must not pull more
    than the caller asked for.
    """

    def __init__(self, judgements: list[PairwiseJudgement]) -> None:
        self._judgements = judgements
        self.pulled = 0

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        list(candidates)
        for judgement in self._judgements:
            self.pulled += 1
            yield judgement

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError


class _BudgetBreachingModule(Matcher[object]):
    """Yields ``yielded``, then raises ``BudgetExceeded`` whose
    ``partial_judgements`` also carries ``tripping`` -- the judgement that
    was produced (and paid for) but never yielded, mirroring
    ``core.presets._SpendCappedMatcher``'s raise-before-yield behavior on the
    call that crosses the cap."""

    def __init__(self, yielded: list[PairwiseJudgement], tripping: PairwiseJudgement) -> None:
        self._yielded = yielded
        self._tripping = tripping

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        list(candidates)
        yield from self._yielded
        exc = BudgetExceeded("budget exceeded")
        exc.partial_judgements = [*self._yielded, self._tripping]
        raise exc

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# JudgementLog.append
# ---------------------------------------------------------------------------


class TestJudgementLogAppend:
    def test_append_writes_one_json_line_per_call(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement("a", "b"), verdict=True)
        log.append(_judgement("c", "d"), verdict=False)
        lines = (tmp_path / "log.jsonl").read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # every line is independently valid JSON

    def test_append_includes_schema_version_v3(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(), verdict=True)
        row = log.read()[0]
        assert row["v"] == 3

    def test_append_records_decision_confidence_and_source_from_judgement(
        self, tmp_path: Path
    ) -> None:
        """The v3 contract columns come straight off the judgement (not derived
        from ``verdict``): a decider's ``decision`` plus its earned confidence."""
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(
            _judgement(
                score=None,
                decision=True,
                confidence=0.82,
                confidence_source="logprob",
            ),
            verdict=True,
        )
        row = log.read()[0]
        assert row["decision"] is True
        assert row["score"] is None
        assert row["confidence"] == pytest.approx(0.82)
        assert row["confidence_source"] == "logprob"

    def test_append_defaults_decision_none_confidence_none_source_none(
        self, tmp_path: Path
    ) -> None:
        """A plain ranker (no decision, no confidence) logs the honest defaults."""
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(score=0.9), verdict=True)
        row = log.read()[0]
        assert row["decision"] is None
        assert row["confidence"] is None
        assert row["confidence_source"] == "none"

    def test_append_reads_cost_from_llm_cost_usd_key(self, tmp_path: Path) -> None:
        """The cost bug: CascadeChainMatcher writes spend under ``llm_cost_usd`` (not
        ``cost_usd``), so a bare ``.get("cost_usd")`` persisted 0.0 for every
        cascade row. The row must carry the real cost."""
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(provenance={"llm_cost_usd": 0.0042, "model": "glm"}), verdict=True)
        row = log.read()[0]
        assert row["cost_usd"] == pytest.approx(0.0042)

    def test_append_cost_prefers_cost_usd_over_llm_cost_usd(self, tmp_path: Path) -> None:
        """First of ``_COST_KEYS`` present wins: ``cost_usd`` takes precedence."""
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(provenance={"cost_usd": 0.01, "llm_cost_usd": 0.99}), verdict=True)
        row = log.read()[0]
        assert row["cost_usd"] == pytest.approx(0.01)

    def test_append_records_usage_vector_in_default_row(self, tmp_path: Path) -> None:
        """The token-usage vector is in the DEFAULT (features=False) row — it is
        non-PII (just counts) and is the whole point of logging, like cost_usd."""
        log = JudgementLog(tmp_path / "log.jsonl")
        usage = {"input_tokens": 120, "output_tokens": 40, "model": "gpt-4o-mini"}
        log.append(_judgement(provenance={"usage": usage}), verdict=True)
        row = log.read()[0]
        assert row["usage"] == usage

    def test_append_usage_is_none_for_non_llm_judge(self, tmp_path: Path) -> None:
        """A judge with no token usage (string/embedding) logs ``usage: null``."""
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(provenance={}), verdict=True)
        row = log.read()[0]
        assert row["usage"] is None

    def test_append_records_pair_ids_score_and_decision_step(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement("left1", "right1", 0.73), verdict=True)
        row = log.read()[0]
        assert row["left_id"] == "left1"
        assert row["right_id"] == "right1"
        assert row["score"] == pytest.approx(0.73)
        assert row["decision_step"] == "test_step"

    def test_append_verdict_reflects_the_passed_bool(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(), verdict=True)
        log.append(_judgement(), verdict=False)
        rows = log.read()
        assert rows[0]["verdict"] is True
        assert rows[1]["verdict"] is False

    def test_append_extracts_model_and_cost_from_provenance(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(
            _judgement(provenance={"model": "gpt-4o-mini", "cost_usd": 0.0021}), verdict=True
        )
        row = log.read()[0]
        assert row["model"] == "gpt-4o-mini"
        assert row["cost_usd"] == pytest.approx(0.0021)

    def test_append_missing_model_is_none(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(provenance={}), verdict=True)
        row = log.read()[0]
        assert row["model"] is None

    def test_append_model_falls_back_to_resolved_pipeline_model(self, tmp_path: Path) -> None:
        """A judgement without its own provenance model logs the caller's resolved id.

        This is the log half of the model-identity contract: the verbs pass
        their resolved ``model`` through ``LoggingMatcher``, so log rows and
        the result's ``model`` field agree even for judges (string/embedding)
        that never stamp ``provenance["model"]`` themselves.
        """
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(provenance={}), verdict=True, model="all-MiniLM-L6-v2")
        row = log.read()[0]
        assert row["model"] == "all-MiniLM-L6-v2"

    def test_append_judge_stamped_model_wins_over_fallback(self, tmp_path: Path) -> None:
        """The judge's own per-call provenance stamp beats the pipeline fallback
        (a cascade step's model is what actually ran for that row)."""
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(
            _judgement(provenance={"model": "gpt-4o-mini"}),
            verdict=True,
            model="pipeline-model",
        )
        row = log.read()[0]
        assert row["model"] == "gpt-4o-mini"

    def test_append_missing_cost_defaults_to_zero(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(provenance={}), verdict=True)
        row = log.read()[0]
        assert row["cost_usd"] == 0.0

    def test_append_timestamp_is_iso_parseable(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(), verdict=True)
        row = log.read()[0]
        datetime.fromisoformat(row["timestamp"])  # raises if malformed

    def test_append_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "nested" / "run" / "log.jsonl")
        log.append(_judgement(), verdict=True)
        assert len(log.read()) == 1

    def test_append_default_excludes_reasoning_and_provenance(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(
            _judgement(reasoning="looks like a match", provenance={"similarities": {"name": 1.0}}),
            verdict=True,
        )
        row = log.read()[0]
        assert "reasoning" not in row
        assert "provenance" not in row

    def test_features_true_includes_reasoning_and_provenance(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl", features=True)
        log.append(
            _judgement(reasoning="looks like a match", provenance={"similarities": {"name": 1.0}}),
            verdict=True,
        )
        row = log.read()[0]
        assert row["reasoning"] == "looks like a match"
        assert row["provenance"] == {"similarities": {"name": 1.0}}


# ---------------------------------------------------------------------------
# JudgementLog run correlation (S5): nullable run_id = active attempt id
# ---------------------------------------------------------------------------


def _run_context() -> RunContext:
    """A minimal, git/dataset-free RunContext (no subprocess, no files)."""
    return RunContext(experiment="s5-judgement-log", dataset_name="fake")


class TestJudgementLogRunCorrelation:
    def test_append_stamps_run_id_from_active_capture_run(self, tmp_path: Path) -> None:
        """Inside a capture_run, the row's run_id equals that run's attempt_id --
        the exact three-way join (RunRecord.attempt_id == JudgementLog.run_id)."""
        log = JudgementLog(tmp_path / "log.jsonl")
        with capture_run(_run_context(), store=None) as handle:
            log.append(_judgement(), verdict=True)
        row = log.read()[0]
        assert row["run_id"] == handle.attempt_id

    def test_append_run_id_is_none_outside_capture_run(self, tmp_path: Path) -> None:
        """Outside any capture_run, run_id is null -- the field is always present."""
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(), verdict=True)
        row = log.read()[0]
        assert row["run_id"] is None

    def test_run_id_resets_when_capture_run_exits(self, tmp_path: Path) -> None:
        """A row logged after the run closes carries run_id=None again."""
        log = JudgementLog(tmp_path / "log.jsonl")
        with capture_run(_run_context(), store=None):
            log.append(_judgement("in", "run"), verdict=True)
        log.append(_judgement("after", "run"), verdict=True)
        rows = log.read()
        assert rows[0]["run_id"] is not None
        assert rows[1]["run_id"] is None

    def test_pre_s5_rows_without_run_id_still_parse(self, tmp_path: Path) -> None:
        """Additive under ``"v": 1``: an old row (no run_id key) still reads back;
        a reader simply sees the key absent (``.get`` -> None)."""
        path = tmp_path / "log.jsonl"
        path.write_text('{"v": 1, "left_id": "a", "right_id": "b", "score": 0.9}\n')
        rows = JudgementLog(path).read()
        assert len(rows) == 1
        assert rows[0].get("run_id") is None


# ---------------------------------------------------------------------------
# JudgementLog.read (round-trip)
# ---------------------------------------------------------------------------


class TestJudgementLogRead:
    def test_read_returns_empty_list_when_file_does_not_exist(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "missing.jsonl")
        assert log.read() == []

    def test_read_round_trips_every_appended_row_in_order(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement("1", "2", 0.1), verdict=False)
        log.append(_judgement("3", "4", 0.9), verdict=True)
        log.append(_judgement("5", "6", 0.5), verdict=False)
        rows = log.read()
        assert [r["left_id"] for r in rows] == ["1", "3", "5"]
        assert [r["score"] for r in rows] == pytest.approx([0.1, 0.9, 0.5])

    def test_read_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        path.write_text('{"v": 1, "left_id": "a"}\n\n{"v": 1, "left_id": "b"}\n')
        log = JudgementLog(path)
        rows = log.read()
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# JudgementLog.read decision-contract backfill (v1/v2 -> v3)
# ---------------------------------------------------------------------------


class TestJudgementLogDecisionBackfill:
    """Additive v2 -> v3: a pre-decision-contract row (no ``decision`` column)
    reads back with ``decision`` backfilled from its ``verdict`` and
    ``confidence``/``confidence_source`` defaulted -- asserting the exact
    backfilled VALUES, not merely that it parses."""

    def test_v2_row_with_true_verdict_backfills_decision_true(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        path.write_text(
            '{"v": 2, "left_id": "a", "right_id": "b", "verdict": true, "score": 1.0}\n',
            encoding="utf-8",
        )
        row = JudgementLog(path).read()[0]
        assert row["decision"] is True
        assert row["confidence"] is None
        assert row["confidence_source"] == "none"

    def test_v2_row_with_false_verdict_backfills_decision_false(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        path.write_text(
            '{"v": 2, "left_id": "a", "right_id": "b", "verdict": false, "score": 0.1}\n',
            encoding="utf-8",
        )
        row = JudgementLog(path).read()[0]
        assert row["decision"] is False

    def test_v1_row_without_verdict_backfills_decision_none(self, tmp_path: Path) -> None:
        """No ``verdict`` to backfill from -> ``decision`` is an honest ``None``
        (an abstain), never a coerced ``False``."""
        path = tmp_path / "log.jsonl"
        path.write_text(
            '{"v": 1, "left_id": "a", "right_id": "b", "score": 0.9}\n', encoding="utf-8"
        )
        row = JudgementLog(path).read()[0]
        assert row["decision"] is None
        assert row["confidence"] is None
        assert row["confidence_source"] == "none"

    def test_v3_row_carries_decision_directly_untouched(self, tmp_path: Path) -> None:
        """A v3 row is trusted as written -- its ``decision`` is NOT re-derived
        from ``verdict`` (here decision=False while verdict=True)."""
        path = tmp_path / "log.jsonl"
        path.write_text(
            '{"v": 3, "left_id": "a", "right_id": "b", "decision": false, '
            '"verdict": true, "score": null, "confidence": 0.4, '
            '"confidence_source": "logprob"}\n',
            encoding="utf-8",
        )
        row = JudgementLog(path).read()[0]
        assert row["decision"] is False  # trusted, not overwritten by verdict
        assert row["confidence"] == pytest.approx(0.4)
        assert row["confidence_source"] == "logprob"

    def test_appended_v3_row_round_trips_decision(self, tmp_path: Path) -> None:
        """End-to-end: a decision-only judgement appended and read back keeps its
        ``decision`` (the v>=3 pass-through branch)."""
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(score=None, decision=False), verdict=False)
        row = log.read()[0]
        assert row["decision"] is False


# ---------------------------------------------------------------------------
# LoggingMatcher
# ---------------------------------------------------------------------------


class TestLoggingModule:
    def test_forward_passes_judgements_through_unchanged(self, tmp_path: Path) -> None:
        judgements = [_judgement("a", "b", 0.9), _judgement("c", "d", 0.1)]
        inner = _CountingModule(judgements)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingMatcher(inner, log=log, threshold=0.5)

        result = list(wrapped.forward(iter([])))

        assert result == judgements

    def test_forward_logs_every_judgement_that_streams_past(self, tmp_path: Path) -> None:
        judgements = [_judgement("a", "b", 0.9), _judgement("c", "d", 0.1)]
        inner = _CountingModule(judgements)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingMatcher(inner, log=log, threshold=0.5)

        list(wrapped.forward(iter([])))

        rows = log.read()
        assert len(rows) == 2
        assert [r["left_id"] for r in rows] == ["a", "c"]

    def test_forward_computes_verdict_from_threshold(self, tmp_path: Path) -> None:
        judgements = [_judgement("a", "b", 0.9), _judgement("c", "d", 0.1)]
        inner = _CountingModule(judgements)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingMatcher(inner, log=log, threshold=0.5)

        list(wrapped.forward(iter([])))

        rows = log.read()
        assert rows[0]["verdict"] is True  # 0.9 >= 0.5
        assert rows[1]["verdict"] is False  # 0.1 < 0.5

    def test_forward_is_lazy_does_not_materialize_the_full_stream(self, tmp_path: Path) -> None:
        judgements = [_judgement(str(i), str(i + 1), 0.9) for i in range(5)]
        inner = _CountingModule(judgements)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingMatcher(inner, log=log, threshold=0.5)

        gen = wrapped.forward(iter([]))
        next(gen)  # pull exactly one judgement

        assert inner.pulled == 1
        assert len(log.read()) == 1

    def test_inspect_scores_delegates_to_the_wrapped_module(self, tmp_path: Path) -> None:
        class _Reporting(Matcher[object]):
            def forward(
                self, candidates: Iterator[ERCandidate[object]]
            ) -> Iterator[PairwiseJudgement]:
                return iter([])

            def inspect_scores(
                self, judgements: list[PairwiseJudgement], sample_size: int = 10
            ) -> ScoreInspectionReport:
                return ScoreInspectionReport(
                    total_judgements=len(judgements),
                    score_distribution={},
                    high_scoring_examples=[],
                    low_scoring_examples=[],
                    recommendations=[],
                )

        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingMatcher(_Reporting(), log=log, threshold=0.5)

        report = wrapped.inspect_scores([_judgement()], sample_size=3)

        assert report.total_judgements == 1


class TestLoggingModuleBudgetExceeded:
    """Regression (codex review, PR #62): a paid judgement that trips a
    ``_SpendCappedMatcher``'s budget is recorded on ``BudgetExceeded.
    partial_judgements`` but never yielded (the cap raises before yielding
    it) -- so a ``LoggingMatcher`` wrapping the cap must not silently drop it
    from the log. Every judgement that was actually produced, including the
    tripping one, must appear in the JSONL exactly once."""

    def test_tripping_judgement_is_logged_before_reraising(self, tmp_path: Path) -> None:
        yielded = [_judgement("a", "b", 0.9)]
        tripping = _judgement("c", "d", 0.95)
        inner = _BudgetBreachingModule(yielded, tripping)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingMatcher(inner, log=log, threshold=0.5)

        with pytest.raises(BudgetExceeded):
            list(wrapped.forward(iter([])))

        rows = log.read()
        assert [(r["left_id"], r["right_id"]) for r in rows] == [("a", "b"), ("c", "d")]

    def test_the_exception_still_carries_partial_judgements_unmodified(
        self, tmp_path: Path
    ) -> None:
        yielded = [_judgement("a", "b", 0.9)]
        tripping = _judgement("c", "d", 0.95)
        inner = _BudgetBreachingModule(yielded, tripping)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingMatcher(inner, log=log, threshold=0.5)

        with pytest.raises(BudgetExceeded) as excinfo:
            list(wrapped.forward(iter([])))

        assert excinfo.value.partial_judgements == [yielded[0], tripping]

    def test_no_duplicate_rows_when_nothing_was_yielded_before_the_breach(
        self, tmp_path: Path
    ) -> None:
        tripping = _judgement("c", "d", 0.95)
        inner = _BudgetBreachingModule([], tripping)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingMatcher(inner, log=log, threshold=0.5)

        with pytest.raises(BudgetExceeded):
            list(wrapped.forward(iter([])))

        rows = log.read()
        assert len(rows) == 1
        assert (rows[0]["left_id"], rows[0]["right_id"]) == ("c", "d")
