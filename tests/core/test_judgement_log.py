"""Tests for langres.core.judgement_log (JudgementLog / LoggingModule).

The opt-in signal-log seam (W0.2): a JSONL sink (``JudgementLog``) plus a
boundary-component ``Module`` wrapper (``LoggingModule``) that logs each
``PairwiseJudgement`` as it streams past, without breaking generator
laziness. Zero-spend throughout -- everything here uses plain fake Modules,
never a real judge.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from langres.core.judgement_log import JudgementLog, LoggingModule
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.reports import ScoreInspectionReport


def _judgement(
    left_id: str = "a",
    right_id: str = "b",
    score: float = 0.9,
    *,
    reasoning: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id=left_id,
        right_id=right_id,
        score=score,
        score_type="heuristic",
        decision_step="test_step",
        reasoning=reasoning,
        provenance=provenance if provenance is not None else {},
    )


class _CountingModule(Module[object]):
    """Yields ``judgements`` one at a time, tracking how many were pulled.

    Used to prove ``LoggingModule.forward`` is lazy -- it must not pull more
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

    def test_append_includes_schema_version_v1(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        log.append(_judgement(), verdict=True)
        row = log.read()[0]
        assert row["v"] == 1

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
# LoggingModule
# ---------------------------------------------------------------------------


class TestLoggingModule:
    def test_forward_passes_judgements_through_unchanged(self, tmp_path: Path) -> None:
        judgements = [_judgement("a", "b", 0.9), _judgement("c", "d", 0.1)]
        inner = _CountingModule(judgements)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingModule(inner, log=log, threshold=0.5)

        result = list(wrapped.forward(iter([])))

        assert result == judgements

    def test_forward_logs_every_judgement_that_streams_past(self, tmp_path: Path) -> None:
        judgements = [_judgement("a", "b", 0.9), _judgement("c", "d", 0.1)]
        inner = _CountingModule(judgements)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingModule(inner, log=log, threshold=0.5)

        list(wrapped.forward(iter([])))

        rows = log.read()
        assert len(rows) == 2
        assert [r["left_id"] for r in rows] == ["a", "c"]

    def test_forward_computes_verdict_from_threshold(self, tmp_path: Path) -> None:
        judgements = [_judgement("a", "b", 0.9), _judgement("c", "d", 0.1)]
        inner = _CountingModule(judgements)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingModule(inner, log=log, threshold=0.5)

        list(wrapped.forward(iter([])))

        rows = log.read()
        assert rows[0]["verdict"] is True  # 0.9 >= 0.5
        assert rows[1]["verdict"] is False  # 0.1 < 0.5

    def test_forward_is_lazy_does_not_materialize_the_full_stream(self, tmp_path: Path) -> None:
        judgements = [_judgement(str(i), str(i + 1), 0.9) for i in range(5)]
        inner = _CountingModule(judgements)
        log = JudgementLog(tmp_path / "log.jsonl")
        wrapped = LoggingModule(inner, log=log, threshold=0.5)

        gen = wrapped.forward(iter([]))
        next(gen)  # pull exactly one judgement

        assert inner.pulled == 1
        assert len(log.read()) == 1

    def test_inspect_scores_delegates_to_the_wrapped_module(self, tmp_path: Path) -> None:
        class _Reporting(Module[object]):
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
        wrapped = LoggingModule(_Reporting(), log=log, threshold=0.5)

        report = wrapped.inspect_scores([_judgement()], sample_size=3)

        assert report.total_judgements == 1
