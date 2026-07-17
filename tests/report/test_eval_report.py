"""Tests for langres.report.eval_report (the $0 EvalReport tearsheet).

Everything here is zero-spend and dependency-light: hand-built judgements or
plain log-row dicts, no judge, no model, no network. The rendering tests assert
by STRUCTURE (svg/section counts, escaping, absence of NaN), never byte equality.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from langres.report.eval_report import EvalError, EvalReport, _histogram, _roc_curve
from langres.core.models import PairwiseJudgement


def _j(
    left: str,
    right: str,
    *,
    score: float | None = None,
    decision: bool | None = None,
    confidence: float | None = None,
    confidence_source: str = "none",
) -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id=left,
        right_id=right,
        score=score,
        decision=decision,
        score_type="prob_llm",
        confidence=confidence,
        confidence_source=confidence_source,  # type: ignore[arg-type]
        decision_step="test",
        provenance={},
    )


_GOLD = {frozenset({"a", "b"}), frozenset({"c", "d"})}


# --------------------------------------------------------------------------- #
# Pure helpers                                                                #
# --------------------------------------------------------------------------- #


def test_histogram_counts_and_closed_right_edge() -> None:
    edges = [0.0, 0.5, 1.0]
    # 1.0 must land in the LAST bin (right edge is closed), not be dropped.
    assert _histogram([0.1, 0.4, 0.6, 1.0], edges) == [2.0, 2.0]


def test_histogram_ignores_non_finite() -> None:
    assert _histogram([0.2, float("nan"), float("inf")], [0.0, 0.5, 1.0]) == [1.0, 0.0]


def test_roc_curve_runs_corner_to_corner() -> None:
    points = _roc_curve([False, False, True, True], [0.1, 0.2, 0.8, 0.9])
    assert points[0] == (0.0, 0.0)
    assert points[-1] == (1.0, 1.0)


def test_roc_curve_single_class_is_empty() -> None:
    assert _roc_curve([True, True], [0.1, 0.9]) == []


# --------------------------------------------------------------------------- #
# from_judgements — the arithmetic                                            #
# --------------------------------------------------------------------------- #


def test_confusion_matrix_at_threshold() -> None:
    """ab=TP (gold, high), ef=FP (non-gold, high), cd=FN (gold, low),
    gh=TN (non-gold, low)."""
    judgements = [
        _j("a", "b", score=0.9),
        _j("c", "d", score=0.2),
        _j("e", "f", score=0.8),
        _j("g", "h", score=0.1),
    ]
    report = EvalReport.from_judgements(judgements, _GOLD, threshold=0.5)
    assert (report.tp, report.fp, report.fn, report.tn) == (1, 1, 1, 1)
    assert report.n_candidates == 4
    assert report.n_ranked == 4
    assert report.n_gold == 2


def test_roc_auc_and_average_precision_match_the_primitives() -> None:
    judgements = [
        _j("a", "b", score=0.9),
        _j("c", "d", score=0.2),
        _j("e", "f", score=0.8),
        _j("g", "h", score=0.1),
    ]
    report = EvalReport.from_judgements(judgements, _GOLD, threshold=0.5)
    # gold={ab,cd}: scores [0.9,0.2] positive, [0.8,0.1] negative -> AUC 0.75.
    assert report.roc_auc == pytest.approx(0.75)
    assert 0.0 <= report.average_precision <= 1.0
    assert report.roc_curve[0] == (0.0, 0.0)


def test_abstention_counts_and_is_excluded_from_tn() -> None:
    """An abstaining judge (no decision, no score) on a non-gold pair is NOT a
    correct negative -- it made no prediction. It counts in n_abstained and is
    absent from tn and from the error list."""
    judgements = [
        _j("a", "b", score=0.9),  # TP
        _j("g", "h"),  # abstain on a non-gold pair
    ]
    report = EvalReport.from_judgements(judgements, _GOLD, threshold=0.5)
    assert report.n_abstained == 1
    assert report.tn == 0  # the abstention is NOT counted as a correct negative
    assert all(not (e.left_id == "g" and e.right_id == "h") for e in report.top_errors)


def test_decider_only_log_has_no_ranking_panels_and_no_nan_in_html() -> None:
    """A pure decider (decision set, score None) has no continuous signal: ROC/AP
    are nan, the ROC curve is empty, yet to_html() renders cleanly with NO literal
    NaN/Infinity (the silent 'empty chart photographs fine' failure mode)."""
    judgements = [
        _j("a", "b", decision=True),
        _j("c", "d", decision=False),
        _j("e", "f", decision=True),
    ]
    report = EvalReport.from_judgements(judgements, _GOLD, threshold=0.5)
    assert report.n_ranked == 0
    assert report.roc_curve == []
    assert math.isnan(report.roc_auc)
    assert math.isnan(report.average_precision)
    out = report.to_html()
    assert "NaN" not in out and "Infinity" not in out
    assert "n/a" in out  # the undefined AUC renders as n/a, not a broken number


def test_confidence_drives_calibration_panel() -> None:
    """Judgements carrying a confidence populate reliability/Brier/ECE; the
    outcome graded is whether the judge's OWN prediction was correct."""
    judgements = [
        _j("a", "b", decision=True, confidence=0.9, confidence_source="logprob"),  # correct
        _j("c", "d", decision=False, confidence=0.6, confidence_source="logprob"),  # wrong (gold)
        _j(
            "e", "f", decision=True, confidence=0.55, confidence_source="logprob"
        ),  # wrong (non-gold)
    ]
    report = EvalReport.from_judgements(judgements, _GOLD, threshold=0.5)
    assert report.n_with_confidence == 3
    assert report.brier is not None and 0.0 <= report.brier <= 1.0
    assert report.ece is not None
    assert len(report.reliability) >= 1
    assert report.confidence_source_counts["logprob"] == 3


def test_no_confidence_leaves_calibration_none() -> None:
    report = EvalReport.from_judgements([_j("a", "b", score=0.9)], _GOLD, threshold=0.5)
    assert report.brier is None
    assert report.ece is None
    assert report.reliability == []
    assert report.n_with_confidence == 0


def test_top_errors_are_ranked_most_confident_first() -> None:
    """A blatant false positive (high score, non-gold) outranks a marginal one."""
    judgements = [
        _j("a", "b", score=0.99),  # non-gold, very confident FP
        _j("c", "d", score=0.55),  # non-gold, marginal FP
    ]
    gold: set[frozenset[str]] = set()  # nothing is gold -> both are FPs
    report = EvalReport.from_judgements(judgements, gold, threshold=0.5, top_n=10)
    assert [(e.left_id, e.right_id) for e in report.top_errors] == [("a", "b"), ("c", "d")]
    assert report.top_errors[0].predicted is True
    assert report.top_errors[0].is_gold is False


def test_single_class_gold_gives_nan_auc_not_a_raise() -> None:
    """Every candidate is gold -> AUC undefined. We return nan, never raise (so an
    all-positive slice degrades to a blank cell, not a crashed report)."""
    judgements = [_j("a", "b", score=0.9), _j("c", "d", score=0.8)]
    report = EvalReport.from_judgements(judgements, _GOLD, threshold=0.5)
    assert math.isnan(report.roc_auc)
    # AP with no negatives is defined as 1.0 (any ordering is trivially perfect).
    assert report.average_precision == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# from_log — the persisted-rows path                                          #
# --------------------------------------------------------------------------- #


def _row(left: str, right: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "v": 3,
        "left_id": left,
        "right_id": right,
        "score": None,
        "verdict": None,
        "decision": None,
        "cost_usd": 0.0,
    }
    row.update(overrides)
    return row


def test_from_log_reconstructs_and_sums_cost() -> None:
    rows = [
        _row("a", "b", score=0.9, decision=True, verdict=True, cost_usd=0.001),
        _row("e", "f", score=0.8, decision=True, verdict=True, cost_usd=0.002),
    ]
    report = EvalReport.from_log(rows, _GOLD, threshold=0.5)
    assert report.n_candidates == 2
    assert report.total_cost_usd == pytest.approx(0.003)


def test_from_log_falls_back_to_legacy_llm_cost_usd_key() -> None:
    """`from_log` accepts arbitrary row dicts, not only current `read()` output.

    Current `JudgementLog.append()` normalises cost to a single top-level
    `cost_usd`, so `read()` rows never carry a top-level `llm_cost_usd`. But a
    hand-built row (or a pre-normalisation log) may put cost under the legacy
    `llm_cost_usd` key; the sum must still see it, mirroring `_COST_KEYS`.
    """
    rows = [_row("a", "b", score=0.9, decision=True, verdict=True, cost_usd=0.0, llm_cost_usd=0.05)]
    report = EvalReport.from_log(rows, _GOLD, threshold=0.5)
    assert report.total_cost_usd == pytest.approx(0.05)


def test_from_log_abstention_row_stays_abstention() -> None:
    """A v3 abstention row (decision present and None) must NOT be coerced to a
    verdict -- it stays an abstention in the report's accounting."""
    rows = [_row("g", "h", score=None, decision=None, verdict=None)]
    report = EvalReport.from_log(rows, _GOLD, threshold=0.5)
    assert report.n_abstained == 1


# --------------------------------------------------------------------------- #
# Rendering — structure, escaping, no NaN                                     #
# --------------------------------------------------------------------------- #


def _full_report() -> EvalReport:
    judgements = [
        _j("a", "b", score=0.9, confidence=0.8, confidence_source="logprob"),
        _j("c", "d", score=0.2, confidence=0.4, confidence_source="logprob"),
        _j("e", "f", score=0.8, confidence=0.7, confidence_source="logprob"),
        _j("g", "h", score=0.1, confidence=0.9, confidence_source="logprob"),
    ]
    return EvalReport.from_judgements(judgements, _GOLD, threshold=0.5)


def test_to_html_is_self_contained_and_has_a_chart_per_panel() -> None:
    out = _full_report().to_html()
    assert out.startswith("<!doctype html>")
    # PR, ROC, histogram, reliability = four inline charts, all confidence present.
    assert out.count("<svg") == 4
    # self-contained: no external asset references.
    assert "http://" not in out.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "https://" not in out
    assert "NaN" not in out and "Infinity" not in out


def test_to_html_escapes_ids_in_the_error_table() -> None:
    judgements = [_j("x&<script>", "b", score=0.99)]  # a hostile id, non-gold -> FP
    report = EvalReport.from_judgements(judgements, set(), threshold=0.5)
    out = report.to_html()
    assert "x&amp;&lt;script&gt;" in out
    assert "<script>" not in out


def test_to_html_reliability_absent_when_no_confidence() -> None:
    report = EvalReport.from_judgements([_j("a", "b", score=0.9)], _GOLD, threshold=0.5)
    out = report.to_html()
    assert "No confidence signal" in out
    # PR, ROC, histogram render; reliability is a note, so three charts.
    assert out.count("<svg") == 3


def test_to_markdown_and_to_dict_and_summary_smoke() -> None:
    report = _full_report()
    md = report.to_markdown()
    assert md.startswith("# Evaluation report")
    assert "Confusion" in md
    assert "NaN" not in md
    d = report.to_dict()
    assert d["threshold"] == 0.5
    assert d["n_candidates"] == 4
    assert isinstance(report.summary, str) and "ROC-AUC" in report.summary


def test_eval_error_is_frozen() -> None:
    err = EvalError(
        left_id="a", right_id="b", predicted=True, is_gold=False, score=0.9, confidence=None
    )
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen -> ValidationError
        err.left_id = "z"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Edge cases and robustness (the core contract tier)                          #
# --------------------------------------------------------------------------- #


def test_empty_judgements_render_cleanly() -> None:
    """No judgements -> every metric is undefined, but nothing raises or leaks NaN."""
    report = EvalReport.from_judgements([], _GOLD, threshold=0.5)

    assert report.n_candidates == 0
    assert report.tp == report.fp == report.tn == 0
    # fn is |gold - predicted|: with nothing predicted, every gold pair is missed.
    assert report.fn == report.n_gold == 2
    assert math.isnan(report.roc_auc)
    assert report.roc_curve == []
    assert report.brier is None and report.ece is None
    out = report.to_html()
    assert "NaN" not in out and "Infinity" not in out


def test_all_scores_equal_gives_chance_auc() -> None:
    """A ranker that scores every pair identically is pure chance: ROC-AUC 0.5."""
    judgements = [
        _j("a", "b", score=0.5),  # gold
        _j("c", "d", score=0.5),  # gold
        _j("e", "f", score=0.5),  # non-gold
        _j("g", "h", score=0.5),  # non-gold
    ]
    report = EvalReport.from_judgements(judgements, _GOLD, threshold=0.5)

    assert report.roc_auc == pytest.approx(0.5)
    assert report.roc_curve[0] == (0.0, 0.0)
    assert report.roc_curve[-1] == (1.0, 1.0)


def test_duplicate_pair_rows_are_collapsed_last_wins_and_not_double_counted() -> None:
    """A log with two rows for one pair counts it once; the later row wins.

    The reviewer's scenario: without dedup a non-gold pair logged both above and
    below threshold is simultaneously an FP (set-based ``classify_pairs``) and a TN
    (the per-judgement walk) -- and it inflates ``n_ranked``/histogram/ROC. Dedup
    (last write wins, as a re-run supersedes) makes the low row the single verdict.
    """
    judgements = [
        _j("e", "f", score=0.9),  # non-gold, high (row 1) -> would be an FP
        _j("e", "f", score=0.3),  # SAME non-gold pair, re-judged low -> supersedes -> TN
        _j("a", "b", score=0.9),  # gold, high -> TP
    ]
    report = EvalReport.from_judgements(judgements, _GOLD, threshold=0.5)

    assert report.n_candidates == 2  # (e,f) collapsed, plus (a,b)
    assert report.n_ranked == 2  # not 3 -- the duplicate score is not double-ranked
    assert report.fp == 0  # last (e,f) row (0.3) wins, so no false positive
    assert report.tn == 1  # (e,f) counted once as a true negative, not also an FP
    assert report.tp == 1  # (a,b)
    # (a,b) is a judged gold that IS predicted, so no judged pair falls into fn:
    # the four cells + abstains then account for exactly the judged pairs.
    assert report.tp + report.fp + report.tn + report.n_abstained == report.n_candidates


def test_misaligned_costs_raises_rather_than_mis_summing() -> None:
    """A costs list of the wrong length is a caller bug, not a silent wrong total."""
    judgements = [_j("a", "b", score=0.9), _j("c", "d", score=0.8)]
    with pytest.raises(ValueError, match="costs must align"):
        EvalReport.from_judgements(judgements, _GOLD, threshold=0.5, costs=[0.01])


def test_to_markdown_escapes_a_pipe_in_a_record_id() -> None:
    """A ``|`` in an id must not break the error-table's column alignment."""
    judgements = [_j("a|b", "c", score=0.99)]  # hostile id, non-gold -> FP
    report = EvalReport.from_judgements(judgements, set(), threshold=0.5)

    md = report.to_markdown()
    assert r"a\|b" in md
    # after removing the escaped pipe, the row has exactly the 6-column borders.
    error_row = next(line for line in md.splitlines() if r"a\|b" in line)
    assert error_row.replace(r"\|", "").count("|") == 7
