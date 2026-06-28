"""Tests for the bootstrap coverage/calibration report.

The builder is pure and deterministic, so every assertion uses synthetic
:class:`GoldPair` data with a known ground truth -- no LLM, no embeddings, no
network. Expected values are hand-computed. Targets 100% coverage of
``langres.bootstrap.report``.
"""

from typing import Any

import pytest
from pydantic import BaseModel

from langres.bootstrap.models import GoldPair, GoldSet
from langres.bootstrap.report import BootstrapReport
from langres.core.models import ERCandidate


class _Rec(BaseModel):
    """Minimal entity schema (id only) for building candidate pairs."""

    id: str


def _cand(left: str, right: str) -> ERCandidate[Any]:
    return ERCandidate(left=_Rec(id=left), right=_Rec(id=right), blocker_name="test")


def _teacher(
    left: str,
    right: str,
    label: bool,
    *,
    confidence: float | None = None,
    provenance: dict[str, object] | None = None,
) -> GoldPair:
    return GoldPair(
        left_id=left,
        right_id=right,
        label=label,
        source="teacher",
        confidence=confidence,
        provenance=provenance or {},
    )


# Shared scenario:
#   truth clusters: {a,b} and {c,d} -> match pairs (a,b) and (c,d)
#   candidates capture a-b and a-c (miss c-d) -> pair-completeness 0.5
#   teacher labels: a-b True (correct), a-c False (correct), c-d False (WRONG),
#                   x-y True (no ground truth -> excluded)
_TRUTH = [{"a", "b"}, {"c", "d"}]


def _scenario_gold() -> GoldSet:
    return GoldSet(
        pairs=[
            _teacher("a", "b", True, confidence=0.9),
            _teacher("a", "c", False, confidence=0.8),
            _teacher("c", "d", False, confidence=0.4),
            _teacher("x", "y", True, confidence=0.5),
        ],
        metadata={"total_cost_usd": 1.25, "mined": 4},
    )


def _scenario_candidates() -> list[ERCandidate[Any]]:
    return [_cand("a", "b"), _cand("a", "c")]


def test_blocking_pair_completeness() -> None:
    report = BootstrapReport.build(_scenario_gold(), _scenario_candidates(), _TRUTH)
    # (a,b) captured, (c,d) missed -> recall 0.5; candidate precision 1/2.
    assert report.blocking.pair_completeness == 0.5
    assert report.blocking.candidate_precision == 0.5
    assert report.blocking.total_candidates == 2
    assert report.blocking.missed_matches == 1


def test_agreement_hand_computed() -> None:
    report = BootstrapReport.build(_scenario_gold(), _scenario_candidates(), _TRUTH)
    a = report.agreement
    assert a is not None
    assert a.n_evaluated == 3  # x-y excluded (no ground truth)
    assert a.accuracy == pytest.approx(2 / 3)
    assert a.precision == 1.0  # TP=1, FP=0
    assert a.recall == 0.5  # TP=1, FN=1
    assert a.f1 == pytest.approx(2 / 3)
    assert a.cohens_kappa == pytest.approx(0.4)
    assert a.mcc == 0.5


def test_calibration_hand_computed() -> None:
    report = BootstrapReport.build(_scenario_gold(), _scenario_candidates(), _TRUTH, n_bins=2)
    c = report.calibration
    assert c is not None
    # confidences [0.9,0.8,0.4], correctness [True,True,False]
    # Brier: (0.01 + 0.04 + 0.16)/3 = 0.07
    assert c.n_evaluated == 3
    assert c.brier == pytest.approx(0.07)
    # quantile 2 bins: [0.4,0.8] acc .5 conf .6; [0.9] acc 1 conf .9 -> ECE 0.1
    assert c.ece == pytest.approx(0.1)
    assert c.n_bins == 2
    assert len(c.reliability) == 2
    assert c.reliability[0].mean_confidence == pytest.approx(0.6)
    assert c.reliability[0].observed_frequency == 0.5
    assert c.reliability[0].count == 2
    assert c.reliability[1].mean_confidence == pytest.approx(0.9)
    assert c.reliability[1].observed_frequency == 1.0
    assert c.reliability[1].count == 1


def test_convergence_curve_deterministic() -> None:
    report = BootstrapReport.build(_scenario_gold(), _scenario_candidates(), _TRUTH)
    # Order by (left,right): a-b, a-c, c-d.
    # n=1 F1=1.0 ; n=2 (a-c TN) F1=1.0 ; n=3 (c-d FN) precision1 recall .5 -> 0.667
    assert [(p.n_labeled, round(p.f1, 4)) for p in report.convergence] == [
        (1, 1.0),
        (2, 1.0),
        (3, round(2 / 3, 4)),
    ]


def test_convergence_includes_false_positive_branch() -> None:
    # Teacher says match where truth says non-match (a-c True but truth non-match).
    gold = GoldSet(
        pairs=[_teacher("a", "c", True, confidence=0.7)],
        metadata={},
    )
    report = BootstrapReport.build(gold, [_cand("a", "b")], _TRUTH)
    a = report.agreement
    assert a is not None
    # FP=1, TP=0 -> precision 0, recall 0 (no truth positives among evaluated)
    assert a.precision == 0.0
    assert a.recall == 0.0
    assert a.f1 == 0.0
    assert report.convergence[0].f1 == 0.0


def test_coverage_and_cost_from_metadata() -> None:
    report = BootstrapReport.build(_scenario_gold(), _scenario_candidates(), _TRUTH)
    cov = report.coverage
    assert cov.total_candidates == 2
    assert cov.mined == 4  # from metadata
    assert cov.labeled == 4
    assert cov.skipped == 0  # total_candidates(2) - labeled(4) floored at 0
    assert cov.with_ground_truth == 3
    assert cov.total_cost_usd == 1.25


def test_build_from_plain_list_uses_defaults() -> None:
    pairs = [_teacher("a", "b", True, confidence=0.9)]
    report = BootstrapReport.build(pairs, [_cand("a", "b"), _cand("e", "f")], _TRUTH)
    cov = report.coverage
    assert cov.labeled == 1
    assert cov.mined == 1  # fallback to labeled when no metadata
    assert cov.skipped == 1  # 2 candidates - 1 labeled
    assert cov.total_cost_usd == 0.0  # no metadata, no provenance cost


def test_cost_summed_from_provenance() -> None:
    pairs = [
        _teacher("a", "b", True, provenance={"cost_usd": 0.10}),
        _teacher("c", "d", True, provenance={"cost": 0.05}),
        _teacher("a", "c", False, provenance={"note": "no cost here"}),
    ]
    report = BootstrapReport.build(pairs, [_cand("a", "b")], _TRUTH)
    assert report.coverage.total_cost_usd == pytest.approx(0.15)


def test_cost_metadata_bool_is_ignored_falls_back_to_provenance() -> None:
    # A bool in metadata must not be read as a cost (isinstance(True, int) is True).
    gold = GoldSet(
        pairs=[_teacher("a", "b", True, provenance={"cost": 0.2})],
        metadata={"total_cost_usd": True},
    )
    report = BootstrapReport.build(gold, [_cand("a", "b")], _TRUTH)
    assert report.coverage.total_cost_usd == pytest.approx(0.2)


def test_cost_non_numeric_provenance_ignored() -> None:
    pairs = [_teacher("a", "b", True, provenance={"cost": "free"})]
    report = BootstrapReport.build(pairs, [_cand("a", "b")], _TRUTH)
    assert report.coverage.total_cost_usd == 0.0


def test_no_ground_truth_yields_none_sections() -> None:
    # No labeled pair's ids appear in truth clusters -> agreement/calibration None.
    pairs = [_teacher("x", "y", True, confidence=0.6)]
    report = BootstrapReport.build(pairs, [_cand("x", "y")], _TRUTH)
    assert report.agreement is None
    assert report.calibration is None
    assert report.convergence == []


def test_calibration_none_when_no_confidence_but_truth_present() -> None:
    # Pair has ground truth but no confidence -> agreement present, calibration None.
    pairs = [_teacher("a", "b", True)]
    report = BootstrapReport.build(pairs, [_cand("a", "b")], _TRUTH)
    assert report.agreement is not None
    assert report.calibration is None


def test_to_markdown_contains_key_numbers() -> None:
    report = BootstrapReport.build(_scenario_gold(), _scenario_candidates(), _TRUTH, n_bins=2)
    md = report.to_markdown()
    assert "# Bootstrap Report" in md
    assert "Pair-completeness (candidate recall): 0.5000" in md
    assert "F1: 0.6667" in md
    assert "Cohen's kappa: 0.4000" in md
    assert "MCC: 0.5000" in md
    assert "Brier score (primary): 0.0700" in md
    assert "ECE (equal-mass, 2 bins): 0.1000" in md
    assert "Total cost (USD): 1.2500" in md
    assert "Final F1 @ 3 labels: 0.6667" in md


def test_render_is_markdown_alias() -> None:
    report = BootstrapReport.build(_scenario_gold(), _scenario_candidates(), _TRUTH)
    assert report.render() == report.to_markdown()


def test_to_markdown_handles_empty_sections() -> None:
    pairs = [_teacher("x", "y", True)]
    report = BootstrapReport.build(pairs, [_cand("x", "y")], _TRUTH)
    md = report.to_markdown()
    assert "No labeled pair had a ground-truth label." in md
    assert "No labeled pair had both a confidence and a ground-truth label." in md
    # Empty convergence -> the convergence section is omitted.
    assert "## Agreement convergence" not in md


def test_report_json_roundtrip() -> None:
    report = BootstrapReport.build(_scenario_gold(), _scenario_candidates(), _TRUTH, n_bins=2)
    restored = BootstrapReport.model_validate_json(report.model_dump_json())
    assert restored == report
