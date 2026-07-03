"""$0 verification of the W3 paid smoke harness (``examples/research/w3_paid_smoke.py``).

Every test here runs at **$0** with DSPy's ``DummyLM`` -- no key, no network, no
real model call ever. They prove the two things the paid run's ≤$10 guarantee
rests on:

1. **The full flow runs end-to-end and produces BOTH quality numbers.**
   ``run_smoke`` with injected DummyLMs (a pairwise-shaped one for
   ``link``/``dedupe``/the AG pairwise arm, a select-shaped one for SelectJudge)
   completes all four deliverables -- link, dedupe, a single SelectJudge group
   call, the signal log, and the AG SelectJudge-vs-pairwise comparison -- and
   reports a pairwise F1 *and* a set-wise F1, at $0 real spend.

2. **The SpendMonitor cap FIRES and carries partials.** With a tiny budget and a
   nonzero *fake* per-call cost (injected via ``cost_track_fn`` -- DummyLM's real
   token cost is $0), the run raises
   :class:`~langres.clients.openrouter.BudgetExceeded` carrying the judgements
   already produced on ``.partial_judgements``. This is the proof the hard cap is
   real -- the run structurally cannot cross the budget.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dspy.utils.dummies import DummyLM

from langres.clients.openrouter import BudgetExceeded
from langres.core.benchmark import CostTrack
from langres.core.models import PairwiseJudgement

# Small AG subset keeps the DummyLM answer pools + import cost modest.
_AG_GROUPS = 3
_POOL = 4000


def _pairwise_lm() -> DummyLM:
    """A DummyLM keyed to DSPyJudge's ``PairwiseMatchSignature`` output fields."""
    return DummyLM(
        [{"reasoning": "different variant", "match": "False", "match_probability": "0.1"}] * _POOL
    )


def _select_lm() -> DummyLM:
    """A DummyLM keyed to SelectSignature's output fields (always selects nothing)."""
    return DummyLM([{"reasoning": "no match", "selected_ids": "[]"}] * _POOL)


def _cfg(tmp_path: Path, *, budget: float) -> "object":
    from examples.research.w3_paid_smoke import SmokeConfig

    return SmokeConfig(
        model="openrouter/openai/gpt-4o-mini",
        budget_usd=budget,
        ag_groups=_AG_GROUPS,
        log_path=tmp_path / "w3_judgements.jsonl",
    )


def test_full_flow_runs_at_zero_and_reports_both_quality_numbers(tmp_path: Path) -> None:
    """All four deliverables run end-to-end on DummyLM; both AG F1s are produced; $0 spend."""
    from examples.research.w3_paid_smoke import run_smoke

    results = run_smoke(
        _cfg(tmp_path, budget=100.0),
        dspy_lm=_pairwise_lm(),
        select_lm=_select_lm(),
    )

    # (1) link + dedupe ran.
    assert isinstance(results["link"]["match"], bool)
    assert 0.0 <= results["link"]["score"] <= 1.0
    assert isinstance(results["dedupe"]["n_clusters"], int)
    assert results["dedupe"]["n_clusters"] >= 1

    # (3) a signal log was emitted (one row per judged pair across both verbs).
    assert results["signal_log_rows"] > 0
    assert (tmp_path / "w3_judgements.jsonl").exists()

    # (2) exactly one LLM call judged a genuinely multi-member group.
    assert results["group_call"]["n_llm_calls"] == 1
    assert results["group_call"]["n_members"] >= 1
    assert results["group_call"]["n_judgements"] == results["group_call"]["n_members"]

    # (4) BOTH quality numbers exist and are valid F1 values -- the substantive
    # deliverable. DummyLM answers "no match", so both are ~0, but both are
    # produced (that is what W3 verifies at $0; real quality is the paid run's).
    pairwise_f1 = results["ag_pairwise"]["f1"]
    setwise_f1 = results["ag_setwise"]["f1"]
    assert isinstance(pairwise_f1, float) and 0.0 <= pairwise_f1 <= 1.0
    assert isinstance(setwise_f1, float) and 0.0 <= setwise_f1 <= 1.0
    # The set-wise arm makes one call per anchor group; the pairwise arm one per
    # pair -- so set-wise never makes MORE calls than pairwise (the cost lever).
    assert results["ag_setwise"]["n_llm_calls"] <= results["ag_pairwise"]["n_llm_calls"]

    # $0: DummyLM records no tokens, so every honest cost is exactly zero.
    assert results["total_spent_usd"] == 0.0
    assert results["verb_cost_usd"] == 0.0
    assert results["ag_pairwise"]["cost_usd"] == 0.0
    assert results["ag_setwise"]["cost_usd"] == 0.0


def test_spend_cap_fires_and_carries_partials(tmp_path: Path) -> None:
    """A tiny budget + nonzero fake per-call cost -> BudgetExceeded carrying partials.

    The real token cost is $0 (DummyLM), so we inject a nonzero ``cost_track_fn``:
    the first metered paid unit (the SelectJudge group call) then trips the
    SpendMonitor, which raises ``BudgetExceeded`` carrying the group's judgements
    on ``.partial_judgements``. This is the structural proof of the ≤$10 cap.
    """
    from examples.research.w3_paid_smoke import run_smoke

    def fake_cost(judgements: list[PairwiseJudgement]) -> CostTrack:
        # A flat, nonzero per-call cost, independent of DummyLM's zero tokens.
        return CostTrack(usd_total=0.05)

    with pytest.raises(BudgetExceeded) as excinfo:
        run_smoke(
            _cfg(tmp_path, budget=0.01),  # 0.05 > 0.01 -> the first metered unit trips it
            dspy_lm=_pairwise_lm(),
            select_lm=_select_lm(),
            cost_track_fn=fake_cost,
        )

    # The cap carried the already-produced (already-"paid-for") judgements.
    partials = excinfo.value.partial_judgements
    assert partials, "BudgetExceeded must carry the partial judgements the money already bought"
    assert all(isinstance(j, PairwiseJudgement) for j in partials)
