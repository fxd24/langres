"""$0 verification of the two flywheel paid-validation scripts (T8).

Every test here runs at **$0** with the deterministic ``SimulatedFrontierJudge``
teacher -- no key, no network, no ``dspy``/``[semantic]`` extra, no real model
call ever. They prove the three things the paid runs' hard-cap guarantee rests on:

1. **Both scripts run the whole closed loop end to end and produce a full report.**
   ``run_fz_smoke`` (committed Fodors-Zagat fixture) and ``run_ag_economics``
   (a bounded Amazon-Google subset, read straight from the vendored CSVs) each
   complete bootstrap -> review -> harvest -> train -> cascade and return a
   well-formed :class:`~examples.flywheel_closed_loop.ClosedLoopReport`, at $0.

2. **The outer SpendMonitor cap FIRES and carries partials.** With a tiny budget
   and the simulated teacher's nonzero *fictional* per-call cost, each run raises
   :class:`~langres.clients.openrouter.BudgetExceeded` carrying the judgements
   already produced on ``.partial_judgements``. This is the structural proof the
   cap protects the orchestrator's real run: the loop cannot cross the budget.

3. **The real path refuses to spend when the run is unsafe.**
   ``preflight_real_model`` rejects a budget above the ceiling and a missing
   ``OPENROUTER_API_KEY`` with a clean reason string *before* any env load or
   network probe -- so a misconfigured paid run fails closed at $0, never mid-call.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

import pytest

# The student is an RFJudge -- skip cleanly where scikit-learn (the [trained]
# extra) is absent rather than fail collection. No dspy/semantic needed: both
# scripts import the real teacher lazily and read AG straight from the CSVs.
pytest.importorskip("sklearn")

from examples.flywheel_closed_loop import ClosedLoopReport, run_closed_loop  # noqa: E402
from examples.research.flywheel_amazon_google import (  # noqa: E402
    materialize_ag_fixtures,
    run_ag_economics,
)
from examples.research.flywheel_fz_smoke import run_fz_smoke  # noqa: E402
from langres.clients.openrouter import BudgetExceeded  # noqa: E402

#: Small AG subset keeps the RFJudge fit + fixture materialization quick.
_AG_MAX_PAIRS = 300


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence the expected circularity/uncompiled warnings the loop narrates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _assert_well_formed(report: ClosedLoopReport) -> None:
    """Every metric a result doc reports is present and in range."""
    assert report.n_heldout > 0
    assert report.n_candidates > 0
    for metric in (report.teacher, report.student, report.cascade):
        assert 0.0 <= metric.f1 <= 1.0
        assert 0.0 <= metric.precision <= 1.0
        assert 0.0 <= metric.recall <= 1.0
    assert 0.0 <= report.escalation_rate <= 1.0
    assert 0.0 <= report.frontier_call_reduction <= 1.0
    # The simulated teacher stamps a fictional per-call cost, so the outer cap
    # metered a nonzero total without tripping the (generous) default budget.
    assert report.teacher_spend_usd > 0.0


# ---------------------------------------------------------------------------
# (1) both scripts run end to end at $0
# ---------------------------------------------------------------------------


def test_fz_smoke_runs_end_to_end_simulated() -> None:
    """The FZ wiring smoke completes the whole loop at $0 and reports every metric."""
    with _quiet():
        report = run_fz_smoke(simulated=True)
    _assert_well_formed(report)


def test_ag_economics_runs_end_to_end_simulated() -> None:
    """The AG economics run completes the whole loop at $0 on a bounded subset."""
    with _quiet():
        report = run_ag_economics(simulated=True, max_pairs=_AG_MAX_PAIRS)
    _assert_well_formed(report)
    # The bound is honored (materialization capped the candidate set).
    assert report.n_candidates <= _AG_MAX_PAIRS


def test_ag_fixtures_span_both_label_classes(tmp_path: Path) -> None:
    """Materialized AG fixtures carry both label classes (else RFJudge.fit crashes)."""
    import json

    out = materialize_ag_fixtures(tmp_path / "ag", max_pairs=_AG_MAX_PAIRS, seed=7)
    payload = json.loads((out / "gold_pairs.json").read_text())
    labels = {pair["label"] for pair in payload["candidate_pairs"]}
    assert labels == {True, False}


def test_simulated_runs_are_deterministic() -> None:
    """Same seed -> identical cascade quality (seeded throughout, zero network)."""
    with _quiet():
        a = run_fz_smoke(simulated=True)
        b = run_fz_smoke(simulated=True)
    assert a.cascade.f1 == b.cascade.f1
    assert a.escalation_rate == b.escalation_rate


# ---------------------------------------------------------------------------
# (2) the outer spend cap fires and carries partials
# ---------------------------------------------------------------------------


def test_fz_cap_trips_when_simulated_cost_exceeds_budget() -> None:
    """A budget below the first fictional call cost -> BudgetExceeded with partials."""
    with _quiet(), pytest.raises(BudgetExceeded) as excinfo:
        run_fz_smoke(simulated=True, budget_usd=0.001)
    partials = excinfo.value.partial_judgements
    assert partials, "the cap must carry the judgements the (fictional) money already bought"


def test_ag_cap_trips_when_simulated_cost_exceeds_budget() -> None:
    """The AG run is likewise structurally bounded by the outer cap."""
    with _quiet(), pytest.raises(BudgetExceeded) as excinfo:
        run_ag_economics(simulated=True, max_pairs=_AG_MAX_PAIRS, budget_usd=0.001)
    assert excinfo.value.partial_judgements


def test_run_closed_loop_cap_is_the_shared_mechanism() -> None:
    """The cap lives on run_closed_loop itself; the default (no cap) is unchanged."""
    with _quiet(), pytest.raises(BudgetExceeded) as excinfo:
        run_closed_loop(seed=0, spend_cap_usd=0.001)
    assert excinfo.value.partial_judgements
    # No cap -> no metered spend, so the default simulated behavior is untouched.
    with _quiet():
        uncapped = run_closed_loop(seed=0)
    assert uncapped.teacher_spend_usd == 0.0


# ---------------------------------------------------------------------------
# (3) the real path refuses to spend when the run is unsafe (no network touched)
# ---------------------------------------------------------------------------


def test_preflight_refuses_budget_over_ceiling() -> None:
    """A budget above the ceiling is refused before any env/network touch."""
    from examples.research._flywheel_paid_common import preflight_real_model

    reason = preflight_real_model("openrouter/openai/gpt-4o-mini", budget_usd=99.0, ceiling_usd=2.0)
    assert reason is not None
    assert "ceiling" in reason


def test_preflight_refuses_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No OPENROUTER_API_KEY -> a clean refusal string, and NO network probe.

    ``load_dotenv`` is stubbed so a developer's real ``.env`` can't repopulate the
    key (which would then fire the real price-probe call). This proves the scripts
    fail cleanly without a key at $0.
    """
    import dotenv

    from examples.research._flywheel_paid_common import preflight_real_model

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reason = preflight_real_model("openrouter/openai/gpt-4o-mini", budget_usd=1.0, ceiling_usd=2.0)
    assert reason is not None
    assert "OPENROUTER_API_KEY" in reason
