"""Exit-criteria test for ``examples/flywheel_closed_loop.py``.

Validates the **plumbing** of the closed loop -- that
bootstrap -> review -> harvest -> train -> cascade runs end to end at $0, escalates
a sane fraction of pairs, and cuts the mixed ``prob_rf`` / ``prob_llm`` stream with a
single threshold **without a scale mismatch** -- NOT the economics. "Cascade beats the
cheap student" is an economic claim reserved for the paid validation run on hard data
(T8): on this easy Fodors-Zagat fixture the student is already near-perfect, so the
cascade's value here is the frontier-call reduction at roughly equal quality, not an F1
gain. The simulated teacher + gold-derived answers make these criteria near-guaranteed
by construction -- that is the point of a plumbing test.
"""

from __future__ import annotations

import warnings

import pytest

# The student is a RandomForestMatcher -- skip cleanly where scikit-learn (the [trained] extra)
# is not installed rather than fail collection.
pytest.importorskip("sklearn")

from examples.flywheel_closed_loop import ClosedLoopReport, run_closed_loop  # noqa: E402


@pytest.fixture(scope="module")
def report() -> ClosedLoopReport:
    """Run the whole $0 loop once (no API calls -- the teacher is simulated)."""
    with warnings.catch_warnings():
        # The BEFORE-arm silver-only circularity warning is expected and asserted
        # via report.circularity_warning_fired; don't let it error the run.
        warnings.simplefilter("ignore")
        return run_closed_loop(seed=0)


def test_escalation_fraction_is_sane(report: ClosedLoopReport) -> None:
    """Some pairs escalate, but the band is not so wide it escalates everything."""
    assert 0.0 < report.escalation_rate <= 0.5


def test_frontier_calls_are_meaningfully_reduced(report: ClosedLoopReport) -> None:
    """The whole point of a cascade: far fewer frontier calls than judging every pair."""
    assert report.frontier_call_reduction >= 0.5


def test_escalated_pairs_are_cut_correctly(report: ClosedLoopReport) -> None:
    """The scale-mismatch guard: the mixed prob_rf/prob_llm stream is thresholded sanely,
    so escalated verdicts are mostly right. A blown scale would tank this, not just F1."""
    assert report.n_escalated > 0
    assert report.escalated_accuracy >= 0.8


def test_cascade_tracks_its_escalation_tier(report: ClosedLoopReport) -> None:
    """Cascade = student off-band + teacher on-band, so its F1 tracks the teacher within
    a small margin. (We do NOT assert cascade > student -- that's an economic claim the
    easy fixture can't honor; see the module docstring.)"""
    assert report.cascade.f1 >= report.teacher.f1 - 0.05
    assert report.student.f1 > 0.0
    assert report.teacher.f1 > 0.0


def test_band_is_data_derived_not_a_magic_constant(report: ClosedLoopReport) -> None:
    """The escalation band comes from the student's own calibration-split scores."""
    assert 0.0 < report.band_low < report.band_high <= 1.0
    assert 0.0 < report.band_fraction < 1.0


def test_circularity_warning_fires_on_silver_only_harvest(report: ClosedLoopReport) -> None:
    """The BEFORE arm (verdicts only, no corrections) trips the silver-only guard."""
    assert report.circularity_warning_fired is True
    assert report.n_corrections > 0


def test_corrected_pairs_do_not_return_in_the_next_queue(report: ClosedLoopReport) -> None:
    """A pair the human already answered is never re-queued for review."""
    assert report.next_queue_has_corrected_pair is False


def test_loop_is_deterministic() -> None:
    """Same seed -> identical escalation, band, and quality (seeded throughout)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = run_closed_loop(seed=0)
        b = run_closed_loop(seed=0)
    assert a.escalation_rate == b.escalation_rate
    assert (a.band_low, a.band_high) == (b.band_low, b.band_high)
    assert a.cascade.f1 == b.cascade.f1
    assert a.student.f1 == b.student.f1
