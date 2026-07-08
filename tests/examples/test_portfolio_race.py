"""Smoke tests for the registry-driven portfolio race example.

Two tiers, matching the example's behavior/smoke coverage:

- **Fast, network- and embedding-free:** the genuinely new logic in this example
  is the registry-driven *selection* (``select_benchmarks``) and the report
  formatting. These assert loadable entries are raced, the external-only
  ``opensanctions`` entry is skipped with a note, and ``--fast`` narrows to the
  bundled subset — all from the static manifest, no dataset load.
- **Slow, embedding-backed:** one true end-to-end ``race_offline`` over the tiny
  12-record fixture, proving the offline surface yields populated, zero-spend
  ``MethodResult``s and renders a table. Marked ``slow`` (it loads MiniLM), so it
  runs in the weekly full suite, not per-PR.
"""

import pytest

from examples.research.portfolio_race import (
    FAST_SUBSET,
    OFFLINE_METHODS,
    _paid_table,
    race_offline,
    select_benchmarks,
)
from langres.core.benchmark import PairTrack


def test_select_benchmarks_full_skips_external_only_with_a_note(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The full selection returns loadable names and skips ``opensanctions`` (noted)."""
    names = select_benchmarks(fast=False)

    # Loadable entries are selected; the external-only one never is.
    assert "opensanctions" not in names
    assert "fodors_zagat" in names
    assert "dblp_acm" in names
    # A visible skip note names the external-only entry and its license reason.
    out = capsys.readouterr().out
    assert "[skip] opensanctions" in out
    assert "CC-BY-NC" in out


def test_select_benchmarks_fast_narrows_to_bundled_subset() -> None:
    """``--fast`` keeps only the fast subset (a strict subset of the full list)."""
    fast = select_benchmarks(fast=True)
    full = select_benchmarks(fast=False)

    assert set(fast) == set(FAST_SUBSET)
    assert set(fast) < set(full)


def test_paid_table_renders_one_row_per_dataset() -> None:
    """The paid pair-level table renders a header plus one row per (name, track)."""
    rows = [
        ("dblp_acm", PairTrack(precision=0.9, recall=0.8, f1=0.85), 0.12),
        ("fodors_zagat", PairTrack(precision=1.0, recall=0.7, f1=0.82), 0.05),
    ]
    table = _paid_table(rows)
    lines = table.splitlines()
    # Header row + separator row + one row per dataset.
    assert len(lines) == 2 + len(rows)
    assert "dblp_acm" in table
    assert "0.8500" in table  # f1 formatted to 4 dp


@pytest.mark.slow
def test_race_offline_tiny_fixture_populates_zero_spend_results() -> None:
    """End-to-end offline race over the 12-record fixture: full, $0 MethodResults."""
    table = race_offline(["tiny_fixture"])

    # One row per offline method on the single fixture.
    assert len(table.results) == len(OFFLINE_METHODS)
    assert {r.dataset for r in table.results} == {"tiny_fixture"}
    for r in table.results:
        assert 0.0 <= r.pair.f1 <= 1.0
        assert r.pair.pr_curve is not None and len(r.pair.pr_curve) > 0
        assert 0.0 <= r.pipeline.bcubed_f1 <= 1.0
        assert r.cost.usd_total == 0.0  # offline: nothing charged

    # to_markdown renders a header (+ separator) plus one row per result.
    assert len(table.to_markdown().splitlines()) == 2 + len(table.results)
