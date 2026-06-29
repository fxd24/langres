"""End-to-end validation of the full M3 harness with ZERO LLM spend.

Drives the importable core of ``examples/m3_zero_spend_race.py``
(``run_zero_spend_race``): every zero-spend scorer (rapidfuzz, weighted_average,
embedding_cosine) raced through ``run_method`` on BOTH ``FodorsZagatBenchmark``
and ``AmazonGoogleBenchmark`` at seed=0, with real MiniLM embeddings.

This is the gate that the whole protocol — the two dataset conformers, the method
registry, and the pair+pipeline tracks — works against Amazon-Google before the
paid W4 race. Asserts every cell yields a fully-populated ``MethodResult`` at zero
spend, and a discrimination check: on the hard Amazon-Google dataset the three
scorers' pair-level F1 must SPREAD (a flat result would mean the benchmark cannot
distinguish methods). Marked ``slow`` (it embeds two corpora) but network-free, so
it runs in CI.
"""

import pytest

from examples.m3_zero_spend_race import (
    _ag_pair_f1_spread,
    format_detailed_report,
    run_zero_spend_race,
)
from langres.core.benchmark import BenchmarkTable
from langres.methods import ZERO_SPEND_METHODS

pytestmark = pytest.mark.slow

#: AG is hard but the three scorers must still separate by at least this much
#: pair-level F1 for the benchmark to be discriminative (observed spread ~0.17).
_MIN_AG_SPREAD = 0.05


@pytest.fixture(scope="module")
def race_table() -> BenchmarkTable:
    """Run the full zero-spend race once and share it across this module's tests.

    The race embeds two corpora (minutes), so computing it once and reusing the
    deterministic ``BenchmarkTable`` keeps the slow suite affordable.
    """
    return run_zero_spend_race(seed=0)


def test_zero_spend_race_populates_every_cell_at_zero_spend(race_table: BenchmarkTable) -> None:
    """Both datasets × three scorers each yield a full MethodResult, $0 spent."""
    table = race_table
    # One cell per (dataset, scorer): 2 datasets × 3 zero-spend methods.
    assert len(table.results) == 2 * len(ZERO_SPEND_METHODS)

    datasets = {r.dataset for r in table.results}
    assert datasets == {"fodors_zagat", "amazon_google"}

    for r in table.results:
        # Pair track populated (incl. the full PR curve over the threshold grid).
        assert 0.0 <= r.pair.precision <= 1.0
        assert 0.0 <= r.pair.recall <= 1.0
        assert 0.0 <= r.pair.f1 <= 1.0
        assert r.pair.pr_curve is not None and len(r.pair.pr_curve) > 0

        # Pipeline track populated (BCubed + closure pairwise + floor + Δ).
        assert 0.0 <= r.pipeline.bcubed_f1 <= 1.0
        assert 0.0 <= r.pipeline.sanity_floor_f1 <= 1.0
        assert 0.0 <= r.pipeline.cluster_pairwise_f1 <= 1.0
        assert r.pipeline.delta_above_floor == pytest.approx(
            r.pipeline.bcubed_f1 - r.pipeline.sanity_floor_f1
        )

        # Zero-spend: nothing was charged on any cell.
        assert r.cost.usd_total == 0.0

    # The detailed report renders one row per cell plus a 2-line header.
    report = format_detailed_report(table)
    assert len(report.splitlines()) == len(table.results) + 2


def test_amazon_google_discriminates_between_scorers(race_table: BenchmarkTable) -> None:
    """On the hard AG dataset the three scorers' pair-level F1 visibly spread."""
    ag_min, ag_max, spread = _ag_pair_f1_spread(race_table)

    assert 0.0 <= ag_min <= ag_max <= 1.0
    assert spread >= _MIN_AG_SPREAD, (
        f"Amazon-Google pair-F1 spread {spread:.4f} is below {_MIN_AG_SPREAD}; "
        "the three zero-spend scorers did not separate (benchmark not discriminative)."
    )
