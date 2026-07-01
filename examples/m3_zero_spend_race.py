"""M3 zero-spend race — the full benchmark harness, end-to-end, with NO LLM spend.

Validates the complete M3 protocol (``run_method`` + the method registry + the
two dataset conformers) before the paid LLM race (W4), using only the three
deterministic, zero-spend scorers:

    rapidfuzz · weighted_average · embedding_cosine   (``ZERO_SPEND_METHODS``)

Each scorer is raced on BOTH datasets at seed=0:

    FodorsZagatBenchmark  — saturated, easy (blocking PC >= 0.99)
    AmazonGoogleBenchmark — hard, unsaturated (blocking PC ~0.84, recall-capped)

For every (dataset, scorer) cell it reports both evaluation tracks: pair-level
Precision/Recall/F1 (pre-clustering, isolating the scorer) AND the pipeline
BCubed P/R/F1, the transitive-closure cluster pairwise-F1, the all-singletons
sanity floor, and Δ above that floor. Real MiniLM embeddings drive blocking; the
scorers make no API call, so the whole run is deterministic and costs $0.

Run:
    uv run python examples/m3_zero_spend_race.py

The committed ``examples/m3_zero_spend_race_output.md`` captures a reference run
so the numbers survive without re-embedding.
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing anything that loads torch/faiss,
# so the run is deterministic and avoids the macOS libomp duplicate-load crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Any, Protocol  # noqa: E402

from langres.core.benchmark import Benchmark, BenchmarkTable, MethodResult, run_method  # noqa: E402
from langres.data.amazon_google import AmazonGoogleBenchmark  # noqa: E402
from langres.data.er_benchmarks import FodorsZagatBenchmark  # noqa: E402
from langres.methods import (  # noqa: E402
    BlockingBenchmark,
    ZERO_SPEND_METHODS,
    make_resolver_factory,
)

SEED = 0


class _RaceBenchmark(Benchmark[Any], BlockingBenchmark, Protocol):
    """A benchmark usable by BOTH the harness and the method registry.

    ``run_method`` needs the :class:`~langres.core.benchmark.Benchmark` contract
    (``name``/``load``/``split``/``threshold_grid``); ``make_resolver_factory``
    needs the :class:`~langres.methods.BlockingBenchmark` contract (``schema`` +
    pinned blocking). Both conformers satisfy this intersection structurally, so
    one heterogeneous tuple can be typed without losing either contract.
    """


#: The two datasets, in race order (easy/saturated first, then hard/unsaturated).
_BENCHMARKS: tuple[_RaceBenchmark, ...] = (FodorsZagatBenchmark(), AmazonGoogleBenchmark())


def run_zero_spend_race(seed: int = SEED) -> BenchmarkTable:
    """Race every zero-spend scorer on both benchmarks, collecting a table.

    Drives the real harness: ``make_resolver_factory(method, bench)`` →
    ``run_method(bench, factory, seed, budget=0.0)``. ``budget=0.0`` is a hard
    assertion that these methods truly spend nothing (``run_method`` raises if any
    measured spend exceeds it). Results are appended in (dataset, method) order.

    Args:
        seed: Split seed (same leakage-free split for every method on a dataset).

    Returns:
        A :class:`BenchmarkTable` with one :class:`MethodResult` per cell
        (``len == len(_BENCHMARKS) * len(ZERO_SPEND_METHODS)``).
    """
    table = BenchmarkTable()
    for bench in _BENCHMARKS:
        for method in ZERO_SPEND_METHODS:
            factory = make_resolver_factory(method, bench)
            # budget=0.0 asserts genuine zero spend (raises if anything is charged).
            result = run_method(bench, factory, seed=seed, budget=0.0)
            table.add(result)
    return table


def format_detailed_report(table: BenchmarkTable) -> str:
    """Render both tracks per cell as a Markdown table (the headline `to_markdown`
    only carries BCubed-F1 + pair-F1; this adds the full P/R and pipeline diagnostics).

    Args:
        table: A table populated by :func:`run_zero_spend_race`.

    Returns:
        A Markdown table string: one row per (dataset, scorer) with pair-level
        Precision/Recall/F1 and pipeline BCubed P/R/F1, cluster pairwise-F1,
        sanity floor, Δ-above-floor, spend, and seconds/pair.
    """
    header = (
        "| dataset | scorer | thr | pair_P | pair_R | pair_F1 "
        "| bc_P | bc_R | bc_F1 | clus_F1 | floor_F1 | Δ_floor | usd | s/pair |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows: list[str] = []
    for r in table.results:
        rows.append(
            f"| {r.dataset} | {r.method} | {r.threshold:.2f} "
            f"| {r.pair.precision:.4f} | {r.pair.recall:.4f} | {r.pair.f1:.4f} "
            f"| {r.pipeline.bcubed_p:.4f} | {r.pipeline.bcubed_r:.4f} | {r.pipeline.bcubed_f1:.4f} "
            f"| {r.pipeline.cluster_pairwise_f1:.4f} | {r.pipeline.sanity_floor_f1:.4f} "
            f"| {r.pipeline.delta_above_floor:+.4f} | {r.cost.usd_total:.4f} "
            f"| {r.latency.seconds_per_pair:.6f} |"
        )
    return "\n".join([header, *rows])


def _ag_pair_f1_spread(table: BenchmarkTable) -> tuple[float, float, float]:
    """Return ``(min, max, spread)`` of Amazon-Google pair-level F1 across scorers.

    The spread (max − min) is the discrimination signal: a hard dataset where the
    three scorers separate tells us the benchmark distinguishes methods (and sets
    the bar the W4 LLM judge must beat).
    """
    ag_f1s = [r.pair.f1 for r in table.results if r.dataset == "amazon_google"]
    return min(ag_f1s), max(ag_f1s), max(ag_f1s) - min(ag_f1s)


def _best_ag_pair_f1(table: BenchmarkTable) -> MethodResult:
    """The Amazon-Google cell with the highest pair-level F1 (the zero-spend bar)."""
    ag = [r for r in table.results if r.dataset == "amazon_google"]
    return max(ag, key=lambda r: r.pair.f1)


def main() -> None:
    """Run the deterministic, zero-spend race and print both report views."""
    print("=" * 78)
    print("M3 zero-spend race — full harness on Fodors-Zagat + Amazon-Google (seed=0)")
    print("=" * 78)

    table = run_zero_spend_race(SEED)

    print("\n## Both tracks per (dataset, scorer)\n")
    print(format_detailed_report(table))

    print("\n## Headline table (BenchmarkTable.to_markdown)\n")
    print(table.to_markdown())

    ag_min, ag_max, spread = _ag_pair_f1_spread(table)
    best = _best_ag_pair_f1(table)
    print("\n## Amazon-Google read-out\n")
    print(
        f"- Pair-level F1 spread across the 3 scorers: {spread:.4f} "
        f"(min {ag_min:.4f}, max {ag_max:.4f}) — {'DISCRIMINATES' if spread >= 0.05 else 'flat'}"
    )
    print(
        f"- Best zero-spend pair-level F1: {best.pair.f1:.4f} "
        f"({best.method}) — the bar W4's LLM judge must beat."
    )
    print(
        f"- Pipeline BCubed recall is blocking-ceiling-limited (AG blocking "
        f"Pair-Completeness caps ~0.84): best AG bcubed_R = "
        f"{max(r.pipeline.bcubed_r for r in table.results if r.dataset == 'amazon_google'):.4f}."
    )
    print(
        f"- Total spend across all {len(table.results)} cells: "
        f"${sum(r.cost.usd_total for r in table.results):.4f} (zero-spend)."
    )


if __name__ == "__main__":
    main()
