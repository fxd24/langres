"""Portfolio race: discover the benchmark portfolio from the registry and race it.

This is the registry-driven counterpart to the hand-written per-dataset race
scripts (``m3_zero_spend_race.py``, ``w2_person_benchmark.py``): instead of a
local, hard-coded ``_BENCHMARKS = (FodorsZagatBenchmark(), AmazonGoogleBenchmark(),
...)`` tuple that every new dataset has to be manually threaded into, it iterates
:func:`langres.data.registry.list_benchmarks` and races **whatever is registered**.

What this actually buys you — honestly:

- ``run_methods`` already killed the per-method racing boilerplate (build a
  factory, ``run_method``, ``table.add`` — one call now). This example adds
  nothing there; it *uses* that.
- The registry adds the missing half: **name -> benchmark discoverability** (a new
  dataset shows up here the moment it is ``register(...)``ed, with no edit to this
  file) and the same **serializable manifest** the CLI / other tools read. That is
  the real delta over a scattered local list — not a new racing capability.

Two surfaces, mirroring ``docs/EXPERIMENTS.md``:

- **Offline race (default).** The two zero-spend methods — ``rapidfuzz`` (string)
  and ``embedding_cosine`` (embedding) — raced via ``run_methods(bench, ...,
  budget=0.0)`` on every loadable benchmark, printed as one
  :meth:`~langres.core.benchmark.BenchmarkTable.to_markdown` table. Non-loadable
  entries (``opensanctions``: CC-BY-NC, never vendored) are skipped with a note.
- **Paid LLM row (``--paid``, OFF by default).** A zero-shot ``LLMJudge`` graded
  **judged-once** via :func:`~langres.core.benchmark.evaluate_judge_on_candidates`
  under a :class:`~langres.core.benchmark.BudgetedModuleRunner` pre-flight cap, all
  metered by one :class:`~langres.clients.openrouter.SpendMonitor`. Judged once —
  never through ``run_methods`` — because ``run_methods`` re-judges per grid
  threshold, which for a paid judge multiplies spend by the grid size (see the KISS
  warning in ``docs/EXPERIMENTS.md``). Requires ``OPENROUTER_API_KEY`` and a priced
  ``--model``; without either it prints why it skipped and does nothing.

Run (offline, $0)::

    uv run python examples/research/portfolio_race.py            # full portfolio
    uv run python examples/research/portfolio_race.py --fast     # FZ + DBLP-ACM only

Run (paid, capped)::

    uv run python examples/research/portfolio_race.py --fast --paid --budget 5

``print`` is allowed in examples (this is an operator tool).
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing anything that pulls torch/faiss
# (the dataset loaders import the embedding stack at module load; macOS libomp
# duplicate-load guard — mirrors examples/research/m3_zero_spend_race.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import logging  # noqa: E402
from typing import Any, Protocol, cast  # noqa: E402

from langres.core.benchmark import (  # noqa: E402
    Benchmark,
    BenchmarkTable,
    DEFAULT_PAIR_GRID,
    PairTrack,
    gold_pairs_from_clusters,
    run_methods,
)
from langres.data.registry import BenchmarkEntry, get_benchmark, list_benchmarks  # noqa: E402
from langres.methods import BlockingBenchmark, make_resolver_factory  # noqa: E402

logger = logging.getLogger("portfolio_race")

#: The two offline, zero-spend scorers: string similarity + local embeddings.
OFFLINE_METHODS: tuple[str, ...] = ("rapidfuzz", "embedding_cosine")
#: The small, fast, in-repo subset for a quick smoke (``--fast``): a saturated
#: restaurant set + a clean bibliographic one — both tiny, both offline.
FAST_SUBSET: frozenset[str] = frozenset({"fodors_zagat", "dblp_acm"})
SEED = 0

#: Paid-row defaults (only used with ``--paid``). gpt-4o-mini is priced in
#: PRICES_PER_1M; a $5 cap is a generous backstop for the small candidate bands.
DEFAULT_LLM_MODEL = "openrouter/openai/gpt-4o-mini"
DEFAULT_PAID_BUDGET_USD = 5.0
#: Generous worst-case tokens/pair sizing the BudgetedModuleRunner pre-flight cap.
WORST_CASE_TOKENS_PER_PAIR = 1200.0


class _RaceBenchmark(Benchmark[Any], BlockingBenchmark, Protocol):
    """A benchmark usable by BOTH the harness and the method registry.

    ``run_methods`` needs the :class:`~langres.core.benchmark.Benchmark` contract
    (``name`` / ``load`` / ``split`` / ``threshold_grid``); ``make_resolver_factory``
    needs the :class:`~langres.methods.BlockingBenchmark` contract (``schema`` +
    pinned blocking). Every registered loader satisfies this intersection, so a
    ``get_benchmark(name)`` result is cast to it (mirrors m3_zero_spend_race's
    ``_RaceBenchmark``).
    """


def select_benchmarks(*, fast: bool) -> list[str]:
    """Registry-driven benchmark selection: every loadable entry (or the fast subset).

    Iterates :func:`list_benchmarks` and returns the loadable names, printing a
    one-line note for each skipped non-loadable (external-only) entry so the
    portfolio's edges are visible rather than silent.

    Args:
        fast: If set, keep only :data:`FAST_SUBSET` (FZ + DBLP-ACM).

    Returns:
        Registered benchmark names to race, in registry (name-sorted) order.
    """
    names: list[str] = []
    for entry in list_benchmarks():
        if not entry.loadable:
            print(f"[skip] {entry.name}: external-only, not bundled — {_fetch_note(entry)}")
            continue
        names.append(entry.name)
    if fast:
        names = [n for n in names if n in FAST_SUBSET]
    return names


def _fetch_note(entry: BenchmarkEntry) -> str:
    """A short reason/where-to-get-it note for a non-loadable entry."""
    hint = entry.fetch_hint or ""
    # First sentence only; split on ". " (period-space) so "CC-BY-NC 4.0" survives.
    return hint.split(". ")[0] if hint else "fetch manually"


def race_offline(names: list[str]) -> BenchmarkTable:
    """Race the offline methods on each named benchmark into one merged table.

    For each name, ``get_benchmark(name)`` is raced through
    :func:`~langres.core.benchmark.run_methods` with ``budget=0.0`` (a hard
    assertion of genuine zero spend), and the per-dataset results are merged into a
    single :class:`~langres.core.benchmark.BenchmarkTable`.

    Args:
        names: Loadable benchmark names (from :func:`select_benchmarks`).

    Returns:
        One table with ``len(names) * len(OFFLINE_METHODS)`` rows.
    """
    table = BenchmarkTable()
    for name in names:
        bench = cast(_RaceBenchmark, get_benchmark(name))
        print(f"[race] {name}: {', '.join(OFFLINE_METHODS)} (offline, $0) ...")
        sub = run_methods(bench, list(OFFLINE_METHODS), seed=SEED, budget=0.0)
        table.results.extend(sub.results)
    return table


def race_paid_llm(
    names: list[str], *, model: str, budget_usd: float
) -> list[tuple[str, PairTrack, float]]:
    """Grade a zero-shot LLM judge, judged once, on each benchmark under one budget.

    For each benchmark the test split is blocked once into a fixed candidate set,
    then an ``LLMJudge`` is graded pairwise via
    :func:`~langres.core.benchmark.evaluate_judge_on_candidates` under a
    :class:`~langres.core.benchmark.BudgetedModuleRunner` whose hard cap is the
    budget *still remaining* (never a fresh floored allowance), so cumulative spend
    across datasets cannot exceed ``budget_usd``. A single :class:`SpendMonitor`
    meters that cumulative spend; once too little is left to fund even one
    worst-case pair the loop stops early, keeping the rows already collected. Judged
    **once** (never via ``run_methods``) so a paid judge is not re-charged per grid
    threshold.

    Args:
        names: Loadable benchmark names to score.
        model: A ``PRICES_PER_1M``-priced OpenRouter model id.
        budget_usd: Hard cumulative spend cap across all datasets.

    Returns:
        ``(name, pair_track, usd_spent)`` per benchmark (pair-level P/R/F1 at the
        best-F1 threshold, and the honest spend for that dataset).
    """
    # Paid-only imports kept local so the offline default never pulls litellm.
    from langres.clients import create_llm_client
    from langres.clients.openrouter import SpendMonitor, per_token_worst_price
    from langres.core.benchmark import BudgetedModuleRunner, evaluate_judge_on_candidates

    monitor = SpendMonitor(budget_usd=budget_usd)
    client = create_llm_client()
    worst_per_token = per_token_worst_price(model)
    # Cost of one worst-case pair — the minimum budget needed to make progress on a
    # dataset. Below this, minting a fresh runner would only overshoot the cumulative
    # cap (SpendMonitor.check runs after each dataset, so it can't pre-empt spend).
    min_progress_usd = WORST_CASE_TOKENS_PER_PAIR * worst_per_token
    rows: list[tuple[str, PairTrack, float]] = []

    for name in names:
        if monitor.remaining < min_progress_usd:
            print(
                f"[stop] budget exhausted: ${monitor.remaining:.4f} left "
                f"< ${min_progress_usd:.4f}/pair worst case; skipping remaining datasets."
            )
            break

        bench = cast(_RaceBenchmark, get_benchmark(name))
        corpus, gold_clusters, _ = bench.load()
        _, test_records, _, test_clusters = bench.split(corpus, gold_clusters, seed=SEED)

        resolver = make_resolver_factory("llm_judge", bench, llm_client=client, llm_model=model)(
            0.5
        )
        candidates = list(resolver._candidates([r.model_dump() for r in test_records]))
        gold_pairs = gold_pairs_from_clusters(test_clusters)

        # Bound both budgets by what's actually left — never mint fresh budget. Hard
        # cap = remaining (so cumulative spend can't exceed ``budget_usd``); soft cap
        # = 90% of remaining for pre-flight headroom. The break above guarantees
        # ``remaining >= min_progress_usd > 0``, so both are positive and soft <= hard
        # (the runner's constructor invariants).
        runner = BudgetedModuleRunner(
            resolver.module,
            budget_usd=monitor.remaining,
            budget_soft_usd=monitor.remaining * 0.9,
            worst_case_units_per_pair=WORST_CASE_TOKENS_PER_PAIR,
        )
        print(f"[paid] {name}: judging {len(candidates)} candidates with {model} ...")
        result, judgements = evaluate_judge_on_candidates(
            resolver.module,
            candidates,
            gold_pairs,
            DEFAULT_PAIR_GRID,
            runner=runner,
            price_per_token_or_pair=worst_per_token,
        )
        monitor.add(result.cost.usd_total)
        monitor.check()
        rows.append((name, result.pair, result.cost.usd_total))
    return rows


def _paid_table(rows: list[tuple[str, PairTrack, float]]) -> str:
    """Render the paid LLM pair-level rows as a small Markdown table."""
    header = (
        "| dataset | judge | pair_P | pair_R | pair_F1 | usd |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    body = [
        f"| {name} | llm_judge | {pair.precision:.4f} | {pair.recall:.4f} "
        f"| {pair.f1:.4f} | {usd:.4f} |"
        for name, pair, usd in rows
    ]
    return "\n".join([header, *body])


def _paid_guard_reason(model: str) -> str | None:
    """Return why the paid row must be skipped, or ``None`` if it can run.

    Checks (a) ``OPENROUTER_API_KEY`` is set and (b) ``model`` is priced (so its
    spend cap is not blind) — the same two guards the paid smoke scripts enforce.
    """
    from dotenv import load_dotenv

    from langres.clients.openrouter import PRICES_PER_1M

    load_dotenv(".env")  # OPENROUTER_API_KEY lives in .env, not Settings.
    if "OPENROUTER_API_KEY" not in os.environ:
        return "OPENROUTER_API_KEY not set"
    if model not in PRICES_PER_1M:
        return f"model {model!r} is not priced in PRICES_PER_1M (cap would be blind)"
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Registry-driven portfolio race: offline methods on every "
        "loadable benchmark, plus an optional budget-capped LLM row."
    )
    parser.add_argument(
        "--fast", action="store_true", help="Race only the fast subset (FZ + DBLP-ACM)."
    )
    parser.add_argument(
        "--paid",
        action="store_true",
        help="Also grade a zero-shot LLMJudge (needs OPENROUTER_API_KEY; costs money).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_LLM_MODEL, help="Priced OpenRouter model for the --paid row."
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_PAID_BUDGET_USD,
        help="Hard spend cap (USD) for the --paid row.",
    )
    args = parser.parse_args()

    print("=" * 78)
    print(f"Portfolio race — {'fast subset' if args.fast else 'full portfolio'} (seed={SEED})")
    print("=" * 78)

    names = select_benchmarks(fast=args.fast)
    if not names:
        print("[fatal] no loadable benchmarks selected; nothing to race.")
        return 1
    print(f"[plan] racing {len(names)} benchmark(s): {', '.join(names)}\n")

    table = race_offline(names)
    print("\n## Offline methods — pipeline BCubed F1 + pair F1 (BenchmarkTable)\n")
    print(table.to_markdown())

    if args.paid:
        reason = _paid_guard_reason(args.model)
        if reason is not None:
            print(f"\n[skip] --paid row: {reason}. Offline table above is complete.")
        else:
            print(f"\n## Paid zero-shot LLM judge — judged once, capped at ${args.budget:.2f}\n")
            rows = race_paid_llm(names, model=args.model, budget_usd=args.budget)
            print(_paid_table(rows))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
