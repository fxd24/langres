"""M5 W2.1 person benchmark — a SECOND entity type resolved config-only at $0.

Proves the M5 "generality" exit: langres resolves a *person* (not a product or
restaurant) with **zero new core code** — only the config-only
:mod:`langres.data.febrl_person` adapter — across five free, local methods, with
no LLM spend.

Two evaluation surfaces, mirroring the M3/W1.2 examples so the numbers are
directly comparable:

    * ``ZERO_SPEND_METHODS`` (rapidfuzz · weighted_average · embedding_cosine) —
      raced through the full harness (``run_method``), reporting BOTH pipeline
      BCubed P/R/F1 and pair-level P/R/F1. ``budget=0.0`` hard-asserts zero spend.
    * the trained family (fellegi_sunter · random_forest) — neither can be raced
      unfit, so they follow the fit-seam pattern from
      ``examples/research/w1_trained_family_race.py``: fit on the train split's own blocked
      candidates (FS unsupervised, RF supervised), then grade the TEST split's
      blocked candidates once via ``evaluate_judge_on_candidates`` (pair-level F1
      only — the judged-once surface has no clustering step, so no BCubed).

Real MiniLM embeddings drive blocking; every scorer is local, so the whole run is
deterministic and costs $0. The committed
``docs/research/20260703_w2_person_benchmark_results.md`` captures a reference run
so the numbers survive without re-embedding.

Run:
    uv run python examples/research/w2_person_benchmark.py
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing anything that loads torch/faiss,
# so the run is deterministic and avoids the macOS libomp duplicate-load crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

from langres.core.benchmark import (  # noqa: E402
    BenchmarkTable,
    JudgePairEval,
    evaluate_judge_on_candidates,
    gold_pairs_from_clusters,
    run_method,
)
from langres.core.models import ERCandidate  # noqa: E402
from langres.core.resolver import Resolver  # noqa: E402
from langres.data.febrl_person import FebrlPersonBenchmark  # noqa: E402
from langres.methods import ZERO_SPEND_METHODS, make_resolver_factory  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 0

#: The trained-family method names (excluded from ZERO_SPEND_METHODS/ALL_METHODS
#: since neither can be raced unfit — see methods.py).
_TRAINED_METHODS: tuple[str, ...] = ("fellegi_sunter", "random_forest")


@dataclass
class TrainedCell:
    """One trained-family (method) result: the graded pairwise eval on test."""

    method: str
    eval: JudgePairEval


def _as_dicts(records: list[Any]) -> list[dict[str, Any]]:
    """Convert schema record instances to plain dicts (what Resolver expects)."""
    return [r.model_dump() for r in records]


def _blocked_candidates_and_labels(
    resolver: Resolver, records: list[Any], gold_pairs: set[frozenset[str]]
) -> tuple[list[ERCandidate[Any]], list[bool]]:
    """Block+compare ``records`` and label each candidate against ``gold_pairs``.

    Uses ``resolver._candidates`` — the same blocker-stream + comparator-attach
    helper ``Resolver.fit``/``predict`` use — so the candidate set is identical to
    what fitting/scoring will see.
    """
    candidates = list(resolver._candidates(_as_dicts(records)))  # noqa: SLF001
    labels = [frozenset({c.left.id, c.right.id}) in gold_pairs for c in candidates]
    return candidates, labels


def run_zero_spend_race(seed: int = SEED) -> BenchmarkTable:
    """Race every zero-spend scorer on FEBRL persons, collecting a table.

    ``budget=0.0`` hard-asserts genuine zero spend (``run_method`` raises if any
    measured spend exceeds it).

    Args:
        seed: Split seed (same leakage-free split for every method).

    Returns:
        A :class:`BenchmarkTable` with one result per zero-spend method.
    """
    bench = FebrlPersonBenchmark()
    table = BenchmarkTable()
    for method in ZERO_SPEND_METHODS:
        factory = make_resolver_factory(method, bench)
        result = run_method(bench, factory, seed=seed, budget=0.0)
        table.add(result)
    return table


def run_trained_family(seed: int = SEED) -> list[TrainedCell]:
    """Fit + evaluate both trained judges on FEBRL persons (pairwise-F1, $0).

    Splits leakage-free, fits on the train split's own blocked candidates (FS
    ignores the derived labels; RF requires them), then grades the TEST split's
    blocked candidates once at the benchmark's threshold grid.

    Args:
        seed: Split seed (same split both methods see).

    Returns:
        One :class:`TrainedCell` per trained method, in order.
    """
    bench = FebrlPersonBenchmark()
    corpus, gold_clusters, _gold_pairs = bench.load()
    train_records, test_records, train_clusters, test_clusters = bench.split(
        corpus, gold_clusters, seed=seed
    )
    train_gold = gold_pairs_from_clusters(train_clusters)
    test_gold = gold_pairs_from_clusters(test_clusters)

    cells: list[TrainedCell] = []
    for method in _TRAINED_METHODS:
        # threshold is unused by evaluate_judge_on_candidates (it sweeps its own
        # grid); 0.5 is an arbitrary placeholder for the factory call.
        resolver = make_resolver_factory(method, bench)(0.5)
        _train_candidates, train_labels = _blocked_candidates_and_labels(
            resolver, train_records, train_gold
        )

        if method == "fellegi_sunter":
            resolver.fit(_as_dicts(train_records))  # fit_unlabeled under the hood
        else:
            resolver.fit(_as_dicts(train_records), labels=train_labels)

        test_candidates = list(resolver._candidates(_as_dicts(test_records)))  # noqa: SLF001
        result, _judgements = evaluate_judge_on_candidates(
            resolver.module, test_candidates, test_gold, grid=bench.threshold_grid
        )
        cells.append(TrainedCell(method=method, eval=result))
        logger.info(
            "febrl_person / %s -> pair F1=%.3f (P=%.3f R=%.3f) @ threshold=%.2f, n=%d",
            method,
            result.pair.f1,
            result.pair.precision,
            result.pair.recall,
            result.best_threshold,
            result.n_candidates,
        )
    return cells


def format_report(table: BenchmarkTable, trained: list[TrainedCell]) -> str:
    """Render one unified Markdown table across all five free methods.

    Zero-spend methods carry both pipeline BCubed-F1 and pair-level P/R/F1; the
    trained family carries pair-level P/R/F1 only (the judged-once surface has no
    clustering step, so BCubed is not defined there — shown as ``—``).

    Args:
        table: The zero-spend race table.
        trained: The trained-family cells.

    Returns:
        A Markdown table string.
    """
    header = (
        "| method | family | bcubed_f1 | pair_P | pair_R | pair_F1 | thr | usd |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows: list[str] = []
    for r in table.results:
        rows.append(
            f"| {r.method} | zero-spend "
            f"| {r.pipeline.bcubed_f1:.4f} "
            f"| {r.pair.precision:.4f} | {r.pair.recall:.4f} | {r.pair.f1:.4f} "
            f"| {r.threshold:.2f} | {r.cost.usd_total:.4f} |"
        )
    for cell in trained:
        rows.append(
            f"| {cell.method} | trained "
            f"| — "
            f"| {cell.eval.pair.precision:.4f} | {cell.eval.pair.recall:.4f} "
            f"| {cell.eval.pair.f1:.4f} "
            f"| {cell.eval.best_threshold:.2f} | {cell.eval.cost.usd_total:.4f} |"
        )
    return "\n".join([header, *rows])


def main() -> None:
    """Run the deterministic, zero-spend person benchmark and print the table."""
    print("=" * 78)
    print("M5 W2.1 person benchmark — FEBRL4, five free methods, $0 (seed=0)")
    print("=" * 78)

    table = run_zero_spend_race(SEED)
    trained = run_trained_family(SEED)

    print("\n## Per-method results (BCubed F1 + pairwise P/R/F1)\n")
    print(format_report(table, trained))

    total_spend = sum(r.cost.usd_total for r in table.results) + sum(
        c.eval.cost.usd_total for c in trained
    )
    print(f"\nTotal spend across all {len(table.results) + len(trained)} cells: ${total_spend:.4f}")


if __name__ == "__main__":
    main()
