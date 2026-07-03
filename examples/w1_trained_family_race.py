"""W1.2 trained-family replication — FellegiSunterJudge + RFJudge, $0.

Races the two W1.2 "trained family" judges — the first learn-with-no-labels
proof (:class:`~langres.core.judges.fellegi_sunter.FellegiSunterJudge`, EM over
random-sampled u + the blocked candidates for m/prior) and a Magellan-style
supervised RandomForest (:class:`~langres.core.modules.rf_judge.RFJudge`) — on
all three $0 replication datasets:

    FodorsZagatBenchmark   — saturated, clean multi-field identity data
    AmazonGoogleBenchmark  — hard, single-text-blob (title+manufacturer)
    AbtBuyBenchmark        — textual-hard (free-text description, often missing)

Unlike the zero-spend race in ``examples/m3_zero_spend_race.py``, neither judge
can be raced through ``run_method`` (see the KISS warning in
``docs/EXPERIMENTS.md``): both need an explicit fit step before ``forward()``
works. This script instead follows the fit-seam pattern documented there:

    1. ``benchmark.split(...)`` — a leakage-free, whole-cluster train/test split.
    2. ``resolver.fit(train_records)`` (FS, unsupervised) or
       ``resolver.fit(train_records, labels=...)`` (RF, supervised) — the
       labels come from the train split's own blocked candidates against its
       gold pairs, not a separate label source.
    3. ``evaluate_judge_on_candidates`` on the TEST split's blocked candidates —
       judged once, graded at the best-F1 grid threshold. This is the same
       pairwise-F1 surface the M4 DSPy probe used, so the numbers are directly
       comparable across judges and to literature bands.

Read the result numbers against ``docs/research/20260702_w1_trained_family_results.md``
(the committed reference run + the literature-band comparison), not as exact
targets — the goal is "in the band, honestly reported", not decimal
replication (docs/research/20260701_er_seam_audit.md).

Run:
    uv run python examples/w1_trained_family_race.py
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing anything that loads torch/faiss,
# so the run is deterministic and avoids the macOS libomp duplicate-load crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, Protocol  # noqa: E402

from langres.core.benchmark import (  # noqa: E402
    Benchmark,
    JudgePairEval,
    evaluate_judge_on_candidates,
    gold_pairs_from_clusters,
)
from langres.core.models import ERCandidate  # noqa: E402
from langres.core.resolver import Resolver  # noqa: E402
from langres.data.abt_buy import AbtBuyBenchmark  # noqa: E402
from langres.data.amazon_google import AmazonGoogleBenchmark  # noqa: E402
from langres.data.er_benchmarks import FodorsZagatBenchmark  # noqa: E402
from langres.methods import BlockingBenchmark, make_resolver_factory  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 0

#: The trained-family method names (see methods.py — deliberately excluded from
#: ZERO_SPEND_METHODS/ALL_METHODS, since neither can be raced unfit).
_METHODS: tuple[str, ...] = ("fellegi_sunter", "random_forest")


class _RaceBenchmark(Benchmark[Any], BlockingBenchmark, Protocol):
    """A benchmark usable by BOTH the harness contract and the method registry.

    Mirrors ``examples/m3_zero_spend_race.py``'s identical intersection type.
    """


#: The three $0 replication datasets (CEO #9), easy -> hard -> textual-hard.
_BENCHMARKS: tuple[_RaceBenchmark, ...] = (
    FodorsZagatBenchmark(),
    AmazonGoogleBenchmark(),
    AbtBuyBenchmark(),
)


@dataclass
class TrainedCell:
    """One (dataset, method) result: the graded pairwise evaluation on test."""

    dataset: str
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
    helper ``Resolver.fit``/``Resolver.predict`` use internally — so the
    candidate set here is byte-identical to what fitting/scoring will see.
    """
    candidates = list(resolver._candidates(_as_dicts(records)))  # noqa: SLF001
    labels = [frozenset({c.left.id, c.right.id}) in gold_pairs for c in candidates]
    return candidates, labels


def run_trained_family_replication(seed: int = SEED) -> list[TrainedCell]:
    """Fit + evaluate both trained judges on all three datasets.

    For each dataset: split leakage-free, fit on the train split's own blocked
    candidates (FS ignores the derived labels; RF requires them), then grade
    the TEST split's blocked candidates once via ``evaluate_judge_on_candidates``
    at the dataset's own threshold grid.

    Args:
        seed: Split seed (same split every method sees, per dataset).

    Returns:
        One :class:`TrainedCell` per (dataset, method), in race order.
    """
    cells: list[TrainedCell] = []
    for bench in _BENCHMARKS:
        corpus, gold_clusters, _gold_pairs = bench.load()
        train_records, test_records, train_clusters, test_clusters = bench.split(
            corpus, gold_clusters, seed=seed
        )
        train_gold = gold_pairs_from_clusters(train_clusters)
        test_gold = gold_pairs_from_clusters(test_clusters)

        for method in _METHODS:
            # threshold is unused by evaluate_judge_on_candidates (it sweeps its
            # own grid); 0.5 is an arbitrary placeholder for the factory call.
            # Uses the class-default hyperparameters (the same construction
            # methods.py wires) -- see docs/research/20260702_w1_trained_family_results.md
            # for what happens (and doesn't help) if FellegiSunterJudge's EM
            # iteration budget is widened past the default.
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
            cells.append(TrainedCell(dataset=bench.name, method=method, eval=result))
            logger.info(
                "%s / %s -> pair F1=%.3f (P=%.3f R=%.3f) @ threshold=%.2f, n=%d",
                bench.name,
                method,
                result.pair.f1,
                result.pair.precision,
                result.pair.recall,
                result.best_threshold,
                result.n_candidates,
            )
    return cells


def format_report(cells: list[TrainedCell]) -> str:
    """Render one Markdown table row per (dataset, method) cell.

    Args:
        cells: Results from :func:`run_trained_family_replication`.

    Returns:
        A Markdown table string.
    """
    header = (
        "| dataset | method | precision | recall | f1 | n_candidates | best_threshold |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = [
        f"| {cell.dataset} | {cell.method} "
        f"| {cell.eval.pair.precision:.4f} | {cell.eval.pair.recall:.4f} | {cell.eval.pair.f1:.4f} "
        f"| {cell.eval.n_candidates} | {cell.eval.best_threshold:.2f} |"
        for cell in cells
    ]
    return "\n".join([header, *rows])


def main() -> None:
    """Run the deterministic, zero-spend trained-family replication and print it."""
    print("=" * 78)
    print("W1.2 trained-family replication — FellegiSunterJudge + RFJudge (seed=0)")
    print("=" * 78)

    cells = run_trained_family_replication(SEED)

    print("\n## Pairwise F1 per (dataset, method)\n")
    print(format_report(cells))


if __name__ == "__main__":
    main()
