"""B2 -- should ``fit(derive_threshold=True)`` be the DEFAULT?

The front door hard-codes ``threshold: float = 0.5`` in six places
(``architectures/fuzzy_string.py``, ``architectures/retrieval.py`` x4,
``architectures/vector_llm_cascade.py``). PR #241 landed
``ERModel.fit(..., derive_threshold=True)``, which derives the match cut from
labels (Youden's J) and **races it against the incumbent** on the ``train``
split, keeping the incumbent unless the candidate strictly wins. The race exists
because a derived cut is not automatically a better cut: on one fixture the
derived cut tied on ``train`` and lost held-out (pair-F1 1.00 -> 0.80).

``derive_threshold`` defaults to ``False``. Should it default to ``True``?
**This script measures that; it does not assume it.** For every registered,
loadable benchmark it runs the real ``fit`` seam at the front door's ``0.5`` and
records four numbers: the incumbent cut and its held-out pair-F1, the derived
cut and *its* held-out pair-F1, and which one the race kept.

Design decisions, each load-bearing:

* **Every number is held-out and entity-disjoint.** ``fit(pairs=..., split=0.3,
  seed=...)`` hands the id-keyed labels to ``curation.harvest.align_pairs``,
  which assigns whole entity-components (union-find over the two ids each pair
  touches) to ``valid`` -- so no entity straddles the boundary. An in-sample cut
  flatters itself; ``fit`` itself prints ``IN-SAMPLE`` in capitals when no split
  is given, for exactly this reason.
* **The label set is EVERY blocked candidate, labeled by the closed-world gold
  partition.** That is the distribution the cut actually operates on at
  inference, and it is the *most generous* labeling budget deriving could ask
  for -- full supervision over the candidate stream. A cut that cannot win here
  will not win from a handful of corrections, so a negative result under this
  regime is the strong form of the negative result.
* **The incumbent is 0.5**, the constant the six architectures hard-code -- not
  each benchmark's tuned operating point. The question is about the *default*,
  so the baseline has to be the default.
* **Selection never sees ``valid``.** That is ``_select_threshold``'s own
  contract (it races on ``train``); this harness only reads what it reported.
  Both ``held_out_f1`` values are therefore clean estimates of two cuts, and the
  ``Δ`` this script reports is a real before/after.
* **Multiple seeds.** A conclusion that flips between seeds is not a conclusion.
  Each (benchmark, method) cell is fitted once per seed with the threshold reset
  to the incumbent first.

Everything here is **$0 in spend**: ``rapidfuzz`` and ``embedding_cosine`` make
no paid call. It is *not* dependency-free or offline on a cold cache -- blocking
is the benchmark's own ``VectorBlocker``, so a run needs ``[semantic]`` (and
``[trained]``, because ``derive_threshold`` goes through scikit-learn).

Run (offline, $0)::

    uv run python examples/research/threshold_default_study.py   # portfolio -> tracked JSON

A narrowed run must name its own ``--out``: the write replaces the file
wholesale, so pointing ``--fast``/``--only`` at the canonical artifact would
shrink the tracked portfolio result to the subset. The CLI refuses rather than
warning::

    uv run python examples/research/threshold_default_study.py --fast --out tmp/fast.json

``print`` is allowed in examples (this is an operator tool).
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing anything that pulls torch/faiss
# (macOS libomp duplicate-load guard -- mirrors closure_diagnostic.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Protocol, cast  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from langres.curation.harvest import LabeledPair  # noqa: E402
from langres.data.benchmark import Benchmark  # noqa: E402
from langres.eval import get_benchmark, list_benchmarks  # noqa: E402
from langres.methods import BlockingBenchmark, make_resolver_factory  # noqa: E402

logger = logging.getLogger("threshold_default_study")

#: The $0 scorers this study races. ``rapidfuzz`` stands in for the string path
#: (``FuzzyString``), ``embedding_cosine`` for the vector path
#: (``architectures/retrieval.py``) -- the two families the six hard-coded
#: ``0.5``s sit on. Both are rankers, so the cut is the whole decision.
DEFAULT_METHODS: tuple[str, ...] = ("rapidfuzz", "embedding_cosine")

#: The constant the six architectures hard-code, and therefore the incumbent
#: this study races the derived cut against.
INCUMBENT_THRESHOLD = 0.5

#: Held-out fraction for ``align_pairs``' entity-disjoint split.
HELD_OUT_SPLIT = 0.3

#: Split seeds. More than one on purpose: a flip between seeds is decision-relevant.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Small, in-repo datasets for a quick pass (``--fast``).
FAST_SUBSET: frozenset[str] = frozenset({"tiny_fixture", "fodors_zagat", "dblp_acm"})

#: The tracked artifact a FULL run refreshes.
CANONICAL_OUT = Path("examples/research/results/threshold_default_study.json")


class _StudyBenchmark(Benchmark[Any], BlockingBenchmark, Protocol):
    """A benchmark satisfying both the harness and the method-registry contracts."""


class CellResult(BaseModel):
    """One (benchmark, method, seed) fit: the four numbers the decision needs.

    Attributes:
        benchmark: Registry name.
        method: The $0 scorer.
        seed: The entity-disjoint split seed.
        n_records: Corpus size.
        n_candidates: Blocked candidate pairs (the label set size before the join).
        n_train / n_valid: Aligned pairs each side of the entity-disjoint split.
        n_valid_positives: Positive labels in ``valid`` (0 makes F1 meaningless).
        gold_coverage: Blocking pair-completeness on the labeled positives.
        incumbent_threshold / incumbent_held_out_f1: The 0.5 default and its
            held-out pair-F1.
        derived_threshold / derived_held_out_f1: Youden's J on ``train``, and
            *its* held-out pair-F1 -- reported whether or not the race kept it.
        incumbent_selection_f1 / derived_selection_f1: The ``train`` pair-F1s the
            race actually compared (selection never reads ``valid``).
        kept: ``"derived"`` (the candidate won) or ``"declined"`` (incumbent held).
        final_threshold: The cut in force on the model after ``fit`` returned.
        delta_f1: ``kept held-out F1 - incumbent held-out F1``. This is the
            effect of switching the default on: ``0.0`` whenever the race
            declined, and NEGATIVE only if the race kept a cut that lost held-out.
        seconds: Wall clock for the fit.
    """

    benchmark: str
    method: str
    seed: int
    n_records: int
    n_candidates: int
    n_train: int
    n_valid: int
    n_valid_positives: int
    gold_coverage: float | None
    incumbent_threshold: float
    incumbent_held_out_f1: float | None
    derived_threshold: float
    derived_held_out_f1: float | None
    incumbent_selection_f1: float | None
    derived_selection_f1: float | None
    kept: str
    final_threshold: float | None
    delta_f1: float | None
    seconds: float


def gold_labels_for_candidates(
    candidates: list[Any], gold_pairs: set[frozenset[str]]
) -> list[LabeledPair]:
    """Label every blocked candidate by closed-world gold membership.

    ``source="correction"`` (not ``"verdict"``): these labels were asserted by the
    dataset's gold partition, not read back off a judge's own cut, so they are
    gold. Stamping ``"verdict"`` would make ``derive_threshold_from_pairs`` warn
    about a circularity that is not present.

    ``score=None`` on purpose -- ``align_pairs`` reads only the two ids and the
    label, and ``fit`` re-scores the aligned candidates through the model's own
    (spend-capped) scorer. Supplying a score here would be a second, unused copy
    that could silently disagree with the one the fit actually uses.

    Args:
        candidates: The blocked candidates (``ERModel.candidates(...)``).
        gold_pairs: The closed-world gold pair set from ``Benchmark.load()``.

    Returns:
        One :class:`LabeledPair` per candidate, in candidate order.
    """
    labels: list[LabeledPair] = []
    for candidate in candidates:
        left, right = str(candidate.left.id), str(candidate.right.id)
        labels.append(
            LabeledPair(
                left_id=left,
                right_id=right,
                score=None,
                label=frozenset({left, right}) in gold_pairs,
                source="correction",
            )
        )
    return labels


def run_cell(
    *,
    benchmark: str,
    method: str,
    seed: int,
    resolver: Any,
    data: list[dict[str, Any]],
    labels: list[LabeledPair],
    n_candidates: int,
) -> CellResult:
    """Fit one cell and extract the race's own report.

    The threshold is reset to :data:`INCUMBENT_THRESHOLD` first, because
    ``resolver`` is reused across seeds and a kept cut from the previous seed
    would silently become the next seed's incumbent -- turning a portfolio study
    into a sequential optimization.

    Args:
        benchmark: Registry name (recorded, not used).
        method: The $0 scorer (recorded, not used).
        seed: The entity-disjoint split seed.
        resolver: The model to fit (mutated: threshold reset, then possibly set).
        data: Raw records as dicts.
        labels: Id-keyed gold labels over the blocked candidates.
        n_candidates: Blocked candidate count (recorded).

    Returns:
        The assembled :class:`CellResult`.

    Raises:
        RuntimeError: If ``fit`` reported no ``threshold_fit`` -- the one thing
            this study exists to read. Better to stop than to report a row whose
            provenance is missing.
    """
    resolver.clusterer.threshold = INCUMBENT_THRESHOLD
    started = time.monotonic()
    resolver.fit(data, pairs=labels, split=HELD_OUT_SPLIT, seed=seed, derive_threshold=True)
    seconds = time.monotonic() - started

    report = resolver.fit_report_
    fit = report.threshold_fit
    if fit is None or fit.previous is None or fit.candidate is None:
        raise RuntimeError(
            f"{benchmark}/{method}/seed={seed}: fit reported no threshold race "
            f"(threshold_fit={fit!r}); there is nothing to measure."
        )
    kept = "derived" if fit.source == "derived" else "declined"
    kept_f1 = fit.candidate.held_out_f1 if kept == "derived" else fit.previous.held_out_f1
    delta = (
        None
        if kept_f1 is None or fit.previous.held_out_f1 is None
        else kept_f1 - fit.previous.held_out_f1
    )
    coverage = report.coverage
    return CellResult(
        benchmark=benchmark,
        method=method,
        seed=seed,
        n_records=len(data),
        n_candidates=n_candidates,
        n_train=report.n_train,
        n_valid=report.n_valid,
        n_valid_positives=(
            0 if report.metrics is None else report.metrics.tp + report.metrics.fn
        ),
        gold_coverage=None if coverage is None else coverage.gold_coverage,
        incumbent_threshold=fit.previous.threshold,
        incumbent_held_out_f1=fit.previous.held_out_f1,
        derived_threshold=fit.candidate.threshold,
        derived_held_out_f1=fit.candidate.held_out_f1,
        incumbent_selection_f1=fit.previous.selection_f1,
        derived_selection_f1=fit.candidate.selection_f1,
        kept=kept,
        final_threshold=resolver.clusterer.threshold,
        delta_f1=delta,
        seconds=seconds,
    )


def run_benchmark(
    name: str, *, methods: tuple[str, ...], seeds: tuple[int, ...]
) -> list[CellResult]:
    """Run every (method, seed) cell for one registered benchmark.

    One resolver per method, reused across seeds: ``BlockerSource.prepare``
    reuses a vector index for an identical corpus, so the embedding pass is paid
    once per method instead of once per seed.

    Args:
        name: Registry name (must be ``loadable``).
        methods: $0 method names.
        seeds: Entity-disjoint split seeds.

    Returns:
        One :class:`CellResult` per (method, seed).
    """
    bench = cast(_StudyBenchmark, get_benchmark(name))
    corpus, _gold_clusters, gold_pairs = bench.load()
    data = [record.model_dump() for record in corpus]

    results: list[CellResult] = []
    for method in methods:
        resolver = make_resolver_factory(method, bench)(INCUMBENT_THRESHOLD)
        candidates = resolver.candidates(data)
        labels = gold_labels_for_candidates(candidates, gold_pairs)
        n_candidates = len(candidates)
        del candidates  # the label list is all that is needed downstream
        for seed in seeds:
            result = run_cell(
                benchmark=name,
                method=method,
                seed=seed,
                resolver=resolver,
                data=data,
                labels=labels,
                n_candidates=n_candidates,
            )
            results.append(result)
            print(
                f"       {method} seed={seed}: incumbent {result.incumbent_threshold:.2f} "
                f"F1={_f1(result.incumbent_held_out_f1)} | derived "
                f"{result.derived_threshold:.4f} F1={_f1(result.derived_held_out_f1)} | "
                f"kept={result.kept} Δ={_f1(result.delta_f1)} ({result.seconds:.1f}s)",
                flush=True,
            )
    return results


def _f1(value: float | None) -> str:
    """Format an F1/delta, or ``n/a`` when the split produced nothing to grade."""
    return "n/a" if value is None else f"{value:+.4f}" if value < 0 else f"{value:.4f}"


def to_markdown(results: list[CellResult]) -> str:
    """Render the per-benchmark table. Never an average -- an average hides the
    single cell that motivated the race in the first place."""
    header = (
        "| benchmark | method | seed | n_train | n_valid | incumbent t | incumbent F1 | "
        "derived t | derived F1 | kept | Δ F1 |"
    )
    lines = [header, "|" + "---|" * 11]
    for r in results:
        lines.append(
            f"| {r.benchmark} | {r.method} | {r.seed} | {r.n_train:,} | {r.n_valid:,} | "
            f"{r.incumbent_threshold:.2f} | {_f1(r.incumbent_held_out_f1)} | "
            f"{r.derived_threshold:.4f} | {_f1(r.derived_held_out_f1)} | "
            f"{r.kept} | {_f1(r.delta_f1)} |"
        )
    return "\n".join(lines)


def select_benchmarks(*, fast: bool, only: list[str] | None) -> list[str]:
    """Registry-driven selection: every loadable entry, or a narrowed subset.

    Args:
        fast: Keep only :data:`FAST_SUBSET`.
        only: Explicit names (wins over ``fast``).

    Returns:
        Registered, loadable benchmark names in registry order.

    Raises:
        SystemExit: If ``only`` selects nothing -- an empty run would otherwise
            write an empty artifact over a real result set.
    """
    names = [entry.name for entry in list_benchmarks() if entry.loadable]
    for entry in list_benchmarks():
        if not entry.loadable:
            print(f"[skip] {entry.name}: external-only, not bundled")
    if only:
        unknown = sorted(set(only) - {entry.name for entry in list_benchmarks()})
        if unknown:
            print(f"[warn] --only names no registered benchmark: {', '.join(unknown)}")
        selected = [name for name in names if name in set(only)]
        if not selected:
            raise SystemExit(
                f"--only {' '.join(only)} selected no loadable benchmark; refusing to run. "
                f"Loadable: {', '.join(names)}"
            )
        return selected
    if fast:
        return [name for name in names if name in FAST_SUBSET]
    return names


def write_results(results: list[CellResult], out: Path) -> None:
    """Persist the machine-readable findings to ``out`` (creating parents)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="only the small in-repo subset")
    parser.add_argument("--only", nargs="+", help="explicit benchmark names")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "where to write the machine-readable findings (a TRACKED path). "
            f"Defaults to {CANONICAL_OUT} for a FULL run; REQUIRED with --fast/--only, "
            "which would otherwise replace the full-portfolio artifact with a subset"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.out is None and (args.fast or args.only):
        parser.error(
            "--fast/--only measure a subset and the write replaces the whole file, "
            f"which would reduce {CANONICAL_OUT} to just those benchmarks. "
            "Pass an explicit --out (e.g. --out tmp/threshold_subset.json)."
        )
    out: Path = args.out if args.out is not None else CANONICAL_OUT

    results: list[CellResult] = []
    failures: list[str] = []
    for name in select_benchmarks(fast=args.fast, only=args.only):
        print(f"[run] {name}: {' '.join(args.methods)} ($0 spend) ...", flush=True)
        try:
            results.extend(
                run_benchmark(
                    name, methods=tuple(args.methods), seeds=tuple(args.seeds)
                )
            )
        except Exception as exc:  # noqa: BLE001 - a broken loader must not kill the sweep
            logger.exception("%s: run_benchmark raised", name)
            print(f"[fail] {name}: {type(exc).__name__}: {exc}", flush=True)
            failures.append(f"{name} ({type(exc).__name__})")
            continue
        # Persist after EVERY benchmark: the large datasets take minutes and a
        # crash on the last one must not throw away the ones already measured.
        write_results(results, out)

    print()
    print(to_markdown(results))
    print()
    print(f"[out] {out}")
    if failures:
        print(f"[fail] {len(failures)} benchmark(s) did not complete: {', '.join(failures)}")


if __name__ == "__main__":
    main()
