"""The threshold-default study -- should ``fit(derive_threshold=True)`` be the DEFAULT?

(A $0 diagnostic in the spirit of the agenda's B thread, but NOT agenda item B1
or B2 -- both of those are answered elsewhere
(``docs/research/20260727_closure_diagnostic.md`` /
``20260727_portfolio_annotation.md``). This question came out of PR #241.)

The front door hard-codes ``threshold: float = 0.5`` in six places
(``architectures/fuzzy_string.py``, ``architectures/retrieval.py`` x4,
``architectures/vector_llm_cascade.py``). PR #241 landed
``ERModel.fit(..., derive_threshold=True)``, which derives the match cut from
labels (Youden's J) and **races it against the incumbent** on the ``train``
split, keeping the incumbent unless the candidate *strictly* wins. The race
exists because a derived cut is not automatically a better cut: on one fixture
the derived cut tied on ``train`` and lost held-out (pair-F1 1.00 -> 0.80).

``derive_threshold`` defaults to ``False``. Should it default to ``True``?
**This script measures that; it does not assume it.** For every registered,
loadable benchmark it runs the real ``fit`` seam against the front door's
``0.5`` and records four numbers: the incumbent cut and its held-out pair-F1,
the derived cut and *its* held-out pair-F1, and which one the race kept.

Design decisions, each load-bearing:

* **Held-out means a DISJOINT CORPUS, not a disjoint slice of one label set.**
  The obvious design -- label every blocked candidate and let
  ``fit(pairs=..., split=0.3)`` hold out 30 % -- was tried first and
  **degenerates**, which is a finding in its own right (§ "the split trap" in
  the write-up). ``align_pairs``' entity-disjoint split assigns whole
  union-find components, and a k-NN candidate graph over a real corpus is
  essentially *one* component: on ``fodors_zagat`` it produced ``n_valid = 0``
  at every seed, and on ``dblp_acm`` it produced 58/18,127 -- inverted at
  seed 0 and 58 held-out pairs at seeds 1-2. Numbers off 58 pairs are noise.
  So the split here is the benchmark's own ``Benchmark.split`` -- whole gold
  clusters to one side, so no entity and no match pair straddles the boundary --
  and the two corpora are blocked, scored and graded independently. Every cell
  still records what ``align_pairs(split=0.3)`` *would* have held out
  (``align_split_train`` / ``align_split_valid``), so this design choice is
  justified by data in the tracked artifact rather than by a claim.
* **The race is untouched.** ``_select_threshold`` selects on ``train`` and
  never reads ``valid``; the ``split=`` argument only affects what ``fit``
  *reports*. So running ``fit(..., split=None)`` on the train corpus gives the
  race exactly the label set a user would give it, and grading on the disjoint
  test corpus is a strictly *stronger* held-out estimate than the one ``fit``
  would have printed.
* **The label set is EVERY blocked candidate on the train corpus**, labeled by
  the closed-world gold partition. That is the distribution the cut operates on
  at inference, and it is the *most generous* labeling budget deriving could
  ask for -- full supervision over the candidate stream. A cut that cannot win
  here will not win from a handful of corrections, so a negative result under
  this regime is the strong form of the negative result.
* **The incumbent is 0.5**, the constant the six architectures hard-code -- not
  each benchmark's tuned operating point. The question is about the *default*,
  so the baseline has to be the default.
* **Two held-out F1s, both reported.** ``held_out_f1_blocked`` restricts gold to
  pairs blocking actually proposed -- the same convention ``fit``'s own
  ``held_out_f1`` uses (its gold comes from the aligned valid candidates), so it
  is the like-for-like number and the primary one. ``held_out_f1_all_gold``
  charges every gold pair blocking missed to recall -- the end-to-end number.
  Both are computed with ``metrics.classify_pairs``, the function ``fit`` uses.
* **Multiple seeds.** A conclusion that flips between seeds is not a conclusion.

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

from langres.core.models import PairwiseJudgement  # noqa: E402
from langres.curation.harvest import LabeledPair, align_pairs  # noqa: E402
from langres.data.benchmark import Benchmark, gold_pairs_from_clusters  # noqa: E402
from langres.eval import get_benchmark, list_benchmarks  # noqa: E402
from langres.methods import BlockingBenchmark, make_resolver_factory  # noqa: E402
from langres.metrics.metrics import classify_pairs  # noqa: E402

logger = logging.getLogger("threshold_default_study")

#: The $0 scorers this study races. ``rapidfuzz`` stands in for the string path
#: (``FuzzyString``), ``embedding_cosine`` for the vector path
#: (``architectures/retrieval.py``) -- the two families the six hard-coded
#: ``0.5``s sit on. Both are rankers, so the cut is the whole decision.
DEFAULT_METHODS: tuple[str, ...] = ("rapidfuzz", "embedding_cosine")

#: The constant the six architectures hard-code, and therefore the incumbent
#: this study races the derived cut against.
INCUMBENT_THRESHOLD = 0.5

#: Split seeds. More than one on purpose: a flip between seeds is decision-relevant.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Small, in-repo datasets for a quick pass (``--fast``).
FAST_SUBSET: frozenset[str] = frozenset({"tiny_fixture", "fodors_zagat", "dblp_acm"})

#: The tracked artifact a FULL run refreshes.
CANONICAL_OUT = Path("examples/research/results/threshold_default_study.json")


class _StudyBenchmark(Benchmark[Any], BlockingBenchmark, Protocol):
    """A benchmark satisfying both the harness and the method-registry contracts."""


class CellResult(BaseModel):
    """One (benchmark, method, seed) fit + its disjoint-corpus grading.

    Attributes:
        benchmark: Registry name.
        method: The $0 scorer.
        seed: Seed for the benchmark's own cluster-disjoint corpus split.
        n_train_records / n_test_records: The two disjoint corpora.
        n_train_pairs: Labeled blocked candidates the race derived + selected on.
        n_test_pairs: Blocked candidates on the held-out corpus (the grading set).
        align_split_train / align_split_valid: what ``align_pairs(split=0.3)``
            would have held out of ``n_train_pairs`` -- the measurement behind
            "the split trap". Not used for any number above; recorded because a
            design choice justified by a degeneracy should carry the degeneracy
            as data rather than as a claim. ``align_split_valid`` near 0 (or
            near ``n_train_pairs``, i.e. inverted) is the trap firing.
        n_test_gold_blocked: Gold pairs among ``n_test_pairs`` (0 makes F1 vacuous).
        incumbent_threshold: The 0.5 default.
        derived_threshold: Youden's J on the train corpus's labeled pairs.
        incumbent_selection_f1 / derived_selection_f1: The ``train`` pair-F1s the
            race itself compared. Selection never reads the held-out corpus.
        kept: ``"derived"`` (candidate won) or ``"declined"`` (incumbent held).
        final_threshold: The cut in force on the model after ``fit`` returned --
            the number a user would actually resolve with.
        incumbent_f1_blocked / derived_f1_blocked / kept_f1_blocked: Held-out
            pair-F1 on the disjoint corpus with gold restricted to blocked pairs
            (``fit``'s own ``held_out_f1`` convention). The primary numbers.
        incumbent_f1_all_gold / derived_f1_all_gold / kept_f1_all_gold: the same,
            charging blocking's misses to recall (the end-to-end view).
        delta_f1_blocked / delta_f1_all_gold: ``kept - incumbent``. This is the
            effect of flipping the default: exactly ``0.0`` whenever the race
            declined, and NEGATIVE only when the race kept a cut that lost.
        seconds: Wall clock for the cell.
    """

    benchmark: str
    method: str
    seed: int
    n_train_records: int
    n_test_records: int
    n_train_pairs: int
    n_test_pairs: int
    align_split_train: int
    align_split_valid: int
    n_test_gold_blocked: int
    incumbent_threshold: float
    derived_threshold: float
    incumbent_selection_f1: float | None
    derived_selection_f1: float | None
    kept: str
    final_threshold: float
    incumbent_f1_blocked: float
    derived_f1_blocked: float
    kept_f1_blocked: float
    incumbent_f1_all_gold: float
    derived_f1_all_gold: float
    kept_f1_all_gold: float
    delta_f1_blocked: float
    delta_f1_all_gold: float
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


def _pair_f1(
    judgements: list[PairwiseJudgement], gold: set[frozenset[str]], threshold: float
) -> float:
    """Pair-F1 at ``threshold`` via ``classify_pairs`` -- the function ``fit`` uses."""
    return classify_pairs(judgements, gold, threshold).f1


def run_cell(
    *,
    benchmark: str,
    method: str,
    seed: int,
    bench: _StudyBenchmark,
    corpus: list[Any],
    gold_clusters: list[set[str]],
    gold_pairs: set[frozenset[str]],
) -> CellResult:
    """Derive on the train corpus, grade both cuts on the disjoint test corpus.

    Args:
        benchmark: Registry name.
        method: The $0 scorer.
        seed: Seed for ``bench.split``.
        bench: The benchmark adapter.
        corpus: The full record list.
        gold_clusters: The closed-world gold partition.
        gold_pairs: The closed-world gold pair set.

    Returns:
        The assembled :class:`CellResult`.

    Raises:
        RuntimeError: If ``fit`` reported no threshold race (nothing to measure),
            or if the held-out corpus contains no blocked gold pair (every F1
            below would be a vacuous 0.0 that a reader would mistake for a
            measurement).
    """
    started = time.monotonic()
    train_records, test_records, _train_clusters, test_clusters = bench.split(
        corpus, gold_clusters, seed=seed
    )
    train_data = [record.model_dump() for record in train_records]
    test_data = [record.model_dump() for record in test_records]

    resolver = make_resolver_factory(method, bench)(INCUMBENT_THRESHOLD)

    # 1. Derive + race, on the train corpus only. split=None because the held-out
    #    estimate comes from the DISJOINT CORPUS below, not from a slice of this
    #    label set -- see the module docstring's "split trap".
    train_candidates = resolver.candidates(train_data)
    train_labels = gold_labels_for_candidates(train_candidates, gold_pairs)
    # The split trap, measured rather than asserted: what fit(split=0.3) WOULD
    # have held out of this same label set. Free here (the candidates are already
    # blocked), and it is the evidence for grading on a disjoint corpus instead.
    trap = align_pairs(train_candidates, train_labels, split=0.3, seed=seed)
    align_split_train, align_split_valid = len(trap.train.labels), len(trap.valid.labels)
    del train_candidates, trap
    resolver.fit(train_data, pairs=train_labels, split=None, seed=seed, derive_threshold=True)
    fit = resolver.fit_report_.threshold_fit
    if fit is None or fit.previous is None or fit.candidate is None:
        raise RuntimeError(
            f"{benchmark}/{method}/seed={seed}: fit reported no threshold race "
            f"(threshold_fit={fit!r}); there is nothing to measure."
        )
    kept = "derived" if fit.source == "derived" else "declined"

    # 2. Grade BOTH cuts on the held-out corpus, scored once.
    judgements = resolver.predict(test_data)
    blocked = {frozenset({j.left_id, j.right_id}) for j in judgements}
    test_gold_all = gold_pairs_from_clusters(test_clusters)
    test_gold_blocked = test_gold_all & blocked
    if not test_gold_blocked:
        raise RuntimeError(
            f"{benchmark}/{method}/seed={seed}: the held-out corpus has no blocked "
            f"gold pair ({len(test_gold_all)} gold, {len(blocked)} blocked), so every "
            "F1 would be a vacuous 0.0. Refusing to report it as a measurement."
        )

    incumbent, derived = fit.previous.threshold, fit.candidate.threshold
    final = float(resolver.clusterer.threshold)
    f1_blocked = {t: _pair_f1(judgements, test_gold_blocked, t) for t in (incumbent, derived)}
    f1_all = {t: _pair_f1(judgements, test_gold_all, t) for t in (incumbent, derived)}
    kept_t = derived if kept == "derived" else incumbent

    return CellResult(
        benchmark=benchmark,
        method=method,
        seed=seed,
        n_train_records=len(train_records),
        n_test_records=len(test_records),
        n_train_pairs=len(train_labels),
        n_test_pairs=len(judgements),
        align_split_train=align_split_train,
        align_split_valid=align_split_valid,
        n_test_gold_blocked=len(test_gold_blocked),
        incumbent_threshold=incumbent,
        derived_threshold=derived,
        incumbent_selection_f1=fit.previous.selection_f1,
        derived_selection_f1=fit.candidate.selection_f1,
        kept=kept,
        final_threshold=final,
        incumbent_f1_blocked=f1_blocked[incumbent],
        derived_f1_blocked=f1_blocked[derived],
        kept_f1_blocked=f1_blocked[kept_t],
        incumbent_f1_all_gold=f1_all[incumbent],
        derived_f1_all_gold=f1_all[derived],
        kept_f1_all_gold=f1_all[kept_t],
        delta_f1_blocked=f1_blocked[kept_t] - f1_blocked[incumbent],
        delta_f1_all_gold=f1_all[kept_t] - f1_all[incumbent],
        seconds=time.monotonic() - started,
    )


def run_benchmark(
    name: str, *, methods: tuple[str, ...], seeds: tuple[int, ...]
) -> list[CellResult]:
    """Run every (method, seed) cell for one registered benchmark.

    Args:
        name: Registry name (must be ``loadable``).
        methods: $0 method names.
        seeds: Corpus-split seeds.

    Returns:
        One :class:`CellResult` per (method, seed) that completed.
    """
    bench = cast(_StudyBenchmark, get_benchmark(name))
    corpus, gold_clusters, gold_pairs = bench.load()

    results: list[CellResult] = []
    for method in methods:
        for seed in seeds:
            result = run_cell(
                benchmark=name,
                method=method,
                seed=seed,
                bench=bench,
                corpus=corpus,
                gold_clusters=gold_clusters,
                gold_pairs=gold_pairs,
            )
            results.append(result)
            print(
                f"       {method} seed={seed}: "
                f"incumbent {result.incumbent_threshold:.2f} "
                f"F1={result.incumbent_f1_blocked:.4f} | "
                f"derived {result.derived_threshold:.4f} "
                f"F1={result.derived_f1_blocked:.4f} | kept={result.kept} "
                f"Δ={result.delta_f1_blocked:+.4f} ({result.seconds:.1f}s)",
                flush=True,
            )
    return results


def to_markdown(results: list[CellResult]) -> str:
    """Render the per-cell table.

    Never an average: an average hides the single cell that motivated the race.
    """
    header = (
        "| benchmark | method | seed | train pairs | test pairs | held-out gold | "
        "incumbent t | incumbent F1 | derived t | derived F1 | kept | Δ F1 |"
    )
    lines = [header, "|" + "---|" * 12]
    for r in results:
        lines.append(
            f"| {r.benchmark} | {r.method} | {r.seed} | {r.n_train_pairs:,} | "
            f"{r.n_test_pairs:,} | {r.n_test_gold_blocked:,} | "
            f"{r.incumbent_threshold:.2f} | "
            f"{r.incumbent_f1_blocked:.4f} | {r.derived_threshold:.4f} | "
            f"{r.derived_f1_blocked:.4f} | {r.kept} | {r.delta_f1_blocked:+.4f} |"
        )
    return "\n".join(lines)


def to_split_trap_markdown(results: list[CellResult]) -> str:
    """Render the split-trap table: what ``align_pairs(split=0.3)`` would hold out.

    One row per (benchmark, seed) -- the split reads only the labeled candidate
    graph, which is the same for both methods (same blocker, same k), so the
    method column would be a duplicated row.
    """
    header = "| benchmark | seed | labeled pairs | align train | align valid | valid share (want ~0.30) |"
    lines = [header, "|" + "---|" * 6]
    seen: set[tuple[str, int]] = set()
    for r in results:
        key = (r.benchmark, r.seed)
        if key in seen:
            continue
        seen.add(key)
        total = r.align_split_train + r.align_split_valid
        share = "n/a" if total == 0 else f"{r.align_split_valid / total:.4f}"
        lines.append(
            f"| {r.benchmark} | {r.seed} | {total:,} | {r.align_split_train:,} | "
            f"{r.align_split_valid:,} | {share} |"
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
    out.write_text(json.dumps([r.model_dump() for r in results], indent=2, sort_keys=True) + "\n")


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
                run_benchmark(name, methods=tuple(args.methods), seeds=tuple(args.seeds))
            )
        except Exception as exc:  # noqa: BLE001 - a broken loader must not kill the sweep
            # ``exception`` (not ``error``): a benchmark that drops out is absent
            # from both the table and the tracked JSON, so the traceback in the
            # run log is the ONLY record of why.
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
    print("The split trap -- what fit(pairs=..., split=0.3) would have held out:")
    print(to_split_trap_markdown(results))
    print()
    print(f"[out] {out}")
    if failures:
        print(f"[fail] {len(failures)} benchmark(s) did not complete: {', '.join(failures)}")


if __name__ == "__main__":
    main()
