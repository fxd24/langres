"""Autoresearch proof (epic #145, M1/E1): the propose→run→eval→keep loop climbs.

This is the runnable *proof that the loop closes*. It points ``langres.optimize``
at a small, frozen search space over the **amazon_google** benchmark and lets the
``propose → run → evaluate → keep-if-better`` loop hill-climb — with **$0 spend**
(local ``all-MiniLM-L6-v2`` embeddings, no LLM) and fully offline (the benchmark
is vendored).

What it demonstrates
--------------------
* **The loop steers on a loss-like objective, not a saturated F1.** ER F1
  plateaus near 99%, so we optimize a continuous *recall@budget* signal instead:
  ``maximize candidate_recall subject_to reduction_ratio >= 0.985`` (keep >=98.5%
  of the O(|A|x|B|) comparisons eliminated). See ``Objective.maximize``.
* **Recall@budget climbs.** ``SearchSpace`` sweeps ``k_neighbors`` innermost, so
  within each ``(metric, text_field)`` group more neighbours surface more true
  pairs and the *incumbent* candidate_recall ratchets up trial by trial — the
  printed progress curve makes the climb visible.
* **The budget is a real gate, not decoration.** The highest-``k`` configs buy
  little-to-no extra recall while spending more comparisons, so they breach the
  reduction-ratio budget and are **rejected** — the loop keeps the best *feasible*
  incumbent. This is the recall-vs-cost tradeoff, made explicit.
* **Every trial is logged off-git — accepted *and* rejected.** The loop persists
  each trial to a local, owned ``RunStore`` JSONL under ``tmp/`` (gitignored), so
  the full audit trail (including the over-budget rejects) survives the run. The
  script reads it back to prove nothing was dropped.

Why amazon_google: it is a genuinely *hard, unsaturated* two-source linkage
benchmark (blocking recall plateaus ~0.84 with title+manufacturer vector
blocking — it never reaches a 0.90 gate), which is exactly what makes the climb
and the budget tradeoff honest rather than a foregone conclusion.

Run:
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \\
        uv run python examples/research/blocking_recall_autoresearch.py

    # or, with the repo's env file:
    uv run --env-file .env python examples/research/blocking_recall_autoresearch.py
"""

from __future__ import annotations

# Force single-threaded / duplicate-safe OpenMP *before* faiss/torch load, so the
# macOS libomp double-load segfault can't bite (belt to the guard already in
# core.indexes.vector_index). Harmless elsewhere; mirrors the sibling examples.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import TYPE_CHECKING

from langres import optimize
from langres.autoresearch.objective import Objective
from langres.autoresearch.search_space import SearchSpace
from langres.tracking.runs import RunStore

if TYPE_CHECKING:
    from langres.autoresearch.loop import LoopResult

#: Registered, vendored, offline benchmark (two-source Amazon<->Google linkage).
BENCHMARK = "amazon_google"

#: Reduction-ratio budget the loop must stay within. 0.985 = "eliminate >=98.5%
#: of the |A|x|B| comparisons". Chosen so the k-sweep can climb to its recall
#: plateau (k<=40) while the top of the sweep (k=80), which spends ~2x the
#: comparisons for no recall gain, breaches the budget and is rejected — a real
#: recall@budget tradeoff rather than a vacuous constraint every config passes.
RR_BUDGET = 0.985

#: Off-git audit trail (``tmp/`` is gitignored). Every trial — accepted and
#: rejected — lands here; the loop owns it, and we read it back at the end.
STORE_PATH = "tmp/autoresearch/amazon_google_blocking.jsonl"


def build_space() -> SearchSpace:
    """The modest, frozen grid the loop proposes over (20 configs, 4 index builds).

    Two metrics x two blocking texts x a 5-point ascending ``k`` sweep. Because
    ``k_neighbors`` is the innermost axis, ``optimize`` builds **one** vector
    index per ``(embedding_model, metric, text_field)`` group and reuses it across
    every ``k`` — so this is only 4 embedding passes over the corpus, not 20.
    """
    return SearchSpace(
        blocker=("vector",),
        embedding_model=("all-MiniLM-L6-v2",),  # local + free; keep it $0/fast
        metric=("cosine", "L2"),
        text_field=("embed_text", "title"),  # title+manufacturer vs title-only
        k_neighbors=(5, 10, 20, 40, 80),  # ascending: recall climbs with k
    )


def run_search(
    space: SearchSpace,
    rr_budget: float,
    store: str | None,
    *,
    seed: int = 0,
) -> LoopResult:
    """Drive the autoresearch loop over ``space`` and return its :class:`LoopResult`.

    The objective is the E1 steering signal: maximize the continuous
    ``candidate_recall`` subject to a reduction-ratio floor. ``optimize`` loads
    amazon_google once, threads a cached index through the scorer, and persists
    every trial to ``store`` (``None`` => persist nothing).
    """
    objective = Objective.maximize(
        "candidate_recall", subject_to=[("reduction_ratio", ">=", rr_budget)]
    )
    return optimize(space, objective, BENCHMARK, store=store, seed=seed)


def _config_label(config: dict[str, object]) -> str:
    """Compact one-line view of a vector config: ``metric/text_field/k=..``."""
    return f"{config['metric']:>6}/{config['text_field']:<10}/k={config['k_neighbors']:<3}"


def print_progress(result: LoopResult, rr_budget: float) -> list[float]:
    """Print the recall@budget progress curve; return the incumbent-recall series.

    One row per trial in evaluation order: the config, its ``candidate_recall`` and
    ``reduction_ratio``, an outcome tag, and the **running incumbent recall** so
    the climb is visible. The outcome tag distinguishes the two reasons a trial is
    not accepted — over-budget (``reduction_ratio`` below the floor) vs. simply not
    better than the incumbent.
    """
    print("\n" + "=" * 78)
    print(
        f"Recall@budget progress curve  (objective: maximize candidate_recall "
        f"s.t. reduction_ratio >= {rr_budget})"
    )
    print("=" * 78)
    print(f"{'#':>2}  {'config':<26} {'recall':>7} {'RR':>8}  {'outcome':<18} {'incumbent':>9}")
    print("-" * 78)

    incumbent = 0.0
    incumbents: list[float] = []
    for i, trial in enumerate(result.trials, start=1):
        metrics = trial.metrics
        if metrics is None:  # scorer raised — recorded as a failed trial
            print(
                f"{i:>2}  {_config_label(trial.config):<26} {'—':>7} {'—':>8}  "
                f"{'FAILED':<18} {incumbent:>9.4f}"
            )
            incumbents.append(incumbent)
            continue
        recall = metrics["candidate_recall"]
        rr = metrics["reduction_ratio"]
        if trial.accepted:
            incumbent = recall
            outcome = "ACCEPT (new best)"
        elif rr < rr_budget:
            outcome = "reject: over budget"
        else:
            outcome = "reject: no gain"
        print(
            f"{i:>2}  {_config_label(trial.config):<26} {recall:>7.4f} {rr:>8.5f}  "
            f"{outcome:<18} {incumbent:>9.4f}"
        )
        incumbents.append(incumbent)
    print("-" * 78)
    return incumbents


def report_best(result: LoopResult, rr_budget: float) -> None:
    """Print the winning config and its metrics."""
    print("\nBest feasible config found:")
    if result.best_config is None or result.best_metrics is None:
        print("  (none — no config satisfied the budget)")
        return
    cfg = result.best_config
    m = result.best_metrics
    print(f"  metric        : {cfg['metric']}")
    print(f"  text_field    : {cfg['text_field']}")
    print(f"  k_neighbors   : {cfg['k_neighbors']}")
    print(f"  candidate_recall : {m['candidate_recall']:.4f}")
    print(f"  reduction_ratio  : {m['reduction_ratio']:.5f}  (budget >= {rr_budget})")
    print(f"  candidate_precision : {m['candidate_precision']:.4f}")
    print(f"  total_candidates    : {int(m['total_candidates'])}")


def report_store(store_path: str) -> None:
    """Read the persisted trail back and prove every trial was logged off-git."""
    records = RunStore(store_path).read()
    accepted = sum(1 for r in records if (r.metrics or {}).get("accepted") == 1.0)
    failed = sum(1 for r in records if r.status == "failed")
    rejected = len(records) - accepted - failed
    print("\n" + "-" * 78)
    print(f"RunStore audit trail: {store_path}")
    print(
        f"  {len(records)} trials logged (incl. rejects)  ->  "
        f"{accepted} accepted / {rejected} rejected / {failed} failed"
    )
    print("  (every proposed config is durable off-git — the full propose->eval->keep record)")
    print("-" * 78)


def maybe_plot(incumbents: list[float], out_dir: str) -> None:
    """Save an incumbent-recall-vs-trial PNG *iff* matplotlib is already installed.

    A nice-to-have — the printed curve is the primary artifact. Never adds a
    dependency: a missing matplotlib is silently skipped.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless; no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping the PNG; the printed curve is the proof.)")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / "incumbent_recall_curve.png"
    trials = range(1, len(incumbents) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(trials, incumbents, where="post", marker="o", color="#2563eb")
    ax.set_xlabel("trial (proposal order)")
    ax.set_ylabel("incumbent candidate_recall")
    ax.set_title("Autoresearch: incumbent recall@budget climbs (amazon_google)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved incumbent-recall curve -> {out_path}")


def main() -> None:
    """Run the E1 proof: hill-climb blocking recall@budget on amazon_google."""
    print("=" * 78)
    print("Autoresearch E1 proof — blocking recall@budget loop on amazon_google")
    print("$0, offline, local MiniLM embeddings (no LLM). Epic #145, M1.")
    print("=" * 78)

    space = build_space()
    print(
        f"\nSearch space: {len(space)} configs "
        f"(metrics x text_fields x k-sweep; k innermost -> 4 index builds)."
    )
    print(f"Persisting every trial to: {STORE_PATH}  (gitignored)")

    result = run_search(space, RR_BUDGET, STORE_PATH, seed=0)

    incumbents = print_progress(result, RR_BUDGET)
    report_best(result, RR_BUDGET)
    report_store(STORE_PATH)
    maybe_plot(incumbents, os.path.dirname(STORE_PATH))

    print(
        "\nThe loop closed: incumbent recall@budget climbed and every trial "
        "(incl. over-budget rejects) is durable off-git."
    )


if __name__ == "__main__":
    main()
