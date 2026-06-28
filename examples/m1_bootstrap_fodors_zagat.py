"""M1 cold-start bootstrapping, end-to-end on Fodors-Zagat (deterministic, free).

This is the runnable proof that the M1 Wave 1-4 pieces compose into one pass:
real semantic blocking (sentence-transformer embeddings + FAISS) -> cross-source
filter -> stratified hard-negative mining -> labeling -> a labeled
:class:`~langres.bootstrap.models.GoldSet` plus a
:class:`~langres.bootstrap.report.BootstrapReport` (blocking pair-completeness,
teacher-vs-truth agreement, calibration), saved to disk and round-tripped.

The default run is **deterministic and costs $0**: it labels with
:class:`~langres.bootstrap.labelers.FakeLabeler`, a similarity-threshold stand-in
for the real teacher whose over-confident confidence keeps the calibration report
non-trivial. Embeddings are real (so blocking pair-completeness is honest) but
involve no network and no API key.

A gated real-GLM branch (``maybe_real_teacher``) shows how Wave 5 swaps in a
budget-capped :class:`~langres.bootstrap.labelers.TeacherLabeler`; it only runs
when ``OPENROUTER_API_KEY`` is set and never costs anything otherwise.

Run it::

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=1 uv run python examples/m1_bootstrap_fodors_zagat.py

``print`` is allowed in examples (this is demonstration, not library code).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langres.bootstrap import (
    BootstrapReport,
    Bootstrapper,
    FakeLabeler,
    GoldSet,
    HardNegativeMiner,
)
from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.data.er_benchmarks import (
    DEFAULT_BLOCKING_K,
    RestaurantSchema,
    load_fodors_zagat,
)


def _cross_source(candidate: Any) -> bool:
    """Keep only candidate pairs whose two records come from different sources.

    The Fodors-Zagat task is a cross-source linkage: intra-source pairs are noise
    (design-review B2). Passed IN to ``Bootstrapper.build`` so the orchestrator
    stays entity-type-agnostic.
    """
    return bool(candidate.left.source != candidate.right.source)


def build_bootstrapper(k_neighbors: int = DEFAULT_BLOCKING_K) -> Bootstrapper:
    """Wire a real-embedding VectorBlocker + miner + FakeLabeler into a Bootstrapper.

    Args:
        k_neighbors: Nearest neighbours per record for the vector blocker
            (``DEFAULT_BLOCKING_K`` clears the >= 0.95 pair-completeness gate).

    Returns:
        A ready-to-run :class:`Bootstrapper` (deterministic, zero-spend labeler).
    """
    blocker: VectorBlocker[RestaurantSchema] = VectorBlocker(
        vector_index=FAISSIndex(
            embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
            metric="cosine",
        ),
        schema=RestaurantSchema,
        text_field="embed_text",
        k_neighbors=k_neighbors,
    )
    miner = HardNegativeMiner(seed=0)
    # MiniLM cosine sims for these cross-source pairs cluster high (~0.78-0.97);
    # 0.9 splits the true matches (which sit at the very top) from the rest,
    # giving the report a realistic, non-degenerate teacher (TP/FP/FN/TN all
    # non-zero) rather than an all-positive collapse.
    labeler = FakeLabeler(threshold=0.9)
    return Bootstrapper(blocker, miner, labeler)


def run_bootstrap(save_dir: Path, *, k_neighbors: int = DEFAULT_BLOCKING_K) -> dict[str, Any]:
    """Run the deterministic bootstrap end-to-end and round-trip the gold set.

    Loads Fodors-Zagat, runs the bootstrapper with the cross-source filter and
    the benchmark ground truth, saves the gold set to ``save_dir``, reloads it,
    and returns everything the demo prints and the test asserts on.

    Args:
        save_dir: Directory to write ``gold_set.json`` into (created if absent).
        k_neighbors: Blocker neighbour count.

    Returns:
        A dict with the gold set, the reloaded gold set, the report, and the
        round-trip flag.
    """
    corpus_records, gold_clusters = load_fodors_zagat()
    # The blocker's declarative schema_factory consumes plain dicts (same shape
    # the data adapter's k-sweep feeds stream()).
    corpus = [r.model_dump() for r in corpus_records]

    bootstrapper = build_bootstrapper(k_neighbors)
    gold, report = bootstrapper.build(
        corpus, candidate_filter=_cross_source, gold_clusters=gold_clusters
    )

    # Save -> reload (round-trip proof).
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "gold_set.json"
    gold.save(path)
    reloaded = GoldSet.load(path)
    roundtrip_ok = reloaded.model_dump() == gold.model_dump()

    return {
        "gold": gold,
        "reloaded": reloaded,
        "report": report,
        "roundtrip_ok": roundtrip_ok,
        "gold_path": path,
    }


def maybe_real_teacher() -> None:
    """Show how Wave 5 swaps in the budget-capped real teacher (gated on a key).

    Stubbed on purpose: it only constructs a :class:`TeacherLabeler` (no labeling
    run) and prints what Wave 5 will finish. It never runs or costs anything
    without ``OPENROUTER_API_KEY``.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n[real teacher] OPENROUTER_API_KEY not set — skipping (default run is free).")
        return

    from langres.bootstrap import TeacherLabeler

    # Wave 5 fills the pinned GLM model + prices and runs this for real. The
    # prices below are positive placeholders (NOT validated rates) so the branch
    # constructs cleanly; Wave 5 pins them from the provider's published rates
    # before any real labeling run.
    teacher = TeacherLabeler.from_env(
        price_per_1m_prompt_tokens=1.0,  # TODO(Wave 5): pin real GLM prompt price
        price_per_1m_completion_tokens=1.0,  # TODO(Wave 5): pin real GLM completion price
        worst_case_tokens_per_pair=2000,
        model="gpt-5-mini",  # TODO(Wave 5): pin the GLM model id
        entity_noun="restaurant",
        budget_usd=5.0,
        budget_soft_usd=4.0,
    )
    print(
        "\n[real teacher] Constructed a budget-capped TeacherLabeler "
        f"(model={teacher._judge.model!r}, budget=${teacher.budget_usd:.2f}). "
        "Wave 5 pins the GLM model/prices and runs build() with it for real."
    )


def main() -> None:
    import logging

    logging.getLogger("langres").setLevel(logging.ERROR)

    print("=" * 78)
    print("M1 cold-start bootstrapping — Fodors-Zagat (real embeddings, $0, deterministic)")
    print("=" * 78)

    results = run_bootstrap(Path("tmp"))
    gold: GoldSet = results["gold"]
    report: BootstrapReport = results["report"]

    pc = report.blocking.pair_completeness
    print(
        f"\nBlocking Pair-Completeness (cross-source recall): {pc:.4f}"
        f"  [target >= 0.95]  {'PASS' if pc >= 0.95 else 'LOW'}"
    )
    print(
        f"  candidates={report.blocking.total_candidates}  "
        f"missed={report.blocking.missed_matches}  "
        f"precision={report.blocking.candidate_precision:.4f}"
    )

    if report.agreement is not None:
        a = report.agreement
        print("\nTeacher(fake)-vs-truth agreement:")
        print(
            f"  acc={a.accuracy:.4f}  F1={a.f1:.4f}  "
            f"kappa={a.cohens_kappa:.4f}  MCC={a.mcc:.4f}  (n={a.n_evaluated})"
        )
    if report.calibration is not None:
        c = report.calibration
        print(
            f"\nCalibration: Brier={c.brier:.4f}  ECE={c.ece:.4f}  "
            f"(n={c.n_evaluated}, {c.n_bins} bins)"
        )

    cov = report.coverage
    matches = gold.metadata["matches"]
    print(
        f"\nLabels: {cov.labeled} total ({matches} match / "
        f"{gold.metadata['non_matches']} non-match)  "
        f"mined={cov.mined}  cost=${cov.total_cost_usd:.4f} (fake = free)"
    )

    print(f"\nGold set saved to {results['gold_path']} and reloaded.")
    assert results["roundtrip_ok"], "Gold-set round-trip differs from the original!"
    print("Save/reload round-trip: gold set is IDENTICAL ✓")

    maybe_real_teacher()

    print("\n" + "=" * 78)
    print("Bootstrap ran end-to-end. ✓")
    print("=" * 78)


if __name__ == "__main__":
    main()
