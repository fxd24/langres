"""W1.3 blocking-algebra + clusterer benchmark (Fodors-Zagat + Amazon-Google).

Zero-spend ($0, no LLM calls anywhere): blocking is KeyBlocker/VectorBlocker,
scoring is WeightedAverageMatcher. Two independent measurements:

1. **Composite blocking.** Does UNIONing a cheap, exact-key ``KeyBlocker``
   with each dataset's existing pinned ``VectorBlocker`` improve
   Pair-Completeness (PC = recall of the blocking stage) over the
   ``VectorBlocker`` alone, and at what Reduction-Ratio (RR) cost? RR is
   measured against the cross-source product (both benchmarks are linkage
   tasks whose true matches are all cross-source; see
   ``langres.data._benchmark_utils.cross_source``), matching the
   already-established PC methodology in ``er_benchmarks.py``/
   ``amazon_google.py`` (``ACHIEVED_PC_AT_DEFAULT_K`` etc.) so these numbers
   are directly comparable to the pinned baselines.

2. **C6 (CorrelationClusterer) vs the base Clusterer.** Head-to-head BCubed
   P/R/F1 on the SAME blocking + judge pipeline for both -- only the
   clusterer differs -- to isolate the over-merge fix's effect. The threshold
   (0.80) reuses the value already tuned for WeightedAverageMatcher in
   ``examples/research/m3_zero_spend_race_output.md``, so the base-Clusterer row here
   should reproduce those numbers.

Reproduce with:
    uv run python examples/research/w1_blocking_algebra.py
"""

import logging
from typing import Any

from pydantic import BaseModel

from langres.core.benchmark import complete_partition
from langres.core.blockers.composite import CompositeBlocker
from langres.core.blockers.key import KeyBlocker
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.clusterers.correlation import CorrelationClusterer
from langres.core.comparator import Comparator
from langres.core.comparators import StringComparator
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.metrics import calculate_bcubed_metrics, evaluate_blocking
from langres.core.models import ERCandidate
from langres.data import _benchmark_utils as _bu
from langres.data.amazon_google import (
    DEFAULT_AG_BLOCKING_K,
    ProductSchema,
    build_product_blocker,
    load_amazon_google,
)
from langres.data.er_benchmarks import (
    DEFAULT_BLOCKING_K,
    RestaurantSchema,
    build_restaurant_blocker,
    load_fodors_zagat,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Reused, not re-tuned: the WeightedAverageMatcher threshold already established
# in examples/research/m3_zero_spend_race_output.md (both FZ and AG).
THRESHOLD = 0.80


def _build_vector_candidates(
    blocker: VectorBlocker[Any], corpus: list[Any], records: list[dict[str, Any]]
) -> list[ERCandidate[Any]]:
    """Build a VectorBlocker's FAISS index over ``embed_text`` and stream it."""
    texts = [r.embed_text for r in corpus]
    blocker.vector_index.create_index(texts)
    return list(blocker.stream(records))


def _pc_rr(
    candidates: list[ERCandidate[Any]],
    gold_clusters: list[set[str]],
    cross_source_denominator: int,
) -> tuple[float, float, int]:
    """Cross-source Pair-Completeness + Reduction-Ratio for one candidate set."""
    filtered = _bu.cross_source(candidates)
    pc = evaluate_blocking(filtered, gold_clusters).candidate_recall
    rr = 1.0 - (len(filtered) / cross_source_denominator)
    return pc, rr, len(filtered)


def run_blocking_comparison() -> list[dict[str, Any]]:
    """Composite (KeyBlocker UNION VectorBlocker) vs VectorBlocker-alone PC/RR."""
    rows: list[dict[str, Any]] = []

    # --- Fodors-Zagat: block by normalized city ---
    fz_corpus, fz_gold = load_fodors_zagat()
    fz_records = [r.model_dump() for r in fz_corpus]
    n_fodors = sum(1 for r in fz_corpus if r.source == "fodors")
    n_zagat = sum(1 for r in fz_corpus if r.source == "zagat")
    fz_denom = n_fodors * n_zagat

    fz_vector = build_restaurant_blocker(DEFAULT_BLOCKING_K)
    fz_vector_candidates = _build_vector_candidates(fz_vector, fz_corpus, fz_records)

    # Reuse fz_vector's already-built index (same k, same corpus) as the
    # composite's second child -- avoids a redundant embedding pass.
    fz_key = KeyBlocker(schema=RestaurantSchema, key_field="city")
    fz_composite = CompositeBlocker(children=[fz_key, fz_vector], op="union")
    fz_composite_candidates = list(fz_composite.stream(fz_records))

    for label, cands in [
        (f"VectorBlocker only (k={DEFAULT_BLOCKING_K}, pinned)", fz_vector_candidates),
        ("KeyBlocker(city) UNION VectorBlocker", fz_composite_candidates),
    ]:
        pc, rr, n = _pc_rr(cands, fz_gold, fz_denom)
        rows.append(
            {"dataset": "fodors_zagat", "blocker": label, "n_candidates": n, "pc": pc, "rr": rr}
        )
        logger.info("fodors_zagat | %-42s | n=%5d | PC=%.4f | RR=%.4f", label, n, pc, rr)

    # --- Amazon-Google: block by normalized manufacturer ---
    ag_corpus, ag_gold, _ag_pairs = load_amazon_google()
    ag_records = [r.model_dump() for r in ag_corpus]
    n_amazon = sum(1 for r in ag_corpus if r.source == "amazon")
    n_google = sum(1 for r in ag_corpus if r.source == "google")
    ag_denom = n_amazon * n_google

    ag_vector = build_product_blocker(DEFAULT_AG_BLOCKING_K)
    ag_vector_candidates = _build_vector_candidates(ag_vector, ag_corpus, ag_records)

    ag_key = KeyBlocker(schema=ProductSchema, key_field="manufacturer")
    ag_composite = CompositeBlocker(children=[ag_key, ag_vector], op="union")
    ag_composite_candidates = list(ag_composite.stream(ag_records))

    for label, cands in [
        (f"VectorBlocker only (k={DEFAULT_AG_BLOCKING_K}, pinned)", ag_vector_candidates),
        ("KeyBlocker(manufacturer) UNION VectorBlocker", ag_composite_candidates),
    ]:
        pc, rr, n = _pc_rr(cands, ag_gold, ag_denom)
        rows.append(
            {"dataset": "amazon_google", "blocker": label, "n_candidates": n, "pc": pc, "rr": rr}
        )
        logger.info("amazon_google | %-42s | n=%5d | PC=%.4f | RR=%.4f", label, n, pc, rr)

    return rows


def _judged_candidates(
    blocker: VectorBlocker[Any], corpus: list[Any], schema: type[BaseModel]
) -> list[Any]:
    """Block + attach comparison + score with WeightedAverageMatcher (no clustering)."""
    records = [r.model_dump() for r in corpus]
    candidates = _build_vector_candidates(blocker, corpus, records)

    comparator: Comparator[Any] = StringComparator.from_schema(schema)
    judge: WeightedAverageMatcher[Any] = WeightedAverageMatcher(
        feature_specs=comparator.feature_specs
    )

    compared = [
        c.model_copy(update={"comparison": comparator.compare(c.left, c.right)}) for c in candidates
    ]
    return list(judge.forward(iter(compared)))


def run_clusterer_comparison() -> list[dict[str, Any]]:
    """CorrelationClusterer (C6) vs the base Clusterer, same pipeline otherwise."""
    rows: list[dict[str, Any]] = []

    fz_corpus, fz_gold = load_fodors_zagat()
    ag_corpus, ag_gold, _ = load_amazon_google()

    datasets: list[tuple[str, list[Any], list[set[str]], VectorBlocker[Any], type[BaseModel]]] = [
        (
            "fodors_zagat",
            fz_corpus,
            fz_gold,
            build_restaurant_blocker(DEFAULT_BLOCKING_K),
            RestaurantSchema,
        ),
        (
            "amazon_google",
            ag_corpus,
            ag_gold,
            build_product_blocker(DEFAULT_AG_BLOCKING_K),
            ProductSchema,
        ),
    ]

    for name, corpus, gold, blocker, schema in datasets:
        judgements = _judged_candidates(blocker, corpus, schema)
        all_ids = [r.id for r in corpus]

        for clusterer_name, clusterer in [
            ("Clusterer (default, transitive closure)", Clusterer(threshold=THRESHOLD)),
            (
                "CorrelationClusterer (C6, pivot algorithm)",
                CorrelationClusterer(threshold=THRESHOLD),
            ),
        ]:
            predicted = clusterer.cluster(judgements)
            completed = complete_partition(predicted, all_ids)
            metrics = calculate_bcubed_metrics(completed, gold)
            rows.append(
                {
                    "dataset": name,
                    "clusterer": clusterer_name,
                    "bc_p": metrics["precision"],
                    "bc_r": metrics["recall"],
                    "bc_f1": metrics["f1"],
                }
            )
            logger.info(
                "%s | %-42s | bc_P=%.4f | bc_R=%.4f | bc_F1=%.4f",
                name,
                clusterer_name,
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            )

    return rows


def main() -> int:
    logger.info("=== Composite blocking: Pair-Completeness / Reduction-Ratio ===")
    blocking_rows = run_blocking_comparison()

    logger.info("")
    logger.info("=== Clusterer comparison: base Clusterer vs CorrelationClusterer (C6) ===")
    clusterer_rows = run_clusterer_comparison()

    logger.info("")
    logger.info("blocking_rows=%s", blocking_rows)
    logger.info("clusterer_rows=%s", clusterer_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
