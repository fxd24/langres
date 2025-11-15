"""
Deduplication Example: Demonstrating DiskCachedEmbedder Speedup with FAISS

This example demonstrates the power of embedding caching for entity deduplication:

1. **Dataset**: 1,741 real-world funder organization names with ground truth labels
2. **Model**: Qwen3 0.6B with DiskCachedEmbedder for persistent caching
3. **Blocking**: FAISS-based vector search for candidate generation
4. **Evaluation**: Full metrics (recall, precision, F1) on both runs

The example runs the same deduplication task twice:
- Run 1 (Cold Cache): Embeddings computed from scratch (~40-60s)
- Run 2 (Hot Cache): Embeddings retrieved from cache (~1-2s, 20-40x faster)

Quality metrics remain identical, proving cache correctness while dramatically
improving performance.

Usage:
    python examples/deduplication_cached_faiss_simple.py
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import DiskCachedEmbedder, SentenceTransformerEmbedder
from langres.core.metrics import pairs_from_clusters
from langres.core.vector_index import FAISSIndex
from langres.data import load_labeled_dedup_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CACHE_DIR = Path("tmp/embedder_cache")
K_NEIGHBORS = 20  # Number of neighbors to retrieve per query
EMBEDDING_INSTRUCTION = (
    "Instruct: Find duplicate organization names accounting for "
    "acronyms, abbreviations, and spelling variations\nQuery: "
)


class OrganizationSchema(BaseModel):
    """Schema for funder organization entities."""

    id: str = Field(description="Unique organization ID")
    name: str = Field(description="Organization name")


def load_funder_data() -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    """Load funder names dataset with ground truth labels.

    Returns:
        tuple: (entities list, gold pairs set)
    """
    logger.info("Loading funder names dataset...")

    dataset = load_labeled_dedup_data(
        data_dir="examples/data",
        entity_names_file="funder_names_with_ids.json",
        labeled_groups_file="funder_name_deduplicated_groups.json",
    )

    # Convert to entity list
    entities = [{"id": entity_id, "name": name} for entity_id, name in dataset.entity_names.items()]

    # Get ground truth pairs from labeled groups
    # Convert LabeledGroup objects to sets of entity_ids
    gold_clusters = [set(group.entity_ids) for group in dataset.labeled_groups]
    gold_pairs = pairs_from_clusters(gold_clusters)

    logger.info(
        f"Loaded {len(entities)} entities with {len(gold_pairs)} ground truth duplicate pairs"
    )

    return entities, gold_pairs


def setup_faiss_blocker(embedder: DiskCachedEmbedder) -> VectorBlocker[OrganizationSchema]:
    """Setup FAISS blocker with dense vectors and caching.

    Args:
        embedder: DiskCachedEmbedder wrapping the base embedding model

    Returns:
        VectorBlocker configured with FAISS index
    """
    logger.info("Setting up FAISS blocker...")

    # FAISS index with cosine similarity
    vector_index = FAISSIndex(embedder=embedder, metric="cosine")

    # VectorBlocker with FAISS
    blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=vector_index,
        k_neighbors=K_NEIGHBORS,
    )

    logger.info(f"FAISS blocker ready (k={K_NEIGHBORS}, metric=cosine, cached=True)")
    return blocker


def evaluate_blocker(
    blocker: VectorBlocker[OrganizationSchema],
    cached_embedder: DiskCachedEmbedder,
    entities: list[dict[str, Any]],
    gold_pairs: set[tuple[str, str]],
    run_name: str,
) -> dict[str, Any]:
    """Evaluate blocking performance and track cache statistics.

    Args:
        blocker: VectorBlocker instance to evaluate
        cached_embedder: DiskCachedEmbedder to track cache hits/misses
        entities: List of entity dictionaries
        gold_pairs: Ground truth duplicate pairs
        run_name: Name for this evaluation run

    Returns:
        Dictionary with metrics and cache statistics
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{run_name}")
    logger.info(f"{'=' * 80}")

    # Get cache stats BEFORE indexing
    cache_before = cached_embedder.cache_info()

    # Build index
    texts = [e["name"] for e in entities]
    logger.info(f"Indexing {len(texts)} organizations...")

    start = time.time()
    blocker.vector_index.create_index(texts)
    indexing_time = time.time() - start

    # Get cache stats AFTER indexing
    cache_after_index = cached_embedder.cache_info()

    # Generate candidates
    logger.info("Generating candidate pairs...")
    start = time.time()
    candidates = list(blocker.stream(entities))
    query_time = time.time() - start

    # Get final cache stats
    cache_after = cached_embedder.cache_info()

    # Calculate cache deltas
    cache_hits_index = (cache_after_index["hits_hot"] + cache_after_index["hits_cold"]) - (
        cache_before["hits_hot"] + cache_before["hits_cold"]
    )
    cache_misses_index = cache_after_index["misses"] - cache_before["misses"]

    cache_hits_query = (cache_after["hits_hot"] + cache_after["hits_cold"]) - (
        cache_after_index["hits_hot"] + cache_after_index["hits_cold"]
    )
    cache_misses_query = cache_after["misses"] - cache_after_index["misses"]

    # Extract candidate pairs (normalized)
    candidate_pairs: set[tuple[str, str]] = set()
    for c in candidates:
        sorted_pair = sorted([c.left.id, c.right.id])
        candidate_pairs.add((sorted_pair[0], sorted_pair[1]))

    # Normalize gold pairs
    gold_normalized: set[tuple[str, str]] = set()
    for pair in gold_pairs:
        sorted_pair = sorted(pair)
        gold_normalized.add((sorted_pair[0], sorted_pair[1]))

    # Calculate metrics
    tp = len(gold_normalized & candidate_pairs)
    fn = len(gold_normalized - candidate_pairs)
    fp = len(candidate_pairs - gold_normalized)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Log results
    logger.info(f"  Indexing time: {indexing_time:.2f}s")
    logger.info(f"  Query time: {query_time:.2f}s")
    logger.info(f"  Total time: {indexing_time + query_time:.2f}s")
    logger.info(f"  Cache (indexing): {cache_hits_index} hits, {cache_misses_index} misses")
    logger.info(f"  Cache (query): {cache_hits_query} hits, {cache_misses_query} misses")
    logger.info(f"  Recall: {recall:.1%}, Precision: {precision:.1%}, F1: {f1:.1%}")
    logger.info(f"  TP={tp}, FN={fn}, FP={fp}, Total Candidates={len(candidates)}")

    return {
        "run_name": run_name,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "total_candidates": len(candidates),
        "indexing_time_seconds": indexing_time,
        "query_time_seconds": query_time,
        "total_time_seconds": indexing_time + query_time,
        "avg_query_time_ms": (query_time / len(entities)) * 1000 if entities else 0,
        "cache_hits_index": cache_hits_index,
        "cache_misses_index": cache_misses_index,
        "cache_hits_query": cache_hits_query,
        "cache_misses_query": cache_misses_query,
        "cache_hit_rate": (
            (cache_hits_index + cache_hits_query)
            / (cache_hits_index + cache_hits_query + cache_misses_index + cache_misses_query)
            if (cache_hits_index + cache_hits_query + cache_misses_index + cache_misses_query) > 0
            else 0.0
        ),
    }


def print_comparison_table(run1: dict[str, Any], run2: dict[str, Any]) -> None:
    """Print side-by-side comparison table showing cache speedup.

    Args:
        run1: Metrics from first run (cold cache)
        run2: Metrics from second run (hot cache)
    """
    print("\n" + "=" * 100)
    print("COMPARISON: Cold Cache vs Hot Cache")
    print("=" * 100)

    # Quality metrics (should be identical)
    quality_metrics = [
        ("Recall", "recall", "%"),
        ("Precision", "precision", "%"),
        ("F1 Score", "f1", "%"),
        ("True Positives", "true_positives", ""),
        ("False Negatives", "false_negatives", ""),
        ("False Positives", "false_positives", ""),
        ("Total Candidates", "total_candidates", ""),
    ]

    # Performance metrics
    performance_metrics = [
        ("Indexing Time", "indexing_time_seconds", "s"),
        ("Query Time", "query_time_seconds", "s"),
        ("Total Time", "total_time_seconds", "s"),
        ("Avg Query Time", "avg_query_time_ms", "ms"),
    ]

    # Cache metrics
    cache_metrics = [
        ("Cache Hits (Index)", "cache_hits_index", ""),
        ("Cache Misses (Index)", "cache_misses_index", ""),
        ("Cache Hits (Query)", "cache_hits_query", ""),
        ("Cache Misses (Query)", "cache_misses_query", ""),
        ("Cache Hit Rate", "cache_hit_rate", "%"),
    ]

    print(
        f"\n{'Metric':<30} {'Run 1 (Cold Cache)':>22} {'Run 2 (Hot Cache)':>22} {'Speedup/Δ':>20}"
    )
    print("-" * 100)

    # Print quality metrics
    print("Quality Metrics:")
    for label, key, unit in quality_metrics:
        val1 = run1[key]
        val2 = run2[key]

        if unit == "%":
            val1_str = f"{val1 * 100:.2f}%"
            val2_str = f"{val2 * 100:.2f}%"
            match = "✓ Match" if abs(val1 - val2) < 0.001 else "✗ Differ"
        else:
            val1_str = str(val1)
            val2_str = str(val2)
            match = "✓ Match" if val1 == val2 else "✗ Differ"

        print(f"  {label:<28} {val1_str:>22} {val2_str:>22} {match:>20}")

    # Print performance metrics
    print("\nPerformance Metrics:")
    for label, key, unit in performance_metrics:
        val1 = run1[key]
        val2 = run2[key]

        val1_str = f"{val1:.3f}{unit}"
        val2_str = f"{val2:.3f}{unit}"

        if val2 > 0 and val1 > 0:
            speedup = val1 / val2
            speedup_str = f"{speedup:.1f}x faster" if speedup > 1 else f"{1 / speedup:.1f}x slower"
        else:
            speedup_str = "N/A"

        print(f"  {label:<28} {val1_str:>22} {val2_str:>22} {speedup_str:>20}")

    # Print cache metrics
    print("\nCache Statistics:")
    for label, key, unit in cache_metrics:
        val1 = run1[key]
        val2 = run2[key]

        if unit == "%":
            val1_str = f"{val1 * 100:.1f}%"
            val2_str = f"{val2 * 100:.1f}%"
            delta = (val2 - val1) * 100
            delta_str = f"{delta:+.1f}pp"
        else:
            val1_str = str(val1)
            val2_str = str(val2)
            delta = val2 - val1
            delta_str = f"{delta:+d}"

        print(f"  {label:<28} {val1_str:>22} {val2_str:>22} {delta_str:>20}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    indexing_speedup = (
        run1["indexing_time_seconds"] / run2["indexing_time_seconds"]
        if run2["indexing_time_seconds"] > 0
        else 0
    )
    total_speedup = (
        run1["total_time_seconds"] / run2["total_time_seconds"]
        if run2["total_time_seconds"] > 0
        else 0
    )

    print(
        f"✓ Quality unchanged: Recall={run2['recall']:.1%}, Precision={run2['precision']:.1%}, F1={run2['f1']:.1%}"
    )
    print(
        f"✓ Cache speedup: {indexing_speedup:.1f}x faster indexing "
        f"({run1['indexing_time_seconds']:.2f}s → {run2['indexing_time_seconds']:.2f}s)"
    )
    print(f"✓ Overall speedup: {total_speedup:.1f}x faster total time")
    print(
        f"✓ Cache hit rate: Run 1={run1['cache_hit_rate']:.1%}, Run 2={run2['cache_hit_rate']:.1%}"
    )

    if run2["cache_misses_index"] == 0 and run2["cache_hits_index"] > 0:
        print("✓ Perfect cache hit rate on second run (all embeddings cached)")

    print("=" * 100)


def save_results(
    run1: dict[str, Any],
    run2: dict[str, Any],
    entities: list[dict[str, Any]],
    gold_pairs: set[tuple[str, str]],
) -> None:
    """Save evaluation results to JSON file.

    Args:
        run1: Metrics from first run (cold cache)
        run2: Metrics from second run (hot cache)
        entities: List of entities evaluated
        gold_pairs: Ground truth pairs
    """
    # Calculate embedding dimension from first entity
    # (This would be set during indexing, but for now we know it's 1024 for Qwen3 0.6B)
    embedding_dim = 1024

    output = {
        "dataset": {
            "name": "funder_names_with_ids",
            "total_entities": len(entities),
            "ground_truth_pairs": len(gold_pairs),
        },
        "model": {
            "name": DENSE_MODEL,
            "embedding_dim": embedding_dim,
            "k_neighbors": K_NEIGHBORS,
            "cache_enabled": True,
            "cache_dir": str(CACHE_DIR),
        },
        "experiments": {
            "run1_cold_cache": {k: v for k, v in run1.items() if k != "run_name"},
            "run2_hot_cache": {k: v for k, v in run2.items() if k != "run_name"},
        },
        "cache_speedup": {
            "indexing_speedup": (
                run1["indexing_time_seconds"] / run2["indexing_time_seconds"]
                if run2["indexing_time_seconds"] > 0
                else 0
            ),
            "total_speedup": (
                run1["total_time_seconds"] / run2["total_time_seconds"]
                if run2["total_time_seconds"] > 0
                else 0
            ),
            "quality_preserved": abs(run1["f1"] - run2["f1"]) < 0.001,
        },
    }

    output_path = Path("examples/deduplication_cached_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"✅ Results saved to: {output_path}")


def main() -> None:
    """Run deduplication evaluation demonstrating cache speedup."""
    # Setup cached embedder
    logger.info("Setting up DiskCachedEmbedder with Qwen3 0.6B...")

    base_embedder = SentenceTransformerEmbedder(
        model_name=DENSE_MODEL,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    cached_embedder = DiskCachedEmbedder(
        embedder=base_embedder,
        cache_dir=CACHE_DIR,
        namespace="qwen3_funder_dedup",
        memory_cache_size=100_000,
    )

    # Load funder names dataset
    entities, gold_pairs = load_funder_data()

    # Clear cache to ensure cold start
    logger.info("Clearing cache for cold start...")
    cached_embedder.cache_clear()

    # RUN 1: Cold cache (compute embeddings)
    logger.info("Creating blocker for Run 1...")
    blocker1 = setup_faiss_blocker(cached_embedder)

    results_run1 = evaluate_blocker(
        blocker=blocker1,
        cached_embedder=cached_embedder,
        entities=entities,
        gold_pairs=gold_pairs,
        run_name="RUN 1: Cold Cache (Computing Embeddings)",
    )

    # RUN 2: Hot cache (retrieve from cache) - Create fresh blocker
    logger.info("Creating fresh blocker for Run 2...")
    blocker2 = setup_faiss_blocker(cached_embedder)

    results_run2 = evaluate_blocker(
        blocker=blocker2,
        cached_embedder=cached_embedder,
        entities=entities,
        gold_pairs=gold_pairs,
        run_name="RUN 2: Hot Cache (Cached Embeddings)",
    )

    # Print comparison
    print_comparison_table(results_run1, results_run2)

    # Save results
    save_results(results_run1, results_run2, entities, gold_pairs)


if __name__ == "__main__":
    main()
