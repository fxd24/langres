"""Compare embedding models for funder organization deduplication.

This example evaluates multiple sentence-transformer embedding models to determine
which provides the best blocking quality for funder names. It uses:
- Qdrant in-memory (no Docker required)
- DiskCachedEmbedder for instant subsequent runs
- Comprehensive BlockerEvaluationReport metrics

Models are compared on:
- Recall: Does it find true matches?
- Separation: Can we distinguish true vs false candidates?
- MAP: How well ranked are true matches?
- Optimal k: Minimum neighbors for 95% recall

First run: Computes embeddings (slow, ~2-5 min per model)
Subsequent runs: Instant (loads from cache)
"""

import logging
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import (
    DiskCachedEmbedder,
    FastEmbedSparseEmbedder,
    SentenceTransformerEmbedder,
)
from langres.core.indexes.hybrid_vector_index import QdrantHybridIndex
from langres.data import load_labeled_dedup_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =========================================================================
# Configuration
# =========================================================================

# Models to compare (add/remove as needed)
MODELS = [
    "all-MiniLM-L6-v2",  # 384d, fast baseline (22M params)
    "all-mpnet-base-v2",  # 768d, better quality (109M params)
    "paraphrase-MiniLM-L3-v2",  # 384d, fastest (17M params)
    "intfloat/multilingual-e5-base",  # 768d, strong multilingual (118M params)
    "BAAI/bge-m3",  # 1024d, multi-lingual, multi-granular (568M params)
]

K_NEIGHBORS = 100
CACHE_DIR = Path("tmp/embedding_cache")
TARGET_RECALL = 0.95


# =========================================================================
# Schema
# =========================================================================


class FunderSchema(BaseModel):
    """Funder organization schema."""

    id: str = Field(description="Funder ID")
    name: str = Field(description="Funder organization name")


# =========================================================================
# Main Comparison Logic
# =========================================================================


def main():
    """Compare embedding models on funder dataset."""
    logger.info("=" * 70)
    logger.info("MULTI-EMBEDDER COMPARISON FOR FUNDER DEDUPLICATION")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Load dataset (once, reused for all models)
    # -------------------------------------------------------------------------
    logger.info("\n📂 Loading funder dataset...")
    dataset = load_labeled_dedup_data(
        "examples/data",
        entity_names_file="funder_names_with_ids.json",
        labeled_groups_file="funder_name_deduplicated_groups.json",
    )

    entities = [{"id": id, "name": name} for id, name in dataset.entity_names.items()]
    texts = list(dataset.entity_names.values())
    gold_clusters = [set(group.entity_ids) for group in dataset.labeled_groups]

    logger.info(f"✓ Loaded {len(entities)} entities, {len(gold_clusters)} clusters")

    # -------------------------------------------------------------------------
    # 2. Initialize shared resources
    # -------------------------------------------------------------------------
    logger.info("\n🔧 Initializing shared resources...")

    # Qdrant in-memory client (no Docker required!)
    client = QdrantClient(":memory:")
    logger.info("✓ Qdrant client created (in-memory mode, no Docker)")

    # Sparse embedder (shared across all models for hybrid search)
    sparse_embedder = FastEmbedSparseEmbedder("Qdrant/bm25")
    logger.info("✓ Sparse embedder (BM25) initialized")

    # -------------------------------------------------------------------------
    # 3. Evaluate each model
    # -------------------------------------------------------------------------
    results = []

    for model_name in MODELS:
        logger.info("\n" + "=" * 70)
        logger.info(f"📊 Evaluating: {model_name}")
        logger.info("=" * 70)

        # Create cached dense embedder (unique namespace per model)
        namespace = model_name.replace("/", "_").replace("-", "_")
        base_embedder = SentenceTransformerEmbedder(model_name=model_name)
        cached_embedder = DiskCachedEmbedder(
            embedder=base_embedder,
            cache_dir=CACHE_DIR,
            namespace=namespace,
            memory_cache_size=2000,  # Keep all 1,741 entities in hot cache
        )

        logger.info(f"  🔹 Dense embedder: {model_name}")
        logger.info(f"  🔹 Cache namespace: {namespace}")

        # Create hybrid index (unique collection name per model)
        collection_name = f"funders_{namespace}"
        index = QdrantHybridIndex(
            client=client,
            collection_name=collection_name,
            dense_embedder=cached_embedder,
            sparse_embedder=sparse_embedder,
        )

        logger.info(f"  🔹 Qdrant collection: {collection_name}")
        logger.info(f"  🔹 Building index (may be slow on first run)...")

        # Build index (creates/caches embeddings)
        index.create_index(texts)

        # Print cache stats
        cache_info = cached_embedder.cache_info()
        total_hits = cache_info["hits_hot"] + cache_info["hits_cold"]
        total_requests = total_hits + cache_info["misses"]
        logger.info(
            f"  ✓ Index built | Cache: {cache_info['hit_rate']:.1%} hit rate "
            f"({total_hits}/{total_requests} hits, {cache_info['misses']} misses)"
        )

        # Create blocker
        blocker = VectorBlocker(
            schema_factory=lambda r: FunderSchema(id=r["id"], name=r["name"]),
            text_field_extractor=lambda x: x.name,
            vector_index=index,
            k_neighbors=K_NEIGHBORS,
        )

        # Generate candidates
        logger.info(f"  🔹 Generating candidates (k={K_NEIGHBORS})...")
        candidates = list(blocker.stream(entities))
        logger.info(f"  ✓ Generated {len(candidates)} candidate pairs")

        # Evaluate
        logger.info("  🔹 Evaluating blocker...")
        report = blocker.evaluate(candidates, gold_clusters)

        # Collect metrics
        optimal_k = report.recall_curve.optimal_k(target_recall=TARGET_RECALL)
        result = {
            "Model": model_name,
            "Recall": f"{report.candidates.recall:.1%}",
            "Precision": f"{report.candidates.precision:.1%}",
            "MAP": f"{report.ranking.map:.3f}",
            "MRR": f"{report.ranking.mrr:.3f}",
            "Separation": f"{report.scores.separation:.3f}",
            "Median Rank": f"{report.rank_distribution.median:.1f}",
            f"Optimal k ({TARGET_RECALL:.0%})": optimal_k,
        }
        results.append(result)

        logger.info("  ✓ Evaluation complete")
        logger.info(f"    • Recall: {result['Recall']}")
        logger.info(f"    • Separation: {result['Separation']}")
        logger.info(f"    • MAP: {result['MAP']}")
        logger.info(f"    • Optimal k: {optimal_k}")

    # -------------------------------------------------------------------------
    # 4. Display comparison table
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("📈 COMPARISON RESULTS")
    logger.info("=" * 70 + "\n")

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

    # -------------------------------------------------------------------------
    # 5. Interpretation guide
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("💡 INTERPRETATION GUIDE")
    logger.info("=" * 70)
    logger.info(
        """
Metric Definitions:
  • Recall: Fraction of true matches found by blocker (higher is better)
  • Precision: Fraction of candidates that are true matches (context: blocker, not classifier)
  • MAP: Mean Average Precision - ranking quality (higher is better)
  • MRR: Mean Reciprocal Rank - typical rank of first true match (higher is better)
  • Separation: Distance between true and false score distributions (higher is better)
  • Median Rank: Typical rank position of true matches (lower is better)
  • Optimal k: Minimum neighbors needed for 95% recall (lower is better)

Success Criteria (from POC goals):
  • Recall ≥ 95%: Excellent (finds most true matches)
  • Separation ≥ 0.2: Good (easy to set threshold)
  • MAP ≥ 0.7: Strong (true matches ranked high)
  • Optimal k ≤ 20: Efficient (low computational cost)

Recommendation:
  1. Find model with HIGHEST recall (critical for not missing true matches)
  2. Among high-recall models, prefer HIGHEST separation (easier downstream classification)
  3. Tie-breaker: LOWEST optimal k (more efficient)
"""
    )

    # -------------------------------------------------------------------------
    # 6. Winner identification
    # -------------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("🏆 WINNER ANALYSIS")
    logger.info("=" * 70)

    # Parse recall back to float for comparison
    df["Recall_float"] = df["Recall"].str.rstrip("%").astype(float)
    df["Separation_float"] = df["Separation"].astype(float)

    best_recall_idx = df["Recall_float"].idxmax()
    best_separation_idx = df["Separation_float"].idxmax()
    best_map_idx = df["MAP"].astype(float).idxmax()

    logger.info(f"  🥇 Best Recall: {df.loc[best_recall_idx, 'Model']}")
    logger.info(f"  🥇 Best Separation: {df.loc[best_separation_idx, 'Model']}")
    logger.info(f"  🥇 Best MAP: {df.loc[best_map_idx, 'Model']}")

    # Overall recommendation (prioritize recall, then separation)
    high_recall_models = df[df["Recall_float"] >= 95.0]
    if not high_recall_models.empty:
        winner_idx = high_recall_models["Separation_float"].idxmax()
        winner = df.loc[winner_idx, "Model"]
        logger.info(f"\n  ✨ RECOMMENDED MODEL: {winner}")
        logger.info(
            f"     Rationale: Achieves ≥95% recall with best separation among high-recall models"
        )
    else:
        winner_idx = best_recall_idx
        winner = df.loc[winner_idx, "Model"]
        logger.info(f"\n  ⚠️  RECOMMENDED MODEL: {winner}")
        logger.info("     Rationale: Best recall (but <95% - consider trying more models)")

    # -------------------------------------------------------------------------
    # 7. Diagnose best model with concrete examples
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("🔍 DIAGNOSING BEST MODEL")
    logger.info("=" * 70)

    # Re-run blocker for winner to get candidates
    # (In production, would store candidates during comparison)
    logger.info(f"\nDiagnosing: {winner}")
    namespace = winner.replace("/", "_").replace("-", "_")
    base_embedder = SentenceTransformerEmbedder(model_name=winner)
    cached_embedder = DiskCachedEmbedder(
        embedder=base_embedder,
        cache_dir=CACHE_DIR,
        namespace=namespace,
        memory_cache_size=2000,
    )

    collection_name = f"funders_{namespace}"
    index = QdrantHybridIndex(
        client=client,
        collection_name=collection_name,
        dense_embedder=cached_embedder,
        sparse_embedder=sparse_embedder,
    )

    # Index should already exist from comparison phase
    blocker = VectorBlocker(
        schema_factory=lambda r: FunderSchema(id=r["id"], name=r["name"]),
        text_field_extractor=lambda x: x.name,
        vector_index=index,
        k_neighbors=K_NEIGHBORS,
    )

    candidates = list(blocker.stream(entities))
    report = blocker.evaluate(candidates, gold_clusters)

    # Create entity dict for text extraction
    entity_dict = {str(e["id"]): e for e in entities}

    # Diagnose
    logger.info("  🔹 Extracting diagnostic examples...")
    diagnostics = report.diagnose(candidates, entity_dict, n_missed=10, n_false_positives=10)

    # Show missed matches
    logger.info(f"\n  📉 Top 10 Missed Matches (Total: {len(diagnostics.missed_matches)})")
    for i, ex in enumerate(diagnostics.missed_matches[:10], 1):
        logger.info(f"    {i}. '{ex.left_text}' ↔ '{ex.right_text}'")
        logger.info(f"       (Cluster {ex.cluster_id})")

    # Show false positives
    if diagnostics.false_positives:
        logger.info(f"\n  ⚠️  Top 10 False Positives (Total: {len(diagnostics.false_positives)})")
        for i, ex in enumerate(diagnostics.false_positives[:10], 1):
            logger.info(f"    {i}. '{ex.left_text}' ↔ '{ex.right_text}'")
            logger.info(f"       (Score: {ex.score:.2f})")
    else:
        logger.info("\n  ✅ No high-scoring false positives (score ≥ 0.7)")

    # Save full diagnostic markdown
    logger.info("\n  💾 Saving full diagnostic report...")
    diag_path = Path("tmp") / f"diagnostics_{namespace}.md"
    diag_path.parent.mkdir(exist_ok=True)
    diag_path.write_text(diagnostics.to_markdown())
    logger.info(f"  ✓ Saved to: {diag_path}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Comparison complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
