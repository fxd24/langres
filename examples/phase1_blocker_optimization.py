"""Phase 1 POC Evaluation: Blocker Optimization.

This script implements Phase 1 of the langres POC evaluation plan. It compares
5 embedding models across multiple k_neighbors values to find the optimal
blocker configuration for the funder deduplication task.

**Objective**: Find the best (model, k) combination for VectorBlocker that:
1. Achieves ≥95% recall (critical - don't miss true matches)
2. Maximizes score separation (≥0.2 target)
3. Achieves strong ranking (MAP ≥0.7)
4. Minimizes k for efficiency

**Models evaluated**:
- all-MiniLM-L6-v2 (384d, fast baseline)
- all-mpnet-base-v2 (768d, better quality)
- paraphrase-MiniLM-L3-v2 (384d, fastest)
- intfloat/multilingual-e5-base (768d, multilingual)
- BAAI/bge-m3 (1024d, multi-lingual/granular)

**K values evaluated**: [10, 20, 50, 100]

**Key metrics tracked**:
- candidates.recall (target: ≥95%)
- candidates.precision
- scores.separation (target: ≥0.2)
- ranking.map (target: ≥0.7)
- ranking.mrr
- recall_curve.optimal_k(target_recall=0.95)

**Outputs**:
1. Comparison table showing Model × K → all metrics
2. Winner identification (best recall + separation at lowest k)
3. Diagnostic markdown files for top configurations
4. Interpretation guide explaining metrics

First run: Computes embeddings (slow, ~2-5 min per model)
Subsequent runs: Instant (loads from cache)
"""

import logging
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
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

# Models to compare (from POC requirements)
MODELS = [
    "all-MiniLM-L6-v2",  # 384d, fast baseline (22M params)
    "all-mpnet-base-v2",  # 768d, better quality (109M params)
    "paraphrase-MiniLM-L3-v2",  # 384d, fastest (17M params)
    "intfloat/multilingual-e5-base",  # 768d, strong multilingual (118M params)
    "BAAI/bge-m3",  # 1024d, multi-lingual, multi-granular (568M params)
]

# K values to test (from POC requirements)
K_VALUES = [20, 30, 50]

CACHE_DIR = Path("tmp/embedding_cache")
DIAGNOSTICS_DIR = Path("tmp")
TARGET_RECALL = 0.95


# =========================================================================
# Schema
# =========================================================================


class FunderSchema(BaseModel):
    """Funder organization schema."""

    id: str = Field(description="Funder ID")
    name: str = Field(description="Funder organization name")


# =========================================================================
# Main Evaluation Logic
# =========================================================================


def main() -> None:
    """Compare embedding models and k values on funder dataset."""
    logger.info("=" * 80)
    logger.info("PHASE 1 POC EVALUATION: BLOCKER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Objective: Find optimal (model, k) configuration for VectorBlocker")
    logger.info(f"Models: {len(MODELS)}")
    logger.info(f"K values: {K_VALUES}")
    logger.info(f"Total configurations: {len(MODELS) * len(K_VALUES)}")
    logger.info("")

    # -------------------------------------------------------------------------
    # 1. Load dataset (once, reused for all models)
    # -------------------------------------------------------------------------
    logger.info("📂 Loading funder dataset...")
    dataset = load_labeled_dedup_data(
        "examples/data",
        entity_names_file="funder_names_with_ids.json",
        labeled_groups_file="funder_name_deduplicated_groups.json",
    )

    entities = [{"id": id, "name": name} for id, name in dataset.entity_names.items()]
    texts = list(dataset.entity_names.values())
    gold_clusters = [set(group.entity_ids) for group in dataset.labeled_groups]

    logger.info(f"✓ Loaded {len(entities)} entities, {len(gold_clusters)} gold clusters")
    logger.info("")

    # -------------------------------------------------------------------------
    # 2. Initialize shared resources
    # -------------------------------------------------------------------------
    logger.info("🔧 Initializing shared resources...")

    # Qdrant in-memory client (no Docker required!)
    client = QdrantClient(":memory:")
    logger.info("✓ Qdrant client created (in-memory mode, no Docker)")

    # Sparse embedder (shared across all models for hybrid search)
    sparse_embedder = FastEmbedSparseEmbedder("Qdrant/bm25")
    logger.info("✓ Sparse embedder (BM25) initialized")
    logger.info("")

    # -------------------------------------------------------------------------
    # 3. Evaluate each (model, k) combination
    # -------------------------------------------------------------------------
    results = []

    for model_name in MODELS:
        logger.info("=" * 80)
        logger.info(f"📊 Evaluating Model: {model_name}")
        logger.info("=" * 80)

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
        logger.info("")

        # Test each k value with this model
        for k in K_VALUES:
            logger.info(f"  ▶ Testing k={k}...")

            # Create blocker with this k value
            blocker = VectorBlocker(
                schema_factory=lambda r: FunderSchema(id=r["id"], name=r["name"]),
                text_field_extractor=lambda x: x.name,
                vector_index=index,  # type: ignore[arg-type]
                k_neighbors=k,
            )

            # Generate candidates
            candidates = list(blocker.stream(entities))
            logger.info(f"    • Generated {len(candidates)} candidate pairs")

            # Evaluate with comprehensive report
            report = blocker.evaluate(candidates, gold_clusters)

            # Extract all key metrics
            optimal_k = report.recall_curve.optimal_k(target_recall=TARGET_RECALL)
            result = {
                "Model": model_name,
                "K": k,
                "Recall": report.candidates.recall,
                "Precision": report.candidates.precision,
                "Separation": report.scores.separation,
                "MAP": report.ranking.map,
                "MRR": report.ranking.mrr,
                "NDCG@10": report.ranking.ndcg_at_10,
                "Median_Rank": report.rank_distribution.median,
                f"Optimal_K_95%": optimal_k,
                "Candidates": report.candidates.total,
                "Missed": report.candidates.missed_matches,
                "False_Pos": report.candidates.false_positives,
            }
            results.append(result)

            logger.info(f"    ✓ Recall: {result['Recall']:.1%}")
            logger.info(f"    ✓ Separation: {result['Separation']:.3f}")
            logger.info(f"    ✓ MAP: {result['MAP']:.3f}")
            logger.info(f"    ✓ Optimal k (95%): {optimal_k}")

            # Generate and save diagnostics for each configuration
            # (We'll filter to top ones later, but save all for completeness)
            entity_dict = {str(e["id"]): e for e in entities}
            diagnostics = report.diagnose(candidates, entity_dict, n_missed=10, n_false_positives=10)

            diag_path = DIAGNOSTICS_DIR / f"diagnostics_blocker_{namespace}_k{k}.md"
            diag_path.parent.mkdir(exist_ok=True)
            diag_path.write_text(diagnostics.to_markdown())
            logger.info(f"    💾 Saved diagnostics to: {diag_path.name}")
            logger.info("")

    # -------------------------------------------------------------------------
    # 4. Display comprehensive comparison table
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("📈 COMPREHENSIVE COMPARISON RESULTS")
    logger.info("=" * 80 + "\n")

    df = pd.DataFrame(results)

    # Format percentages for display
    df_display = df.copy()
    df_display["Recall"] = df["Recall"].apply(lambda x: f"{x:.1%}")
    df_display["Precision"] = df["Precision"].apply(lambda x: f"{x:.1%}")
    df_display["Separation"] = df["Separation"].apply(lambda x: f"{x:.3f}")
    df_display["MAP"] = df["MAP"].apply(lambda x: f"{x:.3f}")
    df_display["MRR"] = df["MRR"].apply(lambda x: f"{x:.3f}")
    df_display["NDCG@10"] = df["NDCG@10"].apply(lambda x: f"{x:.3f}")
    df_display["Median_Rank"] = df["Median_Rank"].apply(lambda x: f"{x:.1f}")

    # Select columns for display
    display_cols = [
        "Model",
        "K",
        "Recall",
        "Precision",
        "Separation",
        "MAP",
        "MRR",
        "Median_Rank",
        "Optimal_K_95%",
    ]
    print(df_display[display_cols].to_markdown(index=False))

    # -------------------------------------------------------------------------
    # 5. Interpretation guide
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("💡 INTERPRETATION GUIDE")
    logger.info("=" * 80)
    logger.info(
        """
Metric Definitions:
  • Recall: Fraction of true matches found by blocker (higher is better)
      - Critical metric - we must not miss true matches!
  • Precision: Fraction of candidates that are true matches (context: blocker, not classifier)
      - Lower is OK for blockers (downstream module filters false positives)
  • Separation: Distance between true and false score distributions (higher is better)
      - Indicates how easily we can set a threshold to filter candidates
  • MAP: Mean Average Precision - ranking quality (higher is better)
      - Measures how well true matches are ranked at the top
  • MRR: Mean Reciprocal Rank - typical rank of first true match (higher is better)
      - High MRR means true matches appear early in ranked lists
  • Median Rank: Typical rank position of true matches (lower is better)
      - Complements MAP/MRR with interpretable position metric
  • Optimal K (95%): Minimum k_neighbors needed for 95% recall (lower is better)
      - Lower optimal k = more efficient (fewer candidates to process downstream)
  • NDCG@10: Normalized Discounted Cumulative Gain at k=10
      - Measures ranking quality in top-10 positions

Success Criteria (from POC goals):
  ✅ Recall ≥ 95%: Excellent (finds most true matches)
  ✅ Separation ≥ 0.2: Good (easy to set threshold)
  ✅ MAP ≥ 0.7: Strong (true matches ranked high)
  ✅ Optimal k ≤ 20: Efficient (low computational cost)

Recommendation Strategy:
  1. Find configurations with HIGHEST recall (≥95% critical for not missing true matches)
  2. Among high-recall configs, prefer HIGHEST separation (easier downstream classification)
  3. Tie-breaker: LOWEST k value (more efficient - fewer candidates to process)
  4. Verify MAP ≥ 0.7 (ensures good ranking quality)
"""
    )

    # -------------------------------------------------------------------------
    # 6. Winner identification with POC criteria
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("🏆 WINNER ANALYSIS (POC Criteria)")
    logger.info("=" * 80)
    logger.info("")

    # Step 1: Find configurations meeting recall threshold
    high_recall_configs = df[df["Recall"] >= TARGET_RECALL]

    if not high_recall_configs.empty:
        logger.info(f"✓ Found {len(high_recall_configs)} configurations with ≥95% recall")
        logger.info("")

        # Step 2: Among high-recall, find best separation
        best_idx = high_recall_configs["Separation"].idxmax()
        winner = df.loc[best_idx]

        logger.info("🥇 WINNER (Best separation among ≥95% recall configs):")
        logger.info(f"   Model: {winner['Model']}")
        logger.info(f"   K: {winner['K']}")
        logger.info(f"   Recall: {winner['Recall']:.1%}")
        logger.info(f"   Separation: {winner['Separation']:.3f}")
        logger.info(f"   MAP: {winner['MAP']:.3f}")
        logger.info(f"   MRR: {winner['MRR']:.3f}")
        logger.info(f"   Optimal k (95%): {winner['Optimal_K_95%']}")
        logger.info("")

        # Check against all POC criteria
        logger.info("POC Success Criteria:")
        logger.info(
            f"  • Recall ≥ 0.95: {'✅ PASS' if winner['Recall'] >= 0.95 else '❌ FAIL'} "
            f"({winner['Recall']:.1%})"
        )
        logger.info(
            f"  • Separation ≥ 0.20: {'✅ PASS' if winner['Separation'] >= 0.20 else '❌ FAIL'} "
            f"({winner['Separation']:.3f})"
        )
        logger.info(
            f"  • MAP ≥ 0.70: {'✅ PASS' if winner['MAP'] >= 0.70 else '❌ FAIL'} "
            f"({winner['MAP']:.3f})"
        )
        logger.info(
            f"  • Optimal k ≤ 20: {'✅ PASS' if winner['Optimal_K_95%'] <= 20 else '⚠️  WARN'} "
            f"({winner['Optimal_K_95%']})"
        )
        logger.info("")

        # Show runners-up
        logger.info("🥈 Top 3 Runners-Up (by separation among ≥95% recall):")
        top_3 = high_recall_configs.nlargest(4, "Separation").iloc[1:4]  # Skip winner
        for i, (_, row) in enumerate(top_3.iterrows(), 2):
            logger.info(
                f"   {i}. {row['Model']} (k={row['K']}): "
                f"Recall={row['Recall']:.1%}, Sep={row['Separation']:.3f}, "
                f"MAP={row['MAP']:.3f}"
            )
        logger.info("")

    else:
        logger.info(f"⚠️  No configurations achieved ≥{TARGET_RECALL:.0%} recall")
        logger.info("    Falling back to best recall configuration...")
        logger.info("")

        # Fallback: best recall config
        best_idx = df["Recall"].idxmax()
        winner = df.loc[best_idx]

        logger.info("🥇 BEST AVAILABLE (Highest recall, but <95%):")
        logger.info(f"   Model: {winner['Model']}")
        logger.info(f"   K: {winner['K']}")
        logger.info(f"   Recall: {winner['Recall']:.1%}")
        logger.info(f"   Separation: {winner['Separation']:.3f}")
        logger.info(f"   MAP: {winner['MAP']:.3f}")
        logger.info("")
        logger.info("   ⚠️  Consider:")
        logger.info("     1. Testing larger k values (e.g., 200, 500)")
        logger.info("     2. Trying additional embedding models")
        logger.info("     3. Using ensemble blocking (multiple models)")
        logger.info("")

    # -------------------------------------------------------------------------
    # 7. Show diagnostics summary for winner
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("🔍 WINNER DIAGNOSTICS SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    winner_namespace = winner["Model"].replace("/", "_").replace("-", "_")
    winner_k = int(winner["K"])
    diag_path = DIAGNOSTICS_DIR / f"diagnostics_blocker_{winner_namespace}_k{winner_k}.md"

    logger.info(f"Full diagnostics saved to: {diag_path}")
    logger.info("")
    logger.info("Quick preview:")
    logger.info(f"  • Total candidates: {winner['Candidates']}")
    logger.info(f"  • Missed matches: {winner['Missed']}")
    logger.info(f"  • False positives: {winner['False_Pos']}")
    logger.info("")
    logger.info(f"View detailed failure examples: cat {diag_path}")
    logger.info("")

    # -------------------------------------------------------------------------
    # 8. Summary and next steps
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("✅ PHASE 1 EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  • Evaluated {len(MODELS)} models × {len(K_VALUES)} k values = {len(results)} configs")
    logger.info(
        f"  • Winner: {winner['Model']} with k={winner['K']} "
        f"(Recall={winner['Recall']:.1%}, Sep={winner['Separation']:.3f})"
    )
    logger.info(f"  • Diagnostics: {DIAGNOSTICS_DIR}/diagnostics_blocker_*.md")
    logger.info("")
    logger.info("Next Steps:")
    logger.info("  1. Review diagnostic examples to understand failure modes")
    logger.info("  2. Proceed to Phase 2: Build RapidfuzzModule baseline")
    logger.info("  3. Proceed to Phase 3: Build LLMJudgeModule for hybrid approach")
    logger.info("")


if __name__ == "__main__":
    main()
