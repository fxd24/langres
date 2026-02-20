"""Comprehensive Blocking Evaluation: Model, Instruction & Architecture Comparison.

This example systematically compares 5 approaches to entity resolution blocking:

1. **FAISS-MiniLM**: Baseline (dense-only semantic search, simple model)
2. **Hybrid-MiniLM**: Add sparse vectors (MiniLM + BM25 keyword matching)
3. **HybridRerank-MiniLM**: Add reranking (MiniLM + BM25 + ColBERT)
4. **Hybrid-Qwen3-NoInst**: Advanced model without instructions (Qwen3 + BM25)
5. **HybridRerank-Qwen3-WithInst**: Advanced model WITH instructions + reranking (Qwen3 + instr + BM25 + ColBERT)

Key comparisons:
- **Sparse vectors impact**: FAISS vs Hybrid (approaches 1 vs 2)
- **Reranking impact**: Hybrid vs Hybrid+Reranking (approaches 2 vs 3)
- **Model quality impact**: MiniLM vs Qwen3 (approaches 2 vs 4)
- **Instructions + Reranking on Qwen3**: Shows combined effect (approaches 4 vs 5)
- **Best reranking approach**: MiniLM vs Qwen3+instructions (approaches 3 vs 5)

Models:
- **Simple**: sentence-transformers/all-MiniLM-L6-v2 (384 dims, 22M params, no instructions)
- **Advanced**: Qwen/Qwen3-Embedding-0.6B (1024 dims, 600M params, WITH instructions)
- **Sparse**: Qdrant/bm25 (keyword matching)
- **Reranking**: colbert-ir/colbertv2.0 (late-interaction)

Dataset: 1,741 real-world funder organization names with ground truth labels.

Expected findings:
- Sparse vectors: +2-5% recall improvement (approach 1 vs 2)
- Reranking: +1-3% precision improvement, 20-40x slower (approach 2 vs 3)
- Qwen3 vs MiniLM: +3-8% F1 improvement (approach 2 vs 4)
- Instructions + Reranking on Qwen3: +3-6% F1 combined improvement (approach 4 vs 5)
- Best quality: HybridRerank-Qwen3-WithInst (approach 5)
- Best trade-off: Hybrid-Qwen3-NoInst (approach 4, good quality without reranking)

Usage:
    python examples/blocking_evaluation_comprehensive_comparison.py
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from langres.clients.settings import Settings
from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import (
    DiskCachedEmbedder,
    FastEmbedLateInteractionEmbedder,
    FastEmbedSparseEmbedder,
    SentenceTransformerEmbedder,
)
from langres.core.hybrid_vector_index import QdrantHybridIndex
from langres.core.metrics import evaluate_blocking_with_ranking, pairs_from_clusters
from langres.core.reranking_vector_index import QdrantHybridRerankingIndex
from langres.core.vector_index import FAISSIndex
from langres.data import load_labeled_dedup_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model configuration
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Simple baseline (384 dims, 22M params)
QWEN3_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # Advanced model (1024 dims, 600M params)
SPARSE_MODEL = "Qdrant/bm25"  # BM25 keyword matching
RERANKING_MODEL = "colbert-ir/colbertv2.0"  # ColBERT late-interaction
K_NEIGHBORS = 20  # Number of neighbors to retrieve
PREFETCH_LIMIT = 20  # For Qdrant hybrid prefetch

# Cache configuration (Qwen3 only)
CACHE_DIR = Path("tmp/cache")
CACHE_NAMESPACE = "qwen3_comprehensive_comparison"
MEMORY_CACHE_SIZE = 10_000

# Instruction prompt (Qwen3 only)
EMBEDDING_INSTRUCTION = (
    "Instruct: Find duplicate organization names accounting for "
    "acronyms, abbreviations, and spelling variations\nQuery: "
)


class OrganizationSchema(BaseModel):
    """Schema for funder organization entities."""

    id: str = Field(description="Unique organization ID")
    name: str = Field(description="Organization name")


def load_funder_data() -> tuple[list[dict[str, Any]], set[tuple[str, str]], list[set[str]]]:
    """Load funder names dataset with ground truth labels.

    Returns:
        tuple: (entities list, gold pairs set, gold clusters list)
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
    gold_clusters = [set(group.entity_ids) for group in dataset.labeled_groups]
    gold_pairs = pairs_from_clusters(gold_clusters)

    logger.info(
        f"Loaded {len(entities)} entities with {len(gold_pairs)} ground truth duplicate pairs"
    )

    return entities, gold_pairs, gold_clusters


def connect_to_qdrant() -> QdrantClient:
    """Connect to managed Qdrant cloud instance.

    Returns:
        QdrantClient: Connected Qdrant client
    """
    logger.info("Connecting to Qdrant cloud...")

    settings = Settings()

    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY must be set in .env file. "
            "See .env.example for configuration."
        )

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    # Verify connection
    collections = client.get_collections()
    logger.info(f"Connected to Qdrant. Existing collections: {len(collections.collections)}")

    return client


def evaluate_blocking_recall(
    blocker: VectorBlocker[OrganizationSchema],
    entities: list[dict[str, Any]],
    gold_pairs: set[tuple[str, str]],
    gold_clusters: list[set[str]],
    name: str,
) -> dict[str, Any]:
    """Evaluate blocking recall, precision, F1, ranking quality, and performance.

    Args:
        blocker: VectorBlocker instance to evaluate
        entities: List of entity dicts to process
        gold_pairs: Set of ground truth duplicate pairs
        gold_clusters: List of ground truth clusters
        name: Name for logging

    Returns:
        dict with evaluation metrics including ranking metrics
    """
    logger.info(f"[{name}] Generating candidates...")

    # Extract texts and build index explicitly
    texts = [entity["name"] for entity in entities]
    blocker.vector_index.create_index(texts)

    # Generate candidates and measure time
    start_time = time.time()
    candidates = list(blocker.stream(entities))
    indexing_time = time.time() - start_time

    logger.info(f"[{name}] Generated {len(candidates)} candidate pairs in {indexing_time:.2f}s")

    # Convert candidates to pair set (normalized: sorted tuples)
    candidate_pairs = {tuple(sorted([c.left.id, c.right.id])) for c in candidates}

    # Normalize gold pairs (ensure sorted tuples)
    gold_normalized = {tuple(sorted(pair)) for pair in gold_pairs}

    # Calculate metrics
    tp = len(gold_normalized & candidate_pairs)  # True positives
    fn = len(gold_normalized - candidate_pairs)  # False negatives (missed pairs)
    fp = len(candidate_pairs - gold_normalized)  # False positives (wrong pairs)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute ranking metrics
    ranking_metrics = evaluate_blocking_with_ranking(
        candidates=candidates,
        gold_clusters=gold_clusters,
        k_values=[20],
    )

    results = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "total_candidates": len(candidates),
        "indexing_time_seconds": indexing_time,
        "avg_query_time_ms": (indexing_time / len(entities)) * 1000,
        # Add ranking metrics
        "map": ranking_metrics["map"],
        "mrr": ranking_metrics["mrr"],
        "ndcg@20": ranking_metrics["ndcg_at_20"],
        "recall@20": ranking_metrics["recall_at_20"],
        "precision@20": ranking_metrics["precision_at_20"],
    }

    logger.info(
        f"[{name}] Recall={recall:.1%}, Precision={precision:.1%}, F1={f1:.1%}, "
        f"TP={tp}, FN={fn}, FP={fp}"
    )
    logger.info(
        f"[{name}] Ranking: MAP={ranking_metrics['map']:.3f}, MRR={ranking_metrics['mrr']:.3f}, "
        f"NDCG@20={ranking_metrics['ndcg_at_20']:.3f}"
    )

    return results


def print_comparison_table(
    faiss_results: dict[str, Any],
    hybrid_minilm_results: dict[str, Any],
    rerank_minilm_results: dict[str, Any],
    hybrid_qwen3_noinst_results: dict[str, Any],
    hybrid_qwen3_inst_results: dict[str, Any],
) -> None:
    """Print five-way comparison table.

    Args:
        faiss_results: Metrics from FAISS-MiniLM
        hybrid_minilm_results: Metrics from Hybrid-MiniLM
        rerank_minilm_results: Metrics from HybridRerank-MiniLM
        hybrid_qwen3_noinst_results: Metrics from Hybrid-Qwen3-NoInst
        hybrid_qwen3_inst_results: Metrics from Hybrid-Qwen3-WithInst
    """
    print("\n" + "=" * 140)
    print("COMPREHENSIVE RESULTS COMPARISON - FIVE APPROACHES")
    print("=" * 140)

    metrics = [
        ("Recall", "recall", "%"),
        ("Precision", "precision", "%"),
        ("F1 Score", "f1", "%"),
        ("True Positives", "true_positives", ""),
        ("False Negatives", "false_negatives", ""),
        ("False Positives", "false_positives", ""),
        ("Total Candidates", "total_candidates", ""),
        ("--- RANKING METRICS ---", None, ""),
        ("MAP", "map", ".3f"),
        ("MRR", "mrr", ".3f"),
        ("NDCG@20", "ndcg@20", ".3f"),
        ("Recall@20", "recall@20", "%"),
        ("Precision@20", "precision@20", "%"),
        ("--- PERFORMANCE ---", None, ""),
        ("Indexing Time", "indexing_time_seconds", "s"),
        ("Avg Query Time", "avg_query_time_ms", "ms"),
    ]

    # Header
    print(
        f"{'Metric':<25} {'FAISS-ML':>15} {'Hybrid-ML':>15} {'Rerank-ML':>15} "
        f"{'Qwen3-NoI':>15} {'Qwen3-I':>15} {'Best':>15}"
    )
    print("-" * 140)

    for label, key, unit in metrics:
        # Handle section headers
        if key is None:
            print(f"\n{label}")
            continue

        faiss_val = faiss_results[key]
        hybrid_ml_val = hybrid_minilm_results[key]
        rerank_ml_val = rerank_minilm_results[key]
        qwen3_noinst_val = hybrid_qwen3_noinst_results[key]
        qwen3_inst_val = hybrid_qwen3_inst_results[key]

        # Determine best (depends on metric type)
        if key in [
            "recall",
            "precision",
            "f1",
            "true_positives",
            "map",
            "mrr",
            "ndcg@20",
            "recall@20",
            "precision@20",
        ]:
            # Higher is better
            best_val = max(
                faiss_val, hybrid_ml_val, rerank_ml_val, qwen3_noinst_val, qwen3_inst_val
            )
            if faiss_val == best_val:
                best_name = "FAISS-ML"
            elif hybrid_ml_val == best_val:
                best_name = "Hybrid-ML"
            elif rerank_ml_val == best_val:
                best_name = "Rerank-ML"
            elif qwen3_noinst_val == best_val:
                best_name = "Qwen3-NoI"
            else:
                best_name = "Qwen3-I"
        elif key in [
            "false_negatives",
            "false_positives",
            "indexing_time_seconds",
            "avg_query_time_ms",
        ]:
            # Lower is better
            best_val = min(
                faiss_val, hybrid_ml_val, rerank_ml_val, qwen3_noinst_val, qwen3_inst_val
            )
            if faiss_val == best_val:
                best_name = "FAISS-ML"
            elif hybrid_ml_val == best_val:
                best_name = "Hybrid-ML"
            elif rerank_ml_val == best_val:
                best_name = "Rerank-ML"
            elif qwen3_noinst_val == best_val:
                best_name = "Qwen3-NoI"
            else:
                best_name = "Qwen3-I"
        else:
            # Neutral (total_candidates)
            best_name = "-"

        # Format values
        if unit == "%":
            faiss_str = f"{faiss_val * 100:.2f}%"
            hybrid_ml_str = f"{hybrid_ml_val * 100:.2f}%"
            rerank_ml_str = f"{rerank_ml_val * 100:.2f}%"
            qwen3_noinst_str = f"{qwen3_noinst_val * 100:.2f}%"
            qwen3_inst_str = f"{qwen3_inst_val * 100:.2f}%"
        elif unit == ".3f":
            faiss_str = f"{faiss_val:.3f}"
            hybrid_ml_str = f"{hybrid_ml_val:.3f}"
            rerank_ml_str = f"{rerank_ml_val:.3f}"
            qwen3_noinst_str = f"{qwen3_noinst_val:.3f}"
            qwen3_inst_str = f"{qwen3_inst_val:.3f}"
        elif unit in ["s", "ms"]:
            faiss_str = f"{faiss_val:.3f}{unit}"
            hybrid_ml_str = f"{hybrid_ml_val:.3f}{unit}"
            rerank_ml_str = f"{rerank_ml_val:.3f}{unit}"
            qwen3_noinst_str = f"{qwen3_noinst_val:.3f}{unit}"
            qwen3_inst_str = f"{qwen3_inst_val:.3f}{unit}"
        else:
            faiss_str = str(faiss_val)
            hybrid_ml_str = str(hybrid_ml_val)
            rerank_ml_str = str(rerank_ml_val)
            qwen3_noinst_str = str(qwen3_noinst_val)
            qwen3_inst_str = str(qwen3_inst_val)

        # Mark best with ✓
        if best_name == "FAISS-ML":
            faiss_str += " ✓"
        elif best_name == "Hybrid-ML":
            hybrid_ml_str += " ✓"
        elif best_name == "Rerank-ML":
            rerank_ml_str += " ✓"
        elif best_name == "Qwen3-NoI":
            qwen3_noinst_str += " ✓"
        elif best_name == "Qwen3-I":
            qwen3_inst_str += " ✓"

        print(
            f"{label:<25} {faiss_str:>15} {hybrid_ml_str:>15} {rerank_ml_str:>15} "
            f"{qwen3_noinst_str:>15} {qwen3_inst_str:>15} {best_name:>15}"
        )

    print("=" * 140)


def save_results(
    faiss_results: dict[str, Any],
    hybrid_minilm_results: dict[str, Any],
    rerank_minilm_results: dict[str, Any],
    hybrid_qwen3_noinst_results: dict[str, Any],
    hybrid_qwen3_inst_results: dict[str, Any],
    entities: list[dict[str, Any]],
    gold_pairs: set[tuple[str, str]],
    minilm_dim: int,
    qwen3_dim: int,
    cache_info: dict[str, Any],
) -> None:
    """Save evaluation results to JSON file.

    Args:
        faiss_results: Metrics from FAISS-MiniLM
        hybrid_minilm_results: Metrics from Hybrid-MiniLM
        rerank_minilm_results: Metrics from HybridRerank-MiniLM
        hybrid_qwen3_noinst_results: Metrics from Hybrid-Qwen3-NoInst
        hybrid_qwen3_inst_results: Metrics from Hybrid-Qwen3-WithInst
        entities: List of entities evaluated
        gold_pairs: Ground truth pairs
        minilm_dim: MiniLM embedding dimension
        qwen3_dim: Qwen3 embedding dimension
        cache_info: Cache performance info from Qwen3 DiskCachedEmbedder
    """
    output = {
        "dataset": {
            "name": "funder_names_with_ids",
            "total_entities": len(entities),
            "ground_truth_pairs": len(gold_pairs),
        },
        "models": {
            "minilm": {
                "name": MINILM_MODEL,
                "params": "22M",
                "embedding_dim": minilm_dim,
                "instruction_support": False,
                "caching": False,
            },
            "qwen3": {
                "name": QWEN3_MODEL,
                "params": "600M",
                "embedding_dim": qwen3_dim,
                "instruction_support": True,
                "caching": True,
                "cache_dir": str(CACHE_DIR),
                "cache_namespace": CACHE_NAMESPACE,
            },
            "sparse": SPARSE_MODEL,
            "reranking": RERANKING_MODEL,
            "k_neighbors": K_NEIGHBORS,
            "prefetch_limit": PREFETCH_LIMIT,
            "instruction_prompt": EMBEDDING_INSTRUCTION,
        },
        "cache_performance": {
            "hit_rate": cache_info.get("hit_rate", 0.0),
            "hits_hot": cache_info.get("hits_hot", 0),
            "hits_cold": cache_info.get("hits_cold", 0),
            "misses": cache_info.get("misses", 0),
            "total_lookups": cache_info.get("hits_hot", 0)
            + cache_info.get("hits_cold", 0)
            + cache_info.get("misses", 0),
        },
        "experiments": {
            "faiss_minilm": faiss_results,
            "hybrid_minilm": hybrid_minilm_results,
            "hybrid_rerank_minilm": rerank_minilm_results,
            "hybrid_qwen3_noinst": hybrid_qwen3_noinst_results,
            "hybrid_qwen3_inst": hybrid_qwen3_inst_results,
        },
        "comparisons": {
            "sparse_vectors_impact": {
                "description": "FAISS-MiniLM vs Hybrid-MiniLM (adding BM25)",
                "recall_improvement": hybrid_minilm_results["recall"] - faiss_results["recall"],
                "precision_improvement": hybrid_minilm_results["precision"]
                - faiss_results["precision"],
                "f1_improvement": hybrid_minilm_results["f1"] - faiss_results["f1"],
                "time_ratio": hybrid_minilm_results["indexing_time_seconds"]
                / faiss_results["indexing_time_seconds"]
                if faiss_results["indexing_time_seconds"] > 0
                else 0,
            },
            "reranking_impact": {
                "description": "Hybrid-MiniLM vs HybridRerank-MiniLM (adding ColBERT)",
                "recall_improvement": rerank_minilm_results["recall"]
                - hybrid_minilm_results["recall"],
                "precision_improvement": rerank_minilm_results["precision"]
                - hybrid_minilm_results["precision"],
                "f1_improvement": rerank_minilm_results["f1"] - hybrid_minilm_results["f1"],
                "time_ratio": rerank_minilm_results["indexing_time_seconds"]
                / hybrid_minilm_results["indexing_time_seconds"]
                if hybrid_minilm_results["indexing_time_seconds"] > 0
                else 0,
            },
            "model_quality_impact": {
                "description": "Hybrid-MiniLM vs Hybrid-Qwen3-NoInst (advanced model)",
                "recall_improvement": hybrid_qwen3_noinst_results["recall"]
                - hybrid_minilm_results["recall"],
                "precision_improvement": hybrid_qwen3_noinst_results["precision"]
                - hybrid_minilm_results["precision"],
                "f1_improvement": hybrid_qwen3_noinst_results["f1"] - hybrid_minilm_results["f1"],
                "time_ratio": hybrid_qwen3_noinst_results["indexing_time_seconds"]
                / hybrid_minilm_results["indexing_time_seconds"]
                if hybrid_minilm_results["indexing_time_seconds"] > 0
                else 0,
            },
            "instructions_plus_reranking_impact": {
                "description": "Hybrid-Qwen3-NoInst vs HybridRerank-Qwen3-WithInst (adding instructions + reranking)",
                "note": "Combined effect of instructions and reranking on Qwen3",
                "recall_improvement": hybrid_qwen3_inst_results["recall"]
                - hybrid_qwen3_noinst_results["recall"],
                "precision_improvement": hybrid_qwen3_inst_results["precision"]
                - hybrid_qwen3_noinst_results["precision"],
                "f1_improvement": hybrid_qwen3_inst_results["f1"]
                - hybrid_qwen3_noinst_results["f1"],
                "time_ratio": hybrid_qwen3_inst_results["indexing_time_seconds"]
                / hybrid_qwen3_noinst_results["indexing_time_seconds"]
                if hybrid_qwen3_noinst_results["indexing_time_seconds"] > 0
                else 0,
            },
        },
        "recommendations": {
            "best_f1": max(
                [
                    ("FAISS-MiniLM", faiss_results["f1"]),
                    ("Hybrid-MiniLM", hybrid_minilm_results["f1"]),
                    ("HybridRerank-MiniLM", rerank_minilm_results["f1"]),
                    ("Hybrid-Qwen3-NoInst", hybrid_qwen3_noinst_results["f1"]),
                    ("Hybrid-Qwen3-WithInst", hybrid_qwen3_inst_results["f1"]),
                ],
                key=lambda x: x[1],
            )[0],
            "fastest": min(
                [
                    ("FAISS-MiniLM", faiss_results["indexing_time_seconds"]),
                    ("Hybrid-MiniLM", hybrid_minilm_results["indexing_time_seconds"]),
                    ("HybridRerank-MiniLM", rerank_minilm_results["indexing_time_seconds"]),
                    ("Hybrid-Qwen3-NoInst", hybrid_qwen3_noinst_results["indexing_time_seconds"]),
                    ("Hybrid-Qwen3-WithInst", hybrid_qwen3_inst_results["indexing_time_seconds"]),
                ],
                key=lambda x: x[1],
            )[0],
        },
    }

    output_path = Path("examples/blocking_evaluation_results_comprehensive.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def main() -> None:
    """Run comprehensive blocking evaluation comparing 5 approaches."""
    print("=" * 140)
    print("COMPREHENSIVE BLOCKING EVALUATION: Model, Instruction & Architecture Comparison")
    print("=" * 140)
    print(f"\nDataset: Funder organization names (1,741 entities)")
    print(f"\nModels:")
    print(f"  Simple:   {MINILM_MODEL} (384 dims, 22M params, no instruction support)")
    print(f"  Advanced: {QWEN3_MODEL} (1024 dims, 600M params, WITH instruction support)")
    print(f"  Sparse:   {SPARSE_MODEL} (BM25 keyword matching)")
    print(f"  Reranking: {RERANKING_MODEL} (ColBERT late-interaction)")
    print(f"\nParameters:")
    print(f"  k-neighbors: {K_NEIGHBORS}")
    print(f"  Prefetch limit: {PREFETCH_LIMIT}")
    print(f"\nCache (Qwen3 only):")
    print(f"  Directory: {CACHE_DIR}")
    print(f"  Namespace: {CACHE_NAMESPACE}")
    print(f"  Memory size: {MEMORY_CACHE_SIZE:,}")
    print(f"\nInstruction: {EMBEDDING_INSTRUCTION[:80]}...")
    print()

    # Load data
    entities, gold_pairs, gold_clusters = load_funder_data()

    # Setup MiniLM embedder (NO caching)
    logger.info(f"Loading {MINILM_MODEL} (simple model, no caching)...")
    minilm_embedder = SentenceTransformerEmbedder(
        model_name=MINILM_MODEL,
        batch_size=256,
        normalize_embeddings=True,
    )
    minilm_dim = minilm_embedder.embedding_dim
    logger.info(f"MiniLM loaded (dim={minilm_dim})")

    # Setup Qwen3 cached embedder
    logger.info(f"Loading {QWEN3_MODEL} (advanced model, WITH caching)...")
    qwen3_base = SentenceTransformerEmbedder(
        model_name=QWEN3_MODEL,
        batch_size=256,
        normalize_embeddings=True,
    )
    qwen3_cached = DiskCachedEmbedder(
        embedder=qwen3_base,
        cache_dir=CACHE_DIR,
        namespace=CACHE_NAMESPACE,
        memory_cache_size=MEMORY_CACHE_SIZE,
    )
    qwen3_dim = qwen3_cached.embedding_dim
    logger.info(f"Qwen3 loaded with caching (dim={qwen3_dim})")
    logger.info(f"Cache ready: {CACHE_DIR / CACHE_NAMESPACE}")

    # Connect to Qdrant
    qdrant_client = connect_to_qdrant()

    # Setup shared sparse embedder
    sparse_embedder = FastEmbedSparseEmbedder(model_name=SPARSE_MODEL)

    # Setup reranking embedder (used only by approach 3)
    logger.info(f"Loading reranking model: {RERANKING_MODEL}...")
    reranking_embedder = FastEmbedLateInteractionEmbedder(model_name=RERANKING_MODEL)
    logger.info("Reranking model loaded")

    # Create blockers
    print("\n" + "-" * 140)
    print("SETTING UP 5 BLOCKING APPROACHES")
    print("-" * 140)

    # Approach 1: FAISS-MiniLM (baseline)
    logger.info("1. FAISS-MiniLM: Dense-only semantic search (baseline)")
    faiss_blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=FAISSIndex(embedder=minilm_embedder, metric="cosine"),
        k_neighbors=K_NEIGHBORS,
    )

    # Approach 2: Hybrid-MiniLM
    logger.info("2. Hybrid-MiniLM: Dense (MiniLM) + Sparse (BM25)")
    hybrid_minilm_blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=QdrantHybridIndex(
            client=qdrant_client,
            collection_name="funder_hybrid_minilm_comp",
            dense_embedder=minilm_embedder,
            sparse_embedder=sparse_embedder,
            fusion="RRF",
            prefetch_limit=PREFETCH_LIMIT,
        ),
        k_neighbors=K_NEIGHBORS,
    )

    # Approach 3: HybridRerank-MiniLM
    logger.info("3. HybridRerank-MiniLM: Dense (MiniLM) + Sparse (BM25) + Reranking (ColBERT)")
    rerank_minilm_blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=QdrantHybridRerankingIndex(
            client=qdrant_client,
            collection_name="funder_rerank_minilm_comp",
            dense_embedder=minilm_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
            fusion="RRF",
            prefetch_limit=PREFETCH_LIMIT,
        ),
        k_neighbors=K_NEIGHBORS,
    )

    # Approach 4: Hybrid-Qwen3-NoInst
    logger.info("4. Hybrid-Qwen3-NoInst: Dense (Qwen3, no instr) + Sparse (BM25)")
    hybrid_qwen3_noinst_blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=QdrantHybridIndex(
            client=qdrant_client,
            collection_name="funder_hybrid_qwen3_noinst_comp",
            dense_embedder=qwen3_cached,
            sparse_embedder=sparse_embedder,
            fusion="RRF",
            prefetch_limit=PREFETCH_LIMIT,
            # Note: QdrantHybridIndex doesn't store query_prompt in constructor
            # Instructions would need to be passed at search time via index.search(query_prompt=...)
        ),
        k_neighbors=K_NEIGHBORS,
    )

    # Approach 5: Hybrid-Qwen3-WithInst
    # Note: QdrantHybridRerankingIndex supports query_prompt in constructor
    logger.info(
        "5. Hybrid-Qwen3-WithInst: Dense (Qwen3, WITH instr) + Sparse (BM25) + Reranking (ColBERT)"
    )
    hybrid_qwen3_inst_blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=QdrantHybridRerankingIndex(
            client=qdrant_client,
            collection_name="funder_hybrid_qwen3_inst_comp",
            dense_embedder=qwen3_cached,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,  # Use ColBERT for instruction comparison
            fusion="RRF",
            prefetch_limit=PREFETCH_LIMIT,
            query_prompt=EMBEDDING_INSTRUCTION,  # WITH instructions (stored in index)
        ),
        k_neighbors=K_NEIGHBORS,
    )

    # Run evaluations
    print("\n" + "-" * 140)
    print("RUNNING EVALUATIONS")
    print("-" * 140)

    # 1. FAISS-MiniLM
    faiss_results = evaluate_blocking_recall(
        faiss_blocker, entities, gold_pairs, gold_clusters, "FAISS-MiniLM"
    )

    # 2. Hybrid-MiniLM
    hybrid_minilm_results = evaluate_blocking_recall(
        hybrid_minilm_blocker, entities, gold_pairs, gold_clusters, "Hybrid-MiniLM"
    )

    # 3. HybridRerank-MiniLM
    rerank_minilm_results = evaluate_blocking_recall(
        rerank_minilm_blocker, entities, gold_pairs, gold_clusters, "HybridRerank-MiniLM"
    )

    # 4. Hybrid-Qwen3-NoInst
    hybrid_qwen3_noinst_results = evaluate_blocking_recall(
        hybrid_qwen3_noinst_blocker, entities, gold_pairs, gold_clusters, "Hybrid-Qwen3-NoInst"
    )

    # Log Qwen3 cache performance
    cache_info = qwen3_cached.cache_info()
    logger.info(
        f"Qwen3 cache after NoInst: {cache_info['hit_rate']:.1%} hit rate, "
        f"{cache_info['hits_hot'] + cache_info['hits_cold']} hits, {cache_info['misses']} misses"
    )

    # 5. Hybrid-Qwen3-WithInst
    hybrid_qwen3_inst_results = evaluate_blocking_recall(
        hybrid_qwen3_inst_blocker, entities, gold_pairs, gold_clusters, "Hybrid-Qwen3-WithInst"
    )

    # Final cache performance
    cache_info = qwen3_cached.cache_info()
    logger.info(
        f"Qwen3 cache final: {cache_info['hit_rate']:.1%} hit rate, "
        f"{cache_info['hits_hot'] + cache_info['hits_cold']} hits, {cache_info['misses']} misses"
    )

    # Print comparison
    print_comparison_table(
        faiss_results,
        hybrid_minilm_results,
        rerank_minilm_results,
        hybrid_qwen3_noinst_results,
        hybrid_qwen3_inst_results,
    )

    # Save results
    save_results(
        faiss_results,
        hybrid_minilm_results,
        rerank_minilm_results,
        hybrid_qwen3_noinst_results,
        hybrid_qwen3_inst_results,
        entities,
        gold_pairs,
        minilm_dim,
        qwen3_dim,
        cache_info,
    )

    # Print detailed analysis
    print("\n" + "=" * 140)
    print("DETAILED ANALYSIS")
    print("=" * 140)

    # A. Sparse Vectors Impact
    print("\n📊 A. IMPACT OF SPARSE VECTORS (Adding BM25)")
    print(f"   Comparison: FAISS-MiniLM vs Hybrid-MiniLM")
    recall_diff = (hybrid_minilm_results["recall"] - faiss_results["recall"]) * 100
    precision_diff = (hybrid_minilm_results["precision"] - faiss_results["precision"]) * 100
    f1_diff = (hybrid_minilm_results["f1"] - faiss_results["f1"]) * 100
    time_ratio = (
        hybrid_minilm_results["indexing_time_seconds"] / faiss_results["indexing_time_seconds"]
    )
    print(
        f"   Recall:     {recall_diff:+.2f}pp (from {faiss_results['recall'] * 100:.2f}% to {hybrid_minilm_results['recall'] * 100:.2f}%)"
    )
    print(
        f"   Precision:  {precision_diff:+.2f}pp (from {faiss_results['precision'] * 100:.2f}% to {hybrid_minilm_results['precision'] * 100:.2f}%)"
    )
    print(
        f"   F1:         {f1_diff:+.2f}pp (from {faiss_results['f1'] * 100:.2f}% to {hybrid_minilm_results['f1'] * 100:.2f}%)"
    )
    print(f"   Time ratio: {time_ratio:.2f}x slower")
    if recall_diff > 2:
        print(
            f"   ✅ BM25 adds significant recall improvement (+{recall_diff:.1f}pp) - keyword matching helps!"
        )
    else:
        print(f"   ⚠️  BM25 adds modest recall improvement (+{recall_diff:.1f}pp)")

    # B. Reranking Impact
    print("\n🎯 B. IMPACT OF RERANKING (Adding ColBERT)")
    print(f"   Comparison: Hybrid-MiniLM vs HybridRerank-MiniLM")
    recall_diff = (rerank_minilm_results["recall"] - hybrid_minilm_results["recall"]) * 100
    precision_diff = (rerank_minilm_results["precision"] - hybrid_minilm_results["precision"]) * 100
    f1_diff = (rerank_minilm_results["f1"] - hybrid_minilm_results["f1"]) * 100
    time_ratio = (
        rerank_minilm_results["indexing_time_seconds"]
        / hybrid_minilm_results["indexing_time_seconds"]
    )
    print(
        f"   Recall:     {recall_diff:+.2f}pp (from {hybrid_minilm_results['recall'] * 100:.2f}% to {rerank_minilm_results['recall'] * 100:.2f}%)"
    )
    print(
        f"   Precision:  {precision_diff:+.2f}pp (from {hybrid_minilm_results['precision'] * 100:.2f}% to {rerank_minilm_results['precision'] * 100:.2f}%)"
    )
    print(
        f"   F1:         {f1_diff:+.2f}pp (from {hybrid_minilm_results['f1'] * 100:.2f}% to {rerank_minilm_results['f1'] * 100:.2f}%)"
    )
    print(f"   Time ratio: {time_ratio:.1f}x slower")
    if precision_diff > 1:
        print(f"   ✅ ColBERT adds significant precision improvement (+{precision_diff:.1f}pp)")
        if time_ratio > 20:
            print(f"   ⚠️  But costs {time_ratio:.0f}x in performance - use selectively!")
    else:
        print(
            f"   ⚠️  ColBERT adds modest precision improvement (+{precision_diff:.1f}pp) at {time_ratio:.0f}x cost"
        )

    # C. Model Quality Impact
    print("\n🚀 C. IMPACT OF ADVANCED MODEL (MiniLM vs Qwen3)")
    print(f"   Comparison: Hybrid-MiniLM vs Hybrid-Qwen3-NoInst")
    recall_diff = (hybrid_qwen3_noinst_results["recall"] - hybrid_minilm_results["recall"]) * 100
    precision_diff = (
        hybrid_qwen3_noinst_results["precision"] - hybrid_minilm_results["precision"]
    ) * 100
    f1_diff = (hybrid_qwen3_noinst_results["f1"] - hybrid_minilm_results["f1"]) * 100
    time_ratio = (
        hybrid_qwen3_noinst_results["indexing_time_seconds"]
        / hybrid_minilm_results["indexing_time_seconds"]
    )
    print(
        f"   Recall:     {recall_diff:+.2f}pp (from {hybrid_minilm_results['recall'] * 100:.2f}% to {hybrid_qwen3_noinst_results['recall'] * 100:.2f}%)"
    )
    print(
        f"   Precision:  {precision_diff:+.2f}pp (from {hybrid_minilm_results['precision'] * 100:.2f}% to {hybrid_qwen3_noinst_results['precision'] * 100:.2f}%)"
    )
    print(
        f"   F1:         {f1_diff:+.2f}pp (from {hybrid_minilm_results['f1'] * 100:.2f}% to {hybrid_qwen3_noinst_results['f1'] * 100:.2f}%)"
    )
    print(f"   Time ratio: {time_ratio:.2f}x slower")
    print(f"   Model size: 22M params (MiniLM) → 600M params (Qwen3) = 27x larger")
    if f1_diff > 3:
        print(f"   ✅ Qwen3 significantly outperforms MiniLM (+{f1_diff:.1f}pp F1)")
    else:
        print(
            f"   ⚠️  Qwen3 marginally better than MiniLM (+{f1_diff:.1f}pp F1) - may not justify 27x size"
        )

    # D. Instructions + Reranking on Qwen3
    print("\n💡 D. IMPACT OF INSTRUCTIONS + RERANKING (Qwen3 only)")
    print(f"   Comparison: Hybrid-Qwen3-NoInst vs HybridRerank-Qwen3-WithInst")
    print(f"   Note: This shows COMBINED effect of instructions + reranking on Qwen3")
    recall_diff = (
        hybrid_qwen3_inst_results["recall"] - hybrid_qwen3_noinst_results["recall"]
    ) * 100
    precision_diff = (
        hybrid_qwen3_inst_results["precision"] - hybrid_qwen3_noinst_results["precision"]
    ) * 100
    f1_diff = (hybrid_qwen3_inst_results["f1"] - hybrid_qwen3_noinst_results["f1"]) * 100
    time_ratio = (
        hybrid_qwen3_inst_results["indexing_time_seconds"]
        / hybrid_qwen3_noinst_results["indexing_time_seconds"]
    )
    print(
        f"   Recall:     {recall_diff:+.2f}pp (from {hybrid_qwen3_noinst_results['recall'] * 100:.2f}% to {hybrid_qwen3_inst_results['recall'] * 100:.2f}%)"
    )
    print(
        f"   Precision:  {precision_diff:+.2f}pp (from {hybrid_qwen3_noinst_results['precision'] * 100:.2f}% to {hybrid_qwen3_inst_results['precision'] * 100:.2f}%)"
    )
    print(
        f"   F1:         {f1_diff:+.2f}pp (from {hybrid_qwen3_noinst_results['f1'] * 100:.2f}% to {hybrid_qwen3_inst_results['f1'] * 100:.2f}%)"
    )
    print(f"   Time ratio: {time_ratio:.1f}x (adds reranking overhead)")
    if f1_diff > 2:
        print(f"   ✅ Instructions + reranking significantly improve Qwen3 (+{f1_diff:.1f}pp F1)")
    else:
        print(f"   ⚠️  Instructions + reranking have modest impact on Qwen3 (+{f1_diff:.1f}pp F1)")

    # E. Cache Performance
    print("\n💾 E. CACHE PERFORMANCE (Qwen3 only)")
    print(f"   Hit rate: {cache_info['hit_rate']:.1%}")
    print(f"   Hot cache hits: {cache_info['hits_hot']:,}")
    print(f"   Cold cache hits: {cache_info['hits_cold']:,}")
    print(f"   Cache misses: {cache_info['misses']:,}")
    print(
        f"   Total lookups: {cache_info['hits_hot'] + cache_info['hits_cold'] + cache_info['misses']:,}"
    )
    if cache_info["hit_rate"] > 0.5:
        print(f"   ✅ Caching is highly effective! Second run will be ~100x faster for embeddings")
    else:
        print(f"   💡 First run - caching will accelerate subsequent runs")

    # F. Model Specifications
    print("\n" + "=" * 140)
    print("MODEL SPECIFICATIONS")
    print("=" * 140)
    print(
        f"\n{'Model':<30} {'Type':<12} {'Params':<10} {'Dims':<8} {'Instructions':<15} {'Caching':<10}"
    )
    print("-" * 140)
    print(
        f"{'all-MiniLM-L6-v2':<30} {'Dense':<12} {'22M':<10} {minilm_dim:<8} {'❌ No':<15} {'❌ No':<10}"
    )
    print(
        f"{'Qwen3-0.6B':<30} {'Dense':<12} {'600M':<10} {qwen3_dim:<8} {'✅ Yes':<15} {'✅ Yes':<10}"
    )
    print(f"{'Qdrant/bm25':<30} {'Sparse':<12} {'N/A':<10} {'Sparse':<8} {'❌ No':<15} {'N/A':<10}")
    print(
        f"{'colbert-ir/colbertv2.0':<30} {'Reranking':<12} {'~110M':<10} {'Multi':<8} {'❌ No':<15} {'N/A':<10}"
    )

    # G. Recommendations
    print("\n" + "=" * 140)
    print("RECOMMENDATIONS")
    print("=" * 140)

    best_f1 = max(
        faiss_results["f1"],
        hybrid_minilm_results["f1"],
        rerank_minilm_results["f1"],
        hybrid_qwen3_noinst_results["f1"],
        hybrid_qwen3_inst_results["f1"],
    )

    if hybrid_qwen3_inst_results["f1"] == best_f1:
        print("\n✅ BEST OVERALL: Hybrid-Qwen3-WithInst")
        print("   - Highest F1 score")
        print("   - Advanced model + instructions + keyword matching")
        print("   - No reranking overhead")
        print("   - Caching makes iteration fast")
    elif rerank_minilm_results["f1"] == best_f1:
        print("\n✅ BEST OVERALL: HybridRerank-MiniLM")
        print("   - Highest F1 score with simple model")
        print("   - ColBERT reranking improves precision")
        print("   - Warning: 20-40x slower than non-reranked approaches")

    print("\n📋 Use Case Recommendations:")
    print("   • Fast prototyping:       FAISS-MiniLM (fastest, good baseline)")
    print("   • Production baseline:    Hybrid-MiniLM (adds keywords, modest overhead)")
    print("   • Quality-critical:       Hybrid-Qwen3-WithInst (best quality without reranking)")
    print("   • Maximum quality:        HybridRerank-MiniLM (if you can afford latency)")
    print("   • Budget-constrained:     Hybrid-MiniLM (best recall/cost trade-off)")

    print("\n" + "=" * 140)
    print("✅ Evaluation complete!")
    print("=" * 140)


if __name__ == "__main__":
    main()
