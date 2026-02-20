"""Blocking Evaluation: FAISS vs Qdrant Hybrid vs CrossEncoder with Caching & Instructions.

This example compares three vector indexing approaches for candidate generation,
demonstrating the impact of disk caching and instruction prompts:
1. FAISS with dense vectors only (semantic search baseline)
2. Qdrant with dense + sparse vectors (hybrid search with RRF fusion)
3. Qdrant with client-side Jina CrossEncoder reranking (fast, quality reranking)

All approaches use the same Qwen3 embedding model with:
- **DiskCachedEmbedder**: Persistent caching for instant re-runs
- **Instruction prompts**: Task-specific instructions for improved matching quality

Dataset: 1,741 real-world funder organization names with ground truth labels.

Metrics evaluated:
- Blocking recall: % of true duplicate pairs captured
- Precision: % of candidate pairs that are true duplicates
- F1 score: Harmonic mean of precision and recall
- Ranking metrics: MAP, MRR, NDCG@20, Recall@20, Precision@20
- Performance: Indexing time and average query latency
- Cache performance: Hit rate, hits, misses

Expected findings:
- Recall: CrossEncoder should achieve highest recall (~84-85%)
- Precision: CrossEncoder significantly improves precision (~5-8% vs 3-4%)
- F1: CrossEncoder should significantly improve F1 score (~9-12% vs ~7%)
- Indexing time: CrossEncoder will be slower, but much faster than ColBERT
- Caching: Second run should be near-instant (embeddings cached)

Trade-off analysis:
- FAISS: Fastest, good baseline, but misses keyword matches
- Hybrid: Better recall via keyword matching, moderate overhead
- CrossEncoder: Best precision/recall, moderate cost with fast Jina model

Performance improvements:
- DiskCachedEmbedder: Second run is ~100x faster (instant cache lookup)
- Instruction prompts: Expected +1-5% quality improvement
- Jina reranker: 10-15x faster than Qwen3 reranker (4-6 minutes vs 1 hour)
- ONNX backend: Additional 2-3x speedup for inference

Usage:
    python examples/blocking_evaluation_with_instructions.py
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from langres.clients.settings import Settings
from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import (
    DiskCachedEmbedder,
    FastEmbedSparseEmbedder,
    SentenceTransformerEmbedder,
)
from langres.core.hybrid_vector_index import QdrantHybridIndex
from langres.core.metrics import evaluate_blocking_with_ranking, pairs_from_clusters
from langres.core.vector_index import FAISSIndex
from langres.data import load_labeled_dedup_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model configuration - all easily changeable
DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # Dense semantic embeddings
SPARSE_MODEL = "Qdrant/bm25"  # BM25 keyword matching
CROSSENCODER_MODEL = "jinaai/jina-reranker-v1-tiny-en"  # Fast Jina reranker (33M params)
K_NEIGHBORS = 20  # Number of neighbors to retrieve
PREFETCH_LIMIT = 20  # For Qdrant hybrid prefetch
CROSSENCODER_PREFETCH = 100  # Fetch more candidates for client-side cross-encoder reranking
CROSSENCODER_BATCH_SIZE = 256  # Larger batch size for fast Jina model

# Cache configuration
CACHE_DIR = Path("tmp/cache")  # Persistent cache directory
CACHE_NAMESPACE = "qwen3_funder_names_with_instructions"  # Unique namespace for this task
MEMORY_CACHE_SIZE = 10_000  # Keep 10k most recent embeddings in memory

# Instruction prompts for Qwen3 models
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
    # Convert LabeledGroup objects to sets of entity_ids
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


def setup_faiss_blocker(
    dense_embedder: DiskCachedEmbedder,
) -> VectorBlocker[OrganizationSchema]:
    """Setup FAISS blocker with dense vectors only.

    Args:
        dense_embedder: Cached Qwen3 embedding provider

    Returns:
        VectorBlocker configured with FAISS index
    """
    logger.info("Setting up FAISS blocker (dense-only with instructions)...")

    # FAISS index with cosine similarity and instruction prompts
    faiss_index = FAISSIndex(
        embedder=dense_embedder,
        metric="cosine",
        query_prompt=EMBEDDING_INSTRUCTION,  # Apply instructions to queries
    )

    # VectorBlocker with FAISS
    blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=faiss_index,
        k_neighbors=K_NEIGHBORS,
    )

    logger.info(f"FAISS blocker ready (k={K_NEIGHBORS}, metric=cosine, with instructions)")
    return blocker


def setup_qdrant_hybrid_blocker(
    dense_embedder: DiskCachedEmbedder,
    qdrant_client: QdrantClient,
) -> VectorBlocker[OrganizationSchema]:
    """Setup Qdrant blocker with dense + sparse hybrid search.

    Args:
        dense_embedder: Cached Qwen3 embedding provider (for dense vectors)
        qdrant_client: Connected Qdrant client

    Returns:
        VectorBlocker configured with Qdrant hybrid index
    """
    logger.info("Setting up Qdrant blocker (dense + sparse hybrid with instructions)...")

    # Sparse embedder (BM25)
    sparse_embedder = FastEmbedSparseEmbedder(model_name=SPARSE_MODEL)

    # Qdrant hybrid index with RRF fusion and instruction prompts
    qdrant_index = QdrantHybridIndex(
        client=qdrant_client,
        collection_name="funder_names_hybrid_instructions_eval",
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        fusion="RRF",  # Reciprocal Rank Fusion
        prefetch_limit=PREFETCH_LIMIT,
        query_prompt=EMBEDDING_INSTRUCTION,  # Apply instructions to queries
    )

    # VectorBlocker with Qdrant hybrid
    blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=qdrant_index,
        k_neighbors=K_NEIGHBORS,
    )

    logger.info(
        f"Qdrant hybrid blocker ready (k={K_NEIGHBORS}, prefetch={PREFETCH_LIMIT}, "
        f"fusion=RRF, sparse={SPARSE_MODEL}, with instructions)"
    )
    return blocker


def evaluate_crossencoder_blocking(
    dense_embedder: DiskCachedEmbedder,
    qdrant_client: QdrantClient,
    cross_encoder: CrossEncoder,
    entities: list[dict[str, Any]],
    gold_pairs: set[tuple[str, str]],
    name: str,
) -> dict[str, Any]:
    """Evaluate blocking with client-side cross-encoder reranking.

    This implements a 3-stage pipeline:
    1. Dense + sparse hybrid search in Qdrant (get top-N candidates)
    2. Fetch candidate texts from Qdrant
    3. Rerank with CrossEncoder on client side

    Args:
        dense_embedder: Cached Qwen3 embedding provider (for dense vectors)
        qdrant_client: Connected Qdrant client
        cross_encoder: Loaded CrossEncoder model
        entities: List of entity dicts to process
        gold_pairs: Set of ground truth duplicate pairs
        name: Name for logging (e.g., "Jina CrossEncoder")

    Returns:
        dict with evaluation metrics
    """
    logger.info(f"[{name}] Setting up hybrid index for initial retrieval...")

    # Sparse embedder (BM25)
    sparse_embedder = FastEmbedSparseEmbedder(model_name=SPARSE_MODEL)

    # Qdrant hybrid index (NO reranking - just dense + sparse)
    qdrant_index = QdrantHybridIndex(
        client=qdrant_client,
        collection_name="funder_names_crossencoder_instructions_eval",
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        fusion="RRF",  # Reciprocal Rank Fusion
        prefetch_limit=CROSSENCODER_PREFETCH,  # Fetch MORE for client-side reranking
        query_prompt=EMBEDDING_INSTRUCTION,  # Apply instructions to queries
    )

    # Build index
    texts = [entity["name"] for entity in entities]
    qdrant_index.create_index(texts)

    logger.info(f"[{name}] Retrieving candidates from hybrid search...")

    # Generate candidates and measure time
    start_time = time.time()

    # PHASE 1: Collect all query-candidate pairs from hybrid search
    all_pairs_to_score = []  # List of (query_text, cand_text, query_id, cand_id)

    for entity in entities:
        query_text = entity["name"]
        query_id = entity["id"]

        # Get top-N candidates from hybrid search
        distances, indices = qdrant_index.search(query_text, k=CROSSENCODER_PREFETCH)

        # Collect candidate texts (skip self-matches)
        for idx in indices:
            if idx >= 0 and idx < len(entities):  # Valid index
                cand_id = entities[idx]["id"]
                if cand_id != query_id:  # Skip self
                    all_pairs_to_score.append(
                        (query_text, entities[idx]["name"], query_id, cand_id)
                    )

    logger.info(f"[{name}] Scoring {len(all_pairs_to_score)} pairs with CrossEncoder in batch...")

    # PHASE 2: Score ALL pairs in one batch (MUCH more efficient!)
    all_candidate_pairs = set()

    if all_pairs_to_score:
        # Format all pairs as [[query, doc], [query, doc], ...]
        formatted_pairs = [
            [query_text, cand_text] for query_text, cand_text, _, _ in all_pairs_to_score
        ]

        # Score everything in one batch call with optimized batch size
        logger.info(
            f"[{name}] Scoring with batch_size={CROSSENCODER_BATCH_SIZE} "
            f"({len(formatted_pairs) // CROSSENCODER_BATCH_SIZE + 1} batches expected)..."
        )
        all_scores = cross_encoder.predict(
            formatted_pairs,
            batch_size=CROSSENCODER_BATCH_SIZE,
            show_progress_bar=True,
        )

        # PHASE 3: Group scores by query and select top-k per query
        query_results: dict[str, list[tuple[float, str]]] = {}  # query_id -> [(score, cand_id)]

        for (query_text, cand_text, query_id, cand_id), score in zip(
            all_pairs_to_score, all_scores
        ):
            if query_id not in query_results:
                query_results[query_id] = []
            query_results[query_id].append((score, cand_id))

        # Select top-k candidates for each query
        for query_id, scored_candidates in query_results.items():
            # Sort by score descending and take top-k
            top_k_candidates = sorted(scored_candidates, reverse=True)[:K_NEIGHBORS]
            for _, cand_id in top_k_candidates:
                pair = tuple(sorted([query_id, cand_id]))
                all_candidate_pairs.add(pair)

    indexing_time = time.time() - start_time

    logger.info(
        f"[{name}] Generated {len(all_candidate_pairs)} candidate pairs in {indexing_time:.2f}s"
    )

    # Normalize gold pairs (ensure sorted tuples)
    gold_normalized = {tuple(sorted(pair)) for pair in gold_pairs}

    # Calculate metrics
    tp = len(gold_normalized & all_candidate_pairs)  # True positives
    fn = len(gold_normalized - all_candidate_pairs)  # False negatives (missed pairs)
    fp = len(all_candidate_pairs - gold_normalized)  # False positives (wrong pairs)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "total_candidates": len(all_candidate_pairs),
        "indexing_time_seconds": indexing_time,
        "avg_query_time_ms": (indexing_time / len(entities)) * 1000,
        # Note: CrossEncoder path doesn't produce ERCandidate objects with scores,
        # so ranking metrics are not computed. Set to 0.0 for consistency.
        "map": 0.0,
        "mrr": 0.0,
        "ndcg@20": 0.0,
        "recall@20": 0.0,
        "precision@20": 0.0,
    }

    logger.info(
        f"[{name}] Recall={recall:.1%}, Precision={precision:.1%}, F1={f1:.1%}, "
        f"TP={tp}, FN={fn}, FP={fp}"
    )
    logger.info(
        f"[{name}] Note: Ranking metrics not computed for CrossEncoder path (different code path)"
    )

    return results


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
        name: Name for logging (e.g., "FAISS", "Qdrant Hybrid")

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
        # Add ranking metrics (note: function returns underscore keys like "ndcg_at_20")
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
        f"NDCG@20={ranking_metrics['ndcg_at_20']:.3f}, Recall@20={ranking_metrics['recall_at_20']:.1%}"
    )

    return results


def print_comparison_table(
    faiss_results: dict[str, Any],
    qdrant_hybrid_results: dict[str, Any],
    qdrant_crossencoder_results: dict[str, Any],
) -> None:
    """Print three-way comparison table.

    Args:
        faiss_results: Metrics from FAISS evaluation
        qdrant_hybrid_results: Metrics from Qdrant hybrid evaluation
        qdrant_crossencoder_results: Metrics from Jina CrossEncoder evaluation
    """
    print("\n" + "=" * 120)
    print("RESULTS COMPARISON - THREE APPROACHES")
    print("=" * 120)

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
    print(f"{'Metric':<25} {'FAISS':>15} {'Hybrid':>15} {'Jina CE':>15} {'Best':>15}")
    print("-" * 120)

    for label, key, unit in metrics:
        # Handle section headers
        if key is None:
            print(f"\n{label}")
            continue

        faiss_val = faiss_results[key]
        hybrid_val = qdrant_hybrid_results[key]
        crossenc_val = qdrant_crossencoder_results[key]

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
            best_val = max(faiss_val, hybrid_val, crossenc_val)
            if faiss_val == best_val:
                best_name = "FAISS"
            elif hybrid_val == best_val:
                best_name = "Hybrid"
            else:
                best_name = "Jina CE"
        elif key in [
            "false_negatives",
            "false_positives",
            "indexing_time_seconds",
            "avg_query_time_ms",
        ]:
            # Lower is better
            best_val = min(faiss_val, hybrid_val, crossenc_val)
            if faiss_val == best_val:
                best_name = "FAISS"
            elif hybrid_val == best_val:
                best_name = "Hybrid"
            else:
                best_name = "Jina CE"
        else:
            # Neutral (total_candidates)
            best_name = "-"

        # Format values
        if unit == "%":
            faiss_str = f"{faiss_val * 100:.2f}%"
            hybrid_str = f"{hybrid_val * 100:.2f}%"
            crossenc_str = f"{crossenc_val * 100:.2f}%"
        elif unit == ".3f":
            faiss_str = f"{faiss_val:.3f}"
            hybrid_str = f"{hybrid_val:.3f}"
            crossenc_str = f"{crossenc_val:.3f}"
        elif unit in ["s", "ms"]:
            faiss_str = f"{faiss_val:.3f}{unit}"
            hybrid_str = f"{hybrid_val:.3f}{unit}"
            crossenc_str = f"{crossenc_val:.3f}{unit}"
        else:
            faiss_str = str(faiss_val)
            hybrid_str = str(hybrid_val)
            crossenc_str = str(crossenc_val)

        # Mark best with ✓
        if best_name == "FAISS":
            faiss_str += " ✓"
        elif best_name == "Hybrid":
            hybrid_str += " ✓"
        elif best_name == "Jina CE":
            crossenc_str += " ✓"

        print(f"{label:<25} {faiss_str:>15} {hybrid_str:>15} {crossenc_str:>15} {best_name:>15}")

    print("=" * 120)


def save_results(
    faiss_results: dict[str, Any],
    qdrant_hybrid_results: dict[str, Any],
    qdrant_crossencoder_results: dict[str, Any],
    entities: list[dict[str, Any]],
    gold_pairs: set[tuple[str, str]],
    embedding_dim: int,
    cache_info: dict[str, Any],
) -> None:
    """Save evaluation results to JSON file.

    Args:
        faiss_results: Metrics from FAISS evaluation
        qdrant_hybrid_results: Metrics from Qdrant hybrid evaluation
        qdrant_crossencoder_results: Metrics from Jina CrossEncoder evaluation
        entities: List of entities evaluated
        gold_pairs: Ground truth pairs
        embedding_dim: Embedding dimension
        cache_info: Cache performance info from DiskCachedEmbedder
    """
    # Calculate rankings
    recall_scores = [
        ("FAISS", faiss_results["recall"]),
        ("Qdrant Hybrid", qdrant_hybrid_results["recall"]),
        ("Jina CrossEncoder", qdrant_crossencoder_results["recall"]),
    ]
    recall_ranking = [name for name, _ in sorted(recall_scores, key=lambda x: x[1], reverse=True)]

    precision_scores = [
        ("FAISS", faiss_results["precision"]),
        ("Qdrant Hybrid", qdrant_hybrid_results["precision"]),
        ("Jina CrossEncoder", qdrant_crossencoder_results["precision"]),
    ]
    precision_ranking = [
        name for name, _ in sorted(precision_scores, key=lambda x: x[1], reverse=True)
    ]

    speed_scores = [
        ("FAISS", faiss_results["indexing_time_seconds"]),
        ("Qdrant Hybrid", qdrant_hybrid_results["indexing_time_seconds"]),
        ("Jina CrossEncoder", qdrant_crossencoder_results["indexing_time_seconds"]),
    ]
    speed_ranking = [name for name, _ in sorted(speed_scores, key=lambda x: x[1])]

    output = {
        "dataset": {
            "name": "funder_names_with_ids",
            "total_entities": len(entities),
            "ground_truth_pairs": len(gold_pairs),
        },
        "models": {
            "dense_model": DENSE_MODEL,
            "sparse_model": SPARSE_MODEL,
            "crossencoder_model": CROSSENCODER_MODEL,
            "embedding_dim": embedding_dim,
            "k_neighbors": K_NEIGHBORS,
            "prefetch_limit": PREFETCH_LIMIT,
            "crossencoder_prefetch": CROSSENCODER_PREFETCH,
            "embedding_instruction": EMBEDDING_INSTRUCTION,
        },
        "caching": {
            "cache_dir": str(CACHE_DIR),
            "namespace": CACHE_NAMESPACE,
            "memory_cache_size": MEMORY_CACHE_SIZE,
            "hit_rate": cache_info.get("hit_rate", 0.0),
            "hits_hot": cache_info.get("hits_hot", 0),
            "hits_cold": cache_info.get("hits_cold", 0),
            "misses": cache_info.get("misses", 0),
            "total_lookups": cache_info.get("hits_hot", 0)
            + cache_info.get("hits_cold", 0)
            + cache_info.get("misses", 0),
        },
        "experiments": {
            "faiss_dense_only": faiss_results,
            "qdrant_hybrid": qdrant_hybrid_results,
            "jina_crossencoder": qdrant_crossencoder_results,
        },
        "comparison": {
            "recall_ranking": recall_ranking,
            "precision_ranking": precision_ranking,
            "speed_ranking": speed_ranking,
            "recall_improvements": {
                "hybrid_vs_faiss": qdrant_hybrid_results["recall"] - faiss_results["recall"],
                "crossencoder_vs_faiss": qdrant_crossencoder_results["recall"]
                - faiss_results["recall"],
                "crossencoder_vs_hybrid": qdrant_crossencoder_results["recall"]
                - qdrant_hybrid_results["recall"],
            },
            "precision_improvements": {
                "hybrid_vs_faiss": qdrant_hybrid_results["precision"] - faiss_results["precision"],
                "crossencoder_vs_faiss": qdrant_crossencoder_results["precision"]
                - faiss_results["precision"],
                "crossencoder_vs_hybrid": qdrant_crossencoder_results["precision"]
                - qdrant_hybrid_results["precision"],
            },
            "f1_improvements": {
                "hybrid_vs_faiss": qdrant_hybrid_results["f1"] - faiss_results["f1"],
                "crossencoder_vs_faiss": qdrant_crossencoder_results["f1"] - faiss_results["f1"],
                "crossencoder_vs_hybrid": qdrant_crossencoder_results["f1"]
                - qdrant_hybrid_results["f1"],
            },
            "indexing_time_ratios": {
                "hybrid_vs_faiss": (
                    qdrant_hybrid_results["indexing_time_seconds"]
                    / faiss_results["indexing_time_seconds"]
                    if faiss_results["indexing_time_seconds"] > 0
                    else 0
                ),
                "crossencoder_vs_faiss": (
                    qdrant_crossencoder_results["indexing_time_seconds"]
                    / faiss_results["indexing_time_seconds"]
                    if faiss_results["indexing_time_seconds"] > 0
                    else 0
                ),
                "crossencoder_vs_hybrid": (
                    qdrant_crossencoder_results["indexing_time_seconds"]
                    / qdrant_hybrid_results["indexing_time_seconds"]
                    if qdrant_hybrid_results["indexing_time_seconds"] > 0
                    else 0
                ),
            },
            "recommendations": {
                "best_recall": recall_ranking[0],
                "best_precision": precision_ranking[0],
                "fastest": speed_ranking[0],
                "best_f1": max(
                    [
                        ("FAISS", faiss_results["f1"]),
                        ("Qdrant Hybrid", qdrant_hybrid_results["f1"]),
                        ("Jina CrossEncoder", qdrant_crossencoder_results["f1"]),
                    ],
                    key=lambda x: x[1],
                )[0],
            },
        },
    }

    output_path = Path("examples/blocking_evaluation_results_with_instructions.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def main() -> None:
    """Run blocking evaluation comparing FAISS vs Qdrant hybrid vs Jina CrossEncoder."""
    print("=" * 120)
    print("BLOCKING EVALUATION: FAISS vs Qdrant Hybrid vs Jina CrossEncoder")
    print("With DiskCachedEmbedder and Instruction Prompts")
    print("=" * 120)
    print(f"\nDataset: Funder organization names")
    print(f"Dense model: {DENSE_MODEL}")
    print(f"Sparse model: {SPARSE_MODEL}")
    print(f"CrossEncoder model: {CROSSENCODER_MODEL}")
    print(f"k-neighbors: {K_NEIGHBORS}")
    print(f"Prefetch limit: {PREFETCH_LIMIT}")
    print(f"CrossEncoder prefetch: {CROSSENCODER_PREFETCH}")
    print(f"CrossEncoder batch size: {CROSSENCODER_BATCH_SIZE}")
    print(f"\nCache directory: {CACHE_DIR}")
    print(f"Cache namespace: {CACHE_NAMESPACE}")
    print(f"Memory cache size: {MEMORY_CACHE_SIZE:,}")
    print(f"\nEmbedding instruction: {EMBEDDING_INSTRUCTION[:80]}...")
    print()

    # Load data
    entities, gold_pairs, gold_clusters = load_funder_data()

    # Setup cached Qwen3 embedder with disk caching
    logger.info(f"Loading {DENSE_MODEL} embedding model with disk caching...")
    base_embedder = SentenceTransformerEmbedder(
        model_name=DENSE_MODEL,
        batch_size=256,
        normalize_embeddings=True,
    )

    cached_embedder = DiskCachedEmbedder(
        embedder=base_embedder,
        cache_dir=CACHE_DIR,
        namespace=CACHE_NAMESPACE,
        memory_cache_size=MEMORY_CACHE_SIZE,
    )

    embedding_dim = cached_embedder.embedding_dim
    logger.info(f"Embedding model loaded (dim={embedding_dim})")
    logger.info(f"Cache ready: {CACHE_DIR / CACHE_NAMESPACE}")

    # Connect to Qdrant cloud
    qdrant_client = connect_to_qdrant()

    # Load CrossEncoder with ONNX backend for fast inference
    logger.info(f"Loading CrossEncoder model: {CROSSENCODER_MODEL} with ONNX backend...")
    cross_encoder = CrossEncoder(CROSSENCODER_MODEL, backend="onnx")
    device_name = getattr(cross_encoder, "device", "unknown")
    logger.info(f"CrossEncoder loaded on device: {device_name} (ONNX-optimized)")

    # Setup blockers
    faiss_blocker = setup_faiss_blocker(cached_embedder)
    qdrant_hybrid_blocker = setup_qdrant_hybrid_blocker(cached_embedder, qdrant_client)

    # Run evaluations
    print("\n" + "-" * 120)
    print("RUNNING EVALUATIONS")
    print("-" * 120)

    faiss_results = evaluate_blocking_recall(
        faiss_blocker, entities, gold_pairs, gold_clusters, "FAISS"
    )

    # Log cache performance after first evaluation
    cache_info_dict = cached_embedder.cache_info()
    logger.info(
        f"Cache performance after FAISS: {cache_info_dict['hit_rate']:.1%} hit rate, "
        f"{cache_info_dict['hits_hot'] + cache_info_dict['hits_cold']} hits, "
        f"{cache_info_dict['misses']} misses"
    )

    qdrant_hybrid_results = evaluate_blocking_recall(
        qdrant_hybrid_blocker, entities, gold_pairs, gold_clusters, "Qdrant Hybrid"
    )

    # Log cache performance after second evaluation
    cache_info_dict = cached_embedder.cache_info()
    logger.info(
        f"Cache performance after Hybrid: {cache_info_dict['hit_rate']:.1%} hit rate, "
        f"{cache_info_dict['hits_hot'] + cache_info_dict['hits_cold']} hits, "
        f"{cache_info_dict['misses']} misses"
    )

    qdrant_crossencoder_results = evaluate_crossencoder_blocking(
        cached_embedder, qdrant_client, cross_encoder, entities, gold_pairs, "Jina CrossEncoder"
    )

    # Final cache performance
    cache_info_dict = cached_embedder.cache_info()
    logger.info(
        f"Final cache performance: {cache_info_dict['hit_rate']:.1%} hit rate, "
        f"{cache_info_dict['hits_hot'] + cache_info_dict['hits_cold']} hits, "
        f"{cache_info_dict['misses']} misses"
    )

    # Print comparison
    print_comparison_table(faiss_results, qdrant_hybrid_results, qdrant_crossencoder_results)

    # Save results
    save_results(
        faiss_results,
        qdrant_hybrid_results,
        qdrant_crossencoder_results,
        entities,
        gold_pairs,
        embedding_dim,
        cache_info_dict,
    )

    # Print key insights
    print("\n" + "=" * 120)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 120)

    # Recall comparison
    print("\n📊 RECALL (% of true duplicates found):")
    faiss_recall = faiss_results["recall"] * 100
    hybrid_recall = qdrant_hybrid_results["recall"] * 100
    crossenc_recall = qdrant_crossencoder_results["recall"] * 100

    print(f"   FAISS (dense-only):              {faiss_recall:.2f}%")
    print(
        f"   Qdrant Hybrid (dense+sparse):    {hybrid_recall:.2f}% "
        f"({hybrid_recall - faiss_recall:+.2f}pp vs FAISS)"
    )
    print(
        f"   Jina CrossEncoder (full attn):   {crossenc_recall:.2f}% "
        f"({crossenc_recall - faiss_recall:+.2f}pp vs FAISS, "
        f"{crossenc_recall - hybrid_recall:+.2f}pp vs Hybrid)"
    )

    # Precision comparison
    print("\n🎯 PRECISION (% of candidates that are true duplicates):")
    faiss_precision = faiss_results["precision"] * 100
    hybrid_precision = qdrant_hybrid_results["precision"] * 100
    crossenc_precision = qdrant_crossencoder_results["precision"] * 100

    print(f"   FAISS (dense-only):              {faiss_precision:.2f}%")
    print(
        f"   Qdrant Hybrid (dense+sparse):    {hybrid_precision:.2f}% "
        f"({hybrid_precision - faiss_precision:+.2f}pp vs FAISS)"
    )
    print(
        f"   Jina CrossEncoder (full attn):   {crossenc_precision:.2f}% "
        f"({crossenc_precision - faiss_precision:+.2f}pp vs FAISS, "
        f"{crossenc_precision - hybrid_precision:+.2f}pp vs Hybrid)"
    )

    # F1 comparison
    print("\n⚖️  F1 SCORE (harmonic mean of precision & recall):")
    faiss_f1 = faiss_results["f1"] * 100
    hybrid_f1 = qdrant_hybrid_results["f1"] * 100
    crossenc_f1 = qdrant_crossencoder_results["f1"] * 100

    print(f"   FAISS (dense-only):              {faiss_f1:.2f}%")
    print(
        f"   Qdrant Hybrid (dense+sparse):    {hybrid_f1:.2f}% "
        f"({hybrid_f1 - faiss_f1:+.2f}pp vs FAISS)"
    )
    print(
        f"   Jina CrossEncoder (full attn):   {crossenc_f1:.2f}% "
        f"({crossenc_f1 - faiss_f1:+.2f}pp vs FAISS, {crossenc_f1 - hybrid_f1:+.2f}pp vs Hybrid)"
    )

    # Performance comparison
    print("\n⏱️  INDEXING TIME:")
    faiss_time = faiss_results["indexing_time_seconds"]
    hybrid_time = qdrant_hybrid_results["indexing_time_seconds"]
    crossenc_time = qdrant_crossencoder_results["indexing_time_seconds"]

    print(f"   FAISS (dense-only):              {faiss_time:.2f}s")
    print(
        f"   Qdrant Hybrid (dense+sparse):    {hybrid_time:.2f}s "
        f"({hybrid_time / faiss_time:.1f}x slower)"
    )
    print(
        f"   Jina CrossEncoder (full attn):   {crossenc_time:.2f}s "
        f"({crossenc_time / faiss_time:.1f}x slower than FAISS, "
        f"{crossenc_time / hybrid_time:.1f}x vs Hybrid)"
    )

    # Cache performance insights
    print("\n💾 CACHE PERFORMANCE:")
    print(f"   Hit rate: {cache_info_dict['hit_rate']:.1%}")
    print(f"   Hot cache hits: {cache_info_dict['hits_hot']:,}")
    print(f"   Cold cache hits: {cache_info_dict['hits_cold']:,}")
    print(f"   Cache misses: {cache_info_dict['misses']:,}")
    print(
        f"   Total lookups: {cache_info_dict['hits_hot'] + cache_info_dict['hits_cold'] + cache_info_dict['misses']:,}"
    )
    print(f"   💡 Second run will be ~100x faster with {cache_info_dict['hit_rate']:.0%} cached!")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")

    best_f1 = max(faiss_f1, hybrid_f1, crossenc_f1)
    if crossenc_f1 == best_f1:
        print(
            "   ✅ Jina CrossEncoder achieved best F1 score - full cross-attention "
            "provides deepest understanding of query-document relevance"
        )
    elif hybrid_f1 == best_f1:
        print(
            "   ✅ Qdrant Hybrid achieved best F1 score - keyword matching improves "
            "recall without reranking overhead"
        )
    else:
        print("   ⚠️  FAISS achieved best F1 score - simpler approach may be sufficient")

    # CrossEncoder vs Hybrid comparison
    if crossenc_precision > hybrid_precision + 1.0:
        print(
            f"   ✅ CrossEncoder significantly improved precision over Hybrid "
            f"(+{crossenc_precision - hybrid_precision:.2f}pp) - reranking adds value"
        )
    else:
        print(
            "   ⚠️  CrossEncoder did not significantly improve precision over Hybrid - "
            "hybrid may be better trade-off"
        )

    print("\n   Use Case Recommendations:")
    print("   • FAISS: Fast baseline, good for initial prototyping")
    print("   • Hybrid: Best recall/cost trade-off, good for production")
    print(
        "   • Jina CrossEncoder: Maximum quality via reranking, moderate cost (10-15x faster than ColBERT)"
    )
    print("   • DiskCachedEmbedder: Essential for iteration - second run is ~100x faster")
    print("   • Instruction prompts: Expected +1-5% quality improvement")

    print("\n=" * 120)
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
