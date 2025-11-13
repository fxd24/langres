"""Blocking Evaluation: FAISS vs Qdrant Hybrid vs Qdrant Reranking.

This example compares three vector indexing approaches for candidate generation:
1. FAISS with dense vectors only (semantic search baseline)
2. Qdrant with dense + sparse vectors (hybrid search with configurable RRF/DBSF fusion)
3. Qdrant with dense + sparse + fusion + late-interaction reranking (ColBERT MaxSim)

All approaches use the same Qwen3 embedding model for dense vectors, allowing
us to isolate the impact of adding sparse vectors (BM25), fusion methods, and
late-interaction reranking (ColBERT token-level similarity).

Dataset: 1,741 real-world funder organization names with ground truth labels.

Metrics evaluated:
- Blocking recall: % of true duplicate pairs captured
- Precision: % of candidate pairs that are true duplicates
- F1 score: Harmonic mean of precision and recall
- Performance: Indexing time and average query latency

Expected findings:
- Recall: Reranking should achieve highest recall (~84-85%)
- Precision: Biggest improvement expected for reranking (~5-8% vs 3-4%)
- F1: Reranking should significantly improve F1 score (~9-12% vs ~7%)
- Indexing time: Reranking will be slowest (~2-3x hybrid, ~20-40x FAISS)

Trade-off analysis:
- FAISS: Fastest, good baseline, but misses keyword matches
- Hybrid: Better recall via keyword matching, moderate overhead, configurable fusion
- Reranking: Best precision/recall via token-level similarity, highest cost

Usage:
    python examples/blocking_evaluation_with_reranking.py
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
    FastEmbedLateInteractionEmbedder,
    FastEmbedSparseEmbedder,
    SentenceTransformerEmbedder,
)
from langres.core.hybrid_vector_index import QdrantHybridIndex
from langres.core.metrics import pairs_from_clusters
from langres.core.reranking_vector_index import QdrantHybridRerankingIndex
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
RERANKING_MODEL = "colbert-ir/colbertv2.0"  # ColBERT late-interaction
CROSSENCODER_MODEL = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"  # Qwen3 cross-encoder
K_NEIGHBORS = 20  # Number of neighbors to retrieve
PREFETCH_LIMIT = 20  # For Qdrant hybrid/reranking prefetch
CROSSENCODER_PREFETCH = 100  # Fetch more candidates for client-side cross-encoder reranking


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
    dense_embedder: SentenceTransformerEmbedder,
) -> VectorBlocker[OrganizationSchema]:
    """Setup FAISS blocker with dense vectors only.

    Args:
        dense_embedder: Qwen3 embedding provider

    Returns:
        VectorBlocker configured with FAISS index
    """
    logger.info("Setting up FAISS blocker (dense-only)...")

    # FAISS index with cosine similarity
    faiss_index = FAISSIndex(embedder=dense_embedder, metric="cosine")

    # VectorBlocker with FAISS
    blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=faiss_index,
        k_neighbors=K_NEIGHBORS,
    )

    logger.info(f"FAISS blocker ready (k={K_NEIGHBORS}, metric=cosine)")
    return blocker


def setup_qdrant_hybrid_blocker(
    dense_embedder: SentenceTransformerEmbedder,
    qdrant_client: QdrantClient,
) -> VectorBlocker[OrganizationSchema]:
    """Setup Qdrant blocker with dense + sparse hybrid search.

    Args:
        dense_embedder: Qwen3 embedding provider (for dense vectors)
        qdrant_client: Connected Qdrant client

    Returns:
        VectorBlocker configured with Qdrant hybrid index
    """
    logger.info("Setting up Qdrant blocker (dense + sparse hybrid)...")

    # Sparse embedder (BM25)
    sparse_embedder = FastEmbedSparseEmbedder(model_name=SPARSE_MODEL)

    # Qdrant hybrid index with RRF fusion
    qdrant_index = QdrantHybridIndex(
        client=qdrant_client,
        collection_name="funder_names_hybrid_eval",
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        fusion="RRF",  # Reciprocal Rank Fusion
        prefetch_limit=PREFETCH_LIMIT,
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
        f"fusion=RRF, sparse={SPARSE_MODEL})"
    )
    return blocker


def setup_qdrant_reranking_blocker(
    dense_embedder: SentenceTransformerEmbedder,
    qdrant_client: QdrantClient,
) -> VectorBlocker[OrganizationSchema]:
    """Setup Qdrant blocker with dense + sparse + fusion + late-interaction reranking.

    This uses a 4-stage pipeline:
    1. Dense prefetch (semantic similarity)
    2. Sparse prefetch (keyword matching)
    3. Fusion (RRF combines dense + sparse results)
    4. Late-interaction reranking (ColBERT token-level MaxSim)

    Args:
        dense_embedder: Qwen3 embedding provider (for dense vectors)
        qdrant_client: Connected Qdrant client

    Returns:
        VectorBlocker configured with Qdrant reranking index
    """
    logger.info("Setting up Qdrant reranking blocker (dense + sparse + fusion + reranking)...")

    # Sparse embedder (BM25)
    sparse_embedder = FastEmbedSparseEmbedder(model_name=SPARSE_MODEL)

    # Late-interaction embedder (ColBERT)
    reranking_embedder = FastEmbedLateInteractionEmbedder(model_name=RERANKING_MODEL)

    # Qdrant reranking index with RRF fusion
    # The 4-stage pipeline: dense prefetch → sparse prefetch → RRF fusion → ColBERT reranking
    qdrant_index = QdrantHybridRerankingIndex(
        client=qdrant_client,
        collection_name="funder_names_reranking_eval",
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        reranking_embedder=reranking_embedder,
        fusion="RRF",  # Reciprocal Rank Fusion (can also use "DBSF")
        prefetch_limit=PREFETCH_LIMIT,
    )

    # VectorBlocker with Qdrant reranking
    blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=qdrant_index,
        k_neighbors=K_NEIGHBORS,
    )

    logger.info(
        f"Qdrant reranking blocker ready (k={K_NEIGHBORS}, prefetch={PREFETCH_LIMIT}, "
        f"sparse={SPARSE_MODEL}, reranking={RERANKING_MODEL})"
    )
    return blocker


def evaluate_crossencoder_blocking(
    dense_embedder: SentenceTransformerEmbedder,
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
        dense_embedder: Qwen3 embedding provider (for dense vectors)
        qdrant_client: Connected Qdrant client
        cross_encoder: Loaded CrossEncoder model
        entities: List of entity dicts to process
        gold_pairs: Set of ground truth duplicate pairs
        name: Name for logging (e.g., "Qwen3 CrossEncoder")

    Returns:
        dict with evaluation metrics
    """
    logger.info(f"[{name}] Setting up hybrid index for initial retrieval...")

    # Sparse embedder (BM25)
    sparse_embedder = FastEmbedSparseEmbedder(model_name=SPARSE_MODEL)

    # Qdrant hybrid index (NO reranking - just dense + sparse)
    qdrant_index = QdrantHybridIndex(
        client=qdrant_client,
        collection_name="funder_names_crossencoder_eval",
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        fusion="RRF",  # Reciprocal Rank Fusion
        prefetch_limit=CROSSENCODER_PREFETCH,  # Fetch MORE for client-side reranking
    )

    # Build index
    texts = [entity["name"] for entity in entities]
    qdrant_index.create_index(texts)

    logger.info(f"[{name}] Generating candidates with CrossEncoder reranking...")

    # Generate candidates and measure time
    start_time = time.time()

    # For each entity, get top-N candidates and rerank
    all_candidate_pairs = set()

    for entity in entities:
        query_text = entity["name"]
        query_id = entity["id"]

        # Step 1: Get top-N candidates from hybrid search
        distances, indices = qdrant_index.search(query_text, k=CROSSENCODER_PREFETCH)

        # Step 2: Fetch candidate texts (skip self-matches)
        candidate_texts = []
        candidate_ids = []
        for idx in indices:
            if idx >= 0 and idx < len(entities):  # Valid index
                cand_id = entities[idx]["id"]
                if cand_id != query_id:  # Skip self
                    candidate_texts.append(entities[idx]["name"])
                    candidate_ids.append(cand_id)

        if not candidate_texts:
            continue

        # Step 3: Format pairs for CrossEncoder
        # CrossEncoder expects list of [query, document] pairs
        pairs = [[query_text, cand_text] for cand_text in candidate_texts]

        # Step 4: Score with CrossEncoder
        scores = cross_encoder.predict(pairs)

        # Step 5: Get top-k after reranking
        top_k_idx = np.argsort(scores)[-K_NEIGHBORS:][::-1]  # Descending order

        # Step 6: Create candidate pairs
        for idx in top_k_idx:
            if idx < len(candidate_ids):
                pair = tuple(sorted([query_id, candidate_ids[idx]]))
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
    }

    logger.info(
        f"[{name}] Recall={recall:.1%}, Precision={precision:.1%}, F1={f1:.1%}, "
        f"TP={tp}, FN={fn}, FP={fp}"
    )

    return results


def evaluate_blocking_recall(
    blocker: VectorBlocker[OrganizationSchema],
    entities: list[dict[str, Any]],
    gold_pairs: set[tuple[str, str]],
    name: str,
) -> dict[str, Any]:
    """Evaluate blocking recall, precision, F1, and performance.

    Args:
        blocker: VectorBlocker instance to evaluate
        entities: List of entity dicts to process
        gold_pairs: Set of ground truth duplicate pairs
        name: Name for logging (e.g., "FAISS", "Qdrant Hybrid", "Qdrant Reranking")

    Returns:
        dict with evaluation metrics
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
    }

    logger.info(
        f"[{name}] Recall={recall:.1%}, Precision={precision:.1%}, F1={f1:.1%}, "
        f"TP={tp}, FN={fn}, FP={fp}"
    )

    return results


def print_comparison_table(
    faiss_results: dict[str, Any],
    qdrant_hybrid_results: dict[str, Any],
    qdrant_reranking_results: dict[str, Any],
    qdrant_crossencoder_results: dict[str, Any],
) -> None:
    """Print four-way comparison table.

    Args:
        faiss_results: Metrics from FAISS evaluation
        qdrant_hybrid_results: Metrics from Qdrant hybrid evaluation
        qdrant_reranking_results: Metrics from ColBERT reranking evaluation
        qdrant_crossencoder_results: Metrics from Qwen3 CrossEncoder evaluation
    """
    print("\n" + "=" * 140)
    print("RESULTS COMPARISON - FOUR APPROACHES")
    print("=" * 140)

    metrics = [
        ("Recall", "recall", "%"),
        ("Precision", "precision", "%"),
        ("F1 Score", "f1", "%"),
        ("True Positives", "true_positives", ""),
        ("False Negatives", "false_negatives", ""),
        ("False Positives", "false_positives", ""),
        ("Total Candidates", "total_candidates", ""),
        ("Indexing Time", "indexing_time_seconds", "s"),
        ("Avg Query Time", "avg_query_time_ms", "ms"),
    ]

    # Header
    print(
        f"{'Metric':<25} {'FAISS':>15} {'Hybrid':>15} {'ColBERT':>15} {'Qwen3 CE':>15} {'Best':>15}"
    )
    print("-" * 140)

    for label, key, unit in metrics:
        faiss_val = faiss_results[key]
        hybrid_val = qdrant_hybrid_results[key]
        rerank_val = qdrant_reranking_results[key]
        crossenc_val = qdrant_crossencoder_results[key]

        # Determine best (depends on metric type)
        if key in ["recall", "precision", "f1", "true_positives"]:
            # Higher is better
            best_val = max(faiss_val, hybrid_val, rerank_val, crossenc_val)
            if faiss_val == best_val:
                best_name = "FAISS"
            elif hybrid_val == best_val:
                best_name = "Hybrid"
            elif rerank_val == best_val:
                best_name = "ColBERT"
            else:
                best_name = "Qwen3 CE"
        elif key in [
            "false_negatives",
            "false_positives",
            "indexing_time_seconds",
            "avg_query_time_ms",
        ]:
            # Lower is better
            best_val = min(faiss_val, hybrid_val, rerank_val, crossenc_val)
            if faiss_val == best_val:
                best_name = "FAISS"
            elif hybrid_val == best_val:
                best_name = "Hybrid"
            elif rerank_val == best_val:
                best_name = "ColBERT"
            else:
                best_name = "Qwen3 CE"
        else:
            # Neutral (total_candidates)
            best_name = "-"

        # Format values
        if unit == "%":
            faiss_str = f"{faiss_val * 100:.2f}%"
            hybrid_str = f"{hybrid_val * 100:.2f}%"
            rerank_str = f"{rerank_val * 100:.2f}%"
            crossenc_str = f"{crossenc_val * 100:.2f}%"
        elif unit in ["s", "ms"]:
            faiss_str = f"{faiss_val:.3f}{unit}"
            hybrid_str = f"{hybrid_val:.3f}{unit}"
            rerank_str = f"{rerank_val:.3f}{unit}"
            crossenc_str = f"{crossenc_val:.3f}{unit}"
        else:
            faiss_str = str(faiss_val)
            hybrid_str = str(hybrid_val)
            rerank_str = str(rerank_val)
            crossenc_str = str(crossenc_val)

        # Mark best with ✓
        if best_name == "FAISS":
            faiss_str += " ✓"
        elif best_name == "Hybrid":
            hybrid_str += " ✓"
        elif best_name == "ColBERT":
            rerank_str += " ✓"
        elif best_name == "Qwen3 CE":
            crossenc_str += " ✓"

        print(
            f"{label:<25} {faiss_str:>15} {hybrid_str:>15} {rerank_str:>15} {crossenc_str:>15} {best_name:>15}"
        )

    print("=" * 140)


def save_results(
    faiss_results: dict[str, Any],
    qdrant_hybrid_results: dict[str, Any],
    qdrant_reranking_results: dict[str, Any],
    qdrant_crossencoder_results: dict[str, Any],
    entities: list[dict[str, Any]],
    gold_pairs: set[tuple[str, str]],
    embedding_dim: int,
) -> None:
    """Save evaluation results to JSON file.

    Args:
        faiss_results: Metrics from FAISS evaluation
        qdrant_hybrid_results: Metrics from Qdrant hybrid evaluation
        qdrant_reranking_results: Metrics from ColBERT reranking evaluation
        qdrant_crossencoder_results: Metrics from Qwen3 CrossEncoder evaluation
        entities: List of entities evaluated
        gold_pairs: Ground truth pairs
        embedding_dim: Embedding dimension
    """
    # Calculate rankings
    recall_scores = [
        ("FAISS", faiss_results["recall"]),
        ("Qdrant Hybrid", qdrant_hybrid_results["recall"]),
        ("ColBERT Reranking", qdrant_reranking_results["recall"]),
        ("Qwen3 CrossEncoder", qdrant_crossencoder_results["recall"]),
    ]
    recall_ranking = [name for name, _ in sorted(recall_scores, key=lambda x: x[1], reverse=True)]

    precision_scores = [
        ("FAISS", faiss_results["precision"]),
        ("Qdrant Hybrid", qdrant_hybrid_results["precision"]),
        ("ColBERT Reranking", qdrant_reranking_results["precision"]),
        ("Qwen3 CrossEncoder", qdrant_crossencoder_results["precision"]),
    ]
    precision_ranking = [
        name for name, _ in sorted(precision_scores, key=lambda x: x[1], reverse=True)
    ]

    speed_scores = [
        ("FAISS", faiss_results["indexing_time_seconds"]),
        ("Qdrant Hybrid", qdrant_hybrid_results["indexing_time_seconds"]),
        ("ColBERT Reranking", qdrant_reranking_results["indexing_time_seconds"]),
        ("Qwen3 CrossEncoder", qdrant_crossencoder_results["indexing_time_seconds"]),
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
            "reranking_model": RERANKING_MODEL,
            "crossencoder_model": CROSSENCODER_MODEL,
            "embedding_dim": embedding_dim,
            "k_neighbors": K_NEIGHBORS,
            "prefetch_limit": PREFETCH_LIMIT,
            "crossencoder_prefetch": CROSSENCODER_PREFETCH,
        },
        "experiments": {
            "faiss_dense_only": faiss_results,
            "qdrant_hybrid": qdrant_hybrid_results,
            "colbert_reranking": qdrant_reranking_results,
            "qwen3_crossencoder": qdrant_crossencoder_results,
        },
        "comparison": {
            "recall_ranking": recall_ranking,
            "precision_ranking": precision_ranking,
            "speed_ranking": speed_ranking,
            "recall_improvements": {
                "hybrid_vs_faiss": qdrant_hybrid_results["recall"] - faiss_results["recall"],
                "colbert_vs_faiss": qdrant_reranking_results["recall"] - faiss_results["recall"],
                "colbert_vs_hybrid": qdrant_reranking_results["recall"]
                - qdrant_hybrid_results["recall"],
                "crossencoder_vs_faiss": qdrant_crossencoder_results["recall"]
                - faiss_results["recall"],
                "crossencoder_vs_hybrid": qdrant_crossencoder_results["recall"]
                - qdrant_hybrid_results["recall"],
                "crossencoder_vs_colbert": qdrant_crossencoder_results["recall"]
                - qdrant_reranking_results["recall"],
            },
            "precision_improvements": {
                "hybrid_vs_faiss": qdrant_hybrid_results["precision"] - faiss_results["precision"],
                "colbert_vs_faiss": qdrant_reranking_results["precision"]
                - faiss_results["precision"],
                "colbert_vs_hybrid": qdrant_reranking_results["precision"]
                - qdrant_hybrid_results["precision"],
                "crossencoder_vs_faiss": qdrant_crossencoder_results["precision"]
                - faiss_results["precision"],
                "crossencoder_vs_hybrid": qdrant_crossencoder_results["precision"]
                - qdrant_hybrid_results["precision"],
                "crossencoder_vs_colbert": qdrant_crossencoder_results["precision"]
                - qdrant_reranking_results["precision"],
            },
            "f1_improvements": {
                "hybrid_vs_faiss": qdrant_hybrid_results["f1"] - faiss_results["f1"],
                "colbert_vs_faiss": qdrant_reranking_results["f1"] - faiss_results["f1"],
                "colbert_vs_hybrid": qdrant_reranking_results["f1"] - qdrant_hybrid_results["f1"],
                "crossencoder_vs_faiss": qdrant_crossencoder_results["f1"] - faiss_results["f1"],
                "crossencoder_vs_hybrid": qdrant_crossencoder_results["f1"]
                - qdrant_hybrid_results["f1"],
                "crossencoder_vs_colbert": qdrant_crossencoder_results["f1"]
                - qdrant_reranking_results["f1"],
            },
            "indexing_time_ratios": {
                "hybrid_vs_faiss": (
                    qdrant_hybrid_results["indexing_time_seconds"]
                    / faiss_results["indexing_time_seconds"]
                    if faiss_results["indexing_time_seconds"] > 0
                    else 0
                ),
                "colbert_vs_faiss": (
                    qdrant_reranking_results["indexing_time_seconds"]
                    / faiss_results["indexing_time_seconds"]
                    if faiss_results["indexing_time_seconds"] > 0
                    else 0
                ),
                "colbert_vs_hybrid": (
                    qdrant_reranking_results["indexing_time_seconds"]
                    / qdrant_hybrid_results["indexing_time_seconds"]
                    if qdrant_hybrid_results["indexing_time_seconds"] > 0
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
                "crossencoder_vs_colbert": (
                    qdrant_crossencoder_results["indexing_time_seconds"]
                    / qdrant_reranking_results["indexing_time_seconds"]
                    if qdrant_reranking_results["indexing_time_seconds"] > 0
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
                        ("ColBERT Reranking", qdrant_reranking_results["f1"]),
                        ("Qwen3 CrossEncoder", qdrant_crossencoder_results["f1"]),
                    ],
                    key=lambda x: x[1],
                )[0],
            },
        },
    }

    output_path = Path("examples/blocking_evaluation_results_reranking.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def main() -> None:
    """Run blocking evaluation comparing FAISS vs Qdrant hybrid vs Qdrant reranking vs Qwen3 CrossEncoder."""
    print("=" * 120)
    print("BLOCKING EVALUATION: FAISS vs Qdrant Hybrid vs ColBERT Reranking vs Qwen3 CrossEncoder")
    print("=" * 120)
    print(f"\nDataset: Funder organization names")
    print(f"Dense model: {DENSE_MODEL}")
    print(f"Sparse model: {SPARSE_MODEL}")
    print(f"ColBERT reranking model: {RERANKING_MODEL}")
    print(f"CrossEncoder model: {CROSSENCODER_MODEL}")
    print(f"k-neighbors: {K_NEIGHBORS}")
    print(f"Prefetch limit: {PREFETCH_LIMIT}")
    print(f"CrossEncoder prefetch: {CROSSENCODER_PREFETCH}")
    print()

    # Load data
    entities, gold_pairs = load_funder_data()

    # Setup shared Qwen3 embedder (used by all approaches for dense vectors)
    logger.info(f"Loading {DENSE_MODEL} embedding model...")
    dense_embedder = SentenceTransformerEmbedder(
        model_name=DENSE_MODEL,
        batch_size=32,
        normalize_embeddings=True,
    )
    embedding_dim = dense_embedder.embedding_dim
    logger.info(f"Embedding model loaded (dim={embedding_dim})")

    # Connect to Qdrant cloud
    qdrant_client = connect_to_qdrant()

    # Load CrossEncoder for client-side reranking
    logger.info(f"Loading CrossEncoder model: {CROSSENCODER_MODEL}...")
    cross_encoder = CrossEncoder(CROSSENCODER_MODEL)
    logger.info("CrossEncoder model loaded")

    # Setup blockers
    faiss_blocker = setup_faiss_blocker(dense_embedder)
    qdrant_hybrid_blocker = setup_qdrant_hybrid_blocker(dense_embedder, qdrant_client)
    qdrant_reranking_blocker = setup_qdrant_reranking_blocker(dense_embedder, qdrant_client)

    # Run evaluations
    print("\n" + "-" * 120)
    print("RUNNING EVALUATIONS")
    print("-" * 120)

    faiss_results = evaluate_blocking_recall(faiss_blocker, entities, gold_pairs, "FAISS")

    qdrant_hybrid_results = evaluate_blocking_recall(
        qdrant_hybrid_blocker, entities, gold_pairs, "Qdrant Hybrid"
    )

    qdrant_reranking_results = evaluate_blocking_recall(
        qdrant_reranking_blocker, entities, gold_pairs, "ColBERT Reranking"
    )

    qdrant_crossencoder_results = evaluate_crossencoder_blocking(
        dense_embedder, qdrant_client, cross_encoder, entities, gold_pairs, "Qwen3 CrossEncoder"
    )

    # Print comparison
    print_comparison_table(
        faiss_results, qdrant_hybrid_results, qdrant_reranking_results, qdrant_crossencoder_results
    )

    # Save results
    save_results(
        faiss_results,
        qdrant_hybrid_results,
        qdrant_reranking_results,
        qdrant_crossencoder_results,
        entities,
        gold_pairs,
        embedding_dim,
    )

    # Print key insights
    print("\n" + "=" * 140)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 140)

    # Recall comparison
    print("\n📊 RECALL (% of true duplicates found):")
    faiss_recall = faiss_results["recall"] * 100
    hybrid_recall = qdrant_hybrid_results["recall"] * 100
    rerank_recall = qdrant_reranking_results["recall"] * 100
    crossenc_recall = qdrant_crossencoder_results["recall"] * 100

    print(f"   FAISS (dense-only):              {faiss_recall:.2f}%")
    print(
        f"   Qdrant Hybrid (dense+sparse):    {hybrid_recall:.2f}% "
        f"({hybrid_recall - faiss_recall:+.2f}pp vs FAISS)"
    )
    print(
        f"   ColBERT Reranking (late-int):    {rerank_recall:.2f}% "
        f"({rerank_recall - faiss_recall:+.2f}pp vs FAISS, "
        f"{rerank_recall - hybrid_recall:+.2f}pp vs Hybrid)"
    )
    print(
        f"   Qwen3 CrossEncoder (full attn):  {crossenc_recall:.2f}% "
        f"({crossenc_recall - faiss_recall:+.2f}pp vs FAISS, "
        f"{crossenc_recall - rerank_recall:+.2f}pp vs ColBERT)"
    )

    # Precision comparison
    print("\n🎯 PRECISION (% of candidates that are true duplicates):")
    faiss_precision = faiss_results["precision"] * 100
    hybrid_precision = qdrant_hybrid_results["precision"] * 100
    rerank_precision = qdrant_reranking_results["precision"] * 100
    crossenc_precision = qdrant_crossencoder_results["precision"] * 100

    print(f"   FAISS (dense-only):              {faiss_precision:.2f}%")
    print(
        f"   Qdrant Hybrid (dense+sparse):    {hybrid_precision:.2f}% "
        f"({hybrid_precision - faiss_precision:+.2f}pp vs FAISS)"
    )
    print(
        f"   ColBERT Reranking (late-int):    {rerank_precision:.2f}% "
        f"({rerank_precision - faiss_precision:+.2f}pp vs FAISS, "
        f"{rerank_precision - hybrid_precision:+.2f}pp vs Hybrid)"
    )
    print(
        f"   Qwen3 CrossEncoder (full attn):  {crossenc_precision:.2f}% "
        f"({crossenc_precision - faiss_precision:+.2f}pp vs FAISS, "
        f"{crossenc_precision - rerank_precision:+.2f}pp vs ColBERT)"
    )

    # F1 comparison
    print("\n⚖️  F1 SCORE (harmonic mean of precision & recall):")
    faiss_f1 = faiss_results["f1"] * 100
    hybrid_f1 = qdrant_hybrid_results["f1"] * 100
    rerank_f1 = qdrant_reranking_results["f1"] * 100
    crossenc_f1 = qdrant_crossencoder_results["f1"] * 100

    print(f"   FAISS (dense-only):              {faiss_f1:.2f}%")
    print(
        f"   Qdrant Hybrid (dense+sparse):    {hybrid_f1:.2f}% "
        f"({hybrid_f1 - faiss_f1:+.2f}pp vs FAISS)"
    )
    print(
        f"   ColBERT Reranking (late-int):    {rerank_f1:.2f}% "
        f"({rerank_f1 - faiss_f1:+.2f}pp vs FAISS, {rerank_f1 - hybrid_f1:+.2f}pp vs Hybrid)"
    )
    print(
        f"   Qwen3 CrossEncoder (full attn):  {crossenc_f1:.2f}% "
        f"({crossenc_f1 - faiss_f1:+.2f}pp vs FAISS, {crossenc_f1 - rerank_f1:+.2f}pp vs ColBERT)"
    )

    # Performance comparison
    print("\n⏱️  INDEXING TIME:")
    faiss_time = faiss_results["indexing_time_seconds"]
    hybrid_time = qdrant_hybrid_results["indexing_time_seconds"]
    rerank_time = qdrant_reranking_results["indexing_time_seconds"]
    crossenc_time = qdrant_crossencoder_results["indexing_time_seconds"]

    print(f"   FAISS (dense-only):              {faiss_time:.2f}s")
    print(
        f"   Qdrant Hybrid (dense+sparse):    {hybrid_time:.2f}s "
        f"({hybrid_time / faiss_time:.1f}x slower)"
    )
    print(
        f"   ColBERT Reranking (late-int):    {rerank_time:.2f}s "
        f"({rerank_time / faiss_time:.1f}x slower than FAISS, "
        f"{rerank_time / hybrid_time:.1f}x slower than Hybrid)"
    )
    print(
        f"   Qwen3 CrossEncoder (full attn):  {crossenc_time:.2f}s "
        f"({crossenc_time / faiss_time:.1f}x slower than FAISS, "
        f"{crossenc_time / rerank_time:.1f}x vs ColBERT)"
    )

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")

    best_f1 = max(faiss_f1, hybrid_f1, rerank_f1, crossenc_f1)
    if crossenc_f1 == best_f1:
        print(
            "   ✅ Qwen3 CrossEncoder achieved best F1 score - full cross-attention "
            "provides deepest understanding of query-document relevance"
        )
    elif rerank_f1 == best_f1:
        print(
            "   ✅ ColBERT Reranking achieved best F1 score - token-level similarity "
            "improves both recall and precision"
        )
    elif hybrid_f1 == best_f1:
        print(
            "   ✅ Qdrant Hybrid achieved best F1 score - keyword matching improves "
            "recall without reranking overhead"
        )
    else:
        print("   ⚠️  FAISS achieved best F1 score - simpler approach may be sufficient")

    # CrossEncoder vs ColBERT comparison
    if crossenc_precision > rerank_precision + 1.0:
        print(
            f"   ✅ CrossEncoder significantly improved precision over ColBERT "
            f"(+{crossenc_precision - rerank_precision:.2f}pp) - deeper attention helps"
        )
    elif rerank_precision > hybrid_precision + 1.0:
        print(
            f"   ✅ ColBERT improved precision over Hybrid (+{rerank_precision - hybrid_precision:.2f}pp) "
            "- late-interaction adds value"
        )
    else:
        print(
            "   ⚠️  Reranking did not significantly improve precision - "
            "hybrid may be better trade-off"
        )

    # Performance warnings
    if crossenc_time / faiss_time > 20:
        print(
            f"   ⚠️  CrossEncoder is {crossenc_time / faiss_time:.0f}x slower than FAISS - "
            "consider cost vs. quality trade-off"
        )
    elif rerank_time / faiss_time > 20:
        print(
            f"   ⚠️  ColBERT is {rerank_time / faiss_time:.0f}x slower than FAISS - "
            "consider cost vs. quality trade-off"
        )

    # CrossEncoder vs ColBERT comparison insight
    if crossenc_f1 > rerank_f1:
        print(
            f"   📊 CrossEncoder outperforms ColBERT (F1: {crossenc_f1:.2f}% vs {rerank_f1:.2f}%) - "
            "full cross-attention > late-interaction for this task"
        )
    else:
        print(
            f"   📊 ColBERT matches/exceeds CrossEncoder (F1: {rerank_f1:.2f}% vs {crossenc_f1:.2f}%) - "
            "late-interaction sufficient, no need for expensive cross-encoder"
        )

    print("\n   Use Case Recommendations:")
    print("   • FAISS: Fast baseline, good for initial prototyping")
    print("   • Hybrid: Best recall/cost trade-off, good for production")
    print("   • ColBERT: Late-interaction reranking, balanced quality/cost")
    print("   • CrossEncoder: Maximum quality via full cross-attention, highest cost")

    print("=" * 140)
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
