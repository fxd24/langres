"""Blocking Evaluation: FAISS (dense-only) vs Qdrant (dense + sparse hybrid).

This example compares two vector indexing approaches for candidate generation:
1. FAISS with dense vectors only (semantic search)
2. Qdrant with dense + sparse vectors (hybrid search with RRF fusion)

Both use the same Qwen3 embedding model for dense vectors, allowing us to
isolate the impact of adding sparse (BM25) vectors to the search.

Dataset: 1,741 real-world funder organization names with ground truth labels.

Metrics evaluated:
- Blocking recall: % of true duplicate pairs captured
- Precision: % of candidate pairs that are true duplicates
- F1 score: Harmonic mean of precision and recall
- Performance: Indexing time and average query latency

Usage:
    python examples/blocking_evaluation_faiss_vs_qdrant.py
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from langres.clients.settings import Settings
from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import FastEmbedSparseEmbedder, SentenceTransformerEmbedder
from langres.core.hybrid_vector_index import QdrantHybridIndex
from langres.core.metrics import pairs_from_clusters
from langres.core.vector_index import FAISSIndex
from langres.data import load_labeled_dedup_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # Tested in tmp/test_qwen3_org_embeddings.py
K_NEIGHBORS = 20  # Number of neighbors to retrieve


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
    qwen_embedder: SentenceTransformerEmbedder,
) -> VectorBlocker[OrganizationSchema]:
    """Setup FAISS blocker with dense vectors only.

    Args:
        qwen_embedder: Qwen3 embedding provider

    Returns:
        VectorBlocker configured with FAISS index
    """
    logger.info("Setting up FAISS blocker (dense-only)...")

    # FAISS index with cosine similarity
    faiss_index = FAISSIndex(embedder=qwen_embedder, metric="cosine")

    # VectorBlocker with FAISS
    blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=faiss_index,
        k_neighbors=K_NEIGHBORS,
    )

    logger.info(f"FAISS blocker ready (k={K_NEIGHBORS}, metric=cosine)")
    return blocker


def setup_qdrant_blocker(
    qwen_embedder: SentenceTransformerEmbedder, qdrant_client: QdrantClient
) -> VectorBlocker[OrganizationSchema]:
    """Setup Qdrant blocker with dense + sparse hybrid search.

    Args:
        qwen_embedder: Qwen3 embedding provider (for dense vectors)
        qdrant_client: Connected Qdrant client

    Returns:
        VectorBlocker configured with Qdrant hybrid index
    """
    logger.info("Setting up Qdrant blocker (dense + sparse hybrid)...")

    # Sparse embedder (BM25)
    bm25_sparse = FastEmbedSparseEmbedder(model_name="Qdrant/bm25")

    # Qdrant hybrid index with RRF fusion
    qdrant_index = QdrantHybridIndex(
        client=qdrant_client,
        collection_name="funder_names_blocking_eval",
        dense_embedder=qwen_embedder,
        sparse_embedder=bm25_sparse,
        fusion="RRF",  # Reciprocal Rank Fusion
        prefetch_limit=20,
    )

    # VectorBlocker with Qdrant
    blocker = VectorBlocker(
        schema_factory=lambda x: OrganizationSchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=qdrant_index,
        k_neighbors=K_NEIGHBORS,
    )

    logger.info(f"Qdrant blocker ready (k={K_NEIGHBORS}, fusion=RRF, sparse=BM25)")
    return blocker


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
        name: Name for logging (e.g., "FAISS", "Qdrant")

    Returns:
        dict with evaluation metrics
    """
    logger.info(f"[{name}] Generating candidates...")

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


def print_comparison_table(faiss_results: dict[str, Any], qdrant_results: dict[str, Any]) -> None:
    """Print side-by-side comparison table.

    Args:
        faiss_results: Metrics from FAISS evaluation
        qdrant_results: Metrics from Qdrant evaluation
    """
    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)

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

    print(f"{'Metric':<25} {'FAISS':>15} {'Qdrant Hybrid':>20} {'Δ':>15}")
    print("-" * 90)

    for label, key, unit in metrics:
        faiss_val = faiss_results[key]
        qdrant_val = qdrant_results[key]

        if unit == "%":
            faiss_str = f"{faiss_val * 100:.2f}%"
            qdrant_str = f"{qdrant_val * 100:.2f}%"
            delta = (qdrant_val - faiss_val) * 100
            delta_str = f"{delta:+.2f}pp"
        elif unit in ["s", "ms"]:
            faiss_str = f"{faiss_val:.3f}{unit}"
            qdrant_str = f"{qdrant_val:.3f}{unit}"
            delta_pct = ((qdrant_val - faiss_val) / faiss_val * 100) if faiss_val > 0 else 0
            delta_str = f"{delta_pct:+.1f}%"
        else:
            faiss_str = str(faiss_val)
            qdrant_str = str(qdrant_val)
            delta = qdrant_val - faiss_val
            delta_str = f"{delta:+d}"

        print(f"{label:<25} {faiss_str:>15} {qdrant_str:>20} {delta_str:>15}")

    print("=" * 90)


def save_results(
    faiss_results: dict[str, Any],
    qdrant_results: dict[str, Any],
    entities: list[dict[str, Any]],
    gold_pairs: set[tuple[str, str]],
    embedding_dim: int,
) -> None:
    """Save evaluation results to JSON file.

    Args:
        faiss_results: Metrics from FAISS evaluation
        qdrant_results: Metrics from Qdrant evaluation
        entities: List of entities evaluated
        gold_pairs: Ground truth pairs
        embedding_dim: Embedding dimension
    """
    output = {
        "dataset": {
            "name": "funder_names_with_ids",
            "total_entities": len(entities),
            "ground_truth_pairs": len(gold_pairs),
        },
        "model": {
            "name": MODEL_NAME,
            "embedding_dim": embedding_dim,
            "k_neighbors": K_NEIGHBORS,
        },
        "experiments": {
            "faiss_dense_only": faiss_results,
            "qdrant_hybrid": qdrant_results,
        },
        "comparison": {
            "recall_improvement": qdrant_results["recall"] - faiss_results["recall"],
            "precision_change": qdrant_results["precision"] - faiss_results["precision"],
            "f1_improvement": qdrant_results["f1"] - faiss_results["f1"],
            "indexing_time_ratio": (
                qdrant_results["indexing_time_seconds"] / faiss_results["indexing_time_seconds"]
                if faiss_results["indexing_time_seconds"] > 0
                else 0
            ),
        },
    }

    output_path = Path("examples/blocking_evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"✅ Results saved to: {output_path}")


def main() -> None:
    """Run blocking evaluation comparing FAISS vs Qdrant hybrid search."""
    print("=" * 90)
    print("BLOCKING EVALUATION: FAISS vs Qdrant Hybrid Search")
    print("=" * 90)
    print(f"\nDataset: Funder organization names")
    print(f"Embedding model: {MODEL_NAME}")
    print(f"k-neighbors: {K_NEIGHBORS}")
    print()

    # Load data
    entities, gold_pairs = load_funder_data()

    # Setup shared Qwen3 embedder (used by both FAISS and Qdrant dense)
    logger.info(f"Loading {MODEL_NAME} embedding model...")
    qwen_embedder = SentenceTransformerEmbedder(
        model_name=MODEL_NAME,
        batch_size=32,
        normalize_embeddings=True,
    )
    embedding_dim = qwen_embedder.embedding_dim
    logger.info(f"Embedding model loaded (dim={embedding_dim})")

    # Connect to Qdrant cloud
    qdrant_client = connect_to_qdrant()

    # Setup blockers
    faiss_blocker = setup_faiss_blocker(qwen_embedder)
    qdrant_blocker = setup_qdrant_blocker(qwen_embedder, qdrant_client)

    # Run evaluations
    print("\n" + "-" * 90)
    print("RUNNING EVALUATIONS")
    print("-" * 90)

    faiss_results = evaluate_blocking_recall(faiss_blocker, entities, gold_pairs, "FAISS")

    qdrant_results = evaluate_blocking_recall(qdrant_blocker, entities, gold_pairs, "Qdrant")

    # Print comparison
    print_comparison_table(faiss_results, qdrant_results)

    # Save results
    save_results(faiss_results, qdrant_results, entities, gold_pairs, embedding_dim)

    # Print key insights
    print("\n" + "=" * 90)
    print("KEY INSIGHTS")
    print("=" * 90)

    recall_diff = (qdrant_results["recall"] - faiss_results["recall"]) * 100
    if recall_diff > 0:
        print(f"✅ Qdrant hybrid search improved recall by {recall_diff:.2f} percentage points")
    else:
        print(
            f"⚠️  Qdrant hybrid search decreased recall by {abs(recall_diff):.2f} percentage points"
        )

    f1_diff = (qdrant_results["f1"] - faiss_results["f1"]) * 100
    if f1_diff > 0:
        print(f"✅ Qdrant improved F1 score by {f1_diff:.2f} percentage points")
    else:
        print(f"⚠️  Qdrant decreased F1 score by {abs(f1_diff):.2f} percentage points")

    time_ratio = (
        qdrant_results["indexing_time_seconds"] / faiss_results["indexing_time_seconds"]
        if faiss_results["indexing_time_seconds"] > 0
        else 0
    )
    if time_ratio > 1.2:
        print(f"⏱️  Qdrant was {time_ratio:.1f}x slower than FAISS (cloud latency)")
    elif time_ratio < 0.8:
        print(f"⚡ Qdrant was {1 / time_ratio:.1f}x faster than FAISS")
    else:
        print(f"⏱️  Qdrant and FAISS had similar performance ({time_ratio:.2f}x)")

    print("=" * 90)
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
