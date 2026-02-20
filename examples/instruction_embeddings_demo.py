"""
Instruction Embeddings Demo: Query Prompts with FAISS Search Functions

This example demonstrates how to use instruction/query prompts for asymmetric
encoding with embedding models, specifically showing:

1. **Asymmetric Encoding Pattern**:
   - Documents (corpus) encoded with prompt=None (generic, broad understanding)
   - Queries encoded with task-specific instructions (focused matching)

2. **Search Functions**:
   - search(single_query) - Find matches for one query
   - search(batch_queries) - Find matches for multiple queries
   - Note: search_all() skipped due to Python 3.13 joblib/loky compatibility issue

3. **Instruction Impact**:
   - Compare results with vs without query instructions
   - Show how instructions improve match quality for organization names

Dataset: 1,741 real-world funder organization names

Usage:
    python examples/instruction_embeddings_demo.py
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from langres.core.embeddings import DiskCachedEmbedder, SentenceTransformerEmbedder
from langres.core.vector_index import FAISSIndex
from langres.data import load_labeled_dedup_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CACHE_DIR = Path("tmp/embedder_cache")
INSTRUCTION = (
    "Instruct: Find duplicate organization names accounting for "
    "acronyms, abbreviations, and spelling variations\nQuery: "
)


def load_funder_names() -> tuple[list[str], list[str]]:
    """Load funder organization names from dataset.

    Returns:
        Tuple of (entity_ids, names)
    """
    logger.info("Loading funder names dataset...")

    dataset = load_labeled_dedup_data(
        data_dir="examples/data",
        entity_names_file="funder_names_with_ids.json",
        labeled_groups_file="funder_name_deduplicated_groups.json",
    )

    # Extract IDs and names in same order
    entity_ids = list(dataset.entity_names.keys())
    names = [dataset.entity_names[id_] for id_ in entity_ids]

    logger.info(f"Loaded {len(names)} organization names")
    return entity_ids, names


def print_search_results(
    query: str,
    entity_ids: list[str],
    names: list[str],
    indices: np.ndarray,
    distances: np.ndarray,
    top_k: int = 5,
) -> None:
    """Print search results in a readable format.

    Args:
        query: The query string
        entity_ids: List of all entity IDs
        names: List of all organization names
        indices: Indices of matches (shape: (k,) or (N, k))
        distances: Distance/similarity scores (shape: (k,) or (N, k))
        top_k: Number of results to show
    """
    print(f"\n  Query: '{query}'")
    print("  " + "-" * 70)

    # Handle both single query and batch query results
    if indices.ndim == 1:
        # Single query result: shape (k,)
        result_indices = indices[:top_k]
        result_distances = distances[:top_k]
    else:
        # Batch query result: shape (N, k) - take first query
        result_indices = indices[0, :top_k]
        result_distances = distances[0, :top_k]

    for rank, (idx, score) in enumerate(zip(result_indices, result_distances), 1):
        match_id = entity_ids[idx]
        match_name = names[idx]
        print(f"  {rank}. [{match_id}] {match_name:<50} (score: {score:.4f})")


def demo_search_all(
    index: FAISSIndex,
    entity_ids: list[str],
    names: list[str],
) -> None:
    """Demonstrate search_all() - deduplication pattern.

    Finds k nearest neighbors for ALL organizations at once.

    Args:
        index: Initialized FAISS index
        entity_ids: List of entity IDs
        names: List of organization names
    """
    print("\n" + "=" * 80)
    print("DEMO 1: search_all() - Deduplication Pattern")
    print("=" * 80)
    print("\nFind 10 nearest neighbors for ALL organizations (efficient batch operation)")

    # Search all corpus items against each other
    distances, indices = index.search_all(k=10)

    print(f"\nResults shape: distances={distances.shape}, indices={indices.shape}")
    print("Showing results for first 3 organizations:\n")

    # Show results for first 3 organizations
    for i in range(3):
        org_id = entity_ids[i]
        org_name = names[i]

        print(f"Organization: [{org_id}] {org_name}")
        print("  Top 5 matches (excluding self):")

        # Skip first result (self) and show next 5
        for rank, (idx, score) in enumerate(zip(indices[i][1:6], distances[i][1:6]), 1):
            match_id = entity_ids[idx]
            match_name = names[idx]
            print(f"    {rank}. [{match_id}] {match_name:<45} (score: {score:.4f})")
        print()


def demo_search_single(
    index: FAISSIndex,
    entity_ids: list[str],
    names: list[str],
) -> None:
    """Demonstrate search() with single query.

    Args:
        index: Initialized FAISS index
        entity_ids: List of entity IDs
        names: List of organization names
    """
    print("\n" + "=" * 80)
    print("DEMO 2: search() - Single Query")
    print("=" * 80)
    print("\nSearch for variations of 'Gates Foundation'")

    query = "Gates Foundation"
    distances, indices = index.search(query, k=5)

    print(f"\nResults shape: distances={distances.shape}, indices={indices.shape}")
    print_search_results(query, entity_ids, names, indices, distances, top_k=5)


def demo_search_batch(
    index: FAISSIndex,
    entity_ids: list[str],
    names: list[str],
) -> None:
    """Demonstrate search() with batch queries.

    Args:
        index: Initialized FAISS index
        entity_ids: List of entity IDs
        names: List of organization names
    """
    print("\n" + "=" * 80)
    print("DEMO 3: search() - Batch Queries")
    print("=" * 80)
    print("\nSearch for multiple organizations at once (efficient batching)")

    queries = [
        "National Science Foundation",
        "NIH",
        "European Union",
    ]

    distances, indices = index.search(queries, k=3)

    print(f"\nResults shape: distances={distances.shape}, indices={indices.shape}")
    print("Showing top 3 matches for each query:\n")

    for i, query in enumerate(queries):
        print(f"  Query {i + 1}: '{query}'")
        print("  " + "-" * 70)

        for rank, (idx, score) in enumerate(zip(indices[i], distances[i]), 1):
            match_id = entity_ids[idx]
            match_name = names[idx]
            print(f"  {rank}. [{match_id}] {match_name:<50} (score: {score:.4f})")
        print()


def demo_comparison_with_without_instructions(
    embedder: DiskCachedEmbedder,
    entity_ids: list[str],
    names: list[str],
) -> None:
    """Compare search results with vs without query instructions.

    Args:
        embedder: Cached embedder
        entity_ids: List of entity IDs
        names: List of organization names
    """
    print("\n" + "=" * 80)
    print("DEMO 4: Impact of Query Instructions")
    print("=" * 80)
    print("\nCompare: Generic embeddings vs Task-specific instructions")

    # Test query with abbreviation/acronym challenge
    query = "WHO"  # World Health Organization

    # Create single index (used for both searches with different prompts)
    logger.info("Creating index...")
    index = FAISSIndex(
        embedder=embedder,
        metric="cosine",
    )
    index.create_index(names)

    # Search with different prompts (key benefit: no need to rebuild index!)
    distances_no_instr, indices_no_instr = index.search(query, k=5, query_prompt=None)
    distances_with_instr, indices_with_instr = index.search(query, k=5, query_prompt=INSTRUCTION)

    # Print results
    print("\n📊 WITHOUT Instructions (Generic Embeddings):")
    print_search_results(query, entity_ids, names, indices_no_instr, distances_no_instr, top_k=5)

    print("\n📊 WITH Instructions (Task-Specific Embeddings):")
    print_search_results(
        query, entity_ids, names, indices_with_instr, distances_with_instr, top_k=5
    )

    print("\n💡 Note: Instructions help the model understand we're looking for")
    print("   organizational duplicates (acronyms, abbreviations, variations)")


def main() -> None:
    """Run instruction embeddings demonstration."""
    print("\n" + "=" * 80)
    print("INSTRUCTION EMBEDDINGS DEMO")
    print("=" * 80)

    # Setup cached embedder
    logger.info("Setting up DiskCachedEmbedder with Qwen3 0.6B...")
    base_embedder = SentenceTransformerEmbedder(
        model_name=DENSE_MODEL,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=False,  # Avoid joblib issues
    )

    cached_embedder = DiskCachedEmbedder(
        embedder=base_embedder,
        cache_dir=CACHE_DIR,
        namespace="instruction_demo",
        memory_cache_size=100_000,
    )

    # Load data
    entity_ids, names = load_funder_names()

    # Create FAISS index (query_prompt moved to search-time)
    logger.info("Creating FAISS index...")
    index = FAISSIndex(
        embedder=cached_embedder,
        metric="cosine",
    )

    # Build index (documents encoded WITHOUT prompt)
    logger.info("Building index (encoding documents generically)...")
    index.create_index(names)
    logger.info("Index built successfully!")

    cache_info = cached_embedder.cache_info()
    logger.info(f"Cache stats: {cache_info['hits_hot']} hot hits, {cache_info['misses']} misses")

    # Run demonstrations
    # NOTE: Skipping search_all() due to Python 3.13 + joblib/loky semaphore issues
    # search_all() triggers a crash when FAISS threading conflicts with lingering joblib workers
    # See: https://github.com/joblib/loky/issues/369
    demo_search_all(index, entity_ids, names)

    demo_search_single(index, entity_ids, names)
    demo_search_batch(index, entity_ids, names)
    demo_comparison_with_without_instructions(cached_embedder, entity_ids, names)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n✓ Demonstrated asymmetric encoding pattern:")
    print("  - Documents: prompt=None (generic, broad understanding)")
    print("  - Queries: task-specific instruction (focused matching)")
    print("\n✓ Showed search functions:")
    print("  - search(single): Find matches for one query")
    print("  - search(batch): Find matches for multiple queries")
    print("  - (search_all() skipped due to Python 3.13 joblib issue)")
    print("\n✓ Compared with/without instructions:")
    print("  - Instructions improve matching quality for abbreviations/acronyms")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
