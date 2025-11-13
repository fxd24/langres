"""Demonstration of DiskCachedEmbedder for persistent embedding caching.

This example shows how to use DiskCachedEmbedder to:
1. Avoid recomputing embeddings across runs
2. Handle large datasets without running out of memory
3. Get cache statistics to monitor performance

Key benefits:
- First run: Computes embeddings (slow)
- Subsequent runs: Loads from cache (fast, no model loading needed!)
- Memory bounded: Only keeps hot cache in RAM (default: 10K embeddings)
- Unlimited disk storage: Can cache millions of embeddings

Run this example multiple times to see caching in action!
"""

import logging
import time
from pathlib import Path

from langres.core.embeddings import DiskCachedEmbedder, SentenceTransformerEmbedder

# Configure logging to see cache operations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Demonstrate DiskCachedEmbedder with a sample dataset."""
    print("=" * 80)
    print("DISK CACHED EMBEDDER DEMONSTRATION")
    print("=" * 80)

    # Sample dataset (in practice, this could be millions of entities)
    company_names = [
        "Apple Inc.",
        "Microsoft Corporation",
        "Google LLC",
        "Amazon.com Inc.",
        "Meta Platforms Inc.",
        "Tesla Inc.",
        "NVIDIA Corporation",
        "Intel Corporation",
        "Advanced Micro Devices Inc.",
        "Qualcomm Inc.",
    ] * 100  # Repeat to get 1,000 entities

    print(f"\nDataset size: {len(company_names)} entities")

    # Setup cache directory
    cache_dir = Path("tmp/embedding_cache")
    print(f"Cache directory: {cache_dir}")

    # Create base embedder (expensive to load model!)
    print("\n[1] Creating base embedder...")
    base_embedder = SentenceTransformerEmbedder(
        model_name="all-MiniLM-L6-v2",
        batch_size=64,
        normalize_embeddings=True,
    )

    # Wrap with disk cache
    print("[2] Creating cached embedder...")
    cached_embedder = DiskCachedEmbedder(
        embedder=base_embedder,
        cache_dir=cache_dir,
        namespace="company_names_demo",
        memory_cache_size=100,  # Small for demo (keeps only 100 in RAM)
    )

    # First encoding (will compute and cache)
    print("\n[3] First encoding (computes embeddings)...")
    start = time.time()
    embeddings_1 = cached_embedder.encode(company_names)
    elapsed_1 = time.time() - start

    print(f"    Shape: {embeddings_1.shape}")
    print(f"    Time: {elapsed_1:.2f}s")

    # Check cache stats
    info_1 = cached_embedder.cache_info()
    print(f"\n    Cache stats after first run:")
    print(f"      Hot hits:  {info_1['hits_hot']}")
    print(f"      Cold hits: {info_1['hits_cold']}")
    print(f"      Misses:    {info_1['misses']}")
    print(f"      Hot size:  {info_1['hot_size']} / {info_1['memory_cache_max']}")
    print(f"      Cold size: {info_1['cold_size']} (on disk)")
    print(f"      Hit rate:  {info_1['hit_rate']:.1%}")

    # Second encoding (will use cache - much faster!)
    print("\n[4] Second encoding (uses cache)...")
    start = time.time()
    embeddings_2 = cached_embedder.encode(company_names)
    elapsed_2 = time.time() - start

    print(f"    Shape: {embeddings_2.shape}")
    print(f"    Time: {elapsed_2:.2f}s ({elapsed_1 / elapsed_2:.1f}x faster!)")

    # Check cache stats again
    info_2 = cached_embedder.cache_info()
    print(f"\n    Cache stats after second run:")
    print(f"      Hot hits:  {info_2['hits_hot']}")
    print(f"      Cold hits: {info_2['hits_cold']}")
    print(f"      Misses:    {info_2['misses']}")
    print(f"      Hot size:  {info_2['hot_size']} / {info_2['memory_cache_max']}")
    print(f"      Cold size: {info_2['cold_size']} (on disk)")
    print(f"      Hit rate:  {info_2['hit_rate']:.1%}")

    # Demonstrate order independence
    print("\n[5] Encoding in different order (still uses cache)...")
    shuffled = company_names[::-1]  # Reverse order
    start = time.time()
    embeddings_3 = cached_embedder.encode(shuffled)
    elapsed_3 = time.time() - start

    print(f"    Time: {elapsed_3:.2f}s")
    print(f"    Results match original: {(embeddings_3 == embeddings_2[::-1]).all()}")

    # Show memory vs disk trade-off
    print("\n[6] Memory management:")
    print(f"    Hot cache limit: {info_2['memory_cache_max']} embeddings")
    print(f"    Hot cache size:  {info_2['hot_size']} embeddings (~40KB in RAM)")
    print(f"    Cold storage:    {info_2['cold_size']} embeddings (on disk)")
    print(f"    Memory saved:    ~{(info_2['cold_size'] - info_2['hot_size']) * 0.04:.1f}KB")

    # Demonstrate persistence
    print("\n[7] Demonstrating persistence across runs...")
    print("    Creating new cached embedder instance (simulating restart)...")

    new_cached = DiskCachedEmbedder(
        embedder=base_embedder,
        cache_dir=cache_dir,
        namespace="company_names_demo",  # Same namespace = same cache!
        memory_cache_size=100,
    )

    start = time.time()
    embeddings_4 = new_cached.encode(company_names[:10])  # Just first 10
    elapsed_4 = time.time() - start

    info_4 = new_cached.cache_info()
    print(f"    Time: {elapsed_4:.3f}s (loaded from disk, no model inference!)")
    print(f"    Cold hits: {info_4['hits_cold']} (loaded from SQLite)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ First run computed {info_1['misses']} embeddings in {elapsed_1:.2f}s")
    print(
        f"✓ Second run used cache (100% hits) in {elapsed_2:.2f}s ({elapsed_1 / elapsed_2:.1f}x faster)"
    )
    print(f"✓ Memory bounded: only {info_2['hot_size']} embeddings in RAM")
    print(f"✓ Disk storage: {info_2['cold_size']} embeddings cached")
    print(f"✓ Persistent: cache survives restarts (SQLite)")
    print(f"✓ Order independent: works with any text order")
    print("\nRun this script again - subsequent runs will be instant!")
    print("=" * 80)


if __name__ == "__main__":
    main()
