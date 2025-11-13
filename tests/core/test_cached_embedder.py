"""Tests for DiskCachedEmbedder with two-tier caching.

Tests cover:
- Hot cache (in-memory LRU)
- Cold storage (SQLite disk cache)
- Cache misses (compute embeddings)
- Persistence across instances
- Future prompt support
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from langres.core.embeddings import DiskCachedEmbedder, FakeEmbedder

logger = logging.getLogger(__name__)


class TestDiskCachedEmbedderInitialization:
    """Tests for DiskCachedEmbedder initialization."""

    def test_disk_cached_embedder_initialization(self, tmp_path):
        """Test that DiskCachedEmbedder initializes correctly."""
        embedder = FakeEmbedder(embedding_dim=128)
        cache_dir = tmp_path / "cache"

        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=cache_dir,
            namespace="test",
            memory_cache_size=100,
        )

        # Cache directory should be created
        assert cache_dir.exists()
        assert cache_dir.is_dir()

        # SQLite database should exist
        db_path = cache_dir / "test.db"
        assert db_path.exists()

        # Should be able to query the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == "embeddings"

        # Cache info should show empty state
        info = cached.cache_info()
        assert info["hits_hot"] == 0
        assert info["hits_cold"] == 0
        assert info["misses"] == 0
        assert info["hot_size"] == 0
        assert info["cold_size"] == 0
        assert info["hit_rate"] == 0.0


class TestDiskCachedEmbedderBasicCaching:
    """Tests for basic caching behavior (cache miss → compute → store)."""

    def test_encode_computes_on_first_call(self, tmp_path):
        """Test that first call computes and caches embeddings."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        texts = ["text1", "text2", "text3"]
        result = cached.encode(texts)

        # Result should match embedder output
        expected = embedder.encode(texts)
        np.testing.assert_array_equal(result, expected)

        # Cache info should show 3 misses
        info = cached.cache_info()
        assert info["misses"] == 3
        assert info["hits_hot"] == 0
        assert info["hits_cold"] == 0
        assert info["hot_size"] == 3
        assert info["cold_size"] == 3

    def test_encode_uses_hot_cache_on_second_call(self, tmp_path):
        """Test that second call uses hot cache (no computation)."""
        # Create a mock embedder to track encode() calls
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        texts = ["text1", "text2"]

        # First call - compute
        result1 = cached.encode(texts)
        info1 = cached.cache_info()
        assert info1["misses"] == 2

        # Second call - should use hot cache
        result2 = cached.encode(texts)
        info2 = cached.cache_info()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

        # Cache stats should show hot hits
        assert info2["misses"] == 2  # No new misses
        assert info2["hits_hot"] == 2  # 2 hot hits
        assert info2["hits_cold"] == 0  # No cold hits

    def test_encode_uses_cold_storage_when_evicted_from_hot(self, tmp_path):
        """Test cold storage lookup when hot cache is full."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
            memory_cache_size=5,  # Small hot cache
        )

        # Encode 10 texts (fills hot cache, evicts first 5)
        first_batch = [f"text{i}" for i in range(10)]
        cached.encode(first_batch)

        info_after_first = cached.cache_info()
        assert info_after_first["misses"] == 10
        assert info_after_first["hot_size"] == 5  # Only last 5 remain in hot
        assert info_after_first["cold_size"] == 10  # All 10 in cold storage

        # Encode first 5 texts again (should hit cold storage)
        second_batch = [f"text{i}" for i in range(5)]
        result = cached.encode(second_batch)

        # Should get valid embeddings
        assert result.shape == (5, 128)

        info_after_second = cached.cache_info()
        assert info_after_second["misses"] == 10  # No new misses
        assert info_after_second["hits_cold"] == 5  # 5 cold hits
        assert info_after_second["hot_size"] == 5  # Still 5 (promoted texts)


class TestDiskCachedEmbedderLRUEviction:
    """Tests for LRU eviction behavior."""

    def test_hot_cache_lru_eviction(self, tmp_path):
        """Test that hot cache evicts least-recently-used entries."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
            memory_cache_size=3,
        )

        # Encode ["a", "b", "c"] → hot cache full
        cached.encode(["a", "b", "c"])
        info1 = cached.cache_info()
        assert info1["hot_size"] == 3

        # Encode ["d"] → should evict "a" (oldest)
        cached.encode(["d"])
        info2 = cached.cache_info()
        assert info2["hot_size"] == 3  # Still 3 (max size)
        assert info2["cold_size"] == 4  # All 4 in cold storage

        # Encode ["a"] again → should hit cold storage
        cached.encode(["a"])
        info3 = cached.cache_info()
        assert info3["hits_cold"] == 1  # "a" came from cold storage


class TestDiskCachedEmbedderPersistence:
    """Tests for persistence across instances."""

    def test_cache_persists_across_instances(self, tmp_path):
        """Test that cache is loaded from disk on new instance."""
        embedder = FakeEmbedder(embedding_dim=128)
        cache_dir = tmp_path / "cache"

        # Instance 1: encode texts
        cached1 = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=cache_dir,
            namespace="test",
        )
        texts = ["text1", "text2", "text3"]
        result1 = cached1.encode(texts)
        info1 = cached1.cache_info()
        assert info1["misses"] == 3
        assert info1["cold_size"] == 3

        # Instance 2: same cache_dir, encode same texts
        cached2 = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=cache_dir,
            namespace="test",
        )
        result2 = cached2.encode(texts)
        info2 = cached2.cache_info()

        # Results should be identical (loaded from cold storage)
        np.testing.assert_array_equal(result1, result2)

        # Should have 0 misses (all from cold storage)
        assert info2["misses"] == 0
        assert info2["hits_cold"] == 3


class TestDiskCachedEmbedderEmptyInput:
    """Tests for empty input handling."""

    def test_encode_empty_list(self, tmp_path):
        """Test encoding empty list returns correct shape."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        result = cached.encode([])

        # Should return empty array with correct shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 128)

        # No cache operations
        info = cached.cache_info()
        assert info["misses"] == 0
        assert info["hits_hot"] == 0
        assert info["hits_cold"] == 0


class TestDiskCachedEmbedderCacheInfo:
    """Tests for cache_info statistics."""

    def test_cache_info_returns_correct_stats(self, tmp_path):
        """Test cache_info returns accurate statistics."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
            memory_cache_size=3,
        )

        # Initial state
        info0 = cached.cache_info()
        assert info0["misses"] == 0
        assert info0["hits_hot"] == 0
        assert info0["hits_cold"] == 0
        assert info0["hot_size"] == 0
        assert info0["cold_size"] == 0
        assert info0["hit_rate"] == 0.0

        # Encode 3 texts (all misses)
        cached.encode(["a", "b", "c"])
        info1 = cached.cache_info()
        assert info1["misses"] == 3
        assert info1["hits_hot"] == 0
        assert info1["hits_cold"] == 0
        assert info1["hot_size"] == 3
        assert info1["cold_size"] == 3
        assert info1["hit_rate"] == 0.0  # 0 hits / 3 total

        # Encode same 3 texts (hot hits)
        cached.encode(["a", "b", "c"])
        info2 = cached.cache_info()
        assert info2["misses"] == 3
        assert info2["hits_hot"] == 3
        assert info2["hits_cold"] == 0
        assert info2["hit_rate"] == 0.5  # 3 hits / 6 total

        # Encode 3 more texts (evicts first 3 from hot)
        cached.encode(["d", "e", "f"])
        info3 = cached.cache_info()
        assert info3["misses"] == 6
        assert info3["hits_hot"] == 3
        assert info3["cold_size"] == 6

        # Encode first 3 again (cold hits)
        cached.encode(["a", "b", "c"])
        info4 = cached.cache_info()
        assert info4["misses"] == 6
        assert info4["hits_hot"] == 3
        assert info4["hits_cold"] == 3
        # hit_rate = (3 hot + 3 cold) / 12 total = 6/12 = 0.5
        assert info4["hit_rate"] == 0.5


class TestDiskCachedEmbedderCacheClear:
    """Tests for cache_clear functionality."""

    def test_cache_clear_empties_both_caches(self, tmp_path):
        """Test cache_clear removes all entries."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        # Populate caches
        cached.encode(["a", "b", "c"])
        info1 = cached.cache_info()
        assert info1["hot_size"] == 3
        assert info1["cold_size"] == 3

        # Clear caches
        cached.cache_clear()

        # Both caches should be empty
        info2 = cached.cache_info()
        assert info2["hot_size"] == 0
        assert info2["cold_size"] == 0
        assert info2["misses"] == 0
        assert info2["hits_hot"] == 0
        assert info2["hits_cold"] == 0
        assert info2["hit_rate"] == 0.0

        # Verify SQLite table is empty
        db_path = tmp_path / "cache" / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0


class TestDiskCachedEmbedderOrderIndependence:
    """Tests for order independence."""

    def test_encode_order_independent(self, tmp_path):
        """Test that encoding works regardless of text order."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        # Encode in one order
        texts1 = ["a", "b", "c"]
        result1 = cached.encode(texts1)

        # Encode in different order
        texts2 = ["c", "a", "b"]
        result2 = cached.encode(texts2)

        # Results should match the input order
        assert result1.shape == (3, 128)
        assert result2.shape == (3, 128)

        # Verify individual embeddings match
        # result1[0] should equal result2[1] (both are "a")
        np.testing.assert_array_equal(result1[0], result2[1])
        # result1[1] should equal result2[2] (both are "b")
        np.testing.assert_array_equal(result1[1], result2[2])
        # result1[2] should equal result2[0] (both are "c")
        np.testing.assert_array_equal(result1[2], result2[0])


class TestDiskCachedEmbedderBatchOptimization:
    """Tests for batch computation optimization."""

    def test_batch_computes_only_missing_embeddings(self, tmp_path):
        """Test that only cache misses are computed in batch."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        # Encode ["a", "b"]
        cached.encode(["a", "b"])
        info1 = cached.cache_info()
        assert info1["misses"] == 2

        # Encode ["b", "c", "d"] (b is cached, c and d are new)
        result = cached.encode(["b", "c", "d"])
        info2 = cached.cache_info()

        # Should have correct shape
        assert result.shape == (3, 128)

        # Cache stats: 1 hot hit (b), 2 new misses (c, d)
        assert info2["misses"] == 4  # 2 from first call + 2 new
        assert info2["hits_hot"] == 1  # b was in hot cache

        # Results should match expected embeddings
        expected = embedder.encode(["b", "c", "d"])
        np.testing.assert_array_equal(result, expected)


class TestDiskCachedEmbedderPromptSupport:
    """Tests for future prompt support (design validation)."""

    def test_hash_text_supports_optional_prompt(self, tmp_path):
        """Test that _hash_text can differentiate text vs text+prompt."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        # Hash same text without prompt
        hash1 = cached._hash_text("same text", prompt=None)

        # Hash same text with prompt
        hash2 = cached._hash_text("same text", prompt="Find duplicates")

        # Hashes should be different
        assert hash1 != hash2

        # Verify that prompt is stored in SQLite schema
        db_path = tmp_path / "cache" / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(embeddings)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "prompt" in columns


class TestDiskCachedEmbedderEmbeddingDim:
    """Tests for embedding_dim property."""

    def test_embedding_dim_returns_correct_dimension(self, tmp_path):
        """Test embedding_dim property returns correct value."""
        embedder = FakeEmbedder(embedding_dim=256)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        assert cached.embedding_dim == 256


class TestDiskCachedEmbedderSerialization:
    """Tests for embedding serialization/deserialization."""

    def test_serialize_deserialize_preserves_embeddings(self, tmp_path):
        """Test that BLOB serialization is lossless."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        # Create random embedding
        original = np.random.rand(128).astype(np.float32)

        # Serialize → Deserialize
        serialized = cached._serialize_embedding(original)
        deserialized = cached._deserialize_embedding(serialized)

        # Should be identical
        np.testing.assert_allclose(original, deserialized, rtol=1e-6)


class TestDiskCachedEmbedderMemoryManagement:
    """Tests for memory management with large datasets."""

    def test_handles_large_dataset_without_oom(self, tmp_path):
        """Test that large datasets don't exhaust memory."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
            memory_cache_size=100,  # Small hot cache
        )

        # Encode 10,000 texts
        large_batch = [f"text{i}" for i in range(10_000)]
        result = cached.encode(large_batch)

        # Should complete without errors
        assert result.shape == (10_000, 128)

        # Hot cache should never exceed max size
        info = cached.cache_info()
        assert info["hot_size"] <= 100

        # All 10,000 should be in cold storage
        assert info["cold_size"] == 10_000


class TestDiskCachedEmbedderNamespaceIsolation:
    """Tests for namespace isolation."""

    def test_different_namespaces_use_separate_databases(self, tmp_path):
        """Test that different namespaces have separate storage."""
        embedder = FakeEmbedder(embedding_dim=128)
        cache_dir = tmp_path / "cache"

        # Create two cached embedders with different namespaces
        cached1 = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=cache_dir,
            namespace="namespace1",
        )
        cached2 = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=cache_dir,
            namespace="namespace2",
        )

        # Encode texts in namespace1
        cached1.encode(["text1", "text2"])
        info1 = cached1.cache_info()
        assert info1["cold_size"] == 2

        # namespace2 should have empty cache
        info2 = cached2.cache_info()
        assert info2["cold_size"] == 0

        # Verify separate database files
        db1_path = cache_dir / "namespace1.db"
        db2_path = cache_dir / "namespace2.db"
        assert db1_path.exists()
        assert db2_path.exists()


class TestDiskCachedEmbedderDuplicateInputs:
    """Tests for handling duplicate inputs in same batch."""

    def test_handles_duplicate_texts_in_batch(self, tmp_path):
        """Test that duplicate texts in same batch are handled correctly."""
        embedder = FakeEmbedder(embedding_dim=128)
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        # Encode batch with duplicates
        texts = ["a", "b", "a", "c", "b"]
        result = cached.encode(texts)

        # Should return correct shape
        assert result.shape == (5, 128)

        # Duplicate texts should have identical embeddings
        np.testing.assert_array_equal(result[0], result[2])  # Both "a"
        np.testing.assert_array_equal(result[1], result[4])  # Both "b"

        # Cache should only store unique texts
        info = cached.cache_info()
        assert info["cold_size"] == 3  # Only a, b, c (unique)
