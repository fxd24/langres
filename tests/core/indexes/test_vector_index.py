"""Tests for vector index implementations."""

import logging

import numpy as np
import pytest

from langres.core.embeddings import FakeEmbedder, SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex, FakeVectorIndex

logger = logging.getLogger(__name__)


class TestFAISSIndex:
    """Tests for FAISSIndex implementation with new API."""

    # ============ NEW API TESTS ============
    def test_create_index_from_texts(self):
        """Test creating index from texts (index owns embedder)."""
        embedder = FakeEmbedder(embedding_dim=128)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]

        # Should not raise
        index.create_index(texts)

    def test_search_single_text_query(self):
        """Test searching with a single text query."""
        embedder = FakeEmbedder(embedding_dim=128)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        corpus_texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon"]
        index.create_index(corpus_texts)

        # Search with single text
        distances, indices = index.search("Apple", k=2)

        # Should return 1D arrays (single query)
        assert distances.shape == (2,)
        assert indices.shape == (2,)
        assert isinstance(distances, np.ndarray)
        assert isinstance(indices, np.ndarray)

    def test_search_batch_text_queries(self):
        """Test searching with batch of text queries (native batching)."""
        embedder = FakeEmbedder(embedding_dim=128)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        corpus_texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(corpus_texts)

        # Search with batch of texts (native batching!)
        query_texts = ["Apple", "Google"]
        distances, indices = index.search(query_texts, k=2)

        # Should return 2D arrays (batch)
        assert distances.shape == (2, 2)  # 2 queries, 2 neighbors each
        assert indices.shape == (2, 2)

    def test_search_all_deduplication_pattern(self):
        """Test search_all for efficient deduplication."""
        embedder = FakeEmbedder(embedding_dim=128)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon"]
        index.create_index(texts)

        # Search all items against all (dedup pattern)
        distances, indices = index.search_all(k=3)

        # Should return shape (N, k) where N = corpus size
        assert distances.shape == (4, 3)
        assert indices.shape == (4, 3)

        # First neighbor should be itself
        assert np.array_equal(indices[:, 0], [0, 1, 2, 3])

    def test_search_before_create_index_raises_error(self):
        """Test that searching before create_index raises error."""
        embedder = FakeEmbedder(embedding_dim=128)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        with pytest.raises(RuntimeError, match="Must call create_index"):
            index.search("Apple", k=3)

    def test_search_all_before_create_index_raises_error(self):
        """Test that search_all before create_index raises error."""
        embedder = FakeEmbedder(embedding_dim=128)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        with pytest.raises(RuntimeError, match="Must call create_index"):
            index.search_all(k=3)

    @pytest.mark.slow
    def test_create_index_with_real_embedder(self):
        """Test create_index with real sentence transformer."""
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corporation", "Google LLC"]
        index.create_index(texts)

        # Search for similar company
        distances, indices = index.search("Apple Company", k=2)

        assert distances.shape == (2,)
        # "Apple Inc." should be most similar
        assert indices[0] == 0


class TestFakeVectorIndex:
    """Tests for FakeVectorIndex test double with new API."""

    # ============ NEW API TESTS ============
    def test_fake_create_index_from_texts(self):
        """Test FakeVectorIndex.create_index with texts."""
        index = FakeVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]

        # Should not raise
        index.create_index(texts)

    def test_fake_search_single_text(self):
        """Test FakeVectorIndex.search with single text."""
        index = FakeVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        distances, indices = index.search("Apple", k=2)

        # Should return 1D arrays
        assert distances.shape == (2,)
        assert indices.shape == (2,)

    def test_fake_search_batch_texts(self):
        """Test FakeVectorIndex.search with batch of texts."""
        index = FakeVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        query_texts = ["Apple", "Google"]
        distances, indices = index.search(query_texts, k=2)

        # Should return 2D arrays
        assert distances.shape == (2, 2)
        assert indices.shape == (2, 2)

    def test_fake_search_all(self):
        """Test FakeVectorIndex.search_all."""
        index = FakeVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon"]
        index.create_index(texts)

        distances, indices = index.search_all(k=3)

        # Should return shape (N, k)
        assert distances.shape == (4, 3)
        assert indices.shape == (4, 3)

        # First neighbor should be itself (deterministic pattern)
        assert np.array_equal(indices[:, 0], [0, 1, 2, 3])

    def test_fake_vector_index_accepts_embeddings(self):
        """Test that FakeVectorIndex accepts np.ndarray for protocol compliance."""
        index = FakeVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Should accept single embedding
        embedding = np.random.rand(128).astype(np.float32)
        distances, indices = index.search(embedding, k=2)
        assert distances.shape == (2,)
        assert indices.shape == (2,)

        # Should accept batch embeddings
        embeddings = np.random.rand(2, 128).astype(np.float32)
        distances, indices = index.search(embeddings, k=2)
        assert distances.shape == (2, 2)
        assert indices.shape == (2, 2)


class TestFAISSIndexInstructionPrompts:
    """Tests for FAISSIndex instruction prompt support (asymmetric encoding)."""

    def test_faiss_index_documents_encoded_without_prompt(self):
        """Test that create_index encodes documents without prompt."""
        from unittest.mock import Mock

        # Create mock embedder to track calls
        embedder = Mock(spec=FakeEmbedder)
        embedder.encode.return_value = np.random.rand(3, 128).astype(np.float32)

        # Note: query_prompt removed from constructor - it's now a search-time parameter
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Verify encode was called with prompt=None (documents shouldn't have prompts)
        embedder.encode.assert_called_once()
        call_args = embedder.encode.call_args
        assert call_args[0][0] == texts  # First positional arg is texts
        # Check that prompt was either not passed or explicitly None
        if "prompt" in call_args[1]:
            assert call_args[1]["prompt"] is None

    def test_faiss_index_queries_encoded_with_prompt(self):
        """Test that search encodes queries with the query-time prompt parameter."""
        from unittest.mock import Mock

        # Create mock embedder to track calls
        embedder = Mock(spec=FakeEmbedder)

        # First call (create_index): return corpus embeddings
        # Second call (search): return query embeddings
        embedder.encode.side_effect = [
            np.random.rand(3, 128).astype(np.float32),  # corpus
            np.random.rand(1, 128).astype(np.float32),  # query
        ]

        query_prompt = "Find duplicate organization names"
        # Note: query_prompt removed from constructor
        index = FAISSIndex(embedder=embedder, metric="cosine")

        # Create index (first encode call)
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Search (second encode call - pass prompt at query time)
        index.search("Apple Company", k=2, query_prompt=query_prompt)

        # Verify second call used the prompt
        assert embedder.encode.call_count == 2
        second_call_args = embedder.encode.call_args_list[1]
        assert second_call_args[1]["prompt"] == query_prompt

    def test_faiss_index_search_all_uses_symmetric_encoding(self):
        """Test that search_all uses symmetric encoding (no re-encoding).

        For deduplication, both query and document sides come from the same corpus,
        so we use symmetric encoding (no prompt) for efficiency and correctness.
        query_prompt can optionally be passed to search_all, but by default it's None.
        """
        # Use a wrapper around FakeEmbedder to track calls
        encode_calls = []
        base_embedder = FakeEmbedder(embedding_dim=128)

        class TrackingEmbedder:
            def encode(self, texts, prompt=None):
                encode_calls.append({"texts": texts, "prompt": prompt})
                # Delegate to FakeEmbedder for proper embeddings
                return base_embedder.encode(texts, prompt=prompt)

            @property
            def embedding_dim(self):
                return base_embedder.embedding_dim

        embedder = TrackingEmbedder()
        # Note: query_prompt removed from constructor
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon"]
        index.create_index(texts)

        # search_all should NOT re-encode - it reuses cached embeddings
        # No query_prompt passed (default=None for symmetric encoding)
        index.search_all(k=3)

        # Verify only one encode call (create_index)
        assert len(encode_calls) == 1

        # Single call (create_index): no prompt (symmetric encoding)
        assert encode_calls[0]["texts"] == texts
        assert encode_calls[0]["prompt"] is None

    def test_faiss_index_no_query_prompt_backward_compatible(self):
        """Test that FAISSIndex without query_prompt uses prompt=None everywhere."""
        from unittest.mock import Mock

        embedder = Mock(spec=FakeEmbedder)
        embedder.encode.side_effect = [
            np.random.rand(3, 128).astype(np.float32),  # corpus
            np.random.rand(1, 128).astype(np.float32),  # query
        ]

        # No query_prompt specified (backward compatible)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)
        index.search("Apple", k=2)

        # Both calls should use prompt=None (or not pass prompt)
        assert embedder.encode.call_count == 2
        for call_args in embedder.encode.call_args_list:
            if "prompt" in call_args[1]:
                assert call_args[1]["prompt"] is None

    @pytest.mark.slow
    def test_faiss_index_different_prompts_affect_search(self):
        """Test that different query prompts produce different search results."""
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

        # Create single index (same for both searches)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        # Same corpus
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Same query, different prompts passed at query time
        query = "Apple Company"

        distances_with, indices_with = index.search(
            query, k=2, query_prompt="Find duplicate organization names accounting for acronyms"
        )
        distances_without, indices_without = index.search(query, k=2, query_prompt=None)

        # Different prompts should produce different distances
        # (indices might be same if ranking is preserved, but distances should differ)
        assert not np.allclose(distances_with, distances_without)


class TestFAISSIndexPrecomputedEmbeddings:
    """Tests for FAISSIndex pre-computed embedding support (performance fix)."""

    def test_faiss_index_search_with_precomputed_embeddings(self):
        """Test that search() accepts np.ndarray input without calling embedder.encode()."""
        # Use tracking embedder to verify NO encode() call when using pre-computed embeddings
        encode_calls = []

        class TrackingEmbedder:
            def encode(self, texts, prompt=None):
                encode_calls.append({"texts": texts, "prompt": prompt})
                # Return fake embeddings
                if isinstance(texts, list):
                    return np.random.rand(len(texts), 128).astype(np.float32)
                return np.random.rand(128).astype(np.float32)

            @property
            def embedding_dim(self):
                return 128

        embedder = TrackingEmbedder()
        index = FAISSIndex(embedder=embedder, metric="cosine")

        # Create index (first encode call)
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Clear tracking
        encode_calls.clear()

        # Search with pre-computed embeddings (should NOT call encode)
        query_embedding = np.random.rand(128).astype(np.float32)
        distances, indices = index.search(query_embedding, k=2)

        # Verify NO encode() call
        assert len(encode_calls) == 0, (
            "search() should not call encode() with pre-computed embeddings"
        )

        # Verify results are valid
        assert distances.shape == (2,)
        assert indices.shape == (2,)

        # Test batch embeddings
        batch_embeddings = np.random.rand(2, 128).astype(np.float32)
        distances, indices = index.search(batch_embeddings, k=2)

        # Still no encode calls
        assert len(encode_calls) == 0
        assert distances.shape == (2, 2)
        assert indices.shape == (2, 2)

    def test_faiss_index_precomputed_embeddings_same_results_as_text(self):
        """Test that pre-computed embeddings produce same results as text queries."""
        embedder = FakeEmbedder(embedding_dim=128)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Encode query manually
        query_text = "Apple"
        query_embedding_single = embedder.encode([query_text])[0]  # Extract single embedding

        # Compare search(text) vs search(embedding) - both single queries
        distances_text, indices_text = index.search(query_text, k=2)
        distances_embed, indices_embed = index.search(query_embedding_single, k=2)

        # Results should be identical
        np.testing.assert_array_equal(indices_text, indices_embed)
        np.testing.assert_allclose(distances_text, distances_embed)

        # Also test batch embeddings
        query_batch = embedder.encode([query_text, "Google"])
        distances_batch, indices_batch = index.search(query_batch, k=2)

        # Batch should have shape (2, 2)
        assert distances_batch.shape == (2, 2)
        assert indices_batch.shape == (2, 2)

    def test_faiss_index_search_all_no_reencoding(self):
        """Test that search_all() does NOT re-encode corpus (uses cached embeddings)."""
        encode_calls = []

        class TrackingEmbedder:
            def encode(self, texts, prompt=None):
                encode_calls.append({"texts": texts, "prompt": prompt})
                # Return fake embeddings
                return np.random.rand(len(texts), 128).astype(np.float32)

            @property
            def embedding_dim(self):
                return 128

        embedder = TrackingEmbedder()
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon"]
        index.create_index(texts)

        # Verify single encode call (create_index)
        assert len(encode_calls) == 1

        # Call search_all - should NOT re-encode
        distances, indices = index.search_all(k=3)

        # Still only one encode call (no re-encoding)
        assert len(encode_calls) == 1, "search_all() should NOT re-encode corpus"

        # Verify results are valid
        assert distances.shape == (4, 3)
        assert indices.shape == (4, 3)

    def test_faiss_index_search_all_with_prompt_no_reencoding(self):
        """Test that search_all() doesn't re-encode even when query_prompt is passed.

        For deduplication, we typically use symmetric encoding (no prompt) because
        both sides are from the same corpus. If a prompt is passed to search_all,
        it would be applied at query time, but by default it's None.
        """
        encode_calls = []

        class TrackingEmbedder:
            def encode(self, texts, prompt=None):
                encode_calls.append({"texts": texts, "prompt": prompt})
                return np.random.rand(len(texts), 128).astype(np.float32)

            @property
            def embedding_dim(self):
                return 128

        embedder = TrackingEmbedder()
        # Note: query_prompt removed from constructor
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon"]
        index.create_index(texts)

        # Verify single encode call (create_index, no prompt)
        assert len(encode_calls) == 1
        assert encode_calls[0]["prompt"] is None

        # Call search_all - should NOT re-encode (should reuse cached embeddings)
        # No query_prompt passed - uses default None
        distances, indices = index.search_all(k=3)

        # Still only one encode call - deduplication uses symmetric encoding
        assert len(encode_calls) == 1, (
            "search_all() should NOT re-encode corpus when using cached embeddings"
        )

    def test_faiss_index_search_text_uses_query_prompt(self):
        """Test that search() with text applies query_prompt when passed at query time."""
        encode_calls = []

        class TrackingEmbedder:
            def encode(self, texts, prompt=None):
                encode_calls.append({"texts": texts, "prompt": prompt})
                return np.random.rand(len(texts), 128).astype(np.float32)

            @property
            def embedding_dim(self):
                return 128

        embedder = TrackingEmbedder()
        query_prompt = "Find duplicate organization names"
        # Note: query_prompt removed from constructor
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        encode_calls.clear()

        # Search with text - pass query_prompt at query time
        index.search("Apple Company", k=2, query_prompt=query_prompt)

        assert len(encode_calls) == 1
        assert encode_calls[0]["prompt"] == query_prompt

    def test_faiss_index_different_prompts_same_index(self):
        """Test using different prompts per query without rebuilding index.

        This demonstrates the key benefit of the refactor: query_prompt as a
        query-time parameter allows trying different prompts on the same index.
        """
        embedder = FakeEmbedder(embedding_dim=128)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        corpus = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(corpus)

        # Same query, different prompts - all without rebuilding index
        query = "Apple"
        results_no_prompt = index.search(query, k=2, query_prompt=None)
        results_with_prompt_a = index.search(query, k=2, query_prompt="Find duplicates")
        results_with_prompt_b = index.search(query, k=2, query_prompt="Match companies")

        # All should work without rebuilding index
        assert results_no_prompt[0].shape == (2,)
        assert results_with_prompt_a[0].shape == (2,)
        assert results_with_prompt_b[0].shape == (2,)


class TestVectorIndexProtocol:
    """Tests for VectorIndex protocol compliance with new API."""

    def test_faiss_index_implements_protocol(self):
        """Test that FAISSIndex implements VectorIndex protocol."""
        embedder = FakeEmbedder(embedding_dim=128)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        assert hasattr(index, "create_index")
        assert hasattr(index, "search")
        assert hasattr(index, "search_all")
        assert callable(index.create_index)
        assert callable(index.search)
        assert callable(index.search_all)

    def test_fake_index_implements_protocol(self):
        """Test that FakeVectorIndex implements VectorIndex protocol."""
        index = FakeVectorIndex()

        assert hasattr(index, "create_index")
        assert hasattr(index, "search")
        assert hasattr(index, "search_all")
        assert callable(index.create_index)
        assert callable(index.search)
        assert callable(index.search_all)

    def test_vector_index_protocol_accepts_union_types(self):
        """Test that VectorIndex protocol accepts str | list[str] | np.ndarray."""
        embedder = FakeEmbedder(embedding_dim=128)
        index = FAISSIndex(embedder=embedder, metric="cosine")

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Protocol should accept str
        distances, indices = index.search("Apple", k=2)
        assert distances.shape == (2,)

        # Protocol should accept list[str]
        distances, indices = index.search(["Apple", "Google"], k=2)
        assert distances.shape == (2, 2)

        # Protocol should accept np.ndarray (single embedding)
        embedding = np.random.rand(128).astype(np.float32)
        distances, indices = index.search(embedding, k=2)
        assert distances.shape == (2,)

        # Protocol should accept np.ndarray (batch embeddings)
        embeddings = np.random.rand(2, 128).astype(np.float32)
        distances, indices = index.search(embeddings, k=2)
        assert distances.shape == (2, 2)


class TestFAISSIndexWithCachedEmbedder:
    """Integration tests for FAISSIndex with DiskCachedEmbedder."""

    def test_faiss_index_with_cached_embedder_and_query_prompts(self, tmp_path):
        """Test FAISSIndex with DiskCachedEmbedder using query prompts."""
        from pathlib import Path

        from langres.core.embeddings import DiskCachedEmbedder

        # Create cached embedder
        base_embedder = FakeEmbedder(embedding_dim=128)
        cached_embedder = DiskCachedEmbedder(
            embedder=base_embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        # Create FAISSIndex with cached embedder
        index = FAISSIndex(embedder=cached_embedder, metric="cosine")

        # Build index with corpus
        corpus = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(corpus)

        # Verify corpus was cached
        info1 = cached_embedder.cache_info()
        assert info1["misses"] == 3  # Corpus texts computed
        assert info1["cold_size"] == 3

        # Query "Apple" with prompt "Find duplicate organizations" → cache miss
        index.search("Apple", k=1, query_prompt="Find duplicate organizations")
        info2 = cached_embedder.cache_info()
        assert info2["misses"] == 4  # +1 for query with prompt

        # Query "Apple" with prompt "Match company names" → another cache miss
        index.search("Apple", k=1, query_prompt="Match company names")
        info3 = cached_embedder.cache_info()
        assert info3["misses"] == 5  # +1 for query with different prompt

        # Query "Apple" with prompt "Find duplicate organizations" again → cache hit
        index.search("Apple", k=1, query_prompt="Find duplicate organizations")
        info4 = cached_embedder.cache_info()
        assert info4["misses"] == 5  # No new miss
        assert info4["hits_hot"] >= 1 or info4["hits_cold"] >= 1  # Should hit cache

        # Query "Apple" with prompt=None → another cache miss (different from prompts)
        index.search("Apple", k=1, query_prompt=None)
        info5 = cached_embedder.cache_info()
        assert info5["misses"] == 6  # +1 for query without prompt

    def test_faiss_index_with_cached_embedder_and_precomputed(self, tmp_path):
        """Test FAISSIndex with DiskCachedEmbedder using pre-computed query embeddings."""
        from pathlib import Path

        from langres.core.embeddings import DiskCachedEmbedder

        # Create cached embedder
        base_embedder = FakeEmbedder(embedding_dim=128)
        cached_embedder = DiskCachedEmbedder(
            embedder=base_embedder,
            cache_dir=tmp_path / "cache",
            namespace="test",
        )

        # Create FAISSIndex with cached embedder
        index = FAISSIndex(embedder=cached_embedder, metric="cosine")

        # Build index with corpus
        corpus = ["Apple Inc.", "Microsoft Corp."]
        index.create_index(corpus)

        # Verify corpus was cached
        info1 = cached_embedder.cache_info()
        assert info1["misses"] == 2  # Corpus texts computed
        assert info1["cold_size"] == 2

        # Create pre-computed query embeddings using the same embedder
        # This ensures they have the correct shape and normalization
        query_texts = ["Apple", "Microsoft"]
        precomputed = base_embedder.encode(query_texts)

        # Search with pre-computed embeddings
        distances, indices = index.search(precomputed, k=1)

        # Verify search worked
        assert distances.shape == (2, 1)
        assert indices.shape == (2, 1)

        # Cache stats should show only corpus items (no query caching for pre-computed)
        info2 = cached_embedder.cache_info()
        assert info2["misses"] == 2  # Still just the corpus
        assert info2["hits_hot"] == 0  # No cache operations for pre-computed
        assert info2["hits_cold"] == 0
        assert info2["cold_size"] == 2  # Still just corpus
