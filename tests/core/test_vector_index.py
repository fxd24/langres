"""Tests for vector index implementations."""

import logging

import numpy as np
import pytest

from langres.core.embeddings import FakeEmbedder, SentenceTransformerEmbedder
from langres.core.vector_index import FAISSIndex, FakeVectorIndex

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
