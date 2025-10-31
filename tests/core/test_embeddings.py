"""Tests for embedding providers."""

import logging

import numpy as np
import pytest

from langres.core.embeddings import (
    EmbeddingProvider,
    FakeEmbedder,
    SentenceTransformerEmbedder,
)

logger = logging.getLogger(__name__)


class TestSentenceTransformerEmbedder:
    """Tests for SentenceTransformerEmbedder implementation."""

    def test_encode_returns_correct_shape(self):
        """Test that encode returns embeddings with correct shape."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        texts = ["Hello world", "Test text", "Another example"]

        embeddings = embedder.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == embedder.embedding_dim

    def test_encode_single_text(self):
        """Test encoding a single text."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        texts = ["Single text"]

        embeddings = embedder.encode(texts)

        assert embeddings.shape == (1, embedder.embedding_dim)

    def test_embedding_dim_property(self):
        """Test that embedding_dim returns correct dimension."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        # all-MiniLM-L6-v2 has 384 dimensions
        assert embedder.embedding_dim == 384

    def test_lazy_model_loading(self):
        """Test that model is not loaded until first encode call."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        # Model should not be loaded yet
        assert embedder._model is None

        # After encode, model should be loaded
        embedder.encode(["test"])
        assert embedder._model is not None

    def test_model_loaded_only_once(self):
        """Test that model is loaded only once and reused."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        embedder.encode(["first call"])
        first_model = embedder._model

        embedder.encode(["second call"])
        second_model = embedder._model

        # Should be the same model instance
        assert first_model is second_model

    def test_encode_empty_list_returns_empty_array(self):
        """Test that encoding empty list returns empty array."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        embeddings = embedder.encode([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0


class TestFakeEmbedder:
    """Tests for FakeEmbedder test double."""

    def test_encode_returns_correct_shape(self):
        """Test that FakeEmbedder returns correct shape."""
        embedder = FakeEmbedder(embedding_dim=128)
        texts = ["text1", "text2", "text3"]

        embeddings = embedder.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 128)

    def test_embedding_dim_property(self):
        """Test that embedding_dim property returns configured dimension."""
        embedder = FakeEmbedder(embedding_dim=256)
        assert embedder.embedding_dim == 256

    def test_deterministic_embeddings_for_same_text(self):
        """Test that FakeEmbedder produces deterministic embeddings."""
        embedder = FakeEmbedder(embedding_dim=128)
        texts = ["same text", "same text"]

        embeddings = embedder.encode(texts)

        # Same text should produce identical embeddings
        np.testing.assert_array_equal(embeddings[0], embeddings[1])

    def test_different_texts_produce_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        embedder = FakeEmbedder(embedding_dim=128)
        texts = ["text1", "text2"]

        embeddings = embedder.encode(texts)

        # Different texts should produce different embeddings
        assert not np.array_equal(embeddings[0], embeddings[1])

    def test_embeddings_are_normalized(self):
        """Test that fake embeddings are L2 normalized (unit vectors)."""
        embedder = FakeEmbedder(embedding_dim=128)
        texts = ["test1", "test2", "test3"]

        embeddings = embedder.encode(texts)

        # Check that each embedding has L2 norm ≈ 1.0
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            assert np.isclose(norm, 1.0, atol=1e-6)

    def test_encode_empty_list(self):
        """Test encoding empty list."""
        embedder = FakeEmbedder(embedding_dim=128)

        embeddings = embedder.encode([])

        assert embeddings.shape == (0, 128)

    def test_protocol_compliance(self):
        """Test that FakeEmbedder implements EmbeddingProvider protocol."""
        embedder = FakeEmbedder(embedding_dim=128)

        # Should have required methods and properties
        assert hasattr(embedder, "encode")
        assert hasattr(embedder, "embedding_dim")
        assert callable(embedder.encode)


class TestEmbeddingProviderProtocol:
    """Tests for EmbeddingProvider protocol compliance."""

    @pytest.mark.parametrize(
        "embedder_class,kwargs",
        [
            (SentenceTransformerEmbedder, {"model_name": "all-MiniLM-L6-v2"}),
            (FakeEmbedder, {"embedding_dim": 128}),
        ],
    )
    def test_protocol_methods_exist(self, embedder_class, kwargs):
        """Test that implementations have required protocol methods."""
        embedder = embedder_class(**kwargs)

        assert hasattr(embedder, "encode")
        assert hasattr(embedder, "embedding_dim")
        assert callable(embedder.encode)

    @pytest.mark.parametrize(
        "embedder_class,kwargs",
        [
            (SentenceTransformerEmbedder, {"model_name": "all-MiniLM-L6-v2"}),
            (FakeEmbedder, {"embedding_dim": 128}),
        ],
    )
    def test_encode_signature(self, embedder_class, kwargs):
        """Test that encode method has correct signature."""
        embedder = embedder_class(**kwargs)
        texts = ["test"]

        result = embedder.encode(texts)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # Should be 2D array
        assert result.shape[0] == len(texts)
