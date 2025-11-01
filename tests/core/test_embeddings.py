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

    @pytest.mark.slow
    def test_encode_returns_correct_shape(self):
        """Test that encode returns embeddings with correct shape."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        texts = ["Hello world", "Test text", "Another example"]

        embeddings = embedder.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == embedder.embedding_dim

    @pytest.mark.slow
    def test_encode_single_text(self):
        """Test encoding a single text."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        texts = ["Single text"]

        embeddings = embedder.encode(texts)

        assert embeddings.shape == (1, embedder.embedding_dim)

    @pytest.mark.slow
    def test_embedding_dim_property(self):
        """Test that embedding_dim returns correct dimension."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        # all-MiniLM-L6-v2 has 384 dimensions
        assert embedder.embedding_dim == 384

    @pytest.mark.slow
    def test_lazy_model_loading(self):
        """Test that model is not loaded until first encode call."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        # Model should not be loaded yet
        assert embedder._model is None

        # After encode, model should be loaded
        embedder.encode(["test"])
        assert embedder._model is not None

    @pytest.mark.slow
    def test_model_loaded_only_once(self):
        """Test that model is loaded only once and reused."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        embedder.encode(["first call"])
        first_model = embedder._model

        embedder.encode(["second call"])
        second_model = embedder._model

        # Should be the same model instance
        assert first_model is second_model

    @pytest.mark.slow
    def test_encode_empty_list_returns_empty_array(self):
        """Test that encoding empty list returns empty array."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        embeddings = embedder.encode([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0

    @pytest.mark.slow
    def test_batch_size_parameter(self):
        """Test that batch_size parameter works without errors."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2", batch_size=64)
        texts = [f"text {i}" for i in range(100)]

        embeddings = embedder.encode(texts)

        assert embeddings.shape == (100, 384)

    @pytest.mark.slow
    def test_normalize_embeddings_true(self):
        """Test that embeddings are L2-normalized when normalize_embeddings=True."""
        embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2", normalize_embeddings=True
        )
        texts = ["hello", "world"]

        embeddings = embedder.encode(texts)

        # Verify L2 norms are 1.0
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    @pytest.mark.slow
    def test_normalize_embeddings_false(self):
        """Test that normalize_embeddings=False parameter works without errors.

        Note: Some models like all-MiniLM-L6-v2 have built-in normalization
        in their architecture, so they return normalized embeddings regardless.
        This test verifies the parameter is accepted and the model runs.
        """
        embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2", normalize_embeddings=False
        )
        texts = ["hello", "world"]

        embeddings = embedder.encode(texts)

        # Verify embeddings are returned with correct shape
        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32

    @pytest.mark.slow
    def test_defensive_numpy_conversion(self, mocker):
        """Test defensive conversion when model returns non-ndarray."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        # Mock model.encode to return a list instead of ndarray
        mock_model = mocker.MagicMock()
        mock_model.encode.return_value = [[0.1] * 384, [0.2] * 384]
        mock_model.get_sentence_embedding_dimension.return_value = 384
        embedder._model = mock_model

        texts = ["text1", "text2"]
        embeddings = embedder.encode(texts)

        # Verify conversion happened and result is ndarray
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32


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
        """Test that fake embeddings are L2 normalized (unit vectors) by default."""
        embedder = FakeEmbedder(embedding_dim=128)
        texts = ["test1", "test2", "test3"]

        embeddings = embedder.encode(texts)

        # Check that each embedding has L2 norm ≈ 1.0
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            assert np.isclose(norm, 1.0, atol=1e-6)

    def test_normalize_embeddings_true(self):
        """Test that FakeEmbedder normalizes when normalize_embeddings=True."""
        embedder = FakeEmbedder(embedding_dim=128, normalize_embeddings=True)
        texts = ["test1", "test2"]

        embeddings = embedder.encode(texts)

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_normalize_embeddings_false(self):
        """Test that FakeEmbedder doesn't normalize when normalize_embeddings=False."""
        embedder = FakeEmbedder(embedding_dim=128, normalize_embeddings=False)
        texts = ["test1", "test2"]

        embeddings = embedder.encode(texts)

        norms = np.linalg.norm(embeddings, axis=1)
        assert not np.allclose(norms, 1.0)

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

    @pytest.mark.slow
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

    @pytest.mark.slow
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
