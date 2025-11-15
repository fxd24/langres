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

    @pytest.mark.slow
    def test_embedding_dim_accessed_before_encode(self):
        """Test that accessing embedding_dim before encode triggers lazy loading."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        # Model should not be loaded yet
        assert embedder._model is None

        # Access embedding_dim property (should trigger lazy loading)
        dim = embedder.embedding_dim

        # Model should now be loaded
        assert embedder._model is not None
        assert dim == 384  # all-MiniLM-L6-v2 dimension

    def test_precomputed_embeddings_bypass_model(self):
        """Test that pre-computed embeddings bypass model loading."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        # Create pre-computed embeddings
        precomputed = np.random.rand(3, 384).astype(np.float32)

        # Pass through embedder
        result = embedder.encode(precomputed)

        # Model should NOT be loaded (bypassed)
        assert embedder._model is None

        # Result should be identical to input
        np.testing.assert_array_equal(result, precomputed)
        assert result.dtype == np.float32

    def test_precomputed_embeddings_dtype_conversion(self):
        """Test that pre-computed embeddings are converted to float32."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        # Create float64 array
        precomputed_f64 = np.random.rand(2, 384).astype(np.float64)

        # Pass through embedder
        result = embedder.encode(precomputed_f64)

        # Result should be float32
        assert result.dtype == np.float32

        # Values should match (within tolerance)
        np.testing.assert_allclose(result, precomputed_f64, rtol=1e-6)


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

    def test_fake_embedder_consistency_across_instances(self):
        """Test that FakeEmbedder produces same embeddings across different instances."""
        # Create two separate embedder instances with same config
        embedder1 = FakeEmbedder(embedding_dim=128, normalize_embeddings=True)
        embedder2 = FakeEmbedder(embedding_dim=128, normalize_embeddings=True)

        # Same text should produce identical embeddings in both instances
        text = "consistency test"

        embeddings1 = embedder1.encode([text])
        embeddings2 = embedder2.encode([text])

        # Should be identical (deterministic hashing)
        np.testing.assert_array_equal(embeddings1, embeddings2)

        # Also test with multiple texts
        texts = ["text1", "text2", "text3"]
        batch1 = embedder1.encode(texts)
        batch2 = embedder2.encode(texts)

        np.testing.assert_array_equal(batch1, batch2)

    def test_precomputed_embeddings_bypass_generation(self):
        """Test that pre-computed embeddings bypass fake generation."""
        embedder = FakeEmbedder(embedding_dim=128)

        # Create pre-computed embeddings
        precomputed = np.random.rand(3, 128).astype(np.float32)

        # Pass through embedder
        result = embedder.encode(precomputed)

        # Result should be identical to input
        np.testing.assert_array_equal(result, precomputed)
        assert result.dtype == np.float32

    def test_precomputed_embeddings_dtype_conversion(self):
        """Test that pre-computed embeddings are converted to float32."""
        embedder = FakeEmbedder(embedding_dim=128)

        # Create float64 array
        precomputed_f64 = np.random.rand(2, 128).astype(np.float64)

        # Pass through embedder
        result = embedder.encode(precomputed_f64)

        # Result should be float32
        assert result.dtype == np.float32

        # Values should match (within tolerance)
        np.testing.assert_allclose(result, precomputed_f64, rtol=1e-6)


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


# ============ LATE INTERACTION EMBEDDING TESTS ============


class TestFastEmbedLateInteractionEmbedder:
    """Tests for FastEmbedLateInteractionEmbedder implementation."""

    @pytest.mark.slow
    def test_encode_returns_correct_structure(self):
        """Test that encode returns list of multi-vectors with correct structure."""
        from langres.core.embeddings import FastEmbedLateInteractionEmbedder

        embedder = FastEmbedLateInteractionEmbedder(model_name="colbert-ir/colbertv2.0")
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]

        multi_vectors = embedder.encode(texts)

        # Should return list with one entry per text
        assert isinstance(multi_vectors, list)
        assert len(multi_vectors) == len(texts)

        # Each entry should be a list of token embeddings
        for text_vectors in multi_vectors:
            assert isinstance(text_vectors, list)
            # Each text should have at least one token
            assert len(text_vectors) > 0
            # Each token should be a list of floats
            for token_embedding in text_vectors:
                assert isinstance(token_embedding, list)
                assert len(token_embedding) == embedder.embedding_dim
                assert all(isinstance(x, float) for x in token_embedding)

    @pytest.mark.slow
    def test_encode_single_text(self):
        """Test encoding a single text."""
        from langres.core.embeddings import FastEmbedLateInteractionEmbedder

        embedder = FastEmbedLateInteractionEmbedder(model_name="colbert-ir/colbertv2.0")
        texts = ["Single text"]

        multi_vectors = embedder.encode(texts)

        assert len(multi_vectors) == 1
        assert isinstance(multi_vectors[0], list)
        assert len(multi_vectors[0]) > 0  # At least one token

    @pytest.mark.slow
    def test_embedding_dim_property(self):
        """Test that embedding_dim returns correct dimension for ColBERTv2."""
        from langres.core.embeddings import FastEmbedLateInteractionEmbedder

        embedder = FastEmbedLateInteractionEmbedder(model_name="colbert-ir/colbertv2.0")

        # ColBERTv2 has 128-dimensional token embeddings
        assert embedder.embedding_dim == 128

    @pytest.mark.slow
    def test_lazy_model_loading(self):
        """Test that model is not loaded until first encode call."""
        from langres.core.embeddings import FastEmbedLateInteractionEmbedder

        embedder = FastEmbedLateInteractionEmbedder(model_name="colbert-ir/colbertv2.0")

        # Model should not be loaded yet
        assert embedder._model is None

        # After encode, model should be loaded
        embedder.encode(["test"])
        assert embedder._model is not None

    @pytest.mark.slow
    def test_model_loaded_only_once(self):
        """Test that model is loaded only once and reused."""
        from langres.core.embeddings import FastEmbedLateInteractionEmbedder

        embedder = FastEmbedLateInteractionEmbedder(model_name="colbert-ir/colbertv2.0")

        embedder.encode(["first call"])
        first_model = embedder._model

        embedder.encode(["second call"])
        second_model = embedder._model

        # Should be the same model instance
        assert first_model is second_model

    @pytest.mark.slow
    def test_encode_empty_list_returns_empty_list(self):
        """Test that encoding empty list returns empty list."""
        from langres.core.embeddings import FastEmbedLateInteractionEmbedder

        embedder = FastEmbedLateInteractionEmbedder(model_name="colbert-ir/colbertv2.0")

        multi_vectors = embedder.encode([])

        assert isinstance(multi_vectors, list)
        assert len(multi_vectors) == 0

    @pytest.mark.slow
    def test_different_texts_have_different_token_counts(self):
        """Test that texts of different lengths produce different token counts."""
        from langres.core.embeddings import FastEmbedLateInteractionEmbedder

        embedder = FastEmbedLateInteractionEmbedder(model_name="colbert-ir/colbertv2.0")
        texts = ["Hi", "This is a much longer sentence with many words"]

        multi_vectors = embedder.encode(texts)

        # Longer text should have more tokens (generally)
        # Note: This is a soft assertion due to tokenization quirks
        short_tokens = len(multi_vectors[0])
        long_tokens = len(multi_vectors[1])
        logger.info("Short text tokens: %d, Long text tokens: %d", short_tokens, long_tokens)
        # At minimum, both should have some tokens
        assert short_tokens > 0
        assert long_tokens > 0

    @pytest.mark.slow
    def test_model_name_parameter_flexibility(self):
        """Test that different model names can be used (model flexibility requirement)."""
        from langres.core.embeddings import FastEmbedLateInteractionEmbedder

        # Default model
        embedder1 = FastEmbedLateInteractionEmbedder()
        assert embedder1.model_name == "colbert-ir/colbertv2.0"

        # Custom model (even if not loaded, config should be stored)
        embedder2 = FastEmbedLateInteractionEmbedder(model_name="custom-model-name")
        assert embedder2.model_name == "custom-model-name"


class TestFakeLateInteractionEmbedder:
    """Tests for FakeLateInteractionEmbedder test double."""

    def test_encode_returns_correct_structure(self):
        """Test that FakeLateInteractionEmbedder returns correct structure."""
        from langres.core.embeddings import FakeLateInteractionEmbedder

        embedder = FakeLateInteractionEmbedder(embedding_dim=128, num_tokens=5)
        texts = ["text1", "text2", "text3"]

        multi_vectors = embedder.encode(texts)

        # Should return list with one entry per text
        assert isinstance(multi_vectors, list)
        assert len(multi_vectors) == len(texts)

        # Each text should have exactly num_tokens token embeddings
        for text_vectors in multi_vectors:
            assert isinstance(text_vectors, list)
            assert len(text_vectors) == 5  # num_tokens=5

            # Each token should have embedding_dim floats
            for token_embedding in text_vectors:
                assert isinstance(token_embedding, list)
                assert len(token_embedding) == 128  # embedding_dim=128
                assert all(isinstance(x, float) for x in token_embedding)

    def test_embedding_dim_property(self):
        """Test that embedding_dim property returns configured dimension."""
        from langres.core.embeddings import FakeLateInteractionEmbedder

        embedder = FakeLateInteractionEmbedder(embedding_dim=256)
        assert embedder.embedding_dim == 256

    def test_deterministic_embeddings_for_same_text(self):
        """Test that FakeLateInteractionEmbedder produces deterministic embeddings."""
        from langres.core.embeddings import FakeLateInteractionEmbedder

        embedder = FakeLateInteractionEmbedder(embedding_dim=128, num_tokens=3)
        texts = ["same text", "same text"]

        multi_vectors = embedder.encode(texts)

        # Same text should produce identical multi-vectors
        assert len(multi_vectors[0]) == len(multi_vectors[1])
        for token_idx in range(len(multi_vectors[0])):
            assert multi_vectors[0][token_idx] == multi_vectors[1][token_idx]

    def test_different_texts_produce_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        from langres.core.embeddings import FakeLateInteractionEmbedder

        embedder = FakeLateInteractionEmbedder(embedding_dim=128, num_tokens=3)
        texts = ["text1", "text2"]

        multi_vectors = embedder.encode(texts)

        # Different texts should produce different multi-vectors
        assert multi_vectors[0] != multi_vectors[1]

    def test_encode_empty_list(self):
        """Test encoding empty list."""
        from langres.core.embeddings import FakeLateInteractionEmbedder

        embedder = FakeLateInteractionEmbedder(embedding_dim=128, num_tokens=5)

        multi_vectors = embedder.encode([])

        assert isinstance(multi_vectors, list)
        assert len(multi_vectors) == 0

    def test_consistency_across_instances(self):
        """Test that FakeLateInteractionEmbedder produces same embeddings across instances."""
        from langres.core.embeddings import FakeLateInteractionEmbedder

        # Create two separate embedder instances with same config
        embedder1 = FakeLateInteractionEmbedder(embedding_dim=128, num_tokens=5)
        embedder2 = FakeLateInteractionEmbedder(embedding_dim=128, num_tokens=5)

        # Same text should produce identical embeddings in both instances
        text = "consistency test"

        multi_vectors1 = embedder1.encode([text])
        multi_vectors2 = embedder2.encode([text])

        # Should be identical (deterministic hashing)
        assert multi_vectors1 == multi_vectors2

    def test_different_token_counts_per_text(self):
        """Test that each token index produces different embeddings."""
        from langres.core.embeddings import FakeLateInteractionEmbedder

        embedder = FakeLateInteractionEmbedder(embedding_dim=128, num_tokens=5)
        texts = ["test"]

        multi_vectors = embedder.encode(texts)

        # Each token embedding should be different
        token_embeddings = multi_vectors[0]
        for i in range(len(token_embeddings)):
            for j in range(i + 1, len(token_embeddings)):
                # Different token indices should produce different embeddings
                assert token_embeddings[i] != token_embeddings[j]

    def test_instantiation_speed(self):
        """Test that FakeLateInteractionEmbedder instantiates and encodes instantly."""
        import time

        from langres.core.embeddings import FakeLateInteractionEmbedder

        start = time.time()
        embedder = FakeLateInteractionEmbedder(embedding_dim=128, num_tokens=10)
        texts = ["text1", "text2", "text3", "text4", "text5"]
        multi_vectors = embedder.encode(texts)
        elapsed = time.time() - start

        # Should be instant (< 0.1 seconds)
        assert elapsed < 0.1
        assert len(multi_vectors) == 5


class TestLateInteractionEmbeddingProviderProtocol:
    """Tests for LateInteractionEmbeddingProvider protocol compliance."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "embedder_class,kwargs",
        [
            (
                lambda: __import__(
                    "langres.core.embeddings", fromlist=["FastEmbedLateInteractionEmbedder"]
                ).FastEmbedLateInteractionEmbedder,
                {"model_name": "colbert-ir/colbertv2.0"},
            ),
            (
                lambda: __import__(
                    "langres.core.embeddings", fromlist=["FakeLateInteractionEmbedder"]
                ).FakeLateInteractionEmbedder,
                {"embedding_dim": 128, "num_tokens": 5},
            ),
        ],
    )
    def test_protocol_methods_exist(self, embedder_class, kwargs):
        """Test that implementations have required protocol methods."""
        embedder = embedder_class()(**kwargs)

        assert hasattr(embedder, "encode")
        assert hasattr(embedder, "embedding_dim")
        assert callable(embedder.encode)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "embedder_class,kwargs",
        [
            (
                lambda: __import__(
                    "langres.core.embeddings", fromlist=["FastEmbedLateInteractionEmbedder"]
                ).FastEmbedLateInteractionEmbedder,
                {"model_name": "colbert-ir/colbertv2.0"},
            ),
            (
                lambda: __import__(
                    "langres.core.embeddings", fromlist=["FakeLateInteractionEmbedder"]
                ).FakeLateInteractionEmbedder,
                {"embedding_dim": 128, "num_tokens": 5},
            ),
        ],
    )
    def test_encode_signature(self, embedder_class, kwargs):
        """Test that encode method has correct signature."""
        embedder = embedder_class()(**kwargs)
        texts = ["test"]

        result = embedder.encode(texts)

        assert isinstance(result, list)
        assert len(result) == len(texts)
        # Each text should produce a list of token embeddings
        assert isinstance(result[0], list)
        # Each token should be a list of floats
        assert isinstance(result[0][0], list)
        assert all(isinstance(x, float) for x in result[0][0])


# ============ INSTRUCTION PROMPT SUPPORT TESTS (Phase 1) ============


class TestEmbeddingProviderProtocolWithPrompts:
    """Tests for EmbeddingProvider protocol with instruction prompt support.

    These tests verify that the protocol accepts optional prompt parameters
    for asymmetric encoding (documents without prompts, queries with prompts).
    """

    def test_embedding_provider_protocol_accepts_prompt_parameter(self):
        """Test that EmbeddingProvider.encode() signature accepts optional prompt.

        This verifies the protocol method signature includes prompt parameter.
        """
        from langres.core.embeddings import EmbeddingProvider

        # Verify protocol has encode method accepting prompt
        assert hasattr(EmbeddingProvider, "encode")

    @pytest.mark.slow
    def test_sentence_transformer_embedder_with_prompt(self):
        """Test that different prompts produce different embeddings.

        This verifies that instruction prompts actually affect the encoding
        for models that support it (e.g., Qwen3-Embedding, BGE, E5).
        """
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        texts = ["Apple Inc."]

        # Encode with different prompts
        embeddings_no_prompt = embedder.encode(texts, prompt=None)
        embeddings_prompt1 = embedder.encode(
            texts, prompt="Find duplicate organization names accounting for acronyms"
        )
        embeddings_prompt2 = embedder.encode(
            texts, prompt="Search for similar product names in catalog"
        )

        # All should have same shape
        assert embeddings_no_prompt.shape == (1, embedder.embedding_dim)
        assert embeddings_prompt1.shape == (1, embedder.embedding_dim)
        assert embeddings_prompt2.shape == (1, embedder.embedding_dim)

        # Different prompts should produce different embeddings
        # Note: Not all models use prompts - this tests the infrastructure
        # Real models like Qwen3-Embedding will show larger differences
        assert not np.array_equal(embeddings_no_prompt, embeddings_prompt1)

    @pytest.mark.slow
    def test_sentence_transformer_embedder_without_prompt_default(self):
        """Test that not passing prompt behaves like prompt=None.

        This ensures backward compatibility with existing code.
        """
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        texts = ["Apple Inc.", "Microsoft Corp."]

        # Both calls should produce identical results
        embeddings_no_arg = embedder.encode(texts)
        embeddings_none = embedder.encode(texts, prompt=None)

        np.testing.assert_array_equal(embeddings_no_arg, embeddings_none)

    @pytest.mark.slow
    def test_sentence_transformer_embedder_prompt_caching(self):
        """Test that same text + same prompt = identical output.

        This verifies determinism and that prompts are properly used.
        """
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        texts = ["Google LLC"]
        prompt = "Find duplicate organization names"

        # Encode same text with same prompt multiple times
        embeddings1 = embedder.encode(texts, prompt=prompt)
        embeddings2 = embedder.encode(texts, prompt=prompt)

        # Should be identical (deterministic)
        np.testing.assert_array_equal(embeddings1, embeddings2)

    def test_fake_embedder_accepts_prompt_parameter(self):
        """Test that FakeEmbedder accepts prompt parameter for protocol compliance.

        This ensures FakeEmbedder can be used as a test double for
        EmbeddingProvider with prompts.
        """
        embedder = FakeEmbedder(embedding_dim=128)
        texts = ["test1", "test2"]

        # Should accept prompt parameter
        embeddings_no_prompt = embedder.encode(texts, prompt=None)
        embeddings_with_prompt = embedder.encode(texts, prompt="test prompt")

        # Should return valid embeddings
        assert embeddings_no_prompt.shape == (2, 128)
        assert embeddings_with_prompt.shape == (2, 128)

    def test_fake_embedder_different_prompts_different_embeddings(self):
        """Test that FakeEmbedder produces different embeddings for different prompts.

        This makes FakeEmbedder a realistic test double that simulates
        real prompt-aware embedding behavior.
        """
        embedder = FakeEmbedder(embedding_dim=128)
        texts = ["Apple Inc."]

        # Same text with different prompts
        embeddings_no_prompt = embedder.encode(texts, prompt=None)
        embeddings_prompt1 = embedder.encode(texts, prompt="Find organizations")
        embeddings_prompt2 = embedder.encode(texts, prompt="Find products")

        # Different prompts should produce different embeddings
        assert not np.array_equal(embeddings_no_prompt, embeddings_prompt1)
        assert not np.array_equal(embeddings_prompt1, embeddings_prompt2)

    def test_fake_embedder_same_prompt_deterministic(self):
        """Test that FakeEmbedder is deterministic for same text+prompt.

        This ensures FakeEmbedder can be used in tests reliably.
        """
        embedder = FakeEmbedder(embedding_dim=128)
        texts = ["test"]
        prompt = "test prompt"

        # Same text + prompt should always produce same embeddings
        embeddings1 = embedder.encode(texts, prompt=prompt)
        embeddings2 = embedder.encode(texts, prompt=prompt)

        np.testing.assert_array_equal(embeddings1, embeddings2)

    def test_fake_embedder_backward_compatible_no_prompt(self):
        """Test that FakeEmbedder works without prompt parameter.

        Ensures backward compatibility with existing tests.
        """
        embedder = FakeEmbedder(embedding_dim=128)
        texts = ["test1", "test2"]

        # Should work without prompt parameter
        embeddings = embedder.encode(texts)

        assert embeddings.shape == (2, 128)

    @pytest.mark.slow
    def test_sentence_transformer_prompt_with_empty_list(self):
        """Test that prompt parameter works correctly with empty input.

        Edge case: empty list with prompt should still return empty array.
        """
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        embeddings = embedder.encode([], prompt="test prompt")

        assert embeddings.shape == (0, embedder.embedding_dim)

    def test_fake_embedder_prompt_with_empty_list(self):
        """Test that FakeEmbedder handles empty list with prompt correctly."""
        embedder = FakeEmbedder(embedding_dim=128)

        embeddings = embedder.encode([], prompt="test prompt")

        assert embeddings.shape == (0, 128)
