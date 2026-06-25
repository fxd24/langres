"""Serialization tests for embedders (Wave 2d).

``FakeEmbedder`` config/from_config is exercised fast (no model). The
``SentenceTransformerEmbedder`` round-trip is marked ``@pytest.mark.slow``
because reconstructing it should NOT load a model at construction time — we
assert ``_model`` stays ``None`` until the first ``encode`` — but a real
``encode`` would download/load weights, so the model-loading assertion lives in
the slow lane.
"""

import numpy as np
import pytest

from langres.core.embeddings import FakeEmbedder, SentenceTransformerEmbedder
from langres.core.registry import get_component


class TestFakeEmbedderConfig:
    """Fast: FakeEmbedder config / from_config round-trip."""

    def test_registered(self) -> None:
        assert get_component("fake_embedder") is FakeEmbedder

    def test_config_shape(self) -> None:
        cfg = FakeEmbedder(embedding_dim=64, normalize_embeddings=False).config()
        assert cfg.embedding_dim == 64
        assert cfg.normalize_embeddings is False

    def test_from_config_roundtrip(self) -> None:
        original = FakeEmbedder(embedding_dim=64, normalize_embeddings=False)
        cfg = original.config()
        rebuilt = FakeEmbedder.from_config(type(cfg).model_validate_json(cfg.model_dump_json()))

        assert rebuilt.embedding_dim == 64
        assert rebuilt.normalize_embeddings is False
        # Deterministic: same text -> same embedding after reconstruction.
        assert np.array_equal(original.encode(["x"]), rebuilt.encode(["x"]))


class TestSentenceTransformerEmbedderConfig:
    """Fast config-shape checks that do NOT load a model."""

    def test_registered(self) -> None:
        assert get_component("sentence_transformer_embedder") is SentenceTransformerEmbedder

    def test_config_shape(self) -> None:
        cfg = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=False,
        ).config()
        assert cfg.model_name == "all-MiniLM-L6-v2"
        assert cfg.batch_size == 16
        assert cfg.show_progress_bar is True
        assert cfg.normalize_embeddings is False

    def test_from_config_does_not_load_model(self) -> None:
        cfg = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2").config()
        rebuilt = SentenceTransformerEmbedder.from_config(
            type(cfg).model_validate_json(cfg.model_dump_json())
        )
        assert rebuilt.model_name == "all-MiniLM-L6-v2"
        assert rebuilt.batch_size == 32
        # Lazy: reconstruction must not load weights.
        assert rebuilt._model is None


@pytest.mark.slow
class TestSentenceTransformerEmbedderRoundtripSlow:
    """Slow: round-trips by model_name and stays lazy until first encode."""

    def test_roundtrip_lazy_then_encodes(self) -> None:
        original = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        cfg = original.config()

        rebuilt = SentenceTransformerEmbedder.from_config(
            type(cfg).model_validate_json(cfg.model_dump_json())
        )

        # Still lazy after reconstruction.
        assert rebuilt._model is None

        # First encode triggers the model load.
        embeddings = rebuilt.encode(["Apple Inc.", "Microsoft Corp."])

        assert rebuilt._model is not None
        assert embeddings.shape == (2, rebuilt.embedding_dim)
