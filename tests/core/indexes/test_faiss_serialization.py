"""Gating test for FAISS heavy-asset persistence (Wave 2d).

Deterministic and fast (NO real model download): uses ``FakeEmbedder`` so the
FAISS index round-trips without loading sentence-transformers. This test stays
in the non-slow lane.

It verifies the Wave 2d contract:

1. ``save_state`` writes the sidecar files (FAISS index, embeddings ``.npy``,
   corpus JSON).
2. ``load_state`` restores the stored vectors verbatim — a loaded index reuses
   the persisted embeddings and never re-embeds.
3. ``search_all`` returns identical neighbor indices before and after a
   save/load cycle.
4. ``config`` -> ``from_config`` round-trips the light construction config
   (metric + nested embedder spec) without touching the heavy state.
"""

import json

import numpy as np

from langres.core.embeddings import FakeEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.registry import get_component
from langres.core.serialization import ComponentSpec, SerializableState

# A small, fixed corpus — deterministic embeddings via FakeEmbedder.
_CORPUS = [
    "Apple Inc.",
    "Apple Incorporated",
    "Microsoft Corp.",
    "Microsoft Corporation",
    "Google LLC",
    "Alphabet Inc.",
    "Amazon.com Inc.",
    "Amazon Web Services",
    "Meta Platforms",
    "Facebook Inc.",
    "Netflix Inc.",
    "Tesla Motors",
    "Tesla Inc.",
    "Nvidia Corporation",
    "Intel Corp.",
]


class TestFAISSIndexSerializableState:
    """save_state / load_state behaviour with a deterministic FakeEmbedder."""

    def test_faiss_index_implements_serializable_state(self) -> None:
        index = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")
        assert isinstance(index, SerializableState)

    def test_save_state_writes_sidecar_files(self, tmp_path) -> None:
        index = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")
        index.create_index(_CORPUS)

        index.save_state(tmp_path)

        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "corpus_embeddings.npy").exists()
        corpus_json = tmp_path / "corpus.json"
        assert corpus_json.exists()

        payload = json.loads(corpus_json.read_text())
        assert payload["corpus_texts"] == _CORPUS
        assert payload["metric"] == "cosine"

    def test_save_state_unbuilt_index_writes_nothing(self, tmp_path) -> None:
        """A never-built index has no heavy state — save is a clean no-op."""
        index = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")

        index.save_state(tmp_path)

        assert list(tmp_path.iterdir()) == []

    def test_load_state_reuses_stored_vectors_without_reembedding(self, tmp_path) -> None:
        """Loaded index reuses persisted embeddings — exact array equality, no re-embed."""
        original = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")
        original.create_index(_CORPUS)
        original.save_state(tmp_path)

        original_embeddings = original._corpus_embeddings
        assert original_embeddings is not None

        # Reconstruct with a fresh embedder that would EXPLODE if asked to embed,
        # proving load_state never re-embeds.
        loaded = FAISSIndex(embedder=_ExplodingEmbedder(), metric="cosine")
        loaded.load_state(tmp_path)

        assert loaded._corpus_embeddings is not None
        assert np.array_equal(loaded._corpus_embeddings, original_embeddings)
        assert loaded._corpus_texts == _CORPUS
        assert loaded.metric == "cosine"

    def test_search_all_identical_before_and_after_roundtrip(self, tmp_path) -> None:
        """search_all returns IDENTICAL neighbor indices after save/load."""
        original = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")
        original.create_index(_CORPUS)
        before_dist, before_idx = original.search_all(k=5)

        original.save_state(tmp_path)

        loaded = FAISSIndex(embedder=_ExplodingEmbedder(), metric="cosine")
        loaded.load_state(tmp_path)
        after_dist, after_idx = loaded.search_all(k=5)

        assert np.array_equal(after_idx, before_idx)
        assert np.allclose(after_dist, before_dist)

    def test_load_state_l2_metric_roundtrip(self, tmp_path) -> None:
        """L2 metric also round-trips (covers the non-cosine index reconstruction)."""
        original = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="L2")
        original.create_index(_CORPUS)
        before_dist, before_idx = original.search_all(k=3)
        original.save_state(tmp_path)

        loaded = FAISSIndex(embedder=_ExplodingEmbedder(), metric="L2")
        loaded.load_state(tmp_path)
        after_dist, after_idx = loaded.search_all(k=3)

        assert loaded.metric == "L2"
        assert np.array_equal(after_idx, before_idx)
        assert np.allclose(after_dist, before_dist)


class TestFAISSIndexConfig:
    """config / from_config round-trip (light construction config only)."""

    def test_config_shape(self) -> None:
        index = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")
        cfg = index.config()

        assert cfg.metric == "cosine"
        assert isinstance(cfg.embedder, ComponentSpec)
        assert cfg.embedder.type_name == "fake_embedder"
        assert cfg.embedder.config["embedding_dim"] == 32

    def test_from_config_rebuilds_embedder_via_registry(self) -> None:
        index = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")
        cfg = index.config()

        rebuilt = FAISSIndex.from_config(cfg)

        assert rebuilt.metric == "cosine"
        assert isinstance(rebuilt.embedder, FakeEmbedder)
        assert rebuilt.embedder.embedding_dim == 32

    def test_config_roundtrips_through_json(self) -> None:
        index = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")
        cfg = index.config()

        restored_cfg = type(cfg).model_validate_json(cfg.model_dump_json())
        rebuilt = FAISSIndex.from_config(restored_cfg)

        assert rebuilt.metric == "cosine"
        assert isinstance(rebuilt.embedder, FakeEmbedder)
        assert rebuilt.embedder.embedding_dim == 32

    def test_faiss_index_registered(self) -> None:
        assert get_component("faiss_index") is FAISSIndex

    def test_full_roundtrip_config_plus_state(self, tmp_path) -> None:
        """End-to-end: config rebuilds the shell, load_state rehydrates vectors."""
        original = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")
        original.create_index(_CORPUS)
        before_dist, before_idx = original.search_all(k=4)

        cfg = original.config()
        original.save_state(tmp_path)

        # Rebuild light shell from config, then rehydrate heavy state from disk.
        rebuilt = FAISSIndex.from_config(type(cfg).model_validate_json(cfg.model_dump_json()))
        rebuilt.load_state(tmp_path)
        after_dist, after_idx = rebuilt.search_all(k=4)

        assert np.array_equal(after_idx, before_idx)
        assert np.allclose(after_dist, before_dist)


class _ExplodingEmbedder:
    """Embedder that fails if ``encode`` is ever called.

    Used to prove ``load_state`` rehydrates stored vectors instead of
    re-embedding the corpus.
    """

    def encode(self, texts, prompt=None):  # type: ignore[no-untyped-def]
        raise AssertionError("load_state must not re-embed: encode() was called")

    @property
    def embedding_dim(self) -> int:
        return 32
