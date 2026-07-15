"""Tests for the autoresearch config->blocker :mod:`factory`.

Core contract code -> high coverage tier. The fast tests inject a ``FakeEmbedder``
via the ``embedder=`` seam so no model is downloaded; one ``@pytest.mark.slow``
smoke exercises the real ``SentenceTransformerEmbedder`` path. Covers a usable
built index, the metric pass-through, VectorBlocker streaming from a config, the
prebuilt-index requirement, the AllPairsBlocker path, and unknown-blocker
fail-loud.
"""

import pytest

from langres.core.autoresearch.factory import build_blocker_from_config, build_index
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import FakeEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.models import CompanySchema

# A tiny, offline dataset. build_index() must be fed texts in the same order as
# the records the resulting blocker streams.
_DATA = [
    {"id": "c1", "name": "Acme Corporation"},
    {"id": "c2", "name": "Acme Corp"},
    {"id": "c3", "name": "Globex Inc"},
    {"id": "c4", "name": "Globex Incorporated"},
]
_TEXTS = [r["name"] for r in _DATA]


def _fake_index(metric: str = "cosine") -> FAISSIndex:
    """Build a ready index over ``_TEXTS`` using a FakeEmbedder (no download)."""
    index = build_index(
        embedding_model="unused-with-fake",
        metric=metric,
        texts=_TEXTS,
        embedder=FakeEmbedder(embedding_dim=16),
    )
    assert isinstance(index, FAISSIndex)
    return index


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------


def test_build_index_returns_a_usable_built_index() -> None:
    index = _fake_index()
    # create_index() was called -> the FAISS index object exists and is searchable.
    assert index._index is not None
    distances, indices = index.search_all(k=2)
    assert distances.shape == (len(_TEXTS), 2)
    assert indices.shape == (len(_TEXTS), 2)


def test_build_index_passes_metric_through() -> None:
    assert _fake_index(metric="cosine").metric == "cosine"
    assert _fake_index(metric="L2").metric == "L2"


# ---------------------------------------------------------------------------
# build_blocker_from_config -- vector
# ---------------------------------------------------------------------------


def test_vector_config_builds_streaming_vector_blocker() -> None:
    index = _fake_index()
    config = {
        "blocker": "vector",
        "embedding_model": "unused-with-fake",
        "metric": "cosine",
        "text_field": "name",
        "k_neighbors": 2,
    }
    blocker = build_blocker_from_config(config, schema=CompanySchema, index=index)

    assert isinstance(blocker, VectorBlocker)
    assert blocker.k_neighbors == 2
    assert blocker.vector_index is index  # index is reused, not rebuilt

    candidates = list(blocker.stream(_DATA))
    assert candidates, "vector blocker should stream at least one candidate pair"
    ids = {frozenset((c.left.id, c.right.id)) for c in candidates}
    assert all(len(pair) == 2 for pair in ids)  # no self-pairs


def test_vector_config_without_index_raises() -> None:
    config = {"blocker": "vector", "text_field": "name", "k_neighbors": 2}
    with pytest.raises(ValueError, match="requires a prebuilt 'index'"):
        build_blocker_from_config(config, schema=CompanySchema, index=None)


# ---------------------------------------------------------------------------
# build_blocker_from_config -- all_pairs
# ---------------------------------------------------------------------------


def test_all_pairs_config_builds_all_pairs_blocker_ignoring_index() -> None:
    config = {"blocker": "all_pairs"}
    blocker = build_blocker_from_config(config, schema=CompanySchema, index=None)

    assert isinstance(blocker, AllPairsBlocker)
    candidates = list(blocker.stream(_DATA))
    # All unique unordered pairs of 4 records: C(4, 2) = 6.
    assert len(candidates) == 6


# ---------------------------------------------------------------------------
# build_blocker_from_config -- errors
# ---------------------------------------------------------------------------


def test_unknown_blocker_raises() -> None:
    with pytest.raises(ValueError, match="unknown blocker"):
        build_blocker_from_config({"blocker": "nope"}, schema=CompanySchema)


# ---------------------------------------------------------------------------
# slow: real SentenceTransformerEmbedder path (covers the non-injected branch)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_build_index_real_sentence_transformer_smoke() -> None:
    """build_index with no injected embedder loads a real model and still streams.

    Marked slow: downloads/loads all-MiniLM-L6-v2. Runs in the full suite, which
    is where the coverage gate (and this branch's coverage) is enforced.
    """
    index = build_index("all-MiniLM-L6-v2", "cosine", _TEXTS)
    config = {"blocker": "vector", "text_field": "name", "k_neighbors": 2}
    blocker = build_blocker_from_config(config, schema=CompanySchema, index=index)
    candidates = list(blocker.stream(_DATA))
    assert candidates
