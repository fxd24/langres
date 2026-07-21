"""Qdrant-backed dense retrieval contract tests."""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient
from testcontainers.qdrant import QdrantContainer

from langres.core.indexes.qdrant_dense_index import QdrantDenseIndex


def test_qdrant_dense_index_searches_vectors_and_filters_groups() -> None:
    index = QdrantDenseIndex(client=QdrantClient(":memory:"), collection_name="dense-test")
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.999, 0.001],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    scores, neighbours = index.search_all(vectors, k=1, groups=["a", "a", "b", "b"])

    assert scores.shape == neighbours.shape == (4, 1)
    assert neighbours[0, 0] == 2
    assert all(
        neighbour < 0 or ["a", "a", "b", "b"][row] != ["a", "a", "b", "b"][neighbour]
        for row, values in enumerate(neighbours)
        for neighbour in values
    )


def test_qdrant_dense_index_rejects_group_count_mismatch() -> None:
    index = QdrantDenseIndex(client=QdrantClient(":memory:"), collection_name="dense-test")

    with pytest.raises(ValueError, match="groups"):
        index.search_all(np.ones((2, 3), dtype=np.float32), k=1, groups=["only-one"])


def test_qdrant_dense_index_returns_no_hits_for_one_vector() -> None:
    index = QdrantDenseIndex(client=QdrantClient(":memory:"), collection_name="dense-test")

    scores, neighbours = index.search_all(np.ones((1, 3), dtype=np.float32), k=5)

    assert scores.shape == neighbours.shape == (1, 0)


def test_qdrant_dense_index_against_real_qdrant_server() -> None:
    with QdrantContainer(image="qdrant/qdrant:v1.15.5") as qdrant:
        client = qdrant.get_client()
        index = QdrantDenseIndex(client=client, collection_name="langres-test")

        scores, neighbours = index.search_all(
            np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
            k=1,
        )

        assert neighbours.tolist() == [[1], [0], [1]]
        assert np.isfinite(scores).all()
