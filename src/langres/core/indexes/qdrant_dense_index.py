"""Qdrant adapter for dense retrieval over precomputed vectors."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np

_MISSING_QDRANT_MESSAGE = (
    "Qdrant retrieval requires the semantic extra: pip install langres[semantic]"
)


class QdrantDenseIndex:
    """Index precomputed dense vectors without owning the embedding model.

    ``Retrieve`` owns the model-bearing :class:`~langres.resources.Embedder`;
    this class owns only vector-store mechanics. A caller may inject a Qdrant
    client for a server or Testcontainer. With no client, Qdrant's in-process
    local mode keeps the out-of-the-box path zero-configuration.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._client = client
        self.collection_name = collection_name or f"langres_retrieve_{uuid4().hex}"
        self._owns_collection = False

    @property
    def client(self) -> Any:
        """Create the optional Qdrant client only when retrieval actually runs."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError(_MISSING_QDRANT_MESSAGE) from exc

            self._client = QdrantClient(":memory:")
        return self._client

    def search_all(
        self,
        vectors: np.ndarray,
        *,
        k: int,
        groups: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return each vector's nearest neighbours, optionally outside its group.

        Group exclusion is expressed in the Qdrant query itself, so disallowed
        hits do not consume a top-k slot. The query record itself is always
        excluded.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a two-dimensional matrix")
        if groups is not None and len(groups) != len(matrix):
            raise ValueError("groups must contain one value per vector")
        if not len(matrix):
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.int64),
            )
        limit = min(k, max(len(matrix) - 1, 0))
        if limit == 0:
            return (
                np.empty((len(matrix), 0), dtype=np.float32),
                np.empty((len(matrix), 0), dtype=np.int64),
            )

        try:
            from qdrant_client.models import (
                Distance,
                FieldCondition,
                Filter,
                HasIdCondition,
                MatchValue,
                PointStruct,
                QueryRequest,
                VectorParams,
            )
        except ImportError as exc:
            raise ImportError(_MISSING_QDRANT_MESSAGE) from exc

        client = self.client
        if client.collection_exists(self.collection_name):
            if not self._owns_collection:
                raise ValueError(
                    f"Qdrant collection {self.collection_name!r} already exists; "
                    "refusing to delete a collection this index did not create"
                )
            client.delete_collection(self.collection_name)
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=matrix.shape[1], distance=Distance.COSINE),
        )
        self._owns_collection = True
        client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=index,
                    vector=vector.tolist(),
                    payload={"_langres_group": groups[index]} if groups is not None else {},
                )
                for index, vector in enumerate(matrix)
            ],
            wait=True,
        )

        requests = []
        for index, vector in enumerate(matrix):
            excluded: list[Any] = [HasIdCondition(has_id=[index])]
            if groups is not None:
                excluded.append(
                    FieldCondition(
                        key="_langres_group",
                        match=MatchValue(value=groups[index]),
                    )
                )
            requests.append(
                QueryRequest(
                    query=vector.tolist(),
                    filter=Filter(must_not=excluded),
                    limit=limit,
                    with_payload=False,
                )
            )

        responses = client.query_batch_points(
            collection_name=self.collection_name,
            requests=requests,
        )
        scores = np.full((len(matrix), limit), np.nan, dtype=np.float32)
        neighbours = np.full((len(matrix), limit), -1, dtype=np.int64)
        for row, response in enumerate(responses):
            for column, point in enumerate(response.points[:limit]):
                scores[row, column] = float(point.score)
                neighbours[row, column] = int(point.id)
        return scores, neighbours


__all__ = ["QdrantDenseIndex"]
