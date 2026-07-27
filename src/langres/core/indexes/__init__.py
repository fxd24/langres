"""Vector index implementations with import-light compatibility exports.

The historical FAISS and hybrid indexes remain available from this package,
but resolving the Qdrant dense research path must not import every backend.
PEP 562 lazy exports preserve the public API without loading FAISS as a side
effect of importing :mod:`langres.core.indexes.qdrant_dense_index`.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langres.core.indexes.hybrid_vector_index import (
        FakeHybridVectorIndex,
        QdrantHybridIndex,
    )
    from langres.core.indexes.qdrant_dense_index import QdrantDenseIndex
    from langres.core.indexes.reranking_vector_index import (
        FakeHybridRerankingVectorIndex,
        QdrantHybridRerankingIndex,
    )
    from langres.core.indexes.vector_index import (
        FAISSIndex,
        FakeVectorIndex,
        VectorIndex,
    )

    def _assert_vector_index_conformance(
        faiss: FAISSIndex,
        fake: FakeVectorIndex,
        hybrid: QdrantHybridIndex,
        hybrid_fake: FakeHybridVectorIndex,
        reranking: QdrantHybridRerankingIndex,
        reranking_fake: FakeHybridRerankingVectorIndex,
    ) -> None:
        """Pin which indexes satisfy :class:`VectorIndex` — checked by mypy.

        Three docstrings in this package claimed "Implements VectorIndex
        protocol" while their ``search`` accepted a strictly narrower
        ``query_texts`` than the protocol. Nothing could contradict them:
        ``VectorBlocker.__init__`` types its parameter as ``VectorIndex`` and its
        docs name ``QdrantHybridIndex`` as a production choice, but the actual
        ``VectorBlocker(vector_index=QdrantHybridIndex(...))`` construction only
        appears under ``examples/``, and CI runs ``uv run mypy src``
        (``.github/workflows/lint.yml:52``) — **src only**. The one gate that
        could catch a Protocol mismatch structurally never saw the call site.

        This function is that call site, living where mypy does look. It is
        ``TYPE_CHECKING``-only: never imported, never called, zero runtime cost.

        The negative cases carry ``# type: ignore[assignment]`` rather than being
        omitted, and that is the load-bearing part. ``strict = true`` implies
        ``warn_unused_ignores``, so if one of those classes is ever widened to
        conform, its ignore becomes unnecessary and **mypy fails** — sending the
        author to the class docstring that says it does not conform. An omitted
        case would just go quietly out of date, which is the failure this whole
        change is about.
        """
        conforms: VectorIndex
        conforms = faiss
        conforms = fake

        # Narrower ``search``: hybrid retrieval needs the query text for the
        # sparse side, so it cannot accept the protocol's ``np.ndarray`` branch.
        conforms = hybrid  # type: ignore[assignment]
        conforms = hybrid_fake  # type: ignore[assignment]
        conforms = reranking  # type: ignore[assignment]
        conforms = reranking_fake  # type: ignore[assignment]


__all__ = [
    "FAISSIndex",
    "FakeVectorIndex",
    "VectorIndex",
    "QdrantDenseIndex",
    "QdrantHybridIndex",
    "FakeHybridVectorIndex",
    "QdrantHybridRerankingIndex",
    "FakeHybridRerankingVectorIndex",
]

_LAZY_SYMBOLS: dict[str, str] = {
    "FAISSIndex": "langres.core.indexes.vector_index",
    "FakeVectorIndex": "langres.core.indexes.vector_index",
    "VectorIndex": "langres.core.indexes.vector_index",
    "QdrantDenseIndex": "langres.core.indexes.qdrant_dense_index",
    "QdrantHybridIndex": "langres.core.indexes.hybrid_vector_index",
    "FakeHybridVectorIndex": "langres.core.indexes.hybrid_vector_index",
    "QdrantHybridRerankingIndex": ("langres.core.indexes.reranking_vector_index"),
    "FakeHybridRerankingVectorIndex": ("langres.core.indexes.reranking_vector_index"),
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_SYMBOLS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value
