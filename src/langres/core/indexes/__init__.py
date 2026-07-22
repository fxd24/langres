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
