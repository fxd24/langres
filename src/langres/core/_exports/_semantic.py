"""The embedding/vector stack (the ``[semantic]`` extra) -- lazy in its entirety.

Every name here pulls torch / sentence-transformers / faiss / qdrant-client, so
nothing in this fragment is imported at module scope: ``__all__`` is empty and
all names resolve through ``langres.core.__getattr__`` on first access.

See ``langres.core._exports`` for the fragment contract.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy names visible to `mypy --strict`
    # without pulling the [semantic] stack into a bare `import langres`.
    from langres.core.embeddings import (
        EmbeddingProvider,
        FakeEmbedder,
        FakeSparseEmbedder,
        FastEmbedSparseEmbedder,
        SentenceTransformerEmbedder,
        SparseEmbeddingProvider,
    )
    from langres.core.indexes import (
        FAISSIndex,
        FakeHybridVectorIndex,
        FakeVectorIndex,
        QdrantHybridIndex,
        VectorIndex,
    )

#: Nothing is eager here by design -- see the module docstring.
__all__: list[str] = []

LAZY_SUBMODULES: tuple[str, ...] = ()

LAZY_SYMBOLS: dict[str, str] = {
    "EmbeddingProvider": "langres.core.embeddings",
    "FakeEmbedder": "langres.core.embeddings",
    "FakeSparseEmbedder": "langres.core.embeddings",
    "FastEmbedSparseEmbedder": "langres.core.embeddings",
    "SentenceTransformerEmbedder": "langres.core.embeddings",
    "SparseEmbeddingProvider": "langres.core.embeddings",
    "FAISSIndex": "langres.core.indexes",
    "FakeHybridVectorIndex": "langres.core.indexes",
    "FakeVectorIndex": "langres.core.indexes",
    "QdrantHybridIndex": "langres.core.indexes",
    "VectorIndex": "langres.core.indexes",
}

EXTRA_BY_SYMBOL: dict[str, str] = dict.fromkeys(LAZY_SYMBOLS, "semantic")

NAMES: tuple[str, ...] = (*__all__, *LAZY_SUBMODULES, *LAZY_SYMBOLS)
