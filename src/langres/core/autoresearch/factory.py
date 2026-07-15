"""Turn one autoresearch config dict into a runnable blocker.

Companion to ``search_space``: :class:`SearchSpace` enumerates config dicts and
this module builds the blocker each one describes. Two pure functions, no global
mutable state — index reuse across ``k`` values is the *caller's* job (it builds
one index and threads it into :func:`build_blocker_from_config`; see the
k-innermost ordering contract on ``SearchSpace.configs``), not a cache here.

**NOT import-light.** This module imports the [semantic] stack
(faiss/sentence-transformers, via the vector index + embedder) at module top, so
it must only ever be imported **lazily** — on demand when a search actually runs,
never from a bare ``import langres`` (unlike the sibling ``search_space``, which
stays pure so it can live on the public API surface). The autoresearch package
``__init__`` intentionally exports nothing, so importing ``langres`` does not
reach this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from langres.core.blocker import Blocker
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import EmbeddingProvider, SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex, VectorIndex


def build_index(
    embedding_model: str,
    metric: str,
    texts: Sequence[str],
    *,
    embedder: EmbeddingProvider | None = None,
) -> VectorIndex:
    """Build a FAISS index over ``texts``, ready for search.

    Mirrors the canonical VectorBlocker build path used across the codebase
    (embedder → ``FAISSIndex(embedder=..., metric=...)`` → ``create_index``;
    see ``core/presets.py:_build_vector_blocker`` and
    ``data/_benchmark_utils.py:sweep_blocking_k``). ``create_index`` is called
    here, so the returned index is immediately usable by a ``VectorBlocker`` —
    an un-built index raises when streamed.

    Args:
        embedding_model: sentence-transformers model name. Used to construct a
            :class:`SentenceTransformerEmbedder` unless ``embedder`` is injected.
        metric: FAISS metric, ``"L2"`` or ``"cosine"``. Passed to ``FAISSIndex``;
            an unknown value raises ``ValueError`` inside ``create_index``.
        texts: Corpus texts to embed and index, in the same order as the records
            the resulting blocker will stream.
        embedder: Optional pre-built embedder that overrides constructing one
            from ``embedding_model``. Tests inject a ``FakeEmbedder`` here to skip
            the model download; production leaves it ``None``.

    Returns:
        A ready-to-search :class:`FAISSIndex` (``create_index`` already called).
    """
    if embedder is None:
        embedder = SentenceTransformerEmbedder(embedding_model)
    index = FAISSIndex(embedder=embedder, metric=cast(Literal["L2", "cosine"], metric))
    index.create_index(list(texts))
    return index


def build_blocker_from_config(
    config: Mapping[str, Any],
    *,
    schema: type[Any],
    index: VectorIndex | None = None,
) -> Blocker[Any]:
    """Construct the blocker a config describes, dispatching on ``config["blocker"]``.

    Named ``build_blocker_from_config`` (not ``build_blocker``) to avoid colliding
    with ``Benchmark.build_blocker`` in the data layer.

    Dispatch:

    - ``"vector"`` — builds a :class:`VectorBlocker` over a **prebuilt** ``index``
      (from :func:`build_index`), mirroring the canonical declarative wiring
      (``schema=`` + ``text_field=`` keeps the blocker config-serializable).
      Raises ``ValueError`` if ``index is None``.
    - ``"all_pairs"`` — builds an :class:`AllPairsBlocker`; ``index`` is unused.

    Args:
        config: A config dict as yielded by ``SearchSpace.configs()``. For the
            vector path it must carry ``text_field`` and ``k_neighbors``.
        schema: The Pydantic record schema, passed declaratively so the blocker
            stays config-serializable.
        index: A prebuilt vector index. **Required** for ``blocker == "vector"``;
            ignored for ``"all_pairs"``.

    Returns:
        A ready-to-stream :class:`Blocker`.

    Raises:
        ValueError: If ``blocker == "vector"`` and ``index is None``, or if
            ``config["blocker"]`` names an unknown blocker kind.
    """
    blocker_kind = config["blocker"]
    if blocker_kind == "vector":
        if index is None:
            raise ValueError(
                "build_blocker_from_config requires a prebuilt 'index' for "
                "blocker='vector' (build one with build_index(...) first)."
            )
        return VectorBlocker(
            vector_index=index,
            schema=schema,
            text_field=config["text_field"],
            k_neighbors=config["k_neighbors"],
        )
    if blocker_kind == "all_pairs":
        return AllPairsBlocker(schema=schema)
    raise ValueError(
        f"unknown blocker {blocker_kind!r} in config (expected 'vector' or 'all_pairs')"
    )
