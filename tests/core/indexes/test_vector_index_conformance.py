"""Does each index actually accept what ``VectorIndex`` promises?

``QdrantHybridIndex``'s docstring claimed "Implements VectorIndex protocol" while
its ``search`` accepted a strictly narrower ``query_texts`` than the protocol —
a conformance claim with nothing able to contradict it.

**Why this is a runtime signature test and not a mypy assertion.** The obvious
check is to assign each implementation to a ``VectorIndex``-typed name and let
the type checker complain. CI runs ``uv run mypy src`` (``.github/workflows/
lint.yml:52``) — *src only*. A mypy assertion living in ``tests/`` is never
checked by anything, which would make it precisely the kind of gate this repo
keeps finding decoupled from the thing it claims to check. So the comparison is
done at runtime, where the test suite can actually observe it.

The expectation table below is deliberately explicit rather than derived from the
classes under test: an expectation regenerated from the thing it is checking
cannot detect that thing changing.
"""

import inspect
import typing

import numpy as np
import pytest

import langres.core.indexes
from langres.core.indexes.hybrid_vector_index import FakeHybridVectorIndex, QdrantHybridIndex
from langres.core.indexes.reranking_vector_index import (
    FakeHybridRerankingVectorIndex,
    QdrantHybridRerankingIndex,
)
from langres.core.indexes.vector_index import FAISSIndex, FakeVectorIndex, VectorIndex


def _accepted_types(func: object, parameter: str) -> set[object]:
    """The set of types ``parameter`` accepts, flattening a union into members."""
    annotation = typing.get_type_hints(func)[parameter]
    args = typing.get_args(annotation)
    return set(args) if args else {annotation}


#: Text-only ``query_texts``: what the hybrid indexes accept. Both forms matter —
#: their docstrings promise a single query *and* a batch.
TEXT_ONLY: frozenset[object] = frozenset({str, list[str]})

#: The protocol's full set: text, batch, or pre-computed vectors.
TEXT_OR_VECTORS: frozenset[object] = frozenset({str, list[str], np.ndarray})

#: ``(class, the EXACT set its ``search`` accepts)``.
#:
#: Deliberately the exact set and not a "conforms?" boolean. A boolean only asks
#: "is this a superset of the protocol", so narrowing ``str | list[str]`` down to
#: bare ``str`` — dropping batch queries — would keep a boolean green while the
#: class docstring still promised both forms. The failure this file exists to
#: catch is precisely a docstring that outlives the signature under it.
#:
#: The hybrid entries are ``TEXT_ONLY`` **on purpose**: a hybrid index cannot
#: serve a pure-vector query, because the sparse side must encode the text. That
#: is a real capability difference and each class docstring says so; if one is
#: ever widened to conform, this fails and sends the author to that docstring.
SEARCH_QUERY_TYPES: tuple[tuple[type, frozenset[object]], ...] = (
    (FAISSIndex, TEXT_OR_VECTORS),
    (FakeVectorIndex, TEXT_OR_VECTORS),
    (QdrantHybridIndex, TEXT_ONLY),
    (FakeHybridVectorIndex, TEXT_ONLY),
    (QdrantHybridRerankingIndex, TEXT_ONLY),
    (FakeHybridRerankingVectorIndex, TEXT_ONLY),
)


@pytest.mark.parametrize(("index_cls", "expected"), SEARCH_QUERY_TYPES)
def test_search_query_texts_conformance(index_cls: type, expected: frozenset[object]) -> None:
    """Each index accepts exactly the ``query_texts`` its docstring claims."""
    protocol_types = _accepted_types(VectorIndex.search, "query_texts")
    actual_types = _accepted_types(index_cls.search, "query_texts")

    assert actual_types == set(expected), (
        f"{index_cls.__name__}.search accepts {sorted(map(str, actual_types))}, "
        f"expected {sorted(map(str, expected))}. If this is intentional, update the "
        "class docstring and SEARCH_QUERY_TYPES together — and the mypy assertion in "
        "langres/core/indexes/__init__.py, which pins the same fact for the checker."
    )
    assert actual_types <= protocol_types, (
        f"{index_cls.__name__}.search accepts {actual_types - protocol_types} "
        "which VectorIndex.search does not declare"
    )


#: Exported index classes deliberately outside the conformance table, each with
#: the reason it is not a ``VectorIndex`` claimant. An explicit list, so removing
#: an entry is a decision someone makes rather than a gap nobody notices.
NOT_VECTOR_INDEX_CLAIMANTS: dict[str, str] = {
    # No `search` at all, and a `search_all(vectors, *, k, groups=...)` of a
    # different shape entirely: the Qdrant dense research path, not a drop-in.
    "QdrantDenseIndex": "different method set — no search(), different search_all() signature",
    # The protocol itself, not an implementation of it.
    "VectorIndex": "the protocol",
}


def test_every_exported_index_is_classified() -> None:
    """A newly added index must be classified, not silently left uncovered.

    The *expected signature* per class cannot be derived from the classes without
    becoming tautological — that is why ``SEARCH_QUERY_TYPES`` is hand-written.
    But the **set of classes** can be, and reconciling it against the package's
    own ``__all__`` is what turns a list that rots closed *quietly* into one that
    rots closed *loudly*: add an index, and this fails until someone decides
    whether it claims the protocol.
    """
    exported = set(langres.core.indexes.__all__)
    classified = {cls.__name__ for cls, _ in SEARCH_QUERY_TYPES} | set(NOT_VECTOR_INDEX_CLAIMANTS)

    assert exported == classified, (
        f"unclassified exports: {sorted(exported - classified)}; "
        f"classified but no longer exported: {sorted(classified - exported)}. "
        "Add each new index to SEARCH_QUERY_TYPES with its exact accepted type set, "
        "or to NOT_VECTOR_INDEX_CLAIMANTS with the reason it is not a claimant."
    )


def test_the_protocol_still_has_the_ndarray_branch_this_test_is_about() -> None:
    """Guard the premise: the whole test is vacuous if the protocol drops ndarray.

    Without this, narrowing ``VectorIndex.search`` to ``str | list[str]`` would
    make every implementation conform and every assertion above pass — a green
    suite reporting agreement it never checked.
    """
    assert np.ndarray in _accepted_types(VectorIndex.search, "query_texts")


def test_non_conforming_indexes_still_provide_the_precomputed_vector_path() -> None:
    """The capability exists on ``QdrantHybridIndex``, under a private name.

    This is the substance of the finding: the protocol's ``np.ndarray`` branch
    and this private argument are two spellings of one capability, and that
    divergence is why neither a reader nor a type checker noticed the
    ``query_prompt`` discard that lived between them.
    """
    parameters = inspect.signature(QdrantHybridIndex.search).parameters
    assert "_dense_embeddings" in parameters
    assert np.ndarray in _accepted_types(QdrantHybridIndex.search, "_dense_embeddings")
