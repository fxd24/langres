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

from langres.core.indexes.hybrid_vector_index import FakeHybridVectorIndex, QdrantHybridIndex
from langres.core.indexes.vector_index import FAISSIndex, FakeVectorIndex, VectorIndex


def _accepted_types(func: object, parameter: str) -> set[object]:
    """The set of types ``parameter`` accepts, flattening a union into members."""
    annotation = typing.get_type_hints(func)[parameter]
    args = typing.get_args(annotation)
    return set(args) if args else {annotation}


#: ``(class, does its ``search`` accept everything the protocol's does?)``.
#: ``QdrantHybridIndex`` is ``False`` **on purpose**: a hybrid index cannot serve
#: a pure-vector query because the sparse side must encode the text. The entry
#: records a real capability difference, and the class docstring says the same.
#: If someone widens it to conform, this test fails and sends them to that
#: docstring — which is the only way the prose and the code stay in agreement.
SEARCH_CONFORMANCE: tuple[tuple[type, bool], ...] = (
    (FAISSIndex, True),
    (FakeVectorIndex, True),
    (QdrantHybridIndex, False),
    (FakeHybridVectorIndex, False),
)


@pytest.mark.parametrize(("index_cls", "conforms"), SEARCH_CONFORMANCE)
def test_search_query_texts_conformance(index_cls: type, conforms: bool) -> None:
    """Each index accepts exactly the ``query_texts`` its docstring claims."""
    protocol_types = _accepted_types(VectorIndex.search, "query_texts")
    actual_types = _accepted_types(index_cls.search, "query_texts")

    assert protocol_types >= actual_types, (
        f"{index_cls.__name__}.search accepts {actual_types - protocol_types} "
        "which VectorIndex.search does not declare"
    )
    assert (actual_types >= protocol_types) is conforms, (
        f"{index_cls.__name__}.search accepts {sorted(map(str, actual_types))}; "
        f"VectorIndex.search declares {sorted(map(str, protocol_types))}. "
        "Conformance changed — update the class docstring and SEARCH_CONFORMANCE together."
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
