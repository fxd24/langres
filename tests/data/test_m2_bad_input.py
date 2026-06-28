"""M2 bad-input contract — the resolver's documented behavior on degenerate input.

These are the DX-hardening assertions from the Wave 3 review: a brainsquad
integrator must know exactly what ``resolve`` does at the edges. They run FAST —
the resolver here mirrors :func:`build_restaurant_resolver` (same
``RestaurantSchema`` + ``Comparator.from_schema`` + ``WeightedAverageJudge`` +
``Clusterer`` wiring) but swaps the MiniLM embedder for a ``FakeEmbedder`` so no
model loads. The two contracts under test are embedder-independent:

- **Empty corpus** -> ``resolve([])`` returns ``[]`` (no clusters). Verified
  identical with the real MiniLM resolver.
- **Missing required field** -> the schema factory raises ``pydantic.ValidationError``
  naming the missing field, and it surfaces at schema-construction time (before
  any embedding).
"""

import pydantic
import pytest

from langres.core import (
    Clusterer,
    Comparator,
    FAISSIndex,
    FakeEmbedder,
    Resolver,
    VectorBlocker,
    WeightedAverageJudge,
)
from langres.data.er_benchmarks import RestaurantSchema


def _fast_restaurant_resolver(threshold: float = 0.8) -> Resolver:
    """A RestaurantSchema resolver with a FakeEmbedder (no MiniLM load).

    Wired exactly like :func:`build_restaurant_resolver` apart from the embedder,
    so the schema-factory and clusterer paths under test are identical.
    """
    index = FAISSIndex(embedder=FakeEmbedder(embedding_dim=16), metric="cosine")
    blocker: VectorBlocker[RestaurantSchema] = VectorBlocker(
        vector_index=index,
        schema=RestaurantSchema,
        text_field="embed_text",
        k_neighbors=5,
    )
    comparator: Comparator[RestaurantSchema] = Comparator.from_schema(RestaurantSchema)
    return Resolver(
        blocker=blocker,
        comparator=comparator,
        module=WeightedAverageJudge(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=threshold),
    )


def test_resolve_empty_corpus_returns_no_clusters() -> None:
    """An empty corpus resolves to an empty cluster list (not an error)."""
    assert _fast_restaurant_resolver().resolve([]) == []


def test_resolve_record_missing_required_field_raises_naming_field() -> None:
    """A record missing required fields raises ValidationError naming them.

    ``name`` (a required ``str``) and ``source`` (a required ``Literal``) are both
    absent, so the schema factory fails fast — before any embedding — with a
    pydantic ``ValidationError`` whose message names the offending fields.
    """
    resolver = _fast_restaurant_resolver()
    # Two records so the path reaches schema construction (one record short-circuits).
    bad_records = [{"id": "f1"}, {"id": "z1"}]

    with pytest.raises(pydantic.ValidationError) as exc_info:
        resolver.resolve(bad_records)

    message = str(exc_info.value)
    assert "name" in message
    assert "source" in message
