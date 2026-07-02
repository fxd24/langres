"""Tests for CompositeBlocker (W1.3): union/intersection/difference algebra.

CompositeBlocker composes 2+ child Blockers' candidate pair-sets via a set
operation (recall-first ``"union"`` default, plus ``"intersection"`` and
``"difference"``), deduping by the canonical undirected pair key
(``frozenset({left.id, right.id})``) with first-seen semantics, and carries
per-pair provenance (which child(ren) produced it) on ``blocker_name``.
Schema-agnostic: exercised with both ``CompanySchema`` and a local
``ProductSchema``.
"""

from typing import Any

import pytest
from pydantic import BaseModel

from langres.core.blocker import Blocker
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.composite import CompositeBlocker
from langres.core.blockers.key import KeyBlocker
from langres.core.groups import ERCandidateGroup
from langres.core.models import CompanySchema, ERCandidate
from langres.core.registry import get_component
from tests.conftest import pairs_from_candidates, pairs_from_groups


class ProductSchema(BaseModel):
    """Second schema for schema-agnostic verification."""

    id: str
    title: str
    manufacturer: str | None = None


COMPANY_DATA = [
    {"id": "a", "name": "Acme", "address": "Zurich"},
    {"id": "b", "name": "Acme Corp", "address": "Zurich"},
    {"id": "c", "name": "Beta Inc", "address": "Geneva"},
    {"id": "d", "name": "Beta", "address": "Bern"},
]


def _key_blocker() -> KeyBlocker[CompanySchema]:
    """Blocks a/b (share 'zurich') only."""
    return KeyBlocker(schema=CompanySchema, key_field="address")


def _all_pairs_blocker() -> AllPairsBlocker[CompanySchema]:
    """Blocks every pair (superset of anything else)."""
    return AllPairsBlocker(schema=CompanySchema)


class _NameFirstLetterBlocker(KeyBlocker[CompanySchema]):
    """A second, disjoint-ish KeyBlocker for intersection/difference tests."""

    def __init__(self) -> None:
        super().__init__(
            schema=CompanySchema,
            key_fn=lambda c: c.name[0] if c.name else None,
        )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_composite_blocker_requires_at_least_two_children() -> None:
    """A composite of 0 or 1 children is not a meaningful algebra expression."""
    with pytest.raises(ValueError, match="at least 2"):
        CompositeBlocker(children=[])

    with pytest.raises(ValueError, match="at least 2"):
        CompositeBlocker(children=[_key_blocker()])


def test_composite_blocker_rejects_unknown_op() -> None:
    """Only union/intersection/difference are valid ops."""
    with pytest.raises(ValueError, match="union.*intersection.*difference|op"):
        CompositeBlocker(
            children=[_key_blocker(), _all_pairs_blocker()],
            op="xor",  # type: ignore[arg-type]
        )


def test_composite_blocker_defaults_to_union() -> None:
    """The recall-maximizing default composition is union."""
    composite = CompositeBlocker(children=[_key_blocker(), _all_pairs_blocker()])
    assert composite.op == "union"


# ---------------------------------------------------------------------------
# Union
# ---------------------------------------------------------------------------


def test_composite_blocker_union_is_pair_set_union() -> None:
    """Union yields every pair produced by ANY child, deduped."""
    composite = CompositeBlocker(children=[_key_blocker(), _NameFirstLetterBlocker()], op="union")
    candidates = list(composite.stream(COMPANY_DATA))

    # key_blocker (by address): {a,b} only.
    # name-first-letter blocker: "A"->{a,b}, "B"->{c,d}.
    # Union: {a,b} (both children agree) + {c,d} (only the letter blocker).
    assert pairs_from_candidates(candidates) == {
        frozenset({"a", "b"}),
        frozenset({"c", "d"}),
    }


def test_composite_blocker_union_dedups_pair_found_by_multiple_children() -> None:
    """A pair produced by 2+ children appears exactly once in the union."""
    composite = CompositeBlocker(children=[_key_blocker(), _all_pairs_blocker()], op="union")
    candidates = list(composite.stream(COMPANY_DATA))

    # AllPairs alone would give C(4,2)=6; union must not double-count {a,b}
    # (also produced by key_blocker).
    assert len(candidates) == 6
    assert len(pairs_from_candidates(candidates)) == 6


def test_composite_blocker_dedups_a_single_child_emitting_the_same_pair_twice() -> None:
    """Defensive dedup: a child that itself yields one pair twice still counts once."""

    class _DuplicateEmittingBlocker(Blocker[CompanySchema]):
        def stream(self, data):  # type: ignore[no-untyped-def]
            left = CompanySchema(id="a", name="Acme")
            right = CompanySchema(id="b", name="Acme Corp")
            yield ERCandidate(left=left, right=right, blocker_name="dup")
            yield ERCandidate(left=left, right=right, blocker_name="dup")

        def inspect_candidates(self, candidates, entities, sample_size=10):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    composite = CompositeBlocker(
        children=[_DuplicateEmittingBlocker(), _all_pairs_blocker()], op="union"
    )
    candidates = list(composite.stream(COMPANY_DATA))

    assert len({(c.left.id, c.right.id) for c in candidates}) == len(candidates)


def test_composite_blocker_union_is_recall_maximizing() -> None:
    """Union's candidate count is >= the max of any single child's count."""
    key_only = list(_key_blocker().stream(COMPANY_DATA))
    letter_only = list(_NameFirstLetterBlocker().stream(COMPANY_DATA))
    composite = CompositeBlocker(children=[_key_blocker(), _NameFirstLetterBlocker()], op="union")
    union_candidates = list(composite.stream(COMPANY_DATA))

    assert len(union_candidates) >= max(len(key_only), len(letter_only))
    # And it's a strict pairs-superset of each child.
    assert pairs_from_candidates(key_only) <= pairs_from_candidates(union_candidates)
    assert pairs_from_candidates(letter_only) <= pairs_from_candidates(union_candidates)


def test_composite_blocker_union_blocker_name_lists_contributing_children() -> None:
    """blocker_name records which child(ren) produced each pair (provenance)."""
    composite = CompositeBlocker(children=[_key_blocker(), _NameFirstLetterBlocker()], op="union")
    candidates = list(composite.stream(COMPANY_DATA))
    by_pair = {frozenset({c.left.id, c.right.id}): c.blocker_name for c in candidates}

    # {a,b} is produced by BOTH children -> both names present.
    ab_name = by_pair[frozenset({"a", "b"})]
    assert "key_blocker" in ab_name
    assert "composite_union" in ab_name

    # {c,d} is produced ONLY by the letter blocker.
    cd_name = by_pair[frozenset({"c", "d"})]
    assert "key_blocker" in cd_name  # both are KeyBlocker subclasses -> same type_name
    assert "composite_union" in cd_name


# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------


def test_composite_blocker_intersection_keeps_only_shared_pairs() -> None:
    """Intersection yields only pairs present in EVERY child."""
    composite = CompositeBlocker(children=[_key_blocker(), _all_pairs_blocker()], op="intersection")
    candidates = list(composite.stream(COMPANY_DATA))

    # key_blocker only ever produces {a,b}; AllPairs is a superset -> intersection = {a,b}.
    assert pairs_from_candidates(candidates) == {frozenset({"a", "b"})}


def test_composite_blocker_intersection_empty_when_no_overlap() -> None:
    """Intersection of children whose pair-sets don't overlap yields no pairs."""

    class _CityBlocker(KeyBlocker[CompanySchema]):
        def __init__(self) -> None:
            super().__init__(schema=CompanySchema, key_field="address")

    # Same city, different first letter -> the city blocker pairs them, the
    # letter blocker doesn't -> intersection is empty.
    disjoint_data = [
        {"id": "1", "name": "Xavier", "address": "Bern"},
        {"id": "2", "name": "Yara", "address": "Bern"},
    ]
    composite = CompositeBlocker(
        children=[_CityBlocker(), _NameFirstLetterBlocker()], op="intersection"
    )
    candidates = list(composite.stream(disjoint_data))

    assert candidates == []


# ---------------------------------------------------------------------------
# Difference
# ---------------------------------------------------------------------------


def test_composite_blocker_difference_keeps_first_minus_rest() -> None:
    """Difference yields pairs in the first child not present in any other child."""
    composite = CompositeBlocker(children=[_all_pairs_blocker(), _key_blocker()], op="difference")
    candidates = list(composite.stream(COMPANY_DATA))

    # AllPairs (6 pairs) minus key_blocker's {a,b} -> the other 5.
    assert pairs_from_candidates(candidates) == pairs_from_candidates(
        list(_all_pairs_blocker().stream(COMPANY_DATA))
    ) - {frozenset({"a", "b"})}
    assert len(candidates) == 5


def test_composite_blocker_difference_blocker_name_shows_subtrahend() -> None:
    """Difference provenance names the surviving (first) child only."""
    composite = CompositeBlocker(children=[_all_pairs_blocker(), _key_blocker()], op="difference")
    candidates = list(composite.stream(COMPANY_DATA))

    assert all("composite_difference" in c.blocker_name for c in candidates)
    assert all("all_pairs_blocker" in c.blocker_name for c in candidates)


# ---------------------------------------------------------------------------
# Dedup / first-seen semantics, empty/edge cases, schema-agnosticism
# ---------------------------------------------------------------------------


def test_composite_blocker_empty_data() -> None:
    """Empty input -> no candidates, for every op."""
    for op in ("union", "intersection", "difference"):
        composite = CompositeBlocker(
            children=[_key_blocker(), _all_pairs_blocker()],
            op=op,
        )
        assert list(composite.stream([])) == []


def test_composite_blocker_is_schema_agnostic_with_product_schema() -> None:
    """The same CompositeBlocker class works with a completely different schema."""

    def product_factory(record: dict[str, Any]) -> ProductSchema:
        return ProductSchema(
            id=record["id"], title=record["title"], manufacturer=record.get("manufacturer")
        )

    data = [
        {"id": "p1", "title": "iPhone 15", "manufacturer": "Apple"},
        {"id": "p2", "title": "iPhone 15 Pro", "manufacturer": "Apple"},
        {"id": "p3", "title": "Galaxy S24", "manufacturer": "Samsung"},
    ]
    key_blocker = KeyBlocker(schema_factory=product_factory, key_field="manufacturer")
    all_pairs = AllPairsBlocker(schema_factory=product_factory)
    composite = CompositeBlocker(children=[key_blocker, all_pairs], op="union")

    candidates = list(composite.stream(data))

    assert isinstance(candidates[0].left, ProductSchema)
    assert pairs_from_candidates(candidates) == {
        frozenset({"p1", "p2"}),
        frozenset({"p1", "p3"}),
        frozenset({"p2", "p3"}),
    }


def test_composite_blocker_nested_composite_children() -> None:
    """A CompositeBlocker can itself be a child of another CompositeBlocker."""
    inner = CompositeBlocker(children=[_key_blocker(), _NameFirstLetterBlocker()], op="union")
    outer = CompositeBlocker(children=[inner, _all_pairs_blocker()], op="intersection")

    candidates = list(outer.stream(COMPANY_DATA))
    # inner union = {a,b},{c,d}; AllPairs is a superset -> intersection = inner's pairs.
    assert pairs_from_candidates(candidates) == {frozenset({"a", "b"}), frozenset({"c", "d"})}


# ---------------------------------------------------------------------------
# Registry / config-registry serialization plumbing
# ---------------------------------------------------------------------------


def test_composite_blocker_registered_under_type_name() -> None:
    """CompositeBlocker is registered under 'composite_blocker'."""
    assert get_component("composite_blocker") is CompositeBlocker


def test_composite_blocker_config_round_trips_simple_children() -> None:
    """config/from_config round-trip for children with plain (non-state) config."""
    original = CompositeBlocker(children=[_key_blocker(), _all_pairs_blocker()], op="intersection")
    rebuilt: CompositeBlocker[CompanySchema] = CompositeBlocker.from_config(original.config)

    assert list(rebuilt.stream(COMPANY_DATA)) == list(original.stream(COMPANY_DATA))
    assert rebuilt.op == "intersection"


def test_composite_blocker_config_raises_for_non_serializable_child() -> None:
    """A child built from an opaque callable propagates its own config error."""
    non_serializable_child = KeyBlocker(schema=CompanySchema, key_fn=lambda c: c.name)
    composite = CompositeBlocker(children=[non_serializable_child, _all_pairs_blocker()])

    with pytest.raises(ValueError, match="key_fn"):
        _ = composite.config


def test_composite_blocker_config_raises_for_child_without_type_name() -> None:
    """A child with no registry type_name cannot be serialized."""

    class _UnregisteredBlocker(Blocker[CompanySchema]):
        def stream(self, data):  # type: ignore[no-untyped-def]
            return iter(())

        def inspect_candidates(self, candidates, entities, sample_size=10):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    composite = CompositeBlocker(children=[_UnregisteredBlocker(), _all_pairs_blocker()])

    with pytest.raises(ValueError, match="type_name"):
        _ = composite.config


# ---------------------------------------------------------------------------
# inspect_candidates
# ---------------------------------------------------------------------------


def test_composite_blocker_inspect_candidates_union_basic_shape() -> None:
    """inspect_candidates reports totals, distribution, examples, recommendations."""
    composite = CompositeBlocker(children=[_key_blocker(), _all_pairs_blocker()], op="union")
    entities = [
        CompanySchema(id=r["id"], name=r["name"], address=r.get("address")) for r in COMPANY_DATA
    ]
    candidates = list(composite.stream(COMPANY_DATA))

    report = composite.inspect_candidates(candidates, entities, sample_size=3)

    assert report.total_candidates == len(candidates)
    assert len(report.examples) == 3
    assert "union" in report.recommendations[0]


def test_composite_blocker_inspect_candidates_intersection_recommendation() -> None:
    """Non-union ops get a precision/recall trade-off warning."""
    composite = CompositeBlocker(children=[_key_blocker(), _all_pairs_blocker()], op="intersection")
    entities = [
        CompanySchema(id=r["id"], name=r["name"], address=r.get("address")) for r in COMPANY_DATA
    ]
    candidates = list(composite.stream(COMPANY_DATA))

    report = composite.inspect_candidates(candidates, entities)

    assert "Pair-Completeness" in report.recommendations[0]


def test_composite_blocker_inspect_candidates_falls_back_to_repr_without_name_field() -> None:
    """A schema with no 'name' attribute falls back to str(entity) for example text."""

    def product_factory(record: dict[str, Any]) -> ProductSchema:
        return ProductSchema(
            id=record["id"], title=record["title"], manufacturer=record.get("manufacturer")
        )

    data = [
        {"id": "p1", "title": "iPhone 15", "manufacturer": "Apple"},
        {"id": "p2", "title": "iPhone 15 Pro", "manufacturer": "Apple"},
    ]
    key_blocker = KeyBlocker(schema_factory=product_factory, key_field="manufacturer")
    all_pairs = AllPairsBlocker(schema_factory=product_factory)
    composite = CompositeBlocker(children=[key_blocker, all_pairs], op="union")
    entities = [product_factory(r) for r in data]
    candidates = list(composite.stream(data))

    report = composite.inspect_candidates(candidates, entities)

    assert report.examples[0]["left_text"] == str(entities[0])


# ---------------------------------------------------------------------------
# stream_groups(): inherited default (W1.0 exactly-once contract)
# ---------------------------------------------------------------------------


def test_composite_blocker_stream_groups_is_a_concrete_iterator() -> None:
    """CompositeBlocker gets a working stream_groups() from the Blocker base default."""
    composite = CompositeBlocker(children=[_key_blocker(), _all_pairs_blocker()], op="union")
    groups = list(composite.stream_groups(COMPANY_DATA))
    assert len(groups) > 0
    assert isinstance(groups[0], ERCandidateGroup)


@pytest.mark.parametrize("op", ["union", "intersection", "difference"])
def test_composite_blocker_stream_groups_pairs_equivalence_property(op: str) -> None:
    """Property (CEO #14): pairs from stream_groups() == pairs from stream(), for every op."""
    composite = CompositeBlocker(
        children=[_key_blocker(), _NameFirstLetterBlocker()],
        op=op,  # type: ignore[arg-type]
    )

    stream_pairs = pairs_from_candidates(composite.stream(COMPANY_DATA))
    group_pairs = pairs_from_groups(composite.stream_groups(COMPANY_DATA))

    assert group_pairs == stream_pairs
