"""Tests for KeyBlocker (W1.3): exact/normalized-key blocking.

KeyBlocker buckets records by a configurable key (declarative ``key_field=``
or a full ``key_fn=`` callable, mirroring the ``schema=``/``schema_factory=``
mutual-exclusion pattern already used by ``AllPairsBlocker``/``VectorBlocker``)
and emits all pairs within each bucket. Schema-agnostic: exercised with both
``CompanySchema`` and a local ``ProductSchema``.
"""

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from langres.core.blocker import Blocker
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
    {"id": "a", "name": "Acme", "city": "Zurich"},
    {"id": "b", "name": "Acme Corp", "city": "  ZURICH  "},
    {"id": "c", "name": "Beta", "city": "Geneva"},
    {"id": "d", "name": "Gamma", "city": None},
]


def _company_factory(record: dict) -> CompanySchema:
    return CompanySchema(id=record["id"], name=record["name"], address=record.get("city"))


# ---------------------------------------------------------------------------
# Basic bucketing behavior
# ---------------------------------------------------------------------------


def test_key_blocker_pairs_only_within_matching_key() -> None:
    """Only records sharing a (normalized) key produce a candidate pair."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")

    candidates = list(blocker.stream(COMPANY_DATA))
    pairs = pairs_from_candidates(candidates)

    # "a"/"b" share the "zurich" key (after normalization); "c" is alone in
    # "geneva"; "d" has no key (city=None) and is excluded entirely.
    assert pairs == {frozenset({"a", "b"})}


def test_key_blocker_normalizes_case_and_whitespace_by_default() -> None:
    """normalize=True (default) lowercases + strips before bucketing."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")
    candidates = list(blocker.stream(COMPANY_DATA))

    assert len(candidates) == 1
    assert {candidates[0].left.id, candidates[0].right.id} == {"a", "b"}


def test_key_blocker_normalize_false_is_exact_match() -> None:
    """normalize=False requires byte-exact key equality."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address", normalize=False)
    candidates = list(blocker.stream(COMPANY_DATA))

    # "Zurich" != "  ZURICH  " verbatim -> no pair.
    assert candidates == []


def test_key_blocker_excludes_records_with_none_key() -> None:
    """A record whose key extraction is None never appears in any pair."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")
    candidates = list(blocker.stream(COMPANY_DATA))

    for cand in candidates:
        assert cand.left.id != "d"
        assert cand.right.id != "d"


def test_key_blocker_blocker_name() -> None:
    """Candidates carry blocker_name == 'key_blocker'."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")
    candidates = list(blocker.stream(COMPANY_DATA))

    assert all(c.blocker_name == "key_blocker" for c in candidates)


def test_key_blocker_empty_data() -> None:
    """Empty input -> no candidates."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")
    assert list(blocker.stream([])) == []


def test_key_blocker_single_record() -> None:
    """A single record never pairs with itself."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")
    assert list(blocker.stream([COMPANY_DATA[0]])) == []


def test_key_blocker_bucket_larger_than_two() -> None:
    """A 3-record bucket yields C(3,2) = 3 pairs."""
    data = [
        {"id": "1", "name": "A", "city": "Bern"},
        {"id": "2", "name": "B", "city": "Bern"},
        {"id": "3", "name": "C", "city": "Bern"},
    ]
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")
    candidates = list(blocker.stream(data))

    assert len(candidates) == 3
    assert pairs_from_candidates(candidates) == {
        frozenset({"1", "2"}),
        frozenset({"1", "3"}),
        frozenset({"2", "3"}),
    }


# ---------------------------------------------------------------------------
# key_fn (callable) construction path + schema-agnosticism
# ---------------------------------------------------------------------------


def test_key_blocker_key_fn_callable_path() -> None:
    """key_fn= gives full control over key extraction, e.g. first-letter blocking."""

    def first_letter(entity: CompanySchema) -> str | None:
        return entity.name[0] if entity.name else None

    data = [
        {"id": "1", "name": "Acme"},
        {"id": "2", "name": "Apex"},
        {"id": "3", "name": "Beta"},
    ]
    blocker = KeyBlocker(schema_factory=_company_factory, key_fn=first_letter)
    candidates = list(blocker.stream(data))

    assert pairs_from_candidates(candidates) == {frozenset({"1", "2"})}


def test_key_blocker_is_schema_agnostic_with_product_schema() -> None:
    """The same KeyBlocker class works with a completely different schema."""

    def product_factory(record: dict) -> ProductSchema:
        return ProductSchema(
            id=record["id"], title=record["title"], manufacturer=record.get("manufacturer")
        )

    data = [
        {"id": "p1", "title": "iPhone 15", "manufacturer": "Apple"},
        {"id": "p2", "title": "iPhone 15 Pro", "manufacturer": "apple"},
        {"id": "p3", "title": "Galaxy S24", "manufacturer": "Samsung"},
    ]
    blocker = KeyBlocker(schema_factory=product_factory, key_field="manufacturer")
    candidates = list(blocker.stream(data))

    assert len(candidates) == 1
    assert isinstance(candidates[0].left, ProductSchema)
    assert {candidates[0].left.id, candidates[0].right.id} == {"p1", "p2"}


# ---------------------------------------------------------------------------
# Constructor validation (mutual exclusion, mirroring AllPairsBlocker)
# ---------------------------------------------------------------------------


def test_key_blocker_requires_exactly_one_schema_arg() -> None:
    """Both schema and schema_factory (or neither) is an error."""
    with pytest.raises(ValueError, match="schema"):
        KeyBlocker(key_field="address")  # neither schema nor schema_factory

    with pytest.raises(ValueError, match="schema"):
        KeyBlocker(
            schema=CompanySchema,
            schema_factory=_company_factory,
            key_field="address",
        )


def test_key_blocker_requires_exactly_one_key_arg() -> None:
    """Both key_field and key_fn (or neither) is an error."""
    with pytest.raises(ValueError, match="key_field.*key_fn|key_fn.*key_field"):
        KeyBlocker(schema=CompanySchema)  # neither key_field nor key_fn

    with pytest.raises(ValueError, match="key_field.*key_fn|key_fn.*key_field"):
        KeyBlocker(schema=CompanySchema, key_field="address", key_fn=lambda e: e.name)


# ---------------------------------------------------------------------------
# Registry / config-registry serialization plumbing
# ---------------------------------------------------------------------------


def test_key_blocker_registered_under_type_name() -> None:
    """KeyBlocker is registered under 'key_blocker'."""
    assert get_component("key_blocker") is KeyBlocker


def test_key_blocker_config_shape() -> None:
    """config exposes schema_type_name + key_field + normalize."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address", normalize=False)

    assert blocker.config == {
        "schema_type_name": "CompanySchema",
        "key_field": "address",
        "normalize": False,
    }


def test_key_blocker_from_config_round_trips() -> None:
    """from_config rebuilds a functionally-equivalent KeyBlocker."""
    original = KeyBlocker(schema=CompanySchema, key_field="address")
    rebuilt = KeyBlocker.from_config(original.config)

    assert list(rebuilt.stream(COMPANY_DATA)) == list(original.stream(COMPANY_DATA))


def test_key_blocker_config_raises_for_schema_factory() -> None:
    """A schema_factory-built KeyBlocker cannot serialize its entity normalization."""
    blocker = KeyBlocker(schema_factory=_company_factory, key_field="address")

    with pytest.raises(ValueError, match="schema_factory"):
        _ = blocker.config


def test_key_blocker_config_raises_for_key_fn() -> None:
    """A key_fn-built KeyBlocker cannot serialize its key extraction."""
    blocker = KeyBlocker(schema=CompanySchema, key_fn=lambda e: e.name)

    with pytest.raises(ValueError, match="key_fn"):
        _ = blocker.config


# ---------------------------------------------------------------------------
# inspect_candidates
# ---------------------------------------------------------------------------


def test_key_blocker_inspect_candidates_basic_shape() -> None:
    """inspect_candidates reports totals, distribution, examples, recommendations."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")
    entities = [_company_factory(r) for r in COMPANY_DATA]
    candidates = list(blocker.stream(COMPANY_DATA))

    report = blocker.inspect_candidates(candidates, entities, sample_size=5)

    assert report.total_candidates == 1
    assert report.examples[0]["left_id"] in {"a", "b"}
    assert report.recommendations


def test_key_blocker_inspect_candidates_handles_zero_candidates() -> None:
    """No candidates (e.g. all keys unique) still produces a valid report."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")
    data = [
        {"id": "1", "name": "A", "city": "Bern"},
        {"id": "2", "name": "B", "city": "Chur"},
    ]
    entities = [_company_factory(r) for r in data]
    candidates = list(blocker.stream(data))

    report = blocker.inspect_candidates(candidates, entities)

    assert report.total_candidates == 0
    assert report.recommendations


# ---------------------------------------------------------------------------
# stream_groups(): inherited default (W1.0 exactly-once contract)
# ---------------------------------------------------------------------------


def test_key_blocker_stream_groups_is_a_concrete_iterator() -> None:
    """KeyBlocker gets a working stream_groups() from the Blocker base default."""
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")
    assert isinstance(blocker, Blocker)

    groups = list(blocker.stream_groups(COMPANY_DATA))
    assert len(groups) == 1
    assert isinstance(groups[0], ERCandidateGroup)


def test_key_blocker_stream_groups_pairs_equivalence_property() -> None:
    """Property (CEO #14): pairs from stream_groups() == pairs from stream()."""
    data = [
        {"id": "1", "name": "A", "city": "Bern"},
        {"id": "2", "name": "B", "city": "Bern"},
        {"id": "3", "name": "C", "city": "Bern"},
        {"id": "4", "name": "D", "city": "Chur"},
    ]
    blocker = KeyBlocker(schema=CompanySchema, key_field="address")

    stream_pairs = pairs_from_candidates(blocker.stream(data))
    group_pairs = pairs_from_groups(blocker.stream_groups(data))

    assert group_pairs == stream_pairs
    assert len(stream_pairs) == 3  # C(3,2) within the "bern" bucket


def _typecheck_stream_return(blocker: KeyBlocker[CompanySchema]) -> Iterator[ERCandidate[CompanySchema]]:
    """mypy-only helper: confirms KeyBlocker.stream's declared return type."""
    return blocker.stream([])
