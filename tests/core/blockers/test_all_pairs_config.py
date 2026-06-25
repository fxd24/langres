"""Config-registry plumbing tests for AllPairsBlocker (Wave 2b).

Covers the declarative ``schema=`` constructor that COEXISTS with the existing
``schema_factory=`` constructor, plus registry serialization
(``config`` / ``from_config``). A ``schema_factory``-constructed blocker is NOT
serializable (its factory is an opaque callable); only ``schema=``-constructed
blockers round-trip.
"""

import pytest
from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.models import CompanySchema
from langres.core.registry import get_component


class _Widget(BaseModel):
    """Schema used only by these tests (distinct registry key)."""

    id: str
    name: str
    sku: str | None = None


def _company_factory(record: dict) -> CompanySchema:
    return CompanySchema(
        id=record["id"],
        name=record["name"],
        address=record.get("address"),
    )


COMPANY_DATA = [
    {"id": "a", "name": "Acme", "address": "1 Main St"},
    {"id": "b", "name": "Acme", "address": "1 Main St"},
    {"id": "c", "name": "Beta", "address": "2 Oak Ave"},
]


def test_schema_constructor_builds_entities() -> None:
    """schema= reconstructs entities from record dicts via model_fields."""
    blocker = AllPairsBlocker(schema=CompanySchema)

    candidates = list(blocker.stream(COMPANY_DATA))

    # 3 records -> 3 pairs (N*(N-1)/2)
    assert len(candidates) == 3
    first = candidates[0]
    assert isinstance(first.left, CompanySchema)
    assert first.left.id == "a"
    assert first.left.name == "Acme"
    assert first.left.address == "1 Main St"
    assert first.right.id == "b"
    assert first.blocker_name == "all_pairs_blocker"


def test_schema_constructor_ignores_unknown_record_keys() -> None:
    """Only fields declared on the schema are pulled from the record dict."""
    blocker = AllPairsBlocker(schema=CompanySchema)
    data = [
        {"id": "a", "name": "Acme", "extra": "ignored"},
        {"id": "b", "name": "Beta", "junk": 99},
    ]

    candidates = list(blocker.stream(data))

    assert len(candidates) == 1
    assert candidates[0].left.name == "Acme"
    assert candidates[0].right.name == "Beta"


def test_schema_factory_still_works() -> None:
    """Existing schema_factory= constructor is unchanged (coexistence)."""
    blocker = AllPairsBlocker(schema_factory=_company_factory)

    candidates = list(blocker.stream(COMPANY_DATA))

    assert len(candidates) == 3
    assert candidates[0].left.id == "a"
    assert candidates[0].right.id == "b"


def test_both_schema_and_factory_raises() -> None:
    """Passing both schema and schema_factory is a clear error."""
    with pytest.raises(ValueError, match="exactly one"):
        AllPairsBlocker(schema=CompanySchema, schema_factory=_company_factory)


def test_neither_schema_nor_factory_raises() -> None:
    """Passing neither schema nor schema_factory is a clear error."""
    with pytest.raises(ValueError, match="exactly one"):
        AllPairsBlocker()


def test_registered_under_type_name() -> None:
    """AllPairsBlocker is registered under 'all_pairs_blocker'."""
    assert get_component("all_pairs_blocker") is AllPairsBlocker


def test_config_shape_for_schema_blocker() -> None:
    """config exposes the schema type name from the schema registry."""
    blocker = AllPairsBlocker(schema=_Widget)

    config = blocker.config

    assert config == {"schema_type_name": "_Widget"}


def test_from_config_reconstructs_equivalent_blocker() -> None:
    """from_config(config) rebuilds a blocker producing identical candidates."""
    blocker = AllPairsBlocker(schema=CompanySchema)
    config = blocker.config

    rebuilt = AllPairsBlocker.from_config(config)

    before = list(blocker.stream(COMPANY_DATA))
    after = list(rebuilt.stream(COMPANY_DATA))

    def _key(cands: list) -> list[tuple[str, str]]:
        return [(c.left.id, c.right.id) for c in cands]

    assert _key(before) == _key(after)


def test_config_roundtrip_reproduces_clusters() -> None:
    """config -> from_config preserves the exact candidate pair set/order."""
    blocker = AllPairsBlocker(schema=CompanySchema)
    rebuilt = AllPairsBlocker.from_config(blocker.config)

    pairs = {frozenset((c.left.id, c.right.id)) for c in rebuilt.stream(COMPANY_DATA)}

    assert pairs == {
        frozenset(("a", "b")),
        frozenset(("a", "c")),
        frozenset(("b", "c")),
    }


def test_factory_blocker_config_raises_not_serializable() -> None:
    """A schema_factory-constructed blocker cannot serialize its config."""
    blocker = AllPairsBlocker(schema_factory=_company_factory)

    with pytest.raises(ValueError, match="not serializable"):
        _ = blocker.config


def test_repeated_schema_registration_is_idempotent() -> None:
    """Constructing two blockers with the same schema must not raise."""
    AllPairsBlocker(schema=CompanySchema)
    # Second construction with the same schema type must not raise a duplicate
    # registration error.
    second = AllPairsBlocker(schema=CompanySchema)
    assert second.config == {"schema_type_name": "CompanySchema"}


def test_schema_name_collision_with_different_class_raises() -> None:
    """Two distinct classes sharing a __name__ is a real collision."""
    from langres.core.blockers.all_pairs import register_schema_idempotent

    class _Collide(BaseModel):
        id: str

    register_schema_idempotent(_Collide)

    # A second, different class with the same __name__ must raise.
    class _Outer:
        class _Collide(BaseModel):  # noqa: N801 - intentional name reuse
            id: str
            extra: str | None = None

    with pytest.raises(ValueError, match="already registered to a different class"):
        register_schema_idempotent(_Outer._Collide)
