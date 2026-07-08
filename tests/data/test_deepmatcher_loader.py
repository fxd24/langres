"""Unit tests for the generic DeepMatcher loader factory internals (Wave B).

Covers the id-scheme-safety path (integer prefixing + non-integer remap), the
schema-driven record builder's required-field fallback, the split-safe id
assertion, the single-char-prefix guard, and the lazy ``build_blocker`` — the
mechanics Wave C loaders depend on but the tiny-fixture happy path does not
exercise (its ids are already integers and never empty).
"""

import pytest

from langres.data._deepmatcher_loader import (
    SourceTable,
    _assert_split_safe_ids,
    _record_from_row,
    _table_id_map,
    make_deepmatcher_benchmark,
)
from langres.data.tiny_fixture import TinyFixtureBenchmark, TinyFixtureSchema


# --- _table_id_map: integer prefixing vs. non-integer remap ---------------------


def test_table_id_map_prefixes_integer_ids_verbatim() -> None:
    rows = [{"id": "0"}, {"id": "5"}, {"id": "42"}]
    assert _table_id_map(rows, "id", "a") == {"0": "a0", "5": "a5", "42": "a42"}


def test_table_id_map_remaps_non_integer_ids_to_synthetic_ints() -> None:
    # WDC-style string ids must be remapped so the final ids stay <prefix><int>
    # and the stratified split's int(id[1:]) never raises.
    rows = [{"id": "prod-x"}, {"id": "0-cat/17"}, {"id": "weird"}]
    id_map = _table_id_map(rows, "id", "w")
    assert id_map == {"prod-x": "w0", "0-cat/17": "w1", "weird": "w2"}


def test_table_id_map_remaps_when_any_id_is_non_integer() -> None:
    # A single non-integer id triggers a full remap (mixed tables are not partially
    # remapped — that would risk id collisions).
    rows = [{"id": "1"}, {"id": "abc"}]
    assert _table_id_map(rows, "id", "b") == {"1": "b0", "abc": "b1"}


def test_table_id_map_rejects_duplicate_raw_ids() -> None:
    # Duplicate raw ids would silently collapse to one final id (dropping a record),
    # so the map raises instead — for both the integer and remap paths.
    with pytest.raises(ValueError, match=r"duplicate raw ids.*'1'"):
        _table_id_map([{"id": "1"}, {"id": "1"}, {"id": "2"}], "id", "a")
    with pytest.raises(ValueError, match=r"duplicate raw ids.*'dup'"):
        _table_id_map([{"id": "dup"}, {"id": "dup"}], "id", "b")


# --- _record_from_row: required-field fallback ----------------------------------


def test_record_from_row_fills_required_field_with_empty_string() -> None:
    # ``name`` is required (no default); an empty cell falls back to "" while an
    # empty optional (``description``) stays absent -> None.
    record = _record_from_row(TinyFixtureSchema, {"name": "  ", "description": ""}, "a", "a9")
    assert record.id == "a9"
    assert record.source == "a"
    assert record.name == ""
    assert record.description is None


def test_record_from_row_ignores_extra_columns_and_strips() -> None:
    record = _record_from_row(
        TinyFixtureSchema,
        {"name": "  Widget ", "description": "desc", "price": "9.99"},
        "b",
        "b3",
    )
    assert record.name == "Widget"
    assert record.description == "desc"
    assert not hasattr(record, "price")


# --- _assert_split_safe_ids -----------------------------------------------------


def test_assert_split_safe_ids_passes_for_char_int_ids() -> None:
    corpus = [TinyFixtureSchema(id="a1", name="x", source="a")]
    _assert_split_safe_ids(corpus, "tiny")  # does not raise


def test_assert_split_safe_ids_raises_on_bad_id() -> None:
    corpus = [TinyFixtureSchema(id="prod_1", name="x", source="a")]
    with pytest.raises(ValueError, match=r"not <char><int>"):
        _assert_split_safe_ids(corpus, "tiny")


# --- make_deepmatcher_benchmark: prefix guard -----------------------------------


@pytest.mark.parametrize("bad_prefix", ["ab", "1", ""])
def test_factory_rejects_non_single_alpha_prefix(bad_prefix: str) -> None:
    with pytest.raises(ValueError, match="single alphabetic char"):
        make_deepmatcher_benchmark(
            name="tiny_fixture",  # schema already registered (idempotent)
            schema=TinyFixtureSchema,
            dataset_package="langres.data.datasets.tiny_fixture",
            table_a=SourceTable(file="tableA.csv", source="a", id_prefix=bad_prefix),
            table_b=SourceTable(file="tableB.csv", source="b", id_prefix="b"),
            split_files={"train": "train.csv"},
            blocking_k=5,
            threshold_grid=(0.5,),
            achieved_pc=1.0,
            gate_met=True,
        )


def test_factory_rejects_equal_prefixes() -> None:
    # Both tables single-alpha but EQUAL -> overlapping raw ids would collide.
    with pytest.raises(ValueError, match="DISTINCT id_prefixes"):
        make_deepmatcher_benchmark(
            name="tiny_fixture",
            schema=TinyFixtureSchema,
            dataset_package="langres.data.datasets.tiny_fixture",
            table_a=SourceTable(file="tableA.csv", source="a", id_prefix="a"),
            table_b=SourceTable(file="tableB.csv", source="b", id_prefix="a"),
            split_files={"train": "train.csv"},
            blocking_k=5,
            threshold_grid=(0.5,),
            achieved_pc=1.0,
            gate_met=True,
        )


# --- build_blocker (lazy [semantic] construction, no model load) ----------------


def test_build_blocker_returns_fresh_unbuilt_vector_blocker() -> None:
    blocker = TinyFixtureBenchmark().build_blocker(5)
    assert type(blocker).__name__ == "VectorBlocker"
    assert blocker.k_neighbors == 5


def test_benchmark_exposes_pinned_blocking_config() -> None:
    benchmark = TinyFixtureBenchmark()
    assert benchmark.name == "tiny_fixture"
    assert benchmark.blocking_k == 5
    assert benchmark.achieved_pc == 1.0
    assert benchmark.gate_met is True
