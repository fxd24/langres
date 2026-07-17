"""Contract tests for the input adapter ported out of the deleted ``verbs.py``.

W4 deleted ``langres.verbs`` and moved its normalization layer to
``langres.core.inputs``. The code was ported wholesale; ``tests/test_verbs.py``
was deleted with the module -- so every rule below arrived in ``core`` (the
95-100% tier) **untested**. These tests re-establish the proof.

Each rule here exists because breaking it corrupts data *silently* rather than
raising, which is why they are asserted individually rather than through one
happy-path smoke test.
"""

from __future__ import annotations

import math

import pytest
from pydantic import BaseModel

from langres.core.inputs import (
    _coerce_scalar,
    _infer_schema,
    _resolve_ids,
    check_no_duplicate_ids,
    normalize_records,
)


class _Company(BaseModel):
    """An explicit schema -- the caller owns its field semantics."""

    id: str
    name: str
    city: str | None = None


class TestCoerceScalar:
    """The value rules for an inferred (all-``str | None``) schema."""

    def test_none_stays_none(self) -> None:
        assert _coerce_scalar(None) is None

    def test_nan_becomes_none_not_the_string_nan(self) -> None:
        """The data-corruption rule: ``"nan"`` scores as a real token.

        pandas hands out ``float('nan')`` for every empty cell. Stringified, two
        unrelated records both carry the literal ``"nan"``, which is a 1.0
        string-similarity match against each other -- silently merging entities
        that share nothing but a missing field.
        """
        assert _coerce_scalar(float("nan")) is None
        assert _coerce_scalar(math.nan) is None

    @pytest.mark.parametrize("value", [["a", "b"], {"a": 1}])
    def test_nested_value_raises_instead_of_being_stringified(self, value: object) -> None:
        """A nested value cannot be a flat inferred field, so it must not be guessed."""
        with pytest.raises(ValueError, match="nested"):
            _coerce_scalar(value)

    def test_nested_error_names_the_escape_hatch(self) -> None:
        """The error must tell the user what to do, not just what went wrong."""
        with pytest.raises(ValueError, match="schema=<YourModel>"):
            _coerce_scalar({"a": 1})

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, "1"), (2.5, "2.5"), (True, "True"), ("x", "x")],
    )
    def test_scalars_stringify(self, value: object, expected: str) -> None:
        assert _coerce_scalar(value) == expected

    def test_infinity_is_not_nan_and_survives(self) -> None:
        """Only NaN is missing-data; +/-inf is a real value and must not vanish."""
        assert _coerce_scalar(float("inf")) == "inf"


class TestResolveIds:
    """Ids are explicit when every record has one, positional when none does."""

    def test_all_records_have_id_uses_them(self) -> None:
        assert _resolve_ids([{"id": "a"}, {"id": "b"}]) == ["a", "b"]

    def test_non_string_ids_are_stringified(self) -> None:
        assert _resolve_ids([{"id": 1}, {"id": 2}]) == ["1", "2"]

    def test_no_record_has_id_falls_back_to_position(self) -> None:
        assert _resolve_ids([{"name": "a"}, {"name": "b"}]) == ["0", "1"]

    def test_mixed_id_presence_raises(self) -> None:
        """The false-collision rule.

        ``str(record.get("id"))`` on two id-less records both read ``"None"`` --
        a duplicate-id collision invented by the normalizer itself. Ambiguous
        input must raise rather than be resolved by a coin-flip.
        """
        with pytest.raises(ValueError, match="consistent id presence"):
            _resolve_ids([{"id": "a"}, {"name": "b"}])

    def test_empty_input_is_not_an_error(self) -> None:
        assert _resolve_ids([]) == []


class TestCheckNoDuplicateIds:
    """The batch-uniqueness contract -- enforced by dedupe, not by compare."""

    def test_unique_ids_pass(self) -> None:
        check_no_duplicate_ids(["a", "b", "c"])

    def test_duplicate_ids_raise_and_name_the_duplicates(self) -> None:
        with pytest.raises(ValueError, match=r"\['a'\]"):
            check_no_duplicate_ids(["a", "b", "a"])

    def test_every_duplicate_is_named_sorted(self) -> None:
        with pytest.raises(ValueError, match=r"\['a', 'b'\]"):
            check_no_duplicate_ids(["a", "b", "a", "b", "c"])

    def test_normalize_does_not_enforce_uniqueness(self) -> None:
        """``compare(a, a)`` is well-defined: an entity against itself.

        If ``normalize_records`` enforced uniqueness, self-comparison -- a
        legitimate operation -- would raise. Only the batch path checks.
        """
        _, records = normalize_records([{"id": "x", "name": "Acme"}, {"id": "x", "name": "Acme"}])
        assert [r["id"] for r in records] == ["x", "x"]


class TestNormalizeRecordsWithExplicitSchema:
    """With a schema the caller owns field semantics -- values pass through."""

    def test_returns_the_caller_schema_untouched(self) -> None:
        schema, _ = normalize_records([{"id": "1", "name": "Acme"}], schema=_Company)
        assert schema is _Company

    def test_values_are_not_coerced(self) -> None:
        """The point of passing a schema is that it, not us, defines the fields.

        A ``None`` city stays ``None`` -- we must not stringify it, and we must
        not apply the inferred path's nested-value rejection to a schema whose
        author may well want a nested field.
        """
        _, records = normalize_records([{"id": "1", "name": "Acme", "city": None}], schema=_Company)
        assert records == [{"id": "1", "name": "Acme", "city": None}]

    def test_ids_are_resolved_by_the_same_rule(self) -> None:
        """Explicit-schema callers get positional ids too -- one id rule, not two."""
        _, records = normalize_records([{"name": "Acme"}, {"name": "Ace"}], schema=_Company)
        assert [r["id"] for r in records] == ["0", "1"]

    def test_mixed_id_presence_still_raises_with_a_schema(self) -> None:
        with pytest.raises(ValueError, match="consistent id presence"):
            normalize_records([{"id": "a"}, {"name": "b"}], schema=_Company)


class TestNormalizeRecordsWithInferredSchema:
    """No schema: infer one from the records' own keys."""

    def test_infers_the_union_of_all_keys(self) -> None:
        schema, _ = normalize_records([{"name": "Acme"}, {"city": "Zurich"}])
        assert set(schema.model_fields) == {"id", "name", "city"}

    def test_missing_field_becomes_none_not_absent(self) -> None:
        """Every record gets every field, so the comparator sees a stable shape."""
        _, records = normalize_records([{"name": "Acme"}, {"city": "Zurich"}])
        assert records == [
            {"id": "0", "name": "Acme", "city": None},
            {"id": "1", "name": None, "city": "Zurich"},
        ]

    def test_nan_is_coerced_through_the_public_entry_point(self) -> None:
        """The NaN rule must hold end-to-end, not just in the private helper."""
        _, records = normalize_records([{"name": "Acme", "city": float("nan")}])
        assert records[0]["city"] is None

    def test_nested_value_raises_through_the_public_entry_point(self) -> None:
        with pytest.raises(ValueError, match="nested"):
            normalize_records([{"name": "Acme", "tags": ["a", "b"]}])

    def test_the_inferred_schema_validates_its_own_output(self) -> None:
        """The returned records must actually satisfy the returned schema."""
        schema, records = normalize_records([{"name": "Acme", "city": float("nan")}])
        instance = schema.model_validate(records[0])
        assert instance.id == "0"

    def test_empty_records_infer_an_id_only_schema(self) -> None:
        schema, records = normalize_records([])
        assert set(schema.model_fields) == {"id"}
        assert records == []


class TestInferredSchemaMemoization:
    """One class per field-set, not one per call."""

    def test_same_field_set_reuses_one_class(self) -> None:
        first = _infer_schema(frozenset({"name", "city"}))
        second = _infer_schema(frozenset({"city", "name"}))
        assert first is second

    def test_different_field_sets_get_different_classes(self) -> None:
        assert _infer_schema(frozenset({"name"})) is not _infer_schema(frozenset({"street"}))

    def test_the_name_is_deterministic_across_key_order(self) -> None:
        """A stable name is what makes the class identifiable in an error message."""
        assert (
            _infer_schema(frozenset({"b", "a"})).__name__
            == _infer_schema(frozenset({"a", "b"})).__name__
        )

    def test_repeated_normalize_calls_reuse_one_class(self) -> None:
        schema_a, _ = normalize_records([{"name": "Acme"}])
        schema_b, _ = normalize_records([{"name": "Ace"}])
        assert schema_a is schema_b
