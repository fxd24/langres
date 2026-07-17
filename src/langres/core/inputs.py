"""The input adapter: raw dicts -> (schema, normalized records).

**Ported wholesale from the deleted ``langres.verbs`` (W4).** This is the layer
that made ``dedupe(records)`` work without a schema, and every rule in it exists
because dropping it silently corrupts data rather than raising:

- ``float('nan')`` -> ``None``, never the string ``"nan"`` (which scores as a
  real token against every other ``"nan"`` and merges unrelated records).
- a nested ``list``/``dict`` value **raises** instead of being stringified into
  ``"{'a': 1}"`` and fuzzy-matched.
- ids are the records' own ``"id"`` when *every* record has one, positional
  otherwise, and a **mix raises** -- ``str(record.get("id"))`` on two id-less
  records both read ``"None"``, a false duplicate-id collision.
- an inferred schema is memoized by field-set, so N calls over one record shape
  reuse one class instead of minting N.

It lives in ``core`` (not in an architecture) because it is a **contract**: every
architecture takes the same records and must normalize them identically. This is
the DRY half of the wave's anti-DRY rule -- ``architectures/`` repeats topology,
never input semantics.

Import-light by construction: stdlib + pydantic only, and it imports nothing from
``langres``, so it is a leaf every model can depend on.

Known limitation (inherited, unchanged): an *inferred* schema is ephemeral -- a
dynamically created class, not importable by name. A model built on one works
in-process, but reloading a saved artifact that references it in a FRESH process
raises the registry's ``SchemaNotRegistered``. Pass ``schema=<YourModel>``
explicitly for anything you intend to persist.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, create_model

__all__ = ["check_no_duplicate_ids", "normalize_records"]

#: Memoization by field-set: repeated calls over the same record shape reuse one
#: ephemeral class instead of minting a new one per call.
_INFERRED_SCHEMA_CACHE: dict[frozenset[str], type[BaseModel]] = {}


def _inferred_schema_name(field_names: frozenset[str]) -> str:
    """Deterministic ``Inferred_<sha8>`` name from a field-set (the cache key)."""
    digest = sha256("|".join(sorted(field_names)).encode()).hexdigest()[:8]
    return f"Inferred_{digest}"


def _infer_schema(field_names: frozenset[str]) -> type[BaseModel]:
    """Build (or reuse) an ephemeral all-``str | None`` schema for ``field_names``."""
    cached = _INFERRED_SCHEMA_CACHE.get(field_names)
    if cached is not None:
        return cached
    fields: dict[str, Any] = {"id": (str, ...)}
    for name in sorted(field_names):
        fields[name] = (str | None, None)
    schema: type[BaseModel] = create_model(_inferred_schema_name(field_names), **fields)
    _INFERRED_SCHEMA_CACHE[field_names] = schema
    return schema


def _coerce_scalar(value: Any) -> str | None:
    """Coerce one raw field value for an inferred (all-``str | None``) schema.

    ``None`` and ``float('nan')`` both become ``None`` -- never the string
    ``"nan"``, which would silently poison string-similarity scoring. A nested
    ``list``/``dict`` cannot be represented by a flat inferred field, so it
    raises with guidance rather than being silently stringified.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (list, dict)):
        raise ValueError(
            f"cannot infer a schema field from a nested {type(value).__name__} "
            f"value ({value!r}). Pass schema=<YourModel> explicitly to control "
            "how nested fields are compared."
        )
    return str(value)


def _field_union(records: Sequence[dict[str, Any]]) -> frozenset[str]:
    """Every key across ``records`` except ``"id"`` (handled separately)."""
    fields: set[str] = set()
    for record in records:
        fields.update(key for key in record if key != "id")
    return frozenset(fields)


def _resolve_ids(records: Sequence[dict[str, Any]]) -> list[str]:
    """Per-record id: the explicit ``"id"`` if EVERY record has one, else positional.

    Raises:
        ValueError: If some records carry an ``"id"`` and others do not
            (ambiguous -- which source should win?).
    """
    has_id = ["id" in record for record in records]
    if all(has_id):
        return [str(record["id"]) for record in records]
    if not any(has_id):
        return [str(i) for i in range(len(records))]
    raise ValueError(
        "some records have an 'id' key and some don't -- schema inference "
        "needs consistent id presence across all records. Add 'id' to every "
        "record (or none), or pass schema=<YourModel> explicitly."
    )


def check_no_duplicate_ids(ids: Sequence[str]) -> None:
    """Raise if ``ids`` repeats -- ``dedupe``'s batch-uniqueness contract.

    Deliberately NOT applied by :func:`normalize_records`: ``compare(a, a)``
    (scoring an entity against itself) is well-defined and must not raise, so
    only the batch verb enforces uniqueness.

    Raises:
        ValueError: On any repeated id, naming the duplicates.
    """
    if len(set(ids)) == len(ids):
        return
    seen: set[str] = set()
    dupes: set[str] = set()
    for i in ids:
        if i in seen:
            dupes.add(i)
        seen.add(i)
    raise ValueError(
        f"duplicate ids in input: {sorted(dupes)}; every record must have a unique id."
    )


def normalize_records(
    records: Sequence[dict[str, Any]], schema: type[BaseModel] | None = None
) -> tuple[type[BaseModel], list[dict[str, Any]]]:
    """Normalize raw records, inferring an ephemeral schema when none is given.

    The ONE entry point every model uses, so ``compare`` and ``dedupe`` -- and
    every architecture -- normalize identically.

    Args:
        records: Raw records (plain dicts).
        schema: An explicit Pydantic schema, or ``None`` to infer an ephemeral
            one from the records' own keys.

    Returns:
        ``(schema, normalized_records)``. With an explicit ``schema`` the values
        are passed through untouched and only ids are resolved -- the caller's
        schema owns its own field semantics (that is the point of passing one).
        With an inferred schema every value is coerced by :func:`_coerce_scalar`.

    Raises:
        ValueError: On inconsistent id presence, or (inferred schema only) a
            nested ``list``/``dict`` value.
    """
    ids = _resolve_ids(records)
    if schema is not None:
        # Explicit schema: resolve ids via the SAME rule, so the caller's path
        # cannot mistake "no id" for "duplicate id" -- str(record.get("id")) on
        # two id-less records both read "None", a false collision.
        return schema, [{**record, "id": rid} for record, rid in zip(records, ids, strict=True)]
    field_names = _field_union(records)
    coerced = [
        {"id": rid, **{name: _coerce_scalar(record.get(name)) for name in field_names}}
        for record, rid in zip(records, ids, strict=True)
    ]
    return _infer_schema(field_names), coerced
