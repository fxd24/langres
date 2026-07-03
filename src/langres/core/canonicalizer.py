"""Golden-record survivorship: merge an entity's records into one master record (M5 / W2.3).

Batch :meth:`Resolver.resolve` and incremental :meth:`Resolver.assign` answer
*which* records are the same entity. The **last mile** — Master Data Creation
(USE_CASES UC4) — is turning that group of records into a single **golden
record**: the field-by-field "which value wins" decision known as *survivorship*.

:class:`Canonicalizer` is that seam. Given a group of records (raw dicts, exactly
the shape ``resolve``/``assign``/``AnchorStore`` pass around), it produces one
golden dict of the same schema shape by resolving each field independently with a
named, reusable **survivorship strategy** (a sensible default, per-field
overridable):

    canon = Canonicalizer(field_strategies={"phone": "most_frequent"})
    golden = canon.canonicalize(entity_records)          # a whole cluster
    golden = canon.enrich(golden, sparse_new_mention)    # the enrichment loop

The **enrichment loop** (the flagship W2.2 → W2.3 flow) is just
:meth:`canonicalize` over two records: when a sparse new mention links to an
existing entity via ``resolver.assign()``, folding the mention into the entity's
golden record fills any field the golden lacked but the mention has — *the same*
survivorship code path, not a parallel one.

The store round-trips through the config-registry artifact seam (no pickle):
:meth:`save` writes a small ``canonicalizer.json`` (version + config); :meth:`load`
rebuilds it from the strategy names in a fresh process.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from langres.core.registry import register

logger = logging.getLogger(__name__)

#: Bump when the ``canonicalizer.json`` layout changes incompatibly.
CANONICALIZER_VERSION = "1"

_MANIFEST_FILENAME = "canonicalizer.json"


def _is_missing(value: Any) -> bool:
    """True iff ``value`` carries no information for survivorship.

    ``None`` and a whitespace-only string are "missing" (a present value always
    beats an absent one). ``0`` / ``False`` / ``0.0`` are **present** — they are
    real values, not gaps.
    """
    return value is None or (isinstance(value, str) and value.strip() == "")


@dataclass(frozen=True)
class FieldContext:
    """The input one survivorship strategy resolves: a field across a group.

    Attributes:
        field: The field name being resolved.
        records: The entity's member records, in stable (caller) order. A
            strategy reads sibling fields off these (e.g. ``most_recent`` reads
            ``timestamp_field``, ``most_complete`` counts each record's non-missing
            fields), so a strategy is a function of the whole group, not just the
            field's values.
        id_field: The identity field, excluded from record-completeness scoring.
        timestamp_field: The field holding a comparable recency key, or ``None``.
            Only ``most_recent`` consults it.
    """

    field: str
    records: list[dict[str, Any]]
    id_field: str
    timestamp_field: str | None

    def present(self) -> list[tuple[int, Any]]:
        """``(index, value)`` for records whose :attr:`field` is non-missing, in order."""
        return [
            (i, record[self.field])
            for i, record in enumerate(self.records)
            if self.field in record and not _is_missing(record[self.field])
        ]


#: A survivorship strategy: pick the winning value for one field of one group.
SurvivorshipStrategy = Callable[[FieldContext], Any]


def _record_completeness(record: dict[str, Any], id_field: str) -> int:
    """Count a record's non-missing attribute fields (identity field excluded)."""
    return sum(1 for key, value in record.items() if key != id_field and not _is_missing(value))


def _first(ctx: FieldContext) -> Any:
    """First non-missing value in group order (source-priority when pre-sorted)."""
    present = ctx.present()
    return present[0][1] if present else None


def _longest(ctx: FieldContext) -> Any:
    """The non-missing value with the greatest string length; first-seen tiebreak."""
    present = ctx.present()
    if not present:
        return None
    # ``max`` keeps the first maximum on ties, and ``present`` is in group order,
    # so the tiebreak is deterministically first-seen.
    return max(present, key=lambda iv: len(str(iv[1])))[1]


def _most_frequent(ctx: FieldContext) -> Any:
    """The most common non-missing value (mode); first-seen tiebreak."""
    present = ctx.present()
    if not present:
        return None
    counts: dict[Any, int] = {}
    first_index: dict[Any, int] = {}
    for index, value in present:
        counts[value] = counts.get(value, 0) + 1
        first_index.setdefault(value, index)
    # Highest count wins; ties broken by earliest first appearance (stable).
    return max(counts, key=lambda value: (counts[value], -first_index[value]))


def _most_complete(ctx: FieldContext) -> Any:
    """Value from the overall-most-complete record; first-seen tiebreak.

    Among records where this field is non-missing, prefer the one carrying the
    most non-missing fields overall (trust the richest source), breaking ties by
    group order. A present value always beats an absent one, so this fills gaps.
    """
    present = ctx.present()
    if not present:
        return None
    best = max(
        present,
        key=lambda iv: (_record_completeness(ctx.records[iv[0]], ctx.id_field), -iv[0]),
    )
    return best[1]


def _most_recent(ctx: FieldContext) -> Any:
    """Value from the record with the greatest ``timestamp_field``; first-seen tiebreak.

    Considers only records whose timestamp *and* this field are non-missing. The
    timestamp is compared as-is (ISO-8601 strings sort chronologically; numeric
    epochs compare numerically), so keep one comparable type per column.
    """
    ts_field = ctx.timestamp_field
    if ts_field is None:  # pragma: no cover - guarded at construction time.
        raise ValueError("most_recent needs a timestamp_field")
    dated = [
        (index, value)
        for index, value in ctx.present()
        if not _is_missing(ctx.records[index].get(ts_field))
    ]
    if not dated:
        return None
    return max(dated, key=lambda iv: (ctx.records[iv[0]][ts_field], -iv[0]))[1]


#: The built-in survivorship strategies, by config name. ``source_priority`` is an
#: alias of ``first`` (the same "first non-missing wins" rule, named for the
#: source-ordered use). Add a strategy here to make it selectable by name in a
#: (serializable) Canonicalizer config.
_STRATEGIES: dict[str, SurvivorshipStrategy] = {
    "most_complete": _most_complete,
    "longest": _longest,
    "most_frequent": _most_frequent,
    "most_recent": _most_recent,
    "first": _first,
    "source_priority": _first,
}

#: Strategies that require a ``timestamp_field`` to be configured.
_TIMESTAMP_STRATEGIES = frozenset({"most_recent"})

#: The default when a field has no explicit override: non-missing wins, richest
#: source breaks ties — the safe "prefer complete data" rule for enrichment.
DEFAULT_STRATEGY = "most_complete"


class CanonicalizerManifest(BaseModel):
    """Typed shape of ``canonicalizer.json`` — the store's serialized config.

    Attributes:
        version: Layout version (see :data:`CANONICALIZER_VERSION`).
        type_name: The registry key, for a self-describing artifact.
        config: The :attr:`Canonicalizer.config` payload (strategies + fields).
    """

    version: str
    type_name: str
    config: dict[str, Any]


@register("canonicalizer")
class Canonicalizer:
    """Merge an entity's records into one golden record via survivorship rules.

    A thin, composable, config-serializable unit: it owns only the survivorship
    policy (a default strategy + per-field overrides) and applies it to a group
    of raw record dicts. It knows nothing about how the group was formed (a
    ``resolve`` cluster, an ``AnchorStore`` entity, a hand-picked list) — so it
    composes with any of them.

    Example:
        canon = Canonicalizer(field_strategies={"name": "longest"})
        golden = canon.canonicalize(entity_records)
        golden = canon.enrich(golden, new_mention)   # fold in a linked mention
        canon.save("artifacts/canon"); Canonicalizer.load("artifacts/canon")
    """

    type_name: ClassVar[str] = "canonicalizer"

    def __init__(
        self,
        *,
        default_strategy: str = DEFAULT_STRATEGY,
        field_strategies: dict[str, str] | None = None,
        id_field: str = "id",
        timestamp_field: str | None = None,
    ) -> None:
        """Configure the survivorship policy.

        Args:
            default_strategy: Strategy name for any field without an override
                (default ``"most_complete"``).
            field_strategies: Per-field ``field -> strategy name`` overrides.
            id_field: The identity field. It is not survivorship'd (identity is
                not merged); the golden record's id is the entity/master id
                (see :meth:`canonicalize`).
            timestamp_field: The recency field ``most_recent`` reads. Required if
                any configured strategy is ``most_recent``.

        Raises:
            ValueError: If a strategy name is unknown, or ``most_recent`` is used
                without a ``timestamp_field``.
        """
        self.default_strategy = default_strategy
        self.field_strategies = dict(field_strategies or {})
        self.id_field = id_field
        self.timestamp_field = timestamp_field

        for name in [default_strategy, *self.field_strategies.values()]:
            if name not in _STRATEGIES:
                available = ", ".join(sorted(_STRATEGIES))
                raise ValueError(f"Unknown survivorship strategy {name!r}. Available: {available}")
            if name in _TIMESTAMP_STRATEGIES and timestamp_field is None:
                raise ValueError(
                    f"Strategy {name!r} needs a `timestamp_field`; none was configured."
                )

    # ------------------------------------------------------------------
    # Canonicalization
    # ------------------------------------------------------------------

    def canonicalize(
        self, records: list[dict[str, Any]], *, entity_id: str | None = None
    ) -> dict[str, Any]:
        """Merge ``records`` (one entity's members) into a single golden record.

        Each attribute field is resolved independently by its configured strategy
        over the whole group; a field only *some* records carry is included (the
        others count as missing for it), which is exactly what makes a sparse
        record enrich a fuller one. The golden record's :attr:`id_field` is the
        stable master id: ``entity_id`` when given, else the first record's id
        (deterministic in group order). A single-record group yields a copy of
        that record.

        Args:
            records: The entity's member records (raw dicts), in stable order.
                Must be non-empty.
            entity_id: The stable master/entity id to stamp on the golden record.
                Defaults to the first record's id.

        Returns:
            One golden record dict: the id field plus every attribute field any
            input record carried, each resolved by survivorship (``None`` where no
            record had a value).

        Raises:
            ValueError: If ``records`` is empty.
        """
        if not records:
            raise ValueError("canonicalize requires at least one record.")

        # Field universe: every attribute key any record carried (id excluded —
        # it is stamped separately as the master id, not survivorship'd).
        fields = [
            key
            for key in dict.fromkeys(k for record in records for k in record)
            if key != self.id_field
        ]

        golden: dict[str, Any] = {}
        if entity_id is not None:
            golden[self.id_field] = entity_id
        elif self.id_field in records[0]:
            golden[self.id_field] = records[0][self.id_field]

        for field in fields:
            strategy = _STRATEGIES[self.field_strategies.get(field, self.default_strategy)]
            golden[field] = strategy(
                FieldContext(
                    field=field,
                    records=records,
                    id_field=self.id_field,
                    timestamp_field=self.timestamp_field,
                )
            )
        return golden

    def enrich(
        self, golden: dict[str, Any], mention: dict[str, Any], *, entity_id: str | None = None
    ) -> dict[str, Any]:
        """Fold a newly-linked ``mention`` into an existing ``golden`` record.

        This is the enrichment loop: a sparse new record that ``resolver.assign``
        linked to an entity is merged into that entity's golden record, filling
        any field the golden lacked but the mention has. It is exactly
        :meth:`canonicalize` over ``[golden, mention]`` — the *same* survivorship
        rules — with ``golden`` first so source-ordered strategies favour the
        established record. The golden record keeps its master id unless a new
        ``entity_id`` is given.

        Args:
            golden: The entity's current golden record.
            mention: The newly-linked record to fold in.
            entity_id: Master id to stamp; defaults to ``golden``'s current id.

        Returns:
            The updated golden record.
        """
        if entity_id is None:
            entity_id = golden.get(self.id_field)
        return self.canonicalize([golden, mention], entity_id=entity_id)

    # ------------------------------------------------------------------
    # Serialization (config-registry seam; no pickle)
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        """Serializable config: the strategy names and field bindings."""
        return {
            "default_strategy": self.default_strategy,
            "field_strategies": dict(self.field_strategies),
            "id_field": self.id_field,
            "timestamp_field": self.timestamp_field,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Canonicalizer":
        """Reconstruct a Canonicalizer from a :attr:`config` payload."""
        return cls(
            default_strategy=config.get("default_strategy", DEFAULT_STRATEGY),
            field_strategies=config.get("field_strategies"),
            id_field=config.get("id_field", "id"),
            timestamp_field=config.get("timestamp_field"),
        )

    def save(self, path: str | Path) -> None:
        """Persist the survivorship policy to ``path`` as a JSON artifact.

        Writes ``canonicalizer.json`` (version + type_name + config). No pickle,
        no code execution on load.

        Args:
            path: Directory to write the artifact into (created if absent).
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = CanonicalizerManifest(
            version=CANONICALIZER_VERSION, type_name=self.type_name, config=self.config
        )
        (out_dir / _MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2))
        logger.info("Saved Canonicalizer artifact to %s", out_dir)

    @classmethod
    def load(cls, path: str | Path) -> "Canonicalizer":
        """Reconstruct a Canonicalizer written by :meth:`save`.

        Args:
            path: Directory containing ``canonicalizer.json``.

        Returns:
            A Canonicalizer equivalent to the one that was saved.

        Raises:
            ValueError: If the artifact's ``version`` differs from the supported
                :data:`CANONICALIZER_VERSION` (an incompatible layout).
        """
        in_dir = Path(path)
        manifest = CanonicalizerManifest.model_validate_json(
            (in_dir / _MANIFEST_FILENAME).read_text()
        )
        if manifest.version != CANONICALIZER_VERSION:
            raise ValueError(
                f"Canonicalizer artifact version {manifest.version!r} differs from "
                f"supported {CANONICALIZER_VERSION!r}; cannot load."
            )
        return cls.from_config(manifest.config)
