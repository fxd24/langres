"""Generic loader factory for DeepMatcher-style two-table ER benchmarks (Wave B).

Four vendored benchmarks — Amazon-Google, Abt-Buy (and the shape FEBRL persons
follow) — share a near-identical anatomy: two source tables (``tableA.csv`` /
``tableB.csv``), fixed literature pair splits (``train`` / ``valid`` / ``test``
with ``ltable_id,rtable_id,label`` rows), a source-prefixed combined corpus, and
a :class:`~langres.core.benchmark.Benchmark` conformer that *also* satisfies the
method-registry ``langres.methods.BlockingBenchmark`` contract. Copying that
~350-line anatomy per dataset is the ~1.4k-line duplication the seam audit warns
against.

:func:`make_deepmatcher_benchmark` owns that anatomy once. A new DeepMatcher-style
dataset becomes ~30–50 lines: a **dataset-namespaced** Pydantic schema (its class
name MUST be unique — :func:`~langres.core.blockers.all_pairs.register_schema_idempotent`
*raises* on a name clash, so never reuse ``ProductSchema`` etc.) plus one factory
call carrying the honest pinned constants. It reuses the six
:mod:`langres.data._benchmark_utils` helpers (CSV reader, cross-source filter,
k-sweep/picker, connected-components partition, stratified split) rather than
re-implementing them.

The genuinely-shared mechanics live here; the existing ``abt_buy`` /
``amazon_google`` / ``er_benchmarks`` / ``febrl_person`` modules are left as-is
(this factory is for the *new* datasets). The heavy ``[semantic]`` stack
(``VectorBlocker`` / ``SentenceTransformerEmbedder`` / ``FAISSIndex``) is imported
**lazily inside** :meth:`DeepMatcherBenchmark.build_blocker`, so a benchmark can be
loaded and split offline (e.g. with an ``AllPairsBlocker``) without pulling faiss.

Id-scheme safety: the shared stratified split parses ``int(id[1:])`` (see
:func:`langres.data._benchmark_utils.stratified_corpus_split`), so every corpus id
must be ``<single-char><int>``. The factory prefixes each table's raw id with a
distinct single-char prefix (so numeric ids reused across ``tableA``/``tableB``
stay globally unique) and, if a source's raw ids are *not* pure integers, remaps
them to synthetic ``<prefix><int>`` ids **before** gold-pair construction — so the
split never raises on a string id.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel

from langres.core.benchmark import gold_pairs_from_clusters
from langres.core.blockers.all_pairs import register_schema_idempotent
from langres.data import _benchmark_utils as _bu

if TYPE_CHECKING:
    from langres.core.blockers.vector import VectorBlocker

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

#: Default embedding model for the vector blocker (matches the existing loaders).
_DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

#: A raw source id is "already split-safe" iff it is a pure non-negative integer;
#: otherwise the factory remaps the table's ids to synthetic sequential integers.
_INT_ID = re.compile(r"^\d+$")

#: A final corpus id must be a single alpha char followed by an integer, so the
#: stratified split's ``id[0], int(id[1:])`` parse holds.
_SPLIT_SAFE_ID = re.compile(r"^[A-Za-z]\d+$")


@dataclass(frozen=True)
class SourceTable:
    """One source table's CSV plus how its records join the combined corpus.

    Attributes:
        file: CSV filename within the dataset package (e.g. ``"tableA.csv"``).
        source: Value assigned to each record's ``source`` field — must be one of
            the schema's ``source`` ``Literal`` options.
        id_prefix: Single alphabetic character prefixed onto the raw source id to
            make it globally unique in the combined corpus (e.g. ``"a"``). MUST be
            exactly one alpha char so the downstream stratified split's
            ``id[0], int(id[1:])`` parse holds; two tables reusing the same numeric
            ids stay distinct because their prefixes differ.
    """

    file: str
    source: str
    id_prefix: str


def _table_id_map(rows: list[dict[str, str]], id_column: str, prefix: str) -> dict[str, str]:
    """Map each table row's raw id to its globally-unique ``<prefix><int>`` id.

    When every raw id is already a pure integer (the DeepMatcher default, where
    ids are 0-based row indices), the id is simply ``f"{prefix}{raw}"``. When any
    raw id is *not* an integer (e.g. WDC's string ids), the whole table is remapped
    to synthetic sequential integers in row order, so the final ids stay
    ``<prefix><int>`` and the stratified split's ``int(id[1:])`` never raises.

    Args:
        rows: The table's header-keyed rows, in file order.
        id_column: Column holding the raw source id.
        prefix: The table's single-char id prefix.

    Returns:
        Mapping of raw source id -> final corpus id.

    Raises:
        ValueError: If a raw id repeats within the table — the returned dict is
            keyed by raw id, so duplicates would silently collapse two distinct
            records onto one final id (and drop rows from the corpus).
    """
    raw_ids = [row[id_column].strip() for row in rows]
    if len(set(raw_ids)) != len(raw_ids):
        dups = sorted({rid for rid in raw_ids if raw_ids.count(rid) > 1})
        raise ValueError(
            f"Table with id_prefix {prefix!r} has duplicate raw ids in column "
            f"{id_column!r}: {dups}. Raw ids must be unique within a table (they "
            "key the id map); dedupe the source or pick a different id column."
        )
    if all(_INT_ID.match(rid) for rid in raw_ids):
        return {rid: f"{prefix}{rid}" for rid in raw_ids}
    # Non-integer source ids: remap deterministically to synthetic ints (row order
    # fixes the mapping) so gold-pair construction and the split both stay safe.
    logger.info(
        "Remapping %d non-integer source ids under prefix %r to synthetic <prefix><int> ids",
        len(raw_ids),
        prefix,
    )
    return {rid: f"{prefix}{i}" for i, rid in enumerate(raw_ids)}


def _record_from_row(
    schema: type[SchemaT], row: dict[str, str], source: str, final_id: str
) -> SchemaT:
    """Build a schema record from a raw CSV row, driven by the schema's fields.

    Reads each declared data field (everything except ``id``/``source`` and the
    computed ``embed_text``, which is not in ``model_fields``): an empty cell maps
    to the field's default (``None`` for optionals), while a *required* field (no
    default, e.g. the always-present title) falls back to ``""`` — mirroring the
    hand-written ``abt_buy`` / ``amazon_google`` builders. Extra CSV columns not on
    the schema are ignored.
    """
    data: dict[str, Any] = {"id": final_id, "source": source}
    for field_name, info in schema.model_fields.items():
        if field_name in ("id", "source"):
            continue
        cleaned = row.get(field_name, "").strip()
        if cleaned:
            data[field_name] = cleaned
        elif info.is_required():
            data[field_name] = ""
        # else: leave absent so the field takes its default (typically None).
    return schema(**data)


def _assert_split_safe_ids(corpus: list[SchemaT], name: str) -> None:
    """Assert every corpus id is ``<char><int>`` (the split's parse contract).

    A guard against a misconfigured prefix or an un-remapped source: with a
    correctly-built id map this always holds, but asserting it here fails loudly
    at load with an actionable message instead of deep inside the stratified split.
    """
    for record in corpus:
        rid = record.id  # type: ignore[attr-defined]
        if not _SPLIT_SAFE_ID.match(rid):
            raise ValueError(
                f"{name}: record id {rid!r} is not <char><int>; the stratified "
                "split parses int(id[1:]) and would raise. Ensure each SourceTable "
                "id_prefix is a single alpha char (the factory remaps non-integer "
                "source ids automatically)."
            )


def _default_class_name(name: str) -> str:
    """Derive a ``<Name>Benchmark`` class name from a snake_case dataset name."""
    camel = "".join(part.capitalize() for part in name.split("_"))
    return f"{camel}Benchmark"


class DeepMatcherBenchmark(Generic[SchemaT]):
    """A DeepMatcher-style benchmark conforming to Benchmark + BlockingBenchmark.

    Produced by :func:`make_deepmatcher_benchmark`, which bakes a dataset's config
    into a zero-arg subclass (so the registry can instantiate it with ``()``).
    Conforms **structurally** to the harness
    :class:`~langres.core.benchmark.Benchmark`
    (``name`` / ``threshold_grid`` / :meth:`load` / :meth:`split`) *and* to
    ``langres.methods.BlockingBenchmark``
    (``schema`` / ``blocking_k`` / :meth:`build_blocker`), so the same instance
    drives both :func:`~langres.core.benchmark.run_method` and the method registry.
    It does not *inherit* ``Benchmark`` because that protocol's record type must
    expose ``id`` — a constraint a schema-generic base cannot state. ``Benchmark``
    conformance is therefore structural, and — since ``Benchmark`` is
    ``@runtime_checkable`` — additionally confirmable via ``isinstance`` (as the
    loader contract test does). ``BlockingBenchmark`` conformance is purely
    structural / duck-typed: ``langres.methods.BlockingBenchmark`` is a plain
    ``Protocol`` that is never ``isinstance``-checked, exactly like the
    hand-written conformers.

    Attributes:
        name: Stable dataset name (e.g. ``"wdc_products"``).
        threshold_grid: Clusterer thresholds swept when racing methods.
        schema: The dataset's Pydantic record type.
        blocking_k: Pinned nearest-neighbour count (blocking held constant).
        achieved_pc: Cross-source Pair-Completeness measured at ``blocking_k``
            (honest, pinned from a real sweep).
        gate_met: Whether ``achieved_pc`` clears the dataset's blocking recall gate.
    """

    name: str
    threshold_grid: tuple[float, ...]
    schema: type[SchemaT]
    blocking_k: int
    achieved_pc: float
    gate_met: bool

    def __init__(
        self,
        *,
        name: str,
        threshold_grid: tuple[float, ...],
        schema: type[SchemaT],
        blocking_k: int,
        achieved_pc: float,
        gate_met: bool,
        load_fn: Callable[[], tuple[list[SchemaT], list[set[str]], set[frozenset[str]]]],
        split_fn: Callable[
            ...,
            tuple[list[SchemaT], list[SchemaT], list[set[str]], list[set[str]]],
        ],
        build_blocker_fn: Callable[[int], VectorBlocker[SchemaT]],
    ) -> None:
        """Store the baked config + closures the factory built."""
        self.name = name
        self.threshold_grid = threshold_grid
        self.schema = schema
        self.blocking_k = blocking_k
        self.achieved_pc = achieved_pc
        self.gate_met = gate_met
        self._load_fn = load_fn
        self._split_fn = split_fn
        self._build_blocker_fn = build_blocker_fn

    def load(self) -> tuple[list[SchemaT], list[set[str]], set[frozenset[str]]]:
        """Return ``(corpus, gold_clusters, gold_pairs)`` for the full dataset.

        ``gold_pairs`` is re-derived from the closed-world partition (every
        within-cluster pair, including the transitive closure of many-to-many
        matches), matching the existing dataset conformers.
        """
        corpus, gold_clusters, _gold_pairs = self._load_fn()
        return corpus, gold_clusters, gold_pairs_from_clusters(gold_clusters)

    def split(
        self,
        corpus: list[SchemaT],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[SchemaT], list[SchemaT], list[set[str]], list[set[str]]]:
        """Leakage-free stratified split via the shared ``stratified_corpus_split``."""
        return self._split_fn(corpus, gold_clusters, seed=seed)

    def build_blocker(self, k_neighbors: int) -> VectorBlocker[SchemaT]:
        """Return a fresh, unbuilt VectorBlocker (MiniLM + FAISS-cosine)."""
        return self._build_blocker_fn(k_neighbors)


def make_deepmatcher_benchmark(
    *,
    name: str,
    schema: type[SchemaT],
    dataset_package: str,
    table_a: SourceTable,
    table_b: SourceTable,
    split_files: dict[str, str],
    blocking_k: int,
    threshold_grid: tuple[float, ...],
    achieved_pc: float,
    gate_met: bool,
    text_field: str = "embed_text",
    id_column: str = "id",
    left_id_column: str = "ltable_id",
    right_id_column: str = "rtable_id",
    label_column: str = "label",
    embed_model: str = _DEFAULT_EMBED_MODEL,
    benchmark_class_name: str | None = None,
) -> tuple[
    Callable[[], tuple[list[SchemaT], list[set[str]], set[frozenset[str]]]],
    Callable[[], dict[str, list[tuple[str, str, int]]]],
    type[DeepMatcherBenchmark[SchemaT]],
]:
    """Build the ``(load, load_pair_splits, <Name>Benchmark)`` triple for a dataset.

    Generalizes the ``abt_buy`` / ``amazon_google`` anatomy: two source tables, a
    source-prefixed combined corpus, fixed literature pair splits, and a benchmark
    conforming to both the harness and registry contracts. The schema is registered
    at call time (import time) so a fresh process can ``Resolver.load`` an artifact
    referencing it.

    Args:
        name: Stable dataset name (also the benchmark's ``name``).
        schema: The dataset's Pydantic record type. Its class name MUST be unique
            across all datasets (registered via ``register_schema_idempotent``,
            which raises on a clash) — always dataset-namespace it (e.g.
            ``WdcProductSchema``, never a shared ``ProductSchema``). Must declare an
            ``id`` field, a ``source`` field, and a computed ``text_field``.
        dataset_package: Importable package holding the CSVs (e.g.
            ``"langres.data.datasets.wdc_products"``).
        table_a: Left source table (its ids become the split ``ltable_id``s).
        table_b: Right source table (its ids become the split ``rtable_id``s).
        split_files: Split name -> CSV filename (e.g.
            ``{"train": "train.csv", "valid": "valid.csv", "test": "test.csv"}``).
        blocking_k: Pinned nearest-neighbour count.
        threshold_grid: Clusterer thresholds swept when racing methods.
        achieved_pc: Cross-source Pair-Completeness measured at ``blocking_k``.
        gate_met: Whether ``achieved_pc`` clears the dataset's blocking recall gate.
        text_field: Attribute holding each record's blocking text.
        id_column: Column in the table CSVs holding the raw source id.
        left_id_column: Column in the split CSVs referencing ``table_a``.
        right_id_column: Column in the split CSVs referencing ``table_b``.
        label_column: Column in the split CSVs holding the ``0``/``1`` label.
        embed_model: SentenceTransformer model id for the vector blocker.
        benchmark_class_name: Optional explicit ``<Name>Benchmark`` class name;
            defaults to a CamelCase of ``name``.

    Returns:
        ``(load, load_pair_splits, benchmark_class)`` — ``load`` returns
        ``(corpus, gold_clusters, gold_pairs)``; ``load_pair_splits`` returns the
        fixed splits as ``{split: [(id_a, id_b, label), ...]}`` with corpus-prefixed
        ids; ``benchmark_class`` is a zero-arg-constructible
        :class:`DeepMatcherBenchmark` subclass.

    Raises:
        ValueError: If either table's ``id_prefix`` is not a single alpha char,
            or if the two tables share the same ``id_prefix``.
    """
    for table in (table_a, table_b):
        if len(table.id_prefix) != 1 or not table.id_prefix.isalpha():
            raise ValueError(
                f"{name}: SourceTable id_prefix must be a single alphabetic char; "
                f"got {table.id_prefix!r} for {table.file!r}."
            )
    if table_a.id_prefix == table_b.id_prefix:
        raise ValueError(
            f"{name}: table_a and table_b must use DISTINCT id_prefixes; both are "
            f"{table_a.id_prefix!r} (equal prefixes + overlapping raw ids collide)."
        )

    # Register the schema now (import time), like the hand-written loaders, so a
    # saved artifact's ``schema_type_name`` round-trips without building a blocker.
    register_schema_idempotent(schema)

    def load_pair_splits() -> dict[str, list[tuple[str, str, int]]]:
        rows_a = _bu.read_csv_rows(dataset_package, table_a.file)
        rows_b = _bu.read_csv_rows(dataset_package, table_b.file)
        map_a = _table_id_map(rows_a, id_column, table_a.id_prefix)
        map_b = _table_id_map(rows_b, id_column, table_b.id_prefix)
        splits: dict[str, list[tuple[str, str, int]]] = {}
        for split_name, filename in split_files.items():
            splits[split_name] = [
                (
                    map_a[row[left_id_column].strip()],
                    map_b[row[right_id_column].strip()],
                    int(row[label_column]),
                )
                for row in _bu.read_csv_rows(dataset_package, filename)
            ]
        return splits

    def load_corpus() -> tuple[list[SchemaT], list[set[str]], set[frozenset[str]]]:
        corpus: list[SchemaT] = []
        for table in (table_a, table_b):
            rows = _bu.read_csv_rows(dataset_package, table.file)
            id_map = _table_id_map(rows, id_column, table.id_prefix)
            corpus.extend(
                _record_from_row(schema, row, table.source, id_map[row[id_column].strip()])
                for row in rows
            )
        _assert_split_safe_ids(corpus, name)

        splits = load_pair_splits()
        gold_pairs: set[frozenset[str]] = {
            frozenset({left, right})
            for split in splits.values()
            for left, right, label in split
            if label == 1
        }
        gold_clusters = _bu.clusters_from_pairs(
            gold_pairs,
            (r.id for r in corpus),  # type: ignore[attr-defined]
        )
        match_clusters = [c for c in gold_clusters if len(c) >= 2]
        logger.info(
            "Loaded %s: %d records, %d gold pairs, %d clusters (%d match + %d singletons)",
            name,
            len(corpus),
            len(gold_pairs),
            len(gold_clusters),
            len(match_clusters),
            len(gold_clusters) - len(match_clusters),
        )
        return corpus, gold_clusters, gold_pairs

    def split_corpus(
        corpus: list[SchemaT],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[SchemaT], list[SchemaT], list[set[str]], list[set[str]]]:
        # ``stratified_corpus_split`` needs ``RecordT: _HasId`` (an ``id: str``);
        # every benchmark schema exposes ``id`` (enforced by ``_assert_split_safe_ids``
        # + the id-scheme contract), but a ``BaseModel``-bound TypeVar can't state
        # that — the same limitation ``fixed_split_pair_benchmark`` handles with
        # targeted ignores.
        return _bu.stratified_corpus_split(  # type: ignore[type-var]
            corpus, gold_clusters, seed=seed
        )

    def build_blocker(k_neighbors: int) -> VectorBlocker[SchemaT]:
        # Lazy [semantic] import: loading + splitting stay faiss-free (e.g. an
        # offline AllPairsBlocker path), and only building the real blocker pulls
        # faiss/sentence-transformers.
        from langres.core.blockers.vector import VectorBlocker
        from langres.core.embeddings import SentenceTransformerEmbedder
        from langres.core.indexes.vector_index import FAISSIndex

        return VectorBlocker(
            vector_index=FAISSIndex(
                embedder=SentenceTransformerEmbedder(embed_model),
                metric="cosine",
            ),
            schema=schema,
            text_field=text_field,
            k_neighbors=k_neighbors,
        )

    class _Benchmark(DeepMatcherBenchmark[SchemaT]):
        """Zero-arg-constructible benchmark with this dataset's config baked in."""

        def __init__(self) -> None:
            super().__init__(
                name=name,
                threshold_grid=threshold_grid,
                schema=schema,
                blocking_k=blocking_k,
                achieved_pc=achieved_pc,
                gate_met=gate_met,
                load_fn=load_corpus,
                split_fn=split_corpus,
                build_blocker_fn=build_blocker,
            )

    class_name = benchmark_class_name or _default_class_name(name)
    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name

    return load_corpus, load_pair_splits, _Benchmark
