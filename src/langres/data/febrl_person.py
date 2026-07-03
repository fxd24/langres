"""FEBRL4 person entity-resolution benchmark adapter (M5 W2.1 — second entity type).

Loads the vendored FEBRL4 **person** subset into a single corpus of
:class:`PersonSchema` records plus cross-source ground-truth pairs. This is the
M5 "generality" exit: a *second* entity type (a person, not a product or
restaurant) resolved with **zero new core code** — only this config-only adapter,
mirroring the Fodors-Zagat and Amazon-Google/Abt-Buy loaders.

Like Fodors-Zagat, this is a clean **1:1 linkage** task: two sources each list
people. ``dataset A`` holds originals (``rec-N-org``) and ``dataset B`` holds one
corrupted duplicate each (``rec-N-dup-0``); the matches we care about are the
cross-source ``org``↔``dup`` pairs, given explicitly by a ``perfectMapping`` file
(same shape as Fodors-Zagat's, so the closed-world gold construction is identical:
2-element match sets + singletons for the unmatched remainder).

The data is **fully synthetic** (ANU name/address frequency tables → fictitious
people, then realistic corruptions), so there is no PII. See
``datasets/febrl_person/SOURCE.md`` for licensing/attribution (recordlinkage
BSD-3-Clause; upstream ANUOS 1.1 — no NonCommercial term).

This module deliberately mirrors :mod:`langres.data.abt_buy`'s patterns (a schema
with a ``@computed_field embed_text``, idempotent schema registration at import, a
combined source-prefixed corpus, a vector blocking k-sweep). The genuinely-shared
mechanics (CSV reading, the cross-source filter, the k-sweep/picker, and the
stratified split) live once in :mod:`langres.data._benchmark_utils`; this module
keeps the person-specific schema, pinned constants, and public wrappers.

:class:`FebrlPersonBenchmark` adapts this loader to the dataset-agnostic
:class:`~langres.core.benchmark.Benchmark` protocol (and
``langres.methods.BlockingBenchmark``) so it races through the same harness as
Fodors-Zagat, Amazon-Google, and Abt-Buy.
"""

import logging
from typing import Literal

from pydantic import BaseModel, computed_field

from langres.core.benchmark import Benchmark, gold_pairs_from_clusters
from langres.core.blockers.all_pairs import register_schema_idempotent
from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.models import ERCandidate
from langres.data import _benchmark_utils as _bu

logger = logging.getLogger(__name__)

__all__ = [
    "ACHIEVED_PC_AT_DEFAULT_K",
    "DEFAULT_PERSON_BLOCKING_K",
    "GATE_MET",
    "PERSON_RECALL_GATE",
    "PERSON_THRESHOLD_GRID",
    "FebrlPersonBenchmark",
    "PersonSchema",
    "build_person_blocker",
    "load_febrl_person",
    "pick_blocking_k",
    "sweep_blocking_k",
]

_DATASET_PACKAGE = "langres.data.datasets.febrl_person"
_PERSON_A_FILE = "person_a.csv"  # originals (rec-N-org)
_PERSON_B_FILE = "person_b.csv"  # duplicates (rec-N-dup-0)
_MAPPING_FILE = "person_perfectMapping.csv"  # id_a,id_b true 1:1 links

PersonSource = Literal["a", "b"]

#: The ten FEBRL person attribute columns (order matches the vendored CSVs).
_PERSON_FIELDS = (
    "given_name",
    "surname",
    "street_number",
    "address_1",
    "address_2",
    "suburb",
    "postcode",
    "state",
    "date_of_birth",
    "soc_sec_id",
)

# Pinned blocking k for the cross-source FEBRL person matches, measured with
# VectorBlocker over SentenceTransformer("all-MiniLM-L6-v2") cosine similarity on
# ``embed_text`` (given_name + surname + suburb). Sweep (k -> Pair-Completeness,
# 500 gold pairs), reproduce via ``sweep_blocking_k`` (see the slow test):
#   k= 5 -> 0.9660
#   k=10 -> 0.9680
#   k=20 -> 0.9780
#   k=30 -> 0.9820
#   k=50 -> 0.9860
# FEBRL persons block cleanly: given_name+surname+suburb is distinctive, so recall
# already clears the 0.95 gate at k=5 (like Fodors-Zagat, which saturates by k=5),
# and only inches up with larger k. DEFAULT_PERSON_BLOCKING_K is the MIN k clearing
# the 0.95 gate (see ``pick_blocking_k``).
DEFAULT_PERSON_BLOCKING_K = 5

#: Pair-Completeness gate the blocking k-sweep must clear (matches the clean
#: Fodors-Zagat 1:1 target of 0.95; FEBRL persons block comparably cleanly).
PERSON_RECALL_GATE = 0.95

#: Cross-source Pair-Completeness achieved at :data:`DEFAULT_PERSON_BLOCKING_K`,
#: recorded from the measured sweep above.
ACHIEVED_PC_AT_DEFAULT_K = 0.9660

#: Whether :data:`ACHIEVED_PC_AT_DEFAULT_K` clears :data:`PERSON_RECALL_GATE`.
GATE_MET = ACHIEVED_PC_AT_DEFAULT_K >= PERSON_RECALL_GATE

#: Candidate Clusterer thresholds swept when racing methods on FEBRL persons.
#: Mirrors the other adapters' grids: the zero-spend judges score in ``[0, 1]``,
#: and this small grid brackets the useful range (tuned on TRAIN only).
PERSON_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class PersonSchema(BaseModel):
    """A single person record from the FEBRL4 benchmark.

    ``id`` is globally unique across both sources (the loader prefixes the raw
    ``rec_id`` with ``a``/``b``). ``embed_text`` is the serializable blocking text
    used by the :class:`VectorBlocker` (referenced as ``text_field``). All
    attribute fields are optional: FEBRL's corruptions can blank any field.

    Attributes:
        id: Globally-unique record id (e.g. ``"a0"`` / ``"b0"``): the FEBRL
            record number prefixed with its source char.
        given_name: First name, if present.
        surname: Family name, if present.
        street_number: Street number as a raw string, if present.
        address_1: First address line, if present.
        address_2: Second address line, if present.
        suburb: Suburb / locality, if present.
        postcode: Postcode as a raw string, if present.
        state: State code, if present.
        date_of_birth: Date of birth as a raw ``YYYYMMDD`` string, if present.
        soc_sec_id: Synthetic social-security id as a raw string, if present.
        source: Originating table (``"a"`` for ``person_a``, ``"b"`` for
            ``person_b``).
    """

    id: str
    given_name: str | None = None
    surname: str | None = None
    street_number: str | None = None
    address_1: str | None = None
    address_2: str | None = None
    suburb: str | None = None
    postcode: str | None = None
    state: str | None = None
    date_of_birth: str | None = None
    soc_sec_id: str | None = None
    source: PersonSource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: given name, surname and suburb joined by spaces.

        Used as the :class:`VectorBlocker` ``text_field`` and as the text fed to
        the vector index. Omits missing components so absent fields don't inject
        empty tokens (mirrors ``RestaurantSchema.embed_text``).
        """
        return " ".join(p for p in [self.given_name, self.surname, self.suburb] if p)


# Register PersonSchema at import time so a fresh process that only imports this
# module (e.g. to ``Resolver.load`` a saved artifact and ``resolve``) finds the
# schema in the registry without first constructing a blocker. A declarative
# VectorBlocker re-registers idempotently, so this is the single source of truth
# for the registry key (same pattern as ``RestaurantSchema`` / ``ProductSchema``).
register_schema_idempotent(PersonSchema)


def _read_csv_rows(filename: str) -> list[dict[str, str]]:
    """Read a packaged FEBRL person CSV into a list of header-keyed row dicts."""
    return _bu.read_csv_rows(_DATASET_PACKAGE, filename)


def _record_from_row(row: dict[str, str], source: PersonSource, prefix: str) -> PersonSchema:
    """Build a :class:`PersonSchema` from a raw CSV row dict.

    The FEBRL CSVs are plainly quoted; empty cells (FEBRL blanks fields as a
    corruption) map to ``None`` for every attribute field. ``id`` is the raw
    ``rec_id`` prefixed with its source char.
    """

    def field(name: str) -> str | None:
        cleaned = row.get(name, "").strip()
        return cleaned or None

    return PersonSchema(
        id=f"{prefix}{row['rec_id'].strip()}",
        source=source,
        **{name: field(name) for name in _PERSON_FIELDS},
    )


def load_febrl_person() -> tuple[list[PersonSchema], list[set[str]], set[frozenset[str]]]:
    """Load the FEBRL person subset as one corpus plus its partition and gold pairs.

    Both sources are combined into a single corpus with globally-unique,
    source-prefixed ids (``a<rec_id>`` for ``person_a`` originals, ``b<rec_id>``
    for ``person_b`` duplicates). Ground truth comes from the explicit
    ``perfectMapping`` file (1:1 ``org``↔``dup`` links): each row yields a
    2-element match set ``{a<id_a>, b<id_b>}``. The mapping is diagonal
    (``id_a == id_b``) because FEBRL4 pairs original record ``N`` with its single
    duplicate ``N``; the ``a``/``b`` source prefix (not the number) distinguishes
    the two sides, so the ids stay globally unique.

    ``gold_clusters`` is the **complete closed-world partition**: the 2-element
    match sets PLUS a singleton ``{id}`` for every record not in any match (mirrors
    :func:`~langres.data.er_benchmarks.load_fodors_zagat`). ``gold_pairs`` is the
    set of those cross-source match pairs as frozensets.

    Returns:
        ``(corpus, gold_clusters, gold_pairs)``.
    """
    corpus: list[PersonSchema] = [
        _record_from_row(row, "a", "a") for row in _read_csv_rows(_PERSON_A_FILE)
    ]
    corpus += [_record_from_row(row, "b", "b") for row in _read_csv_rows(_PERSON_B_FILE)]

    gold_pairs: set[frozenset[str]] = {
        frozenset({f"a{row['id_a'].strip()}", f"b{row['id_b'].strip()}"})
        for row in _read_csv_rows(_MAPPING_FILE)
    }
    match_clusters: list[set[str]] = [set(pair) for pair in gold_pairs]
    matched_ids = {rid for cluster in match_clusters for rid in cluster}
    # Closed-world completion: every unmatched record is its own singleton, so
    # downstream scoring treats it as a known non-match rather than "no truth".
    singletons: list[set[str]] = [{r.id} for r in corpus if r.id not in matched_ids]
    gold_clusters = match_clusters + singletons

    logger.info(
        "Loaded FEBRL person: %d records (%d a + %d b), %d gold pairs, "
        "%d clusters (%d match + %d singletons)",
        len(corpus),
        sum(1 for r in corpus if r.source == "a"),
        sum(1 for r in corpus if r.source == "b"),
        len(gold_pairs),
        len(gold_clusters),
        len(match_clusters),
        len(singletons),
    )
    return corpus, gold_clusters, gold_pairs


def build_person_blocker(
    k_neighbors: int = DEFAULT_PERSON_BLOCKING_K,
) -> VectorBlocker[PersonSchema]:
    """Build the shared FEBRL person VectorBlocker (MiniLM + FAISS-cosine).

    Declarative (``schema=`` + ``text_field=``) so the resulting blocker is
    config-serializable. Each call constructs a *fresh* (unbuilt)
    :class:`FAISSIndex`; embedding only happens when the index is later populated
    from a corpus. Mirrors ``build_restaurant_blocker`` / ``build_abt_buy_blocker``.

    Args:
        k_neighbors: Nearest neighbours per record. Defaults to
            :data:`DEFAULT_PERSON_BLOCKING_K` (clears Pair-Completeness >= 0.95).

    Returns:
        A :class:`VectorBlocker` over ``PersonSchema.embed_text``.
    """
    return VectorBlocker(
        vector_index=FAISSIndex(
            embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
            metric="cosine",
        ),
        schema=PersonSchema,
        text_field="embed_text",
        k_neighbors=k_neighbors,
    )


def _cross_source(candidates: list[ERCandidate[PersonSchema]]) -> list[ERCandidate[PersonSchema]]:
    """Keep only candidate pairs whose two records come from different sources."""
    return _bu.cross_source(candidates)


def sweep_blocking_k(
    corpus: list[PersonSchema],
    gold_clusters: list[set[str]],
    ks: tuple[int, ...] = (5, 10, 20, 30, 50),
) -> dict[int, float]:
    """Measure cross-source Pair-Completeness of vector blocking across ``ks``.

    Person-typed wrapper over
    :func:`langres.data._benchmark_utils.sweep_blocking_k`, binding
    ``PersonSchema`` + ``embed_text``: builds the FAISS index once and reuses it
    across all ``k``, filtering to cross-source pairs before measuring recall.

    Args:
        corpus: Combined record list from :func:`load_febrl_person`.
        gold_clusters: The complete partition from :func:`load_febrl_person`.
        ks: Neighbor counts to sweep.

    Returns:
        Mapping of ``k`` to cross-source Pair-Completeness (``candidate_recall``).
    """
    return _bu.sweep_blocking_k(corpus, gold_clusters, PersonSchema, text_field="embed_text", ks=ks)


def pick_blocking_k(recalls: dict[int, float], threshold: float = PERSON_RECALL_GATE) -> int:
    """Pick the smallest ``k`` whose recall clears ``threshold`` (default the person gate).

    Person-typed wrapper over
    :func:`langres.data._benchmark_utils.pick_blocking_k` that defaults
    ``threshold`` to :data:`PERSON_RECALL_GATE` (0.95). If no ``k`` reaches it,
    returns the ``k`` with the highest recall (honest fallback).

    Args:
        recalls: Mapping of ``k`` to recall, e.g. from :func:`sweep_blocking_k`.
        threshold: Minimum acceptable recall (defaults to :data:`PERSON_RECALL_GATE`).

    Returns:
        The chosen ``k``.

    Raises:
        ValueError: If ``recalls`` is empty.
    """
    return _bu.pick_blocking_k(recalls, threshold)


class FebrlPersonBenchmark(Benchmark[PersonSchema]):
    """FEBRL persons as a :class:`~langres.core.benchmark.Benchmark` conformer.

    Adapts the person loader/splitter to the dataset-agnostic harness so
    :func:`~langres.core.benchmark.run_method` can run any resolver factory
    against it, mirroring ``FodorsZagatBenchmark`` / ``AbtBuyBenchmark`` exactly,
    swapping the restaurant/product schema/loaders for the person ones.

    It also conforms to ``langres.methods.BlockingBenchmark`` by exposing its
    record ``schema`` and pinned blocking config (``blocking_k`` +
    :meth:`build_blocker`), so the method registry can race *any* method against
    it on the identical candidate set.
    """

    name = "febrl_person"
    threshold_grid = PERSON_THRESHOLD_GRID
    #: Record type, exposed for the method registry's Comparator/rapidfuzz fields.
    schema = PersonSchema
    #: Pinned blocking neighbour count (blocking held constant across methods).
    blocking_k = DEFAULT_PERSON_BLOCKING_K

    def build_blocker(self, k_neighbors: int) -> VectorBlocker[PersonSchema]:
        """Return a fresh FEBRL person VectorBlocker (MiniLM + FAISS-cosine)."""
        return build_person_blocker(k_neighbors)

    def load(self) -> tuple[list[PersonSchema], list[set[str]], set[frozenset[str]]]:
        """Return ``(corpus, gold_clusters, gold_pairs)`` for FEBRL persons."""
        corpus, gold_clusters, _gold_pairs = load_febrl_person()
        return corpus, gold_clusters, gold_pairs_from_clusters(gold_clusters)

    def split(
        self,
        corpus: list[PersonSchema],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[PersonSchema], list[PersonSchema], list[set[str]], list[set[str]]]:
        """Leakage-free stratified split via the shared ``stratified_corpus_split``."""
        return _bu.stratified_corpus_split(corpus, gold_clusters, seed=seed)
