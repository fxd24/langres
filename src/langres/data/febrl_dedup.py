"""FEBRL3 person **deduplication** benchmark adapter — the registry's first ``dedup`` task.

Loads the vendored FEBRL3 dataset: **one** table of 5000 synthetic person records
in which 2000 originals are joined by 3000 corrupted duplicates. This is the
sibling of :mod:`langres.data.febrl_person` and the deliberate contrast to it —
same generator, same ten attribute columns, opposite task shape:

===================  =================================  ==============================
                     ``febrl_person`` (FEBRL4)          ``febrl_dedup`` (FEBRL3)
===================  =================================  ==============================
task                 ``linkage``                        ``dedup``
sources              two (``person_a`` / ``person_b``)  one
true matches         all cross-source                   all **intra**-source
cluster sizes        exactly 2                          1–6 (835 singletons)
candidate filter     ``cross_source`` (drops the        none — an intra-source pair
                     intra-source pairs as noise)       is exactly what a match IS
===================  =================================  ==============================

That last row is why a dedup benchmark is not a linkage one relabelled. The
linkage adapters throw away every same-source candidate before scoring, which
hands the matcher a bipartite problem and hides a whole error class. Here the
blocker's own output is the candidate set, and a wrong merge inside one source
counts against you — which is the situation ``dedupe()``, langres's primary
shipped verb, is actually run in.

**Gold labels are entity membership, not a transitive closure.**
``gold_clusters.csv`` names each record's entity directly, so the partition is
read off rather than reconstructed by taking connected components of a pairwise
link file. That distinction is load-bearing in this repo: closure-derived gold
inherits the link file's errors and fuses unrelated records into giant components
(DBLP-Scholar's 37-record component), which inflates every cluster-based metric.
FEBRL3 carries no such artifact — the entity is the generator's own. The fixture
generation script asserts it (``datasets/febrl_dedup/SOURCE.md``): the
``recordlinkage`` pairwise ``links`` index is verified to be *exactly* the set of
within-entity pairs, 6538 = 6538.

The data is **fully synthetic** (ANU name/address frequency tables → fictitious
people, then realistic corruptions), so there is no PII, and it carries no
NonCommercial term. See ``datasets/febrl_dedup/SOURCE.md`` for licensing and
attribution (``recordlinkage`` BSD-3-Clause; upstream ANUOS 1.1).

:class:`FebrlDedupBenchmark` adapts this loader to the dataset-agnostic
:class:`~langres.data.benchmark.Benchmark` protocol (and
``langres.methods.BlockingBenchmark``), so it races through the same harness as
every linkage benchmark.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel, computed_field

from langres.data.benchmark import Benchmark, gold_pairs_from_clusters
from langres.core.blockers.all_pairs import register_schema_idempotent
from langres.data import _benchmark_utils as _bu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langres.core.blockers.vector import VectorBlocker

__all__ = [
    "ACHIEVED_PC_AT_DEFAULT_K",
    "DEDUP_RECALL_GATE",
    "DEDUP_THRESHOLD_GRID",
    "DEFAULT_DEDUP_BLOCKING_K",
    "GATE_MET",
    "PC_REGRESSION_FLOOR",
    "FebrlDedupBenchmark",
    "FebrlDedupSchema",
    "build_dedup_blocker",
    "load_febrl_dedup",
    "pick_blocking_k",
    "sweep_blocking_k",
]

_DATASET_PACKAGE = "langres.data.datasets.febrl_dedup"
_RECORDS_FILE = "records.csv"
_GOLD_FILE = "gold_clusters.csv"

#: The ten FEBRL person attribute columns (order matches the vendored CSV).
#: Deliberately re-declared rather than imported from :mod:`~langres.data.febrl_person`:
#: each dataset module owns its own schema in this package (``abt_buy`` and
#: ``amazon_google`` do the same with their near-identical product schemas), and
#: the column lists coinciding is a property of FEBRL, not a contract between the
#: two fixtures.
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

# Pinned blocking k for FEBRL3 dedup, measured with VectorBlocker over
# SentenceTransformer("all-MiniLM-L6-v2") cosine similarity on ``embed_text``
# (given_name + surname + suburb), over the whole 5000-record corpus with NO
# cross-source filter. Sweep (k -> Pair-Completeness, 6538 gold pairs), measured
# on macOS/MPS; reproduce via ``sweep_blocking_k`` (see the slow test):
#   k= 5 -> 0.8790
#   k=10 -> 0.9137
#   k=20 -> 0.9345
#   k=30 -> 0.9440
#   k=50 -> 0.9552   <- smallest swept k clearing the 0.95 gate
#   k=80 -> 0.9618
# Dedup needs a markedly larger k than the FEBRL4 linkage task (k=20 -> 0.978
# there) for a structural reason, not a tuning one: a gold cluster here holds up
# to 6 records, so one record can owe up to 5 gold pairs and must retrieve ALL of
# its co-referents, while every FEBRL4 cluster is a pair needing one hit. On top
# of that the neighbour list is no longer implicitly halved by a cross-source
# filter, so same-entity neighbours compete with same-source noise for the k
# slots. Recall is also flattening — 30 extra neighbours (50 -> 80) buy 0.7pp —
# so k=50 is the knee, not an arbitrary stop.
DEFAULT_DEDUP_BLOCKING_K = 50

#: Pair-Completeness gate the blocking k-sweep must clear (same 0.95 target the
#: linkage adapters use, so blocking quality is comparable across tasks).
DEDUP_RECALL_GATE = 0.95

#: Pair-Completeness at :data:`DEFAULT_DEDUP_BLOCKING_K`, measured on macOS/MPS.
#: It clears :data:`DEDUP_RECALL_GATE` by only 0.5pp, and the exact value shifts
#: by platform: cross-platform MiniLM float differences reorder borderline
#: neighbours, and :mod:`langres.data.febrl_person` measured that delta at over
#: 1.6pp (its k=5 reads 0.966 on macOS but below 0.95 on the Linux CI runner).
#: A margin that thin is smaller than the known platform noise, so the slow test
#: asserts a **regression floor** well under it rather than re-asserting the gate
#: — a gate that fails on a runner difference checks the runner, not the data.
ACHIEVED_PC_AT_DEFAULT_K = 0.9552

#: Whether :data:`ACHIEVED_PC_AT_DEFAULT_K` clears :data:`DEDUP_RECALL_GATE`
#: (mirrors ``febrl_person.GATE_MET``).
GATE_MET = ACHIEVED_PC_AT_DEFAULT_K >= DEDUP_RECALL_GATE

#: Pair-Completeness floor the slow blocking test asserts at
#: :data:`DEFAULT_DEDUP_BLOCKING_K`. Set ~2.5pp under the measured value — below
#: the gate on purpose (see :data:`ACHIEVED_PC_AT_DEFAULT_K`), but far enough
#: above the k=30 measurement (0.9440) that a real blocking regression, a broken
#: ``embed_text``, or a silently-reintroduced cross-source filter still trips it.
PC_REGRESSION_FLOOR = 0.93

#: Candidate Clusterer thresholds swept when racing methods on FEBRL3 dedup.
#: Mirrors the other adapters' grids: the zero-spend judges score in ``[0, 1]``,
#: and this small grid brackets the useful range (tuned on TRAIN only).
DEDUP_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class FebrlDedupSchema(BaseModel):
    """A single person record from the FEBRL3 deduplication benchmark.

    Note the absence of a ``source`` field: FEBRL3 is a **single-source** corpus,
    so there is no side to filter candidates by — which is precisely what makes it
    a dedup task. ``embed_text`` is the serializable blocking text used by the
    :class:`VectorBlocker` (referenced as ``text_field``). All attribute fields
    are optional: FEBRL's corruptions can blank any of them.

    Attributes:
        id: Opaque record id (``"r0000"``..``"r4999"``). Opaque on purpose — the
            upstream ``rec-<N>-org`` / ``rec-<N>-dup-<K>`` ids encode the entity
            number ``N``, so keeping them would leak the label into the record.
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: given name, surname and suburb joined by spaces.

        Used as the :class:`VectorBlocker` ``text_field`` and as the text fed to
        the vector index. Omits missing components so absent fields don't inject
        empty tokens. Identical composition to ``FebrlPersonSchema.embed_text``,
        so blocking quality is comparable between the linkage and dedup tasks.
        """
        return " ".join(p for p in [self.given_name, self.surname, self.suburb] if p)


# Register FebrlDedupSchema at import time so a fresh process that only imports
# this module (e.g. to ``Resolver.load`` a saved artifact and ``resolve``) finds
# the schema in the registry without first constructing a blocker (same pattern
# as ``FebrlPersonSchema`` / ``ProductSchema``).
register_schema_idempotent(FebrlDedupSchema)


def _read_csv_rows(filename: str) -> list[dict[str, str]]:
    """Read a packaged FEBRL dedup CSV into a list of header-keyed row dicts."""
    return _bu.read_csv_rows(_DATASET_PACKAGE, filename)


def load_febrl_dedup() -> tuple[list[FebrlDedupSchema], list[set[str]], set[frozenset[str]]]:
    """Load FEBRL3 as one corpus plus its gold partition and gold pairs.

    The corpus is a single 5000-record table (no source merge — that is the point
    of a dedup benchmark). ``gold_clusters`` comes straight from
    ``gold_clusters.csv``'s ``record_id,cluster_id`` membership, so it is the
    **complete closed-world partition** by construction: 2000 clusters covering
    every record, of which 835 are singletons (genuinely non-duplicated people)
    and the rest hold 2–6 co-referent records.

    It is **not** derived by transitive closure over a pairwise link file, so it
    carries none of that construction's fusion artifacts — see the module
    docstring and ``datasets/febrl_dedup/SOURCE.md``.

    ``gold_pairs`` is every within-cluster pair (6538 of them), derived from the
    partition rather than read from a second file, so the two can never disagree.

    Returns:
        ``(corpus, gold_clusters, gold_pairs)``.

    Raises:
        BenchmarkDataNotFoundError: If the packaged CSVs are missing.
    """
    corpus = [
        FebrlDedupSchema(
            id=row["id"].strip(),
            **{name: (row.get(name, "").strip() or None) for name in _PERSON_FIELDS},
        )
        for row in _read_csv_rows(_RECORDS_FILE)
    ]

    by_cluster: dict[str, set[str]] = defaultdict(set)
    for row in _read_csv_rows(_GOLD_FILE):
        by_cluster[row["cluster_id"].strip()].add(row["record_id"].strip())
    # Sorted by cluster id so the returned partition order is deterministic.
    gold_clusters = [by_cluster[cid] for cid in sorted(by_cluster)]
    gold_pairs = gold_pairs_from_clusters(gold_clusters)

    logger.info(
        "Loaded FEBRL dedup: %d records, %d gold clusters (%d singletons, "
        "largest %d), %d gold pairs",
        len(corpus),
        len(gold_clusters),
        sum(1 for c in gold_clusters if len(c) == 1),
        max((len(c) for c in gold_clusters), default=0),
        len(gold_pairs),
    )
    return corpus, gold_clusters, gold_pairs


def build_dedup_blocker(
    k_neighbors: int = DEFAULT_DEDUP_BLOCKING_K,
) -> VectorBlocker[FebrlDedupSchema]:
    """Build the shared FEBRL dedup VectorBlocker (MiniLM + FAISS-cosine).

    Declarative (``schema=`` + ``text_field=``) so the resulting blocker is
    config-serializable. Each call constructs a *fresh* (unbuilt)
    :class:`FAISSIndex`; embedding only happens when the index is later populated
    from a corpus. Mirrors ``build_person_blocker``.

    Args:
        k_neighbors: Nearest neighbours per record. Defaults to
            :data:`DEFAULT_DEDUP_BLOCKING_K` (clears Pair-Completeness >= 0.95).

    Returns:
        A :class:`VectorBlocker` over ``FebrlDedupSchema.embed_text``.
    """
    from langres.core.blockers.vector import VectorBlocker
    from langres.core.embeddings import SentenceTransformerEmbedder
    from langres.core.indexes.vector_index import FAISSIndex

    return VectorBlocker(
        vector_index=FAISSIndex(
            embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
            metric="cosine",
        ),
        schema=FebrlDedupSchema,
        text_field="embed_text",
        k_neighbors=k_neighbors,
    )


def sweep_blocking_k(
    corpus: list[FebrlDedupSchema],
    gold_clusters: list[set[str]],
    ks: tuple[int, ...] = (5, 10, 20, 30, 50),
) -> dict[int, float]:
    """Measure Pair-Completeness of vector blocking across ``ks``.

    Dedup-typed wrapper over :func:`langres.data._benchmark_utils.sweep_blocking_k`,
    binding ``FebrlDedupSchema`` + ``embed_text`` and — the difference that
    matters — passing ``cross_source_only=False``. There is only one source here,
    so the linkage adapters' cross-source filter would discard every candidate.

    Args:
        corpus: Record list from :func:`load_febrl_dedup`.
        gold_clusters: The complete partition from :func:`load_febrl_dedup`.
        ks: Neighbor counts to sweep.

    Returns:
        Mapping of ``k`` to Pair-Completeness (``candidate_recall``).
    """
    return _bu.sweep_blocking_k(
        corpus,
        gold_clusters,
        FebrlDedupSchema,
        text_field="embed_text",
        ks=ks,
        cross_source_only=False,
    )


def pick_blocking_k(recalls: dict[int, float], threshold: float = DEDUP_RECALL_GATE) -> int:
    """Pick the smallest ``k`` whose recall clears ``threshold`` (default the dedup gate).

    Dedup-typed wrapper over :func:`langres.data._benchmark_utils.pick_blocking_k`
    that defaults ``threshold`` to :data:`DEDUP_RECALL_GATE` (0.95). If no ``k``
    reaches it, returns the ``k`` with the highest recall (honest fallback).

    Args:
        recalls: Mapping of ``k`` to recall, e.g. from :func:`sweep_blocking_k`.
        threshold: Minimum acceptable recall (defaults to :data:`DEDUP_RECALL_GATE`).

    Returns:
        The chosen ``k``.

    Raises:
        ValueError: If ``recalls`` is empty.
    """
    return _bu.pick_blocking_k(recalls, threshold)


class FebrlDedupBenchmark(Benchmark[FebrlDedupSchema]):
    """FEBRL3 dedup as a :class:`~langres.data.benchmark.Benchmark` conformer.

    Adapts the dedup loader/splitter to the dataset-agnostic harness so
    :func:`~langres.benchmarks.runner.run_method` can run any resolver factory
    against it, mirroring ``FebrlPersonBenchmark`` exactly — the *only*
    differences are the single-source schema and that no cross-source filter is
    applied to candidates.

    It also conforms to ``langres.methods.BlockingBenchmark`` by exposing its
    record ``schema`` and pinned blocking config (``blocking_k`` +
    :meth:`build_blocker`), so the method registry can race *any* method against
    it on the identical candidate set.
    """

    name = "febrl_dedup"
    threshold_grid = DEDUP_THRESHOLD_GRID
    #: Record type, exposed for the method registry's Comparator/rapidfuzz fields.
    schema = FebrlDedupSchema
    #: Pinned blocking neighbour count (blocking held constant across methods).
    blocking_k = DEFAULT_DEDUP_BLOCKING_K

    def build_blocker(self, k_neighbors: int) -> VectorBlocker[FebrlDedupSchema]:
        """Return a fresh FEBRL dedup VectorBlocker (MiniLM + FAISS-cosine)."""
        return build_dedup_blocker(k_neighbors)

    def load(self) -> tuple[list[FebrlDedupSchema], list[set[str]], set[frozenset[str]]]:
        """Return ``(corpus, gold_clusters, gold_pairs)`` for FEBRL3 dedup."""
        return load_febrl_dedup()

    def split(
        self,
        corpus: list[FebrlDedupSchema],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[FebrlDedupSchema], list[FebrlDedupSchema], list[set[str]], list[set[str]]]:
        """Leakage-free stratified split via the shared ``stratified_corpus_split``.

        Stratifies by gold-cluster size — which here spans 1–6 rather than the
        linkage adapters' {1, 2}, so every size band is represented on both
        sides instead of the split being effectively a coin flip over pairs.
        """
        return _bu.stratified_corpus_split(corpus, gold_clusters, seed=seed)
