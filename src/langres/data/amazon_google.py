"""Amazon-Google product entity-resolution benchmark adapter + blocking k-sweep.

Loads the vendored Amazon-Google benchmark into a single corpus of
:class:`ProductSchema` records plus cross-source ground-truth pairs, and provides
a blocking k-sweep that pins the best blocking ``k`` against a Pair-Completeness
>= 0.90 target. This benchmark is hard enough that vector blocking on
title+manufacturer does **not** reach that target within k<=50 (it plateaus
around 0.84); the sweep pins the honest best ``k`` and records the achieved
recall rather than faking the gate (see ``DEFAULT_AG_BLOCKING_K`` / ``GATE_MET``).

Like Fodors-Zagat, this is a *linkage* task: two sources (Amazon ``tableA``,
Google ``tableB``) each list products, and the matches we care about are
cross-source. The k-sweep therefore filters candidate pairs to cross-source ones
before measuring recall (intra-source pairs are noise for this task; mirrors the
Fodors-Zagat adapter in :mod:`langres.data.er_benchmarks`).

Unlike Fodors-Zagat, Amazon-Google is a *harder, unsaturated* benchmark
(reported SOTA pairwise F1 ~0.5-0.75) and is genuinely many-to-many: a single
Amazon product can match several Google listings (and vice versa), so the gold
clusters are the connected components of the match graph — clusters can exceed
two records, not just cross-source pairs.

This module deliberately mirrors the Fodors-Zagat adapter's patterns (schema with
a ``@computed_field embed_text``, idempotent schema registration at import, a
combined source-prefixed corpus, a vector blocking k-sweep). The genuinely-shared
mechanics (CSV reading, the cross-source filter, the k-sweep/picker, and the
stratified split) live once in :mod:`langres.data._benchmark_utils`; this module
keeps the Amazon-Google-specific schema, pinned constants, and public wrappers.

:class:`AmazonGoogleBenchmark` adapts this loader to the dataset-agnostic
:class:`~langres.core.benchmark.Benchmark` protocol so
:func:`~langres.core.benchmark.run_method` can race any resolver factory against
it (it also conforms to ``langres.methods.BlockingBenchmark`` by exposing its
``schema`` + pinned blocking config), mirroring ``FodorsZagatBenchmark``.
"""

import logging
from collections.abc import Iterable
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
    "AG_RECALL_GATE",
    "DEFAULT_AG_BLOCKING_K",
    "DEFAULT_AG_THRESHOLD_GRID",
    "GATE_MET",
    "AmazonGoogleBenchmark",
    "ProductSchema",
    "build_product_blocker",
    "load_amazon_google",
    "load_amazon_google_pair_splits",
    "pick_blocking_k",
    "sweep_blocking_k",
]

_DATASET_PACKAGE = "langres.data.datasets.amazon_google"
_TABLE_A_FILE = "tableA.csv"  # Amazon records
_TABLE_B_FILE = "tableB.csv"  # Google records
#: Fixed literature pair splits (DeepMatcher/Magellan); label 1 = match.
_PAIR_SPLIT_FILES: dict[str, str] = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv",
}

ProductSource = Literal["amazon", "google"]

# Pinned blocking k for the cross-source Amazon-Google matches, measured with
# VectorBlocker over SentenceTransformer("all-MiniLM-L6-v2") cosine similarity on
# ``embed_text`` (title + manufacturer).
#
# Measured sweep (cross-source Pair-Completeness, 1167 gold pairs over the full
# 4589-record corpus; reproduce via ``sweep_blocking_k``):
#   k= 5 -> 0.7568
#   k=10 -> 0.8129
#   k=20 -> 0.8317
#   k=30 -> 0.8367
#   k=50 -> 0.8388
# Amazon-Google is genuinely hard: short, noisy product titles (and frequently
# missing manufacturers on the Google side) mean many true matches are not near
# neighbours, so recall plateaus around 0.84 and *never reaches the 0.90 gate*
# within k<=50 (contrast Fodors-Zagat, saturated at >=0.99 by k=5). Rather than
# fake the gate, DEFAULT_AG_BLOCKING_K is pinned to the HONEST BEST k=50 (highest
# measured PC), and the realised shortfall is recorded in
# ``ACHIEVED_PC_AT_DEFAULT_K`` (0.8388) and ``GATE_MET`` (False). Returns from
# ``pick_blocking_k`` agree: with no k clearing 0.90 it falls back to the
# best-recall k. Recall barely improves past k=20 (0.8317 -> 0.8388 by k=50), so
# larger k buys little; lifting blocking recall needs a richer blocking key (e.g.
# adding price/normalised title), tracked for a later wave.
DEFAULT_AG_BLOCKING_K = 50

#: Pair-Completeness gate the blocking k-sweep aims to clear. Amazon-Google is
#: harder than Fodors-Zagat, so the target is 0.90 (vs 0.95 there). NOTE: this
#: gate is NOT met by title+manufacturer vector blocking (see the sweep above);
#: it is the aspirational target, and ``GATE_MET`` records the honest outcome.
AG_RECALL_GATE = 0.90

#: Cross-source Pair-Completeness achieved at :data:`DEFAULT_AG_BLOCKING_K`,
#: recorded from the measured sweep above so callers can report the realised
#: blocking recall without re-running embeddings.
ACHIEVED_PC_AT_DEFAULT_K = 0.8388

#: Whether :data:`ACHIEVED_PC_AT_DEFAULT_K` clears :data:`AG_RECALL_GATE`. False
#: here: the honest measured best (0.8388) falls short of 0.90 — reported, not
#: hidden.
GATE_MET = ACHIEVED_PC_AT_DEFAULT_K >= AG_RECALL_GATE

#: Candidate Clusterer thresholds swept by ``run_method`` when racing methods on
#: Amazon-Google. Mirrors Fodors-Zagat's ``DEFAULT_THRESHOLD_GRID``: the zero-spend
#: judges score in ``[0, 1]``, and this small grid brackets the useful range
#: (tuned on TRAIN only). Amazon-Google is harder, so the best train threshold
#: typically lands lower than on Fodors-Zagat, but the same grid covers it.
DEFAULT_AG_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class ProductSchema(BaseModel):
    """A single product record from the Amazon-Google benchmark.

    ``id`` is globally unique across both sources (the loader prefixes the raw
    source id with ``a``/``g``). ``embed_text`` is the serializable blocking text
    used by the :class:`VectorBlocker` (referenced as ``text_field``).

    Attributes:
        id: Globally-unique record id (e.g. ``"a534"`` / ``"g219"``).
        title: Product title (always present).
        manufacturer: Manufacturer / brand, if present.
        price: Listed price as a raw string, if present.
        source: Originating table (``"amazon"`` for ``tableA``, ``"google"`` for
            ``tableB``).
    """

    id: str
    title: str
    manufacturer: str | None = None
    price: str | None = None
    source: ProductSource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: title and manufacturer joined by a space.

        Used as the :class:`VectorBlocker` ``text_field`` and as the text fed to
        the vector index. Omits a missing manufacturer so an absent field doesn't
        inject empty tokens (mirrors ``RestaurantSchema.embed_text``).
        """
        return " ".join(p for p in [self.title, self.manufacturer] if p)


# Register ProductSchema at import time so a fresh process that only imports this
# module (e.g. to ``Resolver.load`` a saved artifact and ``resolve``) finds the
# schema in the registry without first constructing a blocker. A declarative
# VectorBlocker re-registers idempotently, so this is the single source of truth
# for the registry key and round-trips a saved artifact's ``schema_type_name``
# (same pattern as ``RestaurantSchema``).
register_schema_idempotent(ProductSchema)


def _read_csv_rows(filename: str) -> list[dict[str, str]]:
    """Read a packaged Amazon-Google CSV into a list of header-keyed row dicts."""
    return _bu.read_csv_rows(_DATASET_PACKAGE, filename)


def _record_from_row(row: dict[str, str], source: ProductSource, prefix: str) -> ProductSchema:
    """Build a :class:`ProductSchema` from a raw CSV row dict.

    The Amazon-Google CSVs are plainly quoted (standard ``csv`` quoting handles
    embedded commas/quotes), so no source-specific unquoting is needed. Empty
    cells map to ``None`` for the optional fields; the required ``title`` instead
    falls back to ``""`` (the real data has no empty titles, but this keeps the
    field non-optional — mirrors ``er_benchmarks._record_from_row``).
    """

    def field(name: str) -> str | None:
        cleaned = row.get(name, "").strip()
        return cleaned or None

    return ProductSchema(
        id=f"{prefix}{row['id'].strip()}",
        title=field("title") or "",
        manufacturer=field("manufacturer"),
        price=field("price"),
        source=source,
    )


def load_amazon_google_pair_splits() -> dict[str, list[tuple[str, str, int]]]:
    """Load the fixed literature pair splits with source-prefixed ids.

    Each split file lists ``ltable_id,rtable_id,label`` rows, where ``ltable_id``
    indexes ``tableA`` (Amazon) and ``rtable_id`` indexes ``tableB`` (Google).
    The ids are mapped to the same ``a``/``g``-prefixed ids the corpus uses, so a
    split row references real corpus records. These FIXED splits are exposed for
    literature-comparable pair-level evaluation (W4); the corpus-level
    :func:`load_amazon_google` ground truth pools all three splits.

    Returns:
        Mapping of split name (``"train"``/``"valid"``/``"test"``) to a list of
        ``(amazon_id, google_id, label)`` tuples (``label`` is ``1`` or ``0``).
    """
    splits: dict[str, list[tuple[str, str, int]]] = {}
    for name, filename in _PAIR_SPLIT_FILES.items():
        splits[name] = [
            (f"a{row['ltable_id'].strip()}", f"g{row['rtable_id'].strip()}", int(row["label"]))
            for row in _read_csv_rows(filename)
        ]
    return splits


def _clusters_from_pairs(gold_pairs: set[frozenset[str]], all_ids: Iterable[str]) -> list[set[str]]:
    """Amazon-Google's connected-components partition (many-to-many linkage).

    Thin wrapper over the shared
    :func:`~langres.data._benchmark_utils.clusters_from_pairs` (also used by
    :mod:`langres.data.abt_buy`) -- kept as a local name so this module's
    existing internal call sites and tests are unaffected.
    """
    return _bu.clusters_from_pairs(gold_pairs, all_ids)


def load_amazon_google() -> tuple[list[ProductSchema], list[set[str]], set[frozenset[str]]]:
    """Load Amazon-Google as one corpus plus its complete partition and gold pairs.

    Both sources are combined into a single corpus with globally-unique,
    source-prefixed ids (``a<id>`` for Amazon ``tableA``, ``g<id>`` for Google
    ``tableB``). Ground truth pools the positive (``label == 1``) rows of all
    three fixed splits into ``gold_pairs`` (a ``set``, so a pair appearing in more
    than one split is counted once); ``gold_clusters`` is the connected components
    of those pairs, singleton-completed over the corpus (a closed-world partition;
    see :func:`_clusters_from_pairs`).

    Returns:
        ``(corpus, gold_clusters, gold_pairs)`` where ``corpus`` is the combined
        record list, ``gold_clusters`` is the complete partition (match
        components + singletons), and ``gold_pairs`` is the set of positive
        cross-source match pairs as frozensets.
    """
    corpus: list[ProductSchema] = [
        _record_from_row(row, "amazon", "a") for row in _read_csv_rows(_TABLE_A_FILE)
    ]
    corpus += [_record_from_row(row, "google", "g") for row in _read_csv_rows(_TABLE_B_FILE)]

    splits = load_amazon_google_pair_splits()
    gold_pairs: set[frozenset[str]] = {
        frozenset({left, right})
        for split in splits.values()
        for left, right, label in split
        if label == 1
    }

    gold_clusters = _clusters_from_pairs(gold_pairs, (r.id for r in corpus))
    match_clusters = [c for c in gold_clusters if len(c) >= 2]

    logger.info(
        "Loaded Amazon-Google: %d records (%d amazon + %d google), "
        "%d gold pairs, %d clusters (%d match components, %d singletons)",
        len(corpus),
        sum(1 for r in corpus if r.source == "amazon"),
        sum(1 for r in corpus if r.source == "google"),
        len(gold_pairs),
        len(gold_clusters),
        len(match_clusters),
        len(gold_clusters) - len(match_clusters),
    )
    return corpus, gold_clusters, gold_pairs


def build_product_blocker(
    k_neighbors: int = DEFAULT_AG_BLOCKING_K,
) -> VectorBlocker[ProductSchema]:
    """Build the shared product VectorBlocker (MiniLM + FAISS-cosine).

    Declarative (``schema=`` + ``text_field=``) so the resulting blocker is
    config-serializable. Each call constructs a *fresh* (unbuilt)
    :class:`FAISSIndex`; embedding only happens when the index is later populated
    from a corpus. :func:`sweep_blocking_k` intentionally does **not** use this
    factory: it shares one pre-built index across every ``k`` to embed the corpus
    once. Mirrors ``build_restaurant_blocker``.

    Args:
        k_neighbors: Nearest neighbours per record. Defaults to
            :data:`DEFAULT_AG_BLOCKING_K` (the best-measured ``k``; the 0.90 gate
            is *not* met — see :data:`ACHIEVED_PC_AT_DEFAULT_K` / :data:`GATE_MET`).

    Returns:
        A :class:`VectorBlocker` over ``ProductSchema.embed_text``.
    """
    return VectorBlocker(
        vector_index=FAISSIndex(
            embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
            metric="cosine",
        ),
        schema=ProductSchema,
        text_field="embed_text",
        k_neighbors=k_neighbors,
    )


def _cross_source(
    candidates: list[ERCandidate[ProductSchema]],
) -> list[ERCandidate[ProductSchema]]:
    """Keep only candidate pairs whose two records come from different sources.

    Product-typed wrapper over :func:`langres.data._benchmark_utils.cross_source`.
    """
    return _bu.cross_source(candidates)


def sweep_blocking_k(
    corpus: list[ProductSchema],
    gold_clusters: list[set[str]],
    ks: tuple[int, ...] = (5, 10, 20, 30, 50),
) -> dict[int, float]:
    """Measure cross-source Pair-Completeness of vector blocking across ``ks``.

    Product-typed wrapper over
    :func:`langres.data._benchmark_utils.sweep_blocking_k`, binding
    ``ProductSchema`` + ``embed_text``: builds the FAISS index once and reuses it
    across all ``k``, filtering to cross-source pairs before measuring recall.
    ``candidate_recall`` *is* Pair-Completeness (fraction of gold match pairs
    surfaced as candidates), the quantity the gate is defined on.

    Args:
        corpus: Combined record list from :func:`load_amazon_google`.
        gold_clusters: The complete partition from :func:`load_amazon_google`.
        ks: Neighbor counts to sweep.

    Returns:
        Mapping of ``k`` to cross-source Pair-Completeness (``candidate_recall``).
    """
    return _bu.sweep_blocking_k(
        corpus, gold_clusters, ProductSchema, text_field="embed_text", ks=ks
    )


def pick_blocking_k(recalls: dict[int, float], threshold: float = AG_RECALL_GATE) -> int:
    """Pick the smallest ``k`` whose recall clears ``threshold`` (default the AG gate).

    Product-typed wrapper over
    :func:`langres.data._benchmark_utils.pick_blocking_k` defaulting ``threshold``
    to the Amazon-Google :data:`AG_RECALL_GATE` (0.90). If no ``k`` reaches it,
    returns the ``k`` with the highest recall (honest fallback — see the sweep
    note above; the gate is not met within ``k<=50``).

    Args:
        recalls: Mapping of ``k`` to recall, e.g. from :func:`sweep_blocking_k`.
        threshold: Minimum acceptable recall (defaults to :data:`AG_RECALL_GATE`).

    Returns:
        The chosen ``k``.

    Raises:
        ValueError: If ``recalls`` is empty.
    """
    return _bu.pick_blocking_k(recalls, threshold)


class AmazonGoogleBenchmark(Benchmark[ProductSchema]):
    """Amazon-Google as a :class:`~langres.core.benchmark.Benchmark` conformer.

    Adapts the product loader/splitter to the dataset-agnostic harness so
    :func:`~langres.core.benchmark.run_method` can run any resolver factory
    against it: ``load`` returns the corpus, the closed-world gold partition, and
    the within-cluster gold match pairs; ``split`` delegates to the shared
    stratified split. Mirrors ``FodorsZagatBenchmark`` exactly, swapping the
    restaurant schema/loaders for the product ones.

    It also conforms to ``langres.methods.BlockingBenchmark`` by exposing its
    record ``schema`` and pinned blocking config (``blocking_k`` +
    :meth:`build_blocker`), so the method registry can race *any* of the five
    methods against it on the identical candidate set. Blocking is pinned to the
    honest best ``k`` (:data:`DEFAULT_AG_BLOCKING_K`); the 0.90 Pair-Completeness
    gate is *not* met (see :data:`ACHIEVED_PC_AT_DEFAULT_K` / :data:`GATE_MET`),
    so end-to-end recall is ceiling-limited at ~0.84 here.
    """

    name = "amazon_google"
    threshold_grid = DEFAULT_AG_THRESHOLD_GRID
    #: Record type, exposed for the method registry's Comparator/rapidfuzz fields.
    schema = ProductSchema
    #: Pinned blocking neighbour count (blocking held constant across methods).
    blocking_k = DEFAULT_AG_BLOCKING_K

    def build_blocker(self, k_neighbors: int) -> VectorBlocker[ProductSchema]:
        """Return a fresh product VectorBlocker (MiniLM + FAISS-cosine).

        Delegates to :func:`build_product_blocker` so the race shares the exact
        blocking config used elsewhere; each call yields a fresh, unbuilt index.
        """
        return build_product_blocker(k_neighbors)

    def load(self) -> tuple[list[ProductSchema], list[set[str]], set[frozenset[str]]]:
        """Return ``(corpus, gold_clusters, gold_pairs)`` for Amazon-Google.

        ``gold_pairs`` is derived from the gold partition (every within-cluster
        pair, including the transitive closure of the many-to-many matches) so the
        protocol's pair semantics match Fodors-Zagat and what ``run_method``
        recomputes per split.
        """
        corpus, gold_clusters, _gold_pairs = load_amazon_google()
        return corpus, gold_clusters, gold_pairs_from_clusters(gold_clusters)

    def split(
        self,
        corpus: list[ProductSchema],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[ProductSchema], list[ProductSchema], list[set[str]], list[set[str]]]:
        """Leakage-free stratified split via the shared ``stratified_corpus_split``."""
        return _bu.stratified_corpus_split(corpus, gold_clusters, seed=seed)
