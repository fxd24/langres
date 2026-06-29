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
combined source-prefixed corpus, a vector blocking k-sweep) in a *separate* file
so the two benchmarks stay decoupled.

# TODO(W3/W4): wrap as Benchmark protocol once core.benchmark lands (integration
# adds the thin protocol adapter; intentionally out of scope for this module).
"""

import csv
import logging
from collections import defaultdict
from collections.abc import Iterable
from importlib import resources
from typing import Literal

from pydantic import BaseModel, computed_field

from langres.core.blockers.all_pairs import register_schema_idempotent
from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.metrics import evaluate_blocking
from langres.core.models import ERCandidate

logger = logging.getLogger(__name__)

__all__ = [
    "ACHIEVED_PC_AT_DEFAULT_K",
    "AG_RECALL_GATE",
    "DEFAULT_AG_BLOCKING_K",
    "GATE_MET",
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
    """Read a packaged benchmark CSV into a list of header-keyed row dicts."""
    text = resources.files(_DATASET_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    return [dict(row) for row in reader]


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
    """Connected components of the match graph, singleton-completed over ``all_ids``.

    Amazon-Google is many-to-many, so the gold clusters are the connected
    components of the undirected graph whose edges are the positive pairs (a
    record reachable from another via a chain of matches shares its entity).
    A tiny union-find computes the components; every corpus id not touched by any
    match becomes its own singleton, yielding the **complete closed-world
    partition** (match components + singletons) — exactly like Fodors-Zagat's
    ``perfectMapping`` completion. Singletons add no positive pairs, so blocking
    Pair-Completeness is unaffected.

    Args:
        gold_pairs: Positive match pairs as 2-element frozensets of corpus ids.
        all_ids: Every corpus id (e.g. ``[r.id for r in corpus]``); order fixes
            the singleton order for determinism.

    Returns:
        The complete partition: match components followed by one singleton per
        unmatched id (in ``all_ids`` order).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in gold_pairs:
        a, b = tuple(pair)
        union(a, b)

    components: dict[str, set[str]] = defaultdict(set)
    for node in parent:
        components[find(node)].add(node)
    match_clusters = list(components.values())

    matched_ids = set(parent)
    singletons = [{rid} for rid in all_ids if rid not in matched_ids]
    return match_clusters + singletons


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
    """Keep only candidate pairs whose two records come from different sources."""
    return [c for c in candidates if c.left.source != c.right.source]


def sweep_blocking_k(
    corpus: list[ProductSchema],
    gold_clusters: list[set[str]],
    ks: tuple[int, ...] = (5, 10, 20, 30, 50),
) -> dict[int, float]:
    """Measure cross-source Pair-Completeness of vector blocking across ``ks``.

    Builds the FAISS index once over ``embed_text`` and reuses it across all
    ``k`` (only ``k_neighbors`` changes). For each ``k`` the candidates are
    filtered to cross-source pairs before recall is measured via
    :func:`~langres.core.metrics.evaluate_blocking`. Mirrors the Fodors-Zagat
    sweep.

    ``candidate_recall`` from :func:`evaluate_blocking` *is* Pair-Completeness
    (fraction of gold match pairs surfaced as candidates), which is the quantity
    the gate is defined on — so this uses ``evaluate_blocking`` rather than
    ``evaluate_blocking_with_ranking`` (the latter reports truncated Recall@K and
    requires per-candidate ``similarity_score``).

    Args:
        corpus: Combined record list from :func:`load_amazon_google`.
        gold_clusters: The complete partition from :func:`load_amazon_google`.
        ks: Neighbor counts to sweep.

    Returns:
        Mapping of ``k`` to cross-source Pair-Completeness (``candidate_recall``).
    """
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    index = FAISSIndex(embedder=embedder, metric="cosine")
    index.create_index([r.embed_text for r in corpus])
    records = [r.model_dump() for r in corpus]

    recalls: dict[int, float] = {}
    for k in ks:
        # Construct a fresh blocker per k (the pre-built FAISS index is reused, so
        # this is cheap); only k_neighbors varies. ``k`` lives on the blocker, not
        # the index, so sharing one index across ks is safe.
        blocker: VectorBlocker[ProductSchema] = VectorBlocker(
            vector_index=index,
            schema=ProductSchema,
            text_field="embed_text",
            k_neighbors=k,
        )
        candidates = _cross_source(list(blocker.stream(records)))
        recall = evaluate_blocking(candidates, gold_clusters).candidate_recall
        recalls[k] = recall
        logger.info("blocking k=%d -> cross-source recall=%.4f", k, recall)
    return recalls


def pick_blocking_k(recalls: dict[int, float], threshold: float = AG_RECALL_GATE) -> int:
    """Pick the smallest ``k`` whose recall clears ``threshold``.

    If no ``k`` reaches ``threshold``, returns the ``k`` with the highest recall
    (the honest best-effort fallback; callers should document the shortfall
    rather than fake the gate). Mirrors the Fodors-Zagat picker.

    Args:
        recalls: Mapping of ``k`` to recall, e.g. from :func:`sweep_blocking_k`.
        threshold: Minimum acceptable recall.

    Returns:
        The chosen ``k``.

    Raises:
        ValueError: If ``recalls`` is empty.
    """
    if not recalls:
        raise ValueError("recalls is empty; nothing to pick from")
    passing = [k for k in sorted(recalls) if recalls[k] >= threshold]
    if passing:
        return passing[0]
    return max(recalls, key=lambda k: recalls[k])
