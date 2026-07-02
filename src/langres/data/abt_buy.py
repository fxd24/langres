"""Abt-Buy product entity-resolution benchmark adapter (M4.5/W1.2 — CEO #9).

Loads the vendored Abt-Buy benchmark into a single corpus of
:class:`AbtBuySchema` records plus cross-source ground-truth pairs. Like
Amazon-Google, this is a *linkage* task: two sources (Abt Electronics
``tableA``, Buy.com ``tableB``) each list products, and the matches we care
about are cross-source.

Unlike Fodors-Zagat (clean, saturated) and closer to (but distinct from)
Amazon-Google, Abt-Buy is the **textual-hard** dataset in the M4.5 replication
matrix (CEO decision #9): records carry a free-text ``description`` field
(frequently missing on the Buy side) rather than a rich set of clean,
comparable columns — it stresses judges that rely on short, noisy text.

This module deliberately mirrors :mod:`langres.data.amazon_google`'s patterns
(schema with a ``@computed_field embed_text``, idempotent schema registration
at import, a combined source-prefixed corpus). The genuinely-shared mechanics
(CSV reading, the cross-source filter, the k-sweep/picker, the connected-
components partition, and the stratified split) live once in
:mod:`langres.data._benchmark_utils`; this module keeps the Abt-Buy-specific
schema, pinned constants, and public wrappers.

:class:`AbtBuyBenchmark` adapts this loader to the dataset-agnostic
:class:`~langres.core.benchmark.Benchmark` protocol (and
``langres.methods.BlockingBenchmark``) so it races through the same harness
as Fodors-Zagat and Amazon-Google.
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
    "ABT_BUY_RECALL_GATE",
    "ABT_BUY_THRESHOLD_GRID",
    "ACHIEVED_PC_AT_DEFAULT_K",
    "DEFAULT_ABT_BUY_BLOCKING_K",
    "GATE_MET",
    "AbtBuyBenchmark",
    "AbtBuySchema",
    "build_abt_buy_blocker",
    "load_abt_buy",
    "load_abt_buy_pair_splits",
    "pick_blocking_k",
    "sweep_blocking_k",
]

_DATASET_PACKAGE = "langres.data.datasets.abt_buy"
_TABLE_A_FILE = "tableA.csv"  # Abt records
_TABLE_B_FILE = "tableB.csv"  # Buy records
#: Fixed literature pair splits (DeepMatcher/Magellan); label 1 = match.
_PAIR_SPLIT_FILES: dict[str, str] = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv",
}

AbtBuySource = Literal["abt", "buy"]

# Pinned blocking k for the cross-source Abt-Buy matches, measured with
# VectorBlocker over SentenceTransformer("all-MiniLM-L6-v2") cosine similarity
# on ``embed_text`` (name + description). Sweep (k -> Pair-Completeness):
# 5=0.6983, 10=0.8218, 20=0.9301, 30=0.9511, 50=0.9780 -- see
# ``sweep_blocking_k`` / ``docs/research/w1_trained_family_results.md``.
DEFAULT_ABT_BUY_BLOCKING_K = 20

#: Pair-Completeness gate the blocking k-sweep aims to clear (mirrors
#: Amazon-Google's harder-than-Fodors-Zagat 0.90 target -- Abt-Buy's missing
#: descriptions make it a comparably hard blocking target).
ABT_BUY_RECALL_GATE = 0.90

#: Cross-source Pair-Completeness actually achieved at :data:`DEFAULT_ABT_BUY_BLOCKING_K`.
ACHIEVED_PC_AT_DEFAULT_K = 0.9301

#: Whether :data:`ACHIEVED_PC_AT_DEFAULT_K` clears :data:`ABT_BUY_RECALL_GATE`.
#: Unlike Amazon-Google (title+manufacturer text is comparatively weak), the
#: Abt-Buy k-sweep *does* clear the 0.90 gate at k=20 -- reported honestly like
#: Amazon-Google's honestly-not-met gate, not assumed.
GATE_MET = True

#: Candidate Clusterer thresholds swept when racing methods on Abt-Buy.
#: Mirrors ``DEFAULT_AG_THRESHOLD_GRID``.
ABT_BUY_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class AbtBuySchema(BaseModel):
    """A single product record from the Abt-Buy benchmark.

    ``id`` is globally unique across both sources (the loader prefixes the raw
    source id with ``a``/``b``). ``embed_text`` is the serializable blocking
    text used by the :class:`VectorBlocker`.

    Attributes:
        id: Globally-unique record id (e.g. ``"a534"`` / ``"b219"``).
        name: Product name (always present).
        description: Free-text description, if present (frequently missing on
            the Buy side).
        price: Listed price as a raw string, if present.
        source: Originating table (``"abt"`` for ``tableA``, ``"buy"`` for
            ``tableB``).
    """

    id: str
    name: str
    description: str | None = None
    price: str | None = None
    source: AbtBuySource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: name and description joined by a space.

        Omits a missing description so an absent field doesn't inject empty
        tokens (mirrors ``ProductSchema.embed_text``).
        """
        return " ".join(p for p in [self.name, self.description] if p)


# Register AbtBuySchema at import time so a fresh process that only imports
# this module (e.g. to ``Resolver.load`` a saved artifact and ``resolve``)
# finds the schema in the registry without first constructing a blocker.
register_schema_idempotent(AbtBuySchema)


def _read_csv_rows(filename: str) -> list[dict[str, str]]:
    """Read a packaged Abt-Buy CSV into a list of header-keyed row dicts."""
    return _bu.read_csv_rows(_DATASET_PACKAGE, filename)


def _record_from_row(row: dict[str, str], source: AbtBuySource, prefix: str) -> AbtBuySchema:
    """Build an :class:`AbtBuySchema` from a raw CSV row dict.

    The Abt-Buy CSVs are plainly quoted (standard ``csv`` quoting handles
    embedded commas/quotes), so no source-specific unquoting is needed. Empty
    cells map to ``None`` for the optional fields; the required ``name``
    instead falls back to ``""`` (the real data has no empty names, but this
    keeps the field non-optional — mirrors ``amazon_google._record_from_row``).
    """

    def field(name: str) -> str | None:
        cleaned = row.get(name, "").strip()
        return cleaned or None

    return AbtBuySchema(
        id=f"{prefix}{row['id'].strip()}",
        name=field("name") or "",
        description=field("description"),
        price=field("price"),
        source=source,
    )


def load_abt_buy_pair_splits() -> dict[str, list[tuple[str, str, int]]]:
    """Load the fixed literature pair splits with source-prefixed ids.

    Each split file lists ``ltable_id,rtable_id,label`` rows, where
    ``ltable_id`` indexes ``tableA`` (Abt) and ``rtable_id`` indexes
    ``tableB`` (Buy). The ids are mapped to the same ``a``/``b``-prefixed ids
    the corpus uses.

    Returns:
        Mapping of split name (``"train"``/``"valid"``/``"test"``) to a list
        of ``(abt_id, buy_id, label)`` tuples (``label`` is ``1`` or ``0``).
    """
    splits: dict[str, list[tuple[str, str, int]]] = {}
    for name, filename in _PAIR_SPLIT_FILES.items():
        splits[name] = [
            (f"a{row['ltable_id'].strip()}", f"b{row['rtable_id'].strip()}", int(row["label"]))
            for row in _read_csv_rows(filename)
        ]
    return splits


def load_abt_buy() -> tuple[list[AbtBuySchema], list[set[str]], set[frozenset[str]]]:
    """Load Abt-Buy as one corpus plus its complete partition and gold pairs.

    Both sources are combined into a single corpus with globally-unique,
    source-prefixed ids (``a<id>`` for Abt ``tableA``, ``b<id>`` for Buy
    ``tableB``). Ground truth pools the positive (``label == 1``) rows of all
    three fixed splits into ``gold_pairs``; ``gold_clusters`` is the connected
    components of those pairs, singleton-completed over the corpus (a
    closed-world partition).

    Returns:
        ``(corpus, gold_clusters, gold_pairs)``.
    """
    corpus: list[AbtBuySchema] = [
        _record_from_row(row, "abt", "a") for row in _read_csv_rows(_TABLE_A_FILE)
    ]
    corpus += [_record_from_row(row, "buy", "b") for row in _read_csv_rows(_TABLE_B_FILE)]

    splits = load_abt_buy_pair_splits()
    gold_pairs: set[frozenset[str]] = {
        frozenset({left, right})
        for split in splits.values()
        for left, right, label in split
        if label == 1
    }

    gold_clusters = _bu.clusters_from_pairs(gold_pairs, (r.id for r in corpus))
    match_clusters = [c for c in gold_clusters if len(c) >= 2]

    logger.info(
        "Loaded Abt-Buy: %d records (%d abt + %d buy), "
        "%d gold pairs, %d clusters (%d match components, %d singletons)",
        len(corpus),
        sum(1 for r in corpus if r.source == "abt"),
        sum(1 for r in corpus if r.source == "buy"),
        len(gold_pairs),
        len(gold_clusters),
        len(match_clusters),
        len(gold_clusters) - len(match_clusters),
    )
    return corpus, gold_clusters, gold_pairs


def build_abt_buy_blocker(
    k_neighbors: int = DEFAULT_ABT_BUY_BLOCKING_K,
) -> VectorBlocker[AbtBuySchema]:
    """Build the shared Abt-Buy VectorBlocker (MiniLM + FAISS-cosine).

    Declarative (``schema=`` + ``text_field=``) so the resulting blocker is
    config-serializable. Each call constructs a *fresh* (unbuilt)
    :class:`FAISSIndex`; embedding only happens when the index is later
    populated from a corpus. Mirrors ``build_product_blocker``.

    Args:
        k_neighbors: Nearest neighbours per record. Defaults to
            :data:`DEFAULT_ABT_BUY_BLOCKING_K`.

    Returns:
        A :class:`VectorBlocker` over ``AbtBuySchema.embed_text``.
    """
    return VectorBlocker(
        vector_index=FAISSIndex(
            embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
            metric="cosine",
        ),
        schema=AbtBuySchema,
        text_field="embed_text",
        k_neighbors=k_neighbors,
    )


def _cross_source(candidates: list[ERCandidate[AbtBuySchema]]) -> list[ERCandidate[AbtBuySchema]]:
    """Keep only candidate pairs whose two records come from different sources."""
    return _bu.cross_source(candidates)


def sweep_blocking_k(
    corpus: list[AbtBuySchema],
    gold_clusters: list[set[str]],
    ks: tuple[int, ...] = (5, 10, 20, 30, 50),
) -> dict[int, float]:
    """Measure cross-source Pair-Completeness of vector blocking across ``ks``.

    Product-typed wrapper over
    :func:`langres.data._benchmark_utils.sweep_blocking_k`, binding
    ``AbtBuySchema`` + ``embed_text``.

    Args:
        corpus: Combined record list from :func:`load_abt_buy`.
        gold_clusters: The complete partition from :func:`load_abt_buy`.
        ks: Neighbor counts to sweep.

    Returns:
        Mapping of ``k`` to cross-source Pair-Completeness.
    """
    return _bu.sweep_blocking_k(corpus, gold_clusters, AbtBuySchema, text_field="embed_text", ks=ks)


def pick_blocking_k(recalls: dict[int, float], threshold: float = ABT_BUY_RECALL_GATE) -> int:
    """Pick the smallest ``k`` whose recall clears ``threshold`` (default the Abt-Buy gate).

    Args:
        recalls: Mapping of ``k`` to recall, e.g. from :func:`sweep_blocking_k`.
        threshold: Minimum acceptable recall (defaults to
            :data:`ABT_BUY_RECALL_GATE`).

    Returns:
        The chosen ``k``.

    Raises:
        ValueError: If ``recalls`` is empty.
    """
    return _bu.pick_blocking_k(recalls, threshold)


class AbtBuyBenchmark(Benchmark[AbtBuySchema]):
    """Abt-Buy as a :class:`~langres.core.benchmark.Benchmark` conformer.

    Mirrors ``AmazonGoogleBenchmark`` exactly, swapping the product schema/
    loaders for the Abt-Buy ones. Also conforms to
    ``langres.methods.BlockingBenchmark`` (``schema`` + ``blocking_k`` +
    :meth:`build_blocker`).
    """

    name = "abt_buy"
    threshold_grid = ABT_BUY_THRESHOLD_GRID
    #: Record type, exposed for the method registry's Comparator/rapidfuzz fields.
    schema = AbtBuySchema
    #: Pinned blocking neighbour count (blocking held constant across methods).
    blocking_k = DEFAULT_ABT_BUY_BLOCKING_K

    def build_blocker(self, k_neighbors: int) -> VectorBlocker[AbtBuySchema]:
        """Return a fresh Abt-Buy VectorBlocker (MiniLM + FAISS-cosine)."""
        return build_abt_buy_blocker(k_neighbors)

    def load(self) -> tuple[list[AbtBuySchema], list[set[str]], set[frozenset[str]]]:
        """Return ``(corpus, gold_clusters, gold_pairs)`` for Abt-Buy."""
        corpus, gold_clusters, _gold_pairs = load_abt_buy()
        return corpus, gold_clusters, gold_pairs_from_clusters(gold_clusters)

    def split(
        self,
        corpus: list[AbtBuySchema],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[AbtBuySchema], list[AbtBuySchema], list[set[str]], list[set[str]]]:
        """Leakage-free stratified split via the shared ``stratified_corpus_split``."""
        return _bu.stratified_corpus_split(corpus, gold_clusters, seed=seed)
