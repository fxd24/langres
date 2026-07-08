"""Walmart-Amazon (structured) product entity-resolution benchmark, via the loader factory.

The **structured** Walmart-Amazon benchmark (the non-``-SM`` variant the EM
literature reports on): two e-commerce sources — Walmart (``tableA``) and Amazon
(``tableB``) — each list electronics products across clean, comparable columns
(``title, category, brand, modelno, price``); the task is to find the
cross-source pairs that refer to the same real-world product. Like Amazon-Google
and Abt-Buy it is a *linkage* task whose true matches are all cross-source, and
it is genuinely many-to-many (an Amazon listing can match several Walmart
listings), so the gold clusters are the connected components of the match graph.

This is a ~30-line dataset module: a dataset-namespaced schema
(:class:`WalmartAmazonSchema` — NEVER the shared ``ProductSchema``, which
``amazon_google`` already registers) plus one
:func:`~langres.data._deepmatcher_loader.make_deepmatcher_benchmark` call carrying
the honest pinned blocking constants. The factory owns the shared anatomy
(source-prefixed corpus, fixed literature pair splits, connected-components gold
clusters, stratified split, lazy ``[semantic]`` blocker) — see
``datasets/walmart_amazon/ATTRIBUTION.md`` for provenance and counts.
"""

from typing import Literal

from pydantic import BaseModel, computed_field

from langres.data._deepmatcher_loader import SourceTable, make_deepmatcher_benchmark

__all__ = [
    "WALMART_AMAZON_ACHIEVED_PC",
    "WALMART_AMAZON_BLOCKING_K",
    "WALMART_AMAZON_GATE_MET",
    "WALMART_AMAZON_RECALL_GATE",
    "WALMART_AMAZON_THRESHOLD_GRID",
    "WalmartAmazonBenchmark",
    "WalmartAmazonSchema",
    "load_walmart_amazon",
    "load_walmart_amazon_pair_splits",
]

_DATASET_PACKAGE = "langres.data.datasets.walmart_amazon"

WalmartAmazonSource = Literal["a", "b"]

# Pinned blocking k for cross-source Walmart-Amazon matches, measured with
# VectorBlocker over SentenceTransformer("all-MiniLM-L6-v2") cosine similarity on
# ``embed_text`` (title + brand + modelno + category).
#
# Measured cross-source Pair-Completeness sweep (1092 gold pairs — the
# transitive closure of 962 positive labels — over the full 24628-record corpus;
# reproduce via ``tmp/measure_walmart_amazon_blocking.py`` /
# ``_benchmark_utils.sweep_blocking_k``):
#   k= 5 -> 0.8013
#   k=10 -> 0.8471
#   k=20 -> 0.8645
#   k=30 -> 0.8709
#   k=50 -> 0.8773
# Walmart-Amazon is a hard, unsaturated benchmark: short product titles plus a
# 22k-record Amazon side (many near-duplicate listings) mean many true matches
# are not top-k neighbours, so recall climbs slowly and *never reaches the 0.90
# gate* within k<=50 (contrast Fodors-Zagat, saturated at >=0.99 by k=5). Rather
# than fake the gate, WALMART_AMAZON_BLOCKING_K is pinned to the HONEST BEST k=50
# (highest measured PC), and the shortfall is recorded in
# WALMART_AMAZON_ACHIEVED_PC (0.8773) and WALMART_AMAZON_GATE_MET (False). Recall
# gains flatten past k=20 (0.8645 -> 0.8773 by k=50), so larger k buys little;
# lifting blocking recall needs a richer blocking key (e.g. adding normalised
# modelno), tracked for a later wave.
WALMART_AMAZON_BLOCKING_K = 50

#: Pair-Completeness gate the blocking k-sweep aims to clear (0.90, as for
#: Amazon-Google). NOTE: this gate is NOT met by title+brand+modelno+category
#: vector blocking (see the sweep above); it is the aspirational target, and
#: ``WALMART_AMAZON_GATE_MET`` records the honest outcome.
WALMART_AMAZON_RECALL_GATE = 0.90

#: Cross-source Pair-Completeness achieved at :data:`WALMART_AMAZON_BLOCKING_K`,
#: recorded from the measured sweep above so callers can report the realised
#: blocking recall without re-running embeddings.
WALMART_AMAZON_ACHIEVED_PC = 0.8773

#: Whether :data:`WALMART_AMAZON_ACHIEVED_PC` clears
#: :data:`WALMART_AMAZON_RECALL_GATE`. False here: the honest measured best
#: (0.8773) falls short of 0.90 — reported, not hidden.
WALMART_AMAZON_GATE_MET = WALMART_AMAZON_ACHIEVED_PC >= WALMART_AMAZON_RECALL_GATE

#: Clusterer thresholds swept when racing methods (mirrors the other adapters).
WALMART_AMAZON_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class WalmartAmazonSchema(BaseModel):
    """A single product record from the structured Walmart-Amazon benchmark.

    ``id`` is globally unique across both sources (the loader prefixes the raw
    source id with ``a``/``b``). ``embed_text`` is the serializable blocking text
    used by the :class:`VectorBlocker` (referenced as ``text_field``).

    Attributes:
        id: Globally-unique record id (e.g. ``"a534"`` / ``"b219"``).
        title: Product title (always present).
        category: Product category, if present.
        brand: Brand / manufacturer, if present.
        modelno: Manufacturer model number, if present.
        price: Listed price as a raw string, if present.
        source: Originating table (``"a"`` for Walmart ``tableA``, ``"b"`` for
            Amazon ``tableB``).
    """

    id: str
    title: str
    category: str | None = None
    brand: str | None = None
    modelno: str | None = None
    price: str | None = None
    source: WalmartAmazonSource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: title, brand, modelno and category joined by a space.

        Omits ``price`` (a bare number is noise for semantic matching) and any
        missing field so an absent value doesn't inject empty tokens (mirrors
        ``ProductSchema.embed_text``).
        """
        return " ".join(p for p in [self.title, self.brand, self.modelno, self.category] if p)


load_walmart_amazon, load_walmart_amazon_pair_splits, WalmartAmazonBenchmark = (
    make_deepmatcher_benchmark(
        name="walmart_amazon",
        schema=WalmartAmazonSchema,
        dataset_package=_DATASET_PACKAGE,
        table_a=SourceTable(file="tableA.csv", source="a", id_prefix="a"),
        table_b=SourceTable(file="tableB.csv", source="b", id_prefix="b"),
        split_files={"train": "train.csv", "valid": "valid.csv", "test": "test.csv"},
        blocking_k=WALMART_AMAZON_BLOCKING_K,
        threshold_grid=WALMART_AMAZON_THRESHOLD_GRID,
        achieved_pc=WALMART_AMAZON_ACHIEVED_PC,
        gate_met=WALMART_AMAZON_GATE_MET,
        benchmark_class_name="WalmartAmazonBenchmark",
    )
)
