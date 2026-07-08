"""WDC-computers product entity-resolution benchmark, built via the loader factory (Wave B).

The **computers** category of the Web Data Commons (WDC) product-matching corpus,
redistributed by the matchbench Hugging Face mirror as a DeepMatcher-style
two-table benchmark (``tableA`` / ``tableB`` + fixed ``train`` / ``valid`` /
``test`` pair splits). Two sources each list web product offers; the task is to
find the cross-source pairs referring to the same real-world product. Like
Amazon-Google and Abt-Buy it is *textual-hard*: each record is a single free-text
``title`` blob (title + specs + brand + retailer noise, frequently with
language-tagged fragments), so the loader's blocking text is that ``title`` alone
— the matchbench mirror ships only ``id,title`` columns, so there is no separate
description/brand/price to concatenate (contrast Amazon-Google's title +
manufacturer). See ``datasets/wdc_computers/ATTRIBUTION.md`` for provenance,
row/pair counts, and the license (CC-BY 4.0 via matchbench).

Like Amazon-Google this is genuinely **many-to-many** (an offer can match several
others), so the gold clusters are the connected components of the match graph and
``gold_pairs`` is their transitive closure — not a strict 1:1 pairing.

This module is ~one factory call plus the schema; the shared anatomy (CSV reading,
id remapping, connected-components partition, stratified split, lazy vector
blocker) lives in :func:`~langres.data._deepmatcher_loader.make_deepmatcher_benchmark`.
It additionally exposes :func:`wdc_slice_map`, a *derived* seen/unseen slice used
by Wave D to demonstrate the honest seen -> unseen F1 drop at a fixed threshold.
"""

from typing import Literal

from pydantic import BaseModel, computed_field

from langres.data._deepmatcher_loader import SourceTable, make_deepmatcher_benchmark

__all__ = [
    "WDC_COMPUTERS_ACHIEVED_PC",
    "WDC_COMPUTERS_BLOCKING_K",
    "WDC_COMPUTERS_GATE_MET",
    "WDC_COMPUTERS_RECALL_GATE",
    "WDC_COMPUTERS_THRESHOLD_GRID",
    "WdcComputersBenchmark",
    "WdcComputersSchema",
    "load_wdc_computers",
    "load_wdc_computers_pair_splits",
    "wdc_slice_map",
]

_DATASET_PACKAGE = "langres.data.datasets.wdc_computers"

WdcComputersSource = Literal["a", "b"]

#: Pair-Completeness gate the blocking k-sweep aims to clear. WDC-computers is a
#: textual-hard, many-to-many product benchmark (like Amazon-Google), so the
#: target is 0.90; :data:`WDC_COMPUTERS_GATE_MET` records the honest outcome.
WDC_COMPUTERS_RECALL_GATE = 0.90

# Pinned blocking k for the cross-source WDC-computers matches, measured with
# VectorBlocker over SentenceTransformer("all-MiniLM-L6-v2") cosine similarity on
# ``embed_text`` (the single ``title`` blob).
#
# Measured cross-source Pair-Completeness sweep (1111 transitive-closure gold
# pairs over the full 4647-record corpus; reproduce via
# ``_bu.sweep_blocking_k(corpus, gold_clusters, WdcComputersSchema,
# text_field="embed_text", ks=(5,10,20,30,50))``):
#   k= 5 -> 0.3375
#   k=10 -> 0.4608
#   k=20 -> 0.5770
#   k=30 -> 0.6526
#   k=50 -> 0.7237
# WDC-computers is genuinely hard: each record is one noisy free-text ``title``
# blob (title + specs + brand + multilingual retailer fragments), so many true
# matches are not near neighbours and recall never reaches the 0.90 gate within
# k<=50 (contrast Fodors-Zagat, saturated by k=5). Rather than fake the gate,
# WDC_COMPUTERS_BLOCKING_K is pinned to the HONEST BEST k=50 (highest measured
# PC); ``pick_blocking_k(pc_by_k, 0.90)`` agrees — with no k clearing the gate it
# falls back to the best-recall k=50. The realised shortfall is recorded in
# WDC_COMPUTERS_ACHIEVED_PC (0.7237) and WDC_COMPUTERS_GATE_MET (False). Recall is
# still climbing at k=50 (0.6526 -> 0.7237 from k=30), so a larger k would help;
# clearing 0.90 likely needs a richer blocking key than the raw title blob,
# tracked for a later wave.
WDC_COMPUTERS_BLOCKING_K = 50

#: Cross-source Pair-Completeness achieved at :data:`WDC_COMPUTERS_BLOCKING_K`,
#: recorded from the measured sweep above so callers can report the realised
#: blocking recall without re-running embeddings.
WDC_COMPUTERS_ACHIEVED_PC = 0.7237

#: Whether :data:`WDC_COMPUTERS_ACHIEVED_PC` clears :data:`WDC_COMPUTERS_RECALL_GATE`.
#: False here: the honest measured best (0.7237) falls short of 0.90 — reported,
#: not hidden (mirrors Amazon-Google's honest miss).
WDC_COMPUTERS_GATE_MET = WDC_COMPUTERS_ACHIEVED_PC >= WDC_COMPUTERS_RECALL_GATE

#: Clusterer thresholds swept when racing methods (mirrors the other adapters).
WDC_COMPUTERS_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class WdcComputersSchema(BaseModel):
    """A single web product-offer record from the WDC-computers benchmark.

    ``id`` is globally unique across both sources (the loader prefixes the raw
    source id with ``a``/``b``). ``embed_text`` is the serializable blocking text
    used by the :class:`VectorBlocker` (referenced as ``text_field``).

    Attributes:
        id: Globally-unique record id (e.g. ``"a534"`` / ``"b219"``).
        title: The web-offer text blob (title + specs + brand + retailer noise);
            the only text column the matchbench mirror provides. Always present.
        source: Originating table (``"a"`` for ``tableA``, ``"b"`` for ``tableB``).
    """

    id: str
    title: str
    source: WdcComputersSource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: the ``title`` blob (the only text field available)."""
        return self.title


load_wdc_computers, load_wdc_computers_pair_splits, WdcComputersBenchmark = (
    make_deepmatcher_benchmark(
        name="wdc_computers",
        schema=WdcComputersSchema,
        dataset_package=_DATASET_PACKAGE,
        table_a=SourceTable(file="tableA.csv", source="a", id_prefix="a"),
        table_b=SourceTable(file="tableB.csv", source="b", id_prefix="b"),
        split_files={"train": "train.csv", "valid": "valid.csv", "test": "test.csv"},
        blocking_k=WDC_COMPUTERS_BLOCKING_K,
        threshold_grid=WDC_COMPUTERS_THRESHOLD_GRID,
        achieved_pc=WDC_COMPUTERS_ACHIEVED_PC,
        gate_met=WDC_COMPUTERS_GATE_MET,
    )
)


def wdc_slice_map(split: str = "test") -> dict[frozenset[str], str]:
    """Tag each pair in ``split`` seen/half_seen/unseen by TRAIN-entity membership.

    An entity (record id) is "seen" if it appears in **any** ``train.csv`` pair
    (positive or negative). Each ``split`` pair is then tagged by how many of its
    two ids are seen: ``"seen"`` (both), ``"unseen"`` (neither), or ``"half_seen"``
    (exactly one). Wave D closes a ``slice_fn`` over this map to show the honest
    seen -> unseen F1 drop at a single fixed threshold.

    The ids are the corpus-prefixed ``<char><int>`` ids the loader factory emits
    (``load_wdc_computers_pair_splits`` remaps raw split ids to the same
    ``a``/``b``-prefixed ids the corpus uses), so a key here matches what a caller
    gets from candidate pairs / :func:`load_wdc_computers`.

    Args:
        split: Which fixed split to tag (``"train"`` / ``"valid"`` / ``"test"``).

    Returns:
        Mapping of each split pair (as a ``frozenset`` of its two prefixed ids) to
        its ``"seen"`` / ``"half_seen"`` / ``"unseen"`` tag.
    """
    splits = load_wdc_computers_pair_splits()
    train_ids = {rid for (left, right, _label) in splits["train"] for rid in (left, right)}
    slice_map: dict[frozenset[str], str] = {}
    for left, right, _label in splits[split]:
        n_seen = (left in train_ids) + (right in train_ids)
        tag = "seen" if n_seen == 2 else "unseen" if n_seen == 0 else "half_seen"
        slice_map[frozenset({left, right})] = tag
    return slice_map
