"""Tests for the WDC-computers benchmark loader + its derived seen/unseen slice.

Runs the shared loader contract (``tests/data/_loader_contract.py``) against the
factory-built ``WdcComputersBenchmark`` and checks the Wave D ``wdc_slice_map``
seen/unseen tagging. Both are fast (CSV parse + partition; no embeddings).
"""

import pytest

from langres.data import _benchmark_utils as _bu
from langres.data.wdc_computers import (
    WDC_COMPUTERS_ACHIEVED_PC,
    WDC_COMPUTERS_BLOCKING_K,
    WdcComputersBenchmark,
    WdcComputersSchema,
    load_wdc_computers,
    wdc_slice_map,
)
from tests.data._loader_contract import assert_loader_contract

#: Pinned as evidence (see ``datasets/wdc_computers/ATTRIBUTION.md``): 2204 tableA
#: + 2443 tableB records, and 1111 transitive-closure within-cluster gold pairs
#: (the many-to-many closed-world partition of 986 pooled positive pairs).
_N_CORPUS = 4647
_N_GOLD_PAIRS = 1111


def test_wdc_computers_satisfies_the_loader_contract() -> None:
    assert_loader_contract(
        WdcComputersBenchmark(),
        expected_corpus_size=_N_CORPUS,
        expected_gold_pairs=_N_GOLD_PAIRS,
    )


def test_wdc_slice_map_tags_test_pairs_seen_and_unseen() -> None:
    """The derived seen/unseen slice is non-empty, well-tagged, and spans the range.

    Every value is a valid tag, and the test split genuinely exhibits both
    ``"seen"`` and ``"unseen"`` pairs (measured: seen=86, half_seen=423,
    unseen=572) — the precondition Wave D's honest seen -> unseen F1-drop demo
    depends on.
    """
    tags = wdc_slice_map("test")
    assert tags, "wdc_slice_map('test') is empty"
    assert set(tags.values()) <= {"seen", "half_seen", "unseen"}, "unexpected slice tag"
    present = set(tags.values())
    assert "seen" in present, "no fully-seen test pairs (Wave D needs both endpoints)"
    assert "unseen" in present, "no fully-unseen test pairs (Wave D needs both endpoints)"


@pytest.mark.slow
def test_sweep_blocking_k_pins_documented_pair_completeness() -> None:
    """Re-measure blocking Pair-Completeness at the pinned k, guarding the constant.

    Mirrors ``test_amazon_google.py``: runs the real embedding sweep at
    ``WDC_COMPUTERS_BLOCKING_K`` and asserts the live cross-source PC matches the
    pinned ``WDC_COMPUTERS_ACHIEVED_PC`` (an honest sub-gate 0.7237 — title-only
    text is hard) within tolerance. Slow (loads MiniLM + embeds the corpus) →
    weekly ``test-full`` only, zero per-PR cost.
    """
    corpus, gold_clusters, _pairs = load_wdc_computers()
    recalls = _bu.sweep_blocking_k(
        corpus,
        gold_clusters,
        WdcComputersSchema,
        text_field="embed_text",
        ks=(WDC_COMPUTERS_BLOCKING_K,),
    )
    assert recalls[WDC_COMPUTERS_BLOCKING_K] == pytest.approx(WDC_COMPUTERS_ACHIEVED_PC, abs=5e-3)
