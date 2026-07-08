"""Fast loader-contract test for the structured Walmart-Amazon benchmark.

Runs the shared DeepMatcher loader contract (``tests/data/_loader_contract.py``)
against ``WalmartAmazonBenchmark``, pinning the vendored corpus / gold-pair counts
as evidence. The contract check is fast (CSV parse + id-scheme + partition + gold
counts, no embeddings); the honest blocking k-sweep is measured out-of-band via
``tmp/measure_walmart_amazon_blocking.py`` and pinned in the loader module.
"""

import pytest

from langres.data import _benchmark_utils as _bu
from langres.data.walmart_amazon import (
    WALMART_AMAZON_ACHIEVED_PC,
    WALMART_AMAZON_BLOCKING_K,
    WalmartAmazonBenchmark,
    WalmartAmazonSchema,
    load_walmart_amazon,
)
from tests.data._loader_contract import assert_loader_contract

#: Vendored corpus size (2554 Walmart + 22074 Amazon) — see ATTRIBUTION.md.
_N_CORPUS = 24628
#: Within-cluster gold pairs: the transitive closure of the 962 positive labels
#: over the connected-components partition (846 match components) — see
#: ATTRIBUTION.md.
_N_GOLD_PAIRS = 1092


def test_walmart_amazon_satisfies_the_loader_contract() -> None:
    assert_loader_contract(
        WalmartAmazonBenchmark(),
        expected_corpus_size=_N_CORPUS,
        expected_gold_pairs=_N_GOLD_PAIRS,
    )


@pytest.mark.slow
def test_sweep_blocking_k_pins_documented_pair_completeness() -> None:
    """Re-measure blocking Pair-Completeness at the pinned k, guarding the constant.

    Mirrors ``test_amazon_google.py``: runs the real embedding sweep at
    ``WALMART_AMAZON_BLOCKING_K`` and asserts the live cross-source PC matches the
    pinned ``WALMART_AMAZON_ACHIEVED_PC`` (an honest sub-gate 0.8773) within
    tolerance. Slow (loads MiniLM + embeds the 24628-record corpus) → weekly
    ``test-full`` only, zero per-PR cost.
    """
    corpus, gold_clusters, _pairs = load_walmart_amazon()
    recalls = _bu.sweep_blocking_k(
        corpus,
        gold_clusters,
        WalmartAmazonSchema,
        text_field="embed_text",
        ks=(WALMART_AMAZON_BLOCKING_K,),
    )
    assert recalls[WALMART_AMAZON_BLOCKING_K] == pytest.approx(WALMART_AMAZON_ACHIEVED_PC, abs=5e-3)
