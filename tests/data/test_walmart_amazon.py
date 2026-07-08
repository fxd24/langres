"""Fast loader-contract test for the structured Walmart-Amazon benchmark.

Runs the shared DeepMatcher loader contract (``tests/data/_loader_contract.py``)
against ``WalmartAmazonBenchmark``, pinning the vendored corpus / gold-pair counts
as evidence. The contract check is fast (CSV parse + id-scheme + partition + gold
counts, no embeddings); the honest blocking k-sweep is measured out-of-band via
``tmp/measure_walmart_amazon_blocking.py`` and pinned in the loader module.
"""

from langres.data.walmart_amazon import WalmartAmazonBenchmark
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
