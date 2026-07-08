"""Fast loader-contract test for the DBLP-ACM benchmark (Wave C).

Runs the shared DeepMatcher loader contract (``tests/data/_loader_contract.py``)
against ``DblpAcmBenchmark``, pinning the vendored corpus / gold-pair counts as
evidence. The contract check is fast and deterministic (CSV parse + id-scheme +
closed-world partition + leakage-free split + gold counts, no embeddings); any
blocking / Pair-Completeness measurement lives in the pinned constants (measured
once, recorded in ``langres.data.dblp_acm``), not in this per-PR test.
"""

import pytest

from langres.data import _benchmark_utils as _bu
from langres.data.dblp_acm import (
    DBLP_ACM_ACHIEVED_PC,
    DBLP_ACM_BLOCKING_K,
    DblpAcmBenchmark,
    DblpAcmSchema,
    load_dblp_acm,
)
from tests.data._loader_contract import assert_loader_contract

#: Vendored counts (see ``src/langres/data/datasets/dblp_acm/ATTRIBUTION.md``):
#: 2616 DBLP + 2294 ACM records; 2220 strictly-1:1 cross-source gold pairs.
_N_CORPUS = 4910
_N_GOLD_PAIRS = 2220


def test_dblp_acm_satisfies_the_loader_contract() -> None:
    assert_loader_contract(
        DblpAcmBenchmark(),
        expected_corpus_size=_N_CORPUS,
        expected_gold_pairs=_N_GOLD_PAIRS,
    )


@pytest.mark.slow
def test_sweep_blocking_k_pins_documented_pair_completeness() -> None:
    """Re-measure blocking Pair-Completeness at the pinned k, guarding the constant.

    Mirrors ``test_amazon_google.py``: runs the real embedding sweep at
    ``DBLP_ACM_BLOCKING_K`` and asserts the live cross-source PC matches the
    pinned ``DBLP_ACM_ACHIEVED_PC`` within tolerance. Slow (loads MiniLM +
    embeds the corpus) → weekly ``test-full`` only, zero per-PR cost.
    """
    corpus, gold_clusters, _pairs = load_dblp_acm()
    recalls = _bu.sweep_blocking_k(
        corpus,
        gold_clusters,
        DblpAcmSchema,
        text_field="embed_text",
        ks=(DBLP_ACM_BLOCKING_K,),
    )
    assert recalls[DBLP_ACM_BLOCKING_K] == pytest.approx(DBLP_ACM_ACHIEVED_PC, abs=5e-3)
