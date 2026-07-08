"""Fast loader-contract test for the DBLP-ACM benchmark (Wave C).

Runs the shared DeepMatcher loader contract (``tests/data/_loader_contract.py``)
against ``DblpAcmBenchmark``, pinning the vendored corpus / gold-pair counts as
evidence. The contract check is fast and deterministic (CSV parse + id-scheme +
closed-world partition + leakage-free split + gold counts, no embeddings); any
blocking / Pair-Completeness measurement lives in the pinned constants (measured
once, recorded in ``langres.data.dblp_acm``), not in this per-PR test.
"""

from langres.data.dblp_acm import DblpAcmBenchmark
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
