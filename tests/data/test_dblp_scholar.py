"""Fast loader-contract test for the DBLP-Scholar benchmark (Wave C).

Runs the shared DeepMatcher loader contract (``tests/data/_loader_contract.py``):
CSV parse + id-scheme + closed-world partition + gold-pair consistency +
leakage-free split, with the vendored corpus / gold-pair counts pinned as
evidence. No embeddings — the (slow) Pair-Completeness sweep that pins the
blocking constants lives off the test path (a tmp/ script; the measured sweep is
recorded in ``langres.data.dblp_scholar`` and the ATTRIBUTION).

Pinned counts: 66879 records (2616 DBLP + 64263 Scholar); 13763 gold pairs — the
within-cluster transitive closure of the 5347 raw cross-source positives (the
benchmark is many-to-many, so ``Benchmark.load`` re-derives more pairs than the
raw split positives). See ``datasets/dblp_scholar/ATTRIBUTION.md``.
"""

from langres.data.dblp_scholar import DblpScholarBenchmark
from tests.data._loader_contract import assert_loader_contract

_N_CORPUS = 66879
_N_GOLD_PAIRS = 13763


def test_dblp_scholar_satisfies_the_loader_contract() -> None:
    assert_loader_contract(
        DblpScholarBenchmark(),
        expected_corpus_size=_N_CORPUS,
        expected_gold_pairs=_N_GOLD_PAIRS,
    )
