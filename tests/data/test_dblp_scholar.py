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

import pytest

from langres.data import _benchmark_utils as _bu
from langres.data.dblp_scholar import (
    DBLP_SCHOLAR_ACHIEVED_PC,
    DBLP_SCHOLAR_BLOCKING_K,
    DblpScholarBenchmark,
    DblpScholarSchema,
    load_dblp_scholar,
)
from tests.data._loader_contract import assert_loader_contract

_N_CORPUS = 66879
_N_GOLD_PAIRS = 13763


def test_dblp_scholar_satisfies_the_loader_contract() -> None:
    assert_loader_contract(
        DblpScholarBenchmark(),
        expected_corpus_size=_N_CORPUS,
        expected_gold_pairs=_N_GOLD_PAIRS,
    )


@pytest.mark.slow
def test_sweep_blocking_k_pins_documented_pair_completeness() -> None:
    """Re-measure blocking Pair-Completeness at the pinned k, guarding the constant.

    Mirrors ``test_amazon_google.py`` but asserts only the SINGLE pinned
    ``DBLP_SCHOLAR_BLOCKING_K`` point — embedding the full 66879-record corpus is
    expensive, so this sweeps one k rather than the full grid. The pinned PC is an
    honest 0.3945 (capped near the ~0.3977 many-to-many artifact ceiling; the low
    number is a metric-definition ceiling, not a blocking failure — see the loader
    module). Slow → weekly ``test-full`` only.

    macOS local run: prefix with ``OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=1``
    (or ``uv run --env-file .env pytest``) to avoid the libomp duplicate-runtime
    abort documented in ``docs/FRICTION_LOG.md``.
    """
    corpus, gold_clusters, _pairs = load_dblp_scholar()
    recalls = _bu.sweep_blocking_k(
        corpus,
        gold_clusters,
        DblpScholarSchema,
        text_field="embed_text",
        ks=(DBLP_SCHOLAR_BLOCKING_K,),
    )
    assert recalls[DBLP_SCHOLAR_BLOCKING_K] == pytest.approx(DBLP_SCHOLAR_ACHIEVED_PC, abs=5e-3)
