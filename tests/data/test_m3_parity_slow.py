"""M3 Wave-1 parity gate: the refactored harness reproduces the M2 baseline.

Runs the full Fodors-Zagat pipeline through the *new* dataset-agnostic
``run_method`` harness (``FodorsZagatBenchmark`` + ``build_restaurant_resolver``)
with real MiniLM embeddings and ZERO spend, and asserts it reproduces the M2
held-out numbers exactly: seed=0, threshold tuned to 0.8, BCubed P/R/F1 =
0.991 / 0.969 / 0.980, merge-nothing floor = 0.932.

This is the no-behavior-change guard for the M2 -> M3 extraction. Marked
``slow`` (it embeds the corpus) but RUNS in CI.
"""

import pytest

from langres.core.benchmark import run_method
from langres.data.er_benchmarks import (
    FodorsZagatBenchmark,
    build_restaurant_resolver,
)

# Pinned M2 baseline (see memory: M2 completion, PR #42 — held-out BCubed F1
# 0.980, floor 0.932 vs true ground truth). abs tol absorbs cross-machine
# FAISS/embedding nondeterminism.
_ATOL = 0.005


@pytest.mark.slow
def test_run_method_reproduces_m2_baseline() -> None:
    result = run_method(
        FodorsZagatBenchmark(),
        build_restaurant_resolver,
        seed=0,
    )

    # Threshold tuned on TRAIN lands at the M2 value.
    assert result.threshold == pytest.approx(0.8)

    # Pipeline track == the M2 held-out BCubed numbers.
    assert result.pipeline.bcubed_p == pytest.approx(0.991, abs=_ATOL)
    assert result.pipeline.bcubed_r == pytest.approx(0.969, abs=_ATOL)
    assert result.pipeline.bcubed_f1 == pytest.approx(0.980, abs=_ATOL)
    assert result.pipeline.sanity_floor_f1 == pytest.approx(0.932, abs=_ATOL)

    # The baseline carries signal over merging nothing.
    assert result.pipeline.delta_above_floor > 0.0
    assert result.pipeline.bcubed_f1 > result.pipeline.sanity_floor_f1

    # Zero-spend: no cost recorded.
    assert result.cost.usd_total == 0.0
