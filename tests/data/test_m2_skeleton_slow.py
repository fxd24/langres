"""M2 walking skeleton — full real-embedding end-to-end regression gate (slow).

Runs the complete M2 pipeline on the real Fodors-Zagat corpus with real MiniLM
embeddings and ZERO spend (no LLM): load -> split (seed=0) -> tune threshold on
TRAIN -> evaluate on the held-out TEST split against the TRUE ``perfectMapping``
ground truth.

This mirrors ``examples/research/m2_walking_skeleton_fodors_zagat.py``. It is marked
``slow`` (it embeds the corpus) but RUNS in CI. The F1 floor below is an
INFORMATIONAL / regression gate pinned from the first real run (measured BCubed
F1 = 0.9799), NOT a quality bar — it exists to catch a regression, not to assert
the baseline is "good enough". M3 is what improves the baseline.
"""

import pytest

from langres.data.er_benchmarks import (
    RECALL_GATE,
    build_restaurant_resolver,
    evaluate_resolver_bcubed,
    load_fodors_zagat,
    split_restaurant_corpus,
    tune_threshold_on_train,
)

# Pinned from the first real run (measured F1 = 0.9799), set ~3 points below as a
# regression gate with margin for cross-machine embedding/FAISS nondeterminism.
# This is informational only — it is NOT the M2 quality bar.
F1_REGRESSION_FLOOR = 0.95


@pytest.mark.slow
def test_m2_skeleton_held_out_bcubed_regression_gate() -> None:
    corpus, gold_clusters = load_fodors_zagat()
    train_records, test_records, train_clusters, test_clusters = split_restaurant_corpus(
        corpus, gold_clusters, test_size=0.3, seed=0
    )

    # Tune on TRAIN only (no test leakage), then score the held-out TEST split.
    best_threshold = tune_threshold_on_train(train_records, train_clusters)
    resolver = build_restaurant_resolver(best_threshold)
    result = evaluate_resolver_bcubed(resolver, test_records, test_clusters)

    # Regression gate (informational, not a quality bar).
    assert result.f1 >= F1_REGRESSION_FLOOR
    # The baseline must carry signal — beat the all-singletons floor.
    assert result.f1 > result.sanity_floor_f1
    # Blocking must surface the true matches it is asked to judge.
    assert result.pair_completeness >= RECALL_GATE
    # Sanity: all reported scores are valid probabilities.
    for value in (result.precision, result.recall, result.f1):
        assert 0.0 <= value <= 1.0
