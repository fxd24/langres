"""Tests for the tiny synthetic fixture benchmark + the loader factory end-to-end.

Runs the shared loader contract (``tests/data/_loader_contract.py``) plus a fast,
fully-offline block -> judge -> cluster pipeline: an ``AllPairsBlocker`` (no
embeddings) + a rapidfuzz judge, asserting the pairwise, blocking (incl. Reduction
Ratio) and clustering (incl. Generalized Merge Distance) metrics all compute.

RR / GMD come from Wave A (``feat/eval-metrics-rr-gmd``, PR #89), which is *not*
merged into this base branch — those two assertions are gated on the metrics being
present in ``langres.core.metrics`` and skipped otherwise (everything else runs).
"""

import inspect
import subprocess
import sys

from langres.core import metrics as _metrics
from langres.data.benchmark import complete_partition
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.metrics import calculate_bcubed_metrics, classify_pairs, evaluate_blocking
from langres.core.matchers.rapidfuzz import RapidfuzzMatcher
from langres.data._benchmark_utils import cross_source
from langres.data.tiny_fixture import (
    TinyFixtureBenchmark,
    TinyFixtureSchema,
    load_tiny_fixture_pair_splits,
)
from tests.data._loader_contract import assert_loader_contract

_N_CORPUS = 12
_N_GOLD_PAIRS = 3


def test_tiny_fixture_satisfies_the_loader_contract() -> None:
    assert_loader_contract(
        TinyFixtureBenchmark(),
        expected_corpus_size=_N_CORPUS,
        expected_gold_pairs=_N_GOLD_PAIRS,
    )


def test_load_and_split_stay_faiss_free() -> None:
    """Importing + loading + splitting the fixture pulls no [semantic] stack.

    The loader factory reuses ``langres.data._benchmark_utils``, whose vector-stack
    imports are lazy (inside ``sweep_blocking_k`` only). A fresh subprocess (mirrors
    ``tests/test_import_budget.py``) proves the offline-load invariant the factory /
    tiny-fixture docstrings promise: ``build_blocker`` is the *only* faiss entry
    point, so ``import + load() + split()`` must leave faiss/sentence_transformers
    out of ``sys.modules``.
    """
    script = (
        "import sys; import langres.data.tiny_fixture as tf; "
        "b = tf.TinyFixtureBenchmark(); "
        "corpus, gold_clusters, _ = b.load(); "
        "b.split(corpus, gold_clusters, seed=0); "
        "leaked = [m for m in ['faiss', 'sentence_transformers'] if m in sys.modules]; "
        "assert not leaked, f'offline load/split leaked heavy modules: {leaked}'; "
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"offline-load import-budget check failed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_pair_splits_round_trip_to_prefixed_corpus_ids() -> None:
    splits = load_tiny_fixture_pair_splits()
    assert set(splits) == {"train", "valid", "test"}
    positives = {
        frozenset({left, right})
        for split in splits.values()
        for left, right, label in split
        if label == 1
    }
    assert positives == {frozenset({"a1", "b1"}), frozenset({"a3", "b2"}), frozenset({"a4", "b3"})}
    # Every split id is a corpus-prefixed <char><int> id.
    for split in splits.values():
        for left, right, _label in split:
            assert left.startswith("a")
            assert right.startswith("b")


def test_end_to_end_offline_block_judge_cluster_metrics() -> None:
    """Block (AllPairs) -> judge (rapidfuzz) -> cluster, and every metric computes."""
    benchmark = TinyFixtureBenchmark()
    corpus, gold_clusters, gold_pairs = benchmark.load()
    records = [record.model_dump() for record in corpus]

    # --- Blocking (offline, no embeddings) ---
    blocker: AllPairsBlocker[TinyFixtureSchema] = AllPairsBlocker(schema=TinyFixtureSchema)
    candidates = list(blocker.stream(records))
    assert len(candidates) == _N_CORPUS * (_N_CORPUS - 1) // 2  # 66 all-pairs

    stats = evaluate_blocking(candidates, gold_clusters)
    # All-pairs blocking captures every true match -> perfect Pair-Completeness.
    assert stats.candidate_recall == 1.0
    assert 0.0 <= stats.candidate_precision <= 1.0

    # Reduction Ratio (Wave A) — gated until feat/eval-metrics-rr-gmd merges. The
    # plan threads RR onto evaluate_blocking via n_left/n_right, exposing it as a
    # CandidateStats field.
    if "n_left" in inspect.signature(evaluate_blocking).parameters:
        n_a = sum(1 for r in corpus if r.source == "a")
        n_b = sum(1 for r in corpus if r.source == "b")
        # Cross-source (linkage) RR compares the emitted cross-source candidates
        # against the |A|*|B| space, so restrict to cross-source pairs first
        # (as the real loaders do). Feeding all-pairs — which includes the
        # n_a(n_a-1)/2 + n_b(n_b-1)/2 same-source pairs — against the |A|*|B|
        # denominator would push the ratio above 1 and RR negative.
        cross_candidates = cross_source(candidates)
        stats_rr = evaluate_blocking(cross_candidates, gold_clusters, n_left=n_a, n_right=n_b)
        assert stats_rr.reduction_ratio is not None
        assert 0.0 <= stats_rr.reduction_ratio <= 1.0

    # --- Judge (rapidfuzz over name; offline) ---
    judge: RapidfuzzMatcher[TinyFixtureSchema] = RapidfuzzMatcher(
        field_extractors={"name": (lambda entity: entity.name, 1.0)},
        algorithm="token_set_ratio",
    )
    judgements = list(judge.forward(iter(candidates)))
    assert len(judgements) == len(candidates)
    assert all(0.0 <= j.score <= 1.0 for j in judgements)

    # --- Pairwise metrics ---
    pair_metrics = classify_pairs(judgements, gold_pairs, threshold=0.5)
    assert 0.0 <= pair_metrics.precision <= 1.0
    assert 0.0 <= pair_metrics.recall <= 1.0
    assert 0.0 <= pair_metrics.f1 <= 1.0

    # --- Clustering metrics ---
    predicted = Clusterer(threshold=0.5).cluster(iter(judgements))
    completed = complete_partition(predicted, [record.id for record in corpus])
    bcubed = calculate_bcubed_metrics(completed, gold_clusters)
    assert 0.0 <= bcubed["f1"] <= 1.0

    # Generalized Merge Distance (Wave A) — gated until feat/eval-metrics-rr-gmd
    # merges. Signature per the plan: generalized_merge_distance(predicted, gold).
    gmd_fn = getattr(_metrics, "generalized_merge_distance", None)
    if gmd_fn is not None:
        assert gmd_fn(completed, gold_clusters) >= 0.0
