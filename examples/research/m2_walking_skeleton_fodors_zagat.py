"""M2 walking skeleton — held-out BCubed baseline on Fodors-Zagat.

Runs the full M2 pipeline end-to-end, deterministically and with ZERO spend
(real MiniLM embeddings for blocking; the WeightedAverageMatcher is a zero-cost
feature-bag scorer — no LLM):

    load -> split (seed=0) -> tune threshold on TRAIN -> build resolver at the
    chosen threshold -> evaluate on the held-out TEST split.

It reports the headline BCubed Precision/Recall/F1 against the dataset's TRUE
closed-world ``perfectMapping`` ground truth (NOT the M1 teacher labels), plus
three honesty diagnostics: the chosen threshold (tuned on train only — no test
leakage), the all-singletons sanity floor (the score of merging nothing), and
the test-split cross-source blocking Pair-Completeness (which caps recall).

Run:
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=1 \\
        uv run python examples/research/m2_walking_skeleton_fodors_zagat.py
"""

from langres.data.er_benchmarks import (
    DEFAULT_THRESHOLD_GRID,
    build_restaurant_resolver,
    evaluate_resolver_bcubed,
    load_fodors_zagat,
    split_restaurant_corpus,
    tune_threshold_on_train,
)

SEED = 0
TEST_SIZE = 0.3


def main() -> None:
    """Run the deterministic, zero-spend M2 skeleton and print the baseline."""
    print("=" * 64)
    print("M2 walking skeleton — Fodors-Zagat held-out BCubed baseline")
    print("=" * 64)

    # 1. Load the complete closed-world corpus + true ground-truth partition.
    corpus, gold_clusters = load_fodors_zagat()
    print(f"\nCorpus: {len(corpus)} records, {len(gold_clusters)} gold clusters")

    # 2. Leakage-free stratified split (whole match clusters kept on one side).
    train_records, test_records, train_clusters, test_clusters = split_restaurant_corpus(
        corpus, gold_clusters, test_size=TEST_SIZE, seed=SEED
    )
    print(
        f"Split (seed={SEED}, test_size={TEST_SIZE}): "
        f"{len(train_records)} train / {len(test_records)} test records"
    )

    # 3. Tune the Clusterer threshold on TRAIN only (test is never touched here).
    print(f"\nTuning threshold on TRAIN over {list(DEFAULT_THRESHOLD_GRID)} …")
    best_threshold = tune_threshold_on_train(train_records, train_clusters)
    print(f"Chosen threshold (best TRAIN BCubed F1): {best_threshold}")

    # 4. Build the resolver at the chosen threshold and score on held-out TEST.
    resolver = build_restaurant_resolver(best_threshold)
    result = evaluate_resolver_bcubed(resolver, test_records, test_clusters)

    print("\n" + "-" * 64)
    print("Held-out TEST BCubed (vs TRUE perfectMapping ground truth)")
    print("-" * 64)
    print(f"  BCubed Precision        : {result.precision:.4f}")
    print(f"  BCubed Recall           : {result.recall:.4f}")
    print(f"  BCubed F1               : {result.f1:.4f}")
    print(f"  Sanity floor (F1)       : {result.sanity_floor_f1:.4f}  (merge nothing)")
    print(f"  Blocking Pair-Complete. : {result.pair_completeness:.4f}  (caps recall)")
    print(f"  Chosen threshold        : {best_threshold}")
    print("-" * 64)

    if result.f1 > result.sanity_floor_f1:
        print("\nThe resolver beats the merge-nothing floor (baseline has signal).")
    else:
        print("\nThe resolver does NOT beat the floor — an honest, weak baseline (M3 improves).")


if __name__ == "__main__":
    main()
