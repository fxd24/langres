"""Proof: a curated training recipe beats random pairs at an equal label budget.

This is the acceptance demonstration for the data-preparation layer. It asks one
question and answers it with numbers, on a real benchmark, for $0:

    Given a fixed budget of N training pairs drawn from a blocked candidate pool,
    does *curating* those N (balance + hard positives + denoise + augmentation)
    beat drawing them at random?

Why it works -- honestly. A blocked entity-resolution pool is dominated by
non-matches (here an all-pairs blocked pool runs ~1:500 positive:negative). A
random draw of N pairs is therefore almost all negatives and usually contains
*no* matches at all, so the trained matcher never learns what a match looks like
and its held-out F1 collapses. The recipe spends the SAME N-pair budget
deliberately: it mines the hard positives, drops likely-mislabeled pairs,
augments the positives, and balances the negatives to 2:1 -- so the matcher
trains on a set that actually contains matches.

To keep the comparison fair and low-variance, the baseline is the **mean over
several random draws** at the same N (a single lucky draw that happens to catch a
few matches can approach the recipe -- see how many draws the recipe beats). The
recipe and every draw come from the SAME pool and are trained by the SAME
:class:`~langres.core.matchers.random_forest_judge.RandomForestMatcher`, then
scored on the SAME held-out, entity-disjoint test split with the existing
:func:`~langres.core.benchmark.evaluate_judge_on_candidates` (best-F1 threshold on
a fixed grid). No LLM, no network, no spend.

Run it (self-contained -- sets the macOS OpenMP guard itself):
    uv run python examples/recipe_lift_proof.py
"""

import os

# On macOS, importing the abt_buy loader pulls faiss + torch (its default blocker
# is a VectorBlocker); the guard avoids their OpenMP double-load crash. Set before
# any heavy import. This example never runs the VectorBlocker -- it blocks with the
# core AllPairsBlocker -- so it stays $0 and offline.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
from statistics import mean, pstdev

from langres.core.benchmark import evaluate_judge_on_candidates
from langres.core.comparator import StringComparator
from langres.core.finetune import LabeledCandidate
from langres.core.matchers.random_forest_judge import RandomForestMatcher
from langres.core.models import ERCandidate
from langres.core.resolver import Resolver
from langres.data.mining import (
    augment_by_attribute,
    denoise_pairs,
    mine_misclassified_pairs,
    sample_negative_pairs,
)
from langres.data.registry import get_benchmark

BENCHMARK = "abt_buy"
SEED = 0
#: Held-out pair-level F1 threshold grid (0.05 .. 1.00) -- the same operating-point
#: sweep for the recipe and every baseline draw, so neither is handed a better cut.
GRID = [i / 20 for i in range(1, 21)]

#: Pool size knobs. A modest slice of the benchmark keeps the run a few seconds
#: while preserving the heavy imbalance the mechanism needs.
N_TRAIN_CLUSTERS = 80  # matched entity clusters whose records seed the TRAIN pool
N_TEST_CLUSTERS = 60  # entity-disjoint clusters for the held-out TEST pool
N_TRAIN_SINGLETONS = 120  # extra unmatched records -> negatives in the TRAIN pool
N_TEST_SINGLETONS = 120
POS_CAP = 40  # positives the recipe mines/augments (the label budget's match half)
N_BASELINE_DRAWS = 10  # random draws averaged for the honest baseline


def build_pool(
    resolver: Resolver,
    by_id: dict[str, object],
    gold_pairs: set[frozenset[str]],
    ids: list[str],
) -> list[LabeledCandidate]:
    """Block a set of records all-pairs and label each pair against gold.

    ``resolver.candidates`` runs the AllPairsBlocker + StringComparator, so every
    candidate carries the ``comparison`` vector the featurizing miners and the
    RandomForest need. A pair is a positive iff its id-set is a gold match.
    """
    records = [by_id[i].model_dump() for i in ids]  # type: ignore[attr-defined]
    candidates = resolver.candidates(records)
    return [(c, frozenset({c.left.id, c.right.id}) in gold_pairs) for c in candidates]


def assemble_recipe(
    train_pool: list[LabeledCandidate], comparator: StringComparator[object]
) -> tuple[list[LabeledCandidate], int]:
    """Curate the same-budget training set from the imbalanced pool.

    Four Wave-A miners compose into one training set:
      1. ``denoise_pairs``          -- drop likely-mislabeled pairs.
      2. ``mine_misclassified_pairs`` -- the AnyMatch hard positives (out-of-fold).
      3. ``augment_by_attribute``   -- blank one field at a time (missing-field
         robustness); it resets ``comparison`` to ``None``, so re-run the
         comparator over the augmented pairs before they are usable.
      4. ``sample_negative_pairs``  -- balance the negatives to 2:1.

    Returns ``(recipe, n_flagged)`` -- the assembled pairs and how many the
    denoiser dropped.
    """
    clean, flagged = denoise_pairs(train_pool, seed=SEED)
    hard = mine_misclassified_pairs(clean, cap=POS_CAP, seed=SEED)
    augmented = [
        (c.model_copy(update={"comparison": comparator.compare(c.left, c.right)}), label)
        for c, label in augment_by_attribute(hard, cap=POS_CAP, seed=SEED)
    ]
    positives = list(hard) + augmented
    negatives = sample_negative_pairs(
        positives + [pair for pair in clean if not pair[1]], ratio=2.0, seed=SEED
    )
    return positives + negatives, len(flagged)


def fit_and_score(
    labeled: list[LabeledCandidate],
    feature_specs: list[object],
    test_candidates: list[ERCandidate[object]],
    gold_pairs: set[frozenset[str]],
) -> float:
    """Fit a RandomForest on ``labeled`` and return its best-F1 on the held-out set."""
    matcher: RandomForestMatcher[object] = RandomForestMatcher(
        feature_specs=feature_specs,  # type: ignore[arg-type]
        random_state=SEED,
    )
    matcher.fit(iter([c for c, _ in labeled]), [label for _, label in labeled])
    result, _ = evaluate_judge_on_candidates(matcher, test_candidates, gold_pairs, GRID)
    return result.pair.f1


def main() -> None:
    print("Recipe-lift proof: curated pairs vs random pairs at an equal budget")
    print("=" * 70)

    bench = get_benchmark(BENCHMARK)
    corpus, clusters, gold_pairs = bench.load()
    by_id = {record.id: record for record in corpus}
    multi = sorted((sorted(c) for c in clusters if len(c) >= 2), key=lambda c: c[0])
    singletons = sorted(next(iter(c)) for c in clusters if len(c) == 1)

    resolver = Resolver.from_schema(bench.schema)  # AllPairsBlocker + StringComparator
    comparator = resolver.comparator
    feature_specs = list(comparator.feature_specs)

    train_ids = [i for c in multi[:N_TRAIN_CLUSTERS] for i in c] + singletons[:N_TRAIN_SINGLETONS]
    test_slice = multi[N_TRAIN_CLUSTERS : N_TRAIN_CLUSTERS + N_TEST_CLUSTERS]
    test_ids = [i for c in test_slice for i in c] + singletons[
        N_TRAIN_SINGLETONS : N_TRAIN_SINGLETONS + N_TEST_SINGLETONS
    ]
    train_pool = build_pool(resolver, by_id, gold_pairs, train_ids)
    test_candidates = [c for c, _ in build_pool(resolver, by_id, gold_pairs, test_ids)]

    n_pos = sum(1 for _, label in train_pool if label)
    n_neg = len(train_pool) - n_pos
    print(f"\nDataset: {BENCHMARK} (all-pairs blocked)")
    print(
        f"Train pool: {len(train_pool):,} pairs "
        f"({n_pos} positive / {n_neg:,} negative -> 1:{n_neg / max(n_pos, 1):.0f} imbalance)"
    )
    print(f"Held-out test pool: {len(test_candidates):,} candidate pairs (entity-disjoint)")

    recipe, n_flagged = assemble_recipe(train_pool, comparator)
    budget = len(recipe)
    recipe_pos = sum(1 for _, label in recipe if label)
    print(f"\nBudget N = {budget} pairs (the recipe's size; every baseline draws the same N).")
    print(
        f"  recipe   : {recipe_pos} positives (hard-mined + augmented) + "
        f"{budget - recipe_pos} negatives (2:1); denoiser dropped {n_flagged}"
    )

    recipe_f1 = fit_and_score(recipe, feature_specs, test_candidates, gold_pairs)

    baseline_f1s: list[float] = []
    baseline_pos: list[int] = []
    for k in range(N_BASELINE_DRAWS):
        draw = random.Random(1000 + k).sample(train_pool, min(budget, len(train_pool)))
        baseline_pos.append(sum(1 for _, label in draw if label))
        baseline_f1s.append(fit_and_score(draw, feature_specs, test_candidates, gold_pairs))
    avg_draw_pos = mean(baseline_pos)
    baseline_mean = mean(baseline_f1s)
    margin = recipe_f1 - baseline_mean
    beats = sum(1 for f1 in baseline_f1s if recipe_f1 > f1)

    print(
        f"  baseline : ~{avg_draw_pos:.1f} positives on average per random draw "
        f"(the rest negatives)"
    )
    print("\nHeld-out pair F1")
    print("-" * 70)
    print(
        f"  BASELINE (mean of {N_BASELINE_DRAWS} random draws) : "
        f"{baseline_mean:.4f}  (+/- {pstdev(baseline_f1s):.4f}, best draw {max(baseline_f1s):.4f})"
    )
    print(f"  RECIPE  (curated, same N)                : {recipe_f1:.4f}")
    print(f"  MARGIN  (recipe - baseline mean)         : {margin:+.4f}")
    print(f"  Recipe beats {beats}/{N_BASELINE_DRAWS} individual random draws.")

    verdict = "LIFT" if margin > 0 else "NO LIFT"
    print(
        f"\n=> {verdict}: at an equal {budget}-pair budget, curation "
        f"{'beats' if margin > 0 else 'does not beat'} random by {margin:+.4f} F1."
    )
    print(
        "   The lift is the balance + hard positives paying for themselves: a random\n"
        "   draw from a 1:%d pool is mostly non-matches, so the matcher barely sees a\n"
        "   match to learn from. A single lucky draw can approach the recipe, which is\n"
        "   why the honest baseline is the MEAN over draws." % (n_neg / max(n_pos, 1))
    )


if __name__ == "__main__":
    main()
