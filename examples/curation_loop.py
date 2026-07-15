"""The failure-mode curation loop: profile errors, mine the failing slice, re-profile.

Where ``recipe_lift_proof.py`` shows that curation *helps*, this example shows the
**loop** you run to decide what to curate -- the data-preparation flywheel, end to
end, for $0:

    1. PROFILE  -- run a matcher, log its judgements, and build a
       FailureModeSection: which slice of the data do the errors concentrate in?
    2. CURATE   -- mine the hard positives + augment + balance (the Wave-A miners),
       targeting that failure mode.
    3. RE-PROFILE -- retrain, re-judge, and confirm the targeted slice's error
       rate actually dropped.

On abt_buy, an initial matcher trained on a naive random draw from the imbalanced
pool is too conservative: it misses matches (all its errors are false negatives),
and those misses concentrate on a stable slice -- the cross-source pairs (where
every true match lives) and the missing-field pairs (where it leans on a field it
does not always have). The failure-mode profile names the worst such slice. The
curation round mines exactly the positives the matcher got wrong and augments them
with blanked fields (missing-field robustness), so the retrained matcher recovers
the misses and that slice's error rate collapses.

To keep the error rates legible, the held-out set is balanced down to ~1:4 (a
1:150 blocked set drowns every slice's error rate in the easy-negative majority);
this is a diagnostic view, not a deployment F1 benchmark. No LLM, no network, no
spend -- the matcher is a RandomForest and the log is a real
:class:`~langres.core.judgement_log.JudgementLog` round-tripped through disk.

Run it (self-contained -- sets the macOS OpenMP guard itself):
    uv run python examples/curation_loop.py
"""

import os

# See recipe_lift_proof.py: importing the abt_buy loader pulls faiss + torch; the
# guard avoids their macOS OpenMP double-load crash. This example blocks with the
# core AllPairsBlocker and never runs the VectorBlocker, so it stays $0 and offline.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
import tempfile
from pathlib import Path

from langres.core.comparator import StringComparator
from langres.core.finetune import LabeledCandidate
from langres.core.judgement_log import JudgementLog, LoggingMatcher
from langres.core.matchers.random_forest_judge import RandomForestMatcher
from langres.core.models import ERCandidate
from langres.core.resolver import Resolver
from langres.data.data_profile import FailureModeSection, FailureSlice, profile_failure_mode
from langres.data.mining import (
    augment_by_attribute,
    denoise_pairs,
    mine_misclassified_pairs,
    sample_negative_pairs,
)
from langres.data.registry import get_benchmark

BENCHMARK = "abt_buy"
SEED = 0
THRESHOLD = 0.5  # fixed deployment operating point (no test-set tuning) for the verdicts
POS_CAP = 40
N_TRAIN_CLUSTERS = 80
N_EVAL_CLUSTERS = 60
N_TRAIN_SINGLETONS = 120
N_EVAL_SINGLETONS = 120
EVAL_NEG_RATIO = 4  # balance the eval set to ~1:4 so slice error rates are legible
INITIAL_BUDGET = 240  # the naive matcher's random training draw


def build_pool(
    resolver: Resolver,
    by_id: dict[str, object],
    gold_pairs: set[frozenset[str]],
    ids: list[str],
) -> list[LabeledCandidate]:
    """Block records all-pairs (comparison attached) and label each pair against gold."""
    records = [by_id[i].model_dump() for i in ids]  # type: ignore[attr-defined]
    candidates = resolver.candidates(records)
    return [(c, frozenset({c.left.id, c.right.id}) in gold_pairs) for c in candidates]


def profile_matcher(
    matcher: RandomForestMatcher[object],
    eval_candidates: list[ERCandidate[object]],
    gold_pairs: set[frozenset[str]],
    records: dict[str, dict[str, object]],
    log_path: Path,
) -> FailureModeSection:
    """Judge the eval set through a real JudgementLog, then profile the failures.

    Wraps the matcher in a :class:`LoggingMatcher` so every verdict is appended to
    a JSONL :class:`JudgementLog` (the flywheel inlet), reads it back, and hands
    the rows + gold + records to :func:`profile_failure_mode` -- exactly the
    log -> harvest -> profile path a production loop follows.
    """
    log = JudgementLog(log_path)
    logged = LoggingMatcher(matcher, log=log, threshold=THRESHOLD)
    list(logged.forward(iter(eval_candidates)))  # drain the stream: writes one row per pair
    section = profile_failure_mode(log.read(), gold_pairs=gold_pairs, records=records)
    assert section is not None  # eval set is non-empty and gold is provided
    return section


def worst_stable_slice(section: FailureModeSection) -> FailureSlice | None:
    """The highest-lift *categorical* slice (field emptiness / source).

    Score bands shift between two matchers (a different score distribution), so
    they are not comparable across the two profiles; the field-emptiness and
    source slices cover the SAME pairs each time, so their error rate is
    apples-to-apples before and after curation.
    """
    stable = [
        s
        for s in section.slices
        if (s.dimension.startswith("empty:") or s.dimension == "source") and s.lift is not None
    ]
    return max(stable, key=lambda s: s.lift or 0.0) if stable else None


def slice_error(section: FailureModeSection, dimension: str, value: str) -> float | None:
    """Look up one slice's error rate in a profile (``None`` if that slice is absent)."""
    for s in section.slices:
        if s.dimension == dimension and s.value == value:
            return s.error_rate
    return None


def curate(
    train_pool: list[LabeledCandidate], comparator: StringComparator[object]
) -> list[LabeledCandidate]:
    """Mine the failing pairs into a targeted training set (the four Wave-A miners).

    Denoise -> mine the hard (misclassified) positives -> augment them with blanked
    fields (missing-field robustness, re-compared) -> balance the negatives to 2:1.
    Hard-positive mining surfaces exactly the matches the matcher got wrong -- the
    false negatives the profile flagged -- so the recipe targets the failure mode.
    """
    clean, _flagged = denoise_pairs(train_pool, seed=SEED)
    hard = mine_misclassified_pairs(clean, cap=POS_CAP, seed=SEED)
    augmented = [
        (c.model_copy(update={"comparison": comparator.compare(c.left, c.right)}), label)
        for c, label in augment_by_attribute(hard, cap=POS_CAP, seed=SEED)
    ]
    positives = list(hard) + augmented
    negatives = sample_negative_pairs(
        positives + [pair for pair in clean if not pair[1]], ratio=2.0, seed=SEED
    )
    return positives + negatives


def fit(
    labeled: list[LabeledCandidate], feature_specs: list[object]
) -> RandomForestMatcher[object]:
    """Fit a RandomForest on the labeled pairs."""
    matcher: RandomForestMatcher[object] = RandomForestMatcher(
        feature_specs=feature_specs,  # type: ignore[arg-type]
        random_state=SEED,
    )
    matcher.fit(iter([c for c, _ in labeled]), [label for _, label in labeled])
    return matcher


def main() -> None:
    print("Failure-mode curation loop: profile -> mine the failing slice -> re-profile")
    print("=" * 76)

    bench = get_benchmark(BENCHMARK)
    corpus, clusters, gold_pairs = bench.load()
    by_id = {record.id: record for record in corpus}
    multi = sorted((sorted(c) for c in clusters if len(c) >= 2), key=lambda c: c[0])
    singletons = sorted(next(iter(c)) for c in clusters if len(c) == 1)

    resolver = Resolver.from_schema(bench.schema)  # AllPairsBlocker + StringComparator
    comparator = resolver.comparator
    feature_specs = list(comparator.feature_specs)

    train_ids = [i for c in multi[:N_TRAIN_CLUSTERS] for i in c] + singletons[:N_TRAIN_SINGLETONS]
    eval_slice = multi[N_TRAIN_CLUSTERS : N_TRAIN_CLUSTERS + N_EVAL_CLUSTERS]
    eval_ids = [i for c in eval_slice for i in c] + singletons[
        N_TRAIN_SINGLETONS : N_TRAIN_SINGLETONS + N_EVAL_SINGLETONS
    ]
    train_pool = build_pool(resolver, by_id, gold_pairs, train_ids)

    # Balance the held-out set to ~1:EVAL_NEG_RATIO so per-slice error rates are not
    # drowned by the easy-negative majority (a diagnostic view, not a deployment F1).
    eval_full = build_pool(resolver, by_id, gold_pairs, eval_ids)
    eval_pos = [p for p in eval_full if p[1]]
    eval_neg = [p for p in eval_full if not p[1]]
    eval_pool = eval_pos + random.Random(SEED).sample(
        eval_neg, min(len(eval_neg), EVAL_NEG_RATIO * len(eval_pos))
    )
    eval_candidates = [c for c, _ in eval_pool]
    records: dict[str, dict[str, object]] = {}
    for candidate, _ in eval_pool:
        records[candidate.left.id] = candidate.left.model_dump()
        records[candidate.right.id] = candidate.right.model_dump()
    print(
        f"\nEval set: {len(eval_candidates)} pairs "
        f"({len(eval_pos)} matches, balanced to ~1:{EVAL_NEG_RATIO})"
    )

    with tempfile.TemporaryDirectory(prefix="langres-curation-") as tmp:
        log_dir = Path(tmp)

        # --- Round 0: naive matcher (random draw from the imbalanced pool) -----------
        initial_draw = random.Random(1000).sample(train_pool, min(INITIAL_BUDGET, len(train_pool)))
        initial = fit(initial_draw, feature_specs)
        before = profile_matcher(
            initial, eval_candidates, gold_pairs, records, log_dir / "before.jsonl"
        )
        print(f"\n[before] naive matcher trained on {len(initial_draw)} random pairs")
        print(
            f"  overall error rate {_pct(before.error_rate)}  "
            f"(FP={before.n_false_positive}, FN={before.n_false_negative})"
        )
        target = worst_stable_slice(before)
        if target is None:
            print("  no stable failing slice surfaced; nothing to target.")
            return
        print(
            f"  worst slice: {target.dimension} = '{target.value}'  "
            f"error rate {_pct(target.error_rate)} (lift {target.lift:.2f}, {target.n} pairs)"
        )

        # --- Curate targeting the failure, retrain, re-profile -----------------------
        recipe = curate(train_pool, comparator)
        curated = fit(recipe, feature_specs)
        after = profile_matcher(
            curated, eval_candidates, gold_pairs, records, log_dir / "after.jsonl"
        )
        print(f"\n[after] curated matcher trained on {len(recipe)} mined+augmented+balanced pairs")
        print(
            f"  overall error rate {_pct(after.error_rate)}  "
            f"(FP={after.n_false_positive}, FN={after.n_false_negative})"
        )
        after_rate = slice_error(after, target.dimension, target.value)
        print(
            f"  targeted slice {target.dimension} = '{target.value}': "
            f"error rate {_pct(target.error_rate)} -> {_pct(after_rate)}"
        )

    dropped = (
        after_rate is not None and target.error_rate is not None and after_rate < target.error_rate
    )
    print(
        f"\n=> The targeted slice's error rate {'DROPPED' if dropped else 'did not drop'} "
        f"({_pct(target.error_rate)} -> {_pct(after_rate)}); "
        f"overall {_pct(before.error_rate)} -> {_pct(after.error_rate)}, "
        f"false negatives {before.n_false_negative} -> {after.n_false_negative}."
    )
    print(
        "   The naive matcher missed matches (its errors are false negatives); mining\n"
        "   those misses back in -- as hard positives + missing-field augmentations --\n"
        "   is the curation round the failure-mode profile told us to run."
    )


def _pct(value: float | None) -> str:
    """A rate as a percentage, or ``n/a`` when the slice was absent."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


if __name__ == "__main__":
    main()
