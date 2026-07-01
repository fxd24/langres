"""M4 experiment loop — the DSPy judge experimentation DX, end-to-end at $0.

This is the script a programmer using langres would write to run a DSPy-judge
experiment. It exercises **both experiment surfaces** on Amazon-Google, entirely
offline and free with DSPy's ``DummyLM`` (no API key, no network):

1. Load the fixed Amazon-Google **pair splits** (train/test candidate sets + gold).
2. Build a ``DSPyJudge(lm=DummyLM(...), model="dummy")`` and ``.compile(...)`` it
   on the train band with ``BootstrapFewShot`` (zero-spend).
3. Evaluate the **compiled** judge on the test band with
   ``evaluate_judge_on_candidates`` — pairwise P/R/F1 at the best-F1 grid
   threshold, judged **once**. This is the SOTA-comparable, blocking-free judge
   surface (see docs/EXPERIMENTS.md).
4. Derive a **data-driven** threshold from the judge's own score distribution
   with ``derive_threshold`` and print it beside a hand-set ``0.5`` (the "kill
   magic constants" demo).
5. Thread a ``SpendMonitor`` through and report **$0.00** cumulative — the
   paid-path budget seam, exercised at zero cost.
6. Race a couple of **cheap zero-spend methods** through ``run_methods`` — the
   full-pipeline (BCubed + pair) race surface — and print the winner.

``DummyLM`` returns canned answers, so the printed judge numbers are illustrative
of the *plumbing*, not a real quality signal (a paid model + MIPROv2 lands in a
later wave). The point is that the whole loop runs green at $0.

Run:
    uv run python examples/m4_experiment_loop.py
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing anything that loads torch /
# sentence-transformers (embedding_cosine in the race pulls MiniLM), so the run is
# deterministic and avoids the macOS libomp duplicate-load crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dspy import Example  # noqa: E402
from dspy.utils.dummies import DummyLM  # noqa: E402

from langres.clients.openrouter import SpendMonitor  # noqa: E402
from langres.core.benchmark import (  # noqa: E402
    evaluate_judge_on_candidates,
    run_methods,
)
from langres.core.calibration import derive_threshold  # noqa: E402
from langres.core.models import ERCandidate, PairwiseJudgement  # noqa: E402
from langres.core.modules.dspy_judge import DSPyJudge  # noqa: E402
from langres.data.amazon_google import (  # noqa: E402
    AmazonGoogleBenchmark,
    ProductSchema,
    load_amazon_google,
    load_amazon_google_pair_splits,
)

#: Pair-level score-threshold grid swept by ``evaluate_judge_on_candidates``.
GRID: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

#: The status-quo hand-set threshold the derived one is compared against.
HAND_SET_THRESHOLD = 0.5

#: A generous pool of canned DummyLM answers — compile AND forward both draw from it.
_CANNED_ANSWERS = [
    {"reasoning": "records describe the same product", "match": "True", "match_probability": "0.9"}
] * 1000


def _dummy_lm() -> DummyLM:
    """A fresh, deterministic DummyLM (identical answer stream on every call)."""
    return DummyLM(list(_CANNED_ANSWERS))


def build_candidates(
    split: str, per_class: int = 8
) -> tuple[list[ERCandidate[ProductSchema]], set[frozenset[str]]]:
    """Build a balanced band of ER candidates from a fixed AG pair split.

    Reuses the corpus-by-id + fixed-pair-rows pattern (no embedding step —
    ``DSPyJudge`` reads the raw records): takes ``per_class`` positive and
    ``per_class`` negative rows so both the trainset and the eval carry both
    labels. Returns the candidates plus the positive gold pairs.
    """
    corpus, _clusters, _gold = load_amazon_google()
    by_id = {record.id: record for record in corpus}
    rows = load_amazon_google_pair_splits()[split]
    positives = [row for row in rows if row[2] == 1][:per_class]
    negatives = [row for row in rows if row[2] == 0][:per_class]

    candidates: list[ERCandidate[ProductSchema]] = []
    gold: set[frozenset[str]] = set()
    for amazon_id, google_id, label in positives + negatives:
        candidates.append(
            ERCandidate(left=by_id[amazon_id], right=by_id[google_id], blocker_name="m4_loop")
        )
        if label == 1:
            gold.add(frozenset({amazon_id, google_id}))
    return candidates, gold


def to_trainset(
    candidates: list[ERCandidate[ProductSchema]], gold: set[frozenset[str]]
) -> list[Example]:
    """Turn labeled candidates into DSPy examples (rendered like ``forward``)."""
    trainset: list[Example] = []
    for candidate in candidates:
        is_match = frozenset({candidate.left.id, candidate.right.id}) in gold
        trainset.append(
            Example(
                left=candidate.left.model_dump_json(indent=2),
                right=candidate.right.model_dump_json(indent=2),
                match=is_match,
            ).with_inputs("left", "right")
        )
    return trainset


def _is_gold(judgement: PairwiseJudgement, gold: set[frozenset[str]]) -> bool:
    """Whether a judgement's pair is a true match (aligns scores with labels)."""
    return frozenset({judgement.left_id, judgement.right_id}) in gold


def main() -> None:
    """Run the whole zero-spend experiment loop and print each surface's output."""
    print("=" * 78)
    print("M4 experiment loop — DSPy judge DX on Amazon-Google, $0 with DummyLM")
    print("=" * 78)

    # --- 1) Load the fixed pair splits (train + test candidate bands). ----------
    train_candidates, train_gold = build_candidates("train")
    test_candidates, test_gold = build_candidates("test")
    print("\n## 1. Pair splits (fixed candidate sets)")
    print(f"- train band: {len(train_candidates)} candidates, {len(train_gold)} gold pairs")
    print(f"- test  band: {len(test_candidates)} candidates, {len(test_gold)} gold pairs")

    # --- 2) Build + compile the DSPy judge (BootstrapFewShot, zero-spend). ------
    judge: DSPyJudge[ProductSchema] = DSPyJudge(
        lm=_dummy_lm(), model="dummy", entity_noun="product"
    )
    judge.compile(to_trainset(train_candidates, train_gold), optimizer="bootstrap")
    print("\n## 2. DSPyJudge compiled")
    print(f"- optimizer=bootstrap  compiled={judge.compiled}  (DummyLM => $0)")

    # --- 3) Evaluate the COMPILED judge on the test band — judged ONCE. ---------
    #     This is the SOTA-comparable, blocking-free judge surface. A paid judge
    #     goes here (optionally under a BudgetedModuleRunner); NEVER through
    #     run_methods, which would rebuild it uncompiled and re-judge per threshold.
    monitor = SpendMonitor(budget_usd=5.0)
    result, judgements = evaluate_judge_on_candidates(judge, test_candidates, test_gold, GRID)
    monitor.add(result.cost.usd_total)  # 5) thread honest spend through the ledger
    monitor.check()
    print("\n## 3. evaluate_judge_on_candidates (compiled judge, judged once)")
    print(
        f"- pairwise  F1={result.pair.f1:.3f}  P={result.pair.precision:.3f}  "
        f"R={result.pair.recall:.3f}  @ best threshold={result.best_threshold:.2f}"
    )
    print(f"- judged {result.n_judged}/{result.n_candidates} pairs  (truncated={result.truncated})")

    # --- 4) Derive a data-driven threshold from the judge's score distribution. -
    scores = [judgement.score for judgement in judgements]
    labels = [_is_gold(judgement, test_gold) for judgement in judgements]
    derived = derive_threshold(scores, labels, method="youden")
    print("\n## 4. derive_threshold — kill the magic constant")
    print(f"- score distribution: min={min(scores):.3f} max={max(scores):.3f} n={len(scores)}")
    print(f"- hand-set threshold: {HAND_SET_THRESHOLD:.3f}  (a guessed constant)")
    print(f"- derived threshold:  {derived:.3f}  (Youden's J off the judge's own scores)")

    # --- 5) Report the budget seam — $0.00 cumulative on the zero-spend path. ---
    print("\n## 5. SpendMonitor (paid-path seam, exercised at $0)")
    print(
        f"- cumulative spend: ${monitor.spent:.2f}  "
        f"(budget ${monitor.budget_usd:.2f}, remaining ${monitor.remaining:.2f})"
    )

    # --- 6) Race cheap zero-spend methods — the full-pipeline race surface. -----
    #     run_methods rebuilds the module per grid threshold, which is fine (and
    #     free) for these cheap methods; budget=0.0 asserts genuine zero spend.
    print("\n## 6. run_methods — full-pipeline race (cheap zero-spend methods)")
    table = run_methods(AmazonGoogleBenchmark(), ["rapidfuzz", "embedding_cosine"], budget=0.0)
    for row in table.results:
        print(
            f"- {row.method:<16} pair-F1={row.pair.f1:.3f}  BCubed-F1={row.pipeline.bcubed_f1:.3f}"
        )
    print(f"- winner (best pair-F1): {table.best().method}")

    print("\nDone — whole loop ran green at $0 (DummyLM, no key, no network).")


if __name__ == "__main__":
    main()
