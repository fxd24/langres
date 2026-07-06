"""M4 DSPyJudge smoke: Signature -> ChainOfThought -> compile -> forward -> eval -> save/load.

The whole flow at **$0** with DSPy's ``DummyLM`` (no key, no network) on a handful
of Amazon-Google test pairs — the plumbing a programmer using langres would drive:

1. build ER candidates from the fixed AG pair split;
2. compile a ``DSPyJudge`` against a tiny gold trainset (``BootstrapFewShot``);
3. score the candidates and grade them with ``evaluate_judge_on_candidates``;
4. save the compiled Resolver, reload it, and assert the reloaded judge scores the
   candidates identically — proving the compiled program round-trips.

Run:
    uv run python examples/research/m4_dspy_judge.py --smoke   # default; $0, DummyLM

The ``DummyLM`` returns canned answers, so the printed JudgePairEval is illustrative
of the *plumbing*, not a real quality signal — the paid first-light (a real model
+ ``MIPROv2``) lands in a later wave.
"""

import argparse
import logging
from pathlib import Path

from dspy import Example
from dspy.utils.dummies import DummyLM

from langres.core import AllPairsBlocker, Clusterer, Resolver
from langres.core.benchmark import evaluate_judge_on_candidates
from langres.core.models import ERCandidate
from langres.core.modules.dspy_judge import DSPyJudge
from langres.data.amazon_google import (
    ProductSchema,
    load_amazon_google,
    load_amazon_google_pair_splits,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("m4_dspy_judge")

#: Compact pair-level threshold grid for the smoke eval.
GRID: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

#: A generous pool of canned DummyLM answers (compile + forward both consume them).
_CANNED_ANSWERS = [
    {"reasoning": "records describe the same product", "match": "True", "match_probability": "0.9"}
] * 200


def _dummy_lm() -> DummyLM:
    """A fresh, deterministic DummyLM (identical answer stream every call)."""
    return DummyLM(list(_CANNED_ANSWERS))


def build_ag_smoke_candidates(
    per_class: int = 6,
) -> tuple[list[ERCandidate[ProductSchema]], set[frozenset[str]]]:
    """Build a balanced handful of ER candidates from the fixed AG test pair split.

    Reuses ``m3_race``'s pattern (corpus by id + fixed pair rows) but skips the
    MiniLM embedding step — ``DSPyJudge`` reads the raw records, not a cosine — so
    the smoke stays fast and $0. Takes ``per_class`` positive and ``per_class``
    negative rows so the trainset carries both labels (bootstrap can collect
    match-demos and the eval has real positives). Returns the candidates plus the
    positive gold pairs.
    """
    corpus, _clusters, _gold = load_amazon_google()
    by_id = {r.id: r for r in corpus}
    all_rows = load_amazon_google_pair_splits()["test"]
    positives = [row for row in all_rows if row[2] == 1][:per_class]
    negatives = [row for row in all_rows if row[2] == 0][:per_class]
    rows = positives + negatives

    candidates: list[ERCandidate[ProductSchema]] = []
    gold: set[frozenset[str]] = set()
    for amazon_id, google_id, label in rows:
        candidates.append(
            ERCandidate(
                left=by_id[amazon_id],
                right=by_id[google_id],
                blocker_name="m4_smoke",
            )
        )
        if label == 1:
            gold.add(frozenset({amazon_id, google_id}))
    return candidates, gold


def _trainset(
    candidates: list[ERCandidate[ProductSchema]], gold: set[frozenset[str]]
) -> list[Example]:
    """Turn candidates into labeled DSPy examples (rendered like ``forward``)."""
    examples: list[Example] = []
    for candidate in candidates:
        is_match = frozenset({candidate.left.id, candidate.right.id}) in gold
        examples.append(
            Example(
                left=candidate.left.model_dump_json(indent=2),
                right=candidate.right.model_dump_json(indent=2),
                match=is_match,
            ).with_inputs("left", "right")
        )
    return examples


def run_smoke() -> None:
    """Run the full zero-spend DSPyJudge plumbing and assert the round-trip is faithful."""
    candidates, gold = build_ag_smoke_candidates()
    logger.info("Built %d AG candidates (%d positive gold pairs).", len(candidates), len(gold))

    # 1) Compile a DSPyJudge against the tiny gold trainset (deterministic under DummyLM).
    judge: DSPyJudge[ProductSchema] = DSPyJudge(lm=_dummy_lm(), entity_noun="product")
    judge.compile(_trainset(candidates, gold), optimizer="bootstrap")
    logger.info("Compiled DSPyJudge (bootstrap). compiled=%s", judge._compiled)

    # 2) Score + grade the candidates at the pair level.
    result, _judgements = evaluate_judge_on_candidates(judge, candidates, gold, GRID)
    logger.info(
        "JudgePairEval: F1=%.3f P=%.3f R=%.3f @ threshold=%.2f (n_judged=%d, cost=$%.4f)",
        result.pair.f1,
        result.pair.precision,
        result.pair.recall,
        result.best_threshold,
        result.n_judged,
        result.cost.usd_total,
    )

    # 3) Save the compiled Resolver, reload it, and re-inject a fresh DummyLM (the LM
    #    is never serialized). The reloaded compiled program must score identically.
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=ProductSchema),
        comparator=None,
        module=judge,
        clusterer=Clusterer(threshold=0.7),
    )
    out_dir = Path("tmp/m4_dspy_smoke_artifact")
    resolver.save(out_dir)
    logger.info("Saved compiled Resolver to %s", out_dir)

    # Re-seed the original judge's LM so both runs start from an identical answer stream.
    judge._lm = _dummy_lm()
    before = [j.score for j in judge.forward(iter(candidates))]

    reloaded = Resolver.load(out_dir)
    assert isinstance(reloaded.module, DSPyJudge)
    reloaded.module._lm = _dummy_lm()
    after = [j.score for j in reloaded.module.forward(iter(candidates))]

    assert before == after, f"reloaded judge diverged:\n before={before}\n after ={after}"
    logger.info("Round-trip OK: reloaded compiled judge scored %d pairs identically.", len(after))
    logger.info("Smoke complete — $0, DummyLM.")


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 DSPyJudge smoke ($0, DummyLM).")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the zero-spend smoke (default when no mode is given).",
    )
    parser.parse_args()
    # Only a zero-spend smoke mode exists on this branch; the paid first-light is a
    # later wave. The flag is accepted for forward-compatibility and clarity.
    run_smoke()


if __name__ == "__main__":
    main()
