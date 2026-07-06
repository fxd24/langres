"""W1.1 SelectJudge benchmark: call-count + honest-cost reduction on Amazon-Google.

SelectJudge (``langres.core.modules.select_judge.SelectJudge``) makes ONE LLM
call per anchor GROUP ("which single candidate, if any, matches the anchor?")
instead of one call per PAIR. This script produces the two artifacts the W1.1
exit criterion calls for, both at $0 with DSPy's ``DummyLM`` (no key, no
network):

1. **Harness plumbing proof** -- ``select_judge`` runs end-to-end through
   :func:`~langres.core.benchmark.run_method` on the real Amazon-Google
   dataset, exactly like any other registered method. This is a PLUMBING
   proof, not a quality claim: the injected DummyLM always answers "no
   match", so recall is trivially 0 -- real-model quality replication is
   explicitly deferred to W3 (the paid gate).

2. **Honest call-count + cost-reduction table + group-size distribution**
   -- driven directly against the blocker's NATIVE
   ``VectorBlocker.stream_groups()`` (not the buffered pairwise->group
   default ``GroupwiseModule.forward()`` uses, and NOT through
   ``run_method``/``BudgetedModuleRunner``, which today only ever sees
   size-1 groups -- see ``docs/TECHNICAL_OVERVIEW.md`` section 8 and
   ``SelectJudge``'s own module docstring). This is the only way to measure
   a real, non-size-1 group structure, so it is the only honest source for
   the call-count claim. The group-size distribution is reported alongside
   the ratio (E3): a skewed distribution (e.g. one giant group and many
   singletons) would inflate an averaged reduction ratio without every
   group actually saving calls.

Both halves run on the IDENTICAL test split (``AmazonGoogleBenchmark``,
seed 0) so the two numbers describe the same population.

Run:
    uv run python examples/research/w1_select_judge_benchmark.py
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from dspy.utils.dummies import DummyLM

from langres.core.benchmark import run_method
from langres.core.modules.select_judge import SelectJudge
from langres.data.amazon_google import AmazonGoogleBenchmark
from langres.methods import make_resolver_factory

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("w1_select_judge_benchmark")

SEED = 0
RESULTS_DIR = Path("data/benchmarks/w1")

#: Generously large so any run (train-tuning + test-scoring, or the
#: group-count measurement) never exhausts the canned-answer queue. Content
#: is deliberately generic ("no match") -- it is valid for ANY group
#: regardless of which candidate ids happen to be in it, since an empty
#: selection never references an id. run_method's threshold tuning re-scores
#: TRAIN once per grid point (each re-derived call reuses the SAME injected
#: DummyLM, since make_resolver_factory shares one llm_client across every
#: factory(threshold) call), so the real consumption is a small multiple of
#: (train groups + test groups), not just one pass -- sized with a large
#: margin rather than computed exactly, since an exhausted queue produces
#: misleading "malformed response" select_errors instead of a clean run.
_POOL_SIZE = 300_000


def _no_match_dummy_lm() -> DummyLM:
    """A DummyLM that always selects nothing -- content-agnostic, valid for any group."""
    return DummyLM([{"reasoning": "no match", "selected_ids": "[]"}] * _POOL_SIZE)


def run_harness_plumbing_proof() -> dict[str, Any]:
    """Prove select_judge runs end-to-end through run_method on real Amazon-Google data."""
    bench = AmazonGoogleBenchmark()
    corpus, gold_clusters, _gold_pairs = bench.load()
    _train, test, _train_clusters, test_clusters = bench.split(corpus, gold_clusters, seed=SEED)
    logger.info(
        "Harness plumbing proof: %d test records, %d test gold clusters",
        len(test),
        len(test_clusters),
    )

    factory = make_resolver_factory("select_judge", bench, llm_client=_no_match_dummy_lm())
    result = run_method(bench, factory, seed=SEED)
    logger.info(
        "run_method OK: pair_f1=%.3f bcubed_f1=%.3f usd_total=%.4f",
        result.pair.f1,
        result.pipeline.bcubed_f1,
        result.cost.usd_total,
    )
    return {
        "n_test_records": len(test),
        "pair_f1": result.pair.f1,
        "bcubed_f1": result.pipeline.bcubed_f1,
        "usd_total": result.cost.usd_total,
    }


def measure_call_count_reduction() -> dict[str, Any]:
    """The honest structural call-count/cost table + group-size distribution (E3, finding #11).

    Computed on the SAME test split ``run_harness_plumbing_proof`` scores, so
    the two halves of this benchmark describe the identical population.
    """
    bench = AmazonGoogleBenchmark()
    corpus, gold_clusters, _gold_pairs = bench.load()
    _train, test, _train_clusters, _test_clusters = bench.split(corpus, gold_clusters, seed=SEED)
    records = [r.model_dump() for r in test]

    blocker = bench.build_blocker(bench.blocking_k)
    entities = [blocker.schema_factory(r) for r in records]
    texts = [blocker.text_field_extractor(e) for e in entities]
    blocker.vector_index.create_index(texts)

    # Property check (CEO #14), with SelectJudge's own consumer of
    # stream_groups() in the loop: the group contract stays lossless over
    # pairs -- no dupes, no losses -- with a real GroupwiseModule attached.
    stream_pairs = {frozenset({c.left.id, c.right.id}) for c in blocker.stream(records)}
    groups = list(blocker.stream_groups(records))
    group_pairs = {frozenset({g.anchor.id, member.id}) for g in groups for member in g.members}
    assert group_pairs == stream_pairs, "stream_groups() pairs must equal stream() pairs"

    non_empty_groups = [g for g in groups if g.members]
    group_sizes = [len(g.members) for g in non_empty_groups]
    n_pairs = sum(group_sizes)
    n_groups = len(non_empty_groups)

    judge: SelectJudge[Any] = SelectJudge(lm=_no_match_dummy_lm(), entity_noun="product")
    judgements = list(judge.forward_groups(iter(non_empty_groups)))
    n_select_calls = len(judge._get_lm().history)
    assert n_select_calls == n_groups, "one SelectJudge LLM call per non-empty group"
    assert len(judgements) == n_pairs, "SelectJudge must yield one judgement per pair in scope"

    size_distribution = Counter(group_sizes)
    reduction_ratio = n_pairs / n_select_calls if n_select_calls else float("nan")

    logger.info(
        "Call-count: %d pairs -> naive %d pairwise calls vs select_judge %d group calls "
        "(%.2fx reduction)",
        n_pairs,
        n_pairs,
        n_select_calls,
        reduction_ratio,
    )
    logger.info(
        "Group sizes: min=%d max=%d mean=%.2f over %d non-empty groups (of %d total)",
        min(group_sizes),
        max(group_sizes),
        sum(group_sizes) / len(group_sizes),
        n_groups,
        len(groups),
    )

    return {
        "n_test_records": len(records),
        "n_groups_total": len(groups),
        "n_groups_nonempty": n_groups,
        "n_pairs": n_pairs,
        "n_pairwise_calls_naive": n_pairs,
        "n_select_calls": n_select_calls,
        "call_reduction_ratio": reduction_ratio,
        "group_size_min": min(group_sizes),
        "group_size_max": max(group_sizes),
        "group_size_mean": sum(group_sizes) / len(group_sizes),
        "group_size_distribution": dict(sorted(size_distribution.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="W1.1 SelectJudge call-count benchmark ($0, DummyLM)."
    )
    parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    harness = run_harness_plumbing_proof()
    reduction = measure_call_count_reduction()

    out = {"seed": SEED, "harness_plumbing_proof": harness, "call_count_reduction": reduction}
    out_path = RESULTS_DIR / "select_judge_amazon_google.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
