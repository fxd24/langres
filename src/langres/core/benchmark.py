# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: the benchmark subsystem split out of ``langres.core``.

The old ``langres.core.benchmark`` was two concerns in one 1.7k-line file. They
now live in their proper homes:

- The benchmark **spec** — the "a dataset *is* a benchmark" contract (the
  :class:`~langres.data.benchmark.Benchmark` protocol,
  :class:`~langres.data.benchmark.PairTrack`, ``gold_pairs_from_clusters``,
  ``complete_partition``, ``BlindCostError``, ``DEFAULT_PAIR_GRID``) → the
  import-light :mod:`langres.data.benchmark`, so a dataset can carry its
  benchmark capability without pulling the runner.
- The generic **harness** — race a method into a table, or score a judge on a
  fixed candidate set → the :mod:`langres.benchmarks` package
  (:mod:`langres.benchmarks.runner`, :mod:`langres.benchmarks.judge_eval`).

This shim re-exports both halves so any unrepointed ``from
langres.core.benchmark import X`` keeps resolving. It imports the spec from
``langres.data.benchmark`` DIRECTLY (never routing ``data`` back through
``core``), so no ``core -> data -> core`` cycle is reintroduced. All src/tests
import from the new homes; only demonstrative ``examples/research/*`` still reach
through here, until the W2 sweep deletes this file.
"""

from langres.benchmarks.judge_eval import (  # pragma: no cover
    BudgetedModuleRunner,
    EvaluationTruncatedError,
    JudgePairEval,
    _apply_truncation_policy,
    _budget_exceeded_message,
    _budget_truncation_message,
    _empty_run_message,
    _gold_in_scope,
    _grade_slices,
    _judge_skips_message,
    _NEGLIGIBLE_WORST_CASE_PRICE,
    _truncation_reason,
    evaluate,
    evaluate_judge_on_candidates,
)
from langres.benchmarks.runner import (  # pragma: no cover
    _COST_KEYS,
    _RANKABLE_METRICS,
    BenchmarkTable,
    LatencyTrack,
    MethodResult,
    PipelineTrack,
    _combined_cost_basis,
    _cost_track,
    _judgement_cost,
    _judgement_cost_basis,
    _judgement_usage,
    _pipeline_track,
    _rank_accessor,
    _sum_usage,
    _validate_cut,
    _validate_grid_point,
    _validated_grid,
    evaluate_resolver_bcubed,
    run_method,
    run_methods,
    tune_threshold_on_train,
)
from langres.core.usage import CostBasis, CostTrack  # pragma: no cover
from langres.data.benchmark import (  # pragma: no cover
    DEFAULT_PAIR_GRID,
    Benchmark,
    BlindCostError,
    PairTrack,
    RecordT,
    _Resolvable,
    complete_partition,
    gold_pairs_from_clusters,
)

__all__ = [  # pragma: no cover
    "DEFAULT_PAIR_GRID",
    "Benchmark",
    "BenchmarkTable",
    "BlindCostError",
    "BudgetedModuleRunner",
    "CostBasis",
    "CostTrack",
    "EvaluationTruncatedError",
    "JudgePairEval",
    "LatencyTrack",
    "MethodResult",
    "PairTrack",
    "PipelineTrack",
    "RecordT",
    "_COST_KEYS",
    "_NEGLIGIBLE_WORST_CASE_PRICE",
    "_RANKABLE_METRICS",
    "_Resolvable",
    "_apply_truncation_policy",
    "_budget_exceeded_message",
    "_budget_truncation_message",
    "_combined_cost_basis",
    "_cost_track",
    "_empty_run_message",
    "_gold_in_scope",
    "_grade_slices",
    "_judge_skips_message",
    "_judgement_cost",
    "_judgement_cost_basis",
    "_judgement_usage",
    "_pipeline_track",
    "_rank_accessor",
    "_sum_usage",
    "_truncation_reason",
    "_validate_cut",
    "_validate_grid_point",
    "_validated_grid",
    "complete_partition",
    "evaluate",
    "evaluate_judge_on_candidates",
    "evaluate_resolver_bcubed",
    "gold_pairs_from_clusters",
    "run_method",
    "run_methods",
    "tune_threshold_on_train",
]
