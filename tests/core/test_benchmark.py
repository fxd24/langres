"""Tests for the dataset-agnostic benchmark harness (``langres.core.benchmark``).

Covers the budgeted module runner (preflight cap, budget stop, blind-cost guard,
per-call isolation, both cost-key paths), the generic ``run_method`` orchestration
on a fully-deterministic fake benchmark (no embeddings, no LLM), ``BenchmarkTable``
rendering, the ``run_methods`` experiment facade + ``BenchmarkTable.best``/``rank``
structured accessors, and the import-cycle guard (core must not import
``langres.data``).
"""

import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest

import langres.core.benchmark as benchmark_module
from langres.core.benchmark import (
    Benchmark,
    BenchmarkTable,
    BlindCostError,
    BudgetedModuleRunner,
    CostTrack,
    LatencyTrack,
    MethodResult,
    PairTrack,
    PipelineTrack,
    _cost_track,
    complete_partition,
    gold_pairs_from_clusters,
    run_method,
    run_methods,
    tune_threshold_on_train,
)
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.indexes.vector_index import FakeVectorIndex
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.reports import ScoreInspectionReport
from langres.core.resolver import Resolver
from langres.methods import ZERO_SPEND_METHODS

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeModule(Module[CompanySchema]):
    """Yields one judgement per candidate with a controllable cost / failure.

    ``cost_key`` selects which provenance key carries the spend (``cost_usd`` or
    ``llm_cost_usd``); ``boom_ids`` forces a ``RuntimeError`` for a candidate
    whose left id is listed (to exercise per-call isolation).
    """

    def __init__(
        self,
        *,
        cost: float = 0.0,
        cost_key: str = "cost_usd",
        boom_ids: frozenset[str] = frozenset(),
        empty_ids: frozenset[str] = frozenset(),
        blind_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._cost = cost
        self._cost_key = cost_key
        self._boom_ids = boom_ids
        self._empty_ids = empty_ids
        self._blind_ids = blind_ids

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for cand in candidates:
            if cand.left.id in self._blind_ids:
                raise BlindCostError(f"untrackable spend for {cand.left.id}")
            if cand.left.id in self._boom_ids:
                raise RuntimeError(f"boom for {cand.left.id}")
            if cand.left.id in self._empty_ids:
                continue  # yield nothing for this candidate
            yield PairwiseJudgement(
                left_id=cand.left.id,
                right_id=cand.right.id,
                score=1.0,
                score_type="heuristic",
                decision_step="fake",
                provenance={self._cost_key: self._cost},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError  # pragma: no cover — unused by the runner


def _candidates(n: int) -> list[ERCandidate[CompanySchema]]:
    return [
        ERCandidate(
            left=CompanySchema(id=f"l{i}", name=f"L{i}"),
            right=CompanySchema(id=f"r{i}", name=f"R{i}"),
            blocker_name="test",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# BudgetedModuleRunner
# ---------------------------------------------------------------------------


def test_runner_rejects_bad_budgets() -> None:
    module = _FakeModule()
    with pytest.raises(ValueError, match="budgets must be positive"):
        BudgetedModuleRunner(module, budget_usd=0.0)
    with pytest.raises(ValueError, match="must not exceed"):
        BudgetedModuleRunner(module, budget_usd=1.0, budget_soft_usd=2.0)
    with pytest.raises(ValueError, match="worst_case_units_per_pair must be positive"):
        BudgetedModuleRunner(module, worst_case_units_per_pair=0.0)


def test_runner_blind_cost_error_on_zero_price() -> None:
    runner = BudgetedModuleRunner(_FakeModule(), budget_usd=10.0, budget_soft_usd=10.0)
    with pytest.raises(BlindCostError, match="blind"):
        runner.run(_candidates(3), price_per_token_or_pair=0.0)


def test_runner_preflight_cap_truncates_in_input_order() -> None:
    # worst_case_per_pair = 1 * 0.5 = 0.5; floor(1.0 / 0.5) = 2 pairs kept.
    runner = BudgetedModuleRunner(_FakeModule(cost=0.0), budget_usd=1.0, budget_soft_usd=1.0)
    out = runner.run(_candidates(5), price_per_token_or_pair=0.5)
    assert len(out) == 2
    assert runner.dropped_by_cap_count == 3
    assert [j.left_id for j in out] == ["l0", "l1"]


def test_runner_budget_stop_returns_paid_work_before_crossing() -> None:
    # Preflight keeps floor(0.9/0.3)=3; actual cost 0.5/pair exceeds the 0.3
    # worst-case estimate, so the per-pair gate stops after 2 pairs.
    runner = BudgetedModuleRunner(_FakeModule(cost=0.5), budget_usd=1.0, budget_soft_usd=0.9)
    out = runner.run(_candidates(4), price_per_token_or_pair=0.3)
    assert len(out) == 2
    assert runner.total_spent_usd == pytest.approx(1.0)
    assert runner.dropped_by_cap_count == 1  # 4 input -> 3 preflight
    assert runner.labeled_count == 2


def test_runner_tallies_llm_cost_usd_key() -> None:
    # CascadeModule writes ``llm_cost_usd``; the tally must read it as a fallback.
    runner = BudgetedModuleRunner(
        _FakeModule(cost=0.2, cost_key="llm_cost_usd"),
        budget_usd=100.0,
        budget_soft_usd=100.0,
    )
    out = runner.run(_candidates(3), price_per_token_or_pair=0.001)
    assert len(out) == 3
    assert runner.total_spent_usd == pytest.approx(0.6)


def test_runner_tallies_cost_usd_key() -> None:
    runner = BudgetedModuleRunner(
        _FakeModule(cost=0.2, cost_key="cost_usd"),
        budget_usd=100.0,
        budget_soft_usd=100.0,
    )
    runner.run(_candidates(3), price_per_token_or_pair=0.001)
    assert runner.total_spent_usd == pytest.approx(0.6)


def test_runner_skips_when_module_yields_nothing() -> None:
    runner = BudgetedModuleRunner(
        _FakeModule(cost=0.0, empty_ids=frozenset({"l1"})),
        budget_usd=100.0,
        budget_soft_usd=100.0,
    )
    out = runner.run(_candidates(3), price_per_token_or_pair=0.001)
    assert [j.left_id for j in out] == ["l0", "l2"]
    assert runner.skipped_count == 1


def test_runner_isolates_per_call_exceptions() -> None:
    runner = BudgetedModuleRunner(
        _FakeModule(cost=0.0, boom_ids=frozenset({"l1"})),
        budget_usd=100.0,
        budget_soft_usd=100.0,
    )
    out = runner.run(_candidates(3), price_per_token_or_pair=0.001)
    # l1 raised -> skipped; l0 and l2 survive (paid work not lost).
    assert [j.left_id for j in out] == ["l0", "l2"]
    assert runner.skipped_count == 1
    assert runner.labeled_count == 2


def test_runner_propagates_blind_cost_error_with_partial() -> None:
    # A module that signals untrackable spend must abort the run, not be swallowed
    # as a skip (else the cap keeps accruing unknowable cost) — and the already-paid
    # judgement(s) must survive on exc.partial.
    runner = BudgetedModuleRunner(
        _FakeModule(blind_ids=frozenset({"l1"})),
        budget_usd=100.0,
        budget_soft_usd=100.0,
    )
    with pytest.raises(BlindCostError, match="untrackable spend") as excinfo:
        runner.run(_candidates(3), price_per_token_or_pair=0.001)
    # l0 was scored (and paid for) before l1 aborted -> recoverable on partial.
    assert [j.left_id for j in excinfo.value.partial] == ["l0"]


# ---------------------------------------------------------------------------
# Fake benchmark + run_method
# ---------------------------------------------------------------------------


class _FakeBenchmark(Benchmark[CompanySchema]):
    """A tiny deterministic CompanySchema benchmark (two dup clusters, no embeds)."""

    name = "fake"
    threshold_grid = (0.3, 0.5, 0.7, 0.9)

    _CORPUS = [
        CompanySchema(id="c1", name="Acme Corporation", address="1 Main St"),
        CompanySchema(id="c1b", name="Acme Corporation", address="1 Main St"),
        CompanySchema(id="c2", name="Zeta Holdings", address="9 Pine Rd"),
        CompanySchema(id="c3", name="Beta Incorporated", address="2 Oak Ave"),
        CompanySchema(id="c3b", name="Beta Incorporated", address="2 Oak Ave"),
        CompanySchema(id="c4", name="Omega Limited", address="7 Elm Blvd"),
    ]
    _GOLD = [{"c1", "c1b"}, {"c2"}, {"c3", "c3b"}, {"c4"}]

    def load(self) -> tuple[list[CompanySchema], list[set[str]], set[frozenset[str]]]:
        return (
            list(self._CORPUS),
            [set(c) for c in self._GOLD],
            gold_pairs_from_clusters([set(c) for c in self._GOLD]),
        )

    def split(
        self,
        corpus: list[CompanySchema],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[CompanySchema], list[CompanySchema], list[set[str]], list[set[str]]]:
        by_id = {r.id: r for r in corpus}
        train_clusters = [{"c1", "c1b"}, {"c2"}]
        test_clusters = [{"c3", "c3b"}, {"c4"}]
        train = [by_id[i] for c in train_clusters for i in sorted(c)]
        test = [by_id[i] for c in test_clusters for i in sorted(c)]
        return train, test, train_clusters, test_clusters


def _resolver_factory(threshold: float) -> Resolver:
    # Name-dominant weights so identical-name duplicates clear the evidence floor.
    return Resolver.from_schema(
        CompanySchema, threshold=threshold, weights={"name": 0.7, "address": 0.3}
    )


def test_run_method_computes_both_tracks_cost_and_latency() -> None:
    result = run_method(_FakeBenchmark(), _resolver_factory, seed=0)

    assert isinstance(result, MethodResult)
    assert result.dataset == "fake"
    assert result.seed == 0
    assert result.threshold in _FakeBenchmark.threshold_grid
    assert result.method == "weighted_average_judge"

    # Pipeline track: the two identical Beta records merge -> perfect on test.
    assert result.pipeline.bcubed_f1 == pytest.approx(1.0)
    assert result.pipeline.delta_above_floor >= 0.0
    assert result.pipeline.bcubed_f1 >= result.pipeline.sanity_floor_f1

    # Pair track: the one gold test pair (c3/c3b) is recovered.
    assert result.pair.recall == pytest.approx(1.0)
    assert result.pair.pr_curve is not None
    assert len(result.pair.pr_curve) == len(_FakeBenchmark.threshold_grid)

    # Zero-spend method: cost is all zeros, optionals stay empty.
    assert result.cost.usd_total == 0.0
    assert result.cost.usd_per_1k_pairs == 0.0
    assert result.cost.escalation_rate is None
    assert result.cost.llm_calls_per_candidate is None

    # Latency populated; optionals empty.
    assert result.latency.seconds_per_pair >= 0.0
    assert result.latency.throttle_seconds is None


def test_run_method_budget_zero_passes_for_zero_spend() -> None:
    # A zero-spend method must satisfy a $0 ceiling without raising.
    result = run_method(_FakeBenchmark(), _resolver_factory, seed=0, budget=0.0)
    assert result.cost.usd_total == 0.0


def _cost_resolver_factory(threshold: float) -> Resolver:
    # A resolver whose module charges a fixed cost per pair, so the budget guard
    # has spend to bound (no comparator needed — _FakeModule reads raw entities).
    return Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=_FakeModule(cost=0.01),
        clusterer=Clusterer(threshold=threshold),
    )


def test_run_method_raises_when_budget_exceeded() -> None:
    with pytest.raises(ValueError, match="exceeding budget"):
        run_method(_FakeBenchmark(), _cost_resolver_factory, seed=0, budget=0.0)


def test_tune_threshold_on_train_empty_grid_raises() -> None:
    with pytest.raises(ValueError, match="thresholds is empty"):
        tune_threshold_on_train(
            _resolver_factory,
            list(_FakeBenchmark._CORPUS),
            [{"c1", "c1b"}],
            thresholds=[],
        )


def test_generic_tune_picks_argmax_train_f1(monkeypatch: pytest.MonkeyPatch) -> None:
    # Map each candidate threshold to a known train F1; the tuner returns the max.
    f1_by_threshold = {0.3: 0.10, 0.4: 0.50, 0.5: 0.90, 0.6: 0.40}

    def fake_eval(resolver: Resolver, records: object, clusters: object) -> PipelineTrack:
        f1 = f1_by_threshold[resolver.clusterer.threshold]
        return PipelineTrack(
            bcubed_p=0.0,
            bcubed_r=0.0,
            bcubed_f1=f1,
            cluster_pairwise_f1=0.0,
            delta_above_floor=0.0,
            sanity_floor_f1=0.0,
        )

    monkeypatch.setattr(benchmark_module, "evaluate_resolver_bcubed", fake_eval)
    best = tune_threshold_on_train(_resolver_factory, [], [], thresholds=tuple(f1_by_threshold))
    assert best == 0.5


def test_generic_tune_breaks_ties_to_first(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_eval(resolver: Resolver, records: object, clusters: object) -> PipelineTrack:
        return PipelineTrack(
            bcubed_p=0.0,
            bcubed_r=0.0,
            bcubed_f1=0.5,
            cluster_pairwise_f1=0.0,
            delta_above_floor=0.0,
            sanity_floor_f1=0.0,
        )

    monkeypatch.setattr(benchmark_module, "evaluate_resolver_bcubed", fake_eval)
    best = tune_threshold_on_train(_resolver_factory, [], [], thresholds=(0.4, 0.5, 0.6))
    assert best == 0.4


# ---------------------------------------------------------------------------
# BenchmarkTable
# ---------------------------------------------------------------------------


def test_benchmark_table_renders_one_row_per_result() -> None:
    table = BenchmarkTable()
    result = run_method(_FakeBenchmark(), _resolver_factory, seed=0)
    table.add(result)
    md = table.to_markdown()

    lines = md.splitlines()
    assert lines[0].startswith("| method | dataset | seed")
    assert "| --- |" in lines[1]
    assert len([line for line in lines if "| fake |" in line]) == 1
    assert "weighted_average_judge" in md


def test_benchmark_table_empty_is_header_only() -> None:
    assert BenchmarkTable().to_markdown().count("\n") == 1  # header + separator


def test_cost_track_empty_judgements_is_zero() -> None:
    # No candidates -> no per-pair division; all spend figures fall back to 0.
    track = _cost_track([])
    assert track.usd_total == 0.0
    assert track.usd_per_1k_pairs == 0.0
    assert track.est_usd_per_100k == 0.0


# ---------------------------------------------------------------------------
# evaluate_judge_on_candidates (direct pair-level judge eval, no blocking)
# ---------------------------------------------------------------------------


class _ScoreModule(Module[CompanySchema]):
    """Yields one judgement per candidate whose score is read from a per-id map.

    Lets a test drive the pair-level threshold sweep deterministically: the score
    map keys on ``left.id`` so each candidate gets a chosen score. ``cost`` is
    written to ``provenance['cost_usd']`` so the cost track is exercised too.
    """

    def __init__(self, scores: dict[str, float], *, cost: float = 0.0) -> None:
        self._scores = scores
        self._cost = cost

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for cand in candidates:
            yield PairwiseJudgement(
                left_id=cand.left.id,
                right_id=cand.right.id,
                score=self._scores[cand.left.id],
                score_type="prob_llm",
                decision_step="fake",
                provenance={"cost_usd": self._cost},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError  # pragma: no cover — unused here


def _labeled_candidates(
    scores: dict[str, float],
) -> tuple[list[ERCandidate[CompanySchema]], set[frozenset[str]]]:
    """Build candidates keyed by id; positive gold pairs are ids starting with 'p'."""
    cands = [
        ERCandidate(
            left=CompanySchema(id=lid, name=lid),
            right=CompanySchema(id=f"r_{lid}", name=lid),
            blocker_name="test",
        )
        for lid in scores
    ]
    gold = {frozenset({lid, f"r_{lid}"}) for lid in scores if lid.startswith("p")}
    return cands, gold


def test_evaluate_judge_on_candidates_picks_best_threshold_and_curve() -> None:
    # Two true matches (p0,p1) scored high; one non-match (n0) scored low.
    scores = {"p0": 0.9, "p1": 0.8, "n0": 0.2}
    cands, gold = _labeled_candidates(scores)
    result, judgements = benchmark_module.evaluate_judge_on_candidates(
        _ScoreModule(scores), cands, gold, grid=(0.5, 0.85)
    )
    assert len(judgements) == 3
    assert result.n_candidates == 3
    assert result.n_judged == 3
    # At threshold 0.5 both matches are caught and the non-match excluded: F1 = 1.0.
    assert result.pair.f1 == pytest.approx(1.0)
    assert result.best_threshold == 0.5
    assert result.pair.pr_curve is not None and len(result.pair.pr_curve) == 2
    assert result.cost.usd_total == 0.0
    assert not result.truncated


def test_evaluate_judge_on_candidates_runs_under_budget_runner() -> None:
    scores = {f"p{i}": 0.9 for i in range(5)}
    cands, gold = _labeled_candidates(scores)
    # worst-case 1 unit * $0.4 = $0.4/pair; floor(0.9/0.4)=2 pairs kept (truncated).
    runner = BudgetedModuleRunner(
        _ScoreModule(scores, cost=0.1), budget_usd=1.0, budget_soft_usd=0.9
    )
    result, judgements = benchmark_module.evaluate_judge_on_candidates(
        _ScoreModule(scores, cost=0.1),
        cands,
        gold,
        grid=(0.5,),
        runner=runner,
        price_per_token_or_pair=0.4,
    )
    assert result.n_candidates == 5
    assert result.n_judged == 2  # preflight cap kept 2
    assert result.truncated
    assert result.cost.usd_total == pytest.approx(0.2)


def test_evaluate_judge_on_candidates_custom_cost_track() -> None:
    # A custom cost_track_fn (e.g. cascade's) is honored.
    scores = {"p0": 0.9}
    cands, gold = _labeled_candidates(scores)

    def _double_cost(js: list[PairwiseJudgement]) -> CostTrack:
        return CostTrack(usd_total=99.0)

    result, _ = benchmark_module.evaluate_judge_on_candidates(
        _ScoreModule(scores, cost=1.0), cands, gold, grid=(0.5,), cost_track_fn=_double_cost
    )
    assert result.cost.usd_total == 99.0


def test_evaluate_judge_on_candidates_handles_empty_candidates() -> None:
    # No candidates -> no judgements -> latency falls back to 0.0 (no div-by-zero).
    result, judgements = benchmark_module.evaluate_judge_on_candidates(
        _ScoreModule({}), [], set(), grid=(0.5,)
    )
    assert judgements == []
    assert result.n_judged == 0
    assert result.latency.seconds_per_pair == 0.0
    assert not result.truncated


def test_evaluate_judge_on_candidates_ignores_gold_outside_candidates() -> None:
    # A gold pair whose candidate was never supplied (a subsample/blocking miss)
    # must not count against the judge: recall is graded only over in-scope pairs.
    # Without this, a 600-pair subsample holding 61 of 234 gold pairs would cap
    # recall at ~0.26 for every method.
    scores = {"p0": 0.9, "n0": 0.2}
    cands, gold = _labeled_candidates(scores)  # gold = {frozenset(p0, r_p0)}
    gold_plus_unseen = gold | {frozenset({"ghost", "r_ghost"})}
    result, _ = benchmark_module.evaluate_judge_on_candidates(
        _ScoreModule(scores), cands, gold_plus_unseen, grid=(0.5,)
    )
    # Only p0 is in scope; it is caught and n0 excluded -> perfect, ghost ignored.
    assert result.pair.recall == pytest.approx(1.0)
    assert result.pair.f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# complete_partition (re-homed to core)
# ---------------------------------------------------------------------------


def test_complete_partition_adds_singletons_for_uncovered_ids() -> None:
    assert complete_partition([{"a", "b"}], ["a", "b", "c"]) == [{"a", "b"}, {"c"}]


# ---------------------------------------------------------------------------
# Import-cycle guard: core.benchmark must not import langres.data
# ---------------------------------------------------------------------------


def test_core_benchmark_does_not_import_langres_data() -> None:
    # A fresh interpreter importing only the harness must not pull in langres.data
    # (the cycle that would break ``import langres`` at import time).
    code = (
        "import langres.core.benchmark, sys; "
        "bad = [m for m in sys.modules if m.startswith('langres.data')]; "
        "assert not bad, bad; print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_import_langres_succeeds() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import langres; print('ok')"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# run_methods experiment facade
# ---------------------------------------------------------------------------


def _company_factory(record: dict[str, Any]) -> CompanySchema:
    return CompanySchema(**{f: record.get(f) for f in CompanySchema.model_fields})


class _FakeBlockingBenchmark(_FakeBenchmark):
    """``_FakeBenchmark`` + the registry ``BlockingBenchmark`` contract.

    Adds ``schema`` + ``blocking_k`` + ``build_blocker`` (a ``FakeVectorIndex``
    blocker — no real embeddings) so ``make_resolver_factory`` can build the
    zero-spend methods on the same tiny corpus. This is exactly the intersection
    ``run_methods`` requires, mirroring the fake in ``tests/test_methods.py``.
    """

    schema = CompanySchema
    blocking_k = 2

    def build_blocker(self, k_neighbors: int) -> VectorBlocker[CompanySchema]:
        return VectorBlocker(
            schema_factory=_company_factory,
            text_field_extractor=lambda e: e.name,
            vector_index=FakeVectorIndex(),
            k_neighbors=k_neighbors,
        )


def test_run_methods_races_zero_spend_methods_one_row_each() -> None:
    table = run_methods(_FakeBlockingBenchmark(), ZERO_SPEND_METHODS, seed=0)

    assert isinstance(table, BenchmarkTable)
    # One populated row per method, all on the same dataset, distinct scorers.
    assert len(table.results) == len(ZERO_SPEND_METHODS)
    assert {r.dataset for r in table.results} == {"fake"}
    assert len({r.method for r in table.results}) == len(ZERO_SPEND_METHODS)
    # Both tracks populate on every row.
    assert all(r.pair.pr_curve is not None for r in table.results)
    assert all(0.0 <= r.pipeline.bcubed_f1 <= 1.0 for r in table.results)
    assert all(r.seed == 0 for r in table.results)


def test_run_methods_budget_zero_asserts_zero_spend() -> None:
    # budget=0.0 is a hard assertion these methods truly spend nothing.
    table = run_methods(_FakeBlockingBenchmark(), ZERO_SPEND_METHODS, seed=0, budget=0.0)
    assert all(r.cost.usd_total == 0.0 for r in table.results)


# ---------------------------------------------------------------------------
# BenchmarkTable.best / rank (structured accessors)
# ---------------------------------------------------------------------------


def _result(
    method: str, *, pair_f1: float, bcubed_f1: float, delta: float = 0.0
) -> MethodResult:
    """A minimal MethodResult carrying only the fields best/rank read."""
    return MethodResult(
        method=method,
        dataset="fake",
        seed=0,
        threshold=0.5,
        pair=PairTrack(precision=pair_f1, recall=pair_f1, f1=pair_f1),
        pipeline=PipelineTrack(
            bcubed_p=bcubed_f1,
            bcubed_r=bcubed_f1,
            bcubed_f1=bcubed_f1,
            cluster_pairwise_f1=bcubed_f1,
            delta_above_floor=delta,
            sanity_floor_f1=0.0,
        ),
        latency=LatencyTrack(seconds_per_pair=0.0),
    )


def _table_with_three() -> BenchmarkTable:
    # pair_f1 winner is 'b'; bcubed_f1 winner is 'a' — orderings differ so `by` matters.
    table = BenchmarkTable()
    table.add(_result("a", pair_f1=0.5, bcubed_f1=0.9))
    table.add(_result("b", pair_f1=0.8, bcubed_f1=0.4))
    table.add(_result("c", pair_f1=0.6, bcubed_f1=0.6))
    return table


def test_best_returns_highest_pair_f1_row_by_default() -> None:
    best = _table_with_three().best()
    assert best is not None
    assert best.method == "b"


def test_best_by_alternate_metric_switches_winner() -> None:
    best = _table_with_three().best(by="bcubed_f1")
    assert best is not None
    assert best.method == "a"


def test_best_by_delta_above_floor() -> None:
    table = BenchmarkTable()
    table.add(_result("a", pair_f1=0.1, bcubed_f1=0.1, delta=0.9))
    table.add(_result("b", pair_f1=0.9, bcubed_f1=0.9, delta=0.1))
    best = table.best(by="delta_above_floor")
    assert best is not None
    assert best.method == "a"


def test_best_empty_table_returns_none() -> None:
    assert BenchmarkTable().best() is None


def test_rank_orders_by_metric_best_first() -> None:
    ranked = _table_with_three().rank(by="pair_f1")
    assert [r.method for r in ranked] == ["b", "c", "a"]


def test_rank_empty_table_is_empty_list() -> None:
    assert BenchmarkTable().rank() == []


def test_best_and_rank_reject_unknown_metric() -> None:
    table = _table_with_three()
    with pytest.raises(ValueError, match="unknown ranking metric"):
        table.best(by="nope")
    with pytest.raises(ValueError, match="unknown ranking metric"):
        table.rank(by="nope")
