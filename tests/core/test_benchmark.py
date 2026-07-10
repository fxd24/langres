"""Tests for the dataset-agnostic benchmark harness (``langres.core.benchmark``).

Covers the budgeted module runner (preflight cap, budget stop, blind-cost guard,
per-call isolation, both cost-key paths), the generic ``run_method`` orchestration
on a fully-deterministic fake benchmark (no embeddings, no LLM), ``BenchmarkTable``
rendering, the ``run_methods`` experiment facade + ``BenchmarkTable.best``/``rank``
structured accessors, and the import-cycle guard (core must not import
``langres.data``).
"""

import logging
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
from langres.core.groups import ERCandidateGroup
from langres.core.indexes.vector_index import FakeVectorIndex
from langres.core.metrics import classify_pairs
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.module import GroupwiseModule, Module
from langres.core.presets import DEFAULT_BUDGET_USD
from langres.core.reports import ScoreInspectionReport
from langres.core.resolver import Resolver
from langres.core.usage import LLMUsage
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
# BudgetedModuleRunner + GroupwiseModule: documented group-atomicity (E5).
#
# BudgetedModuleRunner._score_one() calls ``module.forward(iter([candidate]))``
# ONE candidate at a time (by design -- see its docstring's "per-call
# resilience" point). A GroupwiseModule's forward() derives groups from
# whatever pairwise stream it's given, so under the runner each call sees
# exactly one candidate -> one trivial, size-1 group. This means the
# atomicity guarantee ("never split a group mid-call") holds TRIVIALLY today
# -- there is never more than one pair per call, so there is nothing to
# split -- but it also means the runner does NOT yet amortize a real multi-
# pair group into a single priced call. That cost-saving integration is
# deferred to W1.1 (the first concrete GroupwiseModule, SelectJudge), which
# will extend the runner (or add a group-aware runner) to pre-flight whole
# groups. This test pins down and documents the current, correct-but-not-
# yet-optimized behavior so a future change is a deliberate, measured one.
# ---------------------------------------------------------------------------


class _CountingGroupwiseModule(GroupwiseModule[CompanySchema]):
    """Records every forward_groups() call's groups, for atomicity inspection."""

    def __init__(self) -> None:
        self.calls: list[list[ERCandidateGroup[CompanySchema]]] = []

    def forward_groups(
        self, groups: Iterator[ERCandidateGroup[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        materialized = list(groups)
        self.calls.append(materialized)
        for group in materialized:
            for member in group.members:
                yield PairwiseJudgement(
                    left_id=group.anchor.id,
                    right_id=member.id,
                    score=1.0,
                    score_type="prob_group_llm",
                    decision_step="test",
                    provenance={"cost_usd": 0.01},
                )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError  # pragma: no cover — unused by the runner


def test_runner_never_splits_a_group_mid_call_but_also_never_batches_one() -> None:
    """Documents E5's atomicity guarantee under today's per-candidate runner loop.

    Three candidates share the same left.id (would form ONE group of size 3
    if handed to forward() together) but the runner isolates each candidate
    into its own call -> forward_groups() is invoked 3 times, each with a
    single size-1 group. No group is ever split across calls (each call's
    group is already maximal for what it was given), but no batching happens
    either -- see the module docstring above for why, and the deferral.
    """
    module = _CountingGroupwiseModule()
    runner = BudgetedModuleRunner(module, budget_usd=100.0, budget_soft_usd=100.0)
    candidates = [
        ERCandidate(
            left=CompanySchema(id="anchor", name="Anchor"),
            right=CompanySchema(id=f"m{i}", name=f"Member {i}"),
            blocker_name="test",
        )
        for i in range(3)
    ]

    out = runner.run(candidates, price_per_token_or_pair=0.01)

    assert len(out) == 3
    assert len(module.calls) == 3  # one call per candidate, not one call for the group
    assert all(len(call) == 1 and len(call[0].members) == 1 for call in module.calls)


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
    assert track.usage == LLMUsage()
    assert track.cost_basis == "none"
    assert track.cost_is_real is False


def test_cost_track_backward_compatible_construction_still_works() -> None:
    # ``CostTrack`` is a plain (non-frozen) BaseModel: the new fields are
    # additive-with-defaults, so an existing call site passing only usd_total
    # (see test_evaluate_judge_on_candidates_custom_cost_track below) must not break.
    track = CostTrack(usd_total=99.0)
    assert track.usd_total == 99.0
    assert track.usage == LLMUsage()
    assert track.cost_basis == "none"


# ---------------------------------------------------------------------------
# CostTrack.usage / cost_basis (Task 3: tokens are the fact, dollars derived)
# ---------------------------------------------------------------------------


def _llm_judgement(
    *, lid: str = "a", rid: str = "b", cost_usd: float, cost_is_real: bool, tokens: int = 10
) -> PairwiseJudgement:
    """A judgement shaped like LLMJudge's / DSPyJudge's real provenance."""
    usage = LLMUsage(input_tokens=tokens, output_tokens=tokens, model="m")
    return PairwiseJudgement(
        left_id=lid,
        right_id=rid,
        score=0.9,
        score_type="prob_llm",
        decision_step="fake",
        provenance={
            "cost_usd": cost_usd,
            "cost_is_real": cost_is_real,
            "usage": usage.model_dump(),
        },
    )


def test_cost_track_usage_sums_token_vectors_across_judgements() -> None:
    j1 = _llm_judgement(cost_usd=0.01, cost_is_real=True, tokens=10)
    j2 = _llm_judgement(lid="c", rid="d", cost_usd=0.02, cost_is_real=True, tokens=20)
    track = _cost_track([j1, j2])
    assert track.usage.input_tokens == 30
    assert track.usage.output_tokens == 30
    assert track.cost_basis == "real"
    assert track.cost_is_real is True


def test_cost_track_basis_estimated_when_cost_is_real_false() -> None:
    # LLMJudge's litellm-estimate fallback, or DSPyJudge's normal (non-error)
    # self-reported cost -- neither claims to be the real billed amount.
    j = _llm_judgement(cost_usd=0.01, cost_is_real=False)
    track = _cost_track([j])
    assert track.cost_basis == "estimated"
    assert track.cost_is_real is False


def test_cost_track_basis_untracked_for_dspy_parse_error_path() -> None:
    # DSPyJudge's billed-but-unparseable call: cost_untracked=True regardless of
    # whether cost_usd/cost_is_real are also present.
    j = PairwiseJudgement(
        left_id="a",
        right_id="b",
        score=0.0,
        score_type="prob_llm",
        decision_step="dspy_parse_error",
        provenance={"cost_usd": 0.0, "cost_untracked": True, "parse_error": True},
    )
    track = _cost_track([j])
    assert track.cost_basis == "untracked"
    assert track.cost_is_real is False


def test_cost_track_basis_none_for_judge_with_no_cost_concept() -> None:
    # string/embedding judges never write cost_usd at all.
    j = PairwiseJudgement(
        left_id="a",
        right_id="b",
        score=0.8,
        score_type="sim_cos",
        decision_step="embedding",
        provenance={"similarity_score": 0.8},
    )
    track = _cost_track([j])
    assert track.cost_basis == "none"
    assert track.usage == LLMUsage()


def test_cost_track_basis_mixed_when_judgements_disagree() -> None:
    real = _llm_judgement(cost_usd=0.01, cost_is_real=True)
    estimated = _llm_judgement(lid="c", rid="d", cost_usd=0.01, cost_is_real=False)
    free = PairwiseJudgement(
        left_id="e",
        right_id="f",
        score=0.5,
        score_type="heuristic",
        decision_step="string",
        provenance={},
    )
    track = _cost_track([real, estimated, free])
    assert track.cost_basis == "mixed"
    assert track.cost_is_real is False


def test_cost_track_usage_malformed_falls_back_to_zero_vector() -> None:
    # A corrupt/foreign 'usage' payload (e.g. from a future schema or a buggy
    # custom judge) must not crash cost accounting -- observability never flakes
    # evaluation (mirrors usage.py's own _as_int philosophy).
    j = PairwiseJudgement(
        left_id="a",
        right_id="b",
        score=0.9,
        score_type="prob_llm",
        decision_step="fake",
        provenance={"cost_usd": 0.0, "usage": {"input_tokens": "not-a-number"}},
    )
    track = _cost_track([j])
    assert track.usage == LLMUsage()


# ---------------------------------------------------------------------------
# evaluate_judge_on_candidates (direct pair-level judge eval, no blocking)
# ---------------------------------------------------------------------------


class _ScoreModule(Module[CompanySchema]):
    """Yields one judgement per candidate whose score is read from a per-id map.

    Lets a test drive the pair-level threshold sweep deterministically: the score
    map keys on ``left.id`` so each candidate gets a chosen score. ``cost`` is
    written to ``provenance['cost_usd']`` so the cost track is exercised too.
    """

    def __init__(
        self, scores: dict[str, float], *, cost: float = 0.0, skip: frozenset[str] = frozenset()
    ) -> None:
        self._scores = scores
        self._cost = cost
        self._skip = skip

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for cand in candidates:
            # ``skip`` yields NO judgement for those left ids, modelling a judge
            # that produced no verdict for a candidate (the gold-only slice case).
            if cand.left.id in self._skip:
                continue
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
    assert result.truncation_reason == "budget_cap"  # runner.dropped_by_cap_count > 0
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
    assert result.truncation_reason == "none"


class _AbstainingModule(Module[CompanySchema]):
    """Like ``_ScoreModule`` but flags ids in ``abstain`` with ``parse_error``.

    Models an LLMJudge under ``on_parse_error='abstain'``: the judgement is still
    emitted (score 0.0) but carries ``provenance['parse_error']`` so the evaluator
    can count it instead of silently folding it into the metric.
    """

    def __init__(self, scores: dict[str, float], abstain: frozenset[str]) -> None:
        self._scores = scores
        self._abstain = abstain

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for cand in candidates:
            provenance: dict[str, Any] = {"cost_usd": 0.0}
            if cand.left.id in self._abstain:
                provenance["parse_error"] = True
            yield PairwiseJudgement(
                left_id=cand.left.id,
                right_id=cand.right.id,
                score=self._scores[cand.left.id],
                score_type="prob_llm",
                decision_step="fake",
                provenance=provenance,
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError  # pragma: no cover — unused here


def test_evaluate_surfaces_parse_error_count_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scores = {"p0": 0.9, "p1": 0.0}
    cands, gold = _labeled_candidates(scores)
    with caplog.at_level(logging.WARNING, logger="langres.core.benchmark"):
        result, _ = benchmark_module.evaluate_judge_on_candidates(
            _AbstainingModule(scores, abstain=frozenset({"p1"})), cands, gold, grid=(0.5,)
        )
    assert result.n_parse_errors == 1
    assert result.n_abstained == 1
    assert result.abstention_rate == pytest.approx(1 / 2)  # 1 of 2 judged
    assert any(
        "parse" in r.message.lower() or "abstention" in r.message.lower() for r in caplog.records
    )
    # The warning must be truthful about what actually happens: an abstention's
    # score=0.0 always resolves to a non-match at any positive grid threshold, so
    # it silently becomes a false negative on a true gold pair -- never a graded
    # "verdict". The old wording ("graded on their emitted (abstained) scores")
    # didn't say this.
    assert any("false negative" in r.message.lower() for r in caplog.records)


def test_evaluate_no_parse_errors_is_zero_and_quiet(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scores = {"p0": 0.9}
    cands, gold = _labeled_candidates(scores)
    with caplog.at_level(logging.WARNING, logger="langres.core.benchmark"):
        result = benchmark_module.evaluate(
            _ScoreModule(scores), cands, gold, grid=(0.5,), threshold=0.5
        )
    assert result.n_parse_errors == 0
    assert result.n_abstained == 0
    assert result.abstention_rate == 0.0
    assert not any("parse" in r.message.lower() for r in caplog.records)


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
# evaluate_judge_on_candidates — honest fixed-threshold sliced aggregation
# ---------------------------------------------------------------------------


def _slice_candidates(
    scores: dict[str, float], positives: set[str]
) -> tuple[list[ERCandidate[CompanySchema]], set[frozenset[str]]]:
    """Candidates keyed by left id (right id is ``r_<lid>``); gold = ``positives``."""
    cands = [
        ERCandidate(
            left=CompanySchema(id=lid, name=lid),
            right=CompanySchema(id=f"r_{lid}", name=lid),
            blocker_name="test",
        )
        for lid in scores
    ]
    gold = {frozenset({lid, f"r_{lid}"}) for lid in positives}
    return cands, gold


def _tag_by_base_prefix(pair_key: frozenset[str]) -> str:
    """Tag a ``{lid, r_<lid>}`` pair by the first char of its non-``r_`` id."""
    base = next(i for i in pair_key if not i.startswith("r_"))
    return base[0]


def test_evaluate_judge_no_slice_fn_leaves_slices_none() -> None:
    # Default (no slice_fn): slices stays None and the rest of the result is intact.
    scores = {"p0": 0.9, "n0": 0.2}
    cands, gold = _labeled_candidates(scores)
    result, _ = benchmark_module.evaluate_judge_on_candidates(
        _ScoreModule(scores), cands, gold, grid=(0.5,)
    )
    assert result.slices is None
    assert result.pair.f1 == pytest.approx(1.0)


def test_evaluate_judge_grades_slices_at_fixed_threshold_not_per_slice_argmax() -> None:
    # Two slices A and B. The GLOBAL best-F1 threshold is 0.3 (ties break to the
    # first grid entry). At that FIXED cut slice A scores F1=0.667 — but slice A's
    # OWN argmax threshold (0.6, isolating its one true match) would score F1=1.0.
    # An honest sliced eval must report 0.667 (the fixed cut), never the 1.0 an
    # argmax would fake — that is exactly how a seen->unseen drop gets hidden.
    scores = {"A_p": 0.9, "A_n": 0.5, "B_p": 0.4, "B_n": 0.35}
    cands, gold = _slice_candidates(scores, positives={"A_p", "B_p"})
    result, judgements = benchmark_module.evaluate_judge_on_candidates(
        _ScoreModule(scores), cands, gold, grid=(0.3, 0.6), slice_fn=_tag_by_base_prefix
    )

    assert result.best_threshold == 0.3
    assert result.slices is not None
    assert set(result.slices) == {"A", "B"}

    # Every slice is graded at the ONE global threshold: reconstruct via
    # classify_pairs at result.best_threshold and it must match exactly.
    for tag, track in result.slices.items():
        tag_judged = [j for j in judgements if j.left_id.startswith(tag)]
        tag_gold = {pk for pk in gold if _tag_by_base_prefix(pk) == tag}
        expected = classify_pairs(tag_judged, tag_gold, result.best_threshold)
        assert track.precision == pytest.approx(expected.precision)
        assert track.recall == pytest.approx(expected.recall)
        assert track.f1 == pytest.approx(expected.f1)
        assert track.pr_curve is None  # a single fixed cut has no per-slice curve

    # Proof it is NOT a per-slice argmax: slice A's own best threshold (0.6) beats
    # its fixed-cut grade, so an argmax would have reported the higher number.
    a_judged = [j for j in judgements if j.left_id.startswith("A")]
    a_gold = {pk for pk in gold if _tag_by_base_prefix(pk) == "A"}
    a_argmax_f1 = classify_pairs(a_judged, a_gold, 0.6).f1
    assert a_argmax_f1 == pytest.approx(1.0)
    assert a_argmax_f1 > result.slices["A"].f1


def test_evaluate_judge_slice_fn_none_tag_excludes_pairs_from_every_slice() -> None:
    # slice_fn returns None for X-prefixed pairs -> they land in no slice, even
    # though X0 is a (high-scoring) gold pair that would form its own slice.
    scores = {"A0": 0.9, "B0": 0.8, "X0": 0.9}
    cands, gold = _slice_candidates(scores, positives={"A0", "B0", "X0"})

    def slice_fn(pair_key: frozenset[str]) -> str | None:
        tag = _tag_by_base_prefix(pair_key)
        return None if tag == "X" else tag

    result, _ = benchmark_module.evaluate_judge_on_candidates(
        _ScoreModule(scores), cands, gold, grid=(0.5,), slice_fn=slice_fn
    )
    assert result.slices is not None
    assert set(result.slices) == {"A", "B"}
    assert None not in result.slices


def test_evaluate_judge_gold_only_slice_reports_zero_recall_no_divide_by_zero() -> None:
    # The judge produces NO verdict for G0 (skip), but G0 is a gold candidate, so
    # its slice "G" holds a gold pair with zero judged pairs. Unioning the tag sets
    # keeps slice G reporting recall=0 (a real miss) instead of vanishing — and
    # classify_pairs over an empty predicted set must not divide by zero.
    scores = {"S0": 0.9, "S1": 0.2, "G0": 0.9}
    cands, gold = _slice_candidates(scores, positives={"S0", "G0"})
    result, judgements = benchmark_module.evaluate_judge_on_candidates(
        _ScoreModule(scores, skip=frozenset({"G0"})),
        cands,
        gold,
        grid=(0.5,),
        slice_fn=_tag_by_base_prefix,
    )
    assert not any(j.left_id == "G0" for j in judgements)  # G0 truly unjudged
    assert result.slices is not None
    assert set(result.slices) == {"S", "G"}
    assert result.slices["G"].recall == 0.0
    assert result.slices["G"].precision == 0.0
    assert result.slices["G"].f1 == 0.0
    assert result.slices["S"].f1 == pytest.approx(1.0)  # S0 caught, S1 below cut


# ---------------------------------------------------------------------------
# evaluate() — the bring-your-own-data one-liner over evaluate_judge_on_candidates
# ---------------------------------------------------------------------------


def test_evaluate_returns_pair_eval_with_sane_metrics() -> None:
    # Happy path: two true matches scored high, one non-match low -> at the tuned
    # threshold the judge is perfect. evaluate() returns just the JudgePairEval.
    scores = {"p0": 0.9, "p1": 0.8, "n0": 0.2}
    cands, gold = _labeled_candidates(scores)
    result = benchmark_module.evaluate(_ScoreModule(scores), cands, gold, grid=(0.5, 0.85))
    assert isinstance(result, benchmark_module.JudgePairEval)
    assert result.pair.precision == pytest.approx(1.0)
    assert result.pair.recall == pytest.approx(1.0)
    assert result.pair.f1 == pytest.approx(1.0)
    assert result.best_threshold == 0.5
    assert result.n_candidates == 3
    assert result.slices is None  # no slice_fn passed


def test_evaluate_passes_slice_fn_through() -> None:
    # slice_fn is forwarded: the result carries per-slice tracks graded at the one
    # global best-F1 threshold (evaluate() is a thin passthrough, so this just
    # confirms the kwarg reaches evaluate_judge_on_candidates).
    scores = {"A_p": 0.9, "A_n": 0.5, "B_p": 0.4, "B_n": 0.35}
    cands, gold = _slice_candidates(scores, positives={"A_p", "B_p"})
    result = benchmark_module.evaluate(
        _ScoreModule(scores), cands, gold, grid=(0.3, 0.6), slice_fn=_tag_by_base_prefix
    )
    assert result.slices is not None
    assert set(result.slices) == {"A", "B"}


def test_evaluate_uses_default_pair_grid_when_grid_omitted() -> None:
    # Omitting grid= sweeps DEFAULT_PAIR_GRID (the fine 0.05..0.95, 19 points), so
    # the PR curve has one point per grid threshold and the tuned cut lands inside it.
    # threshold= is also omitted (the sweep/argmax default), which now fires a
    # one-shot UserWarning about the optimistic bias -- assert it does.
    scores = {"p0": 0.9, "p1": 0.85, "n0": 0.1}
    cands, gold = _labeled_candidates(scores)
    with pytest.warns(UserWarning, match="argmax|biased|optimistic"):
        result = benchmark_module.evaluate(_ScoreModule(scores), cands, gold)
    assert result.pair.pr_curve is not None
    assert len(result.pair.pr_curve) == len(benchmark_module.DEFAULT_PAIR_GRID) == 19
    assert result.best_threshold in benchmark_module.DEFAULT_PAIR_GRID
    assert result.graded_threshold == result.best_threshold
    assert result.pair.f1 == pytest.approx(1.0)  # matches separable from the non-match


# ---------------------------------------------------------------------------
# evaluate() — Task 1: threshold=<float> is an honest fixed cut, no argmax
# ---------------------------------------------------------------------------


def test_evaluate_fixed_threshold_grades_once_no_argmax() -> None:
    # p0(0.9) clears 0.5, p1(0.3) does not, n0(0.2) correctly excluded.
    scores = {"p0": 0.9, "p1": 0.3, "n0": 0.2}
    cands, gold = _labeled_candidates(scores)
    result = benchmark_module.evaluate(
        _ScoreModule(scores), cands, gold, grid=(0.05, 0.5, 0.85), threshold=0.5
    )
    assert result.best_threshold is None  # never lie about an argmax that didn't happen
    assert result.graded_threshold == 0.5
    assert result.pair.precision == pytest.approx(1.0)
    assert result.pair.recall == pytest.approx(0.5)  # p1 missed at the fixed cut
    # pr_curve stays populated in fixed mode too (cheap, a later PR needs it).
    assert result.pair.pr_curve is not None
    assert len(result.pair.pr_curve) == 3


def test_evaluate_fixed_threshold_emits_no_bias_warning(recwarn: pytest.WarningsRecorder) -> None:
    scores = {"p0": 0.9, "n0": 0.2}
    cands, gold = _labeled_candidates(scores)
    benchmark_module.evaluate(_ScoreModule(scores), cands, gold, grid=(0.5,), threshold=0.5)
    assert len(recwarn) == 0


def test_evaluate_sweep_mode_warns_about_optimistic_bias() -> None:
    scores = {"p0": 0.9, "n0": 0.2}
    cands, gold = _labeled_candidates(scores)
    with pytest.warns(UserWarning, match="argmax|biased|optimistic"):
        benchmark_module.evaluate(_ScoreModule(scores), cands, gold, grid=(0.5,))


# ---------------------------------------------------------------------------
# evaluate() — Task 1/2: internal spend cap (default $1, budget_usd=, on_truncation=)
# ---------------------------------------------------------------------------


def test_evaluate_default_budget_usd_resolves_to_presets_default() -> None:
    # budget_usd omitted -> resolves to presets.DEFAULT_BUDGET_USD ($1) without
    # the caller passing it: a judge costing exactly that per pair is stopped
    # after the first pair (evaluate() is spend-capped by default, not opt-in).
    scores = {f"p{i}": 0.9 for i in range(3)}
    cands, gold = _labeled_candidates(scores)
    module = _ScoreModule(scores, cost=DEFAULT_BUDGET_USD)
    with pytest.raises(benchmark_module.EvaluationTruncatedError):
        benchmark_module.evaluate(module, cands, gold, grid=(0.5,), threshold=0.5)


def test_evaluate_free_judge_never_truncated_by_tiny_budget() -> None:
    # A zero-cost judge (string/embedding-shaped) never approaches ANY budget,
    # however small -- its real, measured spend stays $0 forever.
    scores = {f"p{i}": 0.9 for i in range(50)}
    cands, gold = _labeled_candidates(scores)
    result = benchmark_module.evaluate(
        _ScoreModule(scores, cost=0.0), cands, gold, grid=(0.5,), budget_usd=0.0001, threshold=0.5
    )
    assert result.truncated is False
    assert result.n_judged == 50
    assert result.truncation_reason == "none"


def test_evaluate_raises_on_budget_truncation_by_default() -> None:
    scores = {f"p{i}": 0.9 for i in range(4)}
    cands, gold = _labeled_candidates(scores)
    with pytest.raises(benchmark_module.EvaluationTruncatedError) as excinfo:
        benchmark_module.evaluate(
            _ScoreModule(scores, cost=0.1),
            cands,
            gold,
            grid=(0.5,),
            budget_usd=0.05,
            threshold=0.5,
        )
    err = excinfo.value
    message = str(err)
    assert "budget_usd" in message
    assert 'on_truncation="return"' in message
    assert err.partial is not None
    assert err.partial.truncated is True
    assert err.partial.n_judged < err.partial.n_candidates
    assert err.partial.truncation_reason in ("budget_cap", "budget_stop")


def test_evaluate_on_truncation_return_gives_partial_silently(
    recwarn: pytest.WarningsRecorder,
) -> None:
    scores = {f"p{i}": 0.9 for i in range(4)}
    cands, gold = _labeled_candidates(scores)
    result = benchmark_module.evaluate(
        _ScoreModule(scores, cost=0.1),
        cands,
        gold,
        grid=(0.5,),
        budget_usd=0.05,
        on_truncation="return",
        threshold=0.5,
    )
    assert result.truncated is True
    assert result.truncation_reason in ("budget_cap", "budget_stop")
    assert len(recwarn) == 0  # "return" is silent


def test_evaluate_on_truncation_warn_logs_and_returns_partial() -> None:
    scores = {f"p{i}": 0.9 for i in range(4)}
    cands, gold = _labeled_candidates(scores)
    with pytest.warns(UserWarning, match="budget_usd"):
        result = benchmark_module.evaluate(
            _ScoreModule(scores, cost=0.1),
            cands,
            gold,
            grid=(0.5,),
            budget_usd=0.05,
            on_truncation="warn",
            threshold=0.5,
        )
    assert result.truncated is True
    assert result.truncation_reason in ("budget_cap", "budget_stop")


def test_evaluate_judge_skips_warns_but_never_raises_even_under_raise_mode() -> None:
    # A candidate whose module call raises is a NON-spend truncation
    # (truncation_reason="judge_skips"): it must only ever warn, never raise --
    # even under the default on_truncation="raise" -- so one bad candidate
    # cannot blow up a run.
    cands = _candidates(3)
    module = _FakeModule(cost=0.0, boom_ids=frozenset({"l1"}))
    with pytest.warns(UserWarning):
        result = benchmark_module.evaluate(module, cands, set(), grid=(0.5,), threshold=0.5)
    assert result.truncated is True
    assert result.truncation_reason == "judge_skips"
    assert result.n_judged == 2


def test_evaluate_raises_when_every_candidate_fails_despite_judge_skips_reason() -> None:
    # A judge that fails on EVERY candidate produces zero judgements. Reporting
    # precision/recall/F1 of 0.0 there is the dishonest cell: it is
    # indistinguishable from a healthy judge that matched nothing. So a zero-
    # judgement run raises even though its reason is "judge_skips", which
    # otherwise only ever warns.
    cands = _candidates(3)
    module = _FakeModule(cost=0.0, boom_ids=frozenset({"l0", "l1", "l2"}))
    with pytest.raises(benchmark_module.EvaluationTruncatedError) as exc:
        benchmark_module.evaluate(module, cands, set(), grid=(0.5,), threshold=0.5)
    assert "0 judgements" in str(exc.value)
    # The partial result is still attached, so a caller can inspect the wreckage.
    partial = exc.value.partial
    assert partial is not None
    assert partial.n_judged == 0
    assert partial.n_candidates == 3


def test_evaluate_zero_judgements_returns_silently_under_on_truncation_return() -> None:
    # "return" means "I know it may be partial, give me what you have" -- an
    # empty result is then a deliberate, opted-into outcome rather than a lie.
    cands = _candidates(3)
    module = _FakeModule(cost=0.0, boom_ids=frozenset({"l0", "l1", "l2"}))
    result = benchmark_module.evaluate(
        module, cands, set(), grid=(0.5,), threshold=0.5, on_truncation="return"
    )
    assert result.n_judged == 0
    assert result.truncated is True


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


def test_run_methods_stamps_the_requested_registry_method_name() -> None:
    """Each row's ``method`` is the registry key passed in, not the module ``type_name``.

    ``run_method`` labels a result from the module's ``type_name`` (e.g.
    ``weighted_average_judge``), but the caller races by the registry key
    (``weighted_average``). ``run_methods`` overwrites the label so
    ``best().method`` is a name ``make_resolver_factory`` accepts and can re-run.
    """
    from langres.methods import make_resolver_factory

    bench = _FakeBlockingBenchmark()
    table = run_methods(bench, ["embedding_cosine", "weighted_average"], seed=0)

    # Rows carry the exact registry keys, in order.
    assert [r.method for r in table.results] == ["embedding_cosine", "weighted_average"]

    # And the winner's label round-trips through the registry (no ValueError).
    best = table.best()
    assert best is not None
    assert best.method in ("embedding_cosine", "weighted_average")
    make_resolver_factory(best.method, bench)  # accepted registry key -> no raise


# ---------------------------------------------------------------------------
# BenchmarkTable.best / rank (structured accessors)
# ---------------------------------------------------------------------------


def _result(method: str, *, pair_f1: float, bcubed_f1: float, delta: float = 0.0) -> MethodResult:
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
