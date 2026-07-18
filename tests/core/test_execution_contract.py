"""Public, slot-neutral execution contract for explicit and classic ER models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.indexes.vector_index import FakeVectorIndex
from langres.core.models import CompanySchema
from langres.core.op import ExecutionEvent, Score, Stage, ThresholdSelect, TopKSelect
from langres.core.op_adapters import BlockerSource, ClustererStage, MatcherScore
from langres.core.pairs import Pairs
from langres.core.resolver import ERModel, Resolver
from langres.core.serialization import ArtifactManifest
from langres.core.spend import SpendMonitor
from langres.core.spend_cap import SpendCappedMatcher
from langres.tracking.judgement_log import JudgementLog
from tests.parity._explicit_chain_fixture import (
    ChainCo,
    RECORDS,
    THRESHOLD,
    CostedNameMatcher,
    build_score_after_select_model,
    chain_ops,
)


def _explicit() -> ERModel:
    ops, _matcher = chain_ops()
    return ERModel.from_topology(ops=ops)


def test_explicit_model_binding_and_schema_come_from_source() -> None:
    model = _explicit()

    assert model.is_bound is True
    assert model.schema is ChainCo


def test_explicit_compare_and_dedupe_log_every_matcher_score(tmp_path: Path) -> None:
    model = _explicit()
    log = JudgementLog(tmp_path / "judgements.jsonl")

    model.compare(RECORDS[0], RECORDS[1], log=log)
    model.dedupe(RECORDS, log=log)

    rows = log.read()
    # compare scores one named pair; the four-record fixture produces six
    # all-pairs candidates during dedupe.
    assert len(rows) == 7
    assert {(row["left_id"], row["right_id"]) for row in rows} >= {("a1", "a2")}


def test_explicit_compare_logs_each_matcher_in_a_multi_score_chain(tmp_path: Path) -> None:
    model = build_score_after_select_model()
    log = JudgementLog(tmp_path / "multi.jsonl")

    model.compare(RECORDS[0], RECORDS[1], log=log)

    assert len(log.read()) == 2


def test_explicit_multi_stage_log_uses_each_stage_identity_and_decision_semantics(
    tmp_path: Path,
) -> None:
    """A retrieval score is not mislabeled as the final match decision/model."""
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(
            CostedNameMatcher(score_type="sim_cos", model="student/model"),
            out_space="sim_cos",
        ),
        TopKSelect(k=5),
        MatcherScore(
            CostedNameMatcher(score_type="prob_llm", model="final/model"),
            out_space="prob_llm",
        ),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    model = ERModel.from_topology(ops=ops)
    log = JudgementLog(tmp_path / "stages.jsonl")

    verdict = model.compare(RECORDS[0], RECORDS[1], log=log)

    assert verdict.backbone == "final/model"
    first, second = log.read()
    assert first["model"] == "student/model"
    assert first["verdict"] is None
    assert first["stage_id"] == "01-matcher_score"
    assert second["model"] == "final/model"
    assert second["verdict"] is True
    assert second["stage_id"] == "03-matcher_score"


def test_explicit_verdict_backbone_is_the_scorer_that_ran_before_an_early_select(
    tmp_path: Path,
) -> None:
    """A later, unexecuted scorer cannot claim an early-select verdict."""
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(
            CostedNameMatcher(score_type="sim_cos", model="student/model"),
            out_space="sim_cos",
        ),
        TopKSelect(k=0),
        MatcherScore(
            CostedNameMatcher(score_type="prob_llm", model="never-ran/model"),
            out_space="prob_llm",
        ),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    model = ERModel.from_topology(ops=ops)
    log = JudgementLog(tmp_path / "early.jsonl")

    verdict = model.compare(RECORDS[0], RECORDS[1], log=log)

    assert verdict.match is False
    assert verdict.backbone == "student/model"
    assert [row["model"] for row in log.read()] == ["student/model"]


def test_explicit_logging_preserves_a_supported_matcher_score_subclass(tmp_path: Path) -> None:
    class _TaggedMatcherScore(MatcherScore[Any]):
        def __init__(self, matcher: Any, *, tag: str) -> None:
            super().__init__(matcher, out_space="prob_llm")
            self.tag = tag

    monitor = SpendMonitor(budget_usd=1.0)
    tagged = _TaggedMatcherScore(
        SpendCappedMatcher(CostedNameMatcher(), monitor=monitor),
        tag="preserve",
    )
    model = ERModel.from_topology(
        ops=[
            BlockerSource(AllPairsBlocker(schema=ChainCo)),
            tagged,
            ThresholdSelect(THRESHOLD),
            ClustererStage(Clusterer(threshold=0.0)),
        ],
        monitor=monitor,
    )
    log = JudgementLog(tmp_path / "subclass.jsonl")

    model.compare(RECORDS[0], RECORDS[1], log=log)

    assert len(log.read()) == 1
    assert tagged.tag == "preserve"


class _CountingScore(Score[Any]):
    def __init__(self, score: float) -> None:
        super().__init__(scope="pair", out_space="heuristic")
        self.score = score
        self.calls = 0

    def forward(self, pairs: Any) -> Any:
        self.calls += 1
        rows = [
            row.model_copy(update={"score": self.score, "score_type": "heuristic"})
            for row in pairs.rows
        ]
        return type(pairs)(store=pairs.store, rows=rows)


def test_explicit_compare_honors_an_early_select_before_later_scores() -> None:
    first = _CountingScore(0.0)
    should_not_run = _CountingScore(1.0)
    model = ERModel.from_topology(
        ops=[
            BlockerSource(AllPairsBlocker(schema=CompanySchema)),
            first,
            ThresholdSelect(0.5),
            should_not_run,
            ThresholdSelect(0.5),
            ClustererStage(Clusterer(threshold=0.0)),
        ]
    )

    verdict = model.compare(
        {"id": "left", "name": "A"},
        {"id": "right", "name": "B"},
    )

    assert verdict.match is False
    assert first.calls == 1
    assert should_not_run.calls == 0


def test_explicit_vector_source_builds_and_reuses_its_index() -> None:
    class CountingIndex(FakeVectorIndex):
        def __init__(self) -> None:
            super().__init__()
            self.builds = 0

        def create_index(self, texts: list[str]) -> None:
            self.builds += 1
            super().create_index(texts)

    index = CountingIndex()
    blocker = VectorBlocker(
        vector_index=index,
        schema=CompanySchema,
        text_field="name",
        k_neighbors=2,
    )
    model = ERModel.from_topology(
        ops=[
            BlockerSource(blocker),
            MatcherScore(CostedNameMatcher(), out_space="prob_llm"),
            # The fixture matcher scores equal names at 0.95.
            ThresholdSelect(THRESHOLD),
            ClustererStage(Clusterer(threshold=0.0)),
        ]
    )

    first = model.dedupe(RECORDS)
    assert index._n_samples == len(RECORDS)
    second = model.dedupe(RECORDS)

    assert first == second
    assert index._n_samples == len(RECORDS)
    assert index.builds == 1

    changed = [dict(record) for record in RECORDS]
    changed[0]["name"] = "Changed corpus"
    model.dedupe(changed)
    assert index.builds == 2


def test_execution_plan_is_stable_safe_and_slot_neutral() -> None:
    first = _explicit().execution_plan()
    second = _explicit().execution_plan()

    assert first == second
    assert first.is_bound is True
    assert first.schema_name == "ChainCo"
    assert [step.spec.role for step in first.steps] == [
        "blocker_source",
        "comparator_score",
        "matcher_score",
        "threshold_select",
        "clusterer_stage",
    ]
    assert len({step.stage_id for step in first.steps}) == len(first.steps)
    assert all("0x" not in step.stage_id for step in first.steps)
    # The whole contract crosses a plain JSON boundary.
    first.model_validate_json(first.model_dump_json())


def test_execution_plan_and_execute_do_not_require_artifact_serialization() -> None:
    """Runnable custom stages and schema factories need registration only to save."""
    source = BlockerSource(
        AllPairsBlocker(schema_factory=lambda record: ChainCo.model_validate(record))
    )
    score = _CountingScore(0.9)
    model = ERModel.from_topology(
        ops=[
            source,
            score,
            ThresholdSelect(0.5),
            ClustererStage(Clusterer(threshold=0.0)),
        ]
    )

    plan = model.execution_plan()
    result = model.execute(RECORDS)

    assert plan.steps[0].spec.role == "blocker_source"
    assert plan.steps[1].spec.role == "score"
    assert result.plan == plan
    assert score.calls == 1


def test_execution_plan_resource_refs_name_actual_models() -> None:
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(CostedNameMatcher(model="student/model"), out_space="sim_cos"),
        TopKSelect(k=5),
        MatcherScore(CostedNameMatcher(model="final/model"), out_space="prob_llm"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]

    plan = ERModel.from_topology(ops=ops).execution_plan()

    matcher_steps = [step for step in plan.steps if step.spec.role == "matcher_score"]
    assert [step.resource_ref for step in matcher_steps] == ["student/model", "final/model"]


def test_execute_uses_the_plan_and_emits_immutable_observer_events() -> None:
    model = _explicit()
    observed: list[ExecutionEvent] = []

    result = model.execute(RECORDS, observer=observed.append)

    assert result.plan == model.execution_plan()
    assert result.clusters == tuple(frozenset(cluster) for cluster in model.resolve(RECORDS))
    assert result.pairs.rows
    assert tuple(observed) == result.events
    assert result.observer_errors == ()
    assert [event.kind for event in observed].count("start") == len(result.plan.steps)
    assert [event.kind for event in observed].count("finish") == len(result.plan.steps)
    assert all(
        event.stage_id in {step.stage_id for step in result.plan.steps} for event in observed
    )
    with pytest.raises(ValidationError):
        observed[0].stage_id = "mutated"  # type: ignore[misc]


def test_observer_failure_is_isolated_explicit_and_cannot_change_results() -> None:
    def broken(_event: ExecutionEvent) -> None:
        raise LookupError("secret record content")

    model = _explicit()
    baseline = model.execute(RECORDS)
    observed = model.execute(RECORDS, observer=broken)

    assert observed.pairs == baseline.pairs
    assert observed.clusters == baseline.clusters
    assert len(observed.observer_errors) == len(observed.events)
    assert {error.exception_type for error in observed.observer_errors} == {"LookupError"}
    assert {error.message for error in observed.observer_errors} == {
        "observer callback raised; exception details suppressed"
    }
    assert "secret record content" not in observed.model_dump_json()


def test_classic_model_exposes_the_same_execution_contract() -> None:
    model = Resolver.from_schema(CompanySchema, threshold=THRESHOLD)

    plan = model.execution_plan()
    result = model.execute(RECORDS)

    assert [step.spec.role for step in plan.steps] == [
        "blocker_source",
        "comparator_score",
        "matcher_score",
        "threshold_select",
        "clusterer_stage",
    ]
    assert result.clusters == tuple(frozenset(cluster) for cluster in model.resolve(RECORDS))


def test_authoring_contracts_are_public_core_exports() -> None:
    import langres.core as core

    expected = {
        "ClusterStage",
        "ExecutionEvent",
        "ExecutionObserver",
        "ExecutionObserverError",
        "ExecutionPlan",
        "ExecutionResult",
        "ExecutionStep",
        "Feasible",
        "Op",
        "Score",
        "Select",
        "Sequential",
        "Source",
        "SpendMonitorBindable",
        "register_op",
    }
    assert expected <= set(core.__all__)
    assert all(hasattr(core, name) for name in expected)


def test_manifest_model_still_validates_after_execution_contract() -> None:
    """Guard the execution models from accidentally widening the artifact leaf."""
    ArtifactManifest(
        artifact_version="2",
        langres_version="test",
        ops=[],
    )
