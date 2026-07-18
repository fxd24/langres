from __future__ import annotations

import pytest

from langres.experiments import (
    CacheIdentityInput,
    EvaluationProtocol,
    ExperimentReport,
    ExperimentRun,
    ReportConstraints,
    ResourceSlotIdentity,
    RuntimeFacts,
    SourceState,
    TokenUsage,
    compute_evaluation_identity,
)


def test_protocol_nested_identity_fields_are_deeply_immutable_and_serialize_stably() -> None:
    protocol = EvaluationProtocol(
        benchmark_ids=("dataset",),
        dataset_fingerprints={"dataset": "sha256:a"},
        split_ids=("fixed",),
        fixed_test_set_id="dataset:test",
        split_seeds=(0,),
        deterministic_resources={"embedder": {"batch_sizes": [16, 32]}},
        threshold_split_id="validation",
        test_split_id="test",
        hardware_cohort="cpu",
        benchmark_version="1",
    )
    before = protocol.model_dump_json()

    with pytest.raises(TypeError, match="immutable"):
        protocol.dataset_fingerprints["dataset"] = "changed"
    with pytest.raises(TypeError, match="immutable"):
        protocol.deterministic_resources["embedder"]["batch_sizes"] = (64,)

    assert protocol.model_dump_json() == before


def test_cache_and_report_nested_identity_fields_are_deeply_immutable() -> None:
    slot = ResourceSlotIdentity(
        slot="embedder",
        base="org/model",
        kind="hf",
        revision="rev",
        runtime_config={"batch_size": 16},
    )
    cache = CacheIdentityInput(
        stage_id="embed",
        execution_plan_id="plan",
        operation_identity={"type": "Embed", "config": {"normalize": True}},
        resource_slots=(slot,),
        source=SourceState(
            git_sha="a" * 40,
            lockfile_hash="lock",
            environment_hash="env",
        ),
        semantics="deterministic",
        input_fingerprint="rows",
    )
    protocol = EvaluationProtocol.smoke(seed=0)
    evaluation_id = compute_evaluation_identity(protocol).evaluation_id
    run = ExperimentRun(
        recipe_id="recipe",
        evaluation_id=evaluation_id,
        architecture="fuzzy",
        variant_id="fuzzy-default",
        benchmark_id="tiny_fixture",
        split_id="smoke",
        split_seed=0,
        repeat_index=0,
        status="completed",
        cohort_id="local-smoke",
        metrics={"pair_f1": 1.0},
    )
    report = ExperimentReport(protocol=protocol, runs=(run,))
    cache_json = cache.model_dump_json()
    report_json = report.model_dump_json()

    with pytest.raises(TypeError, match="immutable"):
        cache.operation_identity["config"]["normalize"] = False
    with pytest.raises(TypeError, match="immutable"):
        slot.runtime_config["batch_size"] = 32
    with pytest.raises(TypeError, match="immutable"):
        report.runs[0].metrics["pair_f1"] = 0.0

    assert cache.model_dump_json() == cache_json
    assert report.model_dump_json() == report_json


def test_default_experiment_run_metrics_mapping_is_immutable() -> None:
    protocol = EvaluationProtocol.smoke()
    run = ExperimentRun(
        recipe_id="recipe",
        evaluation_id=compute_evaluation_identity(protocol).evaluation_id,
        architecture="fuzzy",
        variant_id="fuzzy-default",
        benchmark_id="tiny_fixture",
        split_id="smoke",
        split_seed=0,
        repeat_index=0,
        status="completed",
        cohort_id="local-smoke",
    )

    with pytest.raises(TypeError, match="immutable"):
        run.metrics["pair_f1"] = 1.0


def test_all_default_experiment_mappings_are_immutable() -> None:
    usage = TokenUsage()
    runtime = RuntimeFacts(hardware_cohort="cpu")
    constraints = ReportConstraints()

    with pytest.raises(TypeError, match="immutable"):
        usage.provider_usage["provider"] = {}
    with pytest.raises(TypeError, match="immutable"):
        runtime.library_versions["numpy"] = "2"
    with pytest.raises(TypeError, match="immutable"):
        constraints.minimum_metrics["pair_f1"] = 0.9
