from __future__ import annotations

import pydantic
import pytest

from langres.experiments import (
    EvaluationProtocol,
    ExperimentReport,
    ExperimentRun,
    ReportConstraints,
    compute_evaluation_identity,
)


def _protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        benchmark_ids=("dataset",),
        split_ids=("fixed",),
        fixed_test_set_id="dataset:test:v1",
        split_seeds=(1,),
        threshold_split_id="validation",
        test_split_id="test",
        hardware_cohort="cpu-a",
        benchmark_version="1",
    )


def _run(
    architecture: str,
    *,
    status: str = "completed",
    evaluation_id: str | None = None,
    cohort_id: str = "cpu-a",
    pair_f1: float | None = None,
    wall_seconds: float | None = None,
    usd: float | None = None,
    model_size_bytes: int | None = None,
) -> ExperimentRun:
    resolved_evaluation_id = evaluation_id or compute_evaluation_identity(_protocol()).evaluation_id
    return ExperimentRun(
        recipe_id=f"recipe-{architecture}",
        evaluation_id=resolved_evaluation_id,
        attempt_id=f"attempt-{architecture}",
        architecture=architecture,
        benchmark_id="dataset",
        split_id="fixed",
        split_seed=1,
        repeat_index=0,
        status=status,
        cohort_id=cohort_id,
        metrics={"pair_f1": pair_f1},
        wall_seconds=wall_seconds,
        usd=usd,
        model_size_bytes=model_size_bytes,
    )


def test_report_is_immutable_and_preserves_failed_and_missing_cells() -> None:
    report = ExperimentReport(
        protocol=_protocol(),
        runs=(
            _run("good", pair_f1=0.9, wall_seconds=1.0, usd=0.1),
            _run("failed", status="failed"),
            _run("missing", status="missing"),
        ),
        reproduction_artifact=".langres/runs/eval-a.json",
    )

    markdown = report.to_markdown()

    assert "| failed |" in markdown
    assert "| missing |" in markdown
    assert "Reproduce: langres experiments reproduce .langres/runs/eval-a.json" in markdown
    aggregates = report.aggregate("pair_f1")
    assert next(row for row in aggregates if row.architecture == "good").completed == 1
    assert sum(row.failed for row in aggregates) == 1
    assert sum(row.missing for row in aggregates) == 1
    with pytest.raises(pydantic.ValidationError):
        report.evaluation_id = "changed"  # type: ignore[misc]


def test_constraints_and_pareto_exclude_unknown_required_facts() -> None:
    report = ExperimentReport(
        protocol=_protocol(),
        runs=(
            _run("quality", pair_f1=0.95, wall_seconds=2.0, usd=0.2, model_size_bytes=100),
            _run("speed", pair_f1=0.90, wall_seconds=1.0, usd=0.1, model_size_bytes=200),
            _run("dominated", pair_f1=0.80, wall_seconds=3.0, usd=0.3, model_size_bytes=300),
            _run("unknown", pair_f1=0.99, wall_seconds=None, usd=None),
        ),
    )

    constrained = report.constrained(ReportConstraints(max_usd=0.25))
    assert {run.architecture for run in constrained} == {"quality", "speed"}

    front = report.pareto({"pair_f1": "max", "wall_seconds": "min", "usd": "min"})
    assert {run.architecture for run in front} == {"quality", "speed"}


def test_report_derives_identity_and_rejects_spoofed_or_out_of_protocol_rows() -> None:
    protocol = _protocol()
    expected = compute_evaluation_identity(protocol).evaluation_id
    report = ExperimentReport(protocol=protocol, runs=(_run("a", pair_f1=0.9),))
    assert report.evaluation_id == expected

    with pytest.raises(pydantic.ValidationError, match="does not match protocol"):
        ExperimentReport(evaluation_id="spoofed", protocol=protocol, runs=())
    with pytest.raises(pydantic.ValidationError, match="benchmark_id"):
        ExperimentReport(
            protocol=protocol,
            runs=(_run("a", pair_f1=0.9).model_copy(update={"benchmark_id": "other"}),),
        )
    with pytest.raises(pydantic.ValidationError, match="split_id"):
        ExperimentReport(
            protocol=protocol,
            runs=(_run("a", pair_f1=0.9).model_copy(update={"split_id": "other"}),),
        )
    with pytest.raises(pydantic.ValidationError, match="split_seed"):
        ExperimentReport(
            protocol=protocol,
            runs=(_run("a", pair_f1=0.9).model_copy(update={"split_seed": 999}),),
        )
    with pytest.raises(pydantic.ValidationError, match="cohort_id"):
        ExperimentReport(
            protocol=protocol,
            runs=(_run("a", pair_f1=0.9).model_copy(update={"cohort_id": "gpu-b"}),),
        )


def test_aggregate_honors_median_and_exhaustive_denominators_without_fake_ci() -> None:
    protocol = _protocol().model_copy(update={"aggregation": "median"})
    evaluation_id = compute_evaluation_identity(protocol).evaluation_id
    runs = (
        _run("a", evaluation_id=evaluation_id, pair_f1=0.1),
        _run("a", evaluation_id=evaluation_id, pair_f1=0.9),
        _run("a", evaluation_id=evaluation_id, pair_f1=0.8),
        _run("a", evaluation_id=evaluation_id, status="completed", pair_f1=None),
        _run("a", evaluation_id=evaluation_id, status="failed"),
        _run("a", evaluation_id=evaluation_id, status="missing"),
    )

    [row] = ExperimentReport(protocol=protocol, runs=runs).aggregate("pair_f1")

    assert row.value == pytest.approx(0.8)
    assert row.aggregation == "median"
    assert row.completed == 4
    assert row.observed == 3
    assert row.failed == 1
    assert row.missing == 1
    assert row.total == 6
    assert row.confidence_interval.status == "unavailable"
    assert "paired entity" in (row.confidence_interval.reason or "")


def test_non_finite_metrics_and_bad_pareto_requests_fail_actionably() -> None:
    with pytest.raises(pydantic.ValidationError, match="finite"):
        _run("a", pair_f1=float("nan"))

    report = ExperimentReport(protocol=_protocol(), runs=(_run("a", pair_f1=0.9),))
    with pytest.raises(ValueError, match="direction"):
        report.pareto({"pair_f1": "up"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="unknown Pareto objective"):
        report.pareto({"made_up_metric": "max"})
