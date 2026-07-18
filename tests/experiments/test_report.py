from __future__ import annotations

import pydantic
import pytest

from langres.experiments import (
    EvaluationProtocol,
    ExperimentReport,
    ExperimentRun,
    IncompatibleProtocolError,
    ReportConstraints,
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
    evaluation_id: str = "eval-a",
    cohort_id: str = "cpu-a",
    pair_f1: float | None = None,
    wall_seconds: float | None = None,
    usd: float | None = None,
    model_size_bytes: int | None = None,
) -> ExperimentRun:
    return ExperimentRun(
        recipe_id=f"recipe-{architecture}",
        evaluation_id=evaluation_id,
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
        evaluation_id="eval-a",
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
        evaluation_id="eval-a",
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


def test_pareto_refuses_to_mix_incompatible_cohorts() -> None:
    report = ExperimentReport(
        evaluation_id="eval-a",
        protocol=_protocol(),
        runs=(
            _run("a", pair_f1=0.9, wall_seconds=1.0),
            _run("b", pair_f1=0.8, wall_seconds=0.5, cohort_id="gpu-b"),
        ),
    )

    assert len(report.cohorts) == 2
    with pytest.raises(IncompatibleProtocolError, match="cohort"):
        report.pareto({"pair_f1": "max", "wall_seconds": "min"})
