"""Immutable experiment rows, compatible cohorts, constraints, and Pareto views."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from langres.experiments.measurements import FunnelFacts, StageMeasurement, TokenUsage
from langres.experiments.protocol import EvaluationProtocol
from langres.experiments.statistics import SplitInstability, split_instability

RunStatus = Literal["running", "completed", "failed", "budget_exceeded", "missing"]
Direction = Literal["min", "max"]


class IncompatibleProtocolError(ValueError):
    """Raised when a comparison would mix incompatible evaluation cohorts."""


class ExperimentRun(BaseModel):
    """One immutable matrix cell, including failures and explicit missing cells."""

    model_config = ConfigDict(frozen=True)

    recipe_id: str
    evaluation_id: str
    attempt_id: str | None = None
    cache_id: str | None = None
    architecture: str
    benchmark_id: str
    split_id: str
    split_seed: int
    repeat_index: int = Field(ge=0)
    status: RunStatus
    cohort_id: str
    metrics: dict[str, float | None] = Field(default_factory=dict)
    measurements: tuple[StageMeasurement, ...] = ()
    funnel: FunnelFacts | None = None
    wall_seconds: float | None = Field(default=None, ge=0.0)
    p95_latency_seconds: float | None = Field(default=None, ge=0.0)
    token_usage: TokenUsage | None = None
    usd: float | None = Field(default=None, ge=0.0)
    model_size_bytes: int | None = Field(default=None, ge=0)
    loaded_memory_bytes: int | None = Field(default=None, ge=0)
    warnings: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    def fact(self, name: str) -> float | None:
        """Resolve a quality metric or standard resource/performance fact."""
        if name in self.metrics:
            return self.metrics[name]
        value = getattr(self, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None


class AggregateRow(BaseModel):
    """Aggregate with explicit completion/failure/missing denominators."""

    model_config = ConfigDict(frozen=True)

    architecture: str
    benchmark_id: str
    metric: str
    mean: float | None
    standard_deviation: float | None
    completed: int
    failed: int
    missing: int
    total: int


class CohortView(BaseModel):
    """Rows that may be compared for performance."""

    model_config = ConfigDict(frozen=True)

    evaluation_id: str
    cohort_id: str
    runs: tuple[ExperimentRun, ...]


class ReportConstraints(BaseModel):
    """Optional hard filters applied before ranking or Pareto analysis."""

    model_config = ConfigDict(frozen=True)

    max_p95_latency_seconds: float | None = Field(default=None, ge=0.0)
    max_wall_seconds: float | None = Field(default=None, ge=0.0)
    max_usd: float | None = Field(default=None, ge=0.0)
    max_model_size_bytes: int | None = Field(default=None, ge=0)
    max_loaded_memory_bytes: int | None = Field(default=None, ge=0)
    minimum_metrics: dict[str, float] = Field(default_factory=dict)


class ExperimentReport(BaseModel):
    """All planned cells and derived views; failed/missing rows are never imputed."""

    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    evaluation_id: str
    protocol: EvaluationProtocol
    runs: tuple[ExperimentRun, ...]
    reproduction_artifact: str | None = None

    @model_validator(mode="after")
    def _validate_evaluation(self) -> "ExperimentReport":
        incompatible = {
            run.evaluation_id for run in self.runs if run.evaluation_id != self.evaluation_id
        }
        if incompatible:
            raise ValueError(
                "all report rows must share evaluation_id; separate incompatible protocols "
                f"into their own reports: {sorted(incompatible)}"
            )
        return self

    @property
    def reproduce_command(self) -> str | None:
        if self.reproduction_artifact is None:
            return None
        return f"langres experiments reproduce {self.reproduction_artifact}"

    @property
    def cohorts(self) -> tuple[CohortView, ...]:
        grouped: dict[tuple[str, str], list[ExperimentRun]] = defaultdict(list)
        for run in self.runs:
            grouped[(run.evaluation_id, run.cohort_id)].append(run)
        return tuple(
            CohortView(evaluation_id=key[0], cohort_id=key[1], runs=tuple(rows))
            for key, rows in sorted(grouped.items())
        )

    def aggregate(self, metric: str) -> tuple[AggregateRow, ...]:
        grouped: dict[tuple[str, str], list[ExperimentRun]] = defaultdict(list)
        for run in self.runs:
            grouped[(run.architecture, run.benchmark_id)].append(run)
        output: list[AggregateRow] = []
        for (architecture, benchmark_id), rows in sorted(grouped.items()):
            values = [
                value
                for run in rows
                if run.status == "completed"
                for value in [run.metrics.get(metric)]
                if value is not None
            ]
            failed = sum(run.status in {"failed", "budget_exceeded"} for run in rows)
            missing = sum(run.status in {"missing", "running"} for run in rows)
            output.append(
                AggregateRow(
                    architecture=architecture,
                    benchmark_id=benchmark_id,
                    metric=metric,
                    mean=statistics.fmean(values) if values else None,
                    standard_deviation=statistics.stdev(values) if len(values) > 1 else None,
                    completed=len(values),
                    failed=failed,
                    missing=missing,
                    total=len(rows),
                )
            )
        return tuple(output)

    def split_instability(self, metric: str) -> dict[str, SplitInstability]:
        """Report split-seed sensitivity separately for each architecture/dataset."""
        repeated: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for run in self.runs:
            value = run.metrics.get(metric)
            if run.status != "completed" or value is None:
                continue
            key = f"{run.architecture}:{run.benchmark_id}"
            split_key = f"{run.split_id}:seed-{run.split_seed}"
            repeated[key][split_key].append(value)
        grouped = {
            key: {
                split_key: (
                    statistics.fmean(values)
                    if self.protocol.aggregation == "mean"
                    else statistics.median(values)
                )
                for split_key, values in split_values.items()
            }
            for key, split_values in repeated.items()
        }
        return {key: split_instability(values) for key, values in sorted(grouped.items())}

    def constrained(self, constraints: ReportConstraints) -> tuple[ExperimentRun, ...]:
        return tuple(run for run in self.runs if self._meets_constraints(run, constraints))

    def pareto(
        self,
        objectives: Mapping[str, Direction],
        *,
        constraints: ReportConstraints | None = None,
        cohort_id: str | None = None,
    ) -> tuple[ExperimentRun, ...]:
        """Return the non-dominated completed rows within exactly one cohort."""
        if not objectives:
            raise ValueError("pareto requires at least one objective")
        cohort_ids = {run.cohort_id for run in self.runs if run.status == "completed"}
        if cohort_id is None and len(cohort_ids) > 1:
            raise IncompatibleProtocolError(
                "Pareto comparison spans multiple hardware cohorts; pass cohort_id explicitly"
            )
        selected_cohort = cohort_id or (next(iter(cohort_ids)) if cohort_ids else None)
        candidates = [
            run
            for run in self.runs
            if run.status == "completed"
            and (selected_cohort is None or run.cohort_id == selected_cohort)
            and all(run.fact(name) is not None for name in objectives)
            and (constraints is None or self._meets_constraints(run, constraints))
        ]
        front: list[ExperimentRun] = []
        for candidate in candidates:
            if not any(
                other is not candidate and self._dominates(other, candidate, objectives)
                for other in candidates
            ):
                front.append(candidate)
        return tuple(front)

    @staticmethod
    def _dominates(
        candidate: ExperimentRun,
        other: ExperimentRun,
        objectives: Mapping[str, Direction],
    ) -> bool:
        at_least_as_good = True
        strictly_better = False
        for name, direction in objectives.items():
            candidate_value = candidate.fact(name)
            other_value = other.fact(name)
            assert candidate_value is not None and other_value is not None
            if direction == "max":
                at_least_as_good &= candidate_value >= other_value
                strictly_better |= candidate_value > other_value
            else:
                at_least_as_good &= candidate_value <= other_value
                strictly_better |= candidate_value < other_value
        return at_least_as_good and strictly_better

    @staticmethod
    def _meets_constraints(run: ExperimentRun, constraints: ReportConstraints) -> bool:
        if run.status != "completed":
            return False
        limits: tuple[tuple[str, float | int | None], ...] = (
            ("p95_latency_seconds", constraints.max_p95_latency_seconds),
            ("wall_seconds", constraints.max_wall_seconds),
            ("usd", constraints.max_usd),
            ("model_size_bytes", constraints.max_model_size_bytes),
            ("loaded_memory_bytes", constraints.max_loaded_memory_bytes),
        )
        for name, limit in limits:
            if limit is None:
                continue
            value = run.fact(name)
            if value is None or value > limit:
                return False
        for metric, minimum in constraints.minimum_metrics.items():
            value = run.metrics.get(metric)
            if value is None or value < minimum:
                return False
        return True

    def to_markdown(self) -> str:
        """Render every row, including failed and missing matrix cells."""
        header = (
            "| architecture | benchmark | split | repeat | status | pair_f1 | "
            "bcubed_f1 | seconds | tokens | usd |\n"
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|"
        )
        rows = [header]
        for run in self.runs:
            token_total = None
            if (
                run.token_usage is not None
                and run.token_usage.input_tokens is not None
                and run.token_usage.output_tokens is not None
            ):
                token_total = run.token_usage.input_tokens + run.token_usage.output_tokens
            rows.append(
                "| "
                + " | ".join(
                    (
                        run.architecture,
                        run.benchmark_id,
                        run.split_id,
                        str(run.repeat_index),
                        run.status,
                        _format(run.metrics.get("pair_f1")),
                        _format(run.metrics.get("bcubed_f1")),
                        _format(run.wall_seconds),
                        _format(token_total),
                        _format(run.usd),
                    )
                )
                + " |"
            )
        if self.reproduce_command is not None:
            rows.extend(("", f"Reproduce: {self.reproduce_command}"))
        return "\n".join(rows)


def _format(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"
