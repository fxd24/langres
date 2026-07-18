"""Immutable experiment rows, compatible cohorts, constraints, and Pareto views."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from langres.experiments.identity import compute_evaluation_identity
from langres.experiments.measurements import FunnelFacts, StageMeasurement, TokenUsage
from langres.experiments.protocol import EvaluationProtocol, FrozenDict, freeze_mapping
from langres.experiments.statistics import SplitInstability, split_instability

RunStatus = Literal["running", "completed", "failed", "budget_exceeded", "missing"]
Direction = Literal["min", "max"]


class IncompatibleProtocolError(ValueError):
    """Raised when a comparison would mix incompatible evaluation cohorts."""


class ExperimentRun(BaseModel):
    """One immutable matrix cell, including failures and explicit missing cells."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False, validate_default=True)

    recipe_id: str
    evaluation_id: str
    attempt_id: str | None = None
    cache_id: str | None = None
    architecture: str
    variant_id: str = Field(min_length=1)
    benchmark_id: str
    split_id: str
    threshold_split_id: str | None = None
    evaluation_split_id: str | None = None
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

    @field_validator("metrics", mode="after")
    @classmethod
    def _freeze_metrics(cls, value: dict[str, float | None]) -> FrozenDict:
        return freeze_mapping(value)

    def fact(self, name: str) -> float | None:
        """Resolve a quality metric or standard resource/performance fact."""
        if name in self.metrics:
            return self.metrics[name]
        value = getattr(self, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None


class MetricConfidenceInterval(BaseModel):
    """Availability state for a statistically valid metric interval."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    method: str
    status: Literal["available", "unavailable", "insufficient"]
    lower: float | None = None
    upper: float | None = None
    confidence_level: float | None = None
    samples: int | None = None
    reason: str | None = None


class AggregateRow(BaseModel):
    """Aggregate with explicit completion/failure/missing denominators."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    architecture: str
    variant_id: str
    benchmark_id: str
    split_id: str
    cohort_id: str
    metric: str
    aggregation: Literal["mean", "median"]
    value: float | None
    standard_deviation: float | None
    completed: int
    observed: int
    failed: int
    missing: int
    total: int
    confidence_interval: MetricConfidenceInterval


class CohortView(BaseModel):
    """Rows that may be compared for performance."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    evaluation_id: str
    cohort_id: str
    runs: tuple[ExperimentRun, ...]


class ParetoRow(BaseModel):
    """One architecture aggregate inside a comparable benchmark/split cohort."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    architecture: str
    variant_id: str
    benchmark_id: str
    split_id: str
    cohort_id: str
    aggregation: Literal["mean", "median"]
    objectives: dict[str, float]
    completed: int
    observed: int
    failed: int
    missing: int
    total: int

    @field_validator("objectives", mode="after")
    @classmethod
    def _freeze_objectives(cls, value: dict[str, float]) -> FrozenDict:
        return freeze_mapping(value)

    def fact(self, name: str) -> float | None:
        return self.objectives.get(name)


class ReportConstraints(BaseModel):
    """Optional hard filters applied before ranking or Pareto analysis."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False, validate_default=True)

    max_p95_latency_seconds: float | None = Field(default=None, ge=0.0)
    max_wall_seconds: float | None = Field(default=None, ge=0.0)
    max_usd: float | None = Field(default=None, ge=0.0)
    max_model_size_bytes: int | None = Field(default=None, ge=0)
    max_loaded_memory_bytes: int | None = Field(default=None, ge=0)
    minimum_metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("minimum_metrics", mode="after")
    @classmethod
    def _freeze_minimum_metrics(cls, value: dict[str, float]) -> FrozenDict:
        return freeze_mapping(value)


class ExperimentReport(BaseModel):
    """All planned cells and derived views; failed/missing rows are never imputed."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    version: Literal[1] = 1
    evaluation_id: str = ""
    protocol: EvaluationProtocol
    runs: tuple[ExperimentRun, ...]
    reproduction_artifact: str | None = None

    @model_validator(mode="after")
    def _validate_evaluation(self) -> "ExperimentReport":
        derived_id = compute_evaluation_identity(self.protocol).evaluation_id
        if self.evaluation_id and self.evaluation_id != derived_id:
            raise ValueError(
                f"evaluation_id {self.evaluation_id!r} does not match protocol-derived "
                f"evaluation_id {derived_id!r}"
            )
        object.__setattr__(self, "evaluation_id", derived_id)
        logical_cells: set[tuple[str, str, str, str, int, int]] = set()
        for index, run in enumerate(self.runs):
            if run.evaluation_id != derived_id:
                raise ValueError(
                    f"runs[{index}].evaluation_id does not match protocol-derived "
                    f"evaluation_id {derived_id!r}"
                )
            if run.benchmark_id not in self.protocol.benchmark_ids:
                raise ValueError(
                    f"runs[{index}].benchmark_id={run.benchmark_id!r} is not in protocol "
                    f"benchmark_ids={self.protocol.benchmark_ids!r}"
                )
            if run.split_id not in self.protocol.split_ids:
                raise ValueError(
                    f"runs[{index}].split_id={run.split_id!r} is not in protocol "
                    f"split_ids={self.protocol.split_ids!r}"
                )
            if run.split_seed not in self.protocol.split_seeds:
                raise ValueError(
                    f"runs[{index}].split_seed={run.split_seed!r} is not in protocol "
                    f"split_seeds={self.protocol.split_seeds!r}"
                )
            planned_repeats = self.protocol.repeats_for(run.architecture)
            if run.repeat_index >= planned_repeats:
                raise ValueError(
                    f"runs[{index}].repeat_index={run.repeat_index} is not declared for "
                    f"architecture={run.architecture!r}; planned repeats={planned_repeats}"
                )
            if run.cohort_id != self.protocol.hardware_cohort:
                raise ValueError(
                    f"runs[{index}].cohort_id={run.cohort_id!r} does not match protocol "
                    f"hardware_cohort={self.protocol.hardware_cohort!r}"
                )
            logical_cell = (
                run.architecture,
                run.variant_id,
                run.benchmark_id,
                run.split_id,
                run.split_seed,
                run.repeat_index,
            )
            if logical_cell in logical_cells:
                raise ValueError(
                    "runs contain duplicate logical experiment cell "
                    f"{logical_cell!r}; retries belong in RunStore attempts, not report samples"
                )
            logical_cells.add(logical_cell)
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
        grouped: dict[tuple[str, str, str, str, str], list[ExperimentRun]] = defaultdict(list)
        for run in self.runs:
            grouped[
                (
                    run.architecture,
                    run.variant_id,
                    run.benchmark_id,
                    run.split_id,
                    run.cohort_id,
                )
            ].append(run)
        output: list[AggregateRow] = []
        for (
            architecture,
            variant_id,
            benchmark_id,
            split_id,
            cohort_id,
        ), rows in sorted(grouped.items()):
            values = [
                value
                for run in rows
                if run.status == "completed"
                for value in [run.metrics.get(metric)]
                if value is not None
            ]
            completed = sum(run.status == "completed" for run in rows)
            failed = sum(run.status in {"failed", "budget_exceeded"} for run in rows)
            missing = sum(run.status in {"missing", "running"} for run in rows)
            aggregate_value: float | None = None
            if values:
                aggregate_value = (
                    statistics.fmean(values)
                    if self.protocol.aggregation == "mean"
                    else statistics.median(values)
                )
            if self.protocol.confidence_interval_method == "none":
                interval = MetricConfidenceInterval(
                    method="none",
                    status="unavailable",
                    reason="confidence intervals are disabled by the evaluation protocol",
                )
            else:
                interval = MetricConfidenceInterval(
                    method=self.protocol.confidence_interval_method,
                    status="unavailable",
                    confidence_level=self.protocol.confidence_level,
                    samples=self.protocol.bootstrap_samples,
                    reason=(
                        "paired entity/cluster observations are unavailable on summary rows; "
                        "attach paired fixed-test-set inputs before computing this interval"
                    ),
                )
            output.append(
                AggregateRow(
                    architecture=architecture,
                    variant_id=variant_id,
                    benchmark_id=benchmark_id,
                    split_id=split_id,
                    cohort_id=cohort_id,
                    metric=metric,
                    aggregation=self.protocol.aggregation,
                    value=aggregate_value,
                    standard_deviation=statistics.stdev(values) if len(values) > 1 else None,
                    completed=completed,
                    observed=len(values),
                    failed=failed,
                    missing=missing,
                    total=len(rows),
                    confidence_interval=interval,
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
            key = f"{run.architecture}:{run.variant_id}:{run.benchmark_id}"
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
        benchmark_id: str | None = None,
        split_id: str | None = None,
        include_incomplete: bool = False,
    ) -> tuple[ParetoRow, ...]:
        """Return non-dominated architecture aggregates in one comparable slice."""
        if not objectives:
            raise ValueError("pareto requires at least one objective")
        invalid_directions = {
            name: direction
            for name, direction in objectives.items()
            if direction not in {"min", "max"}
        }
        if invalid_directions:
            raise ValueError(
                f"Pareto objective direction must be 'min' or 'max'; got {invalid_directions!r}"
            )
        standard_facts = {
            "wall_seconds",
            "p95_latency_seconds",
            "usd",
            "model_size_bytes",
            "loaded_memory_bytes",
        }
        available_names = standard_facts | {metric for run in self.runs for metric in run.metrics}
        unknown = sorted(set(objectives) - available_names)
        if unknown:
            raise ValueError(
                f"unknown Pareto objective(s) {unknown}; available objectives are "
                f"{sorted(available_names)}"
            )
        cohort_ids = {run.cohort_id for run in self.runs if run.status == "completed"}
        if cohort_id is None and len(cohort_ids) > 1:
            raise IncompatibleProtocolError(
                "Pareto comparison spans multiple hardware cohorts; pass cohort_id explicitly"
            )
        selected_cohort = cohort_id or (next(iter(cohort_ids)) if cohort_ids else None)
        if cohort_id is not None and cohort_id not in cohort_ids:
            raise IncompatibleProtocolError(
                f"cohort_id {cohort_id!r} is not present; available cohorts are "
                f"{sorted(cohort_ids)}"
            )
        completed = [
            run
            for run in self.runs
            if run.status == "completed"
            and (selected_cohort is None or run.cohort_id == selected_cohort)
        ]
        selected_benchmark = self._select_pareto_slice(
            field="benchmark_id",
            requested=benchmark_id,
            planned=self.protocol.benchmark_ids,
        )
        completed = [
            run
            for run in completed
            if selected_benchmark is None or run.benchmark_id == selected_benchmark
        ]
        selected_split = self._select_pareto_slice(
            field="split_id",
            requested=split_id,
            planned=self.protocol.split_ids,
        )
        completed = [
            run for run in completed if selected_split is None or run.split_id == selected_split
        ]

        selected_rows = [
            run
            for run in self.runs
            if (selected_cohort is None or run.cohort_id == selected_cohort)
            and (selected_benchmark is None or run.benchmark_id == selected_benchmark)
            and (selected_split is None or run.split_id == selected_split)
        ]
        grouped: dict[tuple[str, str], list[ExperimentRun]] = defaultdict(list)
        for run in selected_rows:
            grouped[(run.architecture, run.variant_id)].append(run)
        required_names = set(objectives)
        if constraints is not None:
            if constraints.max_p95_latency_seconds is not None:
                required_names.add("p95_latency_seconds")
            if constraints.max_wall_seconds is not None:
                required_names.add("wall_seconds")
            if constraints.max_usd is not None:
                required_names.add("usd")
            if constraints.max_model_size_bytes is not None:
                required_names.add("model_size_bytes")
            if constraints.max_loaded_memory_bytes is not None:
                required_names.add("loaded_memory_bytes")
            required_names.update(constraints.minimum_metrics)
        candidates: list[ParetoRow] = []
        for (architecture, variant_id), rows in sorted(grouped.items()):
            completed_rows = [run for run in rows if run.status == "completed"]
            failed = sum(run.status in {"failed", "budget_exceeded"} for run in rows)
            planned_repeats = self.protocol.repeats_for(architecture)
            expected_cells = {
                (split_seed, repeat_index)
                for split_seed in self.protocol.split_seeds
                for repeat_index in range(planned_repeats)
            }
            present_cells = {(run.split_seed, run.repeat_index) for run in rows}
            implied_missing = len(expected_cells - present_cells)
            missing = sum(run.status in {"missing", "running"} for run in rows) + implied_missing
            aggregated: dict[str, float] = {}
            observed = len(completed_rows)
            for name in sorted(required_names):
                values = [run.fact(name) for run in completed_rows]
                name_observed = sum(value is not None for value in values)
                observed = min(observed, name_observed)
                observed_values = [value for value in values if value is not None]
                if not observed_values:
                    break
                aggregated[name] = (
                    statistics.fmean(observed_values)
                    if self.protocol.aggregation == "mean"
                    else statistics.median(observed_values)
                )
            else:
                candidate = ParetoRow(
                    architecture=architecture,
                    variant_id=variant_id,
                    benchmark_id=rows[0].benchmark_id,
                    split_id=rows[0].split_id,
                    cohort_id=rows[0].cohort_id,
                    aggregation=self.protocol.aggregation,
                    objectives=aggregated,
                    completed=len(completed_rows),
                    observed=observed,
                    failed=failed,
                    missing=missing,
                    total=len(rows) + implied_missing,
                )
                complete = (
                    candidate.completed == candidate.total
                    and candidate.observed == candidate.completed
                )
                if (include_incomplete or complete) and (
                    constraints is None or self._pareto_meets_constraints(candidate, constraints)
                ):
                    candidates.append(candidate)

        front: list[ParetoRow] = []
        for candidate in candidates:
            if not any(
                other is not candidate and self._dominates(other, candidate, objectives)
                for other in candidates
            ):
                front.append(candidate)
        return tuple(front)

    @staticmethod
    def _dominates(
        candidate: ParetoRow,
        other: ParetoRow,
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
    def _select_pareto_slice(
        *,
        field: Literal["benchmark_id", "split_id"],
        requested: str | None,
        planned: tuple[str, ...],
    ) -> str | None:
        available = set(planned)
        if requested is not None:
            if requested not in available:
                raise IncompatibleProtocolError(
                    f"{field} {requested!r} is not present; available values are "
                    f"{sorted(available)}"
                )
            return requested
        if len(available) > 1:
            raise IncompatibleProtocolError(
                f"Pareto comparison spans multiple {field} values; pass {field} explicitly"
            )
        return next(iter(available)) if available else None

    @staticmethod
    def _pareto_meets_constraints(
        row: ParetoRow,
        constraints: ReportConstraints,
    ) -> bool:
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
            value = row.fact(name)
            if value is None or value > limit:
                return False
        for metric, minimum in constraints.minimum_metrics.items():
            value = row.fact(metric)
            if value is None or value < minimum:
                return False
        return True

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
            "| architecture | variant | benchmark | split | repeat | status | pair_f1 | "
            "bcubed_f1 | seconds | tokens | usd |\n"
            "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|"
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
                        run.variant_id,
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
