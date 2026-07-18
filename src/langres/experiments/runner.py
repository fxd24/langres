"""One architecture-first experiment matrix over the existing execution spine."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langres.benchmarks.runner import evaluate_execution_result
from langres.core.op import ExecutionEvent, ExecutionResult
from langres.core.model_ref import normalize_model_ref
from langres.core.resolver import ERModel
from langres.core.spend import BudgetExceeded, SpendMonitor
from langres.data.benchmark import Benchmark
from langres.data.registry import get_benchmark
from langres.experiments.cache import StageArtifactStore, ordered_input_fingerprint
from langres.experiments.identity import (
    CacheIdentityInput,
    ResourceSlotIdentity,
    compute_cache_identity,
    compute_evaluation_identity,
    compute_recipe_identity,
    detect_source_state,
)
from langres.experiments.measurements import StageMeasurement
from langres.experiments.protocol import EvaluationProtocol, expand_official_proof_matrix
from langres.experiments.report import ExperimentReport, ExperimentRun
from langres.tracking.runs import (
    RunContext,
    RunRecord,
    RunStore,
    capture_run,
    dataset_fingerprint,
    mint_attempt_id,
    resolve_store,
)
from langres.tracking.trackers import ExperimentTracker, TrackerSpec, resolve_tracker

CacheSemantics = Literal["deterministic", "seeded", "stochastic"]


class ExperimentConfigurationError(ValueError):
    """The experiment matrix cannot be executed as declared."""


@dataclass(frozen=True)
class ArchitectureFactory:
    """A named ``threshold -> ERModel`` architecture factory."""

    name: str
    factory: Callable[[float, SpendMonitor], ERModel]
    variant_id: str = "default"
    cache_semantics: CacheSemantics = "deterministic"

    def build(self, threshold: float, monitor: SpendMonitor) -> ERModel:
        model = self.factory(threshold, monitor)
        if not isinstance(model, ERModel):
            raise TypeError(
                f"architecture factory {self.name!r} returned "
                f"{type(model).__name__}, expected ERModel"
            )
        return model


class _PublicationSafeTracker:
    """Preserve local completion when optional publication fails."""

    def __init__(self, tracker: ExperimentTracker) -> None:
        self._tracker = tracker
        self.errors: list[str] = []
        self.name = tracker.name

    def _call(self, method: str, *args: Any, **kwargs: Any) -> None:
        try:
            getattr(self._tracker, method)(*args, **kwargs)
        except Exception as exc:
            self.errors.append(
                f"{method} failed with {type(exc).__name__}; publication incomplete"
            )

    def start_run(self, context: RunContext, *, run_name: str | None = None) -> None:
        self._call("start_run", context, run_name=run_name)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._call("log_params", params)

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None:
        self._call("log_metrics", metrics, step=step)

    def log_artifact(self, key: str, value: str) -> None:
        self._call("log_artifact", key, value)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._call("set_tags", tags)

    def finish(self, *, status: str) -> None:
        self._call("finish", status=status)

    @property
    def run_url(self) -> str | None:
        try:
            return self._tracker.run_url
        except Exception:
            self.errors.append("run_url failed; publication incomplete")
            return None

    @property
    def native(self) -> Any:
        return self._tracker.native


def flatten_numeric(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten nested numeric facts for tracker backends; omit bool/None/text."""
    flattened: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_numeric(value, prefix=name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                flattened[name] = number
    return flattened


def _resource_slots(
    plan: Any,
    *,
    boundary_index: int,
) -> tuple[ResourceSlotIdentity, ...]:
    """Project prefix model references into the strict cache identity contract."""
    slots: list[ResourceSlotIdentity] = []
    for step in plan.steps[:boundary_index]:
        if step.resource_ref is None:
            continue
        raw: object = step.resource_ref
        if step.resource_ref.startswith("{"):
            try:
                raw = json.loads(step.resource_ref)
            except ValueError:
                raw = step.resource_ref
        ref = normalize_model_ref(raw)  # type: ignore[arg-type]
        slots.append(
            ResourceSlotIdentity.from_model_ref(
                f"stage-{step.index}",
                ref,
            )
        )
    return tuple(slots)


def _measurements(result: ExecutionResult, *, cache_hit: bool) -> tuple[StageMeasurement, ...]:
    output: list[StageMeasurement] = []
    first_finished = True
    for event in result.events:
        if event.kind != "finish" or event.duration_seconds is None:
            continue
        count = event.output_count
        throughput = (
            count / event.duration_seconds
            if count is not None and event.duration_seconds > 0
            else None
        )
        output.append(
            StageMeasurement(
                stage_id=event.stage_id,
                operation_kind=event.role,
                wall_seconds=event.duration_seconds,
                items_in=event.input_count,
                items_out=event.output_count,
                pairs_in=event.input_count if event.index > 0 else None,
                pairs_out=event.output_count if event.role != "clusterer_stage" else None,
                throughput_per_second=throughput,
                cache_hit=cache_hit if first_finished else None,
            )
        )
        first_finished = False
    return tuple(output)


class Experiment:
    """Expand and execute an architecture x benchmark x split x seed matrix."""

    def __init__(
        self,
        *,
        architectures: Sequence[ArchitectureFactory],
        protocol: EvaluationProtocol,
        tracker: TrackerSpec = None,
        store: str | Path | RunStore | None = None,
        cache_dir: str | Path | None = None,
        budget_usd: float | None = None,
        resume: bool = True,
        fail_fast: bool = False,
    ) -> None:
        if not architectures:
            raise ExperimentConfigurationError("architectures must not be empty")
        if budget_usd is not None and budget_usd < 0:
            raise ExperimentConfigurationError("budget_usd must be non-negative")
        if protocol.paid_proof:
            expand_official_proof_matrix(protocol)
            expected = {cell.topology for cell in expand_official_proof_matrix(protocol)}
            actual = {architecture.name for architecture in architectures}
            if actual != expected:
                raise ExperimentConfigurationError(
                    "official paid proof requires exactly the four named recipes "
                    "plus CustomTopology"
                )
        allowed_splits = {"train", "test"}
        if protocol.paid_proof:
            allowed_splits.add("official")
        unknown_splits = set(protocol.split_ids) - allowed_splits
        if unknown_splits:
            raise ExperimentConfigurationError(
                "registry benchmarks expose honest 'train' and 'test' splits; "
                f"unsupported split ids: {sorted(unknown_splits)}"
            )
        if protocol.threshold_split_id != "train":
            raise ExperimentConfigurationError(
                "the current benchmark registry returns a train/test split; "
                "threshold_split_id must be 'train', not a validation alias"
            )
        if protocol.test_split_id != "test":
            raise ExperimentConfigurationError(
                "the current benchmark registry returns a train/test split; "
                "test_split_id must be 'test'"
            )
        self.architectures = tuple(architectures)
        self.protocol = protocol
        self.tracker = resolve_tracker(tracker)
        self.store = resolve_store(store)
        self.cache = StageArtifactStore(
            cache_dir
            if cache_dir is not None
            else (
                self.store.path.parent / "stage-cache"
                if self.store is not None
                else Path(".langres-cache")
            )
        )
        self.budget_usd = budget_usd if budget_usd is not None else protocol.budget_usd
        self._monitor = SpendMonitor(
            budget_usd=(
                self.budget_usd if self.budget_usd is not None else float("inf")
            )
        )
        self.resume = resume
        self.fail_fast = fail_fast

    def run(self) -> ExperimentReport:
        """Execute every independent cell, retaining failures and resumable rows."""
        evaluation_id = compute_evaluation_identity(self.protocol).evaluation_id
        source = detect_source_state()
        loaded: dict[str, tuple[Benchmark[Any], list[Any], list[set[str]], str]] = {}
        runs: list[ExperimentRun] = []
        for benchmark_id in self.protocol.benchmark_ids:
            benchmark = get_benchmark(benchmark_id)
            corpus, gold_clusters, gold_pairs = benchmark.load()
            fingerprint = dataset_fingerprint(corpus, [gold_clusters, gold_pairs])
            declared = self.protocol.dataset_fingerprints.get(benchmark_id)
            if declared is not None and declared != fingerprint:
                raise ExperimentConfigurationError(
                    f"dataset fingerprint mismatch for {benchmark_id!r}"
                )
            loaded[benchmark_id] = (benchmark, corpus, gold_clusters, fingerprint)

        for architecture in self.architectures:
            for benchmark_id in self.protocol.benchmark_ids:
                benchmark, corpus, gold_clusters, fingerprint = loaded[benchmark_id]
                for seed in self.protocol.split_seeds:
                    split = benchmark.split(corpus, gold_clusters, seed=seed)
                    train_records, test_records, train_clusters, test_clusters = split
                    split_data = {
                        "train": (train_records, train_clusters),
                        "test": (test_records, test_clusters),
                    }
                    if self.protocol.paid_proof:
                        split_data["official"] = (test_records, test_clusters)
                    for repeat_index in range(
                        self.protocol.repeats_for(architecture.name)
                    ):
                        for split_id in self.protocol.split_ids:
                            records, truth = split_data[split_id]
                            try:
                                run = self._run_cell(
                                    architecture,
                                    benchmark_id=benchmark_id,
                                    dataset_fingerprint_value=fingerprint,
                                    seed=seed,
                                    repeat_index=repeat_index,
                                    split_id=split_id,
                                    records=records,
                                    truth_clusters=truth,
                                    train_records=train_records,
                                    train_clusters=train_clusters,
                                    evaluation_id=evaluation_id,
                                    source=source,
                                )
                            except Exception as exc:
                                if self.fail_fast:
                                    raise
                                run = self._failed_cell(
                                    architecture,
                                    benchmark_id,
                                    split_id,
                                    seed,
                                    repeat_index,
                                    evaluation_id,
                                    budget_exceeded=isinstance(exc, BudgetExceeded),
                                )
                            runs.append(run)
        return ExperimentReport(protocol=self.protocol, runs=tuple(runs))

    def _context(
        self,
        architecture: ArchitectureFactory,
        *,
        benchmark_id: str,
        fingerprint: str,
        split_id: str,
        seed: int,
        repeat_index: int,
        plan: Mapping[str, Any],
    ) -> RunContext:
        return RunContext(
            experiment=f"{architecture.name}:{benchmark_id}:{split_id}",
            tags={
                "architecture": architecture.name,
                "variant_id": architecture.variant_id,
                "split_id": split_id,
                "repeat_index": str(repeat_index),
                "threshold_split_id": "train",
                "evaluation_split_id": (
                    "test" if split_id == "official" else split_id
                ),
            },
            resolver_config={"execution_plan": dict(plan)},
            budget_usd=self.budget_usd,
            method=architecture.name,
            dataset_name=benchmark_id,
            dataset_fingerprint=fingerprint,
            split_id=split_id,
            seeds={"split": seed, "repeat": repeat_index},
        )

    def _run_cell(
        self,
        architecture: ArchitectureFactory,
        *,
        benchmark_id: str,
        dataset_fingerprint_value: str,
        seed: int,
        repeat_index: int,
        split_id: str,
        records: Sequence[Any],
        truth_clusters: list[set[str]],
        train_records: Sequence[Any],
        train_clusters: list[set[str]],
        evaluation_id: str,
        source: Any,
    ) -> ExperimentRun:
        initial = architecture.build(
            self.protocol.threshold_grid[0], self._monitor
        )
        plan = initial.execution_plan()
        context = self._context(
            architecture,
            benchmark_id=benchmark_id,
            fingerprint=dataset_fingerprint_value,
            split_id=split_id,
            seed=seed,
            repeat_index=repeat_index,
            plan=plan.model_dump(mode="json"),
        )
        recipe_id = compute_recipe_identity(context).recipe_id
        existing = self._completed(recipe_id)
        if self.resume and existing is not None:
            return self._from_record(existing, architecture, evaluation_id)
        parent = self._latest(recipe_id)
        if parent is not None:
            context = context.model_copy(
                update={"parent_run_id": parent.attempt_id}
            )
        attempt_id = mint_attempt_id(recipe_id)
        input_fp = ordered_input_fingerprint(records)
        cache_id = self._cache_id(
            architecture,
            initial,
            source=source,
            input_fingerprint=input_fp,
            seed=seed,
            repeat_index=repeat_index,
            attempt_id=attempt_id,
        )
        safe_tracker = _PublicationSafeTracker(self.tracker)
        started = time.perf_counter()
        starting_spend = self._monitor.spent
        with capture_run(
            context,
            store=self.store,
            tracker=safe_tracker,
            recipe_id=recipe_id,
            evaluation_id=evaluation_id,
            cache_id=cache_id,
            protocol=self.protocol.model_dump(mode="json"),
            attempt_id=attempt_id,
            suppress_error_details=True,
        ) as handle:
            selected_threshold = self._tune_threshold(
                architecture,
                train_records,
                train_clusters,
                source=source,
                seed=seed,
                repeat_index=repeat_index,
                attempt_id=attempt_id,
            )
            model = architecture.build(selected_threshold, self._monitor)
            safe_tracker.log_params(
                {
                    "architecture": architecture.name,
                    "variant_id": architecture.variant_id,
                    "benchmark_id": benchmark_id,
                    "split_id": split_id,
                    "seed": seed,
                    "repeat_index": repeat_index,
                    "selected_threshold": selected_threshold,
                }
            )
            result, cache_hit = self._execute_cached(
                model,
                records,
                cache_id=cache_id,
                input_fingerprint=input_fp,
            )
            quality = evaluate_execution_result(
                result,
                records,
                truth_clusters,
                threshold=selected_threshold,
            ).model_dump()
            measurements = _measurements(result, cache_hit=cache_hit)
            numeric = flatten_numeric({"quality": quality})
            numeric["selected_threshold"] = selected_threshold
            handle.log_metrics(numeric, headline_metric=quality.get("pair_f1"))
            handle.record_measurements(
                measurement.model_dump(mode="json") for measurement in measurements
            )
            handle.record_cost(self._monitor.spent - starting_spend)
        wall = time.perf_counter() - started
        warnings = tuple(dict.fromkeys(safe_tracker.errors))
        return ExperimentRun(
            recipe_id=recipe_id,
            evaluation_id=evaluation_id,
            attempt_id=attempt_id,
            cache_id=cache_id,
            architecture=architecture.name,
            variant_id=architecture.variant_id,
            benchmark_id=benchmark_id,
            split_id=split_id,
            threshold_split_id="train",
            evaluation_split_id="test" if split_id == "official" else split_id,
            split_seed=seed,
            repeat_index=repeat_index,
            status="completed",
            cohort_id=self.protocol.hardware_cohort,
            metrics={key: float(value) for key, value in quality.items()},
            measurements=measurements,
            wall_seconds=wall,
            usd=self._monitor.spent - starting_spend,
            warnings=warnings,
        )

    def _tune_threshold(
        self,
        architecture: ArchitectureFactory,
        records: Sequence[Any],
        truth_clusters: list[set[str]],
        *,
        source: Any,
        seed: int,
        repeat_index: int,
        attempt_id: str,
    ) -> float:
        best_threshold = self.protocol.threshold_grid[0]
        best_f1 = -1.0
        first = architecture.build(best_threshold, self._monitor)
        input_fp = ordered_input_fingerprint(records)
        cache_id = self._cache_id(
            architecture,
            first,
            source=source,
            input_fingerprint=input_fp,
            seed=seed,
            repeat_index=repeat_index,
            attempt_id=attempt_id,
        )
        first_result, _ = self._execute_cached(
            first, records, cache_id=cache_id, input_fingerprint=input_fp
        )
        for index, threshold in enumerate(self.protocol.threshold_grid):
            model = architecture.build(threshold, self._monitor)
            if index == 0:
                result = first_result
            elif first_result.checkpoint is not None:
                result = model.execute_from(
                    first_result.checkpoint,
                    cache_id=cache_id,
                    input_fingerprint=input_fp,
                )
            else:
                result = model.execute(
                    [record.model_dump() for record in records]
                )
            evaluated = evaluate_execution_result(
                result, records, truth_clusters, threshold=threshold
            )
            if evaluated.bcubed_f1 > best_f1:
                best_f1 = evaluated.bcubed_f1
                best_threshold = threshold
        return best_threshold

    def _cache_id(
        self,
        architecture: ArchitectureFactory,
        model: ERModel,
        *,
        source: Any,
        input_fingerprint: str,
        seed: int,
        repeat_index: int,
        attempt_id: str,
    ) -> str:
        plan = model.execution_plan()
        if plan.replay_prefix_id is None or plan.replay_boundary is None:
            plan_payload = plan.model_dump(mode="json")
            plan_id = hashlib.sha256(
                json.dumps(
                    plan_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            return compute_cache_identity(
                CacheIdentityInput(
                    stage_id="no-replay-boundary",
                    execution_plan_id=plan_id,
                    operation_identity={"plan": plan_payload},
                    resource_slots=_resource_slots(
                        plan, boundary_index=len(plan.steps)
                    ),
                    source=source,
                    semantics=architecture.cache_semantics,
                    input_fingerprint=input_fingerprint,
                    seed=seed if architecture.cache_semantics == "seeded" else None,
                    repeat_index=(
                        repeat_index if architecture.cache_semantics == "stochastic" else None
                    ),
                    attempt_id=(
                        attempt_id if architecture.cache_semantics == "stochastic" else None
                    ),
                    official=self.protocol.publication_profile == "official",
                )
            ).cache_id
        boundary_index = next(
            step.index for step in plan.steps if step.stage_id == plan.replay_boundary
        )
        return compute_cache_identity(
            CacheIdentityInput(
                stage_id=plan.replay_boundary,
                execution_plan_id=plan.replay_prefix_id,
                operation_identity={
                    "steps": [
                        step.model_dump(mode="json")
                        for step in plan.steps[:boundary_index]
                    ]
                },
                resource_slots=_resource_slots(plan, boundary_index=boundary_index),
                source=source,
                semantics=architecture.cache_semantics,
                input_fingerprint=input_fingerprint,
                seed=seed if architecture.cache_semantics == "seeded" else None,
                repeat_index=(
                    repeat_index if architecture.cache_semantics == "stochastic" else None
                ),
                attempt_id=attempt_id if architecture.cache_semantics == "stochastic" else None,
                official=self.protocol.publication_profile == "official",
            )
        ).cache_id

    def _execute_cached(
        self,
        model: ERModel,
        records: Sequence[Any],
        *,
        cache_id: str,
        input_fingerprint: str,
    ) -> tuple[ExecutionResult, bool]:
        plan = model.execution_plan()
        record_dicts = [record.model_dump() for record in records]
        if plan.replay_prefix_id is None or plan.replay_boundary is None:
            return model.execute(record_dicts), False
        boundary_index = next(
            step.index for step in plan.steps if step.stage_id == plan.replay_boundary
        )
        checkpoint = self.cache.load(
            cache_id,
            prefix_plan_id=plan.replay_prefix_id,
            boundary_index=boundary_index,
            input_fingerprint=input_fingerprint,
        )
        if checkpoint is not None:
            replayed = model.execute_from(
                    checkpoint,
                    cache_id=cache_id,
                    input_fingerprint=input_fingerprint,
                )
            return replayed.model_copy(update={"checkpoint": checkpoint}), True
        result = model.execute(
            record_dicts,
            checkpoint_cache_id=cache_id,
            input_fingerprint=input_fingerprint,
        )
        if result.checkpoint is None:
            raise RuntimeError("declared replay boundary did not emit a checkpoint")
        self.cache.put(result.checkpoint)
        return result, False

    def _completed(self, recipe_id: str) -> RunRecord | None:
        latest = self._latest(recipe_id, status="completed")
        return latest

    def _latest(
        self,
        recipe_id: str,
        *,
        status: str | None = None,
    ) -> RunRecord | None:
        if self.store is None:
            return None
        return next(
            (
                record
                for record in reversed(self.store.read())
                if record.recipe_id == recipe_id
                and (status is None or record.status == status)
            ),
            None,
        )

    def _from_record(
        self,
        record: RunRecord,
        architecture: ArchitectureFactory,
        evaluation_id: str,
    ) -> ExperimentRun:
        metrics: dict[str, float | None] = {
            key.removeprefix("quality."): float(value)
            for key, value in (record.metrics or {}).items()
            if key.startswith("quality.") and isinstance(value, (int, float))
        }
        return ExperimentRun(
            recipe_id=record.recipe_id,
            evaluation_id=evaluation_id,
            attempt_id=record.attempt_id,
            cache_id=record.cache_id,
            architecture=architecture.name,
            variant_id=architecture.variant_id,
            benchmark_id=record.context.dataset_name or "",
            split_id=record.context.split_id or "",
            threshold_split_id=record.context.tags.get("threshold_split_id", "train"),
            evaluation_split_id=record.context.tags.get(
                "evaluation_split_id", record.context.split_id or ""
            ),
            split_seed=int(record.context.seeds.get("split", 0)),
            repeat_index=int(record.context.seeds.get("repeat", 0)),
            status="completed",
            cohort_id=self.protocol.hardware_cohort,
            metrics=metrics,
            wall_seconds=record.duration_seconds,
            warnings=("resumed from completed RunStore attempt",),
        )

    def _failed_cell(
        self,
        architecture: ArchitectureFactory,
        benchmark_id: str,
        split_id: str,
        seed: int,
        repeat_index: int,
        evaluation_id: str,
        *,
        budget_exceeded: bool = False,
    ) -> ExperimentRun:
        return ExperimentRun(
            recipe_id=f"failed:{architecture.name}:{benchmark_id}:{split_id}:{seed}:{repeat_index}",
            evaluation_id=evaluation_id,
            architecture=architecture.name,
            variant_id=architecture.variant_id,
            benchmark_id=benchmark_id,
            split_id=split_id,
            threshold_split_id="train",
            evaluation_split_id="test" if split_id == "official" else split_id,
            split_seed=seed,
            repeat_index=repeat_index,
            status="budget_exceeded" if budget_exceeded else "failed",
            cohort_id=self.protocol.hardware_cohort,
            error_type=(
                "BudgetExceeded" if budget_exceeded else "ExperimentCellError"
            ),
            error_message="cell failed; exception details suppressed",
        )


__all__ = [
    "ArchitectureFactory",
    "Experiment",
    "ExperimentConfigurationError",
    "flatten_numeric",
]
