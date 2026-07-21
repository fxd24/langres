"""One architecture-first experiment matrix over the existing execution spine."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from langres._version import __version__
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
from langres.experiments.measurements import (
    EmbeddingFacts,
    FunnelFacts,
    PriceSnapshot,
    RuntimeFacts,
    StageMeasurement,
    TokenUsage,
)
from langres.experiments.protocol import (
    STOCHASTIC_TOPOLOGIES,
    EvaluationProtocol,
    expand_official_proof_matrix,
)
from langres.experiments.report import (
    ExperimentPlan,
    ExperimentReport,
    ExperimentRun,
    PlannedExperimentCell,
)
from langres.resources.base import GenerationEnvelope, UnknownGenerationCostError
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
MeasurementPhase = Literal["tuning", "evaluation", "partial"]


class ExperimentConfigurationError(ValueError):
    """The experiment matrix cannot be executed as declared."""


class ExperimentCellError(RuntimeError):
    """A sanitized architecture/cell failure with remediation context."""


@dataclass(frozen=True)
class _TuningOutcome:
    threshold: float
    executions: tuple[tuple[ExecutionResult, bool], ...]


@dataclass(frozen=True)
class ArchitectureFactory:
    """A named ``threshold -> ERModel`` architecture factory."""

    name: str
    factory: Callable[[float, SpendMonitor], ERModel]
    variant_id: str = "default"
    cache_semantics: CacheSemantics = "deterministic"
    estimated_usd: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ArchitectureFactory.name must not be empty")
        if not self.variant_id:
            raise ValueError("ArchitectureFactory.variant_id must not be empty")
        if self.estimated_usd is not None and self.estimated_usd < 0:
            raise ValueError("ArchitectureFactory.estimated_usd must be non-negative")

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
        self._finished = False

    def _call(self, method: str, *args: Any, **kwargs: Any) -> None:
        try:
            getattr(self._tracker, method)(*args, **kwargs)
        except Exception as exc:
            self.errors.append(f"{method} failed with {type(exc).__name__}; publication incomplete")

    def start_run(self, context: RunContext, *, run_name: str | None = None) -> None:
        self._call("start_run", context, run_name=run_name)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._call("log_params", params)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        self._call("log_metrics", metrics, step=step)

    def log_artifact(self, key: str, value: str) -> None:
        self._call("log_artifact", key, value)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._call("set_tags", tags)

    def finish(self, *, status: str) -> None:
        if self._finished:
            return
        self._finished = True
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


def _reproduction_artifact_name(index: int, architecture_name: str) -> str:
    """Return a path-safe, stable directory name for one saved architecture."""
    slug = "".join(
        character.lower() if character.isalnum() else "-" for character in architecture_name
    )
    slug = "-".join(part for part in slug.split("-") if part)
    return f"{index:03d}-{slug or 'architecture'}"


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


def _resource_slot(role: str) -> str | None:
    return {
        "retrieve": "embedder",
        "rerank": "reranker",
        "generate": "llm",
    }.get(role)


def _runtime_facts(
    *,
    cohort: str,
    config: Mapping[str, Any] | None,
) -> RuntimeFacts:
    values = config or {}
    return RuntimeFacts(
        hardware_cohort=cohort,
        host=platform.node() or None,
        operating_system=platform.platform(),
        python_version=platform.python_version(),
        langres_version=__version__,
        cpu=platform.processor() or platform.machine() or None,
        device=values.get("device") if isinstance(values.get("device"), str) else None,
        dtype=values.get("dtype") if isinstance(values.get("dtype"), str) else None,
        quantization=(
            values.get("quantization") if isinstance(values.get("quantization"), str) else None
        ),
        batch_size=(
            values.get("batch_size") if isinstance(values.get("batch_size"), int) else None
        ),
        worker_count=(
            values.get("worker_count") if isinstance(values.get("worker_count"), int) else None
        ),
    )


def _measurement_rows(result: ExecutionResult) -> tuple[Any, ...]:
    if result.accounting_rows:
        return result.accounting_rows
    if result.checkpoint is not None:
        return result.checkpoint.rows
    return tuple(result.pairs.rows)


def _generated(result: ExecutionResult) -> bool:
    return any(event.kind == "finish" and event.role == "generate" for event in result.events)


def _token_usage(result: ExecutionResult) -> TokenUsage | None:
    if not _generated(result):
        return None
    usages = [
        row.provenance["usage"]
        for row in _measurement_rows(result)
        if isinstance(row.provenance.get("usage"), Mapping)
    ]
    if not usages:
        return None

    def total(name: str) -> int | None:
        count = 0
        for usage in usages:
            value = usage.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                return None
            count += value
        return count

    first = usages[0]
    provider = first.get("provider")
    model = first.get("model")
    provider_usage: dict[str, dict[str, str | int | float | bool | None]] = {}
    if isinstance(provider, str) or isinstance(model, str):
        provider_usage["generation"] = {
            "provider": provider if isinstance(provider, str) else None,
            "model": model if isinstance(model, str) else None,
        }
    return TokenUsage(
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        cache_read_input_tokens=total("cache_read_input_tokens"),
        cache_creation_input_tokens=total("cache_creation_input_tokens"),
        reasoning_output_tokens=total("reasoning_tokens"),
        provider_usage=provider_usage,
    )


def _merge_token_usage(usages: Sequence[TokenUsage | None]) -> TokenUsage | None:
    observed = [usage for usage in usages if usage is not None]
    if not observed:
        return None

    def total(field: str) -> int | None:
        values = [getattr(usage, field) for usage in observed]
        if any(value is None for value in values):
            return None
        return cast(int, sum(value for value in values if value is not None))

    providers: dict[str, dict[str, str | int | float | bool | None]] = {}
    for index, usage in enumerate(observed):
        for name, facts in usage.provider_usage.items():
            providers[f"{name}-{index}"] = dict(facts)
    return TokenUsage(
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        cache_read_input_tokens=total("cache_read_input_tokens"),
        cache_creation_input_tokens=total("cache_creation_input_tokens"),
        reasoning_output_tokens=total("reasoning_output_tokens"),
        provider_usage=providers,
    )


def _cost_complete(result: ExecutionResult) -> bool:
    if not _generated(result):
        return True
    for row in _measurement_rows(result):
        if row.provenance.get("cost_unknown") is True:
            return False
        if row.provenance.get("cost_required") is True and not isinstance(
            row.provenance.get("cost_usd"), (int, float)
        ):
            return False
    return True


def _observed_usd(result: ExecutionResult) -> float | None:
    if not _generated(result) or not _cost_complete(result):
        return None
    rows = _measurement_rows(result)
    costs = [row.provenance.get("cost_usd") for row in rows]
    measured = [float(value) for value in costs if isinstance(value, (int, float))]
    return sum(measured)


def _embedding_facts(
    result: ExecutionResult,
    *,
    vectors_produced: int | None,
) -> EmbeddingFacts | None:
    for row in _measurement_rows(result):
        retrieve = row.provenance.get("retrieve")
        if not isinstance(retrieve, Mapping):
            continue
        facts = retrieve.get("embedding")
        if not isinstance(facts, Mapping):
            continue
        dimension = facts.get("dimension")
        dtype = facts.get("dtype")
        dimensions = dimension if isinstance(dimension, int) else None
        dtype_name = dtype if isinstance(dtype, str) else None
        bytes_per_scalar = {
            "float16": 2,
            "float32": 4,
            "float64": 8,
            "int8": 1,
        }.get(dtype_name or "")
        bytes_per_vector = (
            dimensions * bytes_per_scalar
            if dimensions is not None and bytes_per_scalar is not None
            else None
        )
        return EmbeddingFacts(
            dimensions=dimensions,
            dtype=dtype_name,
            vectors_produced=vectors_produced,
            bytes_per_vector=bytes_per_vector,
            total_vector_bytes=(
                vectors_produced * bytes_per_vector
                if vectors_produced is not None and bytes_per_vector is not None
                else None
            ),
        )
    return None


def _price_snapshot(
    resource_ref: str | None,
    price_snapshots: Mapping[str, PriceSnapshot],
) -> PriceSnapshot | None:
    """Resolve a tariff by exact resource identity, never substring overlap."""
    if resource_ref is None:
        return None
    identities = [resource_ref]
    try:
        payload = json.loads(resource_ref)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, Mapping) and isinstance(payload.get("base"), str):
        identities.append(payload["base"])
    return next(
        (price_snapshots[identity] for identity in identities if identity in price_snapshots),
        None,
    )


def _measurements(
    result: ExecutionResult,
    *,
    cache_hit: bool,
    hardware_cohort: str,
    resource_runtime: Mapping[str, Mapping[str, Any]],
    price_snapshots: Mapping[str, PriceSnapshot],
    phase: MeasurementPhase,
) -> tuple[StageMeasurement, ...]:
    output: list[StageMeasurement] = []
    first_finished = True
    usage = _token_usage(result)
    observed_usd = _observed_usd(result)
    steps = {step.stage_id: step for step in result.plan.steps}
    for event in result.events:
        if event.kind != "finish" or event.duration_seconds is None:
            continue
        count = event.output_count
        throughput = (
            count / event.duration_seconds
            if count is not None and event.duration_seconds > 0
            else None
        )
        slot = _resource_slot(event.role)
        step = steps[event.stage_id]
        price = None
        derived_usd = None
        if slot == "llm":
            price = _price_snapshot(step.resource_ref, price_snapshots)
            if price is not None and usage is not None:
                derived_usd = price.reprice(
                    usage,
                    requests=event.input_count,
                ).amount
        output.append(
            StageMeasurement(
                stage_id=event.stage_id,
                operation_kind=event.role,
                phase=phase,
                wall_seconds=event.duration_seconds,
                items_in=event.input_count,
                items_out=event.output_count,
                pairs_in=event.input_count if event.index > 0 else None,
                pairs_out=event.output_count if event.role != "clusterer_stage" else None,
                throughput_per_second=throughput,
                cache_hit=cache_hit if first_finished else None,
                resource_slot=slot,
                resource_id=step.resource_ref,
                usage=usage if slot == "llm" else None,
                embedding=(
                    _embedding_facts(result, vectors_produced=event.input_count)
                    if slot == "embedder"
                    else None
                ),
                runtime=(
                    _runtime_facts(
                        cohort=hardware_cohort,
                        config=resource_runtime.get(slot, {}),
                    )
                    if slot is not None
                    else None
                ),
                price=price,
                observed_usd=observed_usd if slot == "llm" else None,
                derived_usd=derived_usd,
                external_calls=event.input_count if slot == "llm" else None,
            )
        )
        first_finished = False
    return tuple(output)


def _funnel(result: ExecutionResult, *, record_count: int) -> FunnelFacts:
    finished = [event for event in result.events if event.kind == "finish"]

    def first(role: str, field: Literal["input_count", "output_count"]) -> int | None:
        event = next((item for item in finished if item.role == role), None)
        return getattr(event, field) if event is not None else None

    rows = _measurement_rows(result)
    parse_seen = any(event.role == "parse" for event in finished) or any(
        "parse_error" in row.provenance for row in rows
    )
    parsed_abstentions = (
        sum(row.provenance.get("parse_error") is True for row in rows) if parse_seen else None
    )
    select_counts = tuple(
        event.output_count
        for event in finished
        if event.role.endswith("select") and event.output_count is not None
    )
    return FunnelFacts(
        possible_pairs=record_count * (record_count - 1) // 2,
        retrieved_pairs=first("retrieve", "output_count"),
        pairs_after_select=select_counts,
        reranker_pairs=first("rerank", "input_count"),
        llm_pairs=first("generate", "input_count"),
        parsed_abstentions=parsed_abstentions,
        selected_match_edges=(select_counts[-1] if select_counts else len(result.pairs.rows)),
        clusters_produced=first("clusterer_stage", "output_count"),
    )


def _envelope_usage(outputs: Sequence[GenerationEnvelope]) -> TokenUsage | None:
    usages = [
        TokenUsage(
            input_tokens=output.usage.input_tokens,
            output_tokens=output.usage.output_tokens,
            cache_read_input_tokens=output.usage.cache_read_input_tokens,
            cache_creation_input_tokens=output.usage.cache_creation_input_tokens,
            reasoning_output_tokens=output.usage.reasoning_tokens,
            provider_usage={
                "generation": {
                    "provider": output.usage.provider,
                    "model": output.usage.model,
                }
            },
        )
        for output in outputs
        if output.usage is not None
    ]
    return _merge_token_usage(usages)


def _partial_generation_facts(
    error: BudgetExceeded | UnknownGenerationCostError,
    *,
    hardware_cohort: str,
    resource_runtime: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[StageMeasurement, ...], TokenUsage | None, float | None]:
    outputs = tuple(output for output in error.outputs if isinstance(output, GenerationEnvelope))
    if not outputs:
        return (), None, None
    usage = _envelope_usage(outputs)
    costs = [output.cost_usd for output in outputs]
    cost = (
        None
        if any(value is None for value in costs)
        else sum(value for value in costs if value is not None)
    )
    models = {output.model_ref.base for output in outputs}
    resource_id = next(iter(models)) if len(models) == 1 else None
    return (
        (
            StageMeasurement(
                stage_id="partial-generate",
                operation_kind="generate",
                phase="partial",
                wall_seconds=None,
                items_out=len(outputs),
                pairs_out=len(outputs),
                resource_slot="llm",
                resource_id=resource_id,
                usage=usage,
                runtime=_runtime_facts(
                    cohort=hardware_cohort,
                    config=resource_runtime.get("llm", {}),
                ),
                observed_usd=cost,
                external_calls=len(outputs),
                warnings=("generation stopped after the budget boundary",),
            ),
        ),
        usage,
        cost,
    )


def _tuning_measurements(
    tuning: _TuningOutcome | None,
    *,
    hardware_cohort: str,
    resource_runtime: Mapping[str, Mapping[str, Any]],
    price_snapshots: Mapping[str, PriceSnapshot],
) -> tuple[StageMeasurement, ...]:
    """Materialize completed tuning facts available before a later failure."""
    if tuning is None:
        return ()
    return tuple(
        measurement
        for execution, cache_hit in tuning.executions
        for measurement in _measurements(
            execution,
            cache_hit=cache_hit,
            hardware_cohort=hardware_cohort,
            resource_runtime=resource_runtime,
            price_snapshots=price_snapshots,
            phase="tuning",
        )
    )


def _tuning_usage(tuning: _TuningOutcome | None) -> tuple[TokenUsage | None, ...]:
    """Return usage observations from completed threshold-tuning executions."""
    if tuning is None:
        return ()
    return tuple(_token_usage(execution) for execution, _cache_hit in tuning.executions)


def _experiment_facts(
    *,
    funnel: FunnelFacts | None,
    token_usage: TokenUsage | None,
    selected_threshold: float | None,
    usd: float | None,
    warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "funnel": funnel.model_dump(mode="json") if funnel is not None else None,
        "token_usage": (token_usage.model_dump(mode="json") if token_usage is not None else None),
        "selected_threshold": selected_threshold,
        "usd": usd,
        "warnings": list(warnings),
    }


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
        price_snapshots: Mapping[str, PriceSnapshot] | None = None,
        reproduction_path: str | Path | None = None,
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
            if actual != expected or len(architectures) != len(expected):
                raise ExperimentConfigurationError(
                    "official paid proof requires exactly the four named recipes "
                    "plus CustomTopology, with no duplicate factories"
                )
            deterministic_repeats = sorted(
                architecture.name
                for architecture in architectures
                if architecture.name in STOCHASTIC_TOPOLOGIES
                and architecture.cache_semantics != "stochastic"
            )
            if deterministic_repeats:
                raise ExperimentConfigurationError(
                    "official paid proof requires cache_semantics='stochastic' "
                    "for repeated LLM topologies; invalid factories: "
                    f"{deterministic_repeats}"
                )
        logical_architectures = [
            (architecture.name, architecture.variant_id) for architecture in architectures
        ]
        if len(logical_architectures) != len(set(logical_architectures)):
            raise ExperimentConfigurationError(
                "architecture factories must use unique (name, variant_id) pairs"
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
        self.price_snapshots = dict(price_snapshots or {})
        self._monitor = SpendMonitor(
            budget_usd=(self.budget_usd if self.budget_usd is not None else float("inf"))
        )
        self.resume = resume
        self.fail_fast = fail_fast
        self.reproduction_path = Path(reproduction_path) if reproduction_path is not None else None

    def plan(self) -> ExperimentPlan:
        """Return the expanded matrix without loading data or executing inference."""
        source = detect_source_state()
        cache_contains_entries = self.cache.root.is_dir() and any(
            path.is_dir() and path.name != "quarantine" for path in self.cache.root.iterdir()
        )

        cells: list[PlannedExperimentCell] = []
        deterministic_attempts = 0
        stochastic_attempts = 0
        cache_hits = 0
        cache_misses = 0
        cache_unknown = 0
        cache_not_replayable = 0
        estimated_usd = 0.0
        estimate_complete = True
        for architecture in self.architectures:
            try:
                model = architecture.build(self.protocol.threshold_grid[0], self._monitor)
            except Exception as exc:
                raise ExperimentConfigurationError(
                    f"Cannot initialize architecture {architecture.name!r} during preflight. "
                    f"Cause: {type(exc).__name__}; configuration could not be inspected. "
                    "Fix: install the resource's optional extra or pass a locally available "
                    "resource, then retry."
                ) from exc
            replayable = model.execution_plan().replay_boundary is not None
            repeats = self.protocol.repeats_for(architecture.name)
            attempts_per_architecture = (
                len(self.protocol.benchmark_ids)
                * len(self.protocol.split_ids)
                * len(self.protocol.split_seeds)
                * repeats
            )
            if architecture.cache_semantics == "stochastic":
                stochastic_attempts += attempts_per_architecture
            else:
                deterministic_attempts += attempts_per_architecture
            if architecture.estimated_usd is None:
                estimate_complete = False
            else:
                estimated_usd += architecture.estimated_usd * attempts_per_architecture

            for benchmark_id in self.protocol.benchmark_ids:
                for split_id in self.protocol.split_ids:
                    for seed in self.protocol.split_seeds:
                        if not replayable:
                            status: Literal["hit", "miss", "unknown", "not_replayable"] = (
                                "not_replayable"
                            )
                            cache_not_replayable += repeats
                        elif cache_contains_entries:
                            # Input fingerprints are intentionally unavailable in
                            # side-effect-free preflight. Existing directories may
                            # belong to another dataset/plan and are never counted
                            # as hits by quantity.
                            status = "unknown"
                            cache_unknown += repeats
                        else:
                            status = "miss"
                            cache_misses += repeats
                        cells.append(
                            PlannedExperimentCell(
                                architecture=architecture.name,
                                variant_id=architecture.variant_id,
                                benchmark_id=benchmark_id,
                                split_id=split_id,
                                split_seed=seed,
                                repeats=repeats,
                                cache_status=status,
                                estimated_usd=(
                                    None
                                    if architecture.estimated_usd is None
                                    else architecture.estimated_usd * repeats
                                ),
                            )
                        )

        total_attempts = deterministic_attempts + stochastic_attempts
        estimate = estimated_usd if estimate_complete else None
        if self.protocol.paid_proof and estimate is None:
            unknown = sorted(
                architecture.name
                for architecture in self.architectures
                if architecture.estimated_usd is None
            )
            raise ExperimentConfigurationError(
                "Cannot start official paid proof without a complete USD preflight "
                f"estimate; missing estimates for {unknown}. "
                "Fix: declare ArchitectureFactory.estimated_usd for every topology."
            )
        if estimate is not None and self.budget_usd is not None and estimate > self.budget_usd:
            raise ExperimentConfigurationError(
                f"Cannot start experiment: estimated maximum USD {estimate:.2f} exceeds "
                f"the hard cap USD {self.budget_usd:.2f}. "
                "Fix: reduce the matrix/model estimates or raise the explicit budget."
            )
        publication_reasons: list[str] = []
        if source.git_dirty:
            publication_reasons.append("source tree is dirty")
        if self.tracker.name not in {"trackio", "multi"}:
            publication_reasons.append("Trackio publication is not configured")
        return ExperimentPlan(
            cells=tuple(cells),
            topology_count=len(self.architectures),
            benchmark_count=len(self.protocol.benchmark_ids),
            cell_count=len(cells),
            deterministic_attempts=deterministic_attempts,
            stochastic_attempts=stochastic_attempts,
            total_attempts=total_attempts,
            estimated_usd=estimate,
            budget_usd=self.budget_usd,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_unknown=cache_unknown,
            cache_not_replayable=cache_not_replayable,
            publication_profile=self.protocol.publication_profile,
            publication_eligible=not publication_reasons,
            publication_reasons=tuple(publication_reasons),
        )

    def run(self, *, dry_run: bool = False) -> ExperimentReport | ExperimentPlan:
        """Execute every independent cell, retaining failures and resumable rows."""
        if dry_run:
            return self.plan()
        self._validate_budget_preflight()
        evaluation_id = compute_evaluation_identity(self.protocol).evaluation_id
        ignored_artifacts: list[Path] = [self.cache.root]
        if self.store is not None:
            ignored_artifacts.append(self.store.path)
        source = detect_source_state(ignored_paths=ignored_artifacts)
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
                    for repeat_index in range(self.protocol.repeats_for(architecture.name)):
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
                                    error=exc,
                                    budget_exceeded=isinstance(exc, BudgetExceeded),
                                )
                            runs.append(run)
        artifact = str(self.reproduction_path) if self.reproduction_path is not None else None
        report = ExperimentReport(
            protocol=self.protocol,
            runs=tuple(runs),
            reproduction_artifact=artifact,
        )
        if self.reproduction_path is not None:
            from langres.experiments.reproduction import (
                ReproductionArchitecture,
                write_reproduction_bundle,
            )

            architecture_snapshots = []
            artifact_root = self.reproduction_path.with_suffix(".models")
            for index, architecture in enumerate(self.architectures):
                model = architecture.build(self.protocol.threshold_grid[0], self._monitor)
                artifact_directory = artifact_root / _reproduction_artifact_name(
                    index, architecture.name
                )
                model.save(artifact_directory)
                architecture_snapshots.append(
                    ReproductionArchitecture(
                        name=architecture.name,
                        variant_id=architecture.variant_id,
                        cache_semantics=architecture.cache_semantics,
                        estimated_usd=architecture.estimated_usd,
                        artifact_path=artifact_directory.relative_to(
                            self.reproduction_path.parent
                        ).as_posix(),
                        execution_plan=model.execution_plan().model_dump(mode="json"),
                    )
                )
            write_reproduction_bundle(
                self.reproduction_path,
                source=source,
                architectures=architecture_snapshots,
                report=report,
            )
        return report

    def _validate_budget_preflight(self) -> None:
        """Reject an unsafe spend declaration before loading data or running cells."""
        estimated_usd = 0.0
        missing: list[str] = []
        for architecture in self.architectures:
            attempts = (
                len(self.protocol.benchmark_ids)
                * len(self.protocol.split_ids)
                * len(self.protocol.split_seeds)
                * self.protocol.repeats_for(architecture.name)
            )
            if architecture.estimated_usd is None:
                missing.append(architecture.name)
            else:
                estimated_usd += architecture.estimated_usd * attempts
        if self.protocol.paid_proof and missing:
            raise ExperimentConfigurationError(
                "Cannot start official paid proof without a complete USD preflight "
                f"estimate; missing estimates for {sorted(missing)}. "
                "Fix: declare ArchitectureFactory.estimated_usd for every topology."
            )
        if self.budget_usd is not None and not missing and estimated_usd > self.budget_usd:
            raise ExperimentConfigurationError(
                f"Cannot start experiment: estimated maximum USD {estimated_usd:.2f} exceeds "
                f"the hard cap USD {self.budget_usd:.2f}. "
                "Fix: reduce the matrix/model estimates or raise the explicit budget."
            )

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
        model_config: Mapping[str, Any] | None = None,
    ) -> RunContext:
        return RunContext(
            experiment=f"{architecture.name}:{benchmark_id}:{split_id}",
            tags={
                "architecture": architecture.name,
                "variant_id": architecture.variant_id,
                "split_id": split_id,
                "repeat_index": str(repeat_index),
                "threshold_split_id": "train",
                "evaluation_split_id": ("test" if split_id == "official" else split_id),
            },
            resolver_config={
                "architecture": architecture.name,
                "variant_id": architecture.variant_id,
                "execution_plan": dict(plan),
                "model_config": dict(model_config) if model_config is not None else None,
            },
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
        try:
            initial = architecture.build(self.protocol.threshold_grid[0], self._monitor)
        except Exception as exc:
            self._capture_initialization_failure(
                architecture,
                benchmark_id=benchmark_id,
                dataset_fingerprint_value=dataset_fingerprint_value,
                seed=seed,
                repeat_index=repeat_index,
                split_id=split_id,
                evaluation_id=evaluation_id,
                error=exc,
            )
            raise
        plan = initial.execution_plan()
        try:
            model_config = initial.config_dict()
        except (TypeError, ValueError):
            model_config = None
        context = self._context(
            architecture,
            benchmark_id=benchmark_id,
            fingerprint=dataset_fingerprint_value,
            split_id=split_id,
            seed=seed,
            repeat_index=repeat_index,
            plan=plan.model_dump(mode="json"),
            model_config=model_config,
        )
        recipe_id = compute_recipe_identity(context).recipe_id
        existing = self._completed(
            recipe_id,
            evaluation_id=evaluation_id,
            variant_id=architecture.variant_id,
        )
        if self.resume and model_config is not None and existing is not None:
            return self._from_record(existing, architecture, evaluation_id)
        parent = self._latest(recipe_id)
        if parent is not None:
            context = context.model_copy(update={"parent_run_id": parent.attempt_id})
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
            suppress_error_details=False,
        ) as handle:
            tuning: _TuningOutcome | None = None
            selected_threshold: float | None = None
            model = initial
            try:
                tuning = self._tune_threshold(
                    architecture,
                    train_records,
                    train_clusters,
                    source=source,
                    seed=seed,
                    repeat_index=repeat_index,
                    attempt_id=attempt_id,
                )
                selected_threshold = tuning.threshold
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
                resource_runtime = getattr(model, "resource_runtime", {})
                if not isinstance(resource_runtime, Mapping):
                    resource_runtime = {}
                measurements = tuple(
                    measurement
                    for execution, execution_cache_hit, phase in (
                        *(
                            (execution, execution_cache_hit, "tuning")
                            for execution, execution_cache_hit in tuning.executions
                        ),
                        (result, cache_hit, "evaluation"),
                    )
                    for measurement in _measurements(
                        execution,
                        cache_hit=execution_cache_hit,
                        hardware_cohort=self.protocol.hardware_cohort,
                        resource_runtime=resource_runtime,
                        price_snapshots=self.price_snapshots,
                        phase=cast(MeasurementPhase, phase),
                    )
                )
                funnel = _funnel(result, record_count=len(records))
                executions = (
                    *(execution for execution, _cache_hit in tuning.executions),
                    result,
                )
                token_usage = _merge_token_usage(
                    tuple(_token_usage(execution) for execution in executions)
                )
                cost_complete = not self._monitor.cost_is_unknown and all(
                    _cost_complete(execution) for execution in executions
                )
                observed_usd = self._monitor.spent - starting_spend if cost_complete else None
                numeric = flatten_numeric(
                    {
                        "quality": quality,
                        "funnel": funnel.model_dump(mode="json"),
                        "token_usage": (
                            token_usage.model_dump(mode="json") if token_usage is not None else {}
                        ),
                        "usd": observed_usd,
                    }
                )
                numeric["selected_threshold"] = selected_threshold
                handle.log_metrics(numeric, headline_metric=quality.get("pair_f1"))
                handle.record_measurements(
                    measurement.model_dump(mode="json") for measurement in measurements
                )
                safe_tracker.finish(status="completed")
                handle.record_experiment_facts(
                    _experiment_facts(
                        funnel=funnel,
                        token_usage=token_usage,
                        selected_threshold=selected_threshold,
                        usd=observed_usd,
                        warnings=safe_tracker.errors,
                    )
                )
                handle.record_cost(observed_usd)
            except BudgetExceeded as exc:
                resource_runtime = getattr(model, "resource_runtime", {})
                if not isinstance(resource_runtime, Mapping):
                    resource_runtime = {}
                partial_measurements, partial_usage, _partial_usd = _partial_generation_facts(
                    exc,
                    hardware_cohort=self.protocol.hardware_cohort,
                    resource_runtime=resource_runtime,
                )
                completed_measurements = _tuning_measurements(
                    tuning,
                    hardware_cohort=self.protocol.hardware_cohort,
                    resource_runtime=resource_runtime,
                    price_snapshots=self.price_snapshots,
                )
                token_usage = _merge_token_usage((*_tuning_usage(tuning), partial_usage))
                observed_usd = (
                    self._monitor.spent - starting_spend
                    if not self._monitor.cost_is_unknown
                    else None
                )
                handle.record_measurements(
                    measurement.model_dump(mode="json")
                    for measurement in (*completed_measurements, *partial_measurements)
                )
                safe_tracker.finish(status="budget_exceeded")
                handle.record_experiment_facts(
                    _experiment_facts(
                        funnel=None,
                        token_usage=token_usage,
                        selected_threshold=selected_threshold,
                        usd=observed_usd,
                        warnings=safe_tracker.errors,
                    )
                )
                handle.record_partial_judgements(
                    judgement.model_dump(mode="json") for judgement in exc.partial_judgements
                )
                handle.set_status("budget_exceeded")
                handle.record_cost(
                    observed_usd,
                    budget_exceeded=True,
                )
                raise
            except UnknownGenerationCostError as exc:
                resource_runtime = getattr(model, "resource_runtime", {})
                if not isinstance(resource_runtime, Mapping):
                    resource_runtime = {}
                partial_measurements, partial_usage, _partial_usd = _partial_generation_facts(
                    exc,
                    hardware_cohort=self.protocol.hardware_cohort,
                    resource_runtime=resource_runtime,
                )
                completed_measurements = _tuning_measurements(
                    tuning,
                    hardware_cohort=self.protocol.hardware_cohort,
                    resource_runtime=resource_runtime,
                    price_snapshots=self.price_snapshots,
                )
                token_usage = _merge_token_usage((*_tuning_usage(tuning), partial_usage))
                handle.record_measurements(
                    measurement.model_dump(mode="json")
                    for measurement in (*completed_measurements, *partial_measurements)
                )
                safe_tracker.finish(status="failed")
                handle.record_experiment_facts(
                    _experiment_facts(
                        funnel=None,
                        token_usage=token_usage,
                        selected_threshold=selected_threshold,
                        usd=None,
                        warnings=safe_tracker.errors,
                    )
                )
                handle.record_cost(None)
                raise self._cell_error(
                    architecture,
                    benchmark_id=benchmark_id,
                    split_id=split_id,
                    repeat_index=repeat_index,
                    phase="execute",
                    error=exc,
                ) from exc
            except ExperimentCellError:
                safe_tracker.finish(status="failed")
                handle.record_cost(
                    (
                        self._monitor.spent - starting_spend
                        if not self._monitor.cost_is_unknown
                        else None
                    )
                )
                raise
            except Exception as exc:
                safe_tracker.finish(status="failed")
                handle.record_cost(
                    (
                        self._monitor.spent - starting_spend
                        if not self._monitor.cost_is_unknown
                        else None
                    )
                )
                raise self._cell_error(
                    architecture,
                    benchmark_id=benchmark_id,
                    split_id=split_id,
                    repeat_index=repeat_index,
                    phase="execute",
                    error=exc,
                ) from exc
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
            funnel=funnel,
            selected_threshold=selected_threshold,
            wall_seconds=wall,
            token_usage=token_usage,
            usd=observed_usd,
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
    ) -> _TuningOutcome:
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
        first_result, first_cache_hit = self._execute_cached(
            first, records, cache_id=cache_id, input_fingerprint=input_fp
        )
        executions: list[tuple[ExecutionResult, bool]] = [(first_result, first_cache_hit)]
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
                executions.append((result, True))
            else:
                result = model.execute([record.model_dump() for record in records])
                executions.append((result, False))
            evaluated = evaluate_execution_result(
                result, records, truth_clusters, threshold=threshold
            )
            if evaluated.bcubed_f1 > best_f1:
                best_f1 = evaluated.bcubed_f1
                best_threshold = threshold
        return _TuningOutcome(
            threshold=best_threshold,
            executions=tuple(executions),
        )

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
                    resource_slots=_resource_slots(plan, boundary_index=len(plan.steps)),
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
                # The checkpoint is the output of the prefix immediately before
                # the tunable Select. Its identity must not inherit that
                # downstream stage's threshold-bearing stage_id.
                stage_id=f"replay-prefix-boundary:{boundary_index}",
                execution_plan_id=plan.replay_prefix_id,
                operation_identity={
                    "steps": [step.model_dump(mode="json") for step in plan.steps[:boundary_index]]
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

    def _completed(
        self,
        recipe_id: str,
        *,
        evaluation_id: str,
        variant_id: str,
    ) -> RunRecord | None:
        if self.store is None:
            return None
        for record in reversed(self.store.read()):
            if (
                record.recipe_id != recipe_id
                or record.status != "completed"
                or record.evaluation_id != evaluation_id
                or record.context.tags.get("variant_id") != variant_id
            ):
                continue
            try:
                stored_protocol = EvaluationProtocol.model_validate(record.protocol)
            except (TypeError, ValueError):
                continue
            if stored_protocol == self.protocol:
                return record
        return None

    def _capture_initialization_failure(
        self,
        architecture: ArchitectureFactory,
        *,
        benchmark_id: str,
        dataset_fingerprint_value: str,
        seed: int,
        repeat_index: int,
        split_id: str,
        evaluation_id: str,
        error: Exception,
    ) -> None:
        """Persist a factory failure before a runnable plan exists."""
        contextual = self._cell_error(
            architecture,
            benchmark_id=benchmark_id,
            split_id=split_id,
            repeat_index=repeat_index,
            phase="initialize",
            error=error,
        )
        context = self._context(
            architecture,
            benchmark_id=benchmark_id,
            fingerprint=dataset_fingerprint_value,
            split_id=split_id,
            seed=seed,
            repeat_index=repeat_index,
            plan={"status": "architecture-initialization-failed"},
        )
        recipe_id = compute_recipe_identity(context).recipe_id
        parent = self._latest(recipe_id)
        if parent is not None:
            context = context.model_copy(update={"parent_run_id": parent.attempt_id})
        attempt_id = mint_attempt_id(recipe_id)
        with capture_run(
            context,
            store=self.store,
            tracker=_PublicationSafeTracker(self.tracker),
            recipe_id=recipe_id,
            evaluation_id=evaluation_id,
            protocol=self.protocol.model_dump(mode="json"),
            attempt_id=attempt_id,
            suppress_error_details=False,
        ):
            raise contextual from error

    @staticmethod
    def _cell_error(
        architecture: ArchitectureFactory,
        *,
        benchmark_id: str,
        split_id: str,
        repeat_index: int,
        phase: Literal["initialize", "execute"],
        error: Exception,
    ) -> ExperimentCellError:
        action = "initialize" if phase == "initialize" else "execute"
        if isinstance(error, ImportError):
            fix = (
                "Install the optional extra required by the declared resource "
                "and retry, or pass a resource already available locally."
            )
        else:
            fix = (
                "Inspect the chained backend exception in debug mode, correct "
                "the declared resource/configuration, and retry this cell."
            )
        return ExperimentCellError(
            f"Cannot {action} architecture {architecture.name!r}. "
            f"Cause: {type(error).__name__} during architecture {action}. "
            f"Fix: {fix} "
            "Resource slot: architecture factory or declared topology stage. "
            f"Cell: {architecture.name} / {benchmark_id} / {split_id} / "
            f"repeat {repeat_index}"
        )

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
                if record.recipe_id == recipe_id and (status is None or record.status == status)
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
        facts = record.experiment_facts or {}
        funnel_payload = facts.get("funnel")
        token_payload = facts.get("token_usage")
        stored_warnings = facts.get("warnings")
        warnings = (
            tuple(str(item) for item in stored_warnings)
            if isinstance(stored_warnings, (list, tuple))
            else ()
        )
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
            measurements=tuple(
                StageMeasurement.model_validate(measurement)
                for measurement in (record.measurements or ())
            ),
            funnel=(
                FunnelFacts.model_validate(funnel_payload)
                if isinstance(funnel_payload, Mapping)
                else None
            ),
            selected_threshold=(
                float(facts["selected_threshold"])
                if isinstance(facts.get("selected_threshold"), (int, float))
                else None
            ),
            partial_judgements=tuple(record.partial_judgements or ()),
            wall_seconds=record.duration_seconds,
            token_usage=(
                TokenUsage.model_validate(token_payload)
                if isinstance(token_payload, Mapping)
                else None
            ),
            usd=(
                float(facts["usd"])
                if isinstance(facts.get("usd"), (int, float))
                else record.spend_usd
            ),
            warnings=(*warnings, "resumed from completed RunStore attempt"),
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
        error: Exception,
        budget_exceeded: bool = False,
    ) -> ExperimentRun:
        persisted = self._latest_cell_attempt(
            architecture,
            benchmark_id=benchmark_id,
            split_id=split_id,
            seed=seed,
            repeat_index=repeat_index,
            evaluation_id=evaluation_id,
        )
        if persisted is not None:
            facts = persisted.experiment_facts or {}
            funnel_payload = facts.get("funnel")
            token_payload = facts.get("token_usage")
            stored_warnings = facts.get("warnings")
            return ExperimentRun(
                recipe_id=persisted.recipe_id,
                evaluation_id=evaluation_id,
                attempt_id=persisted.attempt_id,
                cache_id=persisted.cache_id,
                architecture=architecture.name,
                variant_id=architecture.variant_id,
                benchmark_id=benchmark_id,
                split_id=split_id,
                threshold_split_id="train",
                evaluation_split_id="test" if split_id == "official" else split_id,
                split_seed=seed,
                repeat_index=repeat_index,
                status=(
                    "budget_exceeded"
                    if persisted.status == "budget_exceeded" or budget_exceeded
                    else "failed"
                ),
                cohort_id=self.protocol.hardware_cohort,
                measurements=tuple(
                    StageMeasurement.model_validate(measurement)
                    for measurement in (persisted.measurements or ())
                ),
                funnel=(
                    FunnelFacts.model_validate(funnel_payload)
                    if isinstance(funnel_payload, Mapping)
                    else None
                ),
                selected_threshold=(
                    float(facts["selected_threshold"])
                    if isinstance(facts.get("selected_threshold"), (int, float))
                    else None
                ),
                partial_judgements=tuple(persisted.partial_judgements or ()),
                wall_seconds=persisted.duration_seconds,
                token_usage=(
                    TokenUsage.model_validate(token_payload)
                    if isinstance(token_payload, Mapping)
                    else None
                ),
                usd=(
                    float(facts["usd"])
                    if isinstance(facts.get("usd"), (int, float))
                    else persisted.spend_usd
                ),
                warnings=(
                    tuple(str(item) for item in stored_warnings)
                    if isinstance(stored_warnings, (list, tuple))
                    else ()
                ),
                error_type=persisted.error_type,
                error_message=persisted.error_message,
            )
        return ExperimentRun(
            recipe_id=(
                f"failed:{architecture.name}:{architecture.variant_id}:"
                f"{benchmark_id}:{split_id}:{seed}:{repeat_index}"
            ),
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
            error_type=("BudgetExceeded" if budget_exceeded else type(error).__name__),
            error_message=(
                "Budget exceeded; partial measurements are retained."
                if budget_exceeded
                else str(error)
            ),
        )

    def _latest_cell_attempt(
        self,
        architecture: ArchitectureFactory,
        *,
        benchmark_id: str,
        split_id: str,
        seed: int,
        repeat_index: int,
        evaluation_id: str,
    ) -> RunRecord | None:
        """Return the durable attempt for one failed matrix cell, if present."""
        if self.store is None:
            return None
        return next(
            (
                record
                for record in reversed(self.store.read())
                if record.evaluation_id == evaluation_id
                and record.status in {"failed", "budget_exceeded"}
                and record.context.dataset_name == benchmark_id
                and record.context.split_id == split_id
                and record.context.tags.get("architecture") == architecture.name
                and record.context.tags.get("variant_id") == architecture.variant_id
                and int(record.context.seeds.get("split", -1)) == seed
                and int(record.context.seeds.get("repeat", -1)) == repeat_index
            ),
            None,
        )


__all__ = [
    "ArchitectureFactory",
    "Experiment",
    "ExperimentCellError",
    "ExperimentConfigurationError",
    "flatten_numeric",
]
