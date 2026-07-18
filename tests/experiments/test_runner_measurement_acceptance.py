from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from langres.architectures import RetrieveLLM
from langres.core.model_ref import ModelRef
from langres.core.registry import register
from langres.core.spend import SpendMonitor
from langres.experiments import (
    ArchitectureFactory,
    EvaluationProtocol,
    Experiment,
    ExperimentCellError,
    PriceSnapshot,
)
from langres.experiments.identity import SourceState
from langres.resources import (
    FakeEmbedder,
    EmbeddingBatch,
    GenerationBatch,
    GenerationEnvelope,
    GenerationRequest,
    GenerationUsage,
    LLMRuntimeConfig,
    ResourceRuntimeConfig,
)
from langres.tracking.runs import RunStore


class _MeasurementRecord(BaseModel):
    id: str
    name: str


class _MeasurementBenchmark:
    name = "measurement"
    threshold_grid = (0.5,)

    def load(
        self,
    ) -> tuple[list[_MeasurementRecord], list[set[str]], set[frozenset[str]]]:
        records = [
            _MeasurementRecord(id="a1", name="A"),
            _MeasurementRecord(id="a2", name="A"),
            _MeasurementRecord(id="b1", name="B"),
            _MeasurementRecord(id="b2", name="B"),
        ]
        clusters = [{"a1", "a2"}, {"b1", "b2"}]
        return records, clusters, {frozenset(cluster) for cluster in clusters}

    def split(
        self,
        corpus: list[_MeasurementRecord],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[
        list[_MeasurementRecord],
        list[_MeasurementRecord],
        list[set[str]],
        list[set[str]],
    ]:
        del seed
        return corpus[:2], corpus[2:], gold_clusters[:1], gold_clusters[1:]


@register("test_measured_llm_acceptance")
class _MeasuredLLM:
    type_name = "test_measured_llm_acceptance"
    requires_cost_accounting = False

    def __init__(self) -> None:
        self.model_ref = ModelRef(base="./measured/llm", kind="local")
        self.runtime_config = LLMRuntimeConfig(
            batch_size=1,
            device="cpu",
            dtype="float32",
        )

    @property
    def config(self) -> dict[str, object]:
        return {}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> _MeasuredLLM:
        assert config == {}
        return cls()

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        return GenerationBatch(
            outputs=tuple(
                GenerationEnvelope.from_content(
                    request_id=request.request_id,
                    model_ref=self.model_ref,
                    content="MATCH",
                    usage=GenerationUsage(
                        input_tokens=0,
                        output_tokens=0,
                        cache_read_input_tokens=None,
                        cache_creation_input_tokens=None,
                        reasoning_tokens=0,
                        provider="fake-provider",
                        model="served-model",
                    ),
                    provider="fake-provider",
                    served_model="served-model",
                    cost_usd=0.0,
                    cost_basis="real",
                )
                for request in requests
            ),
            model_ref=self.model_ref,
        )


@register("test_measured_embedder_acceptance")
class _MeasuredEmbedder:
    type_name = "test_measured_embedder_acceptance"

    def __init__(self) -> None:
        self.model_ref = ModelRef(base="./measured/embedder", kind="local")
        self.runtime_config = ResourceRuntimeConfig(batch_size=1024, device="cpu")
        self._delegate = FakeEmbedder(dimension=8)

    @property
    def config(self) -> dict[str, object]:
        return {}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> _MeasuredEmbedder:
        assert config == {}
        return cls()

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        batch = self._delegate.embed(texts)
        return batch.model_copy(update={"model_ref": self.model_ref})


class _CostedLLM:
    def __init__(
        self,
        *,
        cost_usd: float | None,
        requires_cost_accounting: bool,
    ) -> None:
        self.model_ref = ModelRef(
            base="test/accounting-model",
            kind="api" if requires_cost_accounting else "local",
        )
        self.runtime_config = LLMRuntimeConfig(batch_size=1, device="cpu")
        self.requires_cost_accounting = requires_cost_accounting
        self.cost_usd = cost_usd

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        return GenerationBatch(
            outputs=tuple(
                GenerationEnvelope.from_content(
                    request_id=request.request_id,
                    model_ref=self.model_ref,
                    content="MATCH",
                    usage=GenerationUsage(
                        input_tokens=2,
                        output_tokens=1,
                        cache_read_input_tokens=0,
                        cache_creation_input_tokens=0,
                        reasoning_tokens=0,
                        provider="test",
                        model="accounting-model",
                    ),
                    provider="test",
                    served_model="accounting-model",
                    cost_usd=self.cost_usd,
                    cost_basis="real" if self.cost_usd is not None else "none",
                )
                for request in requests
            ),
            model_ref=self.model_ref,
        )


class _TrackerSpy:
    name = "trackio"

    def __init__(self) -> None:
        self.metrics: list[dict[str, float]] = []

    def start_run(self, context: Any, *, run_name: str | None = None) -> None:
        del context, run_name

    def log_params(self, params: Any) -> None:
        del params

    def log_metrics(self, metrics: Any, *, step: int | None = None) -> None:
        del step
        self.metrics.append(dict(metrics))

    def log_artifact(self, key: str, value: str) -> None:
        del key, value

    def set_tags(self, tags: Any) -> None:
        del tags

    def finish(self, *, status: str) -> None:
        del status

    @property
    def run_url(self) -> None:
        return None

    @property
    def native(self) -> None:
        return None


def _protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        benchmark_ids=("measurement",),
        split_ids=("test",),
        fixed_test_set_id="measurement:test:v1",
        split_seeds=(0,),
        threshold_split_id="train",
        test_split_id="test",
        threshold_grid=(0.5,),
        confidence_interval_method="none",
        bootstrap_samples=1,
        hardware_cohort="measurement-cpu",
        benchmark_version="1",
    )


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "langres.experiments.runner.get_benchmark",
        lambda _name: _MeasurementBenchmark(),
    )
    monkeypatch.setattr(
        "langres.experiments.runner.detect_source_state",
        lambda **_kwargs: SourceState(
            git_sha="a" * 40,
            lockfile_hash="lock",
            environment_hash="environment",
        ),
    )


def test_runner_populates_persists_and_publishes_tier_zero_measurement_facts(
    tmp_path: Path,
) -> None:
    llm = _MeasuredLLM()
    tracker = _TrackerSpy()
    price = PriceSnapshot(
        provider="fake-provider",
        model="served-model",
        captured_at=datetime(2026, 7, 18, tzinfo=UTC),
        input_usd_per_token=0.0,
        output_usd_per_token=0.0,
        source="user",
        source_reference="acceptance-fixture",
    )

    def build(threshold: float, monitor: SpendMonitor) -> RetrieveLLM:
        return RetrieveLLM(
            embedder=FakeEmbedder(dimension=8),
            llm=llm,
            schema=_MeasurementRecord,
            retrieve_k=1,
            llm_k=1,
            threshold=threshold,
            monitor=monitor,
        )

    store = RunStore(tmp_path / "runs.jsonl")
    [run] = (
        Experiment(
            architectures=[
                ArchitectureFactory(
                    name="RetrieveLLM",
                    factory=build,
                    estimated_usd=0.0,
                )
            ],
            protocol=_protocol(),
            tracker=tracker,
            store=store,
            cache_dir=tmp_path / "cache",
            price_snapshots={"./measured/llm": price},
        )
        .run()
        .runs
    )

    assert run.funnel is not None
    assert run.funnel.possible_pairs == 1
    assert run.funnel.retrieved_pairs == 1
    assert run.funnel.pairs_after_select == (1, 1)
    assert run.funnel.reranker_pairs is None
    assert run.funnel.llm_pairs == 1
    assert run.funnel.parsed_abstentions == 0
    assert run.funnel.selected_match_edges == 1
    assert run.funnel.clusters_produced == 1

    assert run.token_usage is not None
    assert run.token_usage.input_tokens == 0
    assert run.token_usage.output_tokens == 0
    assert run.token_usage.cache_read_input_tokens is None
    assert run.token_usage.cache_creation_input_tokens is None
    assert run.token_usage.reasoning_output_tokens == 0
    assert run.usd == 0.0

    retrieve = next(item for item in run.measurements if item.operation_kind == "retrieve")
    assert retrieve.resource_slot == "embedder"
    assert retrieve.resource_id is not None and "fake/embedder" in retrieve.resource_id
    assert retrieve.embedding is not None
    assert retrieve.embedding.dimensions == 8
    assert retrieve.embedding.vectors_produced == 2
    assert retrieve.embedding.bytes_per_vector == 32
    assert retrieve.embedding.total_vector_bytes == 64
    assert retrieve.runtime is not None
    assert retrieve.runtime.hardware_cohort == "measurement-cpu"
    assert retrieve.runtime.python_version is not None
    assert retrieve.runtime.device == "cpu"
    assert retrieve.runtime.batch_size == 1024

    generate = next(item for item in run.measurements if item.operation_kind == "generate")
    assert generate.resource_slot == "llm"
    assert generate.resource_id is not None and "measured/llm" in generate.resource_id
    assert generate.external_calls == 1
    assert generate.usage is not None
    assert generate.usage.input_tokens == 0
    assert generate.observed_usd == 0.0
    assert generate.derived_usd == 0.0
    assert generate.price == price
    assert generate.runtime is not None
    assert generate.runtime.dtype == "float32"
    assert generate.runtime.batch_size == 1

    [record] = store.read()
    assert record.measurements is not None
    assert any(item["operation_kind"] == "generate" for item in record.measurements)
    flattened = {key for metrics in tracker.metrics for key in metrics}
    assert "funnel.llm_pairs" in flattened
    assert "token_usage.input_tokens" in flattened
    assert "usd" in flattened


def test_tuning_and_evaluation_usage_cost_and_stage_facts_are_aggregated(
    tmp_path: Path,
) -> None:
    llm = _CostedLLM(cost_usd=0.25, requires_cost_accounting=True)

    def build(threshold: float, monitor: SpendMonitor) -> RetrieveLLM:
        return RetrieveLLM(
            embedder=FakeEmbedder(dimension=8),
            llm=llm,
            schema=_MeasurementRecord,
            retrieve_k=1,
            llm_k=1,
            threshold=threshold,
            monitor=monitor,
        )

    [run] = (
        Experiment(
            architectures=[
                ArchitectureFactory(
                    name="RetrieveLLM",
                    factory=build,
                    estimated_usd=0.5,
                )
            ],
            protocol=_protocol(),
            store=tmp_path / "runs.jsonl",
            cache_dir=tmp_path / "cache",
            budget_usd=1.0,
        )
        .run()
        .runs
    )

    assert run.usd == pytest.approx(0.5)
    assert run.token_usage is not None
    assert run.token_usage.input_tokens == 4
    assert run.token_usage.output_tokens == 2
    generation = [
        measurement for measurement in run.measurements if measurement.operation_kind == "generate"
    ]
    assert [measurement.phase for measurement in generation] == [
        "tuning",
        "evaluation",
    ]
    assert sum(measurement.external_calls or 0 for measurement in generation) == 2
    assert sum(measurement.observed_usd or 0.0 for measurement in generation) == pytest.approx(0.5)


def test_unknown_paid_cost_remains_none_for_an_uncapped_run(tmp_path: Path) -> None:
    llm = _CostedLLM(cost_usd=None, requires_cost_accounting=True)

    def build(threshold: float, monitor: SpendMonitor) -> RetrieveLLM:
        return RetrieveLLM(
            embedder=FakeEmbedder(dimension=8),
            llm=llm,
            schema=_MeasurementRecord,
            retrieve_k=1,
            llm_k=1,
            threshold=threshold,
            monitor=monitor,
        )

    [run] = (
        Experiment(
            architectures=[ArchitectureFactory(name="RetrieveLLM", factory=build)],
            protocol=_protocol(),
            store=tmp_path / "runs.jsonl",
            cache_dir=tmp_path / "cache",
        )
        .run()
        .runs
    )

    assert run.status == "completed"
    assert run.usd is None
    assert any(
        measurement.operation_kind == "generate" and measurement.observed_usd is None
        for measurement in run.measurements
    )


def test_resume_rehydrates_all_durable_typed_report_facts(tmp_path: Path) -> None:
    llm = _MeasuredLLM()

    def build(threshold: float, monitor: SpendMonitor) -> RetrieveLLM:
        return RetrieveLLM(
            embedder=_MeasuredEmbedder(),
            llm=llm,
            schema=_MeasurementRecord,
            retrieve_k=1,
            llm_k=1,
            threshold=threshold,
            monitor=monitor,
        )

    kwargs = {
        "architectures": [
            ArchitectureFactory(
                name="RetrieveLLM",
                factory=build,
                estimated_usd=0.0,
            )
        ],
        "protocol": _protocol(),
        "tracker": _TrackerSpy(),
        "store": tmp_path / "runs.jsonl",
        "cache_dir": tmp_path / "cache",
    }
    first = Experiment(**kwargs).run().runs[0]
    resumed = Experiment(**kwargs).run().runs[0]

    assert resumed.funnel == first.funnel
    assert resumed.token_usage == first.token_usage
    assert resumed.selected_threshold == first.selected_threshold
    assert resumed.measurements == first.measurements
    assert resumed.usd == first.usd
    assert resumed.warnings == (
        *first.warnings,
        "resumed from completed RunStore attempt",
    )


def test_budget_exceeded_persists_recoverable_generation_facts(tmp_path: Path) -> None:
    llm = _CostedLLM(cost_usd=0.6, requires_cost_accounting=True)

    def build(threshold: float, monitor: SpendMonitor) -> RetrieveLLM:
        return RetrieveLLM(
            embedder=FakeEmbedder(dimension=8),
            llm=llm,
            schema=_MeasurementRecord,
            retrieve_k=1,
            llm_k=1,
            threshold=threshold,
            monitor=monitor,
        )

    store = RunStore(tmp_path / "runs.jsonl")
    [run] = (
        Experiment(
            architectures=[
                ArchitectureFactory(
                    name="RetrieveLLM",
                    factory=build,
                    estimated_usd=0.5,
                )
            ],
            protocol=_protocol(),
            store=store,
            cache_dir=tmp_path / "cache",
            budget_usd=0.5,
        )
        .run()
        .runs
    )

    assert run.status == "budget_exceeded"
    assert run.usd == pytest.approx(0.6)
    assert run.token_usage is not None
    assert run.token_usage.input_tokens == 2
    [partial] = [measurement for measurement in run.measurements if measurement.phase == "partial"]
    assert partial.resource_slot == "llm"
    assert partial.external_calls == 1
    assert partial.observed_usd == pytest.approx(0.6)
    [record] = store.read()
    assert record.measurements is not None
    assert record.experiment_facts is not None
    assert record.experiment_facts["token_usage"]["input_tokens"] == 2


def test_budget_exceeded_after_tuning_preserves_total_cell_accounting(
    tmp_path: Path,
) -> None:
    llm = _CostedLLM(cost_usd=0.3, requires_cost_accounting=True)

    def build(threshold: float, monitor: SpendMonitor) -> RetrieveLLM:
        return RetrieveLLM(
            embedder=FakeEmbedder(dimension=8),
            llm=llm,
            schema=_MeasurementRecord,
            retrieve_k=1,
            llm_k=1,
            threshold=threshold,
            monitor=monitor,
        )

    store = RunStore(tmp_path / "runs.jsonl")
    [run] = (
        Experiment(
            architectures=[
                ArchitectureFactory(
                    name="RetrieveLLM",
                    factory=build,
                    estimated_usd=0.5,
                )
            ],
            protocol=_protocol(),
            store=store,
            cache_dir=tmp_path / "cache",
            budget_usd=0.5,
        )
        .run()
        .runs
    )

    assert run.status == "budget_exceeded"
    assert run.selected_threshold == pytest.approx(0.5)
    assert run.usd == pytest.approx(0.6)
    assert run.token_usage is not None
    assert run.token_usage.input_tokens == 4
    generation = [
        measurement for measurement in run.measurements if measurement.operation_kind == "generate"
    ]
    assert [measurement.phase for measurement in generation] == ["tuning", "partial"]
    assert sum(measurement.observed_usd or 0.0 for measurement in generation) == pytest.approx(0.6)
    [record] = store.read()
    assert record.spend_usd == pytest.approx(0.6)
    assert record.experiment_facts is not None
    assert record.experiment_facts["usd"] == pytest.approx(0.6)
    assert record.experiment_facts["selected_threshold"] == pytest.approx(0.5)


def test_unknown_finite_budget_cost_persists_partial_generation_as_unknown(
    tmp_path: Path,
) -> None:
    llm = _CostedLLM(cost_usd=None, requires_cost_accounting=True)

    def build(threshold: float, monitor: SpendMonitor) -> RetrieveLLM:
        return RetrieveLLM(
            embedder=FakeEmbedder(dimension=8),
            llm=llm,
            schema=_MeasurementRecord,
            retrieve_k=1,
            llm_k=1,
            threshold=threshold,
            monitor=monitor,
        )

    store = RunStore(tmp_path / "runs.jsonl")
    [run] = (
        Experiment(
            architectures=[
                ArchitectureFactory(
                    name="RetrieveLLM",
                    factory=build,
                    estimated_usd=0.5,
                )
            ],
            protocol=_protocol(),
            store=store,
            cache_dir=tmp_path / "cache",
            budget_usd=1.0,
        )
        .run()
        .runs
    )

    assert run.status == "failed"
    assert run.usd is None
    assert run.token_usage is not None
    assert run.token_usage.input_tokens == 2
    [partial] = [measurement for measurement in run.measurements if measurement.phase == "partial"]
    assert partial.observed_usd is None
    assert partial.external_calls == 1
    [record] = store.read()
    assert record.spend_usd is None
    assert record.experiment_facts is not None
    assert record.experiment_facts["usd"] is None
    assert record.experiment_facts["token_usage"]["input_tokens"] == 2


def test_failed_cell_error_is_contextual_actionable_sanitized_and_chained(
    tmp_path: Path,
) -> None:
    def broken(_threshold: float, _monitor: SpendMonitor) -> Any:
        raise ImportError("token=provider-secret record payload")

    experiment = Experiment(
        architectures=[ArchitectureFactory(name="BrokenRecipe", factory=broken)],
        protocol=_protocol(),
        store=tmp_path / "runs.jsonl",
        cache_dir=tmp_path / "cache",
    )
    [run] = experiment.run().runs

    assert run.status == "failed"
    assert run.error_type == "ExperimentCellError"
    assert run.error_message is not None
    assert "Cannot initialize architecture 'BrokenRecipe'" in run.error_message
    assert "Cause: ImportError" in run.error_message
    assert "Fix:" in run.error_message
    assert "Cell: BrokenRecipe / measurement / test / repeat 0" in run.error_message
    assert "provider-secret" not in run.error_message
    assert "record payload" not in run.error_message

    [record] = RunStore(tmp_path / "runs.jsonl").read()
    assert record.error_type == "ExperimentCellError"
    assert record.error_message == run.error_message

    fail_fast = Experiment(
        architectures=[ArchitectureFactory(name="BrokenRecipe", factory=broken)],
        protocol=_protocol(),
        cache_dir=tmp_path / "cache-fast",
        fail_fast=True,
    )
    with pytest.raises(ExperimentCellError) as raised:
        fail_fast.run()
    assert isinstance(raised.value.__cause__, ImportError)
