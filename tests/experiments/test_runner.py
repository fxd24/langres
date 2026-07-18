from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.op import Score, ThresholdSelect
from langres.core.op_adapters import BlockerSource, ClustererStage
from langres.core.pairs import Pairs
from langres.core.resolver import ERModel
from langres.core.spend import SpendMonitor
from langres.experiments.identity import SourceState
from langres.experiments.protocol import EvaluationProtocol
from langres.experiments.runner import (
    ArchitectureFactory,
    Experiment,
    ExperimentConfigurationError,
    flatten_numeric,
)
from langres.tracking.runs import RunStore


class _Record(BaseModel):
    id: str
    name: str


class _Benchmark:
    name = "fake"
    threshold_grid = (0.5, 0.95)

    def load(self) -> tuple[list[_Record], list[set[str]], set[frozenset[str]]]:
        records = [
            _Record(id="a1", name="A"),
            _Record(id="a2", name="A"),
            _Record(id="b1", name="B"),
            _Record(id="b2", name="B"),
        ]
        clusters = [{"a1", "a2"}, {"b1", "b2"}]
        return records, clusters, {frozenset(pair) for pair in clusters}

    def split(
        self,
        corpus: list[_Record],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[_Record], list[_Record], list[set[str]], list[set[str]]]:
        del seed
        return corpus[:2], corpus[2:], gold_clusters[:1], gold_clusters[1:]


class _CountingScore(Score[Any]):
    def __init__(self, counter: list[int]) -> None:
        super().__init__(scope="pair", out_space="heuristic")
        self.counter = counter

    def forward(self, pairs: Pairs[Any]) -> Pairs[Any]:
        self.counter[0] += 1
        return Pairs(
            store=pairs.store,
            rows=[
                row.model_copy(update={"score": 0.9, "score_type": "heuristic"})
                for row in pairs.rows
            ],
        )


def _factory(counter: list[int], name: str = "Custom") -> ArchitectureFactory:
    def build(threshold: float, monitor: SpendMonitor) -> ERModel:
        return ERModel.from_topology(
            ops=[
                BlockerSource(AllPairsBlocker(schema=_Record)),
                _CountingScore(counter),
                ThresholdSelect(threshold),
                ClustererStage(Clusterer(threshold=0.0)),
            ],
            replay_boundary=2,
            monitor=monitor,
        )

    return ArchitectureFactory(name=name, factory=build)


def _protocol(
    *,
    benchmarks: tuple[str, ...] = ("one",),
    splits: tuple[str, ...] = ("test",),
    seeds: tuple[int, ...] = (0,),
) -> EvaluationProtocol:
    return EvaluationProtocol(
        benchmark_ids=benchmarks,
        split_ids=splits,
        fixed_test_set_id="fake:test:v1",
        split_seeds=seeds,
        threshold_split_id="train",
        test_split_id="test",
        threshold_grid=(0.5, 0.95),
        confidence_interval_method="none",
        bootstrap_samples=1,
        hardware_cohort="test",
        benchmark_version="1",
    )


@pytest.fixture(autouse=True)
def _offline_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "langres.experiments.runner.get_benchmark",
        lambda _name: _Benchmark(),
    )
    monkeypatch.setattr(
        "langres.experiments.runner.detect_source_state",
        lambda: SourceState(
            git_sha="a" * 40,
            lockfile_hash="lock",
            environment_hash="environment",
        ),
    )


def test_runner_expands_benchmarks_splits_and_seeds(tmp_path: Path) -> None:
    counter = [0]
    report = Experiment(
        architectures=[_factory(counter)],
        protocol=_protocol(
            benchmarks=("one", "two"),
            splits=("train", "test"),
            seeds=(0, 1),
        ),
        store=tmp_path / "runs.jsonl",
        cache_dir=tmp_path / "cache",
    ).run()

    assert len(report.runs) == 8
    assert {run.status for run in report.runs} == {"completed"}
    assert {run.threshold_split_id for run in report.runs} == {"train"}
    assert {run.evaluation_split_id for run in report.runs} == {"train", "test"}


def test_threshold_sweep_scores_each_input_split_once_and_resume_skips_work(
    tmp_path: Path,
) -> None:
    counter = [0]
    kwargs = {
        "architectures": [_factory(counter)],
        "protocol": _protocol(),
        "store": tmp_path / "runs.jsonl",
        "cache_dir": tmp_path / "cache",
    }

    first = Experiment(**kwargs).run()
    assert counter == [2]  # train prefix once + untouched test prefix once
    assert first.runs[0].metrics["bcubed_f1"] == 1.0

    second = Experiment(**kwargs).run()
    assert counter == [2]
    assert second.runs[0].warnings == ("resumed from completed RunStore attempt",)
    assert len(RunStore(tmp_path / "runs.jsonl").read()) == 1


def test_tracker_failure_keeps_local_run_complete_and_numeric_only(tmp_path: Path) -> None:
    class BrokenTracker:
        name = "broken"

        def __init__(self) -> None:
            self.metrics: list[dict[str, float]] = []

        def start_run(self, context: Any, *, run_name: str | None = None) -> None:
            del context, run_name

        def log_params(self, params: Any) -> None:
            del params

        def log_metrics(self, metrics: Any, *, step: int | None = None) -> None:
            del step
            self.metrics.append(dict(metrics))
            raise RuntimeError("remote tracker unavailable token=secret")

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
        def native(self) -> Any:
            return self

    tracker = BrokenTracker()
    [run] = Experiment(
        architectures=[_factory([0])],
        protocol=_protocol(),
        tracker=tracker,
        store=tmp_path / "runs.jsonl",
        cache_dir=tmp_path / "cache",
    ).run().runs

    assert run.status == "completed"
    assert run.warnings == (
        "log_metrics failed with RuntimeError; publication incomplete",
    )
    assert RunStore(tmp_path / "runs.jsonl").read()[0].status == "completed"
    assert tracker.metrics
    assert all(
        isinstance(value, float)
        for metrics in tracker.metrics
        for value in metrics.values()
    )


def test_failure_continues_to_independent_architecture(tmp_path: Path) -> None:
    def broken(_threshold: float, _monitor: SpendMonitor) -> ERModel:
        raise RuntimeError("api_key=secret record payload")

    report = Experiment(
        architectures=[
            ArchitectureFactory(name="Broken", factory=broken),
            _factory([0], name="Healthy"),
        ],
        protocol=_protocol(),
        store=tmp_path / "runs.jsonl",
        cache_dir=tmp_path / "cache",
    ).run()

    assert [run.status for run in report.runs] == ["failed", "completed"]
    assert report.runs[0].error_message == "cell failed; exception details suppressed"


def test_current_registry_does_not_mislabel_train_as_validation() -> None:
    invalid = EvaluationProtocol(
        benchmark_ids=("one",),
        split_ids=("test",),
        fixed_test_set_id="fake:test:v1",
        split_seeds=(0,),
        threshold_split_id="validation",
        test_split_id="test",
        confidence_interval_method="none",
        bootstrap_samples=1,
        hardware_cohort="test",
        benchmark_version="1",
    )

    with pytest.raises(ExperimentConfigurationError, match="validation alias"):
        Experiment(architectures=[_factory([0])], protocol=invalid)


def test_flatten_numeric_omits_text_bool_none_and_non_finite() -> None:
    assert flatten_numeric(
        {
            "quality": {"f1": 0.9, "valid": True, "missing": None},
            "name": "model",
            "bad": float("nan"),
            "count": 2,
        }
    ) == {"quality.f1": 0.9, "count": 2.0}


def test_optional_budget_is_shared_across_every_factory_build(tmp_path: Path) -> None:
    seen: list[SpendMonitor] = []
    base = _factory([0])

    def build(threshold: float, monitor: SpendMonitor) -> ERModel:
        seen.append(monitor)
        return base.factory(threshold, monitor)

    Experiment(
        architectures=[ArchitectureFactory(name="SharedBudget", factory=build)],
        protocol=_protocol(),
        budget_usd=0.25,
        store=tmp_path / "runs.jsonl",
        cache_dir=tmp_path / "cache",
    ).run()

    assert seen
    assert len({id(monitor) for monitor in seen}) == 1
    assert seen[0].budget_usd == 0.25


def test_failed_cell_resume_mints_new_attempt_parent_and_stochastic_cache(
    tmp_path: Path,
) -> None:
    failing = [True]

    class ConditionalScore(_CountingScore):
        def forward(self, pairs: Pairs[Any]) -> Pairs[Any]:
            if failing[0]:
                raise RuntimeError("provider token=secret record payload")
            return super().forward(pairs)

    def build(threshold: float, monitor: SpendMonitor) -> ERModel:
        return ERModel.from_topology(
            ops=[
                BlockerSource(AllPairsBlocker(schema=_Record)),
                ConditionalScore([0]),
                ThresholdSelect(threshold),
                ClustererStage(Clusterer(threshold=0.0)),
            ],
            replay_boundary=2,
            monitor=monitor,
        )

    kwargs = {
        "architectures": [
            ArchitectureFactory(
                name="Retry",
                factory=build,
                cache_semantics="stochastic",
            )
        ],
        "protocol": _protocol(),
        "store": tmp_path / "runs.jsonl",
        "cache_dir": tmp_path / "cache",
    }
    assert Experiment(**kwargs).run().runs[0].status == "failed"
    failing[0] = False
    assert Experiment(**kwargs).run().runs[0].status == "completed"

    first, second = RunStore(tmp_path / "runs.jsonl").read()
    assert first.attempt_id != second.attempt_id
    assert first.cache_id != second.cache_id
    assert second.context.parent_run_id == first.attempt_id
    assert first.error_message is not None
    assert "secret" not in first.error_message
