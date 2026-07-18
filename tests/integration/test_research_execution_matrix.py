from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from langres.architectures import (
    Retrieve,
    RetrieveLLM,
    RetrieveRerank,
    RetrieveRerankLLM,
)
from langres.core.clusterer import Clusterer
from langres.core.op import ThresholdSelect
from langres.core.op_adapters import ClustererStage
from langres.core.resolver import ERModel
from langres.core.registry import register
from langres.core.spend import SpendMonitor
from langres.experiments import ArchitectureFactory, EvaluationProtocol, Experiment
from langres.experiments.identity import SourceState
from langres.resources import (
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
    GenerationBatch,
    GenerationRequest,
    Retrieve as RetrieveOp,
)


class _MatrixRecord(BaseModel):
    id: str
    name: str


class _Benchmark:
    threshold_grid = (0.2, 0.8)

    def __init__(self, name: str) -> None:
        self.name = name

    def load(
        self,
    ) -> tuple[list[_MatrixRecord], list[set[str]], set[frozenset[str]]]:
        records = [
            _MatrixRecord(id=f"{self.name}-a1", name="A"),
            _MatrixRecord(id=f"{self.name}-a2", name="A"),
            _MatrixRecord(id=f"{self.name}-b1", name="B"),
            _MatrixRecord(id=f"{self.name}-b2", name="B"),
        ]
        clusters = [
            {f"{self.name}-a1", f"{self.name}-a2"},
            {f"{self.name}-b1", f"{self.name}-b2"},
        ]
        return records, clusters, {frozenset(cluster) for cluster in clusters}

    def split(
        self,
        corpus: list[_MatrixRecord],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[
        list[_MatrixRecord],
        list[_MatrixRecord],
        list[set[str]],
        list[set[str]],
    ]:
        del seed
        return corpus[:2], corpus[2:], gold_clusters[:1], gold_clusters[1:]


@register("test_matrix_embedder")
class _PersistEmbedder(FakeEmbedder):
    type_name: ClassVar[str] = "test_matrix_embedder"

    @property
    def config(self) -> dict[str, object]:
        return {"dimension": self.dimension}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> _PersistEmbedder:
        return cls(dimension=int(config["dimension"]))  # type: ignore[call-overload]


@register("test_matrix_reranker")
class _CountingReranker(FakeReranker):
    type_name: ClassVar[str] = "test_matrix_reranker"

    @property
    def config(self) -> dict[str, object]:
        return {"scores": dict(self._scores)}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> _CountingReranker:
        scores = config["scores"]
        assert isinstance(scores, dict)
        return cls(scores={str(key): float(value) for key, value in scores.items()})


@register("test_matrix_llm")
class _CountingLLM(FakeLLM):
    type_name: ClassVar[str] = "test_matrix_llm"

    def __init__(self) -> None:
        super().__init__(default_response="MATCH")
        self.calls = 0

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        self.calls += len(requests)
        return super().generate(requests)

    @property
    def config(self) -> dict[str, object]:
        return {
            "responses": dict(self._responses),
            "default_response": self.default_response,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> _CountingLLM:
        instance = cls()
        responses = config["responses"]
        assert isinstance(responses, dict)
        instance._responses = {str(key): str(value) for key, value in responses.items()}
        instance.default_response = str(config["default_response"])
        return instance


def _factories(
    monitors: dict[str, list[SpendMonitor]],
) -> tuple[
    list[ArchitectureFactory],
    _CountingReranker,
    _CountingReranker,
    _CountingLLM,
    _CountingLLM,
]:
    retrieve_embedder = _PersistEmbedder()
    rerank_embedder = _PersistEmbedder()
    rerank = _CountingReranker()
    llm_embedder = _PersistEmbedder()
    llm = _CountingLLM()
    combined_embedder = _PersistEmbedder()
    combined_reranker = _CountingReranker()
    combined_llm = _CountingLLM()
    custom_embedder = _PersistEmbedder()

    def observe(name: str, monitor: SpendMonitor) -> None:
        monitors.setdefault(name, []).append(monitor)

    return (
        [
            ArchitectureFactory(
                name="Retrieve",
                factory=lambda threshold, monitor: (
                    observe("Retrieve", monitor)
                    or Retrieve(
                        embedder=retrieve_embedder,
                        schema=_MatrixRecord,
                        retrieve_k=1,
                        threshold=threshold,
                        monitor=monitor,
                    )
                ),
                estimated_usd=0.0,
            ),
            ArchitectureFactory(
                name="RetrieveRerank",
                factory=lambda threshold, monitor: (
                    observe("RetrieveRerank", monitor)
                    or RetrieveRerank(
                        embedder=rerank_embedder,
                        reranker=rerank,
                        schema=_MatrixRecord,
                        retrieve_k=1,
                        threshold=threshold,
                        monitor=monitor,
                    )
                ),
                estimated_usd=0.0,
            ),
            ArchitectureFactory(
                name="RetrieveLLM",
                factory=lambda threshold, monitor: (
                    observe("RetrieveLLM", monitor)
                    or RetrieveLLM(
                        embedder=llm_embedder,
                        llm=llm,
                        schema=_MatrixRecord,
                        retrieve_k=1,
                        llm_k=1,
                        threshold=threshold,
                        monitor=monitor,
                    )
                ),
                estimated_usd=0.0,
            ),
            ArchitectureFactory(
                name="RetrieveRerankLLM",
                factory=lambda threshold, monitor: (
                    observe("RetrieveRerankLLM", monitor)
                    or RetrieveRerankLLM(
                        embedder=combined_embedder,
                        reranker=combined_reranker,
                        llm=combined_llm,
                        schema=_MatrixRecord,
                        retrieve_k=1,
                        llm_k=1,
                        threshold=threshold,
                        monitor=monitor,
                    )
                ),
                estimated_usd=0.0,
            ),
            ArchitectureFactory(
                name="CustomTopology",
                factory=lambda threshold, monitor: (
                    observe("CustomTopology", monitor)
                    or ERModel.from_topology(
                        ops=[
                            RetrieveOp(custom_embedder, schema=_MatrixRecord, k=1),
                            ThresholdSelect(threshold),
                            ClustererStage(Clusterer(threshold=0.0)),
                        ],
                        replay_boundary=1,
                        monitor=monitor,
                    )
                ),
                estimated_usd=0.0,
            ),
        ],
        rerank,
        combined_reranker,
        llm,
        combined_llm,
    )


@pytest.mark.integration
def test_zero_network_five_topology_two_dataset_matrix_replays_expensive_stages_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "langres.experiments.runner.get_benchmark",
        lambda name: _Benchmark(name),
    )
    monkeypatch.setattr(
        "langres.experiments.runner.detect_source_state",
        lambda: SourceState(
            git_sha="a" * 40,
            lockfile_hash="lock",
            environment_hash="environment",
        ),
    )
    monitors: dict[str, list[SpendMonitor]] = {}
    factories, rerank, combined_reranker, llm, combined_llm = _factories(monitors)
    protocol = EvaluationProtocol(
        benchmark_ids=("dataset-a", "dataset-b"),
        split_ids=("test",),
        fixed_test_set_id="matrix:test:v1",
        split_seeds=(0,),
        threshold_split_id="train",
        test_split_id="test",
        threshold_grid=(0.2, 0.8),
        confidence_interval_method="none",
        bootstrap_samples=1,
        hardware_cohort="zero-network",
        benchmark_version="1",
    )

    report = Experiment(
        architectures=factories,
        protocol=protocol,
        store=tmp_path / "runs.jsonl",
        cache_dir=tmp_path / "cache",
        budget_usd=1.0,
        fail_fast=True,
    ).run()

    assert len(report.runs) == 10
    assert {run.status for run in report.runs} == {"completed"}
    assert {(run.architecture, run.benchmark_id) for run in report.runs} == {
        (architecture, dataset)
        for architecture in (
            "Retrieve",
            "RetrieveRerank",
            "RetrieveLLM",
            "RetrieveRerankLLM",
            "CustomTopology",
        )
        for dataset in ("dataset-a", "dataset-b")
    }

    # Each architecture evaluates one train and one test prefix per dataset.
    # The second threshold replays only Select/Cluster and cannot call a
    # reranker or LLM again.
    assert rerank.calls == 4
    assert combined_reranker.calls == 4
    assert llm.calls == 4
    assert combined_llm.calls == 4

    all_monitors = [monitor for values in monitors.values() for monitor in values]
    assert all_monitors
    assert len({id(monitor) for monitor in all_monitors}) == 1
    assert all(monitor.budget_usd == 1.0 for monitor in all_monitors)

    for factory in factories:
        plan = factory.build(0.5, all_monitors[0]).execution_plan()
        assert plan.replay_boundary is not None
