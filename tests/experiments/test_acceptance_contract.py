from __future__ import annotations

import inspect
import io
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel
from pydantic import ConfigDict

from langres.architectures import Retrieve, RetrieveLLM, RetrieveRerank, RetrieveRerankLLM
from langres.cli import main
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.op import Score, ThresholdSelect
from langres.core.op_adapters import BlockerSource, ClustererStage
from langres.core.pairs import Pairs
from langres.core.resolver import ERModel
from langres.core.registry import register_op
from langres.core.spend import SpendMonitor
from langres.experiments import (
    ArchitectureFactory,
    EvaluationProtocol,
    Experiment,
    ExperimentCellError,
    ExperimentPlan,
    ExperimentReport,
    ReproductionBundle,
    load_reproduction_bundle,
)
from langres.experiments.identity import SourceState


class _AcceptanceRecord(BaseModel):
    id: str
    name: str


class _Benchmark:
    name = "acceptance"
    threshold_grid = (0.2, 0.8)

    def load(
        self,
    ) -> tuple[list[_AcceptanceRecord], list[set[str]], set[frozenset[str]]]:
        records = [
            _AcceptanceRecord(id="a1", name="A"),
            _AcceptanceRecord(id="a2", name="A"),
            _AcceptanceRecord(id="b1", name="B"),
            _AcceptanceRecord(id="b2", name="B"),
        ]
        clusters = [{"a1", "a2"}, {"b1", "b2"}]
        return records, clusters, {frozenset(cluster) for cluster in clusters}

    def split(
        self,
        corpus: list[_AcceptanceRecord],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[
        list[_AcceptanceRecord],
        list[_AcceptanceRecord],
        list[set[str]],
        list[set[str]],
    ]:
        del seed
        return corpus[:2], corpus[2:], gold_clusters[:1], gold_clusters[1:]


class _ScoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


@register_op("test_acceptance_contract_score")
class _Score(Score[Any]):
    config_model: ClassVar[type[BaseModel]] = _ScoreConfig

    def __init__(self) -> None:
        super().__init__(scope="pair", out_space="heuristic")

    @property
    def config(self) -> dict[str, object]:
        return {}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> _Score:
        assert config == {}
        return cls()

    def forward(self, pairs: Pairs[Any]) -> Pairs[Any]:
        return Pairs(
            store=pairs.store,
            rows=[
                row.model_copy(update={"score": 0.9, "score_type": "heuristic"})
                for row in pairs.rows
            ],
        )


def _factory(
    name: str,
    *,
    semantics: str = "deterministic",
    estimated_usd: float = 0.0,
) -> ArchitectureFactory:
    def build(threshold: float, monitor: SpendMonitor) -> ERModel:
        return ERModel.from_topology(
            ops=[
                BlockerSource(AllPairsBlocker(schema=_AcceptanceRecord)),
                _Score(),
                ThresholdSelect(threshold),
                ClustererStage(Clusterer(threshold=0.0)),
            ],
            replay_boundary=2,
            monitor=monitor,
        )

    return ArchitectureFactory(
        name=name,
        factory=build,
        cache_semantics=semantics,  # type: ignore[arg-type]
        estimated_usd=estimated_usd,
    )


def _protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        benchmark_ids=("one", "two"),
        split_ids=("test",),
        fixed_test_set_id="acceptance:test:v1",
        split_seeds=(0,),
        stochastic_repeats=3,
        architecture_repeats={
            "RetrieveLLM": 3,
            "RetrieveRerankLLM": 3,
        },
        threshold_split_id="train",
        test_split_id="test",
        threshold_grid=(0.2, 0.8),
        confidence_interval_method="none",
        bootstrap_samples=1,
        hardware_cohort="acceptance",
        benchmark_version="1",
        budget_usd=20.0,
    )


@pytest.fixture
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langres.experiments.runner.get_benchmark", lambda _name: _Benchmark())
    monkeypatch.setattr(
        "langres.experiments.runner.detect_source_state",
        lambda **_kwargs: SourceState(
            git_sha="a" * 40,
            lockfile_hash="lock",
            environment_hash="environment",
        ),
    )


def test_public_acceptance_imports_and_signatures_are_stable() -> None:
    assert ExperimentPlan.__module__ == "langres.experiments.report"
    assert ExperimentCellError.__module__ == "langres.experiments.runner"
    assert ReproductionBundle.__module__ == "langres.experiments.reproduction"
    assert str(inspect.signature(ArchitectureFactory)) == (
        "(name: 'str', factory: 'Callable[[float, SpendMonitor], ERModel]', "
        "variant_id: 'str' = 'default', "
        "cache_semantics: 'CacheSemantics' = 'deterministic', "
        "estimated_usd: 'float | None' = None) -> None"
    )
    assert str(inspect.signature(Experiment)) == (
        "(*, architectures: 'Sequence[ArchitectureFactory]', "
        "protocol: 'EvaluationProtocol', tracker: 'TrackerSpec' = None, "
        "store: 'str | Path | RunStore | None' = None, "
        "cache_dir: 'str | Path | None' = None, budget_usd: 'float | None' = None, "
        "price_snapshots: 'Mapping[str, PriceSnapshot] | None' = None, "
        "reproduction_path: 'str | Path | None' = None, resume: 'bool' = True, "
        "fail_fast: 'bool' = False) -> 'None'"
    )
    assert str(inspect.signature(Experiment.run)) == (
        "(self, *, dry_run: 'bool' = False) -> 'ExperimentReport | ExperimentPlan'"
    )
    assert str(inspect.signature(Experiment.plan)) == "(self) -> 'ExperimentPlan'"
    assert str(inspect.signature(Retrieve)) == (
        "(*, embedder: 'EmbedderLike', schema: 'type[BaseModel] | None' = None, "
        "retrieve_k: 'int' = 20, threshold: 'float' = 0.5, "
        "text_field: 'str | None' = None, clusterer: 'Clusterer | None' = None, "
        "budget_usd: 'float | None' = None, "
        "monitor: 'SpendMonitor | None' = None) -> 'None'"
    )
    assert str(inspect.signature(RetrieveRerank)) == (
        "(*, embedder: 'EmbedderLike', reranker: 'RerankerLike', "
        "schema: 'type[BaseModel] | None' = None, retrieve_k: 'int' = 20, "
        "threshold: 'float' = 0.5, text_field: 'str | None' = None, "
        "clusterer: 'Clusterer | None' = None, budget_usd: 'float | None' = None, "
        "monitor: 'SpendMonitor | None' = None) -> 'None'"
    )
    assert str(inspect.signature(RetrieveLLM)) == (
        "(*, embedder: 'EmbedderLike', llm: 'LLMLike', "
        "schema: 'type[BaseModel] | None' = None, retrieve_k: 'int' = 20, "
        "llm_k: 'int' = 5, threshold: 'float' = 0.5, "
        "text_field: 'str | None' = None, clusterer: 'Clusterer | None' = None, "
        "budget_usd: 'float | None' = None, "
        "monitor: 'SpendMonitor | None' = None) -> 'None'"
    )
    assert str(inspect.signature(RetrieveRerankLLM)) == (
        "(*, embedder: 'EmbedderLike', reranker: 'RerankerLike', llm: 'LLMLike', "
        "schema: 'type[BaseModel] | None' = None, retrieve_k: 'int' = 20, "
        "llm_k: 'int' = 5, threshold: 'float' = 0.5, "
        "text_field: 'str | None' = None, clusterer: 'Clusterer | None' = None, "
        "budget_usd: 'float | None' = None, "
        "monitor: 'SpendMonitor | None' = None) -> 'None'"
    )


def test_plan_and_dry_run_expose_matrix_budget_cache_and_publication(
    _offline: None,
    tmp_path: Path,
) -> None:
    architectures = [
        _factory("Retrieve", estimated_usd=0.1),
        _factory("RetrieveRerank", estimated_usd=0.2),
        _factory("RetrieveLLM", semantics="stochastic", estimated_usd=0.3),
        _factory("RetrieveRerankLLM", semantics="stochastic", estimated_usd=0.4),
        _factory("CustomTopology", estimated_usd=0.0),
    ]
    experiment = Experiment(
        architectures=architectures,
        protocol=_protocol(),
        tracker=_TrackioSpy(),
        store=tmp_path / "runs.jsonl",
        cache_dir=tmp_path / "cache",
    )

    plan = experiment.plan()

    assert isinstance(plan, ExperimentPlan)
    assert plan.topology_count == 5
    assert plan.benchmark_count == 2
    assert plan.cell_count == 10
    assert plan.deterministic_attempts == 6
    assert plan.stochastic_attempts == 12
    assert plan.total_attempts == 18
    assert plan.paid_concurrency == 1
    assert plan.estimated_usd == pytest.approx(4.8)
    assert plan.budget_usd == 20.0
    assert plan.cache_hits == 0
    assert plan.cache_misses == 18
    assert plan.publication_eligible is True
    assert experiment.run(dry_run=True) == plan
    assert not (tmp_path / "runs.jsonl").exists()


def test_non_dry_paid_proof_runs_preflight_before_loading_data(
    _offline: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol = EvaluationProtocol.official_proof(
        benchmark_ids=("one", "two"),
        dataset_fingerprints={"one": "sha256:one", "two": "sha256:two"},
        fixed_test_set_id="acceptance:test:v1",
    )
    factories = []
    for name in (
        "Retrieve",
        "RetrieveRerank",
        "RetrieveLLM",
        "RetrieveRerankLLM",
        "CustomTopology",
    ):
        base = _factory(name)
        factories.append(
            ArchitectureFactory(
                name=name,
                factory=base.factory,
                cache_semantics=(
                    "stochastic"
                    if name in {"RetrieveLLM", "RetrieveRerankLLM"}
                    else "deterministic"
                ),
                estimated_usd=None if name == "RetrieveLLM" else 0.0,
            )
        )
    loaded = False

    def load(_name: str) -> _Benchmark:
        nonlocal loaded
        loaded = True
        return _Benchmark()

    monkeypatch.setattr("langres.experiments.runner.get_benchmark", load)

    with pytest.raises(
        ValueError,
        match="complete USD preflight estimate",
    ):
        Experiment(
            architectures=factories,
            protocol=protocol,
            store=tmp_path / "runs.jsonl",
            cache_dir=tmp_path / "cache",
        ).run()

    assert loaded is False


def test_plan_never_counts_unrelated_cache_directories_as_hits(
    _offline: None,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    (cache_dir / "unrelated-cache-id").mkdir(parents=True)
    experiment = Experiment(
        architectures=[_factory("CustomTopology")],
        protocol=_protocol().model_copy(
            update={
                "benchmark_ids": ("one",),
                "architecture_repeats": {},
                "stochastic_repeats": 1,
            }
        ),
        cache_dir=cache_dir,
    )

    plan = experiment.plan()

    assert plan.cache_hits == 0
    assert plan.cache_misses == 0
    assert plan.cache_unknown == 1
    assert plan.cells[0].cache_status == "unknown"


def test_reproduction_bundle_and_cli_validate_in_a_clean_subprocess(
    _offline: None,
    tmp_path: Path,
) -> None:
    report = Experiment(
        architectures=[_factory("CustomTopology")],
        protocol=_protocol().model_copy(
            update={
                "benchmark_ids": ("one",),
                "architecture_repeats": {},
                "stochastic_repeats": 1,
            }
        ),
        store=tmp_path / "runs.jsonl",
        cache_dir=tmp_path / "cache",
        reproduction_path=tmp_path / "reproduce.json",
    ).run()

    assert isinstance(report, ExperimentReport)
    assert report.reproduce_command == (
        f"langres experiments reproduce {tmp_path / 'reproduce.json'}"
    )
    bundle = load_reproduction_bundle(tmp_path / "reproduce.json")
    assert isinstance(bundle, ReproductionBundle)
    assert bundle.report == report
    assert bundle.architectures[0].name == "CustomTopology"

    output = io.StringIO()
    assert (
        main(
            ["experiments", "reproduce", str(tmp_path / "reproduce.json")],
            output_stream=output,
        )
        == 0
    )
    assert "verified reproduction bundle" in output.getvalue().lower()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "langres.cli",
            "experiments",
            "reproduce",
            str(tmp_path / "reproduce.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "CustomTopology" in completed.stdout


class _TrackioSpy:
    name = "trackio"

    def start_run(self, context: Any, *, run_name: str | None = None) -> None:
        del context, run_name

    def log_params(self, params: Any) -> None:
        del params

    def log_metrics(self, metrics: Any, *, step: int | None = None) -> None:
        del metrics, step

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
