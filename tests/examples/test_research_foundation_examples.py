"""Offline smoke coverage for the research-foundation copy-paste examples."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from langres.core.model_ref import ModelRef
from langres.core.spend import SpendMonitor
from langres.experiments import expand_official_proof_matrix
from langres.resources import EmbeddingBatch

ROOT = Path(__file__).parents[2]
RESEARCH = ROOT / "examples" / "research"


def _load(name: str, path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def test_all_four_recipe_examples_run_without_network() -> None:
    module = _load("example_research_recipes", ROOT / "examples" / "research_recipes.py")

    recipes = module.build_recipes()
    clusters = {name: list(recipe.dedupe(module.RECORDS)) for name, recipe in recipes.items()}

    assert set(recipes) == {
        "Retrieve",
        "RetrieveRerank",
        "RetrieveLLM",
        "RetrieveRerankLLM",
    }
    assert {"a", "b"} in clusters["RetrieveRerank"]
    assert {"a", "b"} in clusters["RetrieveLLM"]
    assert {"a", "b"} in clusters["RetrieveRerankLLM"]


def _separability_module() -> ModuleType:
    return _load(
        "example_embedding_separability",
        ROOT / "examples" / "embedding_separability.py",
    )


#: Two records per cluster, three clusters, plus two singletons.
_SEP_IDS = ("a1", "a2", "b1", "b2", "c1", "c2", "d1", "e1")
_SEP_CLUSTERS = [{"a1", "a2"}, {"b1", "b2"}, {"c1", "c2"}, {"d1"}, {"e1"}]
_SEP_TEXTS = tuple(record_id[0] for record_id in _SEP_IDS)


class _ByClusterEmbedder:
    """Encodes a record's cluster letter as a one-hot: perfectly separable."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.calls.append(list(texts))
        letters = sorted({text.strip()[-1] for text in texts})
        vectors = np.zeros((len(texts), len(letters)), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, letters.index(text.strip()[-1])] = 1.0
        return EmbeddingBatch(
            vectors=vectors, model_ref=ModelRef(base="test/by-cluster", kind="hf")
        )


class _BlindEmbedder:
    """Maps every record to the same vector: carries no signal at all."""

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        vectors = np.ones((len(texts), 4), dtype=np.float32) / 2.0
        return EmbeddingBatch(vectors=vectors, model_ref=ModelRef(base="test/blind", kind="hf"))


def test_embedding_separability_example_can_tell_a_good_embedder_from_a_blind_one() -> None:
    """The property the old FakeEmbedder(dimension=32) demo did NOT have.

    That version embedded five hand-written strings with hash-derived
    pseudo-random vectors and printed a healthy-looking margin regardless of
    embedder quality — it measured its own hash function. These two embedders
    must land at opposite ends of the AUC scale, or the example is decorative.
    """
    module = _separability_module()

    good_auc, good_margin = module.separability(
        _ByClusterEmbedder(), _SEP_TEXTS, _SEP_IDS, _SEP_CLUSTERS
    )
    blind_auc, blind_margin = module.separability(
        _BlindEmbedder(), _SEP_TEXTS, _SEP_IDS, _SEP_CLUSTERS
    )

    assert good_auc == pytest.approx(1.0)
    assert good_margin > 0.0
    assert blind_auc == pytest.approx(0.5)
    assert blind_margin == pytest.approx(0.0)


def test_embedding_separability_example_encodes_queries_separately_under_an_instruction() -> None:
    """An instruction must produce a real second, prefixed encode pass.

    Documents stay generic and only the query side carries the instruction —
    the asymmetric shape instructional checkpoints document. If the example
    reused one encode for both sides the instruction would be unobservable,
    which is exactly the bug this stream fixed in ``FAISSIndex.search_all``.
    """
    module = _separability_module()
    embedder = _ByClusterEmbedder()

    module.separability(embedder, _SEP_TEXTS, _SEP_IDS, _SEP_CLUSTERS, instruction="find: ")

    assert len(embedder.calls) == 2
    assert embedder.calls[0] == list(_SEP_TEXTS)
    assert embedder.calls[1] == [f"find: {text}" for text in _SEP_TEXTS]


def test_embedding_separability_example_reuses_one_encode_without_an_instruction() -> None:
    """No instruction, no extra pass — the cheap symmetric default."""
    module = _separability_module()
    embedder = _ByClusterEmbedder()

    module.separability(embedder, _SEP_TEXTS, _SEP_IDS, _SEP_CLUSTERS)

    assert len(embedder.calls) == 1


def test_first_experiment_and_generated_table_use_real_reports(tmp_path: Path) -> None:
    first = _load("example_first_experiment", RESEARCH / "first_experiment.py")
    generator = _load("example_generate_smoke_table", RESEARCH / "generate_smoke_table.py")

    first_report = first.run_first_experiment(tmp_path / "first")
    smoke_report = generator.build_report(tmp_path / "table")
    rendered = generator.render(smoke_report)

    assert [run.status for run in first_report.runs] == ["completed"]
    assert len(smoke_report.runs) == 2
    assert all(run.status == "completed" for run in smoke_report.runs)
    assert "Generated by examples/research/generate_smoke_table.py" in rendered
    assert "| RetrieveRerank |" in rendered


def test_matrix_and_paid_proof_plans_are_exact_without_execution() -> None:
    matrix = _load("example_experiment_matrix", RESEARCH / "experiment_matrix.py")
    paid = _load("example_official_paid_proof", RESEARCH / "official_paid_proof.py")

    protocol = matrix.build_protocol()
    proof = paid.build_protocol()

    assert (
        2 * len(protocol.benchmark_ids) * len(protocol.split_ids) * len(protocol.split_seeds) == 16
    )
    assert len(expand_official_proof_matrix(proof)) == 18
    assert proof.budget_usd == 20.0
    assert paid.PAID_CONCURRENCY == 1


def test_matrix_execute_runs_all_16_core_only_cells(tmp_path: Path) -> None:
    script = f"""
import runpy
import sys
sys.path.insert(0, {str(RESEARCH)!r})
sys.argv = [
    "experiment_matrix.py",
    "--execute",
    "--output-dir",
    {str(tmp_path)!r},
]
runpy.run_path({str(RESEARCH / "experiment_matrix.py")!r}, run_name="__main__")
assert "faiss" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "16 cells:" in completed.stdout
    assert completed.stdout.count("| completed |") == 16


def test_research_factories_share_the_exact_cumulative_monitor() -> None:
    sys.path.insert(0, str(RESEARCH))
    try:
        helper = importlib.import_module("_research_foundation")
    finally:
        sys.path.remove(str(RESEARCH))
    factory = helper.retrieve_factory()
    monitor = SpendMonitor(budget_usd=2.0)

    first = factory.build(0.3, monitor)
    second = factory.build(0.7, monitor)
    monitor.add(0.75)

    assert first._spend_monitor is monitor
    assert second._spend_monitor is monitor
    assert first._spend_monitor.spent == second._spend_monitor.spent == 0.75


@pytest.mark.parametrize("revision", ["main", "latest"])
def test_paid_proof_rejects_mutable_hf_revisions(revision: str) -> None:
    paid = _load("example_official_paid_proof", RESEARCH / "official_paid_proof.py")

    with pytest.raises(ValueError, match="40-hex"):
        paid.build_factories(
            embedder_ref=ModelRef(base="org/embedder", kind="hf", revision=revision),
            reranker_ref=ModelRef(base="org/reranker", kind="hf", revision="a" * 40),
            llm_ref=ModelRef(base="provider/model", kind="api"),
        )


def test_paid_proof_plan_prints_guarded_next_command() -> None:
    completed = subprocess.run(
        [sys.executable, str(RESEARCH / "official_paid_proof.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "confirmation phrase: I_ACCEPT_USD_20_MAXIMUM" in completed.stdout
    assert "--execute-paid --confirm I_ACCEPT_USD_20_MAXIMUM" in completed.stdout
    assert "--embedder-revision <40-hex-commit-sha>" in completed.stdout
    assert "--reranker-revision <40-hex-commit-sha>" in completed.stdout


def test_local_hub_round_trip_needs_no_hub_client(tmp_path: Path) -> None:
    module = _load("example_hub_lifecycle", RESEARCH / "hub_lifecycle.py")

    loaded = module.local_round_trip(tmp_path / "bundle")

    assert type(loaded).__name__ == "FuzzyString"
    assert loaded.schema.__name__ == "HubLifecycleCompany"
    assert loaded.clusterer.threshold == 0.7


def test_example_modules_are_import_light() -> None:
    script = """
import runpy
import sys
paths = [
    "examples/embedding_separability.py",
    "examples/research_recipes.py",
    "examples/research/first_experiment.py",
    "examples/research/experiment_matrix.py",
    "examples/research/trackio_reproduction.py",
    "examples/research/reprice_tokens.py",
    "examples/research/hub_lifecycle.py",
    "examples/research/official_paid_proof.py",
]
sys.path.insert(0, "examples/research")
for path in paths:
    runpy.run_path(path, run_name="import_check")
heavy = {"torch", "faiss", "litellm", "trackio", "huggingface_hub"}
assert heavy.isdisjoint(sys.modules), heavy.intersection(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
