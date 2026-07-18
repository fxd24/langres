"""Zero-network proofs for the four research recipe factories."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from langres.architectures import (
    Retrieve,
    RetrieveLLM,
    RetrieveRerank,
    RetrieveRerankLLM,
)
from langres.core.clusterer import Clusterer
from langres.core.model_ref import ModelRef
from langres.core.models import CompanySchema
from langres.core.op import ThresholdSelect, TopKSelect
from langres.core.op_adapters import ClustererStage
from langres.core.resolver import ERModel
from langres.resources import (
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
    Generate,
    EmbeddingBatch,
    Parse,
    Rerank,
    Retrieve as RetrieveOp,
)

RECORDS = [
    {"id": "a", "name": "Acme"},
    {"id": "b", "name": "ACME"},
    {"id": "c", "name": "Globex"},
]
RERANK_SCORES = {
    '["a","b"]': 0.95,
    '["a","c"]': 0.1,
    '["b","c"]': 0.2,
}
LLM_RESPONSES = {
    '["a","b"]': "MATCH",
    '["a","c"]': "NO_MATCH",
    '["b","c"]': "NO_MATCH",
}


class _IdentifierOnly(BaseModel):
    id: str


class _FixedEmbedder(FakeEmbedder):
    def __init__(self, vectors: list[list[float]]) -> None:
        super().__init__(dimension=len(vectors[0]))
        self._vectors = np.asarray(vectors, dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        assert len(texts) == len(self._vectors)
        return EmbeddingBatch(vectors=self._vectors, model_ref=self.model_ref)


class _RecordingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.texts: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.texts.append(tuple(texts))
        return super().embed(texts)


def _canonical(model: ERModel) -> list[list[str]]:
    return sorted(sorted(cluster) for cluster in model.dedupe(RECORDS))


def test_all_four_recipes_run_with_zero_network_resources() -> None:
    embedder = FakeEmbedder()

    retrieve = Retrieve(
        embedder=embedder,
        schema=CompanySchema,
        retrieve_k=2,
        threshold=0.0,
    )
    retrieve_rerank = RetrieveRerank(
        embedder=embedder,
        reranker=FakeReranker(scores=RERANK_SCORES),
        schema=CompanySchema,
        retrieve_k=2,
        threshold=0.8,
    )
    retrieve_llm = RetrieveLLM(
        embedder=embedder,
        llm=FakeLLM(responses=LLM_RESPONSES),
        schema=CompanySchema,
        retrieve_k=2,
        llm_k=2,
    )
    retrieve_rerank_llm = RetrieveRerankLLM(
        embedder=embedder,
        reranker=FakeReranker(scores=RERANK_SCORES),
        llm=FakeLLM(responses=LLM_RESPONSES),
        schema=CompanySchema,
        retrieve_k=2,
        llm_k=2,
    )

    assert _canonical(retrieve) == [["a", "b", "c"]]
    assert _canonical(retrieve_rerank) == [["a", "b"]]
    assert _canonical(retrieve_llm) == [["a", "b"]]
    assert _canonical(retrieve_rerank_llm) == [["a", "b"]]


def test_recipe_resources_are_complete_and_backbone_is_singular_sugar() -> None:
    embedder = FakeEmbedder()
    reranker = FakeReranker()
    llm = FakeLLM()

    retrieve = Retrieve(embedder=embedder, schema=CompanySchema)
    retrieve_rerank = RetrieveRerank(
        embedder=embedder,
        reranker=reranker,
        schema=CompanySchema,
    )
    retrieve_llm = RetrieveLLM(
        embedder=embedder,
        llm=llm,
        schema=CompanySchema,
    )
    retrieve_rerank_llm = RetrieveRerankLLM(
        embedder=embedder,
        reranker=reranker,
        llm=llm,
        schema=CompanySchema,
    )

    assert retrieve.resources == {"embedder": embedder.model_ref}
    assert retrieve.backbone == embedder.model_ref.base
    assert retrieve_rerank.resources == {
        "embedder": embedder.model_ref,
        "reranker": reranker.model_ref,
    }
    assert retrieve_llm.resources == {
        "embedder": embedder.model_ref,
        "llm": llm.model_ref,
    }
    assert retrieve_rerank_llm.resources == {
        "embedder": embedder.model_ref,
        "reranker": reranker.model_ref,
        "llm": llm.model_ref,
    }
    assert retrieve_rerank.backbone is None
    assert retrieve_llm.backbone is None
    assert retrieve_rerank_llm.backbone is None


def test_api_and_local_llm_refs_share_one_recipe_contract() -> None:
    api = RetrieveLLM(
        embedder=FakeEmbedder(),
        llm=ModelRef(base="openai/gpt-4o-mini", kind="api"),
        schema=CompanySchema,
    )
    local = RetrieveLLM(
        embedder=FakeEmbedder(),
        llm=ModelRef(base="./models/tiny", kind="local"),
        schema=CompanySchema,
    )

    assert api.resources["llm"].kind == "api"
    assert local.resources["llm"].kind == "local"
    assert type(api) is type(local) is RetrieveLLM


def test_same_reranker_instance_conforms_before_different_selects() -> None:
    embedder = FakeEmbedder()
    reranker = FakeReranker(scores=RERANK_SCORES)
    recipe = RetrieveRerank(
        embedder=embedder,
        reranker=reranker,
        schema=CompanySchema,
        retrieve_k=2,
        threshold=0.8,
    )
    custom = ERModel.from_topology(
        ops=[
            RetrieveOp(embedder, schema=CompanySchema, k=2),
            Rerank(reranker),
            TopKSelect(1),
            ThresholdSelect(0.0),
            ClustererStage(Clusterer(threshold=0.0)),
        ]
    )

    assert recipe._ops is not None
    assert custom._ops is not None
    assert next(stage for stage in recipe._ops if isinstance(stage, Rerank)).resource is reranker
    assert next(stage for stage in custom._ops if isinstance(stage, Rerank)).resource is reranker
    assert [step.spec.role for step in custom.execution_plan().steps] == [
        "retrieve",
        "rerank",
        "topk_select",
        "threshold_select",
        "clusterer_stage",
    ]
    assert custom.execute(RECORDS).pairs.rows


def test_recipe_accepts_a_custom_downstream_clusterer() -> None:
    clusterer = Clusterer(threshold=0.7)
    recipe = Retrieve(
        embedder=FakeEmbedder(),
        schema=CompanySchema,
        clusterer=clusterer,
    )

    assert recipe._ops is not None
    stage = next(stage for stage in recipe._ops if isinstance(stage, ClustererStage))
    assert type(stage.clusterer) is type(clusterer)
    assert stage.clusterer is not clusterer
    assert stage.clusterer.threshold == 0.0
    assert clusterer.threshold == 0.7


def test_production_recipe_topology_round_trips_without_loading_weights(
    tmp_path: Path,
) -> None:
    recipe = RetrieveRerankLLM(
        embedder=ModelRef(base="org/embedder", kind="hf", revision="embed-sha"),
        reranker=ModelRef(base="org/reranker", kind="hf", revision="rerank-sha"),
        llm=ModelRef(base="org/llm", kind="hf", revision="llm-sha"),
        schema=CompanySchema,
        retrieve_k=20,
        llm_k=5,
    )

    recipe.save(tmp_path)
    loaded = ERModel.load(tmp_path)

    assert type(loaded) is RetrieveRerankLLM
    assert loaded.resources == recipe.resources
    assert [step.spec.role for step in loaded.execution_plan().steps] == [
        "retrieve",
        "rerank",
        "topk_select",
        "generate",
        "parse",
        "threshold_select",
        "clusterer_stage",
    ]


def test_production_recipe_loads_as_exact_class_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    recipe = RetrieveRerankLLM(
        embedder=ModelRef(base="org/embedder", kind="hf", revision="embed-sha"),
        reranker=ModelRef(base="org/reranker", kind="hf", revision="rerank-sha"),
        llm=ModelRef(base="org/llm", kind="hf", revision="llm-sha"),
        schema=CompanySchema,
    )
    recipe.save(tmp_path)

    script = """
import sys
from pathlib import Path

assert "langres.architectures.retrieval" not in sys.modules
from langres.core import ERModel
assert "langres.architectures.retrieval" not in sys.modules

model = ERModel.load(Path(sys.argv[1]))
assert type(model).__name__ == "RetrieveRerankLLM"
assert model.resources["embedder"].revision == "embed-sha"
assert model.resources["reranker"].revision == "rerank-sha"
assert model.resources["llm"].revision == "llm-sha"
assert [step.spec.role for step in model.execution_plan().steps] == [
    "retrieve",
    "rerank",
    "topk_select",
    "generate",
    "parse",
    "threshold_select",
    "clusterer_stage",
]
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        },
    )

    assert result.returncode == 0, result.stderr


def test_recipe_infers_schema_on_first_run_and_exposes_resources_before_binding() -> None:
    embedder = FakeEmbedder()
    recipe = Retrieve(embedder=embedder, retrieve_k=2, threshold=0.0)

    assert recipe.schema is None
    assert recipe.resources == {"embedder": embedder.model_ref}
    assert _canonical(recipe) == [["a", "b", "c"]]
    assert recipe.schema is not None


def test_schema_inference_reuses_coerced_records_for_first_dedupe() -> None:
    embedder = _RecordingEmbedder()
    recipe = Retrieve(embedder=embedder, retrieve_k=1, threshold=0.0)

    recipe.dedupe(
        [
            {"id": "a", "name": 123, "missing": float("nan")},
            {"id": "b", "name": 124, "missing": float("nan")},
        ]
    )

    assert embedder.texts == [("123", "124")]


def test_schema_inference_reuses_coerced_records_for_first_compare() -> None:
    embedder = _RecordingEmbedder()
    recipe = Retrieve(embedder=embedder, threshold=0.0)

    recipe.compare(
        {"id": "a", "name": 123, "missing": float("nan")},
        {"id": "b", "name": 124, "missing": float("nan")},
    )

    assert embedder.texts == [("123", "124")]


def test_explicit_retrieval_text_field_does_not_require_default_comparable_fields() -> None:
    retrieve = RetrieveOp(
        FakeEmbedder(),
        schema=_IdentifierOnly,
        text_field="id",
        k=1,
    )

    pairs = retrieve.forward([{"id": "a"}, {"id": "b"}])

    assert [(row.left_id, row.right_id) for row in pairs.rows] == [("a", "b")]


def test_retrieve_compare_returns_false_when_source_score_is_below_threshold() -> None:
    recipe = Retrieve(
        embedder=_FixedEmbedder([[1.0, 0.0], [-1.0, 0.0]]),
        schema=CompanySchema,
        threshold=0.5,
    )

    verdict = recipe.compare(RECORDS[0], RECORDS[1])

    assert verdict.match is False
    assert verdict.score == 0.0


def test_retrieve_preserves_reverse_hit_anchor_for_downstream_topk() -> None:
    retrieve = RetrieveOp(
        _FixedEmbedder(
            [
                [1.0, 0.0],
                [0.99, 0.1],
                [0.7, -0.7],
            ]
        ),
        schema=CompanySchema,
        k=1,
    )
    records = [
        {"id": "a", "name": "Alpha"},
        {"id": "b", "name": "Beta"},
        {"id": "z", "name": "Zed"},
    ]

    candidates = retrieve.forward(records)
    selected = TopKSelect[CompanySchema](1).forward(candidates)

    assert ("z", "a") in [(row.left_id, row.right_id) for row in candidates.rows]
    assert ("z", "a") in [(row.left_id, row.right_id) for row in selected.rows]


def test_recipe_clusterer_runs_after_the_recipe_cut_without_a_second_threshold() -> None:
    custom_clusterer = Clusterer(threshold=0.9)
    recipe = Retrieve(
        embedder=_FixedEmbedder([[1.0, 0.0], [0.8, 0.6]]),
        schema=CompanySchema,
        threshold=0.5,
        clusterer=custom_clusterer,
    )

    verdict = recipe.compare(RECORDS[0], RECORDS[1])
    result = recipe.dedupe(RECORDS[:2])
    stage = next(stage for stage in recipe._require_ops() if isinstance(stage, ClustererStage))

    assert verdict.match is True
    assert result == [{"a", "b"}]
    assert stage.clusterer is not custom_clusterer
    assert stage.clusterer.threshold == 0.0
    assert custom_clusterer.threshold == 0.9


def test_retrieve_compare_accepts_the_same_record_on_both_sides() -> None:
    recipe = Retrieve(
        embedder=_FixedEmbedder([[1.0, 0.0], [1.0, 0.0]]),
        schema=CompanySchema,
        threshold=0.5,
    )

    verdict = recipe.compare(RECORDS[0], RECORDS[0])

    assert verdict.match is True
    assert verdict.score == 1.0


def test_llm_recipes_reject_nonpositive_candidate_caps() -> None:
    with pytest.raises(ValueError, match="llm_k must be positive"):
        RetrieveLLM(
            embedder=FakeEmbedder(),
            llm=FakeLLM(),
            schema=CompanySchema,
            llm_k=0,
        )
    with pytest.raises(ValueError, match="llm_k must be positive"):
        RetrieveRerankLLM(
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            llm=FakeLLM(),
            schema=CompanySchema,
            llm_k=-1,
        )
