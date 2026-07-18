"""Resource operations persist and execute through the explicit topology."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from langres.core._artifacts import op_spec, rebuild_op
from langres.core.blockers import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.models import CompanySchema
from langres.core.op import ThresholdSelect
from langres.core.op_adapters import BlockerSource, ClustererStage
from langres.core.registry import register
from langres.core.resolver import ERModel
from langres.core.serialization import ComponentSpec, OpSpec
from langres.resources import FakeLLM, FakeReranker, Generate, Parse, Rerank


@register("test_persist_resource_reranker")
class _PersistReranker(FakeReranker):
    """Registered zero-network reranker used only by persistence tests."""

    type_name: ClassVar[str] = "test_persist_resource_reranker"

    @property
    def config(self) -> dict[str, object]:
        return {"scores": dict(self._scores)}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "_PersistReranker":
        scores = config.get("scores")
        if not isinstance(scores, dict):
            raise ValueError("test reranker config requires scores")
        return cls(scores={str(key): float(value) for key, value in scores.items()})


@register("test_persist_resource_llm")
class _PersistLLM(FakeLLM):
    """Registered zero-network LLM used only by persistence tests."""

    type_name: ClassVar[str] = "test_persist_resource_llm"

    @property
    def config(self) -> dict[str, object]:
        return {
            "responses": dict(self._responses),
            "default_response": self.default_response,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "_PersistLLM":
        responses = config.get("responses")
        default_response = config.get("default_response")
        if not isinstance(responses, dict) or not isinstance(default_response, str):
            raise ValueError("test LLM config requires responses and default_response")
        return cls(
            responses={str(key): str(value) for key, value in responses.items()},
            default_response=default_response,
        )


_RECORDS = [
    {"id": "a", "name": "Acme"},
    {"id": "b", "name": "ACME"},
    {"id": "c", "name": "Globex"},
]
_RESPONSES = {
    '["a","b"]': "MATCH",
    '["a","c"]': "NO_MATCH",
    '["b","c"]': "NO_MATCH",
}


def _resource_model(*, include_rerank: bool) -> ERModel:
    body: list[object] = []
    if include_rerank:
        body.append(
            Rerank[CompanySchema](
                _PersistReranker(
                    scores={
                        '["a","b"]': 0.95,
                        '["a","c"]': 0.1,
                        '["b","c"]': 0.2,
                    }
                )
            )
        )
    body.extend(
        [
            Generate[CompanySchema](_PersistLLM(responses=_RESPONSES)),
            Parse[CompanySchema](),
            ThresholdSelect[CompanySchema](0.5),
        ]
    )
    return ERModel.from_topology(
        ops=[
            BlockerSource(AllPairsBlocker(schema=CompanySchema)),
            *body,
            ClustererStage(Clusterer(threshold=0.0)),
        ]
    )


@pytest.mark.parametrize(
    ("include_rerank", "roles"),
    [
        (
            False,
            [
                "blocker_source",
                "generate",
                "parse",
                "threshold_select",
                "clusterer_stage",
            ],
        ),
        (
            True,
            [
                "blocker_source",
                "rerank",
                "generate",
                "parse",
                "threshold_select",
                "clusterer_stage",
            ],
        ),
    ],
)
def test_resource_topology_save_load_execute_round_trip(
    tmp_path: Path,
    *,
    include_rerank: bool,
    roles: list[str],
) -> None:
    model = _resource_model(include_rerank=include_rerank)
    before = model.execute(_RECORDS)

    model.save(tmp_path)
    loaded = ERModel.load(tmp_path)
    after = loaded.execute(_RECORDS)

    assert [step.spec.role for step in before.plan.steps] == roles
    assert [step.spec.role for step in after.plan.steps] == roles
    assert after.clusters == before.clusters == (frozenset({"a", "b"}),)
    assert [(row.left_id, row.right_id, row.decision, row.score) for row in after.pairs.rows] == [
        (row.left_id, row.right_id, row.decision, row.score) for row in before.pairs.rows
    ]
    assert loaded._ops is not None
    loaded_generate = next(stage for stage in loaded._ops if isinstance(stage, Generate))
    assert loaded_generate.spend_monitor is loaded._spend_monitor


def test_resource_op_specs_use_strict_resource_slot_contract(tmp_path: Path) -> None:
    rerank_spec = op_spec(Rerank[CompanySchema](_PersistReranker()))
    generate_spec = op_spec(Generate[CompanySchema](_PersistLLM()))
    parse_spec = op_spec(Parse[CompanySchema]())

    assert (rerank_spec.role, rerank_spec.component.slot) == ("rerank", "resource")
    assert (generate_spec.role, generate_spec.component.slot) == ("generate", "resource")
    assert parse_spec.role == "parse"
    assert parse_spec.component is None

    with pytest.raises(ValueError, match="extra_forbidden"):
        rebuild_op(
            OpSpec(
                role="generate",
                params={"request_builder": "binary_pair", "unexpected": True},
                component=ComponentSpec(
                    type_name="must_not_be_resolved",
                    slot="resource",
                    config={},
                ),
            ),
            state_dir=tmp_path,
        )


def test_resource_op_rebuild_checks_nested_resource_protocol(tmp_path: Path) -> None:
    llm_spec = op_spec(Generate[CompanySchema](_PersistLLM())).component
    assert llm_spec is not None

    with pytest.raises(TypeError, match="requires a Reranker resource"):
        rebuild_op(
            OpSpec(
                role="rerank",
                params={"out_space": "heuristic", "serializer": "json"},
                component=llm_spec,
            ),
            state_dir=tmp_path,
        )
