"""Tests for SelectJudge: a ComEM-style set-wise judge (W1.1).

SelectJudge is a GroupwiseModule: one LLM call per group ("which single
candidate, if any, matches the anchor?") decomposed into K PairwiseJudgements,
instead of K separate pairwise calls. These tests pin down:

- the happy paths (a match selected, no match selected) across two schemas
  (CompanySchema, ProductSchema) -- schema-agnostic per project convention;
- the group-call cost convention (E5): full cost on the first judgement,
  $0 on siblings, sum-over-group == one call's cost;
- select_error handling (CEO #12): malformed LLM response, a selection that
  references a candidate outside the group, and a selection of more than one
  candidate -- all three map to whole-group "no match" + provenance
  ["select_error"], never a raised exception;
- the empty-group short-circuit (no members -> no LLM call, no judgements);
- inspect_scores() delegating to the shared, label-free report helper;
- serialization: SelectJudge is registered (type_name/@register), has a pure
  config (no lm/program), and a Resolver holding one in its module slot
  round-trips through save()/load() -- including in a FRESH PROCESS, so the
  no-pickle config-registry artifact contract (the same one DSPyJudge
  satisfies) holds for SelectJudge too.

DummyLM-driven throughout: $0, no network, matching the project's zero-spend
LLM test convention (see tests/core/modules/test_dspy_judge.py).
"""

import json
import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from dspy.utils.dummies import DummyLM
from pydantic import BaseModel

from langres.core.groups import ERCandidateGroup
from langres.core.models import CompanySchema, PairwiseJudgement
from langres.core.module import GroupwiseModule
from langres.core.modules.select_judge import SelectJudge
from langres.core.reports import ScoreInspectionReport


class ProductSchema(BaseModel):
    """Second schema for schema-agnostic verification (mirrors test_module_groupwise.py)."""

    id: str
    title: str


def _company(entity_id: str, name: str | None = None) -> CompanySchema:
    return CompanySchema(id=entity_id, name=name or f"Company {entity_id}")


def _group(
    anchor_id: str, member_ids: list[str], *, group_id: str | None = None
) -> ERCandidateGroup[CompanySchema]:
    return ERCandidateGroup(
        anchor=_company(anchor_id),
        members=[_company(m) for m in member_ids],
        group_id=group_id or anchor_id,
    )


def _answer(selected_ids: str, *, reasoning: str = "because") -> dict[str, str]:
    """One canned DummyLM answer keyed by SelectSignature's output field names."""
    return {"reasoning": reasoning, "selected_ids": selected_ids}


def _judge(answers: list[dict[str, str]]) -> SelectJudge[CompanySchema]:
    return SelectJudge(lm=DummyLM(answers), entity_noun="company")


# ---------------------------------------------------------------------------
# ABC / spine contract
# ---------------------------------------------------------------------------


def test_select_judge_is_a_groupwise_module() -> None:
    """SelectJudge IS-A GroupwiseModule IS-A Module -- the Resolver spine dispatches unchanged."""
    assert issubclass(SelectJudge, GroupwiseModule)
    assert isinstance(_judge([]), GroupwiseModule)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_forward_groups_scores_selected_member_one_and_others_zero() -> None:
    """The selected candidate id scores 1.0; every other member of the group scores 0.0."""
    judge = _judge([_answer('["m2"]')])
    group = _group("a", ["m1", "m2", "m3"])

    judgements = list(judge.forward_groups(iter([group])))

    by_right_id = {j.right_id: j for j in judgements}
    assert set(by_right_id) == {"m1", "m2", "m3"}
    assert by_right_id["m2"].score == 1.0
    assert by_right_id["m1"].score == 0.0
    assert by_right_id["m3"].score == 0.0
    assert all(j.left_id == "a" for j in judgements)
    assert all(j.score_type == "prob_group_llm" for j in judgements)
    assert all(j.decision_step == "select_judgment" for j in judgements)
    assert all(j.reasoning == "because" for j in judgements)


def test_forward_groups_no_selection_scores_every_member_zero() -> None:
    """An empty selection (no candidate matches the anchor) scores every member 0.0."""
    judge = _judge([_answer("[]")])
    group = _group("a", ["m1", "m2"])

    judgements = list(judge.forward_groups(iter([group])))

    assert len(judgements) == 2
    assert all(j.score == 0.0 for j in judgements)
    assert all(j.decision_step == "select_judgment" for j in judgements)


def test_forward_groups_is_schema_agnostic_with_product_schema() -> None:
    """SelectJudge works with a second, unrelated schema (ProductSchema)."""
    judge: SelectJudge[ProductSchema] = SelectJudge(
        lm=DummyLM([_answer('["p2"]')]), entity_noun="product"
    )
    group = ERCandidateGroup(
        anchor=ProductSchema(id="p1", title="iPhone"),
        members=[ProductSchema(id="p2", title="iPhone Pro"), ProductSchema(id="p3", title="Pixel")],
        group_id="p1",
    )

    judgements = list(judge.forward_groups(iter([group])))

    by_right_id = {j.right_id: j for j in judgements}
    assert by_right_id["p2"].score == 1.0
    assert by_right_id["p3"].score == 0.0


def test_forward_groups_multiple_groups_makes_one_call_each() -> None:
    """Two groups -> exactly two LLM calls (the structural call-count claim)."""
    judge = _judge([_answer('["m1"]'), _answer("[]")])
    groups = [_group("a", ["m1"]), _group("b", ["n1", "n2"])]

    judgements = list(judge.forward_groups(iter(groups)))

    lm = judge._get_lm()
    assert len(lm.history) == 2
    assert len(judgements) == 3  # 1 member in group a + 2 members in group b
    by_group = {j.provenance["group_id"]: [] for j in judgements}
    for j in judgements:
        by_group[j.provenance["group_id"]].append(j)
    assert {j.right_id for j in by_group["a"]} == {"m1"}
    assert {j.right_id for j in by_group["b"]} == {"n1", "n2"}


def test_forward_groups_empty_group_skips_llm_call() -> None:
    """A group with zero members yields nothing and never calls the LLM."""
    judge = _judge([])  # no canned answers -- a call would raise "No more responses"
    group = _group("a", [])

    judgements = list(judge.forward_groups(iter([group])))

    assert judgements == []


def test_forward_groups_empty_stream_yields_nothing() -> None:
    """No groups at all -> no judgements, no LLM calls."""
    judge = _judge([])
    assert list(judge.forward_groups(iter([]))) == []


def test_forward_dispatches_via_groupwise_forward() -> None:
    """The inherited GroupwiseModule.forward() (pairwise IN) reaches forward_groups()."""
    from langres.core.models import ERCandidate

    judge = _judge([_answer('["b"]')])
    candidates = iter(
        [
            ERCandidate(left=_company("a"), right=_company("b"), blocker_name="test"),
            ERCandidate(left=_company("a"), right=_company("c"), blocker_name="test"),
        ]
    )

    judgements = list(judge.forward(candidates))

    by_right_id = {j.right_id: j for j in judgements}
    assert by_right_id["b"].score == 1.0
    assert by_right_id["c"].score == 0.0


# ---------------------------------------------------------------------------
# select_error handling (CEO #12) -- never raise mid-stream
# ---------------------------------------------------------------------------


def test_forward_groups_select_error_on_malformed_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A DSPy parse error (no usable selected_ids at all) -> whole-group no-match + select_error."""
    judge = _judge([{"unexpected": "shape"}])
    group = _group("a", ["m1", "m2"])

    with caplog.at_level(logging.WARNING):
        judgements = list(judge.forward_groups(iter([group])))

    assert len(judgements) == 2
    assert all(j.score == 0.0 for j in judgements)
    assert all(j.decision_step == "select_judge_error" for j in judgements)
    assert all("malformed" in j.provenance["select_error"] for j in judgements)
    assert all(j.provenance["group_id"] == "a" for j in judgements)
    assert any("parse failure" in r.message for r in caplog.records)


def test_forward_groups_select_error_on_unknown_candidate_id() -> None:
    """A selection referencing an id outside the group -> whole-group no-match + select_error."""
    judge = _judge([_answer('["ghost"]')])
    group = _group("a", ["m1", "m2"])

    judgements = list(judge.forward_groups(iter([group])))

    assert len(judgements) == 2
    assert all(j.score == 0.0 for j in judgements)
    assert all(j.decision_step == "select_judge_error" for j in judgements)
    assert all("not in group" in j.provenance["select_error"] for j in judgements)
    assert all("ghost" in j.provenance["select_error"] for j in judgements)


def test_forward_groups_select_error_on_multiple_selected() -> None:
    """A selection naming more than one candidate -> whole-group no-match + select_error."""
    judge = _judge([_answer('["m1", "m2"]')])
    group = _group("a", ["m1", "m2", "m3"])

    judgements = list(judge.forward_groups(iter([group])))

    assert len(judgements) == 3
    assert all(j.score == 0.0 for j in judgements)
    assert all(j.decision_step == "select_judge_error" for j in judgements)
    assert all("expected at most one" in j.provenance["select_error"] for j in judgements)


def test_forward_groups_select_error_does_not_raise_and_continues_to_next_group() -> None:
    """A select_error on one group does not stop the stream -- later groups still score.

    Uses the unknown-candidate-id sub-case (not the malformed-response one): a
    successfully-parsed-but-semantically-invalid answer never triggers DSPy's
    internal parse-retry, so it consumes exactly one canned answer -- keeping
    this test's expectations independent of DSPy's (undocumented) retry count.
    """
    judge = _judge([_answer('["ghost"]'), _answer('["n1"]')])
    groups = [_group("a", ["m1"]), _group("b", ["n1"])]

    judgements = list(judge.forward_groups(iter(groups)))

    by_group = {j.provenance["group_id"]: j for j in judgements}
    assert by_group["a"].decision_step == "select_judge_error"
    assert by_group["b"].decision_step == "select_judgment"
    assert by_group["b"].score == 1.0


# ---------------------------------------------------------------------------
# Group-call cost convention (E5)
# ---------------------------------------------------------------------------


def test_forward_groups_stamps_full_cost_on_first_judgement_only(mocker) -> None:  # type: ignore[no-untyped-def]
    """The single call's cost lands on the first judgement; siblings carry $0."""
    judge = _judge([_answer('["m2"]')])
    mocker.patch.object(judge, "_cost_usd", return_value=0.03)
    group = _group("a", ["m1", "m2", "m3"])

    judgements = list(judge.forward_groups(iter([group])))

    costs = [j.provenance["cost_usd"] for j in judgements]
    assert costs[0] == pytest.approx(0.03)
    assert costs[1:] == [0.0, 0.0]


def test_forward_groups_group_cost_sums_to_exactly_one_calls_cost(mocker) -> None:  # type: ignore[no-untyped-def]
    """sum(cost over the group's judgements) == one call's cost -- E5's core invariant."""
    judge = _judge([_answer('["m2"]')])
    mocker.patch.object(judge, "_cost_usd", return_value=0.05)
    group = _group("a", ["m1", "m2", "m3", "m4"])

    judgements = list(judge.forward_groups(iter([group])))

    total = sum(j.provenance["cost_usd"] for j in judgements)
    assert total == pytest.approx(0.05)


def test_forward_groups_group_cost_sums_correctly_on_select_error_path(mocker) -> None:  # type: ignore[no-untyped-def]
    """The cost convention holds for select_error judgements too -- the call was still billed."""
    judge = _judge([_answer('["ghost"]')])
    mocker.patch.object(judge, "_cost_usd", return_value=0.02)
    group = _group("a", ["m1", "m2"])

    judgements = list(judge.forward_groups(iter([group])))

    total = sum(j.provenance["cost_usd"] for j in judgements)
    assert total == pytest.approx(0.02)


def test_forward_groups_sets_group_id_on_all_judgements() -> None:
    """provenance['group_id'] is set on every judgement in the group, including select_error."""
    judge = _judge([_answer('["m1"]')])
    group = _group("a", ["m1", "m2"], group_id="anchor-a")

    judgements = list(judge.forward_groups(iter([group])))

    assert all(j.provenance["group_id"] == "anchor-a" for j in judgements)


def test_cost_usd_uses_pinned_price_seam() -> None:
    """The honest-cost seam multiplies tokens by the injectable per-1k price (mirrors DSPyJudge)."""
    judge = _judge([])
    assert judge._cost_usd(1000, 500) == 0.0  # default price 0.0 -> $0
    judge.price_per_1k_tokens = 2.0
    assert judge._cost_usd(1000, 500) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# inspect_scores
# ---------------------------------------------------------------------------


def test_inspect_scores_delegates_to_shared_report_helper() -> None:
    """inspect_scores reuses the label-free report helper shared with other judges."""
    judge = _judge([])
    judgements = [
        PairwiseJudgement(
            left_id="a",
            right_id="b",
            score=0.9,
            score_type="prob_group_llm",
            decision_step="select_judgment",
            provenance={},
        ),
        PairwiseJudgement(
            left_id="a",
            right_id="c",
            score=0.1,
            score_type="prob_group_llm",
            decision_step="select_judgment",
            provenance={},
        ),
    ]

    report = judge.inspect_scores(judgements)

    assert isinstance(report, ScoreInspectionReport)
    assert report.total_judgements == 2


# ---------------------------------------------------------------------------
# Lazy LM construction
# ---------------------------------------------------------------------------


def test_get_lm_builds_dspy_lm_lazily(mocker) -> None:  # type: ignore[no-untyped-def]
    """With no injected LM, _get_lm builds a dspy.LM(model, cache=False) once."""
    sentinel = object()
    build = mocker.patch("dspy.LM", return_value=sentinel)
    judge: SelectJudge[CompanySchema] = SelectJudge(model="openrouter/z-ai/glm-5.2")
    assert judge._lm is None
    assert judge._get_lm() is sentinel
    assert judge._get_lm() is sentinel  # cached
    build.assert_called_once_with("openrouter/z-ai/glm-5.2", cache=False, temperature=0.0)


def test_injected_lm_is_used_and_never_rebuilt(mocker) -> None:  # type: ignore[no-untyped-def]
    """An injected LM wins -- dspy.LM is never constructed."""
    build = mocker.patch("dspy.LM")
    lm = DummyLM([])
    judge: SelectJudge[CompanySchema] = SelectJudge(lm=lm)
    assert judge._get_lm() is lm
    build.assert_not_called()


# ---------------------------------------------------------------------------
# Serialization: SelectJudge must satisfy the Resolver's no-pickle
# config-registry artifact contract, exactly like its DSPyJudge sibling.
# ---------------------------------------------------------------------------


def test_registered_under_select_judge_via_lazy_lookup() -> None:
    """get_component("select_judge") resolves SelectJudge via the lazy registry."""
    from langres.core.registry import get_component

    assert get_component("select_judge") is SelectJudge


def test_config_is_pure_and_excludes_lm() -> None:
    """config is a plain, JSON-able dict -- never the injected DSPy LM."""
    judge: SelectJudge[CompanySchema] = SelectJudge(
        model="openrouter/z-ai/glm-5.2", temperature=0.3, entity_noun="product"
    )
    config = judge.config
    assert config == {
        "model": "openrouter/z-ai/glm-5.2",
        "temperature": 0.3,
        "entity_noun": "product",
    }
    assert "lm" not in config


def test_from_config_builds_fresh_uncompiled_judge() -> None:
    """from_config rebuilds a judge with no injected LM (built lazily on first use)."""
    config: dict[str, object] = {
        "model": "openrouter/z-ai/glm-5.2",
        "temperature": 0.3,
        "entity_noun": "product",
    }
    judge = SelectJudge.from_config(config)
    assert judge.model == "openrouter/z-ai/glm-5.2"
    assert judge.temperature == 0.3
    assert judge.entity_noun == "product"
    assert judge._lm is None


def test_resolver_with_select_judge_saves_and_loads(tmp_path: Path) -> None:
    """A Resolver with a SelectJudge in the module slot round-trips in-process."""
    from langres.core import AllPairsBlocker, Clusterer, Resolver

    judge = _judge([_answer('["b"]')])
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=judge,
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.save(tmp_path)

    manifest = json.loads((tmp_path / "resolver.json").read_text())
    module_spec = next(c for c in manifest["components"] if c["slot"] == "module")
    assert module_spec["type_name"] == "select_judge"
    assert "lm" not in module_spec["config"]

    reloaded = Resolver.load(tmp_path)
    assert isinstance(reloaded.module, SelectJudge)
    assert reloaded.module.config == judge.config


@pytest.mark.slow
def test_resolver_load_select_judge_in_fresh_process_and_scores_a_group(tmp_path: Path) -> None:
    """A clean process can Resolver.load a select_judge artifact and score a group.

    Regression for the lazy-registration path (mirrors
    test_resolver_load_dspy_judge_in_fresh_process): ``@register("select_judge")``
    only fires when ``langres.core.modules.select_judge`` is imported, and
    ``langres.core`` deliberately does not import it (that would import
    ``dspy``). ``get_component`` imports it on demand, so a fresh process that
    only does ``from langres.core import Resolver`` still resolves the type --
    and the reloaded judge is actually exercised (forward_groups on a real
    group, DummyLM-injected), not just type-checked.
    """
    from langres.core import AllPairsBlocker, Clusterer, Resolver

    judge = _judge([_answer('["b"]')])
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=judge,
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.save(tmp_path)

    script = f"""
import sys
from dspy.utils.dummies import DummyLM
from langres.core import Resolver
from langres.core.groups import ERCandidateGroup
from langres.core.models import CompanySchema

r = Resolver.load(r"{tmp_path}")
assert type(r.module).__name__ == "SelectJudge", type(r.module).__name__

r.module._lm = DummyLM([{{"reasoning": "match", "selected_ids": '["b"]'}}])
group = ERCandidateGroup(
    anchor=CompanySchema(id="a", name="Acme"),
    members=[CompanySchema(id="b", name="Acme Inc")],
    group_id="a",
)
judgements = list(r.module.forward_groups(iter([group])))
assert len(judgements) == 1, judgements
assert judgements[0].score == 1.0, judgements[0]
assert judgements[0].score_type == "prob_group_llm"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "fresh-process Resolver.load failed for a select_judge artifact.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "UnknownComponentType" not in result.stderr
    assert "OK" in result.stdout
