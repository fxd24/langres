"""Tests for GroupwiseMatcher and the group-call cost-stamping helper.

GroupwiseMatcher IS-A Matcher (E2): its concrete forward() derives groups
internally from the pairwise ERCandidate stream it receives and dispatches to
the abstract forward_groups(), so the existing Resolver execution spine
(Resolver._judgements -> module.forward), inspect_scores, the JudgementLog
boundary, and benchmark dispatch all keep working with ZERO changes -- no
parallel execution path. These tests pin that contract down, including an
end-to-end Resolver.resolve() run through a concrete GroupwiseMatcher.

stamp_group_cost is the E5 group-call cost convention helper: a groupwise
judge stamps the full call cost on the first judgement of a group, $0 on
siblings, and provenance["group_id"] on all, so cost aggregation downstream
sums to exactly one call's cost per group (never K-times over-counted).
"""

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.groups import ERCandidateGroup
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.matcher import GroupwiseMatcher, Matcher, stamp_group_cost
from langres.core.reports import ScoreInspectionReport
from langres.core.resolver import Resolver


class ProductSchema(BaseModel):
    """Second schema for schema-agnostic verification."""

    id: str
    title: str


class _RecordingGroupwiseModule(GroupwiseMatcher[CompanySchema]):
    """Concrete GroupwiseMatcher: matches every member to its anchor, score=1.0."""

    def __init__(self) -> None:
        self.forward_groups_calls: list[list[ERCandidateGroup[CompanySchema]]] = []

    def forward_groups(
        self, groups: Iterator[ERCandidateGroup[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        materialized = list(groups)
        self.forward_groups_calls.append(materialized)
        for group in materialized:
            for member in group.members:
                yield PairwiseJudgement(
                    left_id=group.anchor.id,
                    right_id=member.id,
                    score=1.0,
                    score_type="prob_group_llm",
                    decision_step="test_groupwise",
                    provenance={},
                )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        return ScoreInspectionReport(
            total_judgements=len(judgements),
            score_distribution={
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
            },
            high_scoring_examples=[],
            low_scoring_examples=[],
            recommendations=[],
        )


def _candidates(pairs: list[tuple[str, str]]) -> list[ERCandidate[CompanySchema]]:
    return [
        ERCandidate(
            left=CompanySchema(id=left_id, name=f"Company {left_id}"),
            right=CompanySchema(id=right_id, name=f"Company {right_id}"),
            blocker_name="test_blocker",
        )
        for left_id, right_id in pairs
    ]


# ---------------------------------------------------------------------------
# ABC / spine-preservation contract
# ---------------------------------------------------------------------------


def test_groupwise_module_is_a_module() -> None:
    """GroupwiseMatcher IS-A Matcher (E2): the Resolver spine dispatches unchanged."""
    assert issubclass(GroupwiseMatcher, Matcher)
    assert isinstance(_RecordingGroupwiseModule(), Matcher)


def test_cannot_instantiate_groupwise_module_without_forward_groups() -> None:
    """forward_groups() is abstract; a subclass missing it cannot be instantiated."""

    class IncompleteGroupwiseModule(GroupwiseMatcher[CompanySchema]):
        def inspect_scores(
            self, judgements: list[PairwiseJudgement], sample_size: int = 10
        ) -> ScoreInspectionReport:
            raise NotImplementedError

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteGroupwiseModule()  # type: ignore[abstract]


def test_cannot_instantiate_groupwise_module_directly() -> None:
    """GroupwiseMatcher itself stays abstract."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        GroupwiseMatcher()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# forward() derives groups internally and dispatches to forward_groups()
# ---------------------------------------------------------------------------


def test_forward_groups_pairwise_candidates_and_dispatches() -> None:
    """forward() groups the incoming pairwise stream and calls forward_groups() once."""
    module = _RecordingGroupwiseModule()
    candidates = _candidates([("a", "b"), ("a", "c"), ("d", "e")])

    judgements = list(module.forward(iter(candidates)))

    # forward_groups() was invoked exactly once, with grouped input.
    assert len(module.forward_groups_calls) == 1
    groups = module.forward_groups_calls[0]
    by_anchor = {g.group_id: g for g in groups}
    assert set(by_anchor) == {"a", "d"}
    assert {m.id for m in by_anchor["a"].members} == {"b", "c"}

    # forward() decomposes back to pairwise judgements (set-wise IN, pairwise OUT).
    assert len(judgements) == 3
    assert all(isinstance(j, PairwiseJudgement) for j in judgements)
    pairs = {(j.left_id, j.right_id) for j in judgements}
    assert pairs == {("a", "b"), ("a", "c"), ("d", "e")}


def test_forward_returns_iterator_of_pairwise_judgements() -> None:
    """forward()'s output type is Iterator[PairwiseJudgement] -- unchanged Matcher contract."""
    module = _RecordingGroupwiseModule()
    result = module.forward(iter(_candidates([("a", "b")])))

    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


def test_forward_handles_empty_candidate_stream() -> None:
    """No candidates -> no groups -> no judgements."""
    module = _RecordingGroupwiseModule()
    judgements = list(module.forward(iter([])))
    assert judgements == []


def test_forward_is_schema_agnostic_with_product_schema() -> None:
    """GroupwiseMatcher works with a second, unrelated schema (ProductSchema)."""

    class _ProductGroupwiseModule(GroupwiseMatcher[ProductSchema]):
        def forward_groups(
            self, groups: Iterator[ERCandidateGroup[ProductSchema]]
        ) -> Iterator[PairwiseJudgement]:
            for group in groups:
                for member in group.members:
                    yield PairwiseJudgement(
                        left_id=group.anchor.id,
                        right_id=member.id,
                        score=1.0,
                        score_type="prob_group_llm",
                        decision_step="test",
                        provenance={},
                    )

        def inspect_scores(
            self, judgements: list[PairwiseJudgement], sample_size: int = 10
        ) -> ScoreInspectionReport:
            raise NotImplementedError

    module = _ProductGroupwiseModule()
    candidates = iter(
        [
            ERCandidate(
                left=ProductSchema(id="p1", title="iPhone"),
                right=ProductSchema(id="p2", title="iPhone Pro"),
                blocker_name="test_blocker",
            )
        ]
    )

    judgements = list(module.forward(candidates))

    assert len(judgements) == 1
    assert judgements[0].left_id == "p1"
    assert judgements[0].right_id == "p2"


# ---------------------------------------------------------------------------
# End-to-end: the Resolver spine dispatches to a GroupwiseMatcher unchanged.
# ---------------------------------------------------------------------------


def test_resolver_resolve_dispatches_through_groupwise_module() -> None:
    """A full Resolver (blocker -> module -> clusterer) runs a GroupwiseMatcher
    with zero Resolver changes -- proving the spine is preserved end-to-end."""
    module = _RecordingGroupwiseModule()
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=module,
        clusterer=Clusterer(threshold=0.5),
    )
    records = [
        {"id": "a", "name": "Acme Corp"},
        {"id": "b", "name": "Acme Corporation"},
        {"id": "c", "name": "Beta Inc"},
    ]

    clusters = resolver.resolve(records)

    # module._RecordingGroupwiseModule always scores 1.0 -> all pairs merge.
    assert clusters == [{"a", "b", "c"}]
    assert len(module.forward_groups_calls) == 1


# ---------------------------------------------------------------------------
# stamp_group_cost: the E5 group-call cost convention helper
# ---------------------------------------------------------------------------


def _judgement(left_id: str, right_id: str) -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id=left_id,
        right_id=right_id,
        score=0.9,
        score_type="prob_group_llm",
        decision_step="select_judge",
        provenance={"model": "gpt-4o-mini"},
    )


def test_stamp_group_cost_puts_full_cost_on_first_judgement_only() -> None:
    """First judgement carries the full call cost; siblings carry $0."""
    judgements = [
        _judgement("anchor", "m1"),
        _judgement("anchor", "m2"),
        _judgement("anchor", "m3"),
    ]

    stamped = stamp_group_cost(judgements, call_cost_usd=0.03, group_id="anchor")

    assert stamped[0].provenance["cost_usd"] == pytest.approx(0.03)
    assert stamped[1].provenance["cost_usd"] == 0.0
    assert stamped[2].provenance["cost_usd"] == 0.0


def test_stamp_group_cost_sums_to_exactly_one_calls_cost() -> None:
    """sum(cost over the group) == one call's cost (E5's core invariant)."""
    judgements = [_judgement("anchor", f"m{i}") for i in range(5)]

    stamped = stamp_group_cost(judgements, call_cost_usd=0.05, group_id="anchor")

    total = sum(j.provenance["cost_usd"] for j in stamped)
    assert total == pytest.approx(0.05)


def test_stamp_group_cost_sets_group_id_on_all_judgements() -> None:
    """provenance["group_id"] is set on every judgement in the group, not just the first."""
    judgements = [_judgement("anchor", "m1"), _judgement("anchor", "m2")]

    stamped = stamp_group_cost(judgements, call_cost_usd=0.02, group_id="anchor")

    assert all(j.provenance["group_id"] == "anchor" for j in stamped)


def test_stamp_group_cost_preserves_other_provenance_fields() -> None:
    """Existing provenance keys (e.g. model) survive the stamping."""
    judgements = [_judgement("anchor", "m1")]

    stamped = stamp_group_cost(judgements, call_cost_usd=0.01, group_id="anchor")

    assert stamped[0].provenance["model"] == "gpt-4o-mini"


def test_stamp_group_cost_does_not_mutate_input_judgements() -> None:
    """stamp_group_cost returns new objects; the originals are untouched."""
    original = _judgement("anchor", "m1")

    stamp_group_cost([original], call_cost_usd=0.07, group_id="anchor")

    assert "cost_usd" not in original.provenance
    assert "group_id" not in original.provenance


def test_stamp_group_cost_sets_group_end_only_on_last_judgement() -> None:
    """group_end marks exactly the LAST judgement, so a consumer draining a
    whole group from a lazy stream (E9's _SpendCappedMatcher) can stop at the
    boundary without peeking at (and thereby computing) the next group."""
    judgements = [
        _judgement("anchor", "m1"),
        _judgement("anchor", "m2"),
        _judgement("anchor", "m3"),
    ]

    stamped = stamp_group_cost(judgements, call_cost_usd=0.03, group_id="anchor")

    assert "group_end" not in stamped[0].provenance
    assert "group_end" not in stamped[1].provenance
    assert stamped[2].provenance["group_end"] is True


def test_stamp_group_cost_sets_group_end_on_single_judgement_group() -> None:
    """A size-1 group's only judgement is both first and last -- group_end is True."""
    stamped = stamp_group_cost([_judgement("anchor", "m1")], call_cost_usd=0.01, group_id="anchor")

    assert stamped[0].provenance["cost_usd"] == pytest.approx(0.01)
    assert stamped[0].provenance["group_end"] is True


def test_stamp_group_cost_rejects_empty_list() -> None:
    """An empty judgement list is a caller bug, not a silently accepted no-op."""
    with pytest.raises(ValueError, match="at least one judgement"):
        stamp_group_cost([], call_cost_usd=0.01, group_id="anchor")
