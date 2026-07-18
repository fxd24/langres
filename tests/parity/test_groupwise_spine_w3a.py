"""W3-a spine parity for a GROUPWISE matcher (epic #193).

The W0 goldens (``tests/parity/test_behavior_parity_w0.py``) pin byte-identical
RESOLUTION for PAIRWISE matchers (``FuzzyString``, ``Resolver`` string), where a
matcher emits one judgement per candidate in candidate order -- so routing the
spine through the Op adapters
(:class:`~langres.core.op_adapters.MatcherScore`, whose ``_rescore`` re-emits in
the incoming carrier's row order) is order-preserving and invisible.

A GROUPWISE matcher is the one place the new spine's judgement *order* differs
from the legacy one. :class:`~langres.core.matcher.GroupwiseMatcher`'s
``forward`` regroups the pairwise stream via
:func:`~langres.core.groups.derive_groups_from_pairs` and emits in anchor-group
order, which need not equal the blocker's carrier order. The legacy spine
returned :meth:`~langres.core.resolver.Resolver.predict`'s judgements in that
emission order; the new spine returns them in deterministic carrier (blocker)
order (``MatcherScore`` maps each judgement back onto its row by ``(left_id,
right_id)`` identity). This is an intentional, better contract -- deterministic
and independent of a matcher's internal emission order -- but it is a real
behavior change that no golden covers. This test pins BOTH halves: the groupwise
RESOLUTION results stay correct, and ``predict`` returns carrier order.

``$0``: a scripted :class:`~langres.core.matcher.GroupwiseMatcher` (no LLM, no
network) that deliberately emits each group's judgements in REVERSED member
order, so its emission order differs from the carrier order. That reversal is a
faithful stand-in for any real set-wise matcher whose ``forward`` emission order
!= carrier order -- with ``AllPairsBlocker``'s already-anchor-grouped ``i < j``
stream a real ``SelectMatcher``'s emission happens to equal the carrier order, so
it could not exhibit (and therefore could not pin) the divergence this test
exists to lock down.
"""

from __future__ import annotations

from collections.abc import Iterator

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.groups import ERCandidateGroup
from langres.core.matcher import GroupwiseMatcher
from langres.core.models import CompanySchema, PairwiseJudgement
from langres.core.resolver import Resolver

_THRESHOLD = 0.5

#: Duplicate structure keyed by name: {a1, a2} Acme and {b1, b2} Beta are true
#: duplicates; c1 Gamma is a singleton (dropped by the clusterer).
RECORDS = [
    {"id": "a1", "name": "Acme"},
    {"id": "a2", "name": "Acme"},
    {"id": "b1", "name": "Beta"},
    {"id": "b2", "name": "Beta"},
    {"id": "c1", "name": "Gamma"},
]


class _ReversingGroupwiseMatcher(GroupwiseMatcher[CompanySchema]):
    """A ``$0`` set-wise matcher: score a member ``1.0`` iff its name equals the
    anchor's, else ``0.0``, tagging ``score_type="prob_group_llm"`` like a real
    ComEM-style select judge. It emits each group's judgements in REVERSED member
    order on purpose, so the matcher's emission order differs from the blocker's
    carrier order. No LLM, no network, no cost."""

    def forward_groups(
        self, groups: Iterator[ERCandidateGroup[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for group in groups:
            for member in reversed(group.members):
                yield PairwiseJudgement(
                    left_id=group.anchor.id,
                    right_id=member.id,
                    score=1.0 if member.name == group.anchor.name else 0.0,
                    score_type="prob_group_llm",
                    decision_step="reversing_groupwise",
                    provenance={"group_id": group.group_id},
                )


def _resolver() -> Resolver:
    """A Resolver with the reversing GROUPWISE matcher in the module slot."""
    return Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=_ReversingGroupwiseMatcher(),
        clusterer=Clusterer(threshold=_THRESHOLD),
    )


def _canonical(clusters: list[set[str]]) -> list[list[str]]:
    return sorted(sorted(cluster) for cluster in clusters)


def _emission_order() -> list[tuple[str, str]]:
    """The matcher's OWN (reversed-per-group) emission order via the real
    groupwise path -- what the legacy direct-call spine would have returned."""
    candidates = _resolver().candidates(RECORDS)
    matcher = _ReversingGroupwiseMatcher()
    return [(j.left_id, j.right_id) for j in matcher.forward(iter(candidates))]


def test_groupwise_resolve_and_dedupe_results_are_correct() -> None:
    """(a) A GROUPWISE matcher through the new Op-adapter spine still produces the
    correct clusters and the honest ``DedupeResult`` metadata."""
    resolver = _resolver()

    assert _canonical(resolver.resolve(RECORDS)) == [["a1", "a2"], ["b1", "b2"]]

    result = resolver.dedupe(RECORDS)
    assert _canonical(list(result)) == [["a1", "a2"], ["b1", "b2"]]
    assert result.score_type == "prob_group_llm"
    assert result.threshold == _THRESHOLD


def test_predict_returns_judgements_in_deterministic_carrier_order() -> None:
    """(b) ``predict`` returns judgements in deterministic blocker/carrier order --
    NOT the matcher's (reversed) emission order. Pins the intentional W3-a change:
    ``MatcherScore`` re-maps judgements onto their rows by identity, so the stream
    order is the blocker's, independent of how the matcher emitted."""
    resolver = _resolver()
    carrier_order = [
        (candidate.left.id, candidate.right.id) for candidate in resolver.candidates(RECORDS)
    ]
    predict_order = [
        (judgement.left_id, judgement.right_id) for judgement in resolver.predict(RECORDS)
    ]

    assert predict_order == carrier_order
    # Non-triviality: the matcher emits each group REVERSED, so its emission order
    # differs from the carrier order -- this assertion would fail if the new spine
    # passed the matcher's emission order through instead of re-sorting to carrier.
    assert predict_order != _emission_order()
