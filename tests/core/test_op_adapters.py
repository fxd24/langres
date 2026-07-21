"""Tests for the Op-role adapters over the legacy components (W2, epic #193).

The adapters in :mod:`langres.core.op_adapters` are **additive and unused by the
spine this wave** (the flip that adopts them is W3), so these unit tests are their
ONLY correctness proof. Each adapter holds a legacy component and bridges its
forward through the :class:`~langres.core.pairs.Pairs` carrier; the tests assert
the bridged output matches driving the legacy component directly, and that a
BlockerSource -> ComparatorScore -> MatcherScore -> ClustererStage chain is
behavior-equivalent to the legacy ``Resolver.resolve`` path.
"""

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.clusterers import CorrelationClusterer
from langres.core.comparators import StringComparator
from langres.core.groups import ERCandidateGroup
from langres.core.matcher import GroupwiseMatcher, Matcher
from langres.core.matchers.rapidfuzz import RapidfuzzMatcher
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.op import ClusterStage, Finalize, Score, Source
from langres.core.op_adapters import (
    BlockerSource,
    CanonicalizeFinalize,
    ClustererStage,
    ComparatorScore,
    GroupwiseMatcherScore,
    MatcherScore,
)
from langres.core.pairs import Pairs
from langres.core.resolver import Resolver
from langres.curation.canonicalizer import Canonicalizer

# --------------------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------------------

_RECORDS = [
    {"id": "a", "name": "Acme Inc", "address": "1 Main St"},
    {"id": "b", "name": "Acme Incorporated", "address": "1 Main St"},
    {"id": "c", "name": "Globex", "address": "99 Side Rd"},
    {"id": "d", "name": "Globex Corp", "address": "99 Side Rd"},
]


@pytest.fixture
def resolver() -> Resolver:
    """A default string-matcher Resolver over CompanySchema (blocker+comparator+matcher+clusterer)."""
    return Resolver.from_schema(CompanySchema, matcher="string", threshold=0.7)


def _sorted_clusters(clusters: list[set[str]]) -> list[list[str]]:
    """Order-independent view of a clustering, for equality assertions."""
    return sorted(sorted(cluster) for cluster in clusters)


# --------------------------------------------------------------------------------------
# BlockerSource
# --------------------------------------------------------------------------------------


def test_blocker_source_is_a_source(resolver: Resolver) -> None:
    assert isinstance(BlockerSource(resolver.blocker), Source)


def test_blocker_source_yields_the_same_pairs_as_stream(resolver: Resolver) -> None:
    """BlockerSource(blocker).forward(records) == Pairs.from_candidates(blocker.stream(records))."""
    blocker = resolver.blocker
    expected = Pairs.from_candidates(list(blocker.stream(_RECORDS)))
    got = BlockerSource(blocker).forward(_RECORDS)

    assert [(row.left_id, row.right_id) for row in got.rows] == [
        (row.left_id, row.right_id) for row in expected.rows
    ]
    # Freshly blocked rows are unscored (a blocker similarity is not a judge score).
    assert all(row.score_type is None for row in got.rows)
    # The store holds every referenced entity once.
    assert set(got.store) == {"a", "b", "c", "d"}


# --------------------------------------------------------------------------------------
# ComparatorScore
# --------------------------------------------------------------------------------------


def test_comparator_score_is_a_vector_space_score(resolver: Resolver) -> None:
    adapter = ComparatorScore(resolver.comparator)
    assert isinstance(adapter, Score)
    assert adapter.scope == "pair"
    assert adapter.out_space == "vector"


def test_comparator_score_attaches_comparison_and_leaves_rows_unscored(resolver: Resolver) -> None:
    pairs = BlockerSource(resolver.blocker).forward(_RECORDS)
    assert all(row.comparison is None for row in pairs.rows)  # precondition

    out = ComparatorScore(resolver.comparator).forward(pairs)

    for row in out.rows:
        assert row.comparison is not None  # a ComparisonVector was attached
        assert row.score_type is None  # a vector is not a scalar: rows stay unscored
        assert row.score is None
    # Row identity/order preserved; entities carried by reference (never deep-copied).
    assert [(r.left_id, r.right_id) for r in out.rows] == [
        (r.left_id, r.right_id) for r in pairs.rows
    ]
    assert set(out.store) == set(pairs.store)
    assert all(out.store[k] is pairs.store[k] for k in pairs.store)


def test_comparator_score_matches_direct_compare(resolver: Resolver) -> None:
    """Each attached vector equals comparator.compare(left, right) called directly."""
    comparator = resolver.comparator
    pairs = BlockerSource(resolver.blocker).forward(_RECORDS)
    out = ComparatorScore(comparator).forward(pairs)

    for row in out.rows:
        direct = comparator.compare(row.left, row.right)
        assert row.comparison == direct


# --------------------------------------------------------------------------------------
# MatcherScore
# --------------------------------------------------------------------------------------


def test_matcher_score_is_a_score_with_declared_out_space(resolver: Resolver) -> None:
    # "unknown" is the honest declaration for a generic matcher (W3-b/W3-c): a
    # Matcher has no class-level score_type -- each row is stamped its own family
    # per-judgement in MatcherScore._rescore -- so the family is not knowable
    # pre-run. This mirrors what the spine declares in ModelRun._matcher_score.
    adapter = MatcherScore(resolver.module, out_space="unknown")
    assert isinstance(adapter, Score)
    assert adapter.scope == "pair"
    assert adapter.out_space == "unknown"


def test_matcher_score_rejects_unknown_out_space(resolver: Resolver) -> None:
    with pytest.raises(ValueError, match="out_space"):
        MatcherScore(resolver.module, out_space="not_a_family")  # type: ignore[arg-type]


def test_matcher_score_writes_scores_preserving_id_and_order(resolver: Resolver) -> None:
    """A rapidfuzz (string) MatcherScore writes score/score_type onto each row in order."""
    pairs = ComparatorScore(resolver.comparator).forward(
        BlockerSource(resolver.blocker).forward(_RECORDS)
    )
    # Drive the same matcher directly for the expected per-pair scores.
    matcher = resolver.module
    expected = {(j.left_id, j.right_id): j for j in matcher.forward(iter(pairs.to_candidates()))}

    out = MatcherScore(matcher, out_space="heuristic").forward(pairs)

    assert [(r.left_id, r.right_id) for r in out.rows] == [
        (r.left_id, r.right_id) for r in pairs.rows
    ]
    for row in out.rows:
        judgement = expected[(row.left_id, row.right_id)]
        assert row.score == judgement.score
        assert row.score_type == judgement.score_type == "heuristic"
        assert row.decision_step == judgement.decision_step
        # Round-trips back to an equivalent judgement.
        assert row.to_judgement().score == judgement.score


def test_matcher_score_maps_by_identity_not_position() -> None:
    """A matcher that yields judgements out of input order still lands on the right rows."""

    class _ReversingMatcher(Matcher[CompanySchema]):
        def forward(
            self, candidates: Iterator[ERCandidate[CompanySchema]]
        ) -> Iterator[PairwiseJudgement]:
            materialized = list(candidates)
            for candidate in reversed(materialized):  # deliberately reversed
                # A distinct score per pair so mislabeling would be visible.
                score = 0.9 if candidate.left.id < candidate.right.id else 0.1
                yield PairwiseJudgement(
                    left_id=candidate.left.id,  # type: ignore[attr-defined]
                    right_id=candidate.right.id,  # type: ignore[attr-defined]
                    score=score,
                    score_type="heuristic",
                    decision_step="reversing",
                    provenance={},
                )

    blocker = AllPairsBlocker(schema=CompanySchema)
    pairs = BlockerSource(blocker).forward(_RECORDS)
    out = MatcherScore(_ReversingMatcher(), out_space="heuristic").forward(pairs)

    # Order preserved (input order), and each row got its OWN pair's score.
    assert [(r.left_id, r.right_id) for r in out.rows] == [
        (r.left_id, r.right_id) for r in pairs.rows
    ]
    for row in out.rows:
        assert row.score == (0.9 if row.left_id < row.right_id else 0.1)


def test_matcher_score_over_a_decider_keeps_score_type_and_does_not_fabricate_a_score() -> None:
    """A decider matcher (decision set, score=None) -> row keeps score_type, score stays None."""

    class _DeciderMatcher(Matcher[CompanySchema]):
        def forward(
            self, candidates: Iterator[ERCandidate[CompanySchema]]
        ) -> Iterator[PairwiseJudgement]:
            for candidate in candidates:
                yield PairwiseJudgement(
                    left_id=candidate.left.id,  # type: ignore[attr-defined]
                    right_id=candidate.right.id,  # type: ignore[attr-defined]
                    decision=candidate.left.name == candidate.right.name,
                    score=None,  # a decider fabricates no score
                    score_type="prob_llm",
                    decision_step="decider",
                    provenance={},
                )

    blocker = AllPairsBlocker(schema=CompanySchema)
    pairs = BlockerSource(blocker).forward(_RECORDS)
    out = MatcherScore(_DeciderMatcher(), out_space="prob_llm").forward(pairs)

    for row in out.rows:
        assert row.score is None  # not fabricated
        assert row.score_type == "prob_llm"  # family still set
        assert row.decision is not None
        # A scored (decider) row is projectable to a judgement.
        assert row.to_judgement().decision == row.decision


def test_matcher_score_rejects_a_missing_judgement() -> None:
    """A matcher must return exactly one judgement for every incoming pair."""

    class _SkipFirstMatcher(Matcher[CompanySchema]):
        def forward(
            self, candidates: Iterator[ERCandidate[CompanySchema]]
        ) -> Iterator[PairwiseJudgement]:
            for index, candidate in enumerate(candidates):
                if index == 0:
                    continue  # emit nothing for the first candidate
                yield PairwiseJudgement(
                    left_id=candidate.left.id,  # type: ignore[attr-defined]
                    right_id=candidate.right.id,  # type: ignore[attr-defined]
                    score=0.8,
                    score_type="heuristic",
                    decision_step="skip_first",
                    provenance={},
                )

    blocker = AllPairsBlocker(schema=CompanySchema)
    pairs = BlockerSource(blocker).forward(_RECORDS)
    with pytest.raises(ValueError, match="missing judgements"):
        MatcherScore(_SkipFirstMatcher(), out_space="heuristic").forward(pairs)


def test_matcher_score_accepts_reversed_pair_orientation() -> None:
    """Legacy matchers may identify a requested undirected pair in reverse."""

    class _ReverseMatcher(Matcher[CompanySchema]):
        def forward(
            self, candidates: Iterator[ERCandidate[CompanySchema]]
        ) -> Iterator[PairwiseJudgement]:
            for candidate in candidates:
                yield PairwiseJudgement(
                    left_id=str(candidate.right.id),
                    right_id=str(candidate.left.id),
                    score=0.8,
                    score_type="heuristic",
                    decision_step="reverse",
                    provenance={},
                )

    pairs = BlockerSource(AllPairsBlocker(schema=CompanySchema)).forward(_RECORDS)
    out = MatcherScore(_ReverseMatcher(), out_space="heuristic").forward(pairs)

    assert [(row.left_id, row.right_id) for row in out.rows] == [
        (row.left_id, row.right_id) for row in pairs.rows
    ]
    assert all(row.score == 0.8 for row in out.rows)


def test_matcher_score_allows_empty_noop_only_for_unscored_pairs() -> None:
    """A classic non-trainable matcher can safely leave an unscored carrier alone."""

    class _EmptyMatcher(Matcher[CompanySchema]):
        def forward(
            self, candidates: Iterator[ERCandidate[CompanySchema]]
        ) -> Iterator[PairwiseJudgement]:
            yield from ()

    pairs = BlockerSource(AllPairsBlocker(schema=CompanySchema)).forward(_RECORDS)
    assert MatcherScore(_EmptyMatcher(), out_space="heuristic").forward(pairs) is pairs

    scored = pairs.model_copy(
        update={
            "rows": [
                row.model_copy(update={"score": 0.9, "score_type": "sim_cos"}) for row in pairs.rows
            ]
        }
    )
    with pytest.raises(ValueError, match="missing judgements"):
        MatcherScore(_EmptyMatcher(), out_space="heuristic").forward(scored)


def test_matcher_score_rejects_duplicate_judgements() -> None:
    """Duplicate output identities are ambiguous even if every input pair appears."""

    class _DuplicateMatcher(Matcher[CompanySchema]):
        def forward(
            self, candidates: Iterator[ERCandidate[CompanySchema]]
        ) -> Iterator[PairwiseJudgement]:
            materialized = list(candidates)
            for candidate in materialized:
                judgement = PairwiseJudgement(
                    left_id=str(candidate.left.id),
                    right_id=str(candidate.right.id),
                    score=0.8,
                    score_type="heuristic",
                    decision_step="duplicate",
                    provenance={},
                )
                yield judgement
                if candidate is materialized[0]:
                    yield judgement

    pairs = BlockerSource(AllPairsBlocker(schema=CompanySchema)).forward(_RECORDS)
    with pytest.raises(ValueError, match="duplicate judgements"):
        MatcherScore(_DuplicateMatcher(), out_space="heuristic").forward(pairs)


def test_matcher_score_rejects_an_unexpected_judgement() -> None:
    """A matcher cannot introduce a pair it was never asked to score."""

    class _UnexpectedMatcher(Matcher[CompanySchema]):
        def forward(
            self, candidates: Iterator[ERCandidate[CompanySchema]]
        ) -> Iterator[PairwiseJudgement]:
            for candidate in candidates:
                yield PairwiseJudgement(
                    left_id=str(candidate.left.id),
                    right_id=str(candidate.right.id),
                    score=0.8,
                    score_type="heuristic",
                    decision_step="expected",
                    provenance={},
                )
            yield PairwiseJudgement(
                left_id="not",
                right_id="requested",
                score=0.8,
                score_type="heuristic",
                decision_step="unexpected",
                provenance={},
            )

    pairs = BlockerSource(AllPairsBlocker(schema=CompanySchema)).forward(_RECORDS)
    with pytest.raises(ValueError, match="unexpected judgements"):
        MatcherScore(_UnexpectedMatcher(), out_space="heuristic").forward(pairs)


def test_matcher_score_rejects_duplicate_input_pairs() -> None:
    """The bijection is undefined when the incoming carrier repeats an identity."""
    pairs = BlockerSource(AllPairsBlocker(schema=CompanySchema)).forward(_RECORDS)
    duplicate = Pairs(store=pairs.store, rows=[pairs.rows[0], pairs.rows[0]])
    matcher = Resolver.from_schema(CompanySchema).module

    with pytest.raises(ValueError, match="duplicate input pairs"):
        MatcherScore(matcher, out_space="unknown").forward(duplicate)


# --------------------------------------------------------------------------------------
# GroupwiseMatcherScore
# --------------------------------------------------------------------------------------


class _MatchAllGroupwiseMatcher(GroupwiseMatcher[CompanySchema]):
    """Concrete GroupwiseMatcher: every member matches its anchor (score=1.0)."""

    def forward_groups(
        self, groups: Iterator[ERCandidateGroup[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for group in groups:
            for member in group.members:
                yield PairwiseJudgement(
                    left_id=group.anchor.id,
                    right_id=member.id,
                    score=1.0,
                    score_type="prob_group_llm",
                    decision_step="match_all_group",
                    provenance={"group_id": group.group_id},
                )

    def inspect_scores(self, judgements, sample_size=10):  # type: ignore[no-untyped-def]
        from langres.core.reports import _inspect_scores_impl

        return _inspect_scores_impl(judgements, sample_size)


def test_groupwise_matcher_score_is_a_group_scope_score() -> None:
    adapter = GroupwiseMatcherScore(_MatchAllGroupwiseMatcher())
    assert isinstance(adapter, Score)
    assert adapter.scope == "group"
    assert adapter.out_space == "prob_group_llm"


def test_groupwise_matcher_score_scores_every_row_via_forward_groups() -> None:
    blocker = AllPairsBlocker(schema=CompanySchema)
    pairs = BlockerSource(blocker).forward(_RECORDS)

    out = GroupwiseMatcherScore(_MatchAllGroupwiseMatcher()).forward(pairs)

    assert [(r.left_id, r.right_id) for r in out.rows] == [
        (r.left_id, r.right_id) for r in pairs.rows
    ]
    for row in out.rows:
        assert row.score == 1.0
        assert row.score_type == "prob_group_llm"


# --------------------------------------------------------------------------------------
# ClustererStage
# --------------------------------------------------------------------------------------


def test_clusterer_stage_is_a_cluster_stage(resolver: Resolver) -> None:
    assert isinstance(ClustererStage(resolver.clusterer), ClusterStage)


def test_clusterer_stage_maps_algorithm_from_type() -> None:
    assert ClustererStage(Clusterer(threshold=0.7)).algorithm == "transitive_closure"
    assert ClustererStage(CorrelationClusterer(threshold=0.7)).algorithm == "pivot"
    assert ClustererStage(CorrelationClusterer(threshold=0.7)).is_heuristic is True


def test_clusterer_stage_produces_same_clusters_as_direct_cluster(resolver: Resolver) -> None:
    """ClustererStage.forward(pairs) == clusterer.cluster(equivalent judgements)."""
    pairs = MatcherScore(resolver.module, out_space="heuristic").forward(
        ComparatorScore(resolver.comparator).forward(
            BlockerSource(resolver.blocker).forward(_RECORDS)
        )
    )
    clusterer = resolver.clusterer
    direct = clusterer.cluster([row.to_judgement() for row in pairs.rows])

    staged = ClustererStage(clusterer).forward(pairs)

    assert _sorted_clusters(staged) == _sorted_clusters(direct)


def test_clusterer_stage_skips_unscored_rows() -> None:
    """Unscored rows carry no judgement; ClustererStage clusters only scored rows."""
    blocker = AllPairsBlocker(schema=CompanySchema)
    pairs = BlockerSource(blocker).forward(_RECORDS)  # all rows unscored (score_type None)

    # No scored rows -> no edges -> no clusters (would raise if to_judgement ran on them).
    assert ClustererStage(Clusterer(threshold=0.7)).forward(pairs) == []


# --------------------------------------------------------------------------------------
# CanonicalizeFinalize
# --------------------------------------------------------------------------------------


def test_canonicalize_finalize_is_a_finalize() -> None:
    fin = CanonicalizeFinalize(Canonicalizer(), store={})
    assert isinstance(fin, Finalize)


class _ReverseSortedSet(set[str]):
    """A ``set`` that iterates in REVERSE-sorted order.

    Stands in for a real cluster set whose hash-seeded iteration order happens to
    be non-sorted, so the determinism test below exercises — deterministically,
    without depending on ``PYTHONHASHSEED`` — the worst-case order that
    :meth:`CanonicalizeFinalize.forward`'s ``sorted()`` must normalize away.
    """

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(sorted(super().__iter__(), reverse=True))


def test_canonicalize_finalize_fuses_a_cluster() -> None:
    """A cluster of complementary records fuses into one golden record (survivorship)."""
    store = {
        "a": CompanySchema(id="a", name="Acme", address="1 Main St", phone=None),
        "b": CompanySchema(id="b", name="Acme Inc", address=None, phone="555-1234"),
    }
    golden = CanonicalizeFinalize(Canonicalizer(), store=store).forward([{"a", "b"}])

    assert isinstance(golden, CompanySchema)  # a GoldenRecord (BaseModel)
    # most_complete default: each field filled from whichever record carried it.
    assert golden.address == "1 Main St"
    assert golden.phone == "555-1234"


def test_canonicalize_finalize_is_deterministic_regardless_of_cluster_iteration_order() -> None:
    """The golden id + first-seen tiebreaks must not depend on set iteration order.

    The cluster is fed in reverse-sorted iteration order. Without ``forward``'s
    ``sorted()`` normalization the first record would be ``"b"`` — so the golden id
    (defaults to ``records[0]["id"]``) and the ``most_complete`` first-seen tiebreak
    on an equally-complete field would both follow ``"b"``. This test pins the
    ``"a"`` (sorted-first) outcome, so it FAILS without the ``sorted()`` fix.
    """
    store = {
        # Both records equally complete (name + address present), differing only in
        # name -> most_complete breaks the tie by group order, so which name wins is
        # purely a function of record order (hence of the sorted() normalization).
        "a": CompanySchema(id="a", name="Acme A", address="1 Main St"),
        "b": CompanySchema(id="b", name="Acme B", address="1 Main St"),
    }
    golden = CanonicalizeFinalize(Canonicalizer(), store=store).forward(
        [_ReverseSortedSet({"a", "b"})]
    )

    assert golden.id == "a"  # sorted-first id, NOT iteration-first ("b")
    assert golden.name == "Acme A"  # first-seen tiebreak follows the sorted-first record


def test_canonicalize_finalize_requires_exactly_one_cluster() -> None:
    store = {"a": CompanySchema(id="a", name="Acme"), "b": CompanySchema(id="b", name="Globex")}
    fin = CanonicalizeFinalize(Canonicalizer(), store=store)
    with pytest.raises(ValueError, match="exactly one cluster"):
        fin.forward([{"a"}, {"b"}])
    with pytest.raises(ValueError, match="empty cluster"):
        fin.forward([set()])


# --------------------------------------------------------------------------------------
# Round-trip: the adapter chain is behavior-equivalent to Resolver.resolve
# --------------------------------------------------------------------------------------


def test_adapter_chain_matches_legacy_resolver_resolve(resolver: Resolver) -> None:
    """BlockerSource -> ComparatorScore -> MatcherScore -> ClustererStage == Resolver.resolve."""
    legacy = resolver.resolve(_RECORDS)

    pairs = BlockerSource(resolver.blocker).forward(_RECORDS)
    pairs = ComparatorScore(resolver.comparator).forward(pairs)
    pairs = MatcherScore(resolver.module, out_space="heuristic").forward(pairs)
    adapter_clusters = ClustererStage(resolver.clusterer).forward(pairs)

    assert _sorted_clusters(adapter_clusters) == _sorted_clusters(legacy)


def test_adapter_chain_with_rapidfuzz_and_pivot_clusterer() -> None:
    """A second chain (rapidfuzz Score + CorrelationClusterer) also matches its legacy drive."""
    blocker = AllPairsBlocker(schema=CompanySchema)
    comparator = StringComparator.from_schema(CompanySchema)
    matcher: RapidfuzzMatcher[CompanySchema] = RapidfuzzMatcher(
        field_extractors={"name": (lambda e: e.name, 1.0)}
    )
    clusterer = CorrelationClusterer(threshold=0.7)

    # Legacy drive.
    candidates = [
        c.model_copy(update={"comparison": comparator.compare(c.left, c.right)})
        for c in blocker.stream(_RECORDS)
    ]
    legacy = clusterer.cluster(list(matcher.forward(iter(candidates))))

    # Adapter chain over the same instances.
    pairs = MatcherScore(matcher, out_space="heuristic").forward(
        ComparatorScore(comparator).forward(BlockerSource(blocker).forward(_RECORDS))
    )
    adapter_clusters = ClustererStage(clusterer).forward(pairs)

    assert _sorted_clusters(adapter_clusters) == _sorted_clusters(legacy)
