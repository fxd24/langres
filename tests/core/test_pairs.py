"""Tests for the ``Pairs`` carrier (W1-T1, #193) and the extracted score aliases.

Covers the carrier contract at the core tier (behavior + edges + errors):

- the id-referenced ``PairRow`` / ``Pairs`` shape and typed ``left``/``right``;
- the ``score_type=None`` "blocked, not yet scored" lifecycle;
- the additive bridges (``from_candidates`` / ``to_candidates`` /
  ``to_judgement``) and the ``F-W1a`` rule that a judge score never masquerades
  as a blocker ``similarity_score``;
- the ``F-W1b`` construction-time store binding (a copied row still resolves
  entities) and the ``F-W1c`` reference-not-deepcopy invariant;
- ``_store`` is never serialized;
- the frozen 7-value ``ScoreType`` and the schema-invariance of the extraction.
"""

import json
import typing
from pathlib import Path

import pytest

from langres.core.matchers import EmbeddingScoreMatcher
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.pairs import Pairs, PairRow, RecordStore
from langres.core.score_type import ConfidenceSource, ScoreType

# The historical, frozen sets — spelled literally here so a widening of the
# aliases fails LOUDLY against this test, not silently.
_HISTORICAL_SCORE_TYPES = {
    "sim_cos",
    "prob_llm",
    "heuristic",
    "calibrated_prob",
    "prob_fs",
    "prob_rf",
    "prob_group_llm",
}
_HISTORICAL_CONFIDENCE_SOURCES = {
    "none",
    "unrequested",
    "logprob",
    "calibrated",
    "heuristic",
}

_SCHEMA_GOLDEN = Path(__file__).parent / "goldens" / "pairwise_judgement_schema.json"


def _company(entity_id: str, name: str) -> CompanySchema:
    return CompanySchema(id=entity_id, name=name)


def _candidate(
    left: CompanySchema,
    right: CompanySchema,
    *,
    blocker_name: str = "allpairs",
    similarity_score: float | None = None,
    comparison: object = None,
) -> ERCandidate[CompanySchema]:
    return ERCandidate(
        left=left,
        right=right,
        blocker_name=blocker_name,
        similarity_score=similarity_score,
        comparison=comparison,  # type: ignore[arg-type]
    )


class TestScoreTypeAliases:
    """The extracted aliases are the frozen historical sets, no widening."""

    def test_score_type_is_the_frozen_seven(self) -> None:
        assert set(typing.get_args(ScoreType)) == _HISTORICAL_SCORE_TYPES

    def test_confidence_source_is_the_historical_five(self) -> None:
        assert set(typing.get_args(ConfidenceSource)) == _HISTORICAL_CONFIDENCE_SOURCES

    def test_pairwise_judgement_json_schema_unchanged_vs_golden(self) -> None:
        """The alias extraction must not move a byte of PairwiseJudgement's schema."""
        current = json.dumps(PairwiseJudgement.model_json_schema(), indent=2, sort_keys=True) + "\n"
        golden = _SCHEMA_GOLDEN.read_text()
        assert current == golden

    def test_models_uses_the_aliases_not_a_new_inline_literal(self) -> None:
        """The judgement's score_type accepts exactly the frozen set and nothing else."""
        enum = PairwiseJudgement.model_json_schema()["properties"]["score_type"]["enum"]
        assert set(enum) == _HISTORICAL_SCORE_TYPES


class TestPairRowMaterialization:
    """``left``/``right`` resolve the typed entity from the owning store."""

    def test_left_and_right_return_typed_entities(self) -> None:
        a, b = _company("a", "Acme"), _company("b", "Acme Inc")
        pairs = Pairs.from_candidates([_candidate(a, b)])
        row = pairs.rows[0]
        assert row.left.name == "Acme"
        assert row.right.name == "Acme Inc"

    def test_unbound_row_raises_on_entity_access(self) -> None:
        """A bare PairRow (not obtained from a Pairs) has no store to resolve against."""
        row: PairRow[CompanySchema] = PairRow(left_id="a", right_id="b", blocker_name="ap")
        with pytest.raises(RuntimeError, match="not bound to a record store"):
            _ = row.left

    def test_iter_yields_rows_and_len_counts_them(self) -> None:
        a, b, c = _company("a", "A"), _company("b", "B"), _company("c", "C")
        pairs = Pairs.from_candidates([_candidate(a, b), _candidate(a, c)])
        assert len(pairs) == 2
        rows = list(pairs)
        assert [r.left_id for r in rows] == ["a", "a"]
        # Every iterated row is bound (F-W1b): entity access works without error.
        assert rows[0].right.name == "B"

    def test_record_store_alias_is_importable(self) -> None:
        assert RecordStore is not None


class TestStoreBindingAtConstruction:
    """F-W1b: the store binds at construction, so a copied row still resolves."""

    def test_copied_row_resolves_entities(self) -> None:
        a, b = _company("a", "Acme"), _company("b", "Beta")
        pairs = Pairs(
            store={"a": a, "b": b},
            rows=[
                PairRow(
                    left_id="a", right_id="b", blocker_name="vec", score=0.9, score_type="sim_cos"
                )
            ],
        )
        copied = pairs.rows[0].model_copy(update={"reasoning": "changed"})
        # Does NOT raise and returns the entity — the private store survived the copy.
        assert copied.left is a
        assert copied.right.name == "Beta"

    def test_binding_happens_before_iteration(self) -> None:
        """Accessing rows[0].left directly (never iterating) still resolves."""
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs(
            store={"a": a, "b": b}, rows=[PairRow(left_id="a", right_id="b", blocker_name="ap")]
        )
        assert pairs.rows[0].left is a


class TestFromCandidates:
    """Folding legacy ERCandidates into id-rows + store."""

    def test_references_entities_never_deepcopies(self) -> None:
        """F-W1c: the store must hold the SAME entity objects, not copies."""
        a, b = _company("a", "Acme"), _company("b", "Beta")
        pairs = Pairs.from_candidates([_candidate(a, b)])
        assert pairs.store["a"] is a
        assert pairs.store["b"] is b

    def test_shared_entity_stored_once_first_wins(self) -> None:
        a, b, c = _company("a", "A"), _company("b", "B"), _company("c", "C")
        pairs = Pairs.from_candidates([_candidate(a, b), _candidate(a, c)])
        assert set(pairs.store) == {"a", "b", "c"}
        assert pairs.store["a"] is a

    def test_similarity_becomes_score_with_none_score_type(self) -> None:
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs.from_candidates([_candidate(a, b, similarity_score=0.75)])
        row = pairs.rows[0]
        assert row.score == 0.75
        assert row.score_type is None  # blocked, not yet scored
        assert row.blocker_name == "allpairs"

    def test_empty_candidates_yields_empty_pairs(self) -> None:
        pairs: Pairs[CompanySchema] = Pairs.from_candidates([])
        assert len(pairs) == 0
        assert pairs.store == {}


class TestToCandidates:
    """Projecting back to inline-entity ERCandidate form, keyed on score_type (F-W1a)."""

    def test_unscored_row_similarity_flows_back(self) -> None:
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs.from_candidates([_candidate(a, b, similarity_score=0.6)])
        cands = pairs.to_candidates()
        assert cands[0].similarity_score == 0.6
        # entities referenced from the store, not copied
        assert cands[0].left is a

    def test_scored_row_score_does_not_leak_into_similarity(self) -> None:
        """F-W1a: a SCORED row's judge score must NOT become similarity_score."""
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs(
            store={"a": a, "b": b},
            rows=[
                PairRow(
                    left_id="a",
                    right_id="b",
                    blocker_name="vec",
                    score=0.95,
                    score_type="prob_llm",
                    decision_step="llm",
                )
            ],
        )
        cands = pairs.to_candidates()
        assert cands[0].similarity_score is None

    def test_scored_candidates_do_not_corrupt_embedding_score_matcher(self) -> None:
        """The F-W1a payoff: a downstream EmbeddingScoreMatcher (reads
        ``candidate.similarity_score``) must NOT silently read the judge score.

        With similarity_score correctly ``None``, the matcher raises its clear
        "needs a VectorBlocker" ValueError instead of scoring on a judge number
        that was never a cosine similarity.
        """
        a, b = _company("a", "A"), _company("b", "B")
        scored = Pairs(
            store={"a": a, "b": b},
            rows=[
                PairRow(
                    left_id="a",
                    right_id="b",
                    blocker_name="vec",
                    score=0.95,
                    score_type="prob_llm",
                    decision_step="llm",
                )
            ],
        )
        matcher: EmbeddingScoreMatcher[CompanySchema] = EmbeddingScoreMatcher()
        with pytest.raises(ValueError, match="similarity_score"):
            list(matcher.forward(iter(scored.to_candidates())))

    def test_unscored_candidates_do_score_on_embedding_score_matcher(self) -> None:
        """The contrast: an UNSCORED row's blocker similarity DOES flow through."""
        a, b = _company("a", "A"), _company("b", "B")
        unscored = Pairs.from_candidates(
            [_candidate(a, b, blocker_name="vec", similarity_score=0.88)]
        )
        matcher: EmbeddingScoreMatcher[CompanySchema] = EmbeddingScoreMatcher()
        judgements = list(matcher.forward(iter(unscored.to_candidates())))
        assert judgements[0].score == 0.88
        assert judgements[0].score_type == "sim_cos"


class TestRoundTrip:
    """from_candidates(to_candidates(...)) is lossless with the score_type=None lifecycle."""

    def test_round_trip_is_lossless(self) -> None:
        a, b, c = _company("a", "A"), _company("b", "B"), _company("c", "C")
        cands = [
            _candidate(a, b, blocker_name="ap", similarity_score=0.5),
            _candidate(b, c, blocker_name="ap", similarity_score=None),
        ]
        p1 = Pairs.from_candidates(cands)
        p2 = Pairs.from_candidates(p1.to_candidates())
        assert p1.rows == p2.rows
        assert set(p1.store) == set(p2.store)
        # entity identity preserved through the whole round trip
        assert p2.store["a"] is a


class TestToJudgement:
    """Projecting a SCORED row to a PairwiseJudgement (ids only)."""

    def test_scored_row_projects_all_fields(self) -> None:
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs(
            store={"a": a, "b": b},
            rows=[
                PairRow(
                    left_id="a",
                    right_id="b",
                    blocker_name="vec",
                    score=0.9,
                    score_type="sim_cos",
                    decision=True,
                    confidence=0.7,
                    confidence_source="logprob",
                    decision_step="embedding_match",
                    reasoning="close",
                    provenance={"k": "v"},
                )
            ],
        )
        j = pairs.rows[0].to_judgement()
        assert isinstance(j, PairwiseJudgement)
        assert (j.left_id, j.right_id) == ("a", "b")
        assert j.score == 0.9
        assert j.score_type == "sim_cos"
        assert j.decision is True
        assert j.confidence == 0.7
        assert j.confidence_source == "logprob"
        assert j.decision_step == "embedding_match"
        assert j.reasoning == "close"
        assert j.provenance == {"k": "v"}

    def test_unscored_row_refuses_to_project(self) -> None:
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs.from_candidates([_candidate(a, b, similarity_score=0.5)])
        with pytest.raises(ValueError, match="score_type is None"):
            pairs.rows[0].to_judgement()


class TestVerdictHelpers:
    """is_abstain / predicted_match delegate to the models-layer semantics."""

    def test_scored_ranker_predicts_via_threshold(self) -> None:
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs(
            store={"a": a, "b": b},
            rows=[
                PairRow(
                    left_id="a", right_id="b", blocker_name="v", score=0.8, score_type="sim_cos"
                )
            ],
        )
        row = pairs.rows[0]
        assert row.predicted_match(0.5) is True
        assert row.predicted_match(0.9) is False
        assert row.is_abstain is False

    def test_scored_decider_predicts_via_decision(self) -> None:
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs(
            store={"a": a, "b": b},
            rows=[
                PairRow(
                    left_id="a",
                    right_id="b",
                    blocker_name="v",
                    score=None,
                    score_type="prob_llm",
                    decision=False,
                )
            ],
        )
        row = pairs.rows[0]
        # decision wins over the (absent) score; threshold is irrelevant
        assert row.predicted_match(0.0) is False
        assert row.is_abstain is False

    def test_scored_but_signalless_row_abstains(self) -> None:
        """A scored family tag with neither score nor decision is an abstention."""
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs(
            store={"a": a, "b": b},
            rows=[PairRow(left_id="a", right_id="b", blocker_name="v", score_type="prob_llm")],
        )
        row = pairs.rows[0]
        assert row.is_abstain is True
        assert row.predicted_match(0.5) is None

    def test_unscored_row_abstains_and_never_thresholds_similarity(self) -> None:
        """F-W1a: an unscored row's blocker similarity is NOT a match verdict."""
        a, b = _company("a", "A"), _company("b", "B")
        # a high blocker similarity must NOT read as a match through predicted_match
        pairs = Pairs.from_candidates([_candidate(a, b, similarity_score=0.99)])
        row = pairs.rows[0]
        assert row.is_abstain is True
        assert row.predicted_match(0.5) is None

    def test_unscored_row_with_explicit_decision_is_a_match(self) -> None:
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs(
            store={"a": a, "b": b},
            rows=[PairRow(left_id="a", right_id="b", blocker_name="ap", decision=True)],
        )
        row = pairs.rows[0]
        assert row.is_abstain is False
        assert row.predicted_match(0.5) is True


class TestSerialization:
    """The ``_store`` private attr is NEVER serialized."""

    def test_row_model_dump_excludes_store(self) -> None:
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs.from_candidates([_candidate(a, b, similarity_score=0.5)])
        dumped = pairs.rows[0].model_dump()
        assert "_store" not in dumped
        assert dumped["left_id"] == "a"

    def test_pairs_model_dump_excludes_store_from_rows(self) -> None:
        a, b = _company("a", "A"), _company("b", "B")
        pairs = Pairs.from_candidates([_candidate(a, b, similarity_score=0.5)])
        dumped = pairs.model_dump()
        assert "store" in dumped  # the owned store IS a field
        assert all("_store" not in row for row in dumped["rows"])

    def test_defaults_are_lifecycle_friendly(self) -> None:
        row: PairRow[CompanySchema] = PairRow(left_id="a", right_id="b", blocker_name="ap")
        assert row.score is None
        assert row.score_type is None
        assert row.decision is None
        assert row.confidence is None
        assert row.confidence_source == "none"
        assert row.decision_step == ""
        assert row.reasoning is None
        assert row.comparison is None
        assert row.provenance == {}
