"""Resource-to-Op adapter tests over the existing Pairs carrier."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from langres.core.model_ref import ModelRef
from langres.core.models import CompanySchema, ERCandidate
from langres.core.op import Score, ThresholdSelect, TopKSelect
from langres.core.pairs import Pairs
from langres.resources import (
    FakeLLM,
    FakeReranker,
    Generate,
    GenerationBatch,
    GenerationEnvelope,
    LLMMatcherAdapter,
    Parse,
    ParsedGeneration,
    Rerank,
    RerankBatch,
    parse_binary_response,
    parse_score_response,
)


def _pairs() -> Pairs[CompanySchema]:
    acme = CompanySchema(id="a", name="Acme")
    acme_copy = CompanySchema(id="b", name="ACME")
    globex = CompanySchema(id="c", name="Globex")
    return Pairs.from_candidates(
        [
            ERCandidate(left=acme, right=acme_copy, blocker_name="test"),
            ERCandidate(left=acme, right=globex, blocker_name="test"),
        ]
    )


def test_rerank_is_a_reusable_score_whose_following_select_owns_role() -> None:
    resource = FakeReranker(scores={'["a","b"]': 0.9, '["a","c"]': 0.2})
    rerank = Rerank[CompanySchema](resource)

    rescored = rerank.forward(_pairs())
    top = TopKSelect[CompanySchema](1).forward(rescored)
    matched = ThresholdSelect[CompanySchema](0.8).forward(rescored)

    assert isinstance(rerank, Score)
    assert [(row.left_id, row.right_id) for row in top.rows] == [("a", "b")]
    assert [(row.left_id, row.right_id) for row in matched.rows] == [("a", "b")]
    assert all(row.score_type == "heuristic" for row in rescored.rows)
    assert resource.calls == 2


def test_resource_ops_reject_duplicate_ids_before_inference() -> None:
    source = _pairs()
    duplicate = Pairs(store=source.store, rows=[source.rows[0], source.rows[0]])
    reranker = FakeReranker()

    with pytest.raises(ValueError, match="Rerank requires unique pair_ids"):
        Rerank[CompanySchema](reranker).forward(duplicate)
    assert reranker.calls == 0
    with pytest.raises(ValueError, match="Generate requires unique request_ids"):
        Generate[CompanySchema](FakeLLM()).forward(duplicate)


def test_resource_ops_reject_model_identity_drift() -> None:
    wrong_ref = ModelRef(base="./wrong", kind="local")

    class _WrongReranker(FakeReranker):
        def rerank(self, pairs):
            batch = super().rerank(pairs)
            return RerankBatch(
                pair_ids=batch.pair_ids,
                scores=batch.scores,
                model_ref=wrong_ref,
            )

    class _WrongLLM(FakeLLM):
        def generate(self, requests):
            outputs = tuple(
                GenerationEnvelope.from_content(
                    request_id=request.request_id,
                    model_ref=wrong_ref,
                    content="MATCH",
                )
                for request in requests
            )
            return GenerationBatch(outputs=outputs, model_ref=wrong_ref)

    with pytest.raises(ValueError, match="different model_ref"):
        Rerank[CompanySchema](_WrongReranker()).forward(_pairs())
    with pytest.raises(ValueError, match="different model_ref"):
        Generate[CompanySchema](_WrongLLM()).forward(_pairs())


def test_generate_and_parse_exchange_typed_private_envelopes_in_provenance() -> None:
    generated = Generate[CompanySchema](
        FakeLLM(
            responses={
                '["a","b"]': "MATCH",
                '["a","c"]': "NO_MATCH",
            }
        )
    ).forward(_pairs())

    for row in generated.rows:
        envelope = row.provenance["_langres_generation"]
        assert envelope.content in {"MATCH", "NO_MATCH"}
        assert envelope.content not in envelope.model_dump_json()

    parsed = Parse[CompanySchema](parse_binary_response).forward(generated)

    assert [row.decision for row in parsed.rows] == [True, False]
    assert all(row.score is None for row in parsed.rows)
    assert all(row.score_type == "prob_llm" for row in parsed.rows)
    assert all("_langres_generation" not in row.provenance for row in parsed.rows)
    assert all("generation" in row.provenance for row in parsed.rows)
    assert all("raw_content" not in row.provenance["generation"] for row in parsed.rows)


def test_score_parser_preserves_score_and_reasoning() -> None:
    parsed = parse_score_response("MATCH\nScore: 0.75\nReasoning: same address")
    assert parsed.score == 0.75
    assert parsed.decision is None
    assert parsed.reasoning == "same address"
    assert parse_score_response("maybe").score is None


def test_parsed_generation_rejects_ambiguous_or_invalid_score() -> None:
    with pytest.raises(ValidationError, match="both decision and score"):
        ParsedGeneration(decision=True, score=0.9)
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        ParsedGeneration(score=1.1)


def test_parse_abstains_without_fabricating_a_score_on_malformed_content() -> None:
    generated = Generate[CompanySchema](FakeLLM(default_response="maybe")).forward(_pairs())

    parsed = Parse[CompanySchema](parse_binary_response, on_parse_error="abstain").forward(
        generated
    )

    assert all(row.decision is None for row in parsed.rows)
    assert all(row.score is None for row in parsed.rows)
    assert all(row.provenance["parse_error"] is True for row in parsed.rows)


def test_parse_raise_policy_and_missing_envelope_are_explicit() -> None:
    generated = Generate[CompanySchema](FakeLLM(default_response="maybe")).forward(_pairs())
    with pytest.raises(ValueError, match="Could not parse"):
        Parse[CompanySchema](parse_binary_response, on_parse_error="raise").forward(generated)
    with pytest.raises(ValueError, match="requires a local GenerationEnvelope"):
        Parse[CompanySchema](parse_binary_response).forward(_pairs())
    with pytest.raises(ValueError, match="must be 'abstain' or 'raise'"):
        Parse[CompanySchema](on_parse_error="ignore")  # type: ignore[arg-type]


def test_parse_restores_a_declared_local_payload_and_handles_parser_exception() -> None:
    generated = Generate[CompanySchema](FakeLLM(default_response="MATCH")).forward(_pairs())
    rows = []
    for row in generated.rows:
        envelope = row.provenance["_langres_generation"]
        provenance = dict(row.provenance)
        provenance["_langres_generation"] = envelope.local_payload()
        rows.append(row.model_copy(update={"provenance": provenance}))
    cached = Pairs(store=generated.store, rows=rows)

    def _broken_parser(content: str) -> ParsedGeneration:
        raise RuntimeError("broken")

    parsed = Parse[CompanySchema](_broken_parser).forward(cached)
    assert all(row.provenance["parse_error"] is True for row in parsed.rows)
    with pytest.raises(RuntimeError, match="broken"):
        Parse[CompanySchema](_broken_parser, on_parse_error="raise").forward(cached)


def test_llm_matcher_adapter_preserves_the_legacy_matcher_contract() -> None:
    candidates = _pairs().to_candidates()
    matcher = LLMMatcherAdapter[CompanySchema](
        FakeLLM(
            responses={
                '["a","b"]': "MATCH",
                '["a","c"]': "NO_MATCH",
            }
        ),
        parser=parse_binary_response,
    )

    judgements = list(matcher.forward(iter(candidates)))

    assert [judgement.decision for judgement in judgements] == [True, False]
    assert all(judgement.score_type == "prob_llm" for judgement in judgements)
