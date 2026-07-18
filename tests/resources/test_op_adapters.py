"""Resource-to-Op adapter tests over the existing Pairs carrier."""

from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from langres.core.model_ref import ModelRef
from langres.core.models import CompanySchema, ERCandidate
from langres.core.op import Score, Spending, ThresholdSelect, TopKSelect
from langres.core.pairs import Pairs
from langres.core.spend import BudgetExceeded, SpendMonitor, UnknownSpendError
from langres.core.spend_cap import SpendCappedMatcher
from langres.resources import (
    FakeLLM,
    FakeReranker,
    Generate,
    GenerationBatch,
    GenerationEnvelope,
    LLMMatcherAdapter,
    LiteLLM,
    Parse,
    ParsedGeneration,
    Rerank,
    RerankBatch,
    UnknownGenerationCostError,
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


def test_builtin_resource_ops_expose_safe_round_trip_params() -> None:
    reranker = FakeReranker(scores={'["a","b"]': 0.9, '["a","c"]': 0.2})
    rerank = Rerank[CompanySchema](reranker)
    llm = FakeLLM(default_response="MATCH")
    generate = Generate[CompanySchema](llm)
    parse = Parse[CompanySchema]()

    rebuilt_rerank = Rerank[CompanySchema].from_config(reranker, rerank.config)
    rebuilt_generate = Generate[CompanySchema].from_config(llm, generate.config)
    rebuilt_parse = Parse[CompanySchema].from_config(parse.config)

    assert rebuilt_rerank.config == rerank.config
    assert rebuilt_generate.config == generate.config
    assert rebuilt_parse.config == parse.config
    assert [
        row.decision for row in rebuilt_parse.forward(rebuilt_generate.forward(_pairs())).rows
    ] == [
        True,
        True,
    ]


def test_resource_op_persistence_rejects_custom_callables() -> None:
    def custom_serializer(record):
        return record.model_dump_json()

    def custom_builder(row):
        return GenerationRequest.user("custom", "prompt")

    def custom_parser(content):
        return ParsedGeneration(decision=True)

    with pytest.raises(TypeError, match="custom callable"):
        _ = Rerank[CompanySchema](FakeReranker(), serializer=custom_serializer).config
    with pytest.raises(TypeError, match="custom callable"):
        _ = Generate[CompanySchema](FakeLLM(), request_builder=custom_builder).config
    with pytest.raises(TypeError, match="custom callable"):
        _ = Parse[CompanySchema](custom_parser).config


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
    assert all("cost_usd" not in row.provenance for row in parsed.rows)


def test_generate_and_parse_defaults_are_compatible() -> None:
    generated = Generate[CompanySchema](
        FakeLLM(
            responses={
                '["a","b"]': "MATCH",
                '["a","c"]': "NO_MATCH",
            }
        )
    ).forward(_pairs())

    parsed = Parse[CompanySchema]().forward(generated)

    assert [row.decision for row in parsed.rows] == [True, False]
    assert all(row.provenance["parse_error"] is False for row in parsed.rows)


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
        )
    )

    judgements = list(matcher.forward(iter(candidates)))

    assert [judgement.decision for judgement in judgements] == [True, False]
    assert all(judgement.score_type == "prob_llm" for judgement in judgements)


def test_llm_matcher_adapter_yields_between_provider_calls_for_outer_cap() -> None:
    class _CostedLLM(FakeLLM):
        calls = 0

        def generate(self, requests):
            self.calls += 1
            batch = super().generate(requests)
            return batch.model_copy(
                update={
                    "outputs": tuple(
                        output.model_copy(update={"cost_usd": 0.6, "cost_basis": "real"})
                        for output in batch.outputs
                    )
                }
            )

    resource = _CostedLLM(default_response="MATCH")
    matcher = SpendCappedMatcher(
        LLMMatcherAdapter[CompanySchema](resource),
        budget_usd=0.5,
    )

    with pytest.raises(BudgetExceeded) as exc_info:
        list(matcher.forward(iter(_pairs().to_candidates())))

    assert resource.calls == 1
    assert len(exc_info.value.partial_judgements) == 1


def test_llm_matcher_adapter_unknown_paid_cost_poison_outer_cap() -> None:
    class _UnknownPaidLLM(FakeLLM):
        requires_cost_accounting = True
        calls = 0

        def generate(self, requests):
            self.calls += 1
            return super().generate(requests)

    resource = _UnknownPaidLLM(default_response="MATCH")
    matcher = SpendCappedMatcher(
        LLMMatcherAdapter[CompanySchema](resource),
        budget_usd=0.0,
    )

    with pytest.raises(UnknownSpendError) as exc_info:
        list(matcher.forward(iter(_pairs().to_candidates())))

    assert resource.calls == 1
    assert len(exc_info.value.partial_judgements) == 1
    assert "cost_usd" not in exc_info.value.partial_judgements[0].provenance
    assert exc_info.value.partial_judgements[0].provenance["cost_unknown"] is True
    with pytest.raises(UnknownSpendError):
        list(matcher.forward(iter(_pairs().to_candidates())))
    assert resource.calls == 1


def test_llm_matcher_adapter_meters_paid_call_before_parse_error() -> None:
    class _MalformedCostedLLM(FakeLLM):
        requires_cost_accounting = True
        calls = 0

        def generate(self, requests):
            self.calls += 1
            batch = super().generate(requests)
            return batch.model_copy(
                update={
                    "outputs": tuple(
                        output.model_copy(update={"cost_usd": 0.6, "cost_basis": "real"})
                        for output in batch.outputs
                    )
                }
            )

    resource = _MalformedCostedLLM(default_response="malformed")
    matcher = SpendCappedMatcher(
        LLMMatcherAdapter[CompanySchema](resource, on_parse_error="raise"),
        budget_usd=1.0,
    )
    one_candidate = iter(_pairs().to_candidates()[:1])

    with pytest.raises(ValueError, match="Could not parse"):
        list(matcher.forward(one_candidate))
    assert matcher.monitor.spent == pytest.approx(0.6)
    assert resource.calls == 1

    with pytest.raises(BudgetExceeded):
        list(matcher.forward(iter(_pairs().to_candidates()[:1])))
    assert matcher.monitor.spent == pytest.approx(1.2)
    assert resource.calls == 2

    with pytest.raises(BudgetExceeded):
        list(matcher.forward(iter(_pairs().to_candidates()[:1])))
    assert resource.calls == 2


def test_generate_binds_one_spend_monitor_and_tallies_envelope_costs() -> None:
    class _CostedLLM(FakeLLM):
        def generate(self, requests):
            batch = super().generate(requests)
            return batch.model_copy(
                update={
                    "outputs": tuple(
                        output.model_copy(update={"cost_usd": 0.3, "cost_basis": "real"})
                        for output in batch.outputs
                    )
                }
            )

    monitor = SpendMonitor(budget_usd=1.0)
    operation = Generate[CompanySchema](_CostedLLM()).bind_spend_monitor(monitor)

    operation.forward(_pairs())

    assert operation.spend_monitor is monitor
    assert monitor.spent == pytest.approx(0.6)
    with pytest.raises(ValueError, match="different SpendMonitor"):
        operation.bind_spend_monitor(SpendMonitor(budget_usd=1.0))


def test_generate_checks_bound_budget_before_another_paid_call() -> None:
    resource = FakeLLM()
    monitor = SpendMonitor(budget_usd=0.0)
    monitor.add(0.01)
    operation = Generate[CompanySchema](resource).bind_spend_monitor(monitor)

    with pytest.raises(BudgetExceeded):
        operation.forward(_pairs())


def test_generate_bound_budget_limits_overshoot_to_one_paid_call() -> None:
    class _CostedLLM(FakeLLM):
        calls = 0

        def generate(self, requests):
            self.calls += 1
            batch = super().generate(requests)
            return batch.model_copy(
                update={
                    "outputs": tuple(
                        output.model_copy(update={"cost_usd": 0.6, "cost_basis": "real"})
                        for output in batch.outputs
                    )
                }
            )

    resource = _CostedLLM()
    monitor = SpendMonitor(budget_usd=0.5)
    operation = Generate[CompanySchema](resource).bind_spend_monitor(monitor)

    with pytest.raises(BudgetExceeded):
        operation.forward(_pairs())

    assert resource.calls == 1
    assert monitor.spent == pytest.approx(0.6)


def test_generate_finite_api_budget_stops_after_unknown_cost_and_retains_output() -> None:
    class _UnknownCostAPI(FakeLLM):
        requires_cost_accounting = True
        calls = 0

        def __init__(self) -> None:
            super().__init__(default_response="MATCH")
            self.model_ref = ModelRef(base="openrouter/openai/gpt-4o-mini", kind="api")

        def generate(self, requests):
            self.calls += 1
            return super().generate(requests)

    resource = _UnknownCostAPI()
    operation = Generate[CompanySchema](resource).bind_spend_monitor(SpendMonitor(budget_usd=0.5))

    with pytest.raises(UnknownGenerationCostError) as exc_info:
        operation.forward(_pairs())

    assert resource.calls == 1
    assert len(exc_info.value.outputs) == 1
    assert exc_info.value.outputs[0].content == "MATCH"
    assert exc_info.value.outputs[0].cost_usd is None
    assert operation.spend_monitor is not None
    assert operation.spend_monitor.spent == 0.0
    assert operation.spend_monitor.cost_is_unknown is True
    with pytest.raises(UnknownSpendError):
        operation.forward(_pairs())
    assert resource.calls == 1


def test_generate_unknown_cost_error_retains_prior_paid_outputs() -> None:
    class _PartiallyMeteredAPI(FakeLLM):
        requires_cost_accounting = True
        calls = 0

        def generate(self, requests):
            self.calls += 1
            batch = super().generate(requests)
            cost = 0.1 if self.calls == 1 else None
            return batch.model_copy(
                update={
                    "outputs": tuple(
                        output.model_copy(
                            update={
                                "cost_usd": cost,
                                "cost_basis": "real" if cost is not None else "none",
                            }
                        )
                        for output in batch.outputs
                    )
                }
            )

    resource = _PartiallyMeteredAPI(default_response="MATCH")
    monitor = SpendMonitor(budget_usd=1.0)
    operation = Generate[CompanySchema](resource).bind_spend_monitor(monitor)

    with pytest.raises(UnknownGenerationCostError) as exc_info:
        operation.forward(_pairs())

    assert resource.calls == 2
    assert [output.request_id for output in exc_info.value.outputs] == [
        '["a","b"]',
        '["a","c"]',
    ]
    assert exc_info.value.outputs[0].cost_usd == pytest.approx(0.1)
    assert exc_info.value.outputs[1].cost_usd is None
    assert monitor.spent == pytest.approx(0.1)
    assert monitor.cost_is_unknown is True


def test_generate_unknown_api_cost_is_nonfatal_when_unbound_or_uncapped() -> None:
    class _UnknownCostAPI(FakeLLM):
        def __init__(self) -> None:
            super().__init__(default_response="MATCH")
            self.model_ref = ModelRef(base="served-model", kind="endpoint", api_base="local")

    resource = _UnknownCostAPI()
    assert len(Generate[CompanySchema](resource).forward(_pairs()).rows) == 2

    uncapped = Generate[CompanySchema](resource).bind_spend_monitor(
        SpendMonitor(budget_usd=float("inf"))
    )
    assert len(uncapped.forward(_pairs()).rows) == 2
    assert uncapped.spend_monitor is not None
    assert uncapped.spend_monitor.spent == 0.0


def test_parse_marks_unknown_paid_cost_without_numeric_none_metadata() -> None:
    class _UnknownPaidLLM(FakeLLM):
        requires_cost_accounting = True

    parsed = Parse[CompanySchema]().forward(
        Generate[CompanySchema](_UnknownPaidLLM(default_response="MATCH")).forward(_pairs())
    )

    assert all("cost_usd" not in row.provenance for row in parsed.rows)
    assert all(row.provenance["cost_unknown"] is True for row in parsed.rows)


def test_generate_matches_topology_spend_binding_contract_when_available() -> None:
    core_op = importlib.import_module("langres.core.op")
    bindable = getattr(core_op, "SpendMonitorBindable", None)
    if bindable is None:
        pytest.skip("cross-branch gate: topology SpendMonitorBindable is not merged yet")

    operation = Generate[CompanySchema](FakeLLM())
    monitor = SpendMonitor(budget_usd=1.0)

    assert isinstance(operation, Spending)
    assert isinstance(operation, bindable)
    assert operation.bind_spend_monitor(monitor) is operation
    assert operation.spend_monitor is monitor


def test_generate_rejects_zero_fallback_for_litellm_hf_ref_paid_capability() -> None:
    class _Client:
        calls = 0

        def completion(self, **kwargs):
            self.calls += 1
            return type(
                "_Response",
                (),
                {
                    "choices": [
                        type(
                            "_Choice",
                            (),
                            {"message": type("_Message", (), {"content": "MATCH"})()},
                        )()
                    ],
                    "usage": None,
                },
            )()

        def completion_cost(self, completion_response):
            return 0.0

    client = _Client()
    resource = LiteLLM(
        ModelRef(base="openrouter/unpriced/model", kind="hf"),
        client=client,
    )
    operation = Generate[CompanySchema](resource).bind_spend_monitor(SpendMonitor(budget_usd=1.0))

    with pytest.raises(UnknownGenerationCostError):
        operation.forward(_pairs())

    assert client.calls == 1
