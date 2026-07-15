"""Tests for LLMMatcher's paper-replication seams + the token-usage vector.

Task 2 makes ``LLMMatcher`` able to run a published paper's prompt without a
subclass fork: an injectable ``response_parser`` (default Score-regex; a shipped
binary yes/no parser), an injectable ``record_serializer``, a ``system_prompt``
seam, explicit parse-failure semantics (``on_parse_error`` + a ``parse_error``
provenance flag), and safe ``{left}``/``{right}`` substitution that tolerates
literal braces (a paper's JSON schema). Task 1 adds the full ``usage`` vector to
provenance.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from langres.core.models import CompanySchema, ERCandidate
from langres.core.matchers.llm_judge import (
    LLMMatcher,
    LLMParseError,
    ParsedVerdict,
    default_record_serializer,
    parse_binary_yes_no,
    parse_score_response,
)


def _pair() -> ERCandidate[CompanySchema]:
    return ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )


def _response(content: str, *, prompt_tokens: int = 100, completion_tokens: int = 50) -> Mock:
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = content
    resp.usage = Mock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.prompt_tokens_details = None
    resp.usage.completion_tokens_details = None
    return resp


def _judge(client: Mock, **kwargs: object) -> LLMMatcher[CompanySchema]:
    return LLMMatcher(client=client, model="gpt-4o-mini", **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Shipped response parsers
# ---------------------------------------------------------------------------


class TestScoreParser:
    def test_parses_score_and_reasoning(self) -> None:
        v = parse_score_response("MATCH\nScore: 0.9\nReasoning: same company")
        assert v.score == 0.9
        assert v.reasoning == "same company"

    def test_clamps_out_of_range_score(self) -> None:
        assert parse_score_response("Score: 1.7").score == 1.0

    def test_reasoning_falls_back_to_full_content(self) -> None:
        v = parse_score_response("Score: 0.8\nThese are similar companies.")
        assert v.score == 0.8
        assert v.reasoning is not None and "similar companies" in v.reasoning

    def test_no_score_line_signals_failure_with_none(self) -> None:
        """The default parser now ABSTAINS (score=None) instead of silently 0.5."""
        v = parse_score_response("These entities might be the same, I'm not sure.")
        assert v.score is None


class TestBinaryYesNoParser:
    def test_yes_maps_to_true_decision(self) -> None:
        v = parse_binary_yes_no("Yes, they are the same.")
        assert v.decision is True
        assert v.score is None  # a decider does not rank -> no fabricated score

    def test_no_maps_to_false_decision(self) -> None:
        v = parse_binary_yes_no("No.")
        assert v.decision is False
        assert v.score is None

    def test_strips_punctuation_and_is_case_insensitive(self) -> None:
        assert parse_binary_yes_no("YES!").decision is True

    def test_absence_of_yes_is_false(self) -> None:
        """The published yes/no family is total: no 'yes' => False (not a match)."""
        assert parse_binary_yes_no("These are different products.").decision is False

    @pytest.mark.parametrize(
        "answer",
        # Intra-word punctuation the paper's check_for_prediction DELETES (not
        # replaces-with-space), so each collapses to "yes" -> MATCH. These are
        # exactly the cases where the old two parsers disagreed. `_` matters:
        # it is `\w` (kept by a naive [^\w\s] scrub) but IS in
        # string.punctuation (deleted by the paper).
        ["ye-s", "ye.s", "y-e-s", "Ye's", "ye_s", "Y.E.S."],
    )
    def test_intra_word_punctuation_deleted_like_check_for_prediction(self, answer: str) -> None:
        """Punctuation is DELETED (paper semantics), so "ye-s" -> "yes" -> MATCH.

        Pins the unified contract against the old ``re.sub(r"[^\\w\\s]", " ", ...)``
        which replaced punctuation with a space and split these into non-matches.
        """
        assert parse_binary_yes_no(answer).decision is True

    def test_crude_substring_match_is_deliberate(self) -> None:
        """Fidelity over cleverness: "Not yes" contains "yes" so it MATCHES.

        This crudeness is inherited from the reference ``check_for_prediction``
        (substring test, no negation handling). We keep it because reproducing
        the paper's reported F1 requires the paper's exact parser.
        """
        assert parse_binary_yes_no("Not yes").decision is True

    def test_is_total_never_returns_none(self) -> None:
        """No 'yes' is a confident non-match (decision=False), never abstain (None)."""
        v = parse_binary_yes_no("Absolutely not, different entities entirely.")
        assert v.decision is False
        assert v.decision is not None  # totality is about the DECISION now
        assert v.score is None

    def test_totality_means_raise_mode_never_fires(self) -> None:
        """Because this parser is total, on_parse_error='raise' can't abort a run."""
        client = Mock()
        client.completion.return_value = _response("No, these are different.")
        judge = _judge(client, response_parser=parse_binary_yes_no, on_parse_error="raise")
        j = list(judge.forward([_pair()]))[0]  # does NOT raise LLMParseError
        assert j.decision is False
        assert j.score is None


class TestDefaultRecordSerializer:
    def test_is_indented_json(self) -> None:
        out = default_record_serializer(CompanySchema(id="c1", name="Acme"))
        assert '"name": "Acme"' in out
        assert '"id": "c1"' in out


# ---------------------------------------------------------------------------
# response_parser wiring
# ---------------------------------------------------------------------------


class TestResponseParserWiring:
    def test_binary_parser_yields_match_decision(self) -> None:
        client = Mock()
        client.completion.return_value = _response("Yes")
        judge = _judge(client, response_parser=parse_binary_yes_no)
        j = list(judge.forward([_pair()]))[0]
        # A binary judge decides; it does not fabricate a 0/1 score.
        assert j.decision is True
        assert j.score is None
        # confidence not requested, but a decision judge COULD expose logprobs.
        assert j.confidence is None
        assert j.confidence_source == "unrequested"
        assert "parse_error" not in j.provenance

    def test_binary_parser_yields_no_match_decision(self) -> None:
        client = Mock()
        client.completion.return_value = _response("No, these are different.")
        judge = _judge(client, response_parser=parse_binary_yes_no)
        j = list(judge.forward([_pair()]))[0]
        assert j.decision is False
        assert j.score is None

    def test_default_parser_still_reads_score(self) -> None:
        client = Mock()
        client.completion.return_value = _response("MATCH\nScore: 0.42\nReasoning: x")
        j = list(_judge(client).forward([_pair()]))[0]
        # The rating family ranks (score) and does not decide; it has no
        # confidence notion, so confidence_source is "none".
        assert j.score == 0.42
        assert j.decision is None
        assert j.confidence_source == "none"


# ---------------------------------------------------------------------------
# Parse-failure semantics
# ---------------------------------------------------------------------------


class TestParseFailureSemantics:
    def test_default_abstains_with_flag_not_silent_half(self) -> None:
        """Default ``on_parse_error='abstain'`` => all-None abstain, flagged and distinguishable."""
        client = Mock()
        client.completion.return_value = _response("I cannot tell.")  # no Score:
        j = list(_judge(client).forward([_pair()]))[0]
        # An abstain is neither a decision nor a score -> is_abstain, not a fake 0.0.
        assert j.is_abstain
        assert j.decision is None
        assert j.score is None
        assert j.confidence is None
        assert j.provenance["parse_error"] is True
        assert j.decision_step == "llm_judgment"

    def test_abstain_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        client = Mock()
        client.completion.return_value = _response("no score here")
        with caplog.at_level(logging.WARNING):
            list(_judge(client).forward([_pair()]))
        assert any("parse" in r.message.lower() for r in caplog.records)

    def test_successful_parse_has_no_parse_error_flag(self) -> None:
        client = Mock()
        client.completion.return_value = _response("Score: 0.7")
        j = list(_judge(client).forward([_pair()]))[0]
        assert "parse_error" not in j.provenance

    def test_raise_mode_raises_typed_error(self) -> None:
        client = Mock()
        client.completion.return_value = _response("garbage, no score")
        judge = _judge(client, on_parse_error="raise")
        with pytest.raises(LLMParseError):
            list(judge.forward([_pair()]))

    def test_invalid_on_parse_error_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="on_parse_error"):
            _judge(Mock(), on_parse_error="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# system_prompt seam
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_no_system_prompt_sends_single_user_message(self) -> None:
        client = Mock()
        client.completion.return_value = _response("Score: 0.9")
        list(_judge(client).forward([_pair()]))
        messages = client.completion.call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_system_prompt_sends_system_then_user(self) -> None:
        client = Mock()
        client.completion.return_value = _response("Score: 0.9")
        judge = _judge(client, system_prompt="You are a strict matcher.")
        list(judge.forward([_pair()]))
        messages = client.completion.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a strict matcher."}
        assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# record_serializer seam
# ---------------------------------------------------------------------------


class TestRecordSerializer:
    def test_custom_serializer_controls_prompt_rendering(self) -> None:
        client = Mock()
        client.completion.return_value = _response("Score: 0.9")
        judge = _judge(
            client,
            record_serializer=lambda e: f"NAME={e.name}",  # type: ignore[attr-defined]
            prompt_template="A: {left}\nB: {right}",
        )
        list(judge.forward([_pair()]))
        prompt = client.completion.call_args.kwargs["messages"][0]["content"]
        assert "NAME=Acme Corporation" in prompt
        assert "NAME=Acme Corp" in prompt
        # The default serializer would have leaked the id/source; this one didn't.
        assert '"id"' not in prompt

    def test_default_serializer_is_json_dump(self) -> None:
        client = Mock()
        client.completion.return_value = _response("Score: 0.9")
        list(_judge(client).forward([_pair()]))
        prompt = client.completion.call_args.kwargs["messages"][0]["content"]
        assert '"name": "Acme Corporation"' in prompt


# ---------------------------------------------------------------------------
# Safe {left}/{right} substitution (literal braces in a paper's prompt)
# ---------------------------------------------------------------------------


class TestSafeTemplateSubstitution:
    def test_literal_braces_do_not_raise(self) -> None:
        """A paper prompt with a JSON output schema (literal { }) must not blow up."""
        client = Mock()
        client.completion.return_value = _response("Score: 0.9")
        template = 'Compare {left} and {right}. Reply as {{"match": true}}.'
        judge = _judge(client, prompt_template=template)
        list(judge.forward([_pair()]))  # would KeyError under str.format
        prompt = client.completion.call_args.kwargs["messages"][0]["content"]
        assert '{"match": true}' in prompt  # braces preserved verbatim

    def test_both_records_are_substituted(self) -> None:
        client = Mock()
        client.completion.return_value = _response("Score: 0.9")
        judge = _judge(
            client,
            record_serializer=lambda e: e.name,  # type: ignore[attr-defined]
            prompt_template="L={left} R={right}",
        )
        list(judge.forward([_pair()]))
        prompt = client.completion.call_args.kwargs["messages"][0]["content"]
        assert prompt == "L=Acme Corporation R=Acme Corp"

    def test_missing_placeholders_raise_at_construction(self) -> None:
        with pytest.raises(ValueError, match=r"\{left\}.*\{right\}|placeholder"):
            _judge(Mock(), prompt_template="No placeholders here")

    def test_placeholder_literal_inside_a_record_is_not_substituted(self) -> None:
        """A record whose text contains ``{right}`` must survive verbatim.

        Chained ``str.replace`` would rescan the inserted left record and
        overwrite its ``{right}`` token with the right record — silent,
        data-dependent prompt corruption.
        """
        client = Mock()
        client.completion.return_value = _response("Score: 0.9")
        judge = _judge(
            client,
            record_serializer=lambda e: "PRE {right} POST" if e.id == "c1" else "RIGHT",  # type: ignore[attr-defined]
            prompt_template="L={left} R={right}",
        )
        list(judge.forward([_pair()]))
        prompt = client.completion.call_args.kwargs["messages"][0]["content"]
        assert prompt == "L=PRE {right} POST R=RIGHT"


# ---------------------------------------------------------------------------
# temperature default (ER papers use 0)
# ---------------------------------------------------------------------------


def test_temperature_defaults_to_zero() -> None:
    judge = LLMMatcher(client=Mock(), model="gpt-4o-mini")
    assert judge.temperature == 0.0


# ---------------------------------------------------------------------------
# Task 1: usage vector in provenance
# ---------------------------------------------------------------------------


class TestUsageVectorInProvenance:
    def test_provenance_carries_usage_vector(self) -> None:
        client = Mock()
        client.completion.return_value = _response(
            "Score: 0.9", prompt_tokens=120, completion_tokens=40
        )
        j = list(_judge(client).forward([_pair()]))[0]
        usage = j.provenance["usage"]
        assert usage["input_tokens"] == 120
        assert usage["output_tokens"] == 40
        assert usage["model"] == "gpt-4o-mini"
        assert usage["cache_read_input_tokens"] == 0
        assert usage["reasoning_tokens"] == 0

    def test_legacy_token_keys_still_present_and_equal_totals(self) -> None:
        """Purely additive: prompt_tokens/completion_tokens still there (readers depend)."""
        client = Mock()
        client.completion.return_value = _response(
            "Score: 0.9", prompt_tokens=120, completion_tokens=40
        )
        j = list(_judge(client).forward([_pair()]))[0]
        assert j.provenance["prompt_tokens"] == 120
        assert j.provenance["completion_tokens"] == 40
        assert j.provenance["prompt_tokens"] == j.provenance["usage"]["input_tokens"]
        assert j.provenance["completion_tokens"] == j.provenance["usage"]["output_tokens"]

    def test_usage_records_serving_provider(self) -> None:
        client = Mock()
        resp = _response("Score: 0.9")
        resp._hidden_params = {
            "additional_headers": {"llm_provider-x-litellm-response-cost": 0.0004}
        }
        resp.provider = "DeepInfra"
        client.completion.return_value = resp
        judge = LLMMatcher(client=client, model="openrouter/z-ai/glm-5.2")
        j = list(judge.forward([_pair()]))[0]
        assert j.provenance["usage"]["provider"] == "DeepInfra"

    def test_usage_is_zero_vector_when_usage_missing(self) -> None:
        client = Mock()
        resp = _response("Score: 0.9")
        resp.usage = None
        client.completion.return_value = resp
        j = list(_judge(client).forward([_pair()]))[0]
        assert j.provenance["usage"]["input_tokens"] == 0
        assert j.provenance["usage"]["output_tokens"] == 0


class TestParsedVerdictModel:
    def test_holds_score_and_reasoning(self) -> None:
        v = ParsedVerdict(score=0.5, reasoning="because")
        assert v.score == 0.5
        assert v.decision is None  # rating family: score XOR decision
        assert v.reasoning == "because"

    def test_holds_decision_for_the_binary_family(self) -> None:
        v = ParsedVerdict(decision=True, reasoning="Yes")
        assert v.decision is True
        assert v.score is None  # decider does not rank

    def test_both_none_means_parse_failure_abstain(self) -> None:
        v = ParsedVerdict()
        assert v.score is None
        assert v.decision is None
        assert v.reasoning is None
