"""Resource adapters over the existing ``Pairs -> Pairs`` Op algebra."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

from langres.core.matcher import Matcher
from langres.core.model_ref import to_config
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.op import Op, Score, Spending
from langres.core.pairs import PairRow, Pairs
from langres.core.score_type import ScoreType
from langres.resources.base import (
    GenerationEnvelope,
    GenerationRequest,
    LLM,
    RerankRequest,
    Reranker,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)
PairSerializer = Callable[[BaseModel], str]
RequestBuilder = Callable[[PairRow[Any]], GenerationRequest]
GenerationParser = Callable[[str], "ParsedGeneration"]

GENERATION_ENVELOPE_KEY = "_langres_generation"
GENERATION_SUMMARY_KEY = "generation"

_SCORE_RE = re.compile(r"(?im)^\s*score\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*$")
_REASONING_RE = re.compile(r"(?im)^\s*reasoning\s*:\s*(.+?)\s*$")


class ParsedGeneration(BaseModel):
    """Typed parser output: score, decision, or explicit abstention."""

    model_config = ConfigDict(frozen=True)

    decision: bool | None = None
    score: float | None = None
    reasoning: str | None = None

    @model_validator(mode="after")
    def _one_verdict_shape(self) -> "ParsedGeneration":
        if self.decision is not None and self.score is not None:
            raise ValueError("ParsedGeneration cannot carry both decision and score")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("ParsedGeneration.score must be in [0, 1]")
        return self


def parse_binary_response(content: str) -> ParsedGeneration:
    """Parse an exact binary response; malformed content becomes abstention."""
    token = content.strip().splitlines()[0].strip().upper() if content.strip() else ""
    if token in {"MATCH", "YES"}:
        return ParsedGeneration(decision=True)
    if token in {"NO_MATCH", "NO MATCH", "NO"}:
        return ParsedGeneration(decision=False)
    return ParsedGeneration(reasoning="unparseable binary response")


def parse_score_response(content: str) -> ParsedGeneration:
    """Parse the legacy ``Score: 0..1`` response shape."""
    score_match = _SCORE_RE.search(content)
    reasoning_match = _REASONING_RE.search(content)
    reasoning = reasoning_match.group(1) if reasoning_match is not None else None
    if score_match is None:
        return ParsedGeneration(reasoning=reasoning or "unparseable score response")
    return ParsedGeneration(score=float(score_match.group(1)), reasoning=reasoning)


def _serialize_record(record: BaseModel) -> str:
    return json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _pair_id(row: PairRow[Any]) -> str:
    return json.dumps([row.left_id, row.right_id], separators=(",", ":"))


def _require_unique(values: list[str], *, field: str, operation: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        preview = ", ".join(repr(value) for value in sorted(duplicates)[:3])
        raise ValueError(
            f"{operation} requires unique {field}; duplicate ids: {preview}. "
            "Deduplicate pair rows or provide a request builder with stable unique ids."
        )


def _default_generation_request(row: PairRow[Any]) -> GenerationRequest:
    prompt = (
        "Determine whether these records refer to the same entity.\n"
        f"Record A: {_serialize_record(row.left)}\n"
        f"Record B: {_serialize_record(row.right)}\n"
        "Answer MATCH or NO_MATCH."
    )
    return GenerationRequest.user(_pair_id(row), prompt)


class Rerank(Score[SchemaT], Generic[SchemaT]):
    """Adapt one reranker resource to a scalar ``Score``.

    The operation only rescales rows. It does not know whether the following
    selection is top-k retrieval or a final match threshold; that role belongs
    solely to the following ``Select``.
    """

    def __init__(
        self,
        resource: Reranker,
        *,
        out_space: ScoreType = "heuristic",
        serializer: PairSerializer = _serialize_record,
    ) -> None:
        super().__init__(scope="pair", out_space=out_space)
        self.resource = resource
        self.serializer = serializer

    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Rerank all rows once, preserving row identity and order."""
        requests = [
            RerankRequest(
                pair_id=_pair_id(row),
                left=self.serializer(row.left),
                right=self.serializer(row.right),
            )
            for row in pairs.rows
        ]
        _require_unique(
            [request.pair_id for request in requests],
            field="pair_ids",
            operation="Rerank",
        )
        batch = self.resource.rerank(requests)
        if batch.model_ref != self.resource.model_ref:
            raise ValueError(
                "Reranker returned a batch with a different model_ref from the "
                "resource. Resource identity must remain stable for provenance "
                "and cache keys."
            )
        expected_ids = tuple(request.pair_id for request in requests)
        if batch.pair_ids != expected_ids:
            raise ValueError(
                "Reranker changed request identity/order. A Rerank resource must "
                "return one score per input pair in the same order."
            )
        rows = []
        for row, score in zip(pairs.rows, batch.scores, strict=True):
            provenance = dict(row.provenance)
            provenance["reranker"] = {
                "model_ref": to_config(batch.model_ref),
                "runtime": self.resource.runtime_config.model_dump(mode="json"),
            }
            rows.append(
                row.model_copy(
                    update={
                        "score": score,
                        "score_type": self.out_space,
                        "decision": None,
                        "decision_step": "rerank",
                        "provenance": provenance,
                    }
                )
            )
        return Pairs(store=pairs.store, rows=rows)


class Generate(Op[SchemaT], Spending, Generic[SchemaT]):
    """Invoke an LLM and attach private typed envelopes to pair provenance."""

    def __init__(
        self,
        resource: LLM,
        *,
        request_builder: RequestBuilder = _default_generation_request,
    ) -> None:
        self.resource = resource
        self.request_builder = request_builder

    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Generate once per pair and keep scores unchanged for the Parse stage."""
        requests = [self.request_builder(row) for row in pairs.rows]
        _require_unique(
            [request.request_id for request in requests],
            field="request_ids",
            operation="Generate",
        )
        batch = self.resource.generate(requests)
        if batch.model_ref != self.resource.model_ref:
            raise ValueError(
                "LLM returned a batch with a different model_ref from the resource. "
                "Resource identity must remain stable for provenance and cache keys."
            )
        expected_ids = tuple(request.request_id for request in requests)
        output_ids = tuple(output.request_id for output in batch.outputs)
        if output_ids != expected_ids:
            raise ValueError(
                "LLM changed request identity/order. An LLM resource must return "
                "one envelope per request in the same order."
            )
        rows = []
        for row, envelope in zip(pairs.rows, batch.outputs, strict=True):
            provenance = dict(row.provenance)
            provenance[GENERATION_ENVELOPE_KEY] = envelope
            rows.append(row.model_copy(update={"provenance": provenance}))
        return Pairs(store=pairs.store, rows=rows)


class Parse(Score[SchemaT], Generic[SchemaT]):
    """Parse private generation envelopes into decisions/scores/abstentions."""

    def __init__(
        self,
        parser: GenerationParser = parse_score_response,
        *,
        on_parse_error: Literal["abstain", "raise"] = "abstain",
    ) -> None:
        if on_parse_error not in {"abstain", "raise"}:
            raise ValueError("on_parse_error must be 'abstain' or 'raise'")
        super().__init__(scope="pair", out_space="prob_llm")
        self.parser = parser
        self.on_parse_error = on_parse_error

    def _envelope(self, row: PairRow[SchemaT]) -> GenerationEnvelope:
        value = row.provenance.get(GENERATION_ENVELOPE_KEY)
        if isinstance(value, GenerationEnvelope):
            return value
        if isinstance(value, dict) and "raw_content" in value:
            return GenerationEnvelope.from_local_payload(value)
        raise ValueError(
            "Parse requires a local GenerationEnvelope from the immediately "
            "preceding Generate operation."
        )

    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Parse each local envelope and remove its raw form from provenance."""
        rows = []
        for row in pairs.rows:
            envelope = self._envelope(row)
            try:
                parsed = self.parser(envelope.content)
            except Exception:
                if self.on_parse_error == "raise":
                    raise
                parsed = ParsedGeneration(reasoning="generation parser raised")
            parse_error = parsed.decision is None and parsed.score is None
            if parse_error and self.on_parse_error == "raise":
                raise ValueError(
                    f"Could not parse generation for pair {row.left_id!r}, {row.right_id!r}"
                )

            provenance = dict(row.provenance)
            provenance.pop(GENERATION_ENVELOPE_KEY, None)
            provenance[GENERATION_SUMMARY_KEY] = envelope.model_dump(mode="json")
            provenance["parse_error"] = parse_error
            if envelope.usage is not None:
                provenance["usage"] = envelope.usage.model_dump(mode="json")
                if envelope.usage.input_tokens is not None:
                    provenance["prompt_tokens"] = envelope.usage.input_tokens
                if envelope.usage.output_tokens is not None:
                    provenance["completion_tokens"] = envelope.usage.output_tokens
            if envelope.cost_usd is not None:
                provenance["cost_usd"] = envelope.cost_usd
                provenance["cost_is_real"] = envelope.cost_basis == "real"
            rows.append(
                row.model_copy(
                    update={
                        "decision": parsed.decision,
                        "score": parsed.score,
                        "score_type": "prob_llm",
                        "decision_step": "llm_parse",
                        "reasoning": parsed.reasoning,
                        "provenance": provenance,
                    }
                )
            )
        return Pairs(store=pairs.store, rows=rows)


class LLMMatcherAdapter(Matcher[SchemaT], Generic[SchemaT]):
    """Expose ``LLM + Generate + Parse`` through the legacy Matcher contract."""

    def __init__(
        self,
        resource: LLM,
        *,
        parser: GenerationParser = parse_score_response,
        request_builder: RequestBuilder = _default_generation_request,
        on_parse_error: Literal["abstain", "raise"] = "abstain",
    ) -> None:
        self.resource = resource
        self._generate = Generate[SchemaT](resource, request_builder=request_builder)
        self._parse = Parse[SchemaT](parser, on_parse_error=on_parse_error)
        self.model_ref = resource.model_ref
        self.model = resource.model_ref.base

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Materialize candidates through resource Ops and yield judgements."""
        pairs = Pairs.from_candidates(list(candidates))
        parsed = self._parse.forward(self._generate.forward(pairs))
        for row in parsed.rows:
            yield row.to_judgement()
