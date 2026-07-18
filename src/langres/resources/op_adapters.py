"""Resource adapters over the existing ``Pairs -> Pairs`` Op algebra."""

from __future__ import annotations

import json
import math
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
from langres.core.spend import SpendMonitor, attach_spend_observation
from langres.resources.base import (
    GenerationBatch,
    GenerationEnvelope,
    GenerationRequest,
    LLM,
    RerankRequest,
    Reranker,
    UnknownGenerationCostError,
    require_unique_ids,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)
PairSerializer = Callable[[BaseModel], str]
RequestBuilder = Callable[[PairRow[Any]], GenerationRequest]
GenerationParser = Callable[[str], "ParsedGeneration"]

GENERATION_ENVELOPE_KEY = "_langres_generation"
GENERATION_COST_REQUIRED_KEY = "_langres_generation_cost_required"
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


def _default_generation_request(row: PairRow[Any]) -> GenerationRequest:
    prompt = (
        "Determine whether these records refer to the same entity.\n"
        f"Record A: {_serialize_record(row.left)}\n"
        f"Record B: {_serialize_record(row.right)}\n"
        "Answer MATCH or NO_MATCH."
    )
    return GenerationRequest.user(_pair_id(row), prompt)


def _requires_cost_accounting(resource: LLM) -> bool:
    """Resolve paid-transport risk from an explicit resource capability first."""
    declared = getattr(resource, "requires_cost_accounting", None)
    if declared is not None:
        return bool(declared)
    return resource.model_ref.kind in {"api", "endpoint"}


PAIR_SERIALIZERS: dict[str, PairSerializer] = {"json": _serialize_record}
REQUEST_BUILDERS: dict[str, RequestBuilder] = {"binary_pair": _default_generation_request}
GENERATION_PARSERS: dict[str, GenerationParser] = {
    "binary": parse_binary_response,
    "score": parse_score_response,
}


def _registered_callable_name(
    value: Callable[..., Any],
    registry: dict[str, Callable[..., Any]],
    *,
    role: str,
) -> str:
    for name, registered in registry.items():
        if value is registered:
            return name
    raise TypeError(
        f"{role} uses a custom callable and cannot be serialized safely. "
        f"Use one of the registered names: {', '.join(sorted(registry))}."
    )


def _resolve_callable(
    config: dict[str, object],
    key: str,
    registry: dict[str, Callable[..., Any]],
    *,
    role: str,
) -> Callable[..., Any]:
    name = config.get(key)
    if not isinstance(name, str) or name not in registry:
        raise ValueError(
            f"{role} config requires {key!r} to be one of "
            f"{', '.join(sorted(registry))}; got {name!r}."
        )
    return registry[name]


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

    @property
    def config(self) -> dict[str, object]:
        """Safe topology params; arbitrary serializers fail closed."""
        return {
            "out_space": self.out_space,
            "serializer": _registered_callable_name(
                self.serializer,
                PAIR_SERIALIZERS,
                role="Rerank",
            ),
        }

    @classmethod
    def from_config(
        cls,
        resource: Reranker,
        config: dict[str, object],
    ) -> "Rerank[SchemaT]":
        """Rebuild from trusted registered params plus a resource component."""
        serializer = _resolve_callable(
            config,
            "serializer",
            PAIR_SERIALIZERS,
            role="Rerank",
        )
        out_space = config.get("out_space")
        if not isinstance(out_space, str):
            raise ValueError("Rerank config requires a string 'out_space'")
        return cls(
            resource,
            out_space=out_space,  # type: ignore[arg-type]  # Score validates the family
            serializer=serializer,
        )

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
        require_unique_ids(
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
        self._spend_monitor: SpendMonitor | None = None

    @property
    def config(self) -> dict[str, object]:
        """Safe topology params; arbitrary prompt builders fail closed."""
        return {
            "request_builder": _registered_callable_name(
                self.request_builder,
                REQUEST_BUILDERS,
                role="Generate",
            )
        }

    @classmethod
    def from_config(
        cls,
        resource: LLM,
        config: dict[str, object],
    ) -> "Generate[SchemaT]":
        """Rebuild from trusted registered params plus an LLM resource."""
        builder = _resolve_callable(
            config,
            "request_builder",
            REQUEST_BUILDERS,
            role="Generate",
        )
        return cls(resource, request_builder=builder)

    @property
    def spend_monitor(self) -> SpendMonitor | None:
        """The topology-owned spend ledger, once bound."""
        return self._spend_monitor

    def bind_spend_monitor(self, monitor: SpendMonitor) -> "Generate[SchemaT]":
        """Bind generation spend to exactly one cumulative ledger."""
        if self._spend_monitor is not None and self._spend_monitor is not monitor:
            raise ValueError(
                "Generate is already bound to a different SpendMonitor. "
                "Build a separate Generate operation for a separate model budget."
            )
        self._spend_monitor = monitor
        return self

    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Generate once per pair and keep scores unchanged for the Parse stage."""
        requests = [self.request_builder(row) for row in pairs.rows]
        require_unique_ids(
            [request.request_id for request in requests],
            field="request_ids",
            operation="Generate",
        )
        outputs = (
            list(self._validated_outputs(requests, self.resource.generate(requests)))
            if self._spend_monitor is None
            else self._generate_one_at_a_time(requests)
        )
        rows = []
        for row, envelope in zip(pairs.rows, outputs, strict=True):
            provenance = dict(row.provenance)
            provenance[GENERATION_ENVELOPE_KEY] = envelope
            provenance[GENERATION_COST_REQUIRED_KEY] = _requires_cost_accounting(self.resource)
            rows.append(row.model_copy(update={"provenance": provenance}))
        return Pairs(store=pairs.store, rows=rows)

    def _generate_one_at_a_time(
        self,
        requests: list[GenerationRequest],
    ) -> list[GenerationEnvelope]:
        """Meter each provider request, limiting overshoot to one paid call."""
        assert self._spend_monitor is not None
        outputs: list[GenerationEnvelope] = []
        for request in requests:
            self._spend_monitor.check()
            batch = self.resource.generate([request])
            validated = self._validated_outputs([request], batch)
            unknown_cost = any(output.cost_usd is None for output in validated)
            finite_budget = math.isfinite(self._spend_monitor.budget_usd)
            paid_transport = _requires_cost_accounting(self.resource)
            if unknown_cost and finite_budget and paid_transport:
                self._spend_monitor.mark_unknown(
                    "Generation succeeded, but provider cost is unknown. The "
                    "finite spend ledger is permanently blocked."
                )
                raise UnknownGenerationCostError(
                    "Generation succeeded, but provider cost is unknown. A finite "
                    "budget cannot safely continue because unknown spend must not "
                    "be counted as $0. The successful output is available on "
                    "exception.outputs.",
                    outputs=(*outputs, *validated),
                )
            measured_costs = [
                output.cost_usd for output in validated if output.cost_usd is not None
            ]
            self._spend_monitor.add(sum(measured_costs))
            self._spend_monitor.check()
            outputs.extend(validated)
        return outputs

    def _validated_outputs(
        self,
        expected: list[GenerationRequest],
        batch: GenerationBatch,
    ) -> tuple[GenerationEnvelope, ...]:
        """Validate model, cardinality, identity, and order for one resource call."""
        if batch.model_ref != self.resource.model_ref:
            raise ValueError(
                "LLM returned a batch with a different model_ref from the resource. "
                "Resource identity must remain stable for provenance and cache keys."
            )
        expected_ids = tuple(request.request_id for request in expected)
        output_ids = tuple(output.request_id for output in batch.outputs)
        if output_ids != expected_ids:
            raise ValueError(
                "LLM changed request identity/order/cardinality. An LLM resource must "
                "return one envelope per request in the same order."
            )
        return batch.outputs


class Parse(Score[SchemaT], Generic[SchemaT]):
    """Parse private generation envelopes into decisions/scores/abstentions."""

    def __init__(
        self,
        parser: GenerationParser = parse_binary_response,
        *,
        on_parse_error: Literal["abstain", "raise"] = "abstain",
    ) -> None:
        if on_parse_error not in {"abstain", "raise"}:
            raise ValueError("on_parse_error must be 'abstain' or 'raise'")
        super().__init__(scope="pair", out_space="prob_llm")
        self.parser = parser
        self.on_parse_error = on_parse_error

    @property
    def config(self) -> dict[str, object]:
        """Safe topology params; arbitrary response parsers fail closed."""
        return {
            "parser": _registered_callable_name(
                self.parser,
                GENERATION_PARSERS,
                role="Parse",
            ),
            "on_parse_error": self.on_parse_error,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "Parse[SchemaT]":
        """Rebuild only a named built-in parser."""
        parser = _resolve_callable(
            config,
            "parser",
            GENERATION_PARSERS,
            role="Parse",
        )
        on_parse_error = config.get("on_parse_error")
        if on_parse_error not in {"abstain", "raise"}:
            raise ValueError("Parse config requires on_parse_error='abstain' or 'raise'")
        return cls(
            parser,
            on_parse_error=on_parse_error,
        )

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
            cost_required = row.provenance.get(GENERATION_COST_REQUIRED_KEY, False) is True
            try:
                parsed = self.parser(envelope.content)
            except Exception as exc:
                if self.on_parse_error == "raise":
                    attach_spend_observation(
                        exc,
                        cost_usd=envelope.cost_usd,
                        cost_required=cost_required,
                    )
                    raise
                parsed = ParsedGeneration(reasoning="generation parser raised")
            parse_error = parsed.decision is None and parsed.score is None
            if parse_error and self.on_parse_error == "raise":
                parse_error_exc = ValueError(
                    f"Could not parse generation for pair {row.left_id!r}, {row.right_id!r}"
                )
                attach_spend_observation(
                    parse_error_exc,
                    cost_usd=envelope.cost_usd,
                    cost_required=cost_required,
                )
                raise parse_error_exc

            provenance = dict(row.provenance)
            provenance.pop(GENERATION_ENVELOPE_KEY, None)
            provenance.pop(GENERATION_COST_REQUIRED_KEY, None)
            provenance[GENERATION_SUMMARY_KEY] = envelope.model_dump(mode="json")
            provenance["cost_required"] = cost_required
            if envelope.cost_usd is None:
                if cost_required:
                    provenance["cost_unknown"] = True
            else:
                provenance["cost_usd"] = envelope.cost_usd
            provenance["parse_error"] = parse_error
            if envelope.usage is not None:
                provenance["usage"] = envelope.usage.model_dump(mode="json")
                if envelope.usage.input_tokens is not None:
                    provenance["prompt_tokens"] = envelope.usage.input_tokens
                if envelope.usage.output_tokens is not None:
                    provenance["completion_tokens"] = envelope.usage.output_tokens
            if envelope.cost_usd is not None:
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
        parser: GenerationParser = parse_binary_response,
        request_builder: RequestBuilder = _default_generation_request,
        on_parse_error: Literal["abstain", "raise"] = "abstain",
    ) -> None:
        self.resource = resource
        self._generate = Generate[SchemaT](resource, request_builder=request_builder)
        self._parse = Parse[SchemaT](parser, on_parse_error=on_parse_error)
        self.model_ref = resource.model_ref
        self.model = resource.model_ref.base

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Generate, parse, and yield each candidate before pulling the next one."""
        for candidate in candidates:
            pairs = Pairs.from_candidates([candidate])
            parsed = self._parse.forward(self._generate.forward(pairs))
            yield parsed.rows[0].to_judgement()
