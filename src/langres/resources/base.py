"""Import-light resource contracts and batch carriers.

Resources own model identity and runtime configuration. They do not own their
position in an entity-resolution topology: an embedder, reranker, or LLM can be
reused by multiple operations without changing the resource itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from langres.core.model_ref import ModelRef

RuntimeDType = Literal["float16", "float32", "bfloat16"]


def require_unique_ids(
    values: Sequence[str],
    *,
    field: str,
    operation: str,
) -> None:
    """Reject ambiguous request identities before a resource can perform work."""
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
            "Deduplicate inputs or provide stable unique ids."
        )


def _read(value: Any, key: str) -> Any:
    """Read a field from a mapping or response object without importing an SDK."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _optional_int(value: Any) -> int | None:
    """Preserve an unknown token count as ``None`` and a measured zero as zero."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_known(*values: Any) -> int | None:
    """Return the first present, integer-like usage field."""
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return None


class ResourceRuntimeConfig(BaseModel):
    """Weightless runtime settings shared by model-backed resources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(default=32, ge=1)
    device: str | None = None
    dtype: RuntimeDType | None = None
    local_files_only: bool = False


class SentenceTransformerRuntimeConfig(ResourceRuntimeConfig):
    """Runtime settings for dense Sentence Transformers."""

    backend: Literal["torch", "onnx", "openvino"] = "torch"
    normalize_embeddings: bool = True
    show_progress_bar: bool = False


class RerankerRuntimeConfig(ResourceRuntimeConfig):
    """Runtime settings for a CrossEncoder reranker."""

    backend: Literal["torch", "onnx", "openvino"] = "torch"
    max_length: int | None = Field(default=None, ge=1)
    show_progress_bar: bool = False


class LLMRuntimeConfig(ResourceRuntimeConfig):
    """Runtime settings shared by served and in-process generation."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=64, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    seed: int | None = Field(default=None, ge=0)


class EmbeddingFacts(BaseModel):
    """Facts measured directly from one embedding batch."""

    model_config = ConfigDict(frozen=True)

    dimension: int = Field(ge=1)
    dtype: str
    normalized: bool | None = None


class EmbeddingBatch(BaseModel):
    """Dense vectors produced for one ordered text batch."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    vectors: np.ndarray
    model_ref: ModelRef
    facts: EmbeddingFacts | None = None

    @field_validator("vectors", mode="before")
    @classmethod
    def _matrix(cls, value: Any) -> np.ndarray:
        vectors = np.asarray(value)
        if vectors.ndim != 2:
            raise ValueError("EmbeddingBatch.vectors must be a two-dimensional matrix")
        return vectors

    @model_validator(mode="after")
    def _facts_match_vectors(self) -> "EmbeddingBatch":
        if self.facts is not None and self.vectors.shape[1] != self.facts.dimension:
            raise ValueError("EmbeddingBatch facts.dimension must equal vectors.shape[1]")
        return self


class RerankRequest(BaseModel):
    """One ordered text pair sent to a reranker."""

    model_config = ConfigDict(frozen=True)

    pair_id: str = Field(min_length=1)
    left: str
    right: str


Score = Annotated[float, Field(ge=0.0, le=1.0)]


class RerankBatch(BaseModel):
    """One bounded score per ordered rerank request."""

    model_config = ConfigDict(frozen=True)

    pair_ids: tuple[str, ...]
    scores: tuple[Score, ...]
    model_ref: ModelRef

    @model_validator(mode="after")
    def _one_score_per_pair(self) -> "RerankBatch":
        if len(self.pair_ids) != len(self.scores):
            raise ValueError("RerankBatch requires exactly one score per pair_id")
        if len(set(self.pair_ids)) != len(self.pair_ids):
            raise ValueError(
                "RerankBatch.pair_ids must be unique; duplicate identities make "
                "score-to-pair mapping ambiguous"
            )
        return self


class ChatMessage(BaseModel):
    """One role/content message in a generation request."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class GenerationRequest(BaseModel):
    """One ordered request sent to an LLM resource."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)

    @classmethod
    def user(cls, request_id: str, content: str) -> "GenerationRequest":
        """Construct a single-user-message request."""
        return cls(
            request_id=request_id,
            messages=(ChatMessage(role="user", content=content),),
        )

    def message_dicts(self) -> list[dict[str, str]]:
        """Return the backend-neutral role/content dictionaries."""
        return [message.model_dump() for message in self.messages]


class GenerationUsage(BaseModel):
    """Optional token facts from one generation.

    Providers do not expose every token subset. Missing values remain ``None``
    so experiment data can distinguish an unknown measurement from a measured
    zero. Input/output totals are the already-normalized inclusive totals from
    LiteLLM; cache and reasoning counts are subsets and must not be added again.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    provider: str | None = None
    model: str = ""

    @classmethod
    def from_response(
        cls,
        response: Any,
        *,
        model: str,
        provider: str | None = None,
    ) -> "GenerationUsage":
        """Extract standard, cache, and reasoning usage without SDK imports.

        LiteLLM exposes cache creation in two provider-dependent shapes:
        ``prompt_tokens_details.cache_creation_tokens`` and the Anthropic-style
        top-level ``cache_creation_input_tokens``. Cache reads likewise use
        either nested ``cached_tokens`` or top-level
        ``cache_read_input_tokens``.
        """
        usage = _read(response, "usage")
        prompt_details = _read(usage, "prompt_tokens_details")
        completion_details = _read(usage, "completion_tokens_details")
        return cls(
            input_tokens=_first_known(
                _read(usage, "prompt_tokens"),
                _read(usage, "input_tokens"),
            ),
            output_tokens=_first_known(
                _read(usage, "completion_tokens"),
                _read(usage, "output_tokens"),
            ),
            cache_read_input_tokens=_first_known(
                _read(prompt_details, "cached_tokens"),
                _read(usage, "cache_read_input_tokens"),
            ),
            cache_creation_input_tokens=_first_known(
                _read(prompt_details, "cache_creation_tokens"),
                _read(usage, "cache_creation_input_tokens"),
            ),
            reasoning_tokens=_first_known(_read(completion_details, "reasoning_tokens")),
            provider=provider,
            model=model,
        )


class GenerationEnvelope(BaseModel):
    """Versioned generation metadata with raw content kept private by default.

    Normal ``model_dump``/``model_dump_json`` output contains usage and serving
    identity but never generated content. Local replay code must explicitly call
    :meth:`local_payload`; this makes publication of raw prompts/records/content
    an opt-in action rather than an accidental side effect of serialization.
    """

    model_config = ConfigDict(frozen=True)

    version: Literal["1"] = "1"
    request_id: str = Field(min_length=1)
    model_ref: ModelRef
    usage: GenerationUsage | None = None
    provider: str | None = None
    served_model: str | None = None
    provider_request_id: str | None = None
    finish_reason: str | None = None
    cost_usd: float | None = Field(default=None, ge=0.0)
    cost_basis: Literal["real", "estimated", "none"] = "none"

    _raw_content: str = PrivateAttr(default="")

    @classmethod
    def from_content(
        cls,
        *,
        request_id: str,
        model_ref: ModelRef,
        content: str,
        usage: GenerationUsage | None = None,
        provider: str | None = None,
        served_model: str | None = None,
        provider_request_id: str | None = None,
        finish_reason: str | None = None,
        cost_usd: float | None = None,
        cost_basis: Literal["real", "estimated", "none"] = "none",
    ) -> "GenerationEnvelope":
        """Build an envelope whose generated content stays process-local."""
        envelope = cls(
            request_id=request_id,
            model_ref=model_ref,
            usage=usage,
            provider=provider,
            served_model=served_model,
            provider_request_id=provider_request_id,
            finish_reason=finish_reason,
            cost_usd=cost_usd,
            cost_basis=cost_basis,
        )
        object.__setattr__(envelope, "_raw_content", content)
        return envelope

    @property
    def content(self) -> str:
        """Raw generated content, available only on the local envelope object."""
        return self._raw_content

    def local_payload(self) -> dict[str, Any]:
        """Serialize for a declared local replay cache, including raw content."""
        payload = self.model_dump(mode="json")
        payload["raw_content"] = self.content
        return payload

    @classmethod
    def from_local_payload(cls, payload: dict[str, Any]) -> "GenerationEnvelope":
        """Restore an envelope from :meth:`local_payload`."""
        values = dict(payload)
        raw_content = values.pop("raw_content")
        if not isinstance(raw_content, str):
            raise TypeError("GenerationEnvelope raw_content must be a string")
        envelope = cls.model_validate(values)
        object.__setattr__(envelope, "_raw_content", raw_content)
        return envelope


class GenerationBatch(BaseModel):
    """Ordered generation envelopes from one LLM invocation batch."""

    model_config = ConfigDict(frozen=True)

    outputs: tuple[GenerationEnvelope, ...]
    model_ref: ModelRef

    @model_validator(mode="after")
    def _same_model(self) -> "GenerationBatch":
        if any(output.model_ref != self.model_ref for output in self.outputs):
            raise ValueError("GenerationBatch outputs must share the batch model_ref")
        request_ids = tuple(output.request_id for output in self.outputs)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError(
                "GenerationBatch request_ids must be unique; duplicate identities "
                "make output-to-request mapping ambiguous"
            )
        return self


@runtime_checkable
class Embedder(Protocol):
    """A resource that maps ordered text to dense vectors."""

    @property
    def model_ref(self) -> ModelRef:
        """Stable weightless identity for this model slot."""
        ...  # pragma: no cover

    @property
    def runtime_config(self) -> ResourceRuntimeConfig:
        """Weightless runtime settings for this resource."""
        ...  # pragma: no cover

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed ``texts`` without changing their order."""
        ...  # pragma: no cover


@runtime_checkable
class Reranker(Protocol):
    """A resource that assigns one bounded score to each text pair."""

    @property
    def model_ref(self) -> ModelRef:
        """Stable weightless identity for this model slot."""
        ...  # pragma: no cover

    @property
    def runtime_config(self) -> ResourceRuntimeConfig:
        """Weightless runtime settings for this resource."""
        ...  # pragma: no cover

    def rerank(self, pairs: Sequence[RerankRequest]) -> RerankBatch:
        """Score ``pairs`` without selecting any of them."""
        ...  # pragma: no cover


@runtime_checkable
class LLM(Protocol):
    """A resource that generates responses plus token/serving facts."""

    @property
    def model_ref(self) -> ModelRef:
        """Stable weightless identity for this model slot."""
        ...  # pragma: no cover

    @property
    def runtime_config(self) -> ResourceRuntimeConfig:
        """Weightless runtime settings for this resource."""
        ...  # pragma: no cover

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        """Generate one envelope per request, preserving request order."""
        ...  # pragma: no cover
