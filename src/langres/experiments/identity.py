"""Canonical experiment identities without a second run persistence system."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from langres.core.model_ref import BackboneKind, ModelRef
from langres.experiments.protocol import EvaluationProtocol, FrozenDict, freeze_mapping
from langres.tracking.runs import RunContext, RunRecord, compute_recipe_id

CacheSemantics = Literal["deterministic", "seeded", "stochastic"]
SourceClaim = Literal["clean", "dirty"]
_IDENTITY_LENGTH = 16
_EXPERIMENT_RECIPE_FIELDS = (
    "experiment",
    "resolver_config",
    "llm_model",
    "cascade_band",
    "blocking_k",
    "method",
    "dataset_name",
    "dataset_fingerprint",
    "split_id",
    "seeds",
)
_SECRET_KEY = re.compile(
    r"(?:^|_)(?:api_key|access_token|auth|authorization|bearer|credential|"
    r"password|private_key|secret|signature|sig|token)(?:$|_)",
    re.IGNORECASE,
)
_SAFE_ENDPOINT_QUERY_KEYS = frozenset(
    {
        "api_version",
        "deployment",
        "location",
        "project",
        "region",
        "version",
    }
)
_EXECUTION_POLICY_KEYS = frozenset(
    {
        "budget_usd",
        "budget_soft_usd",
        "concurrency",
        "max_retries",
        "paid_proof",
        "publication_profile",
        "retry_policy",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (set, frozenset)):
        raise TypeError(
            "sets are not valid JSON identity values; use a deterministically ordered list"
        )
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def _content_id(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:_IDENTITY_LENGTH]


def _without_execution_policy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_execution_policy(item)
            for key, item in value.items()
            if str(key) not in _EXECUTION_POLICY_KEYS
        }
    if isinstance(value, (list, tuple)):
        return tuple(_without_execution_policy(item) for item in value)
    return value


class RecipeIdentity(BaseModel):
    """Logical architecture/data/seed lineage backed by ``RunContext``."""

    model_config = ConfigDict(frozen=True)
    recipe_id: str
    legacy_recipe_id: str


class EvaluationIdentity(BaseModel):
    """One statistically comparable question."""

    model_config = ConfigDict(frozen=True)
    evaluation_id: str


class SourceState(BaseModel):
    """Code/lock provenance used by byte-reuse identity."""

    model_config = ConfigDict(frozen=True)

    git_sha: str | None = Field(default=None, min_length=1)
    git_dirty: bool = False
    dirty_tree_hash: str | None = Field(default=None, min_length=1)
    lockfile_hash: str | None = Field(default=None, min_length=1)
    environment_hash: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_dirty_claim(self) -> "SourceState":
        if self.git_dirty and not self.dirty_tree_hash:
            raise ValueError("dirty source requires dirty_tree_hash")
        if not self.git_dirty and self.dirty_tree_hash is not None:
            raise ValueError("clean source must not provide dirty_tree_hash")
        return self

    @property
    def claim(self) -> SourceClaim:
        return "dirty" if self.git_dirty else "clean"


def _normalized_key(value: object) -> str:
    """Normalize config/header/query spellings before classifying secrets."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _is_secret_key(value: object) -> bool:
    return _SECRET_KEY.search(_normalized_key(value)) is not None


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): ("<redacted>" if _is_secret_key(key) else _redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return tuple(_redact_secrets(item) for item in value)
    return value


def _safe_endpoint(value: str | None) -> str | None:
    """Retain routing identity and safe query config while dropping endpoint secrets."""
    if value is None:
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("endpoint must be an absolute URL with scheme and host")
    hostname = parsed.hostname.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    safe_query = urlencode(
        sorted(
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if _normalized_key(key) in _SAFE_ENDPOINT_QUERY_KEYS
        )
    )
    return urlunsplit((parsed.scheme.lower(), f"{hostname}{port}", parsed.path, safe_query, ""))


class ResourceSlotIdentity(BaseModel):
    """Complete, secret-safe identity for one model-bearing architecture slot."""

    model_config = ConfigDict(frozen=True)

    slot: str = Field(min_length=1)
    base: str = Field(min_length=1)
    kind: BackboneKind
    revision: str | None = Field(default=None, min_length=1)
    adapter: str | None = Field(default=None, min_length=1)
    adapter_revision: str | None = Field(default=None, min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    endpoint: str | None = Field(default=None, min_length=1)
    content_digest: str | None = Field(default=None, min_length=1)
    runtime_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("endpoint", mode="before")
    @classmethod
    def _sanitize_endpoint(cls, value: Any) -> str | None:
        if value is None or isinstance(value, str):
            return _safe_endpoint(value)
        raise ValueError("endpoint must be a string or None")

    @field_validator("runtime_config", mode="after")
    @classmethod
    def _freeze_runtime_config(cls, value: dict[str, Any]) -> FrozenDict:
        return freeze_mapping(_redact_secrets(value))

    @model_validator(mode="after")
    def _validate_resource(self) -> "ResourceSlotIdentity":
        if self.kind == "endpoint" and self.endpoint is None:
            raise ValueError("kind='endpoint' requires endpoint")
        if self.kind != "endpoint" and self.endpoint is not None:
            raise ValueError("endpoint is only valid for kind='endpoint'")
        if self.adapter_revision is not None and self.adapter is None:
            raise ValueError("adapter_revision requires adapter")
        return self

    @classmethod
    def from_model_ref(
        cls,
        slot: str,
        ref: ModelRef,
        *,
        provider: str | None = None,
        content_digest: str | None = None,
        runtime_config: Mapping[str, Any] | None = None,
    ) -> "ResourceSlotIdentity":
        """Project the established ``ModelRef`` into cache identity."""
        inferred_provider = provider
        if inferred_provider is None and ref.kind in {"api", "endpoint"} and "/" in ref.base:
            inferred_provider = ref.base.split("/", 1)[0]
        return cls(
            slot=slot,
            base=ref.base,
            kind=ref.kind,
            revision=ref.revision,
            adapter=ref.adapter,
            provider=inferred_provider,
            endpoint=ref.api_base,
            content_digest=content_digest,
            runtime_config=dict(runtime_config or {}),
        )


class CacheIdentityInput(BaseModel):
    """Every fact that can change one immutable stage output."""

    model_config = ConfigDict(frozen=True)

    stage_id: str = Field(min_length=1)
    execution_plan_id: str = Field(min_length=1)
    operation_identity: dict[str, Any] = Field(min_length=1)
    resource_slots: tuple[ResourceSlotIdentity, ...]
    source: SourceState
    semantics: CacheSemantics
    input_fingerprint: str = Field(min_length=1)
    prompt_revision: str | None = None
    parser_revision: str | None = None
    seed: int | None = None
    repeat_index: int | None = Field(default=None, ge=0)
    attempt_id: str | None = None
    official: bool = False

    @field_validator("operation_identity", mode="after")
    @classmethod
    def _freeze_operation_identity(cls, value: dict[str, Any]) -> FrozenDict:
        return freeze_mapping(_redact_secrets(value))

    @model_validator(mode="after")
    def _validate_semantics(self) -> "CacheIdentityInput":
        slots = [resource.slot for resource in self.resource_slots]
        if len(slots) != len(set(slots)):
            raise ValueError("resource_slots must use unique slot names")
        if self.official:
            if self.source.git_dirty or self.source.git_sha is None:
                raise ValueError("official cache publication requires a clean source commit")
            if self.source.lockfile_hash is None:
                raise ValueError("official cache publication requires lockfile_hash")
            if self.source.environment_hash is None:
                raise ValueError("official cache publication requires environment_hash")
            for resource in self.resource_slots:
                if resource.kind == "hf" and resource.revision is None:
                    raise ValueError(
                        f"official cache resource {resource.slot!r} requires a pinned revision"
                    )
                if resource.kind == "local" and resource.content_digest is None:
                    raise ValueError(
                        f"official local cache resource {resource.slot!r} requires content_digest"
                    )
                if resource.adapter is not None and resource.adapter_revision is None:
                    raise ValueError(
                        f"official cache resource {resource.slot!r} requires a pinned "
                        "adapter_revision"
                    )
                if resource.kind in {"api", "endpoint"} and resource.provider is None:
                    raise ValueError(
                        f"official served resource {resource.slot!r} requires provider identity"
                    )
        if self.semantics == "deterministic":
            if (
                self.seed is not None
                or self.repeat_index is not None
                or self.attempt_id is not None
            ):
                raise ValueError(
                    "deterministic cache identity must not contain seed, repeat_index, or attempt_id"
                )
        elif self.semantics == "seeded":
            if self.seed is None:
                raise ValueError("seeded cache identity requires seed")
            if self.repeat_index is not None or self.attempt_id is not None:
                raise ValueError(
                    "seeded cache identity must not contain repeat_index or attempt_id"
                )
        elif self.repeat_index is None or not self.attempt_id:
            raise ValueError("stochastic cache identity requires repeat_index and attempt_id")
        return self


class CacheIdentity(BaseModel):
    """Content id plus the reuse claim its semantics permits."""

    model_config = ConfigDict(frozen=True)

    cache_id: str
    semantics: CacheSemantics
    source_claim: SourceClaim
    official: bool
    reusable: bool
    counts_as_independent_repeat: bool


class AttemptIdentity(BaseModel):
    """Identity projection over the existing ``RunRecord``."""

    model_config = ConfigDict(frozen=True)

    attempt_id: str
    recipe_id: str
    evaluation_id: str | None = None
    cache_id: str | None = None

    @classmethod
    def from_record(cls, record: RunRecord) -> "AttemptIdentity":
        return cls(
            attempt_id=record.attempt_id,
            recipe_id=record.recipe_id,
            evaluation_id=record.evaluation_id,
            cache_id=record.cache_id,
        )


def compute_recipe_identity(context: RunContext) -> RecipeIdentity:
    """Hash logical experiment recipe fields, excluding spend/execution policy.

    ``legacy_recipe_id`` preserves the established tracking hash (which includes
    ``budget_usd``) for existing stores and callers. New experiment capture paths
    should persist :attr:`RecipeIdentity.recipe_id`.
    """
    payload = {
        field: _without_execution_policy(getattr(context, field))
        for field in _EXPERIMENT_RECIPE_FIELDS
    }
    return RecipeIdentity(
        recipe_id=_content_id(payload),
        legacy_recipe_id=compute_recipe_id(context),
    )


def compute_evaluation_identity(protocol: EvaluationProtocol) -> EvaluationIdentity:
    """Hash only fields that define the statistical comparison cohort."""
    return EvaluationIdentity(evaluation_id=_content_id(protocol.evaluation_payload()))


def compute_cache_identity(inputs: CacheIdentityInput) -> CacheIdentity:
    """Content-address a stage output with clean/dirty and stochastic semantics."""
    payload = inputs.model_dump(mode="json", exclude={"official"})
    payload["resource_slots"] = sorted(payload["resource_slots"], key=lambda item: item["slot"])
    return CacheIdentity(
        cache_id=_content_id(payload),
        semantics=inputs.semantics,
        source_claim=inputs.source.claim,
        official=inputs.official and not inputs.source.git_dirty,
        reusable=True,
        counts_as_independent_repeat=inputs.semantics != "stochastic",
    )
