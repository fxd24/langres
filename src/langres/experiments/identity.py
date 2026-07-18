"""Canonical experiment identities without a second run persistence system."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from langres.experiments.protocol import EvaluationProtocol
from langres.tracking.runs import RunContext, RunRecord, compute_recipe_id

CacheSemantics = Literal["deterministic", "seeded", "stochastic"]
SourceClaim = Literal["clean", "dirty"]
_IDENTITY_LENGTH = 16


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def _content_id(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:_IDENTITY_LENGTH]


class RecipeIdentity(BaseModel):
    """Logical architecture/data/seed lineage backed by ``RunContext``."""

    model_config = ConfigDict(frozen=True)
    recipe_id: str


class EvaluationIdentity(BaseModel):
    """One statistically comparable question."""

    model_config = ConfigDict(frozen=True)
    evaluation_id: str


class SourceState(BaseModel):
    """Code/lock provenance used by byte-reuse identity."""

    model_config = ConfigDict(frozen=True)

    git_sha: str | None = None
    git_dirty: bool = False
    dirty_tree_hash: str | None = None
    lockfile_hash: str | None = None
    environment_hash: str | None = None

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


class CacheIdentityInput(BaseModel):
    """Every fact that can change one immutable stage output."""

    model_config = ConfigDict(frozen=True)

    stage_id: str = Field(min_length=1)
    source: SourceState
    semantics: CacheSemantics
    input_fingerprint: str = Field(min_length=1)
    resource_revisions: dict[str, str] = Field(default_factory=dict)
    prompt_revision: str | None = None
    parser_revision: str | None = None
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    repeat_index: int | None = Field(default=None, ge=0)
    attempt_id: str | None = None
    official: bool = False

    @model_validator(mode="after")
    def _validate_semantics(self) -> "CacheIdentityInput":
        if self.official and (self.source.git_dirty or self.source.git_sha is None):
            raise ValueError("official cache publication requires a clean source commit")
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
    """Wrap the established tracking identity instead of forking its rules."""
    return RecipeIdentity(recipe_id=compute_recipe_id(context))


def compute_evaluation_identity(protocol: EvaluationProtocol) -> EvaluationIdentity:
    """Hash only fields that define the statistical comparison cohort."""
    return EvaluationIdentity(evaluation_id=_content_id(protocol.evaluation_payload()))


def compute_cache_identity(inputs: CacheIdentityInput) -> CacheIdentity:
    """Content-address a stage output with clean/dirty and stochastic semantics."""
    payload = inputs.model_dump(mode="json", exclude={"official"})
    return CacheIdentity(
        cache_id=_content_id(payload),
        semantics=inputs.semantics,
        source_claim=inputs.source.claim,
        official=inputs.official and not inputs.source.git_dirty,
        reusable=inputs.semantics != "stochastic",
        counts_as_independent_repeat=inputs.semantics != "stochastic",
    )
