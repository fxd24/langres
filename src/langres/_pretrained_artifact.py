"""Pure contracts and validation helpers for shareable langres artifacts.

The existing ``resolver.json`` remains the source of truth for model topology.
This module adds a strict outer bundle envelope without changing those frozen
local-persistence bytes.  It intentionally imports no Hub client.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from langres._version import __version__ as LANGRES_VERSION
from langres.core.model_ref import IN_PROCESS_KINDS, ModelRef, normalize_model_ref, to_config
from langres.core.serialization import ArtifactManifest, ArtifactSource, ComponentSpec

BUNDLE_VERSION = "1"
BUNDLE_MANIFEST = "langres-artifact.json"
MODEL_CARD = "README.md"
MEASUREMENT_SUMMARY = "measurement-summary.json"
RESOLVER_MANIFEST = "resolver.json"

MAX_MANIFEST_BYTES = 1_000_000
MAX_RESOLVER_BYTES = 10_000_000
MAX_SUMMARY_BYTES = 1_000_000
MAX_CARD_BYTES = 100_000
MAX_FILES = 256
MAX_FILE_BYTES = 2_000_000_000
MAX_TOTAL_BYTES = 10_000_000_000
MAX_CARD_TEXT = 20_000
MAX_SUMMARY_ITEMS = 256

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ClaimLevel = Literal["reference-only", "frozen-weights", "benchmark-reproducible"]
JsonFact = str | int | float | bool | None

_FIXED_ROOTS = frozenset(
    {
        RESOLVER_MANIFEST,
        BUNDLE_MANIFEST,
        MODEL_CARD,
        MEASUREMENT_SUMMARY,
    }
)
_RESOURCE_COMPONENTS: dict[str, tuple[str, str]] = {
    "resource_sentence_transformer": ("model", "semantic"),
    "resource_cross_encoder_reranker": ("model", "semantic"),
    "resource_litellm": ("model", "llm"),
    "resource_transformers_llm": ("model", "semantic"),
    "sentence_transformer_embedder": ("model_ref", "semantic"),
    "fastembed_sparse_embedder": ("model_name", "semantic"),
    "fastembed_late_interaction_embedder": ("model_name", "semantic"),
    "llm_judge": ("model", "llm"),
    "dspy_judge": ("model", "llm"),
    "select_judge": ("model", "llm"),
}
_COMPONENT_EXTRAS: dict[str, str] = {
    "calibrator": "trained",
    "faiss_index": "semantic",
    "random_forest": "trained",
    "vector_blocker": "semantic",
}
_HF_MODEL_NAME_COMPONENTS = frozenset(
    {
        "fastembed_sparse_embedder",
        "fastembed_late_interaction_embedder",
    }
)
_INPROCESS_LLM_COMPONENTS = frozenset({"llm_judge", "resource_transformers_llm"})
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_CREDENTIAL_KEY = re.compile(
    r"(?:^|_)(?:api_key|access_token|auth|authorization|bearer|credential|"
    r"credentials|authentication|cookie|openai_key|password|private_key|secret|"
    r"set_cookie|signature|sig|subscription_key|token)(?:$|_)",
    re.IGNORECASE,
)


class PretrainedArtifactError(ValueError):
    """A bundle is unsafe, malformed, incompatible, or incomplete."""


class ArtifactEligibilityError(PretrainedArtifactError):
    """The requested sharing claim is not supported by the supplied facts."""


def safe_relative_path(value: str) -> str:
    """Validate and normalize one manifest-owned POSIX relative path."""
    if "\\" in value or "\x00" in value:
        raise ValueError("artifact paths must be normalized POSIX paths")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe artifact path {value!r}")
    root = path.parts[0]
    if root not in _FIXED_ROOTS:
        raise ValueError(f"artifact path {value!r} is outside the state-free pretrained allowlist")
    if (
        root
        in {
            RESOLVER_MANIFEST,
            BUNDLE_MANIFEST,
            MODEL_CARD,
            MEASUREMENT_SUMMARY,
        }
        and len(path.parts) != 1
    ):
        raise ValueError(f"artifact root file {root!r} cannot contain child paths")
    return path.as_posix()


class ArtifactFile(BaseModel):
    """One exact regular file in the validated bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: str
    size: int = Field(ge=0, le=MAX_FILE_BYTES)
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        return safe_relative_path(value)


class ModelCardSpec(BaseModel):
    """Human-authored facts required for a useful, honest model card."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    intended_use: str = "Research and evaluation of entity-resolution architectures with langres."
    limitations: tuple[str, ...] = (
        "Quality and latency depend on the dataset, thresholds, hardware, and serving setup.",
    )

    @field_validator("intended_use")
    @classmethod
    def _bounded_use(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 2_000:
            raise ValueError("intended_use must contain 1..2000 characters")
        return value

    @field_validator("limitations")
    @classmethod
    def _bounded_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 32:
            raise ValueError("limitations must contain 1..32 entries")
        normalized = tuple(item.strip() for item in value)
        if any(not item or len(item) > 1_000 for item in normalized):
            raise ValueError("each limitation must contain 1..1000 characters")
        return normalized


class MeasurementSummary(BaseModel):
    """Bounded publication facts; never raw records, prompts, or generations."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    protocol_id: str | None = None
    evaluation_id: str | None = None
    dataset_ids: tuple[str, ...] = ()
    quality: dict[str, JsonFact] = Field(default_factory=dict)
    cost: dict[str, JsonFact] = Field(default_factory=dict)
    tokens: dict[str, JsonFact] = Field(default_factory=dict)
    performance: dict[str, JsonFact] = Field(default_factory=dict)
    hardware: dict[str, JsonFact] = Field(default_factory=dict)
    size: dict[str, JsonFact] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _bounded(self) -> "MeasurementSummary":
        mappings = (
            self.quality,
            self.cost,
            self.tokens,
            self.performance,
            self.hardware,
            self.size,
        )
        if sum(len(mapping) for mapping in mappings) > MAX_SUMMARY_ITEMS:
            raise ValueError(f"measurement summary exceeds {MAX_SUMMARY_ITEMS} facts")
        if len(self.dataset_ids) > 64:
            raise ValueError("measurement summary exceeds 64 dataset ids")
        for mapping in mappings:
            for key, value in mapping.items():
                if not key or len(key) > 200:
                    raise ValueError("measurement fact keys must contain 1..200 characters")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("measurement facts must be finite")
                if isinstance(value, str) and len(value) > 2_000:
                    raise ValueError("measurement string facts are limited to 2000 characters")
        return self


def _is_immutable_hf_ref(ref: ModelRef) -> bool:
    if (
        ref.kind != "hf"
        or _HF_REPO_ID.fullmatch(ref.base) is None
        or ref.revision is None
        or re.fullmatch(r"[0-9a-f]{40}", ref.revision) is None
    ):
        return False
    if ref.adapter is None:
        return True
    return (
        _HF_REPO_ID.fullmatch(ref.adapter) is not None
        and ref.adapter_revision is not None
        and re.fullmatch(r"[0-9a-f]{40}", ref.adapter_revision) is not None
    )


def _validate_benchmark_claim(
    model_refs: Iterable[ModelRef],
    summary: MeasurementSummary | None,
) -> None:
    refs = tuple(model_refs)
    if summary is None:
        raise ArtifactEligibilityError("benchmark-reproducible requires a measurement summary")
    if (
        not summary.protocol_id
        or not summary.evaluation_id
        or not summary.dataset_ids
        or not summary.quality
    ):
        raise ArtifactEligibilityError(
            "benchmark-reproducible requires protocol_id, evaluation_id, dataset_ids, "
            "and at least one quality metric"
        )
    if any(ref.kind != "hf" for ref in refs):
        raise ArtifactEligibilityError(
            "benchmark-reproducible supports only immutable Hugging Face resources; "
            "API, endpoint, and local references must use claim_level='reference-only'"
        )
    if any(not _is_immutable_hf_ref(ref) for ref in refs):
        raise ArtifactEligibilityError(
            "benchmark-reproducible requires every base and adapter to be a full "
            "'organization/repository' Hugging Face id pinned to an immutable "
            "40-character commit SHA"
        )


def _version_minor(version: str) -> tuple[int, int]:
    parsed = re.fullmatch(
        r"(?:\d+!)?"
        r"(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?:0|[1-9]\d*)"
        r"(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?"
        r"(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?",
        version,
        flags=re.IGNORECASE,
    )
    if parsed is None:
        raise ValueError(f"unsupported langres version format {version!r}")
    return int(parsed["major"]), int(parsed["minor"])


def _compatibility_for(version: str) -> str:
    major, minor = _version_minor(version)
    return f">={major}.{minor}.0,<{major}.{minor + 1}.0"


class PretrainedManifest(BaseModel):
    """Strict outer envelope for one unchanged Resolver artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    bundle_version: Literal["1"] = "1"
    langres_version: str = LANGRES_VERSION
    langres_compatibility: str
    resolver_path: Literal["resolver.json"] = "resolver.json"
    claim_level: ClaimLevel = "reference-only"
    required_extras: tuple[str, ...] = ()
    model_refs: tuple[ModelRef, ...] = ()
    sensitive_config_included: bool = False
    measurement_summary_path: Literal["measurement-summary.json"] | None = None
    model_card_path: Literal["README.md"] = "README.md"
    files: tuple[ArtifactFile, ...]

    @model_validator(mode="after")
    def _coherent(self) -> "PretrainedManifest":
        expected_compatibility = _compatibility_for(self.langres_version)
        if self.langres_compatibility != expected_compatibility:
            raise ValueError(
                "langres_compatibility must be the conservative minor-version interval "
                f"{expected_compatibility!r}"
            )
        if _version_minor(LANGRES_VERSION) != _version_minor(self.langres_version):
            raise PretrainedArtifactError(
                f"artifact requires langres {self.langres_compatibility}; "
                f"installed version is {LANGRES_VERSION}"
            )
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact manifest contains duplicate file paths")
        required = {self.resolver_path, self.model_card_path}
        if self.measurement_summary_path is not None:
            required.add(self.measurement_summary_path)
        missing = required.difference(paths)
        if missing:
            raise ValueError(f"artifact manifest is missing required files: {sorted(missing)}")
        if len(self.files) > MAX_FILES:
            raise ValueError(f"artifact exceeds {MAX_FILES} files")
        if sum(item.size for item in self.files) > MAX_TOTAL_BYTES:
            raise ValueError(f"artifact exceeds {MAX_TOTAL_BYTES} total bytes")
        if self.claim_level == "frozen-weights":
            raise ArtifactEligibilityError(
                "frozen-weights is not supported yet: local resource paths are not copied "
                "and rebased into the bundle. Use claim_level='reference-only'."
            )
        if self.claim_level == "benchmark-reproducible":
            if self.measurement_summary_path is None:
                raise ArtifactEligibilityError(
                    "benchmark-reproducible requires a measurement summary"
                )
            mutable = [ref for ref in self.model_refs if ref.kind != "hf"]
            if mutable:
                raise ArtifactEligibilityError(
                    "benchmark-reproducible supports only immutable Hugging Face resources; "
                    "API, endpoint, and local references must use claim_level='reference-only'"
                )
            if any(not _is_immutable_hf_ref(ref) for ref in self.model_refs):
                raise ArtifactEligibilityError(
                    "benchmark-reproducible requires every base and adapter to be a full "
                    "'organization/repository' Hugging Face id pinned to an immutable "
                    "40-character commit SHA"
                )
        return self


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_regular_files(root: Path, *, exclude_manifest: bool = True) -> list[ArtifactFile]:
    files: list[ArtifactFile] = []
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root).as_posix()
        if exclude_manifest and relative == BUNDLE_MANIFEST:
            continue
        if candidate.is_symlink():
            raise PretrainedArtifactError(f"artifact contains a symlink: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise PretrainedArtifactError(f"artifact contains a non-regular file: {relative}")
        safe_relative_path(relative)
        stat = candidate.stat()
        if stat.st_size > MAX_FILE_BYTES:
            raise PretrainedArtifactError(
                f"artifact file {relative!r} exceeds {MAX_FILE_BYTES} bytes"
            )
        files.append(ArtifactFile(path=relative, size=stat.st_size, sha256=sha256_file(candidate)))
    if len(files) > MAX_FILES:
        raise PretrainedArtifactError(f"artifact exceeds {MAX_FILES} files")
    if sum(item.size for item in files) > MAX_TOTAL_BYTES:
        raise PretrainedArtifactError(f"artifact exceeds {MAX_TOTAL_BYTES} total bytes")
    return files


def read_manifest(path: Path) -> PretrainedManifest:
    data = path.read_bytes()
    if len(data) > MAX_MANIFEST_BYTES:
        raise PretrainedArtifactError(f"{BUNDLE_MANIFEST} exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        return PretrainedManifest.model_validate_json(data)
    except Exception as exc:
        if isinstance(exc, ArtifactEligibilityError):
            raise
        raise PretrainedArtifactError(f"invalid {BUNDLE_MANIFEST}: {exc}") from None


def validate_bundle(root: Path) -> PretrainedManifest:
    """Validate the full tree and checksums before component reconstruction."""
    if root.is_symlink() or not root.is_dir():
        raise PretrainedArtifactError(f"artifact root must be a regular directory: {root}")
    manifest_path = root / BUNDLE_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PretrainedArtifactError(f"artifact is missing regular {BUNDLE_MANIFEST}")
    manifest = read_manifest(manifest_path)
    actual = {item.path: item for item in _iter_regular_files(root)}
    expected = {item.path: item for item in manifest.files}
    if actual.keys() != expected.keys():
        raise PretrainedArtifactError(
            "artifact file allowlist mismatch: "
            f"missing={sorted(expected.keys() - actual.keys())}, "
            f"unexpected={sorted(actual.keys() - expected.keys())}"
        )
    for relative, expected_file in expected.items():
        actual_file = actual[relative]
        if actual_file.size != expected_file.size or actual_file.sha256 != expected_file.sha256:
            raise PretrainedArtifactError(f"artifact checksum/size mismatch for {relative!r}")
    bounded_metadata = {
        manifest.resolver_path: MAX_RESOLVER_BYTES,
        manifest.model_card_path: MAX_CARD_BYTES,
    }
    if manifest.measurement_summary_path is not None:
        bounded_metadata[manifest.measurement_summary_path] = MAX_SUMMARY_BYTES
    for relative, limit in bounded_metadata.items():
        if expected[relative].size > limit:
            raise PretrainedArtifactError(
                f"artifact metadata file {relative!r} exceeds {limit} bytes"
            )
    resolver_manifest = ArtifactManifest.model_validate_json(
        (root / manifest.resolver_path).read_bytes()
    )
    credential_paths = credential_config_paths(resolver_manifest)
    if credential_paths:
        raise PretrainedArtifactError(
            "resolver.json contains credential-bearing transport configuration that cannot "
            "be loaded from a pretrained artifact. Fields: " + ", ".join(credential_paths[:10])
        )
    actual_refs, actual_extras = resource_facts(resolver_manifest)
    if actual_refs != manifest.model_refs or actual_extras != manifest.required_extras:
        raise PretrainedArtifactError("outer resource identity does not match resolver.json")
    actual_sensitive = bool(sensitive_config_paths(resolver_manifest))
    if actual_sensitive != manifest.sensitive_config_included:
        raise PretrainedArtifactError(
            "outer sensitive-config declaration does not match resolver.json"
        )
    summary: MeasurementSummary | None = None
    if manifest.measurement_summary_path is not None:
        try:
            summary = MeasurementSummary.model_validate_json(
                (root / manifest.measurement_summary_path).read_bytes()
            )
        except Exception as exc:
            raise PretrainedArtifactError(f"invalid measurement summary: {exc}") from None
    if manifest.claim_level == "benchmark-reproducible":
        _validate_benchmark_claim(manifest.model_refs, summary)
    return manifest


def component_specs(resolver_manifest: ArtifactManifest) -> tuple[ComponentSpec, ...]:
    """Return top-level and recursively nested component specs."""
    specs = list(resolver_manifest.components)
    if resolver_manifest.ops is not None:
        specs.extend(spec.component for spec in resolver_manifest.ops if spec.component is not None)

    def nested_specs(value: object) -> Iterable[ComponentSpec]:
        if isinstance(value, Mapping):
            if isinstance(value.get("type_name"), str) and isinstance(value.get("config"), Mapping):
                nested = ComponentSpec.model_validate(value)
                yield nested
                yield from nested_specs(nested.config)
                return
            for nested_value in value.values():
                yield from nested_specs(nested_value)
        elif isinstance(value, (list, tuple)):
            for nested_value in value:
                yield from nested_specs(nested_value)

    expanded: list[ComponentSpec] = []
    for spec in specs:
        expanded.append(spec)
        expanded.extend(nested_specs(spec.config))
    return tuple(expanded)


def sensitive_config_paths(resolver_manifest: ArtifactManifest) -> tuple[str, ...]:
    """Find serialized prompt-bearing fields that require explicit publication consent."""
    found: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                child = f"{path}.{key}" if path else str(key)
                if "prompt" in str(key).lower() and nested is not None and nested != "":
                    found.append(child)
                visit(nested, child)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    for index, spec in enumerate(component_specs(resolver_manifest)):
        visit(spec.config, f"component[{index}].{spec.type_name}")
    for index, op_spec in enumerate(resolver_manifest.ops or ()):
        visit(op_spec.params, f"op[{index}].{op_spec.role}")
    return tuple(sorted(set(found)))


def credential_config_paths(resolver_manifest: ArtifactManifest) -> tuple[str, ...]:
    """Find credentials embedded in serialized transport configuration."""
    found: list[str] = []

    def credential_key(value: object) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
        return _CREDENTIAL_KEY.search(normalized) is not None

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_text = str(key)
                child = f"{path}.{key_text}"
                if credential_key(key_text) and nested is not None and nested != "":
                    found.append(child)
                visit(nested, child)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    def credential_bearing_url(value: str) -> bool:
        try:
            parsed = urlsplit(value)
            if parsed.username is not None or parsed.password is not None:
                return True
            return any(credential_key(key) for key, _ in parse_qsl(parsed.query))
        except ValueError:
            # A malformed authority may still contain serialized userinfo. Fail
            # closed on that recognizable credential shape.
            return "@" in value.partition("://")[2].partition("/")[0]

    def visit_api_bases(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_text = str(key)
                child = f"{path}.{key_text}"
                if (
                    key_text.lower() == "api_base"
                    and isinstance(nested, str)
                    and credential_bearing_url(nested)
                ):
                    found.append(child)
                visit_api_bases(nested, child)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit_api_bases(nested, f"{path}[{index}]")

    for index, spec in enumerate(component_specs(resolver_manifest)):
        component_path = f"component[{index}].{spec.type_name}"
        for field in ("provider", "extra_body"):
            value = spec.config.get(field)
            if value is not None:
                visit(value, f"{component_path}.{field}")
        visit_api_bases(spec.config, component_path)
    return tuple(sorted(set(found)))


def resource_facts(
    resolver_manifest: ArtifactManifest,
) -> tuple[tuple[ModelRef, ...], tuple[str, ...]]:
    """Extract model refs and install extras from every nested component."""
    expanded = component_specs(resolver_manifest)

    refs: list[ModelRef] = []
    extras: set[str] = set()
    for spec in expanded:
        component_extra = _COMPONENT_EXTRAS.get(spec.type_name)
        if component_extra is not None:
            extras.add(component_extra)
        metadata = _RESOURCE_COMPONENTS.get(spec.type_name)
        if metadata is None:
            if spec.type_name.startswith("resource_") or {
                "model",
                "model_name",
                "model_ref",
            }.intersection(spec.config):
                raise PretrainedArtifactError(
                    f"model-bearing component {spec.type_name!r} has no explicit "
                    "pretrained resource metadata; refusing to omit its identity"
                )
            continue
        field, extra = metadata
        raw = spec.config.get(field)
        if spec.type_name == "sentence_transformer_embedder" and raw is None:
            raw = {
                "base": spec.config.get("model_name"),
                "kind": "hf",
            }
        if spec.type_name in _HF_MODEL_NAME_COMPONENTS and isinstance(raw, str):
            raw = {"base": raw, "kind": "hf"}
        if raw is None:
            raise PretrainedArtifactError(
                f"resource component {spec.type_name!r} has no model identity"
            )
        ref = normalize_model_ref(raw)  # type: ignore[arg-type]
        refs.append(ref)
        extras.add(extra)
        if spec.type_name in _INPROCESS_LLM_COMPONENTS and ref.kind in IN_PROCESS_KINDS:
            extras.add("semantic")
            if ref.adapter is not None:
                extras.add("finetune")
    unique = {json.dumps(to_config(ref), sort_keys=True): ref for ref in refs}
    return tuple(unique[key] for key in sorted(unique)), tuple(sorted(extras))


def build_manifest(
    root: Path,
    *,
    claim_level: ClaimLevel,
    measurement_summary: MeasurementSummary | None,
    allow_sensitive_config: bool,
) -> PretrainedManifest:
    resolver = ArtifactManifest.model_validate_json((root / RESOLVER_MANIFEST).read_text())
    model_refs, extras = resource_facts(resolver)
    credential_paths = credential_config_paths(resolver)
    if credential_paths:
        raise ArtifactEligibilityError(
            "resolver.json contains credential-bearing transport configuration that cannot "
            "be published. Remove these fields and inject credentials at runtime instead. "
            "Fields: " + ", ".join(credential_paths[:10])
        )
    sensitive_paths = sensitive_config_paths(resolver)
    if sensitive_paths and not allow_sensitive_config:
        raise ArtifactEligibilityError(
            "resolver.json contains prompt-bearing configuration that is excluded from upload "
            "by default. Inspect it, then pass allow_sensitive_config=True to publish it. "
            "Fields: " + ", ".join(sensitive_paths[:10])
        )
    if claim_level == "frozen-weights":
        raise ArtifactEligibilityError(
            "frozen-weights is not supported yet: local resource paths are not copied "
            "and rebased into the bundle. Use claim_level='reference-only'."
        )
    if claim_level == "benchmark-reproducible":
        _validate_benchmark_claim(model_refs, measurement_summary)
    summary_path: Literal["measurement-summary.json"] | None = (
        "measurement-summary.json" if measurement_summary is not None else None
    )
    return PretrainedManifest(
        langres_compatibility=_compatibility_for(LANGRES_VERSION),
        claim_level=claim_level,
        required_extras=extras,
        model_refs=model_refs,
        sensitive_config_included=bool(sensitive_paths),
        measurement_summary_path=summary_path,
        files=tuple(_iter_regular_files(root)),
    )


def render_model_card(
    manifest: PretrainedManifest,
    card: ModelCardSpec,
    summary: MeasurementSummary | None,
) -> str:
    """Render deterministic, bounded Markdown from explicitly selected facts."""
    refs = (
        "\n".join(
            f"- `{ref.base}` ({ref.kind}, revision={ref.revision or 'unpinned'})"
            for ref in manifest.model_refs
        )
        or "- No typed model resources recorded."
    )
    limitations = "\n".join(f"- {item}" for item in card.limitations)
    metrics = ""
    if summary is not None:
        identity = (
            f"- Protocol: `{summary.protocol_id or 'unspecified'}`\n"
            f"- Evaluation: `{summary.evaluation_id or 'unspecified'}`\n"
            "- Datasets: "
            + (
                ", ".join(f"`{dataset}`" for dataset in summary.dataset_ids)
                if summary.dataset_ids
                else "unspecified"
            )
        )
        facts: list[str] = []
        for section in ("quality", "cost", "tokens", "performance", "hardware", "size"):
            values = getattr(summary, section)
            if values:
                facts.append(
                    f"### {section.title()}\n\n"
                    + "\n".join(f"- `{key}`: {values[key]}" for key in sorted(values))
                )
        metrics = (
            "\n\n## Measurement summary\n\n"
            + identity
            + "\n\n"
            + ("\n\n".join(facts) if facts else "No numeric facts supplied.")
        )
    text = (
        "# langres entity-resolution artifact\n\n"
        f"**Claim level:** `{manifest.claim_level}`\n\n"
        f"**Prompt-bearing configuration included:** "
        f"`{str(manifest.sensitive_config_included).lower()}`\n\n"
        "## Intended use\n\n"
        f"{card.intended_use}\n\n"
        "## Model resources\n\n"
        f"{refs}\n\n"
        "## Limitations\n\n"
        f"{limitations}"
        f"{metrics}\n"
    )
    if len(text) > MAX_CARD_TEXT:
        raise PretrainedArtifactError(f"generated model card exceeds {MAX_CARD_TEXT} characters")
    return text


def dump_summary(path: Path, summary: MeasurementSummary) -> None:
    path.write_text(summary.model_dump_json(indent=2) + "\n")


def preflight_resolver_manifest(path: Path) -> None:
    """Resolve every registered type before ERModel.load constructs anything."""
    from langres.core.registry import (
        get_component,
        get_model,
        get_op_serializer,
    )

    manifest = ArtifactManifest.model_validate_json(path.read_bytes())
    try:
        if manifest.model_class is not None:
            get_model(manifest.model_class)
        for spec in component_specs(manifest):
            get_component(spec.type_name)
        for op in manifest.ops or ():
            get_op_serializer(op.role)
    except Exception as exc:
        raise PretrainedArtifactError(
            "artifact references an unavailable registered type. Install the package/extra "
            f"that owns it and register it before loading. Cause: {exc}"
        ) from None


def validate_remote_inventory(
    manifest: PretrainedManifest,
    inventory: Mapping[str, int | None],
) -> tuple[str, ...]:
    """Validate Hub metadata before downloading the full snapshot."""
    expected = {BUNDLE_MANIFEST, *(item.path for item in manifest.files)}
    missing = expected.difference(inventory)
    if missing:
        raise PretrainedArtifactError(
            f"remote artifact inventory mismatch: missing={sorted(missing)}"
        )
    for item in manifest.files:
        remote_size = inventory[item.path]
        if remote_size is None:
            raise PretrainedArtifactError(f"remote metadata has unknown size for {item.path!r}")
        if remote_size != item.size:
            raise PretrainedArtifactError(f"remote metadata size mismatch for {item.path!r}")
    metadata_limits = {
        manifest.resolver_path: MAX_RESOLVER_BYTES,
        manifest.model_card_path: MAX_CARD_BYTES,
    }
    if manifest.measurement_summary_path is not None:
        metadata_limits[manifest.measurement_summary_path] = MAX_SUMMARY_BYTES
    for relative, limit in metadata_limits.items():
        remote_size = inventory[relative]
        if remote_size is not None and remote_size > limit:
            raise PretrainedArtifactError(
                f"remote metadata file {relative!r} exceeds {limit} bytes"
            )
    manifest_size = inventory[BUNDLE_MANIFEST]
    if manifest_size is None or manifest_size > MAX_MANIFEST_BYTES:
        raise PretrainedArtifactError("remote bundle manifest size is unknown or oversized")
    return tuple(sorted(expected))


def validate_remote_manifest_inventory(inventory: Mapping[str, int | None]) -> None:
    """Bound the bootstrap manifest before asking a Hub client to download it."""
    if BUNDLE_MANIFEST not in inventory:
        raise PretrainedArtifactError(
            f"remote artifact inventory mismatch: missing={[BUNDLE_MANIFEST]}"
        )
    manifest_size = inventory[BUNDLE_MANIFEST]
    if manifest_size is None or manifest_size > MAX_MANIFEST_BYTES:
        raise PretrainedArtifactError("remote bundle manifest size is unknown or oversized")


def copy_allowlisted_files(
    source: Path,
    destination: Path,
    paths: Iterable[str],
) -> None:
    """Copy exact validated paths out of a client download directory."""
    import shutil

    if source.is_symlink() or not source.is_dir():
        raise PretrainedArtifactError("download root must be a regular directory")
    destination.mkdir(parents=True, exist_ok=False)
    for relative in paths:
        safe_relative_path(relative)
        src = source.joinpath(*PurePosixPath(relative).parts)
        relative_path = src.relative_to(source)
        ancestors = [
            source.joinpath(*relative_path.parts[:index])
            for index in range(1, len(relative_path.parts))
        ]
        if (
            any(parent.is_symlink() for parent in ancestors)
            or src.is_symlink()
            or not src.is_file()
        ):
            raise PretrainedArtifactError(
                f"downloaded artifact path {relative!r} is missing, a symlink, or non-regular"
            )
        dest = destination.joinpath(*PurePosixPath(relative).parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
