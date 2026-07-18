"""Safe local and Hugging Face lifecycle for langres pretrained artifacts.

The Hub client is an optional transport. Local ``save_pretrained`` and
``from_pretrained`` use only stdlib/Pydantic and the existing Resolver loader;
``huggingface_hub`` is imported only when a remote operation is requested.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from langres._pretrained_artifact import (
    BUNDLE_MANIFEST,
    MODEL_CARD,
    RESOLVER_MANIFEST,
    ArtifactEligibilityError,
    ClaimLevel,
    MeasurementSummary,
    ModelCardSpec,
    PretrainedArtifactError,
    PretrainedManifest,
    build_manifest,
    copy_allowlisted_files,
    dump_summary,
    preflight_resolver_manifest,
    read_manifest,
    render_model_card,
    validate_bundle,
    validate_remote_manifest_inventory,
    validate_remote_inventory,
)
from langres.core.resolver import ERModel
from langres.core.serialization import ArtifactSource


def _missing_hub_extra() -> PretrainedArtifactError:
    return PretrainedArtifactError(
        "Remote Hugging Face operations require the optional Hub client. "
        "Install it with: pip install 'langres[hub]'"
    )


def _provider_error(
    action: str,
    repo_id: str,
    *,
    revision: str | None,
    cause: Exception,
) -> PretrainedArtifactError:
    detail = f" at revision {revision!r}" if revision is not None else ""
    return PretrainedArtifactError(
        f"Hugging Face {action} failed for repository {repo_id!r}{detail} ({type(cause).__name__})"
    )


@dataclass(frozen=True)
class RemoteFile:
    """One path/size fact returned by remote repository metadata."""

    path: str
    size: int | None


@dataclass(frozen=True)
class RemoteRepository:
    """Resolved immutable repository metadata."""

    repo_id: str
    requested_revision: str | None
    resolved_revision: str
    files: tuple[RemoteFile, ...]


@dataclass(frozen=True)
class UploadResult:
    """Transport-neutral result of one atomic Hub commit."""

    repo_id: str
    revision: str
    commit_oid: str
    url: str


@runtime_checkable
class HubTransport(Protocol):
    """Minimal injected seam used by the real client and zero-network fakes."""

    def resolve(
        self,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> RemoteRepository:
        """Resolve a moving revision to an immutable commit and file inventory."""
        ...

    def snapshot_download(
        self,
        repo_id: str,
        *,
        revision: str,
        local_dir: Path,
        allow_patterns: Sequence[str],
        token: str | None,
    ) -> Path:
        """Download only the exact allowlisted paths at one immutable commit."""
        ...

    def create_repo(
        self,
        repo_id: str,
        *,
        private: bool,
        token: str | None,
    ) -> str:
        """Create or locate one model repository and return its canonical id."""
        ...

    def upload_folder(
        self,
        repo_id: str,
        *,
        folder_path: Path,
        revision: str,
        allow_patterns: Sequence[str],
        commit_message: str,
        commit_description: str | None,
        parent_commit: str | None,
        token: str | None,
    ) -> UploadResult:
        """Commit an exact validated bundle."""
        ...


class HuggingFaceHubTransport:
    """Lazy adapter over the official ``huggingface_hub`` Python client."""

    def resolve(
        self,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> RemoteRepository:
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as exc:
            raise _missing_hub_extra() from exc

        try:
            info = HfApi().model_info(
                repo_id,
                revision=revision,
                files_metadata=True,
                token=token,
            )
        except Exception as exc:
            raise _provider_error(
                "revision resolution",
                repo_id,
                revision=revision,
                cause=exc,
            ) from exc
        if not info.sha:
            raise PretrainedArtifactError(
                f"Hugging Face did not return an immutable SHA for {repo_id!r}"
            )
        siblings = info.siblings or ()
        return RemoteRepository(
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=info.sha,
            files=tuple(RemoteFile(path=item.rfilename, size=item.size) for item in siblings),
        )

    def snapshot_download(
        self,
        repo_id: str,
        *,
        revision: str,
        local_dir: Path,
        allow_patterns: Sequence[str],
        token: str | None,
    ) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:
            raise _missing_hub_extra() from exc

        try:
            result = snapshot_download(
                repo_id,
                repo_type="model",
                revision=revision,
                local_dir=local_dir,
                allow_patterns=list(allow_patterns),
                token=token,
            )
        except Exception as exc:
            raise _provider_error(
                "snapshot download",
                repo_id,
                revision=revision,
                cause=exc,
            ) from exc
        return Path(result)

    def create_repo(
        self,
        repo_id: str,
        *,
        private: bool,
        token: str | None,
    ) -> str:
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as exc:
            raise _missing_hub_extra() from exc

        try:
            result = HfApi().create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
                token=token,
            )
        except Exception as exc:
            raise _provider_error(
                "repository creation",
                repo_id,
                revision=None,
                cause=exc,
            ) from exc
        return str(getattr(result, "repo_id", repo_id))

    def upload_folder(
        self,
        repo_id: str,
        *,
        folder_path: Path,
        revision: str,
        allow_patterns: Sequence[str],
        commit_message: str,
        commit_description: str | None,
        parent_commit: str | None,
        token: str | None,
    ) -> UploadResult:
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as exc:
            raise _missing_hub_extra() from exc

        try:
            commit = HfApi().upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=folder_path,
                revision=revision,
                allow_patterns=list(allow_patterns),
                commit_message=commit_message,
                commit_description=commit_description,
                parent_commit=parent_commit,
                token=token,
            )
        except Exception as exc:
            raise _provider_error(
                "upload",
                repo_id,
                revision=revision,
                cause=exc,
            ) from exc
        oid = getattr(commit, "oid", None)
        if not isinstance(oid, str) or not oid:
            raise PretrainedArtifactError("Hugging Face upload returned no commit OID")
        return UploadResult(
            repo_id=repo_id,
            revision=revision,
            commit_oid=oid,
            url=str(commit),
        )


def _coerce_summary(
    value: MeasurementSummary | dict[str, object] | None,
) -> MeasurementSummary | None:
    if value is None or isinstance(value, MeasurementSummary):
        return value
    return MeasurementSummary.model_validate(value)


def _coerce_card(value: ModelCardSpec | dict[str, object] | None) -> ModelCardSpec:
    if value is None:
        return ModelCardSpec()
    if isinstance(value, ModelCardSpec):
        return value
    return ModelCardSpec.model_validate(value)


def _write_bundle(
    model: ERModel,
    root: Path,
    *,
    measurement_summary: MeasurementSummary | None,
    model_card: ModelCardSpec,
    claim_level: ClaimLevel,
    allow_sensitive_config: bool,
) -> PretrainedManifest:
    model.save(root)
    state_files = [
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != RESOLVER_MANIFEST
    ]
    if state_files:
        raise ArtifactEligibilityError(
            "pretrained bundles are state-free and cannot publish Resolver sidecars, "
            "because they may contain records, prompts, or native binary state. "
            "Publish a fresh configuration-only model. Found: "
            + ", ".join(sorted(state_files)[:10])
        )
    if measurement_summary is not None:
        dump_summary(root / "measurement-summary.json", measurement_summary)

    # Build once to derive resource identities for the card, then rebuild after
    # writing the card so README is included in the exact file allowlist.
    placeholder = root / MODEL_CARD
    placeholder.write_text("")
    manifest = build_manifest(
        root,
        claim_level=claim_level,
        measurement_summary=measurement_summary,
        allow_sensitive_config=allow_sensitive_config,
    )
    placeholder.write_text(render_model_card(manifest, model_card, measurement_summary))
    manifest = build_manifest(
        root,
        claim_level=claim_level,
        measurement_summary=measurement_summary,
        allow_sensitive_config=allow_sensitive_config,
    )
    (root / BUNDLE_MANIFEST).write_text(manifest.model_dump_json(indent=2) + "\n")
    validate_bundle(root)
    preflight_resolver_manifest(root / RESOLVER_MANIFEST)
    return manifest


def _promote_directory(stage: Path, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        stage.replace(destination)
        return
    suffix = (
        " In-place overwrite is intentionally unsupported; publish to a new directory "
        "and switch references after validation."
        if overwrite
        else ""
    )
    raise FileExistsError(f"{destination} already exists.{suffix}")


def save_pretrained(
    model: ERModel,
    path: str | Path,
    *,
    measurement_summary: MeasurementSummary | dict[str, object] | None = None,
    model_card: ModelCardSpec | dict[str, object] | None = None,
    claim_level: ClaimLevel = "reference-only",
    allow_sensitive_config: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write an atomic, validated bundle around the model's Resolver artifact."""
    destination = Path(path).expanduser()
    if destination.is_symlink():
        raise PretrainedArtifactError(f"artifact destination cannot be a symlink: {destination}")
    destination = destination.absolute()
    summary = _coerce_summary(measurement_summary)
    card = _coerce_card(model_card)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    promoted = False
    try:
        _write_bundle(
            model,
            stage,
            measurement_summary=summary,
            model_card=card,
            claim_level=claim_level,
            allow_sensitive_config=allow_sensitive_config,
        )
        _promote_directory(stage, destination, overwrite=overwrite)
        promoted = True
    finally:
        if not promoted and stage.exists():
            shutil.rmtree(stage)
    return destination


def _load_local(
    path: Path,
    *,
    source: ArtifactSource,
    model_cls: type[ERModel],
) -> ERModel:
    manifest = validate_bundle(path)
    preflight_resolver_manifest(path / manifest.resolver_path)
    model = model_cls.load(path)
    model.pretrained_source_ = source
    return model


def _remote_from_pretrained(
    repo_id: str,
    *,
    revision: str | None,
    token: str | None,
    transport: HubTransport,
    model_cls: type[ERModel],
) -> ERModel:
    remote = transport.resolve(repo_id, revision=revision, token=token)
    source = ArtifactSource(
        kind="hub",
        location=repo_id,
        requested_revision=revision,
        resolved_revision=remote.resolved_revision,
    )
    inventory = {item.path: item.size for item in remote.files}
    if len(inventory) != len(remote.files):
        raise PretrainedArtifactError("remote repository metadata contains duplicate paths")
    validate_remote_manifest_inventory(inventory)
    with tempfile.TemporaryDirectory(prefix="langres-hub-") as temp:
        temp_root = Path(temp)
        bootstrap = temp_root / "bootstrap"
        downloaded = transport.snapshot_download(
            repo_id,
            revision=remote.resolved_revision,
            local_dir=bootstrap,
            allow_patterns=(BUNDLE_MANIFEST,),
            token=token,
        )
        bootstrap_manifest = downloaded / BUNDLE_MANIFEST
        if bootstrap_manifest.is_symlink() or not bootstrap_manifest.is_file():
            raise PretrainedArtifactError(
                f"bootstrap download did not return a regular {BUNDLE_MANIFEST}"
            )
        manifest = read_manifest(bootstrap_manifest)
        allowlist = validate_remote_inventory(manifest, inventory)

        snapshot = temp_root / "snapshot"
        downloaded = transport.snapshot_download(
            repo_id,
            revision=remote.resolved_revision,
            local_dir=snapshot,
            allow_patterns=allowlist,
            token=token,
        )
        clean = temp_root / "validated"
        copy_allowlisted_files(downloaded, clean, allowlist)
        return _load_local(
            clean,
            source=source,
            model_cls=model_cls,
        )


def _from_pretrained_as(
    model_cls: type[ERModel],
    repo_or_path: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
    transport: HubTransport | None = None,
) -> ERModel:
    """Load a bundle through the class that owns the persistence method."""
    candidate = Path(repo_or_path).expanduser()
    if candidate.exists():
        if candidate.is_symlink():
            raise PretrainedArtifactError(f"artifact root cannot be a symlink: {candidate}")
        if revision is not None:
            raise ValueError("revision is only valid for a Hub repo id, not a local path")
        return _load_local(
            candidate.absolute(),
            source=ArtifactSource(kind="local", location=str(candidate.absolute())),
            model_cls=model_cls,
        )
    if isinstance(repo_or_path, Path):
        raise FileNotFoundError(f"local pretrained artifact does not exist: {candidate}")
    if (
        str(repo_or_path).startswith((".", "~", "/", "\\"))
        or re.match(r"^[A-Za-z]:[\\/]", str(repo_or_path)) is not None
    ):
        raise FileNotFoundError(f"local pretrained artifact does not exist: {candidate}")
    resolved_transport = transport or HuggingFaceHubTransport()
    return _remote_from_pretrained(
        str(repo_or_path),
        revision=revision,
        token=token,
        transport=resolved_transport,
        model_cls=model_cls,
    )


def from_pretrained(
    repo_or_path: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
    transport: HubTransport | None = None,
) -> ERModel:
    """Load a validated local bundle or an immutable allowlisted Hub snapshot."""
    return _from_pretrained_as(
        ERModel,
        repo_or_path,
        revision=revision,
        token=token,
        transport=transport,
    )


def push_to_hub(
    model: ERModel,
    repo_id: str,
    *,
    private: bool = False,
    revision: str = "main",
    commit_message: str = "Upload langres pretrained artifact",
    commit_description: str | None = None,
    parent_commit: str | None = None,
    token: str | None = None,
    measurement_summary: MeasurementSummary | dict[str, object] | None = None,
    model_card: ModelCardSpec | dict[str, object] | None = None,
    claim_level: ClaimLevel = "reference-only",
    allow_sensitive_config: bool = False,
    transport: HubTransport | None = None,
) -> UploadResult:
    """Build a fresh validated bundle and upload only its exact allowlist."""
    resolved_transport = transport or HuggingFaceHubTransport()
    with tempfile.TemporaryDirectory(prefix="langres-push-") as temp:
        bundle = Path(temp) / "artifact"
        save_pretrained(
            model,
            bundle,
            measurement_summary=measurement_summary,
            model_card=model_card,
            claim_level=claim_level,
            allow_sensitive_config=allow_sensitive_config,
        )
        manifest = validate_bundle(bundle)
        allowlist = tuple(sorted({BUNDLE_MANIFEST, *(item.path for item in manifest.files)}))
        canonical_repo = resolved_transport.create_repo(
            repo_id,
            private=private,
            token=token,
        )
        return resolved_transport.upload_folder(
            canonical_repo,
            folder_path=bundle,
            revision=revision,
            allow_patterns=allowlist,
            commit_message=commit_message,
            commit_description=commit_description,
            parent_commit=parent_commit,
            token=token,
        )


__all__ = [
    "ArtifactEligibilityError",
    "ArtifactSource",
    "ClaimLevel",
    "HubTransport",
    "HuggingFaceHubTransport",
    "MeasurementSummary",
    "ModelCardSpec",
    "PretrainedArtifactError",
    "RemoteFile",
    "RemoteRepository",
    "UploadResult",
    "from_pretrained",
    "push_to_hub",
    "save_pretrained",
]
