"""Immutable local artifacts for explicitly declared execution replay boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from langres.core.op import ExecutionCheckpoint

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class ScoreCacheError(RuntimeError):
    """A stage artifact was invalid, corrupt, or would violate immutability."""


class StageArtifactManifest(BaseModel):
    """Identity and checksum envelope for one checkpoint payload."""

    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    cache_id: str
    prefix_plan_id: str
    boundary_index: int
    boundary_stage_id: str
    input_fingerprint: str
    payload_sha256: str


def ordered_input_fingerprint(records: Sequence[Any]) -> str:
    """Hash input rows in caller order while canonicalizing mapping key order."""
    payload = [
        record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        for record in records
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class StageArtifactStore:
    """Atomic, immutable, checksummed checkpoint storage with quarantine."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _entry(self, cache_id: str) -> Path:
        if not _SAFE_ID.fullmatch(cache_id):
            raise ScoreCacheError(
                "cache_id contains unsafe path characters; expected letters, "
                "numbers, dot, underscore, or hyphen"
            )
        return self.root / cache_id

    @staticmethod
    def _payload(checkpoint: ExecutionCheckpoint) -> bytes:
        return checkpoint.model_dump_json().encode("utf-8")

    @staticmethod
    def _validate_pairs(checkpoint: ExecutionCheckpoint) -> None:
        seen: set[tuple[str, str]] = set()
        duplicates: set[tuple[str, str]] = set()
        for row in checkpoint.rows:
            key = (row.left_id, row.right_id)
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        if duplicates:
            preview = ", ".join(repr(pair) for pair in sorted(duplicates)[:3])
            raise ScoreCacheError(
                "checkpoint contains duplicate ordered pair ids; refusing cache "
                f"commit: {preview}"
            )

    def put(self, checkpoint: ExecutionCheckpoint) -> Path:
        """Atomically commit one checkpoint; never overwrite different bytes."""
        self._validate_pairs(checkpoint)
        entry = self._entry(checkpoint.cache_id)
        payload = self._payload(checkpoint)
        checksum = hashlib.sha256(payload).hexdigest()
        manifest = StageArtifactManifest(
            cache_id=checkpoint.cache_id,
            prefix_plan_id=checkpoint.prefix_plan_id,
            boundary_index=checkpoint.boundary_index,
            boundary_stage_id=checkpoint.boundary_stage_id,
            input_fingerprint=checkpoint.input_fingerprint,
            payload_sha256=checksum,
        )
        if entry.exists():
            try:
                existing_payload = (entry / "checkpoint.json").read_bytes()
                existing_manifest = StageArtifactManifest.model_validate_json(
                    (entry / "manifest.json").read_text(encoding="utf-8")
                )
                if (
                    existing_payload == payload
                    and existing_manifest.payload_sha256 == checksum
                ):
                    return entry
            except (OSError, ValueError):
                # A corrupt pre-existing entry is never overwritten implicitly.
                pass
            raise ScoreCacheError(
                f"immutable cache entry {checkpoint.cache_id!r} already exists "
                "with different content"
            )

        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.root))
        try:
            payload_path = temporary / "checkpoint.json"
            manifest_path = temporary / "manifest.json"
            payload_path.write_bytes(payload)
            manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
            for path in (payload_path, manifest_path):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            os.replace(temporary, entry)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return entry

    def load(
        self,
        cache_id: str,
        *,
        prefix_plan_id: str,
        boundary_index: int,
        input_fingerprint: str,
    ) -> ExecutionCheckpoint | None:
        """Load a strictly matching artifact; quarantine any malformed entry."""
        entry = self._entry(cache_id)
        if not entry.exists():
            return None
        try:
            manifest = StageArtifactManifest.model_validate_json(
                (entry / "manifest.json").read_text(encoding="utf-8")
            )
            payload = (entry / "checkpoint.json").read_bytes()
            checksum = hashlib.sha256(payload).hexdigest()
            if checksum != manifest.payload_sha256:
                raise ScoreCacheError("checkpoint payload checksum mismatch")
            checkpoint = ExecutionCheckpoint.model_validate_json(payload)
            self._validate_pairs(checkpoint)
            expected = {
                "cache_id": cache_id,
                "prefix_plan_id": prefix_plan_id,
                "boundary_index": boundary_index,
                "input_fingerprint": input_fingerprint,
            }
            for name, value in expected.items():
                if getattr(manifest, name) != value or getattr(checkpoint, name) != value:
                    raise ScoreCacheError(f"checkpoint {name} identity mismatch")
            if (
                checkpoint.boundary_stage_id != manifest.boundary_stage_id
                or checkpoint.prefix_plan_id != manifest.prefix_plan_id
            ):
                raise ScoreCacheError("checkpoint manifest/payload identity mismatch")
            return checkpoint
        except (OSError, ValueError, ScoreCacheError):
            self._quarantine(entry)
            return None

    def _quarantine(self, entry: Path) -> None:
        quarantine = self.root / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(str(entry).encode("utf-8")).hexdigest()[:12]
        target = quarantine / f"{entry.name}-{digest}"
        suffix = 1
        while target.exists():
            target = quarantine / f"{entry.name}-{digest}-{suffix}"
            suffix += 1
        os.replace(entry, target)


__all__ = [
    "ScoreCacheError",
    "StageArtifactManifest",
    "StageArtifactStore",
    "ordered_input_fingerprint",
]
