from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from langres.core.resolver import ERModel
from langres.hub import RemoteFile, RemoteRepository, UploadResult


class Entity(BaseModel):
    id: str
    name: str


@pytest.fixture
def model() -> ERModel:
    return ERModel.from_schema(Entity)


class FakeHubTransport:
    """Filesystem-backed zero-network Hub transport."""

    def __init__(self, root: Path, *, sha: str = "a" * 40) -> None:
        self.root = root
        self.sha = sha
        self.repo_id: str | None = None
        self.resolve_calls: list[tuple[str, str | None]] = []
        self.download_calls: list[tuple[str, str, tuple[str, ...]]] = []
        self.create_calls: list[tuple[str, bool]] = []
        self.upload_calls: list[dict[str, object]] = []

    def resolve(
        self,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> RemoteRepository:
        del token
        self.resolve_calls.append((repo_id, revision))
        files = tuple(
            RemoteFile(path=item.relative_to(self.root).as_posix(), size=item.stat().st_size)
            for item in sorted(self.root.rglob("*"))
            if item.is_file()
        )
        return RemoteRepository(
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=self.sha,
            files=files,
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
        del token
        self.download_calls.append((repo_id, revision, tuple(allow_patterns)))
        local_dir.mkdir(parents=True, exist_ok=True)
        for relative in allow_patterns:
            source = self.root / relative
            destination = local_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return local_dir

    def create_repo(
        self,
        repo_id: str,
        *,
        private: bool,
        token: str | None,
    ) -> str:
        del token
        self.repo_id = repo_id
        self.create_calls.append((repo_id, private))
        self.root.mkdir(parents=True, exist_ok=True)
        return repo_id

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
        del token
        self.upload_calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "allow_patterns": tuple(allow_patterns),
                "commit_message": commit_message,
                "commit_description": commit_description,
                "parent_commit": parent_commit,
            }
        )
        for relative in allow_patterns:
            source = folder_path / relative
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return UploadResult(
            repo_id=repo_id,
            revision=revision,
            commit_oid=self.sha,
            url=f"https://huggingface.co/{repo_id}/commit/{self.sha}",
        )
