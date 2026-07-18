from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from langres.architectures import RetrieveRerankLLM
from langres.core.model_ref import ModelRef
from langres.core.resolver import ERModel
from langres.hub import (
    RemoteFile,
    RemoteRepository,
    UploadResult,
    from_pretrained,
    push_to_hub,
)


class _RecipeHubEntity(BaseModel):
    id: str
    name: str


class _FakeHub:
    def __init__(self, root: Path, *, revision: str) -> None:
        self.root = root
        self.revision = revision
        self.resolved: list[tuple[str, str | None]] = []

    def resolve(
        self,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> RemoteRepository:
        del token
        self.resolved.append((repo_id, revision))
        return RemoteRepository(
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=self.revision,
            files=tuple(
                RemoteFile(
                    path=path.relative_to(self.root).as_posix(),
                    size=path.stat().st_size,
                )
                for path in sorted(self.root.rglob("*"))
                if path.is_file()
            ),
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
        del repo_id, token
        assert revision == self.revision
        for relative in allow_patterns:
            destination = local_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / relative, destination)
        return local_dir

    def create_repo(
        self,
        repo_id: str,
        *,
        private: bool,
        token: str | None,
    ) -> str:
        del private, token
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
        del commit_message, commit_description, parent_commit, token
        for relative in allow_patterns:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder_path / relative, destination)
        return UploadResult(
            repo_id=repo_id,
            revision=revision,
            commit_oid=self.revision,
            url=f"https://huggingface.co/{repo_id}/commit/{self.revision}",
        )


@pytest.mark.integration
def test_complete_recipe_round_trips_locally_and_through_a_pinned_fake_hub(
    tmp_path: Path,
) -> None:
    recipe = RetrieveRerankLLM(
        embedder=ModelRef(base="org/embed", kind="hf", revision="1" * 40),
        reranker=ModelRef(base="org/rerank", kind="hf", revision="2" * 40),
        llm=ModelRef(base="org/llm", kind="hf", revision="3" * 40),
        schema=_RecipeHubEntity,
        retrieve_k=12,
        llm_k=3,
        threshold=0.73,
    )

    local = tmp_path / "local"
    recipe.save_pretrained(local)
    local_loaded = ERModel.from_pretrained(local)
    assert type(local_loaded) is RetrieveRerankLLM
    assert local_loaded.config_dict() == recipe.config_dict()
    assert local_loaded.resources == recipe.resources
    assert local_loaded.execution_plan().replay_boundary is not None

    pinned_revision = "a" * 40
    transport = _FakeHub(tmp_path / "remote", revision=pinned_revision)
    push_to_hub(recipe, "acme/research-recipe", transport=transport)
    hub_loaded = from_pretrained(
        "acme/research-recipe",
        revision="release-1",
        transport=transport,
    )

    assert type(hub_loaded) is RetrieveRerankLLM
    assert hub_loaded.config_dict() == recipe.config_dict()
    assert hub_loaded.resources == recipe.resources
    assert hub_loaded.pretrained_source_ is not None
    assert hub_loaded.pretrained_source_.requested_revision == "release-1"
    assert hub_loaded.pretrained_source_.resolved_revision == pinned_revision
    assert transport.resolved == [("acme/research-recipe", "release-1")]
