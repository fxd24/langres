from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from langres.hub import HuggingFaceHubTransport, PretrainedArtifactError


class _Commit:
    oid = "c" * 40

    def __str__(self) -> str:
        return "https://huggingface.co/acme/model/commit/" + self.oid


class _Api:
    calls: list[tuple[str, dict[str, object]]] = []

    def model_info(self, repo_id: str, **kwargs: object) -> object:
        self.calls.append(("model_info", {"repo_id": repo_id, **kwargs}))
        return SimpleNamespace(
            sha="a" * 40,
            siblings=[SimpleNamespace(rfilename="resolver.json", size=12)],
        )

    def create_repo(self, **kwargs: object) -> object:
        self.calls.append(("create_repo", dict(kwargs)))
        return SimpleNamespace(repo_id=kwargs["repo_id"])

    def upload_folder(self, **kwargs: object) -> object:
        self.calls.append(("upload_folder", dict(kwargs)))
        return _Commit()


def test_real_transport_forwards_pins_allowlists_and_commit_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    _Api.calls = []
    snapshot_calls: list[dict[str, object]] = []

    def _snapshot_download(repo_id: str, **kwargs: object) -> str:
        snapshot_calls.append({"repo_id": repo_id, **kwargs})
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot_download)
    transport = HuggingFaceHubTransport()

    info = transport.resolve("acme/model", revision="tag", token="secret")
    assert info.resolved_revision == "a" * 40
    assert info.files[0].size == 12

    transport.snapshot_download(
        "acme/model",
        revision=info.resolved_revision,
        local_dir=tmp_path,
        allow_patterns=("resolver.json",),
        token="secret",
    )
    assert snapshot_calls[0]["revision"] == "a" * 40
    assert snapshot_calls[0]["allow_patterns"] == ["resolver.json"]

    assert transport.create_repo("acme/model", private=True, token="secret") == "acme/model"
    result = transport.upload_folder(
        "acme/model",
        folder_path=tmp_path,
        revision="main",
        allow_patterns=("resolver.json",),
        commit_message="publish",
        commit_description="facts",
        parent_commit="b" * 40,
        token="secret",
    )
    assert result.commit_oid == "c" * 40
    upload = next(kwargs for name, kwargs in _Api.calls if name == "upload_folder")
    assert upload["parent_commit"] == "b" * 40
    assert upload["allow_patterns"] == ["resolver.json"]


def test_provider_errors_are_sanitized_with_repository_context(monkeypatch) -> None:
    class FailingApi:
        def model_info(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("secret provider detail")

    monkeypatch.setattr(huggingface_hub, "HfApi", FailingApi)
    with pytest.raises(
        PretrainedArtifactError,
        match=r"revision resolution failed.*acme/model.*release.*RuntimeError",
    ) as error:
        HuggingFaceHubTransport().resolve(
            "acme/model",
            revision="release",
            token="secret",
        )
    assert "secret provider detail" not in str(error.value)
