from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from langres._pretrained_artifact import (
    BUNDLE_MANIFEST,
    ArtifactEligibilityError,
    MAX_MANIFEST_BYTES,
    MAX_RESOLVER_BYTES,
    PretrainedArtifactError,
)
from langres.core.resolver import ERModel
from langres.hub import from_pretrained, push_to_hub, save_pretrained

from .conftest import Entity, FakeHubTransport


def test_unexpected_remote_file_is_never_downloaded(model: ERModel, tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    transport = FakeHubTransport(remote)
    push_to_hub(model, "acme/model", transport=transport)
    (remote / "records.jsonl").write_text('{"secret": true}\n')

    loaded = from_pretrained("acme/model", transport=transport)
    assert loaded.config_dict() == model.config_dict()
    assert len(transport.download_calls) == 2
    assert transport.download_calls[0][2] == (BUNDLE_MANIFEST,)
    assert "records.jsonl" not in transport.download_calls[1][2]


def test_state_sidecars_are_never_published(
    model: ERModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_save = model.save

    def save_with_sensitive_state(path: str | Path) -> None:
        original_save(path)
        state = Path(path) / "module"
        state.mkdir()
        (state / "program.json").write_text('{"prompt": "private"}')

    monkeypatch.setattr(model, "save", save_with_sensitive_state)
    destination = tmp_path / "bundle"
    with pytest.raises(ArtifactEligibilityError, match="state-free"):
        save_pretrained(model, destination)
    assert not destination.exists()


def test_prompt_bearing_config_requires_explicit_publication_consent(
    tmp_path: Path,
) -> None:
    pytest.importorskip("litellm")
    prompt = "PRIVATE PROMPT {left} versus {right}"
    system = "PRIVATE SYSTEM INSTRUCTION"
    model = ERModel.from_schema(
        Entity,
        matcher="prompt_llm",
        model="openai/gpt-5-mini",
        prompt_template=prompt,
        system_prompt=system,
    )

    with pytest.raises(ArtifactEligibilityError, match="allow_sensitive_config=True"):
        save_pretrained(model, tmp_path / "rejected")
    bundle = save_pretrained(
        model,
        tmp_path / "consented",
        allow_sensitive_config=True,
    )
    resolver = (bundle / "resolver.json").read_text()
    assert prompt in resolver
    assert system in resolver
    assert json.loads((bundle / BUNDLE_MANIFEST).read_text())["sensitive_config_included"]


def test_symlink_in_local_bundle_is_rejected(model: ERModel, tmp_path: Path) -> None:
    bundle = save_pretrained(model, tmp_path / "bundle")
    target = tmp_path / "outside"
    target.write_text("secret")
    link = bundle / "module"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PretrainedArtifactError, match="symlink"):
        from_pretrained(bundle)


def test_symlink_destination_is_rejected(model: ERModel, tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "bundle"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PretrainedArtifactError, match="destination cannot be a symlink"):
        save_pretrained(model, link, overwrite=True)
    assert not (target / BUNDLE_MANIFEST).exists()


def test_manifest_traversal_is_rejected(model: ERModel, tmp_path: Path) -> None:
    bundle = save_pretrained(model, tmp_path / "bundle")
    payload = json.loads((bundle / BUNDLE_MANIFEST).read_text())
    payload["files"][0]["path"] = "../resolver.json"
    (bundle / BUNDLE_MANIFEST).write_text(json.dumps(payload))
    with pytest.raises(PretrainedArtifactError, match="unsafe artifact path"):
        from_pretrained(bundle)


def test_oversized_manifest_is_rejected(model: ERModel, tmp_path: Path) -> None:
    bundle = save_pretrained(model, tmp_path / "bundle")
    (bundle / BUNDLE_MANIFEST).write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
    with pytest.raises(PretrainedArtifactError, match="exceeds"):
        from_pretrained(bundle)


def test_oversized_resolver_is_rejected_before_parsing(model: ERModel, tmp_path: Path) -> None:
    bundle = save_pretrained(model, tmp_path / "bundle")
    content = b" " * (MAX_RESOLVER_BYTES + 1)
    (bundle / "resolver.json").write_bytes(content)
    manifest = json.loads((bundle / BUNDLE_MANIFEST).read_text())
    import hashlib

    for item in manifest["files"]:
        if item["path"] == "resolver.json":
            item["size"] = len(content)
            item["sha256"] = hashlib.sha256(content).hexdigest()
    (bundle / BUNDLE_MANIFEST).write_text(json.dumps(manifest))

    with pytest.raises(PretrainedArtifactError, match="metadata file.*exceeds"):
        from_pretrained(bundle)


def test_unknown_component_fails_preflight_before_model_load(
    model: ERModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = save_pretrained(model, tmp_path / "bundle")
    resolver = json.loads((bundle / "resolver.json").read_text())
    resolver["components"][0]["type_name"] = "attacker_component"
    (bundle / "resolver.json").write_text(json.dumps(resolver))

    # Refresh the outer checksum so this reaches registry preflight rather than
    # being rejected earlier as ordinary corruption.
    manifest = json.loads((bundle / BUNDLE_MANIFEST).read_text())
    import hashlib

    content = (bundle / "resolver.json").read_bytes()
    for item in manifest["files"]:
        if item["path"] == "resolver.json":
            item["size"] = len(content)
            item["sha256"] = hashlib.sha256(content).hexdigest()
    (bundle / BUNDLE_MANIFEST).write_text(json.dumps(manifest))

    called = False

    def _load(_cls, _path):
        nonlocal called
        called = True
        raise AssertionError("must not reconstruct")

    monkeypatch.setattr(ERModel, "load", classmethod(_load))
    with pytest.raises(PretrainedArtifactError, match="unavailable registered type"):
        from_pretrained(bundle)
    assert called is False


def test_unknown_nested_component_fails_preflight_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("litellm")
    from langres.architectures import VectorLLMCascade
    from langres.core.model_ref import ModelRef

    model = VectorLLMCascade(
        schema=Entity,
        embedder=ModelRef(
            base="sentence-transformers/all-MiniLM-L6-v2",
            kind="hf",
            revision="e" * 40,
        ),
        llm=ModelRef(base="openai/gpt-5-mini", kind="api"),
    )
    bundle = save_pretrained(
        model,
        tmp_path / "bundle",
        allow_sensitive_config=True,
    )
    resolver = json.loads((bundle / "resolver.json").read_text())
    cascade = next(item for item in resolver["components"] if item["type_name"] == "cascade_judge")
    cascade["config"]["student"] = {"type_name": "attacker_nested", "config": {}}
    (bundle / "resolver.json").write_text(json.dumps(resolver))

    manifest = json.loads((bundle / BUNDLE_MANIFEST).read_text())
    import hashlib

    content = (bundle / "resolver.json").read_bytes()
    for item in manifest["files"]:
        if item["path"] == "resolver.json":
            item["size"] = len(content)
            item["sha256"] = hashlib.sha256(content).hexdigest()
    (bundle / BUNDLE_MANIFEST).write_text(json.dumps(manifest))

    called = False

    def _load(_cls, _path):
        nonlocal called
        called = True
        raise AssertionError("must not reconstruct")

    monkeypatch.setattr(ERModel, "load", classmethod(_load))
    with pytest.raises(PretrainedArtifactError, match="unavailable registered type"):
        from_pretrained(bundle)
    assert called is False


def test_outer_resource_identity_cannot_disagree_with_resolver(
    model: ERModel, tmp_path: Path
) -> None:
    bundle = save_pretrained(model, tmp_path / "bundle")
    manifest = json.loads((bundle / BUNDLE_MANIFEST).read_text())
    manifest["required_extras"] = ["attacker-extra"]
    (bundle / BUNDLE_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(PretrainedArtifactError, match="outer resource identity"):
        from_pretrained(bundle)


def test_importing_hub_adapter_keeps_optional_client_lazy() -> None:
    script = (
        "import sys; import langres.hub; "
        "assert 'huggingface_hub' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'litellm' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_hub_extra_has_install_remedy() -> None:
    script = """
import builtins
from langres.hub import HuggingFaceHubTransport, PretrainedArtifactError

original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "huggingface_hub":
        raise ModuleNotFoundError("No module named 'huggingface_hub'")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked

try:
    HuggingFaceHubTransport().resolve("acme/model", revision=None, token=None)
except PretrainedArtifactError as exc:
    assert "pip install 'langres[hub]'" in str(exc)
else:
    raise AssertionError("missing optional dependency was not translated")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
