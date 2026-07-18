from __future__ import annotations

from pathlib import Path

from langres._pretrained_artifact import BUNDLE_MANIFEST
from langres.core.resolver import ERModel
from langres.hub import from_pretrained, push_to_hub

from .conftest import FakeHubTransport


def test_push_then_load_pins_revision_and_uses_exact_allowlist(
    model: ERModel, tmp_path: Path
) -> None:
    remote = tmp_path / "remote"
    transport = FakeHubTransport(remote)

    result = push_to_hub(
        model,
        "acme/entity-resolver",
        private=True,
        revision="research",
        commit_message="publish result",
        commit_description="reproduction bundle",
        parent_commit="b" * 40,
        token="hf_secret",
        transport=transport,
    )
    assert result.commit_oid == "a" * 40
    assert transport.create_calls == [("acme/entity-resolver", True)]
    upload = transport.upload_calls[0]
    assert upload["revision"] == "research"
    assert upload["parent_commit"] == "b" * 40
    assert BUNDLE_MANIFEST in upload["allow_patterns"]
    assert "hf_secret" not in repr(transport.__dict__)

    loaded = from_pretrained(
        "acme/entity-resolver",
        revision="release-1",
        token="hf_secret",
        transport=transport,
    )
    assert loaded.config_dict() == model.config_dict()
    assert loaded.pretrained_source_ is not None
    assert loaded.pretrained_source_.requested_revision == "release-1"
    assert loaded.pretrained_source_.resolved_revision == "a" * 40
    assert transport.resolve_calls == [("acme/entity-resolver", "release-1")]
    assert len(transport.download_calls) == 2
    assert transport.download_calls[0][1] == "a" * 40
    assert transport.download_calls[0][2] == (BUNDLE_MANIFEST,)
    assert transport.download_calls[1][1] == "a" * 40
    assert set(transport.download_calls[1][2]) == {
        item.relative_to(remote).as_posix() for item in remote.rglob("*") if item.is_file()
    }


def test_model_method_push_uses_same_transport(model: ERModel, tmp_path: Path) -> None:
    transport = FakeHubTransport(tmp_path / "remote")
    result = model.push_to_hub("acme/method", transport=transport)
    assert result.repo_id == "acme/method"


def test_repeated_push_ignores_stale_files_outside_new_manifest(
    model: ERModel, tmp_path: Path
) -> None:
    transport = FakeHubTransport(tmp_path / "remote")
    push_to_hub(
        model,
        "acme/repeated",
        measurement_summary={"quality": {"pair_f1": 0.9}},
        transport=transport,
    )
    assert (transport.root / "measurement-summary.json").is_file()

    push_to_hub(model, "acme/repeated", transport=transport)
    loaded = from_pretrained("acme/repeated", transport=transport)

    assert loaded.config_dict() == model.config_dict()
    assert "measurement-summary.json" not in transport.download_calls[-1][2]
