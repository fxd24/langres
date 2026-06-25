"""Unit tests for the resolver serialization contracts (M0 Wave 1).

Covers:
- SerializableState Protocol (the optional heavy-component capability)
- ComponentSpec / ArtifactManifest typed models
- ARTIFACT_VERSION constant
"""

from pathlib import Path

from langres.core.serialization import (
    ARTIFACT_VERSION,
    ArtifactManifest,
    ComponentSpec,
    SerializableState,
)


class TestArtifactVersion:
    def test_artifact_version_is_zero(self) -> None:
        assert ARTIFACT_VERSION == "0"


class TestComponentSpec:
    def test_round_trip(self) -> None:
        spec = ComponentSpec(
            type_name="all_pairs_blocker",
            config_version="1",
            config={"schema": "CompanySchema"},
        )
        restored = ComponentSpec.model_validate_json(spec.model_dump_json())
        assert restored == spec

    def test_config_version_defaults(self) -> None:
        spec = ComponentSpec(type_name="x", config={})
        assert spec.config_version == "1"


class TestArtifactManifest:
    def test_construct_and_round_trip(self) -> None:
        manifest = ArtifactManifest(
            artifact_version=ARTIFACT_VERSION,
            langres_version="0.1.0",
            components=[
                ComponentSpec(type_name="all_pairs_blocker", config={}),
                ComponentSpec(type_name="clusterer", config={"threshold": 0.7}),
            ],
        )
        restored = ArtifactManifest.model_validate_json(manifest.model_dump_json())
        assert restored == manifest
        assert restored.artifact_version == "0"
        assert len(restored.components) == 2

    def test_checksums_optional_default_empty(self) -> None:
        manifest = ArtifactManifest(
            artifact_version=ARTIFACT_VERSION,
            langres_version="0.1.0",
            components=[],
        )
        assert manifest.checksums == {}


class TestSerializableStateProtocol:
    def test_runtime_checkable_recognizes_implementer(self, tmp_path: Path) -> None:
        class _Heavy:
            def save_state(self, state_dir: Path) -> None:
                (state_dir / "marker").write_text("ok")

            def load_state(self, state_dir: Path) -> None:
                pass

        heavy = _Heavy()
        assert isinstance(heavy, SerializableState)
        heavy.save_state(tmp_path)
        assert (tmp_path / "marker").read_text() == "ok"

    def test_runtime_checkable_rejects_non_implementer(self) -> None:
        class _Light:
            pass

        assert not isinstance(_Light(), SerializableState)
