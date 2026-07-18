"""Unit tests for the resolver serialization contracts (M0 Wave 1).

Covers:
- SerializableState Protocol (the optional heavy-component capability)
- ComponentSpec / OpSpec / ArtifactManifest typed models
- ARTIFACT_VERSION / CLASSIC_ARTIFACT_VERSION constants
"""

from pathlib import Path

from langres.core.serialization import (
    ARTIFACT_VERSION,
    CLASSIC_ARTIFACT_VERSION,
    ArtifactManifest,
    ComponentSpec,
    OpSpec,
    SerializableState,
)


class TestArtifactVersion:
    def test_reader_max_is_two(self) -> None:
        # Bumped "1" -> "2" when the explicit Op-chain layout landed (#193 persist v2).
        assert ARTIFACT_VERSION == "2"

    def test_classic_layout_frozen_at_one(self) -> None:
        # The classic four-slot layout never restamps, so its bytes/recipe_id hold.
        assert CLASSIC_ARTIFACT_VERSION == "1"


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


class TestOpSpec:
    def test_round_trip_with_component(self) -> None:
        spec = OpSpec(
            role="matcher_score",
            params={"out_space": "prob_llm"},
            component=ComponentSpec(type_name="costed_name_matcher", config={"cost_each": 0.0}),
        )
        restored = OpSpec.model_validate_json(spec.model_dump_json())
        assert restored == spec

    def test_params_and_component_default_empty(self) -> None:
        # A Select carries only its role + params; component defaults to None.
        spec = OpSpec(role="threshold_select", params={"threshold": 0.5})
        assert spec.component is None
        # An entirely bare spec is valid too (params default to {}).
        assert OpSpec(role="x").params == {}


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
        assert restored.artifact_version == ARTIFACT_VERSION
        assert len(restored.components) == 2
        assert restored.ops is None  # a components manifest carries no ops

    def test_ops_round_trip(self) -> None:
        manifest = ArtifactManifest(
            artifact_version=ARTIFACT_VERSION,
            langres_version="0.1.0",
            ops=[
                OpSpec(
                    role="blocker_source",
                    component=ComponentSpec(type_name="all_pairs_blocker", config={}),
                ),
                OpSpec(role="threshold_select", params={"threshold": 0.5}),
            ],
        )
        restored = ArtifactManifest.model_validate_json(manifest.model_dump_json())
        assert restored == manifest
        assert restored.ops is not None
        assert restored.components == []  # an ops manifest carries no components

    def test_ops_defaults_none_so_old_v1_json_reads_classic(self) -> None:
        # A pre-#193 v1 resolver.json (components, no ``ops`` key) validates with
        # ops=None -> the classic read path (F4).
        legacy_json = (
            '{"artifact_version": "1", "langres_version": "0.3.0", "model_class": null, '
            '"components": [{"type_name": "clusterer", "config": {"threshold": 0.7}}], '
            '"checksums": {}}'
        )
        manifest = ArtifactManifest.model_validate_json(legacy_json)
        assert manifest.ops is None
        assert len(manifest.components) == 1

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
