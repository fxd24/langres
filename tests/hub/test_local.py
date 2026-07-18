from __future__ import annotations

import json
from pathlib import Path

import pytest

from langres._pretrained_artifact import (
    BUNDLE_MANIFEST,
    ArtifactEligibilityError,
    MeasurementSummary,
    PretrainedArtifactError,
    _version_minor,
    validate_bundle,
)
from langres.architectures import VectorLLMCascade
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.model_ref import ModelRef
from langres.core.op import ThresholdSelect
from langres.core.op_adapters import BlockerSource, ClustererStage
from langres.core.resolver import ERModel
from langres.hub import from_pretrained, save_pretrained
from langres.resources import Generate, Parse, TransformersLLM

from .conftest import Entity, FakeHubTransport


def test_save_pretrained_preserves_resolver_bytes_and_loads(model: ERModel, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    model.save(raw)
    bundle = save_pretrained(model, tmp_path / "bundle")

    assert (bundle / "resolver.json").read_bytes() == (raw / "resolver.json").read_bytes()
    assert (bundle / BUNDLE_MANIFEST).is_file()
    assert (bundle / "README.md").is_file()
    manifest = validate_bundle(bundle)
    assert manifest.claim_level == "reference-only"

    loaded = from_pretrained(bundle)
    assert isinstance(loaded, ERModel)
    assert loaded.config_dict() == model.config_dict()
    assert loaded.pretrained_source_ is not None
    assert loaded.pretrained_source_.kind == "local"
    assert loaded.pretrained_source_.location == str(bundle.resolve())


def test_method_shims_match_free_functions(model: ERModel, tmp_path: Path) -> None:
    bundle = model.save_pretrained(tmp_path / "bundle")
    loaded = ERModel.from_pretrained(bundle)
    assert loaded.config_dict() == model.config_dict()


def test_unregistered_subclass_from_pretrained_preserves_requested_class(
    tmp_path: Path,
) -> None:
    class ResearchResolver(ERModel):
        pass

    model = ResearchResolver.from_schema(Entity)
    transport = FakeHubTransport(tmp_path / "remote")
    model.push_to_hub("acme/research-resolver", transport=transport)

    loaded = ResearchResolver.from_pretrained(
        "acme/research-resolver",
        transport=transport,
    )

    assert type(loaded) is ResearchResolver
    assert loaded.config_dict() == model.config_dict()


def test_measurement_summary_is_bounded_and_published(model: ERModel, tmp_path: Path) -> None:
    summary = MeasurementSummary(
        protocol_id="official-v1",
        evaluation_id="eval-1",
        dataset_ids=("amazon-google:test",),
        quality={"pair_f1": 0.91},
        tokens={"input_tokens": 123, "output_tokens": 45},
        cost={"usd": None},
        hardware={"accelerator": "cpu"},
    )
    bundle = save_pretrained(
        model,
        tmp_path / "bundle",
        measurement_summary=summary,
    )
    payload = json.loads((bundle / "measurement-summary.json").read_text())
    card = (bundle / "README.md").read_text()
    assert payload["tokens"]["input_tokens"] == 123
    assert "pair_f1" in card
    assert "input_tokens" in card


def test_frozen_weights_claim_fails_before_promotion(model: ERModel, tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    with pytest.raises(ArtifactEligibilityError, match="frozen-weights"):
        save_pretrained(model, destination, claim_level="frozen-weights")
    assert not destination.exists()


def test_existing_destination_requires_explicit_overwrite(model: ERModel, tmp_path: Path) -> None:
    destination = save_pretrained(model, tmp_path / "bundle")
    marker = destination / "old-marker.txt"
    marker.write_text("old")
    with pytest.raises(FileExistsError):
        save_pretrained(model, destination)
    assert marker.read_text() == "old"

    with pytest.raises(FileExistsError, match="intentionally unsupported"):
        save_pretrained(model, destination, overwrite=True)
    assert marker.read_text() == "old"


def test_checksum_failure_happens_before_resolver_load(
    model: ERModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = save_pretrained(model, tmp_path / "bundle")
    (bundle / "resolver.json").write_text("{}")
    called = False

    def _load(_cls, _path):
        nonlocal called
        called = True
        raise AssertionError("must not construct components")

    monkeypatch.setattr(ERModel, "load", classmethod(_load))
    with pytest.raises(PretrainedArtifactError, match="checksum/size mismatch"):
        from_pretrained(bundle)
    assert called is False


def test_local_revision_is_rejected(model: ERModel, tmp_path: Path) -> None:
    bundle = save_pretrained(model, tmp_path / "bundle")
    with pytest.raises(ValueError, match="only valid for a Hub repo"):
        from_pretrained(bundle, revision="main")


def test_missing_path_object_is_not_mistaken_for_hub_repo(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        from_pretrained(missing)


@pytest.mark.parametrize("missing", ("./missing", "/tmp/langres-missing", "~/langres-missing"))
def test_missing_path_like_string_is_not_mistaken_for_hub_repo(missing: str) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        from_pretrained(missing)


@pytest.mark.parametrize(
    ("version", "expected"),
    (
        ("0.0.0.dev0", (0, 0)),
        ("0.3.0rc1", (0, 3)),
        ("0.3.0.post1", (0, 3)),
        ("1.2.3+local.1", (1, 2)),
    ),
)
def test_version_compatibility_accepts_pep440_versions(
    version: str, expected: tuple[int, int]
) -> None:
    assert _version_minor(version) == expected


def _transformer_topology(
    *,
    revision: str | None,
    adapter: str | None = None,
    adapter_revision: str | None = None,
) -> ERModel:
    return ERModel.from_topology(
        ops=[
            BlockerSource(AllPairsBlocker(schema=Entity)),
            Generate(
                TransformersLLM(
                    ModelRef(
                        base="acme/local-matcher",
                        kind="hf",
                        revision=revision,
                        adapter=adapter,
                        adapter_revision=adapter_revision,
                    )
                )
            ),
            Parse(),
            ThresholdSelect(0.5),
            ClustererStage(Clusterer(threshold=0.5)),
        ]
    )


def test_benchmark_claim_requires_pinned_hf_resources_and_summary(
    tmp_path: Path,
) -> None:
    summary = MeasurementSummary(
        protocol_id="protocol-v1",
        evaluation_id="evaluation-v1",
        dataset_ids=("fixture:test",),
        quality={"pair_f1": 0.9},
    )
    bundle = save_pretrained(
        _transformer_topology(revision="a" * 40),
        tmp_path / "pinned",
        measurement_summary=summary,
        claim_level="benchmark-reproducible",
    )
    manifest = validate_bundle(bundle)
    assert manifest.claim_level == "benchmark-reproducible"
    assert manifest.model_refs[0].revision == "a" * 40
    assert manifest.required_extras == ("semantic",)

    with pytest.raises(ArtifactEligibilityError, match="commit SHA"):
        save_pretrained(
            _transformer_topology(revision=None),
            tmp_path / "unpinned",
            measurement_summary=summary,
            claim_level="benchmark-reproducible",
        )


def test_benchmark_claim_requires_identified_nonempty_measurement(
    model: ERModel, tmp_path: Path
) -> None:
    with pytest.raises(ArtifactEligibilityError, match="protocol_id"):
        save_pretrained(
            model,
            tmp_path / "empty-summary",
            measurement_summary=MeasurementSummary(),
            claim_level="benchmark-reproducible",
        )


def test_benchmark_claim_requires_adapter_revision(tmp_path: Path) -> None:
    summary = MeasurementSummary(
        protocol_id="protocol-v1",
        evaluation_id="evaluation-v1",
        dataset_ids=("fixture:test",),
        quality={"pair_f1": 0.9},
    )
    model = _transformer_topology(
        revision="b" * 40,
        adapter="acme/adapter",
    )
    with pytest.raises(ArtifactEligibilityError, match="commit SHA"):
        save_pretrained(
            model,
            tmp_path / "unpinned-adapter",
            measurement_summary=summary,
            claim_level="benchmark-reproducible",
        )


def test_benchmark_claim_rejects_local_adapter_even_with_sha_like_revision(
    tmp_path: Path,
) -> None:
    summary = MeasurementSummary(
        protocol_id="protocol-v1",
        evaluation_id="evaluation-v1",
        dataset_ids=("fixture:test",),
        quality={"pair_f1": 0.9},
    )
    model = _transformer_topology(
        revision="a" * 40,
        adapter="./private-local-adapter",
        adapter_revision="b" * 40,
    )
    with pytest.raises(ArtifactEligibilityError, match="organization/repository"):
        save_pretrained(
            model,
            tmp_path / "local-adapter",
            measurement_summary=summary,
            claim_level="benchmark-reproducible",
        )


def test_nested_legacy_architecture_resources_are_published(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("litellm")
    model = VectorLLMCascade(
        schema=Entity,
        embedder=ModelRef(
            base="sentence-transformers/all-MiniLM-L6-v2",
            kind="hf",
            revision="e" * 40,
        ),
        llm=ModelRef(base="openai/gpt-5-mini", kind="api"),
    )
    manifest = validate_bundle(
        save_pretrained(
            model,
            tmp_path / "nested",
            allow_sensitive_config=True,
        )
    )

    assert {ref.base for ref in manifest.model_refs} == {
        "sentence-transformers/all-MiniLM-L6-v2",
        "openai/gpt-5-mini",
    }
    assert manifest.required_extras == ("llm", "semantic")
    assert manifest.sensitive_config_included is True


@pytest.mark.parametrize(
    ("model_ref", "expected_extras"),
    [
        (ModelRef(base="openai/gpt-5-mini", kind="api"), ("llm",)),
        (
            ModelRef(base="acme/local-llm", kind="hf", revision="a" * 40),
            ("llm", "semantic"),
        ),
        (ModelRef(base="./local-llm", kind="local"), ("llm", "semantic")),
        (
            ModelRef(
                base="acme/base",
                kind="hf",
                revision="a" * 40,
                adapter="acme/adapter",
                adapter_revision="b" * 40,
            ),
            ("finetune", "llm", "semantic"),
        ),
    ],
)
def test_legacy_llm_extras_follow_runtime_route_and_adapter(
    model_ref: ModelRef,
    expected_extras: tuple[str, ...],
    tmp_path: Path,
) -> None:
    pytest.importorskip("litellm")
    from langres.core.matchers.llm_judge import LLMMatcher

    model = ERModel.from_schema(Entity, matcher=LLMMatcher(model=model_ref))
    manifest = validate_bundle(
        save_pretrained(
            model,
            tmp_path / model_ref.kind,
            allow_sensitive_config=True,
        )
    )
    assert manifest.required_extras == expected_extras


def test_transformers_resource_adapter_requires_finetune_extra(tmp_path: Path) -> None:
    model = _transformer_topology(
        revision="a" * 40,
        adapter="acme/adapter",
        adapter_revision="b" * 40,
    )
    manifest = validate_bundle(save_pretrained(model, tmp_path / "adapter"))
    assert manifest.required_extras == ("finetune", "semantic")


def test_benchmark_claim_rejects_moving_hf_revisions(tmp_path: Path) -> None:
    summary = MeasurementSummary(
        protocol_id="protocol-v1",
        evaluation_id="evaluation-v1",
        dataset_ids=("fixture:test",),
        quality={"pair_f1": 0.9},
    )
    with pytest.raises(ArtifactEligibilityError, match="commit SHA"):
        save_pretrained(
            _transformer_topology(revision="main"),
            tmp_path / "moving",
            measurement_summary=summary,
            claim_level="benchmark-reproducible",
        )


def test_incompatible_minor_version_is_rejected(model: ERModel, tmp_path: Path) -> None:
    bundle = save_pretrained(model, tmp_path / "bundle")
    manifest_path = bundle / BUNDLE_MANIFEST
    payload = json.loads(manifest_path.read_text())
    payload["langres_version"] = "99.0.0"
    payload["langres_compatibility"] = ">=99.0.0,<99.1.0"
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(PretrainedArtifactError, match="installed version"):
        from_pretrained(bundle)


def test_malformed_compatibility_is_rejected(model: ERModel, tmp_path: Path) -> None:
    bundle = save_pretrained(model, tmp_path / "bundle")
    manifest_path = bundle / BUNDLE_MANIFEST
    payload = json.loads(manifest_path.read_text())
    payload["langres_compatibility"] = "*"
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(PretrainedArtifactError, match="langres_compatibility"):
        from_pretrained(bundle)
