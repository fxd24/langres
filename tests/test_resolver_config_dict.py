"""S2 — ``Resolver.config_dict()``: in-memory config snapshot, no disk I/O.

``config_dict()`` factors the in-memory :class:`ArtifactManifest` assembly out of
:meth:`Resolver.save` so the experiment-tracking layer can snapshot a pipeline's
declared config (``RunContext.resolver_config``) **without** writing an artifact.
These tests lock the S2 acceptance invariants:

1. ``config_dict()`` == the dict ``save()`` manifests (equal to ``resolver.json``),
   and it round-trips through the registry contract.
2. ``config_dict()`` writes **nothing** to disk.
3. It composes with the existing ``save``/``load`` round-trip unchanged.
4. An unserializable slot component raises ``TypeError`` (NOT swallowed here —
   the best-effort catch lives in the tracking capture path, a different stream).
"""

import json
from pathlib import Path

import pytest


def _company_resolver() -> "object":
    """A 4-slot Resolver over built-in, registered components."""
    from langres.core import (
        AllPairsBlocker,
        Clusterer,
        Comparator,
        Resolver,
        WeightedAverageJudge,
    )
    from langres.core.models import CompanySchema

    comparator = Comparator.from_schema(CompanySchema)
    return Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=comparator,
        module=WeightedAverageJudge(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=0.7),
    )


def test_config_dict_matches_saved_manifest(tmp_path: Path) -> None:
    """The snapshot equals the exact dict ``save()`` writes to ``resolver.json``."""
    resolver = _company_resolver()
    snapshot = resolver.config_dict()  # type: ignore[attr-defined]

    resolver.save(tmp_path)  # type: ignore[attr-defined]
    on_disk = json.loads((tmp_path / "resolver.json").read_text())

    assert snapshot == on_disk


def test_config_dict_has_expected_shape() -> None:
    """Snapshot carries the artifact metadata + ordered slot specs."""
    import langres

    snapshot = _company_resolver().config_dict()  # type: ignore[attr-defined]

    assert snapshot["artifact_version"] == "1"
    assert snapshot["langres_version"] == langres.__version__
    assert [c["type_name"] for c in snapshot["components"]] == [
        "all_pairs_blocker",
        "comparator",
        "weighted_average_judge",
        "clusterer",
    ]
    assert [c["slot"] for c in snapshot["components"]] == [
        "blocker",
        "comparator",
        "module",
        "clusterer",
    ]
    # Clusterer config carries its threshold verbatim (declared config, no I/O).
    clusterer_spec = next(c for c in snapshot["components"] if c["slot"] == "clusterer")
    assert clusterer_spec["config"]["threshold"] == 0.7


def test_config_dict_is_json_serializable() -> None:
    """The snapshot is plain JSON data — the tracking layer hashes it via json.dumps."""
    snapshot = _company_resolver().config_dict()  # type: ignore[attr-defined]
    # Must not raise: recipe_id hashing canonicalizes this with json.dumps().
    round_tripped = json.loads(json.dumps(snapshot))
    assert round_tripped == snapshot


def test_config_dict_writes_nothing_to_disk(tmp_path: Path) -> None:
    """Building the snapshot creates no files — it is a pure in-memory read."""
    resolver = _company_resolver()

    snapshot = resolver.config_dict()  # type: ignore[attr-defined]

    assert list(tmp_path.iterdir()) == []  # target dir untouched
    assert not (tmp_path / "resolver.json").exists()
    assert snapshot["components"]  # returned a real, populated manifest


def test_config_dict_omits_absent_comparator() -> None:
    """``comparator=None`` yields a 3-slot snapshot, mirroring ``save()``'s ``_slots``."""
    from langres.core import AllPairsBlocker, Clusterer, Resolver, WeightedAverageJudge
    from langres.core.feature import FeatureSpec
    from langres.core.models import CompanySchema

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=WeightedAverageJudge(feature_specs=[FeatureSpec(name="name")]),
        clusterer=Clusterer(threshold=0.7),
    )

    assert [c["slot"] for c in resolver.config_dict()["components"]] == [
        "blocker",
        "module",
        "clusterer",
    ]


def test_config_dict_composes_with_save_load_round_trip(tmp_path: Path) -> None:
    """save/load still works and reloads to an identical snapshot (parity guard)."""
    from langres.core import Resolver
    from langres.core.serialization import ArtifactManifest

    resolver = _company_resolver()
    snapshot = resolver.config_dict()  # type: ignore[attr-defined]

    # The snapshot validates as the very contract load() reads.
    assert ArtifactManifest.model_validate(snapshot).artifact_version == "1"

    resolver.save(tmp_path)  # type: ignore[attr-defined]
    reloaded = Resolver.load(tmp_path)
    assert reloaded.config_dict() == snapshot


def test_config_dict_raises_on_unserializable_component() -> None:
    """An unregistered slot component raises ``TypeError`` (not swallowed here)."""
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    resolver = Resolver.from_schema(CompanySchema)

    class _UnregisteredModule:
        """Stands in for any component lacking ``type_name``/@register."""

    resolver.module = _UnregisteredModule()  # type: ignore[assignment]

    with pytest.raises(TypeError, match="_UnregisteredModule is not serializable"):
        resolver.config_dict()
