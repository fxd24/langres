"""S2 — ``Resolver.config_dict()``: hash-safe config snapshot, no disk I/O.

``config_dict()`` factors the in-memory :class:`ArtifactManifest` assembly out of
:meth:`Resolver.save` so the experiment-tracking layer can snapshot a pipeline's
declared config (``RunContext.resolver_config``) **without** writing an artifact.
These tests lock its acceptance invariants:

1. ``config_dict()`` returns only the reproducible ``components`` payload and
   **omits** the volatile version/provenance envelope (``artifact_version`` /
   ``langres_version``) that ``save()`` still writes to ``resolver.json``. Because
   the snapshot feeds ``RunContext.resolver_config`` — inside
   :func:`compute_recipe_id`'s hash domain — the resulting ``recipe_id`` must be
   **stable across a package / artifact-schema version bump** (idempotent replay).
2. ``config_dict()`` writes **nothing** to disk.
3. It composes with the existing ``save``/``load`` round-trip unchanged, and
   ``save()`` still records the version provenance on disk.
4. An unserializable slot component raises ``TypeError`` (NOT swallowed here —
   the best-effort catch lives in the tracking capture path, a different stream).
"""

import json
from pathlib import Path

import pytest


def _company_resolver() -> "object":
    """A 4-slot Resolver over built-in, registered components."""
    from langres.core import Clusterer, Resolver
    from langres.core.comparators import StringComparator
    from langres.core.blockers import AllPairsBlocker
    from langres.core.matchers import WeightedAverageMatcher
    from langres.core.models import CompanySchema

    comparator = StringComparator.from_schema(CompanySchema)
    return Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=comparator,
        matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=0.7),
    )


def test_config_dict_components_match_saved_manifest_minus_version(tmp_path: Path) -> None:
    """The snapshot's ``components`` equal ``resolver.json``'s, minus the version envelope.

    ``config_dict()`` is the reproducible config only: it carries the same ordered
    component specs ``save()`` writes, but omits the volatile ``artifact_version``
    / ``langres_version`` fields (those stay on disk / on the RunContext, unhashed).
    """
    resolver = _company_resolver()
    snapshot = resolver.config_dict()  # type: ignore[attr-defined]

    resolver.save(tmp_path)  # type: ignore[attr-defined]
    on_disk = json.loads((tmp_path / "resolver.json").read_text())

    # Same reproducible payload...
    assert snapshot == {"components": on_disk["components"]}
    # ...but the snapshot deliberately drops the version/provenance envelope.
    assert "artifact_version" not in snapshot
    assert "langres_version" not in snapshot


def test_config_dict_has_expected_shape() -> None:
    """Snapshot carries the ordered slot specs and NO version/provenance envelope."""
    snapshot = _company_resolver().config_dict()  # type: ignore[attr-defined]

    # Hash-safe: only the reproducible component payload, no volatile version keys.
    assert set(snapshot.keys()) == {"components"}
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
    from langres.core import Clusterer, Resolver
    from langres.core.blockers import AllPairsBlocker
    from langres.core.matchers import WeightedAverageMatcher
    from langres.core.feature import FeatureSpec
    from langres.core.models import CompanySchema

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=WeightedAverageMatcher(feature_specs=[FeatureSpec(name="name")]),
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

    resolver.save(tmp_path)  # type: ignore[attr-defined]

    # The written artifact validates as the full contract load() reads (with the
    # version envelope config_dict() omits).
    manifest = ArtifactManifest.model_validate_json((tmp_path / "resolver.json").read_text())
    assert manifest.artifact_version == "1"

    reloaded = Resolver.load(tmp_path)
    assert reloaded.config_dict() == snapshot


def test_config_dict_recipe_id_stable_across_version_bump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """recipe_id must NOT change when ``langres``/``ARTIFACT_VERSION`` bump.

    This is the core hygiene invariant: ``config_dict()`` feeds
    ``RunContext.resolver_config``, which is inside ``compute_recipe_id``'s hash
    domain. If the snapshot leaked the version envelope, a package or
    artifact-schema bump would silently fork ``recipe_id`` and break idempotent
    replay. Version/provenance live on the RunContext, unhashed.
    """
    from langres.core import _model_persist as persist_mod
    from langres.core.runs import RunContext, compute_recipe_id

    resolver = _company_resolver()

    def _recipe_id() -> str:
        context = RunContext(
            experiment="exp",
            dataset_name="ds",
            resolver_config=resolver.config_dict(),  # type: ignore[attr-defined]
        )
        return compute_recipe_id(context)

    baseline = _recipe_id()

    # Simulate a package release + artifact-schema bump. Both constants are
    # patched on `langres.core._model_persist` -- where they are *looked up* --
    # because that is where `_build_manifest` reads them from. (It read them from
    # `core.resolver` until the model layer was split by responsibility; the
    # persistence half, and with it the version envelope, moved there. The patch
    # target follows the lookup, always.) Patching `langres.__version__` instead
    # would no-op silently (the manifest binds the version from the
    # `langres._version` leaf, not through the root) and this test would pass
    # while simulating nothing.
    monkeypatch.setattr(persist_mod, "LANGRES_VERSION", "999.999.999")
    monkeypatch.setattr(persist_mod, "ARTIFACT_VERSION", "999")

    assert _recipe_id() == baseline


def test_save_manifest_still_carries_version_provenance(tmp_path: Path) -> None:
    """``save()`` STILL writes ``artifact_version`` + ``langres_version`` to disk.

    The hash-safe ``config_dict()`` drops them, but the on-disk artifact must keep
    them for reconstruction / ``_check_versions`` (unchanged provenance).
    """
    import langres

    resolver = _company_resolver()
    resolver.save(tmp_path)  # type: ignore[attr-defined]
    on_disk = json.loads((tmp_path / "resolver.json").read_text())

    assert on_disk["artifact_version"] == "1"
    assert on_disk["langres_version"] == langres.__version__


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
