r"""M0 EXIT TEST — Resolver save/load round-trip (GREEN, Wave 3).

The Resolver, the concrete Comparator, and WeightedAverageJudge exist, so the
xfail markers are gone and these tests assert the real behavior.

The 4-slot resolver is built with **name-dominant weights** matching Approach 1
(``name`` 0.6 / ``address`` 0.2 / ``phone`` 0.1 / ``website`` 0.1). Equal
weights would gate out the name-only ``c4``/``c4_partial`` pair via the evidence
floor (a single present feature at weight 0.25 < 0.5); name at 0.6 clears the
floor so the missing-fields group is recovered. The bare
``Resolver.from_schema(CompanySchema)`` one-liner (equal weights) is covered
separately and only asserts the >= 0.70 accuracy floor (it need not recover c4).

    comparator = Comparator.from_schema(CompanySchema, weights=NAME_DOMINANT_WEIGHTS)
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=comparator,
        module=WeightedAverageJudge(feature_specs=comparator.feature_specs),  # scorer slot
        clusterer=Clusterer(threshold=0.7),
    )
    clusters_before = resolver.resolve(COMPANY_RECORDS)
    resolver.save(tmp_path)
    reloaded = Resolver.load(tmp_path)
    clusters_after = reloaded.resolve(COMPANY_RECORDS)
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from langres.core.metrics import calculate_bcubed_metrics
from tests.fixtures.companies import COMPANY_RECORDS, EXPECTED_DUPLICATE_GROUPS

# Name-dominant weights mirroring Approach 1 — required for the name-only
# missing-fields group (c4/c4_partial) to clear the evidence floor.
NAME_DOMINANT_WEIGHTS = {"name": 0.6, "address": 0.2, "phone": 0.1, "website": 0.1}


def _canonical(clusters: list[set[str]]) -> frozenset[frozenset[str]]:
    """Order-independent canonical form of a clustering for equality checks."""
    return frozenset(frozenset(c) for c in clusters)


def _wrongly_merged_pairs(
    predicted: list[set[str]], gold_groups: list[set[str]]
) -> list[tuple[str, str]]:
    """Pairs co-clustered in a prediction that are NOT in the same gold group.

    Reasons in pair-space over IDs (the Clusterer drops singletons, so length
    bands are meaningless). An entity not appearing in any gold group is treated
    as its own singleton group, so co-clustering it with anything is "wrong".
    """
    id_to_gold: dict[str, int] = {}
    for i, group in enumerate(gold_groups):
        for member in group:
            id_to_gold[member] = i

    next_singleton = len(gold_groups)
    wrong: list[tuple[str, str]] = []
    for cluster in predicted:
        members = sorted(cluster)
        for a_idx, a in enumerate(members):
            for b in members[a_idx + 1 :]:
                ga = id_to_gold.get(a)
                if ga is None:
                    ga = next_singleton
                    next_singleton += 1
                gb = id_to_gold.get(b, None)
                if gb is None:
                    gb = next_singleton
                    next_singleton += 1
                if ga != gb:
                    wrong.append((a, b))
    return wrong


def test_resolver_roundtrip_in_process(tmp_path: Path) -> None:
    """A-D: in-process save/load round-trip, accuracy, over-merge, provenance."""
    from langres.core import (
        AllPairsBlocker,
        Clusterer,
        Comparator,
        Resolver,
        WeightedAverageJudge,
    )
    from langres.core.models import CompanySchema

    comparator = Comparator.from_schema(CompanySchema, weights=NAME_DOMINANT_WEIGHTS)
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=comparator,
        module=WeightedAverageJudge(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=0.7),
    )

    clusters_before = resolver.resolve(COMPANY_RECORDS)

    resolver.save(tmp_path)
    reloaded = Resolver.load(tmp_path)
    clusters_after = reloaded.resolve(COMPANY_RECORDS)

    # A. Identical clustering before vs after reload (canonicalize BOTH sides).
    assert _canonical(clusters_before) == _canonical(clusters_after)

    # B. Accuracy floor.
    metrics = calculate_bcubed_metrics(clusters_after, EXPECTED_DUPLICATE_GROUPS)
    assert metrics["f1"] >= 0.70

    # C. Structural over-merge check (pair-space, not length band).
    assert _wrongly_merged_pairs(clusters_after, EXPECTED_DUPLICATE_GROUPS) == []

    # D. Artifact provenance.
    import langres

    manifest = json.loads((tmp_path / "resolver.json").read_text())
    assert manifest["artifact_version"] == "1"
    assert manifest["langres_version"] == langres.__version__
    type_names = [component["type_name"] for component in manifest["components"]]
    assert type_names == [
        "all_pairs_blocker",
        "comparator",
        "weighted_average_judge",
        "clusterer",
    ]
    for component in manifest["components"]:
        assert "type_name" in component
        assert "config" in component
    # Clusterer config round-trips its threshold exactly.
    clusterer_spec = next(c for c in manifest["components"] if c["type_name"] == "clusterer")
    assert clusterer_spec["config"]["threshold"] == 0.7
    assert reloaded.clusterer.threshold == 0.7


def test_resolver_roundtrip_fresh_process(tmp_path: Path) -> None:
    """E: reload in a fresh subprocess to catch registry/import side-effects."""
    from langres.core import (
        AllPairsBlocker,
        Clusterer,
        Comparator,
        Resolver,
        WeightedAverageJudge,
    )
    from langres.core.models import CompanySchema

    comparator = Comparator.from_schema(CompanySchema, weights=NAME_DOMINANT_WEIGHTS)
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=comparator,
        module=WeightedAverageJudge(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=0.7),
    )
    clusters_before = resolver.resolve(COMPANY_RECORDS)
    resolver.save(tmp_path)

    script = (
        "import json, sys\n"
        "from langres.core import Resolver\n"
        "from tests.fixtures.companies import COMPANY_RECORDS\n"
        f"reloaded = Resolver.load({str(tmp_path)!r})\n"
        "clusters = reloaded.resolve(COMPANY_RECORDS)\n"
        "out = sorted(sorted(c) for c in clusters)\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    clusters_subprocess = [set(c) for c in json.loads(proc.stdout)]
    assert _canonical(clusters_before) == _canonical(clusters_subprocess)


def test_resolver_from_schema_one_liner() -> None:
    """Convenience constructor: Resolver.from_schema(CompanySchema, threshold=...).

    Uses the bare (equal-weight) comparator: it need not recover the name-only
    c4/c4_partial group, only clear the >= 0.70 accuracy floor.
    """
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    resolver = Resolver.from_schema(CompanySchema, threshold=0.7)
    clusters = resolver.resolve(COMPANY_RECORDS)
    metrics = calculate_bcubed_metrics(clusters, EXPECTED_DUPLICATE_GROUPS)
    assert metrics["f1"] >= 0.70
    assert resolver.clusterer.threshold == 0.7
    # No over-merging even on the bare path.
    assert _wrongly_merged_pairs(clusters, EXPECTED_DUPLICATE_GROUPS) == []


def test_resolver_from_schema_name_dominant_recovers_all_groups() -> None:
    """Name-dominant weights via from_schema recover the missing-fields group."""
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    resolver = Resolver.from_schema(CompanySchema, threshold=0.7, weights=NAME_DOMINANT_WEIGHTS)
    clusters = resolver.resolve(COMPANY_RECORDS)
    # c4/c4_partial (name-only) is recovered: perfect BCubed on this fixture.
    assert calculate_bcubed_metrics(clusters, EXPECTED_DUPLICATE_GROUPS)["f1"] == 1.0


def test_resolver_predict_returns_judgements() -> None:
    """predict() exposes the scored judgements before clustering (observability)."""
    from langres.core import Resolver
    from langres.core.models import CompanySchema, PairwiseJudgement

    resolver = Resolver.from_schema(CompanySchema, weights=NAME_DOMINANT_WEIGHTS)
    judgements = resolver.predict(COMPANY_RECORDS)
    assert all(isinstance(j, PairwiseJudgement) for j in judgements)
    # AllPairs over 15 records -> 15*14/2 = 105 pairs.
    assert len(judgements) == 105


def test_resolver_fit_is_noop_returns_self() -> None:
    """fit() is a no-op that returns self (sklearn convention; optimization is M3+)."""
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    resolver = Resolver.from_schema(CompanySchema)
    assert resolver.fit(COMPANY_RECORDS) is resolver


def test_resolver_load_rejects_newer_artifact(tmp_path: Path) -> None:
    """A strictly-newer artifact_version is a hard error on load."""
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    Resolver.from_schema(CompanySchema).save(tmp_path)
    manifest_path = tmp_path / "resolver.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_version"] = "99"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="newer than this langres build"):
        Resolver.load(tmp_path)


def test_resolver_load_rejects_older_artifact(tmp_path: Path) -> None:
    """A strictly-older artifact_version is a clean error, not a raw KeyError.

    M0.5 bumped ARTIFACT_VERSION to "1" with an incompatible config schema. An
    artifact written by pre-M0.5 code (version "0", old configs) must fail with
    the clean version error rather than falling through to a KeyError when
    rebuilding components from the changed config.
    """
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    Resolver.from_schema(CompanySchema).save(tmp_path)
    manifest_path = tmp_path / "resolver.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_version"] = "0"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="predates the supported layout"):
        Resolver.load(tmp_path)


def test_resolver_save_rejects_unserializable_component(tmp_path: Path) -> None:
    """save() raises a clear TypeError naming an unregistered slot component.

    A component without a registry ``type_name`` (e.g. CascadeModule in the
    module slot) must fail with a readable message instead of an opaque
    AttributeError from the spec/save path.
    """
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    resolver = Resolver.from_schema(CompanySchema)

    class _UnregisteredModule:
        """Stands in for any component lacking `type_name`/@register."""

    resolver.module = _UnregisteredModule()  # type: ignore[assignment]

    with pytest.raises(TypeError, match="_UnregisteredModule is not serializable"):
        resolver.save(tmp_path)


def test_resolver_load_warns_on_langres_version_skew(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A langres_version mismatch warns but still loads (forward-compatible)."""
    import logging

    from langres.core import Resolver
    from langres.core.models import CompanySchema

    Resolver.from_schema(CompanySchema).save(tmp_path)
    manifest_path = tmp_path / "resolver.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["langres_version"] = "0.0.1"
    manifest_path.write_text(json.dumps(manifest))

    with caplog.at_level(logging.WARNING):
        reloaded = Resolver.load(tmp_path)
    assert reloaded.clusterer.threshold == 0.7
    assert any("0.0.1" in rec.message for rec in caplog.records)


def _vector_resolver() -> "object":
    """Build a Resolver over a VectorBlocker + FAISSIndex (FakeEmbedder, fast)."""
    from langres.core import (
        Clusterer,
        Comparator,
        FAISSIndex,
        FakeEmbedder,
        Resolver,
        VectorBlocker,
        WeightedAverageJudge,
    )
    from langres.core.models import CompanySchema

    index = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")
    blocker: VectorBlocker[CompanySchema] = VectorBlocker(
        vector_index=index,
        schema=CompanySchema,
        text_field="name",
        k_neighbors=5,
    )
    comparator = Comparator.from_schema(CompanySchema, weights=NAME_DOMINANT_WEIGHTS)
    return Resolver(
        blocker=blocker,
        comparator=comparator,
        module=WeightedAverageJudge(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=0.7),
    )


def test_resolver_builds_vector_index_transparently() -> None:
    """resolve() builds an index-backed blocker's index without a manual call."""
    resolver = _vector_resolver()
    # The blocker's index starts unbuilt; resolve() must build it transparently.
    assert not resolver.blocker._index_is_built()  # type: ignore[attr-defined]
    clusters = resolver.resolve(COMPANY_RECORDS)
    assert resolver.blocker._index_is_built()  # type: ignore[attr-defined]
    # No over-merging from the vector path.
    assert _wrongly_merged_pairs(clusters, EXPECTED_DUPLICATE_GROUPS) == []


def test_resolver_rebuilds_vector_index_on_new_corpus() -> None:
    """Reusing a Resolver on a DIFFERENT record list rebuilds the index.

    An already-built index must not be reused against a new corpus — that would
    score the new records against the old corpus (wrong pairs / IndexError). The
    Resolver compares the new corpus texts to the index's stored ``_corpus_texts``
    and rebuilds when they differ.
    """
    from langres.core.models import CompanySchema

    resolver = _vector_resolver()

    records_a = [
        {"id": "a1", "name": "Acme Corp"},
        {"id": "a2", "name": "Acme Corp"},
        {"id": "a3", "name": "Beta LLC"},
    ]
    records_b = [
        {"id": "b1", "name": "Zenith Inc"},
        {"id": "b2", "name": "Zenith Inc"},
        {"id": "b3", "name": "Omega GmbH"},
    ]

    clusters_a = resolver.resolve(records_a)
    ids_a = {i for cluster in clusters_a for i in cluster}
    assert ids_a <= {r["id"] for r in records_a}

    # Reuse the same resolver on a disjoint corpus: must rebuild, no IndexError,
    # and the result references B's ids — never A's stale corpus.
    clusters_b = resolver.resolve(records_b)
    ids_b = {i for cluster in clusters_b for i in cluster}
    assert ids_b, "expected at least one B cluster"
    assert ids_b <= {r["id"] for r in records_b}
    assert ids_b.isdisjoint({r["id"] for r in records_a})
    # The index now holds B's corpus, not A's.
    expected_texts_b = [CompanySchema(**r).name for r in records_b]
    assert resolver.blocker.vector_index._corpus_texts == expected_texts_b  # type: ignore[attr-defined]


def test_resolver_roundtrip_with_faiss_state(tmp_path: Path) -> None:
    """An index-backed Resolver persists + restores its FAISS state (sidecar)."""
    from langres.core import Resolver

    resolver = _vector_resolver()
    clusters_before = resolver.resolve(COMPANY_RECORDS)

    resolver.save(tmp_path)
    # The blocker slot wrote a sidecar with the built FAISS index files.
    assert (tmp_path / "blocker" / "index.faiss").exists()
    assert (tmp_path / "blocker" / "corpus_embeddings.npy").exists()

    reloaded = Resolver.load(tmp_path)
    # Loaded index is already built (state restored) — resolve reuses it.
    assert reloaded.blocker._index_is_built()  # type: ignore[attr-defined]
    clusters_after = reloaded.resolve(COMPANY_RECORDS)
    assert _canonical(clusters_before) == _canonical(clusters_after)


def test_resolver_save_unbuilt_vector_index_writes_no_sidecar(tmp_path: Path) -> None:
    """Saving a VectorBlocker Resolver BEFORE the index is built writes no sidecar.

    The FAISS index has no state to persist until ``create_index`` runs, so
    ``save`` must not leave an empty ``blocker/`` dir behind — otherwise ``load``
    would try to read a missing ``index.faiss``. Loading the artifact succeeds,
    and resolve() then builds the index transparently.
    """
    from langres.core import Resolver

    resolver = _vector_resolver()
    # Do NOT resolve() first — the index is unbuilt.
    assert not resolver.blocker._index_is_built()  # type: ignore[attr-defined]

    resolver.save(tmp_path)
    # No empty/partial blocker sidecar was written.
    assert not (tmp_path / "blocker").exists()

    # Load succeeds (no missing-file error) and the reloaded index is unbuilt.
    reloaded = Resolver.load(tmp_path)
    assert not reloaded.blocker._index_is_built()  # type: ignore[attr-defined]
    # resolve() still works — it builds the index transparently.
    reloaded.resolve(COMPANY_RECORDS)
    assert reloaded.blocker._index_is_built()  # type: ignore[attr-defined]


def test_resolver_without_comparator_uses_plain_module() -> None:
    """comparator=None drives a self-contained Module directly (no compare stage)."""
    from collections.abc import Iterator

    from langres.core import AllPairsBlocker, Clusterer, Resolver
    from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
    from langres.core.module import Module
    from langres.core.reports import ScoreInspectionReport

    class ExactNameModule(Module[CompanySchema]):
        """Tiny self-contained scorer: 1.0 iff names are identical."""

        def forward(
            self, candidates: Iterator[ERCandidate[CompanySchema]]
        ) -> Iterator[PairwiseJudgement]:
            for pair in candidates:
                score = 1.0 if pair.left.name == pair.right.name else 0.0
                yield PairwiseJudgement(
                    left_id=pair.left.id,
                    right_id=pair.right.id,
                    score=score,
                    score_type="heuristic",
                    decision_step="exact_name",
                    provenance={},
                )

        def inspect_scores(
            self, judgements: list[PairwiseJudgement], sample_size: int = 10
        ) -> ScoreInspectionReport:  # pragma: no cover - not exercised here
            raise NotImplementedError

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=ExactNameModule(),
        clusterer=Clusterer(threshold=0.5),
    )
    clusters = resolver.resolve(COMPANY_RECORDS)
    # Only exact-name duplicates merge: c1/c1_dup1, c4/c4_partial, c5/c5_addr_var.
    canon = _canonical(clusters)
    assert frozenset({"c1", "c1_dup1"}) in canon
    assert frozenset({"c4", "c4_partial"}) in canon
    assert frozenset({"c5", "c5_addr_var"}) in canon


def test_resolver_save_without_comparator_omits_slot(tmp_path: Path) -> None:
    """A comparator=None Resolver writes a 3-component manifest (no comparator)."""
    from langres.core import AllPairsBlocker, Clusterer, Resolver, WeightedAverageJudge
    from langres.core.feature import FeatureSpec
    from langres.core.models import CompanySchema

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=WeightedAverageJudge(feature_specs=[FeatureSpec(name="name")]),
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.save(tmp_path)
    manifest = json.loads((tmp_path / "resolver.json").read_text())
    type_names = [c["type_name"] for c in manifest["components"]]
    assert type_names == ["all_pairs_blocker", "weighted_average_judge", "clusterer"]
    # Loads back with comparator=None.
    reloaded = Resolver.load(tmp_path)
    assert reloaded.comparator is None


def test_resolver_load_rejects_incompatible_string_version(tmp_path: Path) -> None:
    """A non-integer artifact_version that differs from supported is rejected."""
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    Resolver.from_schema(CompanySchema).save(tmp_path)
    manifest_path = tmp_path / "resolver.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_version"] = "1.0-beta"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="differs from supported"):
        Resolver.load(tmp_path)


def test_resolver_round_trips_comparator_subclass_with_custom_type_name(
    tmp_path: Path,
) -> None:
    """A registered Comparator subclass whose type_name is NOT "comparator" loads right.

    Regression for slot identification: load() must map specs to slots by the
    recorded slot name, not by a hard-coded ``type_name == "comparator"``. With
    positional/type_name identification, a comparator registered as
    "phonetic_comparator" was misread as the module and the real module was
    dropped. Here we assert the reloaded comparator IS the subclass AND the
    module slot is the real WeightedAverageJudge.
    """
    from langres.core import (
        AllPairsBlocker,
        Clusterer,
        Comparator,
        Resolver,
        WeightedAverageJudge,
        register,
    )
    from langres.core.feature import ComparisonLevel, ComparisonVector, FeatureSpec
    from langres.core.models import CompanySchema

    @register("phonetic_comparator")
    class PhoneticComparator(Comparator[CompanySchema]):
        """Trivial Comparator with a custom (non-"comparator") type_name."""

        type_name = "phonetic_comparator"

        def __init__(self, feature_specs: list[FeatureSpec] | None = None) -> None:
            self._specs = feature_specs or [FeatureSpec(name="name", weight=1.0)]

        @property
        def feature_specs(self) -> list[FeatureSpec]:
            return self._specs

        def compare(self, left: CompanySchema, right: CompanySchema) -> ComparisonVector:
            same = (left.name or "")[:1].lower() == (right.name or "")[:1].lower()
            return ComparisonVector(
                levels={"name": ComparisonLevel.PRESENT},
                similarities={"name": 1.0 if same else 0.0},
            )

        @property
        def config(self) -> dict[str, object]:
            return {"feature_specs": [s.model_dump() for s in self._specs]}

        @classmethod
        def from_config(cls, config: dict[str, object]) -> "PhoneticComparator":
            specs = [FeatureSpec.model_validate(s) for s in config["feature_specs"]]  # type: ignore[union-attr]
            return cls(feature_specs=specs)

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=PhoneticComparator(),
        module=WeightedAverageJudge(feature_specs=[FeatureSpec(name="name", weight=1.0)]),
        clusterer=Clusterer(threshold=0.5),
    )
    resolver.save(tmp_path)

    # Manifest records the slot names self-describingly.
    manifest = json.loads((tmp_path / "resolver.json").read_text())
    slots = {c["slot"]: c["type_name"] for c in manifest["components"]}
    assert slots == {
        "blocker": "all_pairs_blocker",
        "comparator": "phonetic_comparator",
        "module": "weighted_average_judge",
        "clusterer": "clusterer",
    }

    reloaded = Resolver.load(tmp_path)
    # The comparator slot is the custom subclass, NOT misread as the module.
    assert isinstance(reloaded.comparator, PhoneticComparator)
    # The module slot is the real module, NOT the comparator.
    assert isinstance(reloaded.module, WeightedAverageJudge)


def test_resolver_load_rejects_manifest_missing_required_slot(tmp_path: Path) -> None:
    """A slot-bearing manifest missing a required slot (blocker) is a hard error."""
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    Resolver.from_schema(CompanySchema, threshold=0.7).save(tmp_path)
    manifest_path = tmp_path / "resolver.json"
    manifest = json.loads(manifest_path.read_text())
    # Drop the blocker component entirely, keeping the others' slots intact.
    manifest["components"] = [c for c in manifest["components"] if c["slot"] != "blocker"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing required slot"):
        Resolver.load(tmp_path)


def test_resolver_loads_legacy_manifest_without_slot(tmp_path: Path) -> None:
    """A hand-written/legacy manifest with no ``slot`` still loads (positional fallback)."""
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    Resolver.from_schema(CompanySchema, threshold=0.7).save(tmp_path)
    manifest_path = tmp_path / "resolver.json"
    manifest = json.loads(manifest_path.read_text())
    # Strip the self-describing slot to simulate an older artifact.
    for component in manifest["components"]:
        component.pop("slot", None)
    manifest_path.write_text(json.dumps(manifest))

    reloaded = Resolver.load(tmp_path)
    assert reloaded.clusterer.threshold == 0.7
    assert reloaded.comparator is not None  # comparator slot recovered by type_name


def test_resolver_load_rejects_legacy_manifest_without_module(tmp_path: Path) -> None:
    """A legacy (slot-less) manifest with no identifiable module spec is rejected."""
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    Resolver.from_schema(CompanySchema, threshold=0.7).save(tmp_path)
    manifest_path = tmp_path / "resolver.json"
    manifest = json.loads(manifest_path.read_text())
    # Strip slots (legacy path) and keep only blocker + clusterer: with the
    # comparator and module gone, positional identification finds no module.
    kept = [
        c for c in manifest["components"] if c["type_name"] in ("all_pairs_blocker", "clusterer")
    ]
    for component in kept:
        component.pop("slot", None)
    manifest["components"] = kept
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="cannot identify a module spec"):
        Resolver.load(tmp_path)


def test_resolver_persists_module_that_owns_state_directly(tmp_path: Path) -> None:
    """A scorer Module that is itself SerializableState round-trips its state.

    Validates the *direct* (non-nested) state path: a slot component that both
    declares a BaseModel ``config()`` and implements ``SerializableState`` is
    saved to / restored from its own sidecar dir, exercising the
    config_model + direct-state branches of the Resolver's save/load helpers.
    """
    from collections.abc import Iterator
    from pathlib import Path as _Path

    from pydantic import BaseModel as _BaseModel

    from langres.core import AllPairsBlocker, Clusterer, Resolver, register
    from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
    from langres.core.module import Module
    from langres.core.reports import ScoreInspectionReport

    class _StatefulConfig(_BaseModel):
        base: float = 0.0

    @register("stateful_test_module")
    class StatefulModule(Module[CompanySchema]):
        """Scores every pair at ``base + bump``; ``bump`` is restored from state."""

        type_name = "stateful_test_module"
        config_model = _StatefulConfig

        def __init__(self, base: float = 0.0) -> None:
            self.base = base
            self.bump = 0.0

        def config(self) -> _StatefulConfig:
            return _StatefulConfig(base=self.base)

        @classmethod
        def from_config(cls, config: _StatefulConfig) -> "StatefulModule":
            return cls(base=config.base)

        def save_state(self, state_dir: _Path) -> None:
            (state_dir / "bump.txt").write_text(str(self.bump))

        def load_state(self, state_dir: _Path) -> None:
            self.bump = float((state_dir / "bump.txt").read_text())

        def forward(
            self, candidates: Iterator[ERCandidate[CompanySchema]]
        ) -> Iterator[PairwiseJudgement]:
            for pair in candidates:
                yield PairwiseJudgement(
                    left_id=pair.left.id,
                    right_id=pair.right.id,
                    score=min(1.0, self.base + self.bump),
                    score_type="heuristic",
                    decision_step="stateful",
                    provenance={},
                )

        def inspect_scores(
            self, judgements: list[PairwiseJudgement], sample_size: int = 10
        ) -> ScoreInspectionReport:  # pragma: no cover - not exercised
            raise NotImplementedError

    module = StatefulModule(base=0.4)
    module.bump = 0.6  # state to persist (would be lost without save_state)
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=module,
        clusterer=Clusterer(threshold=0.5),
    )
    resolver.save(tmp_path)
    assert (tmp_path / "module" / "bump.txt").exists()

    reloaded = Resolver.load(tmp_path)
    assert reloaded.module.base == 0.4  # type: ignore[attr-defined]
    assert reloaded.module.bump == 0.6  # type: ignore[attr-defined] - restored from state
    # Every pair now scores base+bump = 1.0 >= 0.5 -> all records collapse.
    clusters = reloaded.resolve(COMPANY_RECORDS)
    assert len(clusters) == 1


def test_resolver_round_trips_glinker_adapter_slot(tmp_path: Path) -> None:
    """An external adapter (Pydantic-model config attr) round-trips through save/load.

    Regression guard for the serialization seam: a component whose ``config`` is a
    plain instance attribute holding a Pydantic model (not a property/method) must
    serialize via its registered ``type_name`` and rebuild with matching config.
    The adapter sits in the blocker slot; save/load never calls its stubbed
    ``stream``/``forward``.
    """
    from langres.core import AllPairsBlocker, Clusterer, Resolver, WeightedAverageJudge
    from langres.core.adapters.glinker import GLinkerAdapter, GLinkerConfig
    from langres.core.feature import FeatureSpec
    from langres.core.models import CompanySchema

    adapter: GLinkerAdapter[CompanySchema] = GLinkerAdapter(
        GLinkerConfig(model_name="urchade/gliner_small-v2.1", threshold=0.42)
    )
    resolver = Resolver(
        blocker=adapter,
        comparator=None,
        module=WeightedAverageJudge(feature_specs=[FeatureSpec(name="name")]),
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.save(tmp_path)

    manifest = json.loads((tmp_path / "resolver.json").read_text())
    blocker_spec = manifest["components"][0]
    assert blocker_spec["type_name"] == "glinker_adapter"
    assert blocker_spec["config"] == {
        "model_name": "urchade/gliner_small-v2.1",
        "threshold": 0.42,
    }

    reloaded = Resolver.load(tmp_path)
    assert isinstance(reloaded.blocker, GLinkerAdapter)
    # config is now a plain dict (convention, matching every other component);
    # the underlying GLinkerConfig is on _config.
    assert reloaded.blocker.config == {
        "model_name": "urchade/gliner_small-v2.1",
        "threshold": 0.42,
    }
    assert reloaded.blocker._config.model_name == "urchade/gliner_small-v2.1"
    assert reloaded.blocker._config.threshold == 0.42
