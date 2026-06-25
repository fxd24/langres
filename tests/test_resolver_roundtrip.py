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

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=Comparator.from_schema(CompanySchema, weights=NAME_DOMINANT_WEIGHTS),
        module=WeightedAverageJudge(),   # the scorer slot, typed Module
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

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=Comparator.from_schema(CompanySchema, weights=NAME_DOMINANT_WEIGHTS),
        module=WeightedAverageJudge(),
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
    assert manifest["artifact_version"] == "0"
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

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=Comparator.from_schema(CompanySchema, weights=NAME_DOMINANT_WEIGHTS),
        module=WeightedAverageJudge(),
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

    resolver = Resolver.from_schema(
        CompanySchema, threshold=0.7, weights=NAME_DOMINANT_WEIGHTS
    )
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
