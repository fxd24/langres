r"""M0 EXIT TEST — Resolver save/load round-trip (written RED, marked xfail).

This test pins the TARGET Resolver API that lands in Wave 3. It is marked
``xfail`` so the suite stays GREEN now; Wave 3 removes the xfail once Resolver,
the concrete Comparator, and WeightedAverageJudge exist.

The target API (none of this exists yet — imports live inside the test bodies
so collection does not error before Wave 3):

    from langres.core import (
        Resolver, AllPairsBlocker, Comparator, WeightedAverageJudge, Clusterer,
    )
    from langres.core.models import CompanySchema

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=Comparator.from_schema(CompanySchema),
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


@pytest.mark.xfail(reason="Resolver lands in Wave 3", strict=False)
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
        comparator=Comparator.from_schema(CompanySchema),
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
    assert "artifact_version" in manifest
    assert manifest["langres_version"] == langres.__version__
    for component in manifest["components"]:
        assert "type_name" in component
        assert "config" in component
    assert reloaded.clusterer.threshold == 0.7


@pytest.mark.xfail(reason="Resolver lands in Wave 3", strict=False)
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
        comparator=Comparator.from_schema(CompanySchema),
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


@pytest.mark.xfail(reason="Resolver lands in Wave 3", strict=False)
def test_resolver_from_schema_one_liner(tmp_path: Path) -> None:
    """Convenience constructor: Resolver.from_schema(CompanySchema, threshold=...)."""
    from langres.core import Resolver
    from langres.core.models import CompanySchema

    resolver = Resolver.from_schema(CompanySchema, threshold=0.7)
    clusters = resolver.resolve(COMPANY_RECORDS)
    metrics = calculate_bcubed_metrics(clusters, EXPECTED_DUPLICATE_GROUPS)
    assert metrics["f1"] >= 0.70
    assert resolver.clusterer.threshold == 0.7
