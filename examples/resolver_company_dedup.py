"""Resolver north-star example: company deduplication with save/load.

Run it:

    uv run --env-file .env python examples/resolver_company_dedup.py

What it shows (the ergonomic path):

1. Build a full dedup pipeline from a schema in ONE line via
   ``Resolver.from_schema`` — declarative blocker + missing-aware comparator +
   heuristic scorer + clusterer, with ``id`` excluded and weights baked in.
2. Resolve the company fixture into entity clusters.
3. Persist the whole pipeline to ``artifacts/company_v0`` (a human-readable
   ``resolver.json`` — no pickle, no code execution).
4. Reload it and re-resolve — the clustering is identical.
5. Report BCubed F1 against the known duplicate groups.

``print`` is allowed in examples (this is demonstration, not library code).
"""

from pathlib import Path

from langres.core import Resolver
from langres.core.metrics import calculate_bcubed_metrics
from langres.core.models import CompanySchema
from tests.fixtures.companies import COMPANY_RECORDS, EXPECTED_DUPLICATE_GROUPS

# Name-dominant weights (Approach 1): name carries enough evidence on its own to
# recover the name-only "missing fields" duplicate group (c4 / c4_partial).
NAME_DOMINANT_WEIGHTS = {"name": 0.6, "address": 0.2, "phone": 0.1, "website": 0.1}

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "company_v0"


def _print_clusters(title: str, clusters: list[set[str]]) -> None:
    print(f"\n{title}")
    for cluster in sorted(sorted(c) for c in clusters):
        print(f"  {cluster}")


def main() -> None:
    # 1. One-line pipeline: blocker + comparator + scorer + clusterer.
    resolver = Resolver.from_schema(CompanySchema, threshold=0.7, weights=NAME_DOMINANT_WEIGHTS)

    # 2. Resolve raw records into entity clusters (singletons dropped).
    clusters = resolver.resolve(COMPANY_RECORDS)
    _print_clusters("Clusters (fresh resolver):", clusters)

    # 3. Persist the whole pipeline (human-readable JSON manifest).
    resolver.save(ARTIFACT_DIR)
    print(f"\nSaved artifact to: {ARTIFACT_DIR}")
    print("resolver.json (first lines):")
    for line in (ARTIFACT_DIR / "resolver.json").read_text().splitlines()[:12]:
        print(f"  {line}")

    # 4. Reload and re-resolve — must be identical.
    reloaded = Resolver.load(ARTIFACT_DIR)
    clusters_reloaded = reloaded.resolve(COMPANY_RECORDS)
    _print_clusters("Clusters (reloaded resolver):", clusters_reloaded)

    canon = frozenset(frozenset(c) for c in clusters)
    canon_reloaded = frozenset(frozenset(c) for c in clusters_reloaded)
    assert canon == canon_reloaded, "Reloaded clustering differs from the original!"
    print("\nReload round-trip: clustering is IDENTICAL ✓")

    # 5. Accuracy against the known duplicate groups.
    metrics = calculate_bcubed_metrics(clusters_reloaded, EXPECTED_DUPLICATE_GROUPS)
    print(
        f"\nBCubed — precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
