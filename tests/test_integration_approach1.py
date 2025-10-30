r"""End-to-end integration test for Approach 1: Classical String Matching.

This test validates the complete pipeline:
1. AllPairsBlocker generates candidates
2. RapidfuzzModule scores pairs
3. Clusterer forms clusters
4. BCubed F1 >= 0.70 (POC success criterion)
"""

import logging

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.metrics import calculate_bcubed_metrics
from langres.core.models import CompanySchema
from langres.core.modules.rapidfuzz import RapidfuzzModule
from tests.fixtures.companies import COMPANY_RECORDS, EXPECTED_DUPLICATE_GROUPS

logger = logging.getLogger(__name__)


def test_approach1_end_to_end_pipeline():
    """Test the complete Approach 1 pipeline achieves BCubed F1 >= 0.70."""

    # 1. Schema factory for companies
    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(
            id=record["id"],
            name=record["name"],
            address=record.get("address"),
            phone=record.get("phone"),
            website=record.get("website"),
        )

    # 2. AllPairsBlocker: Generate all candidate pairs
    blocker = AllPairsBlocker(schema_factory=company_factory)
    candidates = blocker.stream(COMPANY_RECORDS)

    # 3. RapidfuzzModule: Score pairs with weighted field similarity
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 0.6),
            "address": (lambda x: x.address or "", 0.2),
            "phone": (lambda x: x.phone or "", 0.1),
            "website": (lambda x: x.website or "", 0.1),
        },
        threshold=0.7,
        algorithm="token_sort_ratio",
    )

    judgements = list(module.forward(candidates))

    # 4. Clusterer: Form entity clusters
    clusterer = Clusterer(threshold=0.7)
    predicted_clusters = clusterer.cluster(judgements)

    # 5. Calculate BCubed metrics
    metrics = calculate_bcubed_metrics(predicted_clusters, EXPECTED_DUPLICATE_GROUPS)

    logger.info("\nApproach 1 Results:")
    logger.info("  BCubed Precision: %.3f", metrics["precision"])
    logger.info("  BCubed Recall:    %.3f", metrics["recall"])
    logger.info("  BCubed F1:        %.3f", metrics["f1"])
    logger.info("\nPredicted clusters: %d", len(predicted_clusters))
    logger.info("Gold clusters:      %d", len(EXPECTED_DUPLICATE_GROUPS))

    # Verify BCubed F1 >= 0.70 (POC success criterion)
    assert metrics["f1"] >= 0.70, (
        f"BCubed F1 {metrics['f1']:.3f} is below target 0.70. "
        f"Approach 1 does not meet POC success criteria."
    )


def test_approach1_pipeline_components():
    """Test that pipeline components work together correctly."""

    # Schema factory
    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(
            id=record["id"],
            name=record["name"],
            address=record.get("address"),
        )

    # Test data: 3 companies with 1 duplicate pair
    data = [
        {"id": "c1", "name": "Acme Corporation", "address": "123 Main St"},
        {"id": "c2", "name": "Acme Corp", "address": "123 Main Street"},
        {"id": "c3", "name": "TechStart Industries", "address": "456 Oak Ave"},
    ]

    # 1. Blocker generates candidates
    blocker = AllPairsBlocker(schema_factory=company_factory)
    candidates = list(blocker.stream(data))
    assert len(candidates) == 3  # C(3,2) = 3 pairs

    # 2. Module scores pairs
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 0.7),
            "address": (lambda x: x.address or "", 0.3),
        },
        threshold=0.5,
    )
    judgements = list(module.forward(iter(candidates)))
    assert len(judgements) == 3

    # 3. Clusterer forms clusters
    clusterer = Clusterer(threshold=0.7)
    clusters = clusterer.cluster(judgements)

    # Should have 2 clusters: {c1, c2} and {c3}
    # (c1, c2 should cluster due to high name similarity)
    assert len(clusters) >= 1  # At least one cluster should form


def test_approach1_perfect_matches():
    """Test pipeline with perfect matches (all exact duplicates)."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    # All records are exact duplicates
    data = [
        {"id": "c1", "name": "Acme Corporation"},
        {"id": "c2", "name": "Acme Corporation"},
        {"id": "c3", "name": "Acme Corporation"},
    ]

    blocker = AllPairsBlocker(schema_factory=company_factory)
    module = RapidfuzzModule(field_extractors={"name": (lambda x: x.name, 1.0)}, threshold=0.5)
    clusterer = Clusterer(threshold=0.9)

    candidates = blocker.stream(data)
    judgements = list(module.forward(candidates))
    clusters = clusterer.cluster(judgements)

    # All should cluster together (perfect matches)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_approach1_no_matches():
    """Test pipeline with no matches (all different companies)."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    # All records are completely different
    data = [
        {"id": "c1", "name": "Acme Corporation"},
        {"id": "c2", "name": "TechStart Industries"},
        {"id": "c3", "name": "DataFlow Solutions"},
    ]

    blocker = AllPairsBlocker(schema_factory=company_factory)
    module = RapidfuzzModule(field_extractors={"name": (lambda x: x.name, 1.0)}, threshold=0.5)
    clusterer = Clusterer(threshold=0.7)

    candidates = blocker.stream(data)
    judgements = list(module.forward(candidates))
    clusters = clusterer.cluster(judgements)

    # No clusters should form (all below threshold)
    # Each entity is its own singleton cluster (or no clusters if we only return multi-entity clusters)
    # Actually, Clusterer only returns connected components, so no edges = no clusters
    assert len(clusters) == 0  # No pairs meet the threshold
