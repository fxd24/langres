"""End-to-end integration test for Approach 1: Classical String Matching.

This test validates the complete pipeline:
1. AllPairsBlocker generates candidates
2. RapidfuzzModule scores pairs
3. Clusterer forms clusters
4. BCubed F1 >= 0.70 (POC success criterion)
"""

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.models import CompanySchema
from langres.core.modules.rapidfuzz import RapidfuzzModule
from tests.fixtures.companies import COMPANY_RECORDS, EXPECTED_DUPLICATE_GROUPS


def calculate_bcubed_precision(
    predicted_clusters: list[set[str]], gold_clusters: list[set[str]]
) -> float:
    """Calculate BCubed Precision.

    BCubed Precision measures how many items in each predicted cluster
    belong to the same gold cluster.

    Args:
        predicted_clusters: List of predicted entity clusters
        gold_clusters: List of gold-standard entity clusters

    Returns:
        BCubed Precision score in range [0.0, 1.0]
    """
    # Build gold cluster lookup: entity_id -> cluster_id
    gold_lookup: dict[str, int] = {}
    for cluster_id, cluster in enumerate(gold_clusters):
        for entity_id in cluster:
            gold_lookup[entity_id] = cluster_id

    # Calculate precision for each entity
    total_precision = 0.0
    entity_count = 0

    for pred_cluster in predicted_clusters:
        for entity in pred_cluster:
            # Count how many entities in this predicted cluster share
            # the same gold cluster as this entity
            same_cluster_count = sum(
                1
                for other in pred_cluster
                if gold_lookup.get(entity) == gold_lookup.get(other)
            )

            # Precision for this entity = same_cluster / cluster_size
            precision = same_cluster_count / len(pred_cluster)
            total_precision += precision
            entity_count += 1

    return total_precision / entity_count if entity_count > 0 else 0.0


def calculate_bcubed_recall(
    predicted_clusters: list[set[str]], gold_clusters: list[set[str]]
) -> float:
    """Calculate BCubed Recall.

    BCubed Recall measures how many items from the same gold cluster
    are placed in the same predicted cluster.

    Args:
        predicted_clusters: List of predicted entity clusters
        gold_clusters: List of gold-standard entity clusters

    Returns:
        BCubed Recall score in range [0.0, 1.0]
    """
    # Build predicted cluster lookup: entity_id -> cluster_id
    pred_lookup: dict[str, int] = {}
    for cluster_id, cluster in enumerate(predicted_clusters):
        for entity_id in cluster:
            pred_lookup[entity_id] = cluster_id

    # Calculate recall for each entity
    total_recall = 0.0
    entity_count = 0

    for gold_cluster in gold_clusters:
        for entity in gold_cluster:
            # Count how many entities in this gold cluster are also
            # in the same predicted cluster as this entity
            same_cluster_count = sum(
                1
                for other in gold_cluster
                if pred_lookup.get(entity) == pred_lookup.get(other)
            )

            # Recall for this entity = same_cluster / gold_cluster_size
            recall = same_cluster_count / len(gold_cluster)
            total_recall += recall
            entity_count += 1

    return total_recall / entity_count if entity_count > 0 else 0.0


def calculate_bcubed_f1(
    predicted_clusters: list[set[str]], gold_clusters: list[set[str]]
) -> dict[str, float]:
    """Calculate BCubed Precision, Recall, and F1.

    Args:
        predicted_clusters: List of predicted entity clusters
        gold_clusters: List of gold-standard entity clusters

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    precision = calculate_bcubed_precision(predicted_clusters, gold_clusters)
    recall = calculate_bcubed_recall(predicted_clusters, gold_clusters)

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


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

    # 5. Calculate BCubed F1
    metrics = calculate_bcubed_f1(predicted_clusters, EXPECTED_DUPLICATE_GROUPS)

    print("\nApproach 1 Results:")
    print(f"  BCubed Precision: {metrics['precision']:.3f}")
    print(f"  BCubed Recall:    {metrics['recall']:.3f}")
    print(f"  BCubed F1:        {metrics['f1']:.3f}")
    print(f"\nPredicted clusters: {len(predicted_clusters)}")
    print(f"Gold clusters:      {len(EXPECTED_DUPLICATE_GROUPS)}")

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
    module = RapidfuzzModule(
        field_extractors={"name": (lambda x: x.name, 1.0)}, threshold=0.5
    )
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
    module = RapidfuzzModule(
        field_extractors={"name": (lambda x: x.name, 1.0)}, threshold=0.5
    )
    clusterer = Clusterer(threshold=0.7)

    candidates = blocker.stream(data)
    judgements = list(module.forward(candidates))
    clusters = clusterer.cluster(judgements)

    # No clusters should form (all below threshold)
    # Each entity is its own singleton cluster (or no clusters if we only return multi-entity clusters)
    # Actually, Clusterer only returns connected components, so no edges = no clusters
    assert len(clusters) == 0  # No pairs meet the threshold
