"""Train/test splitting utilities for entity resolution datasets.

This module provides functions for splitting labeled entity resolution data
into train and test sets while preserving important properties:
- Stratification by cluster size (ensures representative samples)
- No data leakage (entity groups stay together)
- Balanced distribution of singletons and duplicates
"""

import logging
import random
from collections import defaultdict
from typing import Any

from langres.data.schemas import LabeledDeduplicationDataset

logger = logging.getLogger(__name__)


def stratified_dedup_split(
    dataset_or_names: LabeledDeduplicationDataset | dict[str, str],
    duplicate_groups: list[dict[str, Any]] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[set[str]], list[set[str]]]:
    """Split deduplication data with stratification by cluster size.

    This function performs a stratified train/test split that:
    1. Preserves cluster size distribution (singletons, pairs, triples, etc.)
    2. Prevents data leakage (all variants of an entity stay together)
    3. Ensures both train and test have representative samples of all cluster sizes

    Args:
        dataset_or_names: Either a LabeledDeduplicationDataset object (new API)
                         or a dict mapping entity ID to name (legacy API)
        duplicate_groups: List of duplicate groups with "variant_ids" key (legacy API only).
                         Must be None when using LabeledDeduplicationDataset.
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_records, test_records, train_clusters, test_clusters) where:
        - train_records: List of dicts with "id" and "name" keys for training
        - test_records: List of dicts with "id" and "name" keys for testing
        - train_clusters: List of sets of entity IDs (ground truth clusters)
        - test_clusters: List of sets of entity IDs (ground truth clusters)

    Examples:
        >>> # New API: Use dataset object
        >>> dataset = LabeledDeduplicationDataset(...)
        >>> train_rec, test_rec, train_cls, test_cls = stratified_dedup_split(
        ...     dataset, test_size=0.2, random_state=42
        ... )

        >>> # Legacy API: Use dicts and lists
        >>> all_names = {"1": "Acme", "2": "Acme Inc", "3": "Widget Co"}
        >>> groups = [{"variant_ids": [1, 2]}]
        >>> train_rec, test_rec, train_cls, test_cls = stratified_dedup_split(
        ...     all_names, groups, test_size=0.5, random_state=42
        ... )
    """
    # Handle both new API (dataset object) and legacy API (dict + list)
    if isinstance(dataset_or_names, LabeledDeduplicationDataset):
        # New API: Extract from dataset object
        if duplicate_groups is not None:
            msg = "duplicate_groups must be None when using LabeledDeduplicationDataset"
            raise ValueError(msg)

        all_names = dataset_or_names.entity_names
        # Convert LabeledGroup objects to legacy format for processing
        duplicate_groups = [
            {
                "variant_ids": [int(id_) for id_ in group.entity_ids],
                "canonical_name": group.canonical_name,
            }
            for group in dataset_or_names.labeled_groups
        ]
    else:
        # Legacy API: Use provided dict and list
        if duplicate_groups is None:
            msg = "duplicate_groups is required when not using LabeledDeduplicationDataset"
            raise ValueError(msg)

        all_names = dataset_or_names
    # Set random seed for reproducibility
    random.seed(random_state)

    # Extract all IDs that appear in duplicate groups
    grouped_ids: set[str] = set()
    for group in duplicate_groups:
        grouped_ids.update(str(id_) for id_ in group["variant_ids"])

    # Identify singleton IDs (not in any duplicate group)
    all_ids = set(all_names.keys())
    singleton_ids = all_ids - grouped_ids

    logger.info(
        "Dataset statistics: %d total entities, %d in groups, %d singletons",
        len(all_ids),
        len(grouped_ids),
        len(singleton_ids),
    )

    # Stratify duplicate groups by size
    groups_by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in duplicate_groups:
        size = len(group["variant_ids"])
        groups_by_size[size].append(group)

    # Log stratification statistics
    logger.info("Cluster size distribution:")
    logger.info("  Singletons (size 1): %d", len(singleton_ids))
    for size in sorted(groups_by_size.keys()):
        logger.info(
            "  Size %d: %d groups (%d entities)",
            size,
            len(groups_by_size[size]),
            size * len(groups_by_size[size]),
        )

    # Initialize train/test splits
    train_ids: set[str] = set()
    test_ids: set[str] = set()
    train_clusters: list[set[str]] = []
    test_clusters: list[set[str]] = []

    # Split singletons (each singleton is its own cluster)
    singleton_list = list(singleton_ids)
    random.shuffle(singleton_list)
    n_test_singletons = max(1, int(len(singleton_list) * test_size))

    test_singleton_ids = singleton_list[-n_test_singletons:]
    train_singleton_ids = singleton_list[:-n_test_singletons]

    train_ids.update(train_singleton_ids)
    test_ids.update(test_singleton_ids)

    # Each singleton is its own cluster (set of size 1)
    train_clusters.extend([{id_} for id_ in train_singleton_ids])
    test_clusters.extend([{id_} for id_ in test_singleton_ids])

    logger.info(
        "Singleton split: %d train, %d test (%.1f%% test)",
        len(train_singleton_ids),
        len(test_singleton_ids),
        100 * len(test_singleton_ids) / len(singleton_list) if singleton_list else 0,
    )

    # Split each size stratum of duplicate groups
    for size in sorted(groups_by_size.keys()):
        groups = groups_by_size[size]
        random.shuffle(groups)

        n_test_groups = max(1, int(len(groups) * test_size))
        test_groups = groups[-n_test_groups:]
        train_groups = groups[:-n_test_groups]

        # Add to train
        for group in train_groups:
            group_ids = {str(id_) for id_ in group["variant_ids"]}
            train_ids.update(group_ids)
            train_clusters.append(group_ids)

        # Add to test
        for group in test_groups:
            group_ids = {str(id_) for id_ in group["variant_ids"]}
            test_ids.update(group_ids)
            test_clusters.append(group_ids)

        logger.info(
            "  Size %d split: %d train groups, %d test groups (%.1f%% test)",
            size,
            len(train_groups),
            len(test_groups),
            100 * len(test_groups) / len(groups),
        )

    # Convert to record format (list of dicts with "id" and "name")
    train_records = [{"id": id_, "name": all_names[id_]} for id_ in sorted(train_ids)]
    test_records = [{"id": id_, "name": all_names[id_]} for id_ in sorted(test_ids)]

    # Verify no data leakage
    assert len(train_ids & test_ids) == 0, (
        "Data leakage detected: overlapping IDs in train and test"
    )

    # Verify all IDs are accounted for
    assert train_ids | test_ids == all_ids, "Missing IDs in split"

    logger.info(
        "Final split: %d train records (%d clusters), %d test records (%d clusters)",
        len(train_records),
        len(train_clusters),
        len(test_records),
        len(test_clusters),
    )

    return train_records, test_records, train_clusters, test_clusters
