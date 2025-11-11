"""Data loaders for entity resolution datasets.

This module provides functions for loading labeled entity resolution data
from various sources and formats.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langres.data.schemas import LabeledDeduplicationDataset, LabeledGroup

logger = logging.getLogger(__name__)


def load_iteration_05_data(
    data_dir: Path | str | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Load iteration 05 labeled deduplication data.

    This loads the manually curated Swiss funder deduplication dataset
    (iteration 05) which includes:
    - 1,741 total entity names
    - 1,305 unique entities after deduplication
    - 230 duplicate groups (with 2-29 variants each)
    - 1,075 singleton entities (appear only once)

    The data has been verified against the Zefix business registry
    to ensure high-quality ground truth labels.

    Args:
        data_dir: Path to directory containing iteration_05 data files.
                  If None, defaults to "tmp/dedup_iteration_05"

    Returns:
        Tuple of (all_names, duplicate_groups, summary) where:
        - all_names: Dict mapping entity ID (str) to name (str)
        - duplicate_groups: List of duplicate groups, each with:
            * canonical_name: str
            * variant_ids: list[int]
            * variant_names: list[str]
            * note: str (optional)
        - summary: Dict with deduplication statistics

    Raises:
        FileNotFoundError: If data directory or required files don't exist
        ValueError: If data format is invalid

    Example:
        >>> all_names, groups, summary = load_iteration_05_data()
        >>> print(f"Loaded {len(all_names)} names, {len(groups)} groups")
        Loaded 1741 names, 230 groups
    """
    # Default to tmp/dedup_iteration_05 if not specified
    if data_dir is None:
        data_dir = Path("tmp/dedup_iteration_05")
    else:
        data_dir = Path(data_dir)

    # Validate directory exists
    if not data_dir.exists():
        msg = f"Data directory not found: {data_dir}"
        raise FileNotFoundError(msg)

    # Load all names
    names_path = data_dir / "all_names_with_ids.json"
    if not names_path.exists():
        msg = f"Required file not found: {names_path}"
        raise FileNotFoundError(msg)

    with open(names_path) as f:
        names_data = json.load(f)

    # Validate structure
    if "names" not in names_data:
        msg = f"Invalid format in {names_path}: missing 'names' key"
        raise ValueError(msg)

    all_names: dict[str, str] = names_data["names"]

    # Load duplicate groups
    groups_path = data_dir / "deduplicated_groups.json"
    if not groups_path.exists():
        msg = f"Required file not found: {groups_path}"
        raise FileNotFoundError(msg)

    with open(groups_path) as f:
        groups_data = json.load(f)

    # Validate structure
    if "groups" not in groups_data:
        msg = f"Invalid format in {groups_path}: missing 'groups' key"
        raise ValueError(msg)

    duplicate_groups: list[dict[str, Any]] = groups_data["groups"]

    # Load summary (optional)
    summary_path = data_dir / "deduplication_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    logger.info(
        "Loaded iteration_05 data from %s: %d names, %d duplicate groups",
        data_dir,
        len(all_names),
        len(duplicate_groups),
    )

    # Log summary statistics if available
    if "deduplication_summary" in summary:
        stats = summary["deduplication_summary"]
        logger.info(
            "  Total unique entities: %d (%.1f%% deduplication rate)",
            stats["output"]["total_unique_entities"],
            stats["reduction"]["reduction_percentage"],
        )

    return all_names, duplicate_groups, summary


def load_labeled_dedup_data(
    data_dir: Path | str,
    entity_names_file: str = "entity_names.json",
    labeled_groups_file: str = "labeled_groups.json",
) -> LabeledDeduplicationDataset:
    """Load labeled deduplication dataset from directory.

    This is a generic loader that works with any labeled deduplication dataset
    following the standard format. It replaces dataset-specific loaders like
    load_iteration_05_data() with a general-purpose, schema-driven approach.

    Args:
        data_dir: Directory containing labeled data files
        entity_names_file: JSON file with {id -> name} mapping (default: "entity_names.json")
        labeled_groups_file: JSON file with ground truth entity groups (default: "labeled_groups.json")

    Returns:
        LabeledDeduplicationDataset with entity_names and labeled_groups

    Raises:
        FileNotFoundError: If data directory or required files don't exist
        ValueError: If data format is invalid

    Expected JSON formats:
        entity_names.json:
            {"names": {"1": "Company A", "2": "Company A Inc", ...}}

        labeled_groups.json:
            {"groups": [
                {
                    "canonical_name": "Company A",
                    "entity_ids": ["1", "2"],  # Or variant_ids for legacy
                    "entity_names": ["Company A", "Company A Inc"]  # Or variant_names for legacy
                },
                ...
            ]}

    Note:
        Handles both new format (entity_ids/entity_names) and legacy format
        (variant_ids/variant_names) for backward compatibility.

    Example:
        >>> # Load with default file names
        >>> dataset = load_labeled_dedup_data("data/my_dataset")
        >>> print(f"{len(dataset.entity_names)} entities, {dataset.num_unique_entities} unique")

        >>> # Load with custom file names
        >>> dataset = load_labeled_dedup_data(
        ...     "data/custom",
        ...     entity_names_file="names.json",
        ...     labeled_groups_file="groups.json"
        ... )
    """
    # Convert to Path
    data_dir = Path(data_dir)

    # Validate directory exists
    if not data_dir.exists():
        msg = f"Data directory not found: {data_dir}"
        raise FileNotFoundError(msg)

    # Load entity names
    names_path = data_dir / entity_names_file
    if not names_path.exists():
        msg = f"Required file not found: {names_path}"
        raise FileNotFoundError(msg)

    with open(names_path) as f:
        names_data = json.load(f)

    # Validate structure
    if "names" not in names_data:
        msg = f"Invalid format in {names_path}: missing 'names' key"
        raise ValueError(msg)

    entity_names: dict[str, str] = names_data["names"]

    # Load labeled groups
    groups_path = data_dir / labeled_groups_file
    if not groups_path.exists():
        msg = f"Required file not found: {groups_path}"
        raise FileNotFoundError(msg)

    with open(groups_path) as f:
        groups_data = json.load(f)

    # Validate structure
    if "groups" not in groups_data:
        msg = f"Invalid format in {groups_path}: missing 'groups' key"
        raise ValueError(msg)

    raw_groups: list[dict[str, Any]] = groups_data["groups"]

    # Convert to LabeledGroup objects (handle both new and legacy formats)
    labeled_groups: list[LabeledGroup] = []
    for raw_group in raw_groups:
        # Handle legacy format (variant_ids, variant_names) vs new format (entity_ids, entity_names)
        entity_ids = raw_group.get("entity_ids")
        entity_names_list = raw_group.get("entity_names")

        # Fall back to legacy field names if new ones not present
        if entity_ids is None:
            entity_ids = raw_group.get("variant_ids")
            if entity_ids is not None:
                # Convert integers to strings if needed (legacy format used ints)
                entity_ids = [str(id_) for id_ in entity_ids]

        if entity_names_list is None:
            entity_names_list = raw_group.get("variant_names")

        # Create LabeledGroup
        labeled_group = LabeledGroup(
            canonical_name=raw_group["canonical_name"],
            entity_ids=entity_ids,  # type: ignore
            entity_names=entity_names_list,  # type: ignore
            note=raw_group.get("note"),
        )
        labeled_groups.append(labeled_group)

    logger.info(
        "Loaded labeled deduplication data from %s: %d entities, %d labeled groups",
        data_dir,
        len(entity_names),
        len(labeled_groups),
    )

    return LabeledDeduplicationDataset(
        entity_names=entity_names,
        labeled_groups=labeled_groups,
    )


def load_labeled_dedup_data_legacy(
    data_dir: Path | str,
) -> LabeledDeduplicationDataset:
    """Load data in legacy iteration_05 format.

    This is a convenience function for loading data with the legacy iteration_05
    file naming convention. It's equivalent to calling load_labeled_dedup_data()
    with specific file names.

    Args:
        data_dir: Directory containing iteration_05 data files

    Returns:
        LabeledDeduplicationDataset with entity_names and labeled_groups

    Raises:
        FileNotFoundError: If data directory or required files don't exist
        ValueError: If data format is invalid

    Example:
        >>> dataset = load_labeled_dedup_data_legacy("tmp/dedup_iteration_05")
        >>> print(f"Loaded {dataset.num_unique_entities} unique entities")
    """
    return load_labeled_dedup_data(
        data_dir,
        entity_names_file="all_names_with_ids.json",
        labeled_groups_file="deduplicated_groups.json",
    )
