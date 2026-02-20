"""Data loaders for entity resolution datasets.

This module provides functions for loading labeled entity resolution data
following the langres standard schema.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langres.data.schemas import LabeledDeduplicationDataset, LabeledGroup

logger = logging.getLogger(__name__)


def load_labeled_dedup_data(
    data_dir: Path | str,
    entity_names_file: str = "entity_names.json",
    labeled_groups_file: str = "labeled_groups.json",
) -> LabeledDeduplicationDataset:
    """Load labeled deduplication dataset from directory.

    This is a generic loader that works with any labeled deduplication dataset
    following the langres standard schema format.

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
                    "entity_ids": ["1", "2"],
                    "entity_names": ["Company A", "Company A Inc"],
                    "note": "Optional verification note"
                },
                ...
            ]}

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

    # Convert to LabeledGroup objects
    labeled_groups: list[LabeledGroup] = []
    for raw_group in raw_groups:
        # Support both "entity_ids"/"entity_names" and "variant_ids"/"variant_names"
        # Convert IDs to strings (some data files use integers)
        raw_ids = raw_group.get("entity_ids") or raw_group.get("variant_ids", [])
        group_entity_ids = [str(id_val) for id_val in raw_ids]
        group_entity_names = raw_group.get("entity_names") or raw_group.get("variant_names", [])

        labeled_group = LabeledGroup(
            canonical_name=raw_group["canonical_name"],
            entity_ids=group_entity_ids,
            entity_names=group_entity_names,
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
