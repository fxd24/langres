"""Data loaders for entity resolution datasets.

This module provides functions for loading labeled entity resolution data
from various sources and formats.
"""

import json
import logging
from pathlib import Path
from typing import Any

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
