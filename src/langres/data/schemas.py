"""Data schemas for entity resolution datasets.

This module defines Pydantic models for labeled entity resolution datasets,
providing validation and type safety for ground truth data.
"""

from pydantic import BaseModel


class LabeledGroup(BaseModel):
    """Ground truth label: a group of entity IDs that refer to same real-world entity.

    This represents a labeled duplicate group where multiple entity name strings
    (with different IDs) have been identified as referring to the same real-world entity.

    Attributes:
        canonical_name: The canonical/preferred name for this entity
        entity_ids: List of entity IDs that belong to this group
        entity_names: List of entity name strings (corresponding to entity_ids)
        note: Optional note or explanation (e.g., verification source)

    Example:
        >>> group = LabeledGroup(
        ...     canonical_name="Acme Corporation",
        ...     entity_ids=["1", "2", "3"],
        ...     entity_names=["Acme", "Acme Inc", "Acme Corp"],
        ...     note="Verified against business registry"
        ... )
    """

    canonical_name: str
    entity_ids: list[str]
    entity_names: list[str]
    note: str | None = None


class LabeledDeduplicationDataset(BaseModel):
    """Labeled dataset for entity deduplication.

    Contains:
    - entity_names: All entity name strings with unique IDs
    - labeled_groups: Ground truth clusters (which IDs refer to same entity)

    Entity IDs that don't appear in any labeled_groups are treated as singletons
    (unique entities that appear only once).

    Attributes:
        entity_names: Dictionary mapping entity_id -> name string
        labeled_groups: List of ground truth entity groups

    Example:
        >>> dataset = LabeledDeduplicationDataset(
        ...     entity_names={
        ...         "1": "Company A",
        ...         "2": "Company A Inc",
        ...         "3": "Company B",
        ...     },
        ...     labeled_groups=[
        ...         LabeledGroup(
        ...             canonical_name="Company A",
        ...             entity_ids=["1", "2"],
        ...             entity_names=["Company A", "Company A Inc"],
        ...         )
        ...     ]
        ... )
        >>> dataset.num_unique_entities
        2
    """

    entity_names: dict[str, str]
    labeled_groups: list[LabeledGroup]

    @property
    def num_unique_entities(self) -> int:
        """Number of unique real-world entities (labels + singletons).

        This counts:
        - Each labeled group as 1 unique entity
        - Each entity ID not in any group as 1 unique entity (singleton)

        Returns:
            Total number of unique real-world entities

        Example:
            >>> # 2 entity IDs in 1 group + 1 singleton = 2 unique entities
            >>> dataset = LabeledDeduplicationDataset(
            ...     entity_names={"1": "A", "2": "A Inc", "3": "B"},
            ...     labeled_groups=[
            ...         LabeledGroup(
            ...             canonical_name="A",
            ...             entity_ids=["1", "2"],
            ...             entity_names=["A", "A Inc"],
            ...         )
            ...     ]
            ... )
            >>> dataset.num_unique_entities
            2
        """
        # Collect all IDs that appear in labeled groups
        grouped_ids = {id_ for group in self.labeled_groups for id_ in group.entity_ids}

        # Count singletons (IDs not in any group)
        singleton_count = len(self.entity_names) - len(grouped_ids)

        # Total unique entities = labeled groups + singletons
        return len(self.labeled_groups) + singleton_count
