"""Tests for langres.data.schemas module."""

import pytest
from pydantic import ValidationError

from langres.data.schemas import LabeledDeduplicationDataset, LabeledGroup


class TestLabeledGroup:
    """Tests for LabeledGroup Pydantic model."""

    def test_valid_labeled_group(self) -> None:
        """Test creating a valid LabeledGroup."""
        group = LabeledGroup(
            canonical_name="Company A",
            entity_ids=["1", "2", "3"],
            entity_names=["Company A", "Company A Inc", "Company A Ltd"],
        )

        assert group.canonical_name == "Company A"
        assert group.entity_ids == ["1", "2", "3"]
        assert group.entity_names == ["Company A", "Company A Inc", "Company A Ltd"]
        assert group.note is None

    def test_labeled_group_with_note(self) -> None:
        """Test LabeledGroup with optional note field."""
        group = LabeledGroup(
            canonical_name="Acme Corp",
            entity_ids=["10", "11"],
            entity_names=["Acme", "Acme Corporation"],
            note="Verified against business registry",
        )

        assert group.note == "Verified against business registry"

    def test_labeled_group_missing_required_fields(self) -> None:
        """Test that missing required fields raise ValidationError."""
        # Missing canonical_name
        with pytest.raises(ValidationError):
            LabeledGroup(
                entity_ids=["1", "2"],
                entity_names=["A", "B"],
            )

        # Missing entity_ids
        with pytest.raises(ValidationError):
            LabeledGroup(
                canonical_name="Company A",
                entity_names=["A", "B"],
            )

        # Missing entity_names
        with pytest.raises(ValidationError):
            LabeledGroup(
                canonical_name="Company A",
                entity_ids=["1", "2"],
            )

    def test_labeled_group_type_validation(self) -> None:
        """Test type validation for LabeledGroup fields."""
        # canonical_name must be string
        with pytest.raises(ValidationError):
            LabeledGroup(
                canonical_name=123,  # type: ignore
                entity_ids=["1"],
                entity_names=["A"],
            )

        # entity_ids must be list of strings
        with pytest.raises(ValidationError):
            LabeledGroup(
                canonical_name="A",
                entity_ids="not_a_list",  # type: ignore
                entity_names=["A"],
            )

        # entity_names must be list of strings
        with pytest.raises(ValidationError):
            LabeledGroup(
                canonical_name="A",
                entity_ids=["1"],
                entity_names=[123],  # type: ignore
            )


class TestLabeledDeduplicationDataset:
    """Tests for LabeledDeduplicationDataset Pydantic model."""

    def test_valid_dataset(self) -> None:
        """Test creating a valid LabeledDeduplicationDataset."""
        entity_names = {
            "1": "Company A",
            "2": "Company A Inc",
            "3": "Company B",
            "4": "Company C",
        }

        labeled_groups = [
            LabeledGroup(
                canonical_name="Company A",
                entity_ids=["1", "2"],
                entity_names=["Company A", "Company A Inc"],
            )
        ]

        dataset = LabeledDeduplicationDataset(
            entity_names=entity_names,
            labeled_groups=labeled_groups,
        )

        assert len(dataset.entity_names) == 4
        assert dataset.entity_names["1"] == "Company A"
        assert len(dataset.labeled_groups) == 1
        assert dataset.labeled_groups[0].canonical_name == "Company A"

    def test_dataset_missing_required_fields(self) -> None:
        """Test that missing required fields raise ValidationError."""
        # Missing entity_names
        with pytest.raises(ValidationError):
            LabeledDeduplicationDataset(
                labeled_groups=[],
            )

        # Missing labeled_groups
        with pytest.raises(ValidationError):
            LabeledDeduplicationDataset(
                entity_names={},
            )

    def test_dataset_type_validation(self) -> None:
        """Test type validation for dataset fields."""
        # entity_names must be dict
        with pytest.raises(ValidationError):
            LabeledDeduplicationDataset(
                entity_names="not_a_dict",  # type: ignore
                labeled_groups=[],
            )

        # labeled_groups must be list
        with pytest.raises(ValidationError):
            LabeledDeduplicationDataset(
                entity_names={},
                labeled_groups="not_a_list",  # type: ignore
            )

    def test_num_unique_entities_with_groups_and_singletons(self) -> None:
        """Test num_unique_entities property with both grouped and singleton entities."""
        entity_names = {
            "1": "Company A",
            "2": "Company A Inc",  # Grouped with 1
            "3": "Company B",
            "4": "Company B Ltd",  # Grouped with 3
            "5": "Company C",  # Singleton
            "6": "Company D",  # Singleton
        }

        labeled_groups = [
            LabeledGroup(
                canonical_name="Company A",
                entity_ids=["1", "2"],
                entity_names=["Company A", "Company A Inc"],
            ),
            LabeledGroup(
                canonical_name="Company B",
                entity_ids=["3", "4"],
                entity_names=["Company B", "Company B Ltd"],
            ),
        ]

        dataset = LabeledDeduplicationDataset(
            entity_names=entity_names,
            labeled_groups=labeled_groups,
        )

        # 2 labeled groups + 2 singletons = 4 unique entities
        assert dataset.num_unique_entities == 4

    def test_num_unique_entities_no_groups(self) -> None:
        """Test num_unique_entities when there are no labeled groups (all singletons)."""
        entity_names = {
            "1": "Company A",
            "2": "Company B",
            "3": "Company C",
        }

        dataset = LabeledDeduplicationDataset(
            entity_names=entity_names,
            labeled_groups=[],  # No duplicate groups
        )

        # All 3 entities are singletons
        assert dataset.num_unique_entities == 3

    def test_num_unique_entities_no_singletons(self) -> None:
        """Test num_unique_entities when all entities are in groups (no singletons)."""
        entity_names = {
            "1": "Company A",
            "2": "Company A Inc",
            "3": "Company B",
            "4": "Company B Ltd",
        }

        labeled_groups = [
            LabeledGroup(
                canonical_name="Company A",
                entity_ids=["1", "2"],
                entity_names=["Company A", "Company A Inc"],
            ),
            LabeledGroup(
                canonical_name="Company B",
                entity_ids=["3", "4"],
                entity_names=["Company B", "Company B Ltd"],
            ),
        ]

        dataset = LabeledDeduplicationDataset(
            entity_names=entity_names,
            labeled_groups=labeled_groups,
        )

        # Only 2 labeled groups, no singletons
        assert dataset.num_unique_entities == 2

    def test_num_unique_entities_empty_dataset(self) -> None:
        """Test num_unique_entities with empty dataset."""
        dataset = LabeledDeduplicationDataset(
            entity_names={},
            labeled_groups=[],
        )

        assert dataset.num_unique_entities == 0

    def test_num_unique_entities_large_group(self) -> None:
        """Test num_unique_entities with a large group (multiple variants of same entity)."""
        entity_names = {
            "1": "Acme",
            "2": "Acme Inc",
            "3": "Acme Corp",
            "4": "Acme Corporation",
            "5": "Widget Co",  # Singleton
        }

        labeled_groups = [
            LabeledGroup(
                canonical_name="Acme",
                entity_ids=["1", "2", "3", "4"],
                entity_names=["Acme", "Acme Inc", "Acme Corp", "Acme Corporation"],
            )
        ]

        dataset = LabeledDeduplicationDataset(
            entity_names=entity_names,
            labeled_groups=labeled_groups,
        )

        # 1 labeled group (4 entities collapsed to 1) + 1 singleton = 2 unique entities
        assert dataset.num_unique_entities == 2
