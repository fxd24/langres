"""Tests for new generic loaders in langres.data.loaders module."""

import json
from pathlib import Path

import pytest

from langres.data.loaders import load_labeled_dedup_data, load_labeled_dedup_data_legacy
from langres.data.schemas import LabeledDeduplicationDataset, LabeledGroup


class TestLoadLabeledDedupData:
    """Tests for load_labeled_dedup_data function."""

    def test_load_valid_data_new_format(self, tmp_path: Path) -> None:
        """Test loading valid data with new format (entity_ids, entity_names)."""
        # Create test data with new format
        entity_names_data = {
            "names": {
                "1": "Company A",
                "2": "Company A Inc",
                "3": "Company B",
            }
        }

        labeled_groups_data = {
            "groups": [
                {
                    "canonical_name": "Company A",
                    "entity_ids": ["1", "2"],
                    "entity_names": ["Company A", "Company A Inc"],
                }
            ]
        }

        # Write test files
        with open(tmp_path / "entity_names.json", "w") as f:
            json.dump(entity_names_data, f)

        with open(tmp_path / "labeled_groups.json", "w") as f:
            json.dump(labeled_groups_data, f)

        # Load data
        dataset = load_labeled_dedup_data(tmp_path)

        # Verify it returns LabeledDeduplicationDataset
        assert isinstance(dataset, LabeledDeduplicationDataset)
        assert len(dataset.entity_names) == 3
        assert dataset.entity_names["1"] == "Company A"
        assert len(dataset.labeled_groups) == 1
        assert dataset.labeled_groups[0].canonical_name == "Company A"
        assert dataset.labeled_groups[0].entity_ids == ["1", "2"]
        assert dataset.num_unique_entities == 2  # 1 group + 1 singleton

    def test_load_valid_data_legacy_format(self, tmp_path: Path) -> None:
        """Test loading data with legacy format (variant_ids, variant_names)."""
        # Create test data with legacy format
        entity_names_data = {
            "names": {
                "1": "Acme",
                "2": "Acme Inc",
                "3": "Widget Co",
            }
        }

        labeled_groups_data = {
            "groups": [
                {
                    "canonical_name": "Acme",
                    "variant_ids": ["1", "2"],  # Legacy field name
                    "variant_names": ["Acme", "Acme Inc"],  # Legacy field name
                }
            ]
        }

        # Write test files
        with open(tmp_path / "entity_names.json", "w") as f:
            json.dump(entity_names_data, f)

        with open(tmp_path / "labeled_groups.json", "w") as f:
            json.dump(labeled_groups_data, f)

        # Load data
        dataset = load_labeled_dedup_data(tmp_path)

        # Verify backward compatibility - should convert legacy format
        assert isinstance(dataset, LabeledDeduplicationDataset)
        assert len(dataset.entity_names) == 3
        assert len(dataset.labeled_groups) == 1
        assert dataset.labeled_groups[0].entity_ids == ["1", "2"]
        assert dataset.labeled_groups[0].entity_names == ["Acme", "Acme Inc"]

    def test_load_custom_file_names(self, tmp_path: Path) -> None:
        """Test loading data with custom file names."""
        entity_names_data = {"names": {"1": "A", "2": "B"}}
        labeled_groups_data = {"groups": []}

        # Write to custom file names
        with open(tmp_path / "custom_names.json", "w") as f:
            json.dump(entity_names_data, f)

        with open(tmp_path / "custom_groups.json", "w") as f:
            json.dump(labeled_groups_data, f)

        # Load with custom file names
        dataset = load_labeled_dedup_data(
            tmp_path,
            entity_names_file="custom_names.json",
            labeled_groups_file="custom_groups.json",
        )

        assert len(dataset.entity_names) == 2
        assert len(dataset.labeled_groups) == 0

    def test_missing_directory(self) -> None:
        """Test error handling when directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            load_labeled_dedup_data("/nonexistent/path")

    def test_missing_entity_names_file(self, tmp_path: Path) -> None:
        """Test error handling when entity_names file is missing."""
        # Create only groups file
        groups_data = {"groups": []}
        with open(tmp_path / "labeled_groups.json", "w") as f:
            json.dump(groups_data, f)

        with pytest.raises(FileNotFoundError, match="entity_names.json"):
            load_labeled_dedup_data(tmp_path)

    def test_missing_labeled_groups_file(self, tmp_path: Path) -> None:
        """Test error handling when labeled_groups file is missing."""
        # Create only names file
        names_data = {"names": {"1": "A"}}
        with open(tmp_path / "entity_names.json", "w") as f:
            json.dump(names_data, f)

        with pytest.raises(FileNotFoundError, match="labeled_groups.json"):
            load_labeled_dedup_data(tmp_path)

    def test_invalid_entity_names_format(self, tmp_path: Path) -> None:
        """Test error handling for invalid entity_names file format."""
        # Create invalid names file (missing 'names' key)
        names_data = {"invalid_key": {}}
        with open(tmp_path / "entity_names.json", "w") as f:
            json.dump(names_data, f)

        groups_data = {"groups": []}
        with open(tmp_path / "labeled_groups.json", "w") as f:
            json.dump(groups_data, f)

        with pytest.raises(ValueError, match="missing 'names' key"):
            load_labeled_dedup_data(tmp_path)

    def test_invalid_labeled_groups_format(self, tmp_path: Path) -> None:
        """Test error handling for invalid labeled_groups file format."""
        names_data = {"names": {"1": "A"}}
        with open(tmp_path / "entity_names.json", "w") as f:
            json.dump(names_data, f)

        # Create invalid groups file (missing 'groups' key)
        groups_data = {"invalid_key": []}
        with open(tmp_path / "labeled_groups.json", "w") as f:
            json.dump(groups_data, f)

        with pytest.raises(ValueError, match="missing 'groups' key"):
            load_labeled_dedup_data(tmp_path)

    def test_string_path_handling(self, tmp_path: Path) -> None:
        """Test that function accepts both Path and str for data_dir."""
        # Create minimal valid data
        names_data = {"names": {"1": "A"}}
        with open(tmp_path / "entity_names.json", "w") as f:
            json.dump(names_data, f)

        groups_data = {"groups": []}
        with open(tmp_path / "labeled_groups.json", "w") as f:
            json.dump(groups_data, f)

        # Test with Path object
        dataset1 = load_labeled_dedup_data(tmp_path)
        assert len(dataset1.entity_names) == 1

        # Test with string path
        dataset2 = load_labeled_dedup_data(str(tmp_path))
        assert len(dataset2.entity_names) == 1

    def test_load_with_note_field(self, tmp_path: Path) -> None:
        """Test loading groups with optional note field."""
        entity_names_data = {"names": {"1": "A", "2": "A Inc"}}

        labeled_groups_data = {
            "groups": [
                {
                    "canonical_name": "A",
                    "entity_ids": ["1", "2"],
                    "entity_names": ["A", "A Inc"],
                    "note": "Verified from business registry",
                }
            ]
        }

        with open(tmp_path / "entity_names.json", "w") as f:
            json.dump(entity_names_data, f)

        with open(tmp_path / "labeled_groups.json", "w") as f:
            json.dump(labeled_groups_data, f)

        dataset = load_labeled_dedup_data(tmp_path)

        assert dataset.labeled_groups[0].note == "Verified from business registry"


class TestLoadLabeledDedupDataLegacy:
    """Tests for load_labeled_dedup_data_legacy convenience function."""

    def test_legacy_function_uses_correct_file_names(self, tmp_path: Path) -> None:
        """Test that legacy function uses iteration_05 file naming convention."""
        # Create data with iteration_05 file names
        names_data = {"names": {"1": "Company A", "2": "Company B"}}
        groups_data = {
            "groups": [
                {
                    "canonical_name": "Company A",
                    "variant_ids": ["1"],  # Legacy format
                    "variant_names": ["Company A"],
                }
            ]
        }

        # Write to legacy file names
        with open(tmp_path / "all_names_with_ids.json", "w") as f:
            json.dump(names_data, f)

        with open(tmp_path / "deduplicated_groups.json", "w") as f:
            json.dump(groups_data, f)

        # Load using legacy function
        dataset = load_labeled_dedup_data_legacy(tmp_path)

        assert len(dataset.entity_names) == 2
        assert len(dataset.labeled_groups) == 1
        assert dataset.labeled_groups[0].canonical_name == "Company A"

    def test_legacy_function_missing_files(self, tmp_path: Path) -> None:
        """Test that legacy function fails appropriately with missing files."""
        # Empty directory
        with pytest.raises(FileNotFoundError, match="all_names_with_ids.json"):
            load_labeled_dedup_data_legacy(tmp_path)

    def test_legacy_function_returns_dataset_object(self, tmp_path: Path) -> None:
        """Test that legacy function returns LabeledDeduplicationDataset."""
        names_data = {"names": {"1": "A"}}
        groups_data = {"groups": []}

        with open(tmp_path / "all_names_with_ids.json", "w") as f:
            json.dump(names_data, f)

        with open(tmp_path / "deduplicated_groups.json", "w") as f:
            json.dump(groups_data, f)

        dataset = load_labeled_dedup_data_legacy(tmp_path)

        assert isinstance(dataset, LabeledDeduplicationDataset)
        assert len(dataset.entity_names) == 1
