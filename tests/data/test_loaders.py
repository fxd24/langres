"""Tests for langres.data.loaders module."""

import json
import tempfile
from pathlib import Path

import pytest

from langres.data.loaders import load_iteration_05_data


class TestLoadIteration05Data:
    """Tests for load_iteration_05_data function."""

    def test_load_valid_data(self, tmp_path: Path) -> None:
        """Test loading valid iteration_05 data."""
        # Create test data files
        names_data = {
            "description": "Test data",
            "total_count": 3,
            "names": {
                "1": "Company A",
                "2": "Company B",
                "3": "Company C",
            },
        }

        groups_data = {
            "groups": [
                {
                    "canonical_name": "Company A",
                    "variant_ids": [1, 2],
                    "variant_names": ["Company A", "Company B"],
                }
            ]
        }

        summary_data = {
            "deduplication_summary": {
                "iteration": 5,
                "output": {"total_unique_entities": 2},
                "reduction": {"reduction_percentage": 33.33},
            }
        }

        # Write test files
        with open(tmp_path / "all_names_with_ids.json", "w") as f:
            json.dump(names_data, f)

        with open(tmp_path / "deduplicated_groups.json", "w") as f:
            json.dump(groups_data, f)

        with open(tmp_path / "deduplication_summary.json", "w") as f:
            json.dump(summary_data, f)

        # Load data
        all_names, duplicate_groups, summary = load_iteration_05_data(tmp_path)

        # Verify loaded data
        assert len(all_names) == 3
        assert all_names["1"] == "Company A"
        assert len(duplicate_groups) == 1
        assert duplicate_groups[0]["canonical_name"] == "Company A"
        assert summary["deduplication_summary"]["iteration"] == 5

    def test_missing_directory(self) -> None:
        """Test error handling when directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            load_iteration_05_data("/nonexistent/path")

    def test_missing_names_file(self, tmp_path: Path) -> None:
        """Test error handling when names file is missing."""
        # Create only groups file
        groups_data = {"groups": []}
        with open(tmp_path / "deduplicated_groups.json", "w") as f:
            json.dump(groups_data, f)

        with pytest.raises(FileNotFoundError, match="all_names_with_ids.json"):
            load_iteration_05_data(tmp_path)

    def test_missing_groups_file(self, tmp_path: Path) -> None:
        """Test error handling when groups file is missing."""
        # Create only names file
        names_data = {"names": {"1": "A"}}
        with open(tmp_path / "all_names_with_ids.json", "w") as f:
            json.dump(names_data, f)

        with pytest.raises(FileNotFoundError, match="deduplicated_groups.json"):
            load_iteration_05_data(tmp_path)

    def test_invalid_names_format(self, tmp_path: Path) -> None:
        """Test error handling for invalid names file format."""
        # Create invalid names file (missing 'names' key)
        names_data = {"invalid_key": {}}
        with open(tmp_path / "all_names_with_ids.json", "w") as f:
            json.dump(names_data, f)

        groups_data = {"groups": []}
        with open(tmp_path / "deduplicated_groups.json", "w") as f:
            json.dump(groups_data, f)

        with pytest.raises(ValueError, match="missing 'names' key"):
            load_iteration_05_data(tmp_path)

    def test_invalid_groups_format(self, tmp_path: Path) -> None:
        """Test error handling for invalid groups file format."""
        names_data = {"names": {"1": "A"}}
        with open(tmp_path / "all_names_with_ids.json", "w") as f:
            json.dump(names_data, f)

        # Create invalid groups file (missing 'groups' key)
        groups_data = {"invalid_key": []}
        with open(tmp_path / "deduplicated_groups.json", "w") as f:
            json.dump(groups_data, f)

        with pytest.raises(ValueError, match="missing 'groups' key"):
            load_iteration_05_data(tmp_path)

    def test_missing_summary_optional(self, tmp_path: Path) -> None:
        """Test that summary file is optional."""
        # Create required files only
        names_data = {"names": {"1": "A"}}
        with open(tmp_path / "all_names_with_ids.json", "w") as f:
            json.dump(names_data, f)

        groups_data = {"groups": []}
        with open(tmp_path / "deduplicated_groups.json", "w") as f:
            json.dump(groups_data, f)

        # Should succeed without summary file
        all_names, duplicate_groups, summary = load_iteration_05_data(tmp_path)

        assert len(all_names) == 1
        assert len(duplicate_groups) == 0
        assert summary == {}  # Empty dict when summary not present

    def test_default_path(self) -> None:
        """Test that default path is tmp/dedup_iteration_05."""
        # This test will fail if the actual data doesn't exist, which is expected
        # We're just verifying the default path behavior
        try:
            load_iteration_05_data()
        except FileNotFoundError as e:
            # Should reference tmp/dedup_iteration_05 in error
            assert "tmp/dedup_iteration_05" in str(e) or "dedup_iteration_05" in str(e)

    def test_string_path_handling(self, tmp_path: Path) -> None:
        """Test that function accepts both Path and str for data_dir."""
        # Create minimal valid data
        names_data = {"names": {"1": "A"}}
        with open(tmp_path / "all_names_with_ids.json", "w") as f:
            json.dump(names_data, f)

        groups_data = {"groups": []}
        with open(tmp_path / "deduplicated_groups.json", "w") as f:
            json.dump(groups_data, f)

        # Test with Path object
        result1 = load_iteration_05_data(tmp_path)
        assert len(result1[0]) == 1

        # Test with string path
        result2 = load_iteration_05_data(str(tmp_path))
        assert len(result2[0]) == 1
