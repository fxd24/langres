"""Tests for langres.data.splitting module."""

import pytest

from langres.data.splitting import stratified_dedup_split


class TestStratifiedDedupSplit:
    """Tests for stratified_dedup_split function."""

    def test_basic_split(self) -> None:
        """Test basic train/test split functionality."""
        # Create sample data
        all_names = {
            "1": "Company A",
            "2": "Company A Inc",
            "3": "Company B",
            "4": "Company B Ltd",
            "5": "Company C",
            "6": "Company D",
        }

        duplicate_groups = [
            {
                "canonical_name": "Company A",
                "variant_ids": [1, 2],
                "variant_names": ["Company A", "Company A Inc"],
            },
            {
                "canonical_name": "Company B",
                "variant_ids": [3, 4],
                "variant_names": ["Company B", "Company B Ltd"],
            },
        ]

        # Split with 50% test size for easier verification
        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.5, random_state=42
        )

        # Verify all records are accounted for
        train_ids = {r["id"] for r in train_records}
        test_ids = {r["id"] for r in test_records}
        assert train_ids | test_ids == set(all_names.keys())

        # Verify no data leakage
        assert len(train_ids & test_ids) == 0

        # Verify clusters match records
        train_cluster_ids = {id_ for cluster in train_clusters for id_ in cluster}
        test_cluster_ids = {id_ for cluster in test_clusters for id_ in cluster}
        assert train_cluster_ids == train_ids
        assert test_cluster_ids == test_ids

    def test_no_data_leakage(self) -> None:
        """Verify that entity groups are never split across train/test."""
        all_names = {
            "1": "Acme",
            "2": "Acme Inc",
            "3": "Acme Corp",
            "4": "Widget Co",
            "5": "Widget Company",
        }

        duplicate_groups = [
            {
                "canonical_name": "Acme",
                "variant_ids": [1, 2, 3],
                "variant_names": ["Acme", "Acme Inc", "Acme Corp"],
            },
            {
                "canonical_name": "Widget",
                "variant_ids": [4, 5],
                "variant_names": ["Widget Co", "Widget Company"],
            },
        ]

        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.5, random_state=42
        )

        # Check that Acme group (1,2,3) is entirely in train or test
        acme_ids = {"1", "2", "3"}
        train_ids = {r["id"] for r in train_records}
        test_ids = {r["id"] for r in test_records}

        # Either all in train or all in test
        assert acme_ids.issubset(train_ids) or acme_ids.issubset(test_ids)

        # Same for Widget group
        widget_ids = {"4", "5"}
        assert widget_ids.issubset(train_ids) or widget_ids.issubset(test_ids)

    def test_stratification_by_size(self) -> None:
        """Verify that cluster size distribution is preserved."""
        # Create data with different cluster sizes
        all_names = {}
        duplicate_groups = []

        # Add 10 pairs (size 2)
        for i in range(10):
            id1, id2 = str(i * 2 + 1), str(i * 2 + 2)
            all_names[id1] = f"Pair{i}_A"
            all_names[id2] = f"Pair{i}_B"
            duplicate_groups.append(
                {
                    "canonical_name": f"Pair{i}",
                    "variant_ids": [int(id1), int(id2)],
                    "variant_names": [f"Pair{i}_A", f"Pair{i}_B"],
                }
            )

        # Add 10 triples (size 3)
        for i in range(10):
            id1, id2, id3 = str(100 + i * 3), str(101 + i * 3), str(102 + i * 3)
            all_names[id1] = f"Triple{i}_A"
            all_names[id2] = f"Triple{i}_B"
            all_names[id3] = f"Triple{i}_C"
            duplicate_groups.append(
                {
                    "canonical_name": f"Triple{i}",
                    "variant_ids": [int(id1), int(id2), int(id3)],
                    "variant_names": [f"Triple{i}_A", f"Triple{i}_B", f"Triple{i}_C"],
                }
            )

        # Add 10 singletons
        for i in range(10):
            id_ = str(200 + i)
            all_names[id_] = f"Singleton{i}"

        # Split with 20% test
        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.2, random_state=42
        )

        # Count cluster sizes in train and test
        train_sizes = [len(cluster) for cluster in train_clusters]
        test_sizes = [len(cluster) for cluster in test_clusters]

        # Both should have all cluster sizes represented
        assert 1 in train_sizes  # Singletons
        assert 2 in train_sizes  # Pairs
        assert 3 in train_sizes  # Triples

        assert 1 in test_sizes  # Singletons
        assert 2 in test_sizes  # Pairs
        assert 3 in test_sizes  # Triples

        # Verify approximate 80/20 split for each size
        # Singletons
        train_singletons = sum(1 for s in train_sizes if s == 1)
        test_singletons = sum(1 for s in test_sizes if s == 1)
        assert 6 <= train_singletons <= 9  # Approximately 80% of 10
        assert 1 <= test_singletons <= 4  # Approximately 20% of 10

    def test_reproducibility(self) -> None:
        """Verify that same random_state produces same split."""
        all_names = {str(i): f"Entity{i}" for i in range(100)}
        duplicate_groups = [
            {
                "canonical_name": f"Group{i}",
                "variant_ids": [i * 2, i * 2 + 1],
                "variant_names": [f"Entity{i * 2}", f"Entity{i * 2 + 1}"],
            }
            for i in range(20)
        ]

        # Run twice with same seed
        result1 = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.2, random_state=42
        )
        result2 = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.2, random_state=42
        )

        # Should be identical
        train_ids_1 = {r["id"] for r in result1[0]}
        train_ids_2 = {r["id"] for r in result2[0]}
        assert train_ids_1 == train_ids_2

        # Different seed should give different split
        result3 = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.2, random_state=123
        )
        train_ids_3 = {r["id"] for r in result3[0]}
        assert train_ids_1 != train_ids_3

    def test_record_format(self) -> None:
        """Verify output records have correct format."""
        all_names = {"1": "Company A", "2": "Company B"}
        duplicate_groups = []

        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.5, random_state=42
        )

        # Check record structure
        for record in train_records + test_records:
            assert "id" in record
            assert "name" in record
            assert isinstance(record["id"], str)
            assert isinstance(record["name"], str)
            assert record["name"] == all_names[record["id"]]

    def test_cluster_format(self) -> None:
        """Verify output clusters are sets of strings."""
        all_names = {"1": "A", "2": "A Inc", "3": "B"}
        duplicate_groups = [
            {"canonical_name": "A", "variant_ids": [1, 2], "variant_names": ["A", "A Inc"]}
        ]

        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.5, random_state=42
        )

        # Check cluster structure
        for cluster in train_clusters + test_clusters:
            assert isinstance(cluster, set)
            for id_ in cluster:
                assert isinstance(id_, str)

    def test_empty_groups(self) -> None:
        """Test handling of dataset with no duplicate groups."""
        all_names = {"1": "A", "2": "B", "3": "C", "4": "D"}
        duplicate_groups = []  # No duplicates

        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.25, random_state=42
        )

        # Should split singletons
        assert len(train_records) == 3  # 75%
        assert len(test_records) == 1  # 25%

        # All clusters should be size 1
        assert all(len(c) == 1 for c in train_clusters)
        assert all(len(c) == 1 for c in test_clusters)

    def test_test_size_edge_cases(self) -> None:
        """Test edge cases for test_size parameter."""
        all_names = {"1": "A", "2": "B", "3": "C"}
        duplicate_groups = []

        # Very small test_size (but should still get at least 1 test sample)
        _, test_records, _, _ = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.01, random_state=42
        )
        assert len(test_records) >= 1

        # Large test_size
        _, test_records, _, _ = stratified_dedup_split(
            all_names, duplicate_groups, test_size=0.99, random_state=42
        )
        assert len(test_records) >= 1
