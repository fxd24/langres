"""Tests for updated stratified_dedup_split with dataset object parameter."""

import pytest

from langres.data.schemas import LabeledDeduplicationDataset, LabeledGroup
from langres.data.splitting import stratified_dedup_split


class TestStratifiedDedupSplitWithDataset:
    """Tests for stratified_dedup_split accepting LabeledDeduplicationDataset."""

    def test_split_with_dataset_object(self) -> None:
        """Test that function accepts LabeledDeduplicationDataset object."""
        dataset = LabeledDeduplicationDataset(
            entity_names={
                "1": "Company A",
                "2": "Company A Inc",
                "3": "Company B",
                "4": "Company B Ltd",
                "5": "Company C",
                "6": "Company D",
            },
            labeled_groups=[
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
            ],
        )

        # Split with 50% test size for easier verification
        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            dataset, test_size=0.5, random_state=42
        )

        # Verify all records are accounted for
        train_ids = {r["id"] for r in train_records}
        test_ids = {r["id"] for r in test_records}
        assert train_ids | test_ids == set(dataset.entity_names.keys())

        # Verify no data leakage
        assert len(train_ids & test_ids) == 0

        # Verify clusters match records
        train_cluster_ids = {id_ for cluster in train_clusters for id_ in cluster}
        test_cluster_ids = {id_ for cluster in test_clusters for id_ in cluster}
        assert train_cluster_ids == train_ids
        assert test_cluster_ids == test_ids

    def test_split_preserves_group_integrity(self) -> None:
        """Test that labeled groups are never split across train/test."""
        dataset = LabeledDeduplicationDataset(
            entity_names={
                "1": "Acme",
                "2": "Acme Inc",
                "3": "Acme Corp",
                "4": "Widget Co",
                "5": "Widget Company",
            },
            labeled_groups=[
                LabeledGroup(
                    canonical_name="Acme",
                    entity_ids=["1", "2", "3"],
                    entity_names=["Acme", "Acme Inc", "Acme Corp"],
                ),
                LabeledGroup(
                    canonical_name="Widget",
                    entity_ids=["4", "5"],
                    entity_names=["Widget Co", "Widget Company"],
                ),
            ],
        )

        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            dataset, test_size=0.5, random_state=42
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

    def test_split_with_singletons_only(self) -> None:
        """Test splitting dataset with no labeled groups (all singletons)."""
        dataset = LabeledDeduplicationDataset(
            entity_names={"1": "A", "2": "B", "3": "C", "4": "D"},
            labeled_groups=[],  # No duplicate groups
        )

        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            dataset, test_size=0.25, random_state=42
        )

        # Should split singletons
        assert len(train_records) == 3  # 75%
        assert len(test_records) == 1  # 25%

        # All clusters should be size 1
        assert all(len(c) == 1 for c in train_clusters)
        assert all(len(c) == 1 for c in test_clusters)

    def test_split_stratification_by_cluster_size(self) -> None:
        """Test that cluster size distribution is preserved."""
        entity_names = {}
        labeled_groups = []

        # Add 10 pairs (size 2)
        for i in range(10):
            id1, id2 = str(i * 2 + 1), str(i * 2 + 2)
            entity_names[id1] = f"Pair{i}_A"
            entity_names[id2] = f"Pair{i}_B"
            labeled_groups.append(
                LabeledGroup(
                    canonical_name=f"Pair{i}",
                    entity_ids=[id1, id2],
                    entity_names=[f"Pair{i}_A", f"Pair{i}_B"],
                )
            )

        # Add 10 triples (size 3)
        for i in range(10):
            id1, id2, id3 = str(100 + i * 3), str(101 + i * 3), str(102 + i * 3)
            entity_names[id1] = f"Triple{i}_A"
            entity_names[id2] = f"Triple{i}_B"
            entity_names[id3] = f"Triple{i}_C"
            labeled_groups.append(
                LabeledGroup(
                    canonical_name=f"Triple{i}",
                    entity_ids=[id1, id2, id3],
                    entity_names=[f"Triple{i}_A", f"Triple{i}_B", f"Triple{i}_C"],
                )
            )

        # Add 10 singletons
        for i in range(10):
            id_ = str(200 + i)
            entity_names[id_] = f"Singleton{i}"

        dataset = LabeledDeduplicationDataset(
            entity_names=entity_names,
            labeled_groups=labeled_groups,
        )

        # Split with 20% test
        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            dataset, test_size=0.2, random_state=42
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

    def test_split_reproducibility(self) -> None:
        """Test that same random_state produces same split."""
        dataset = LabeledDeduplicationDataset(
            entity_names={str(i): f"Entity{i}" for i in range(100)},
            labeled_groups=[
                LabeledGroup(
                    canonical_name=f"Group{i}",
                    entity_ids=[str(i * 2), str(i * 2 + 1)],
                    entity_names=[f"Entity{i * 2}", f"Entity{i * 2 + 1}"],
                )
                for i in range(20)
            ],
        )

        # Run twice with same seed
        result1 = stratified_dedup_split(dataset, test_size=0.2, random_state=42)
        result2 = stratified_dedup_split(dataset, test_size=0.2, random_state=42)

        # Should be identical
        train_ids_1 = {r["id"] for r in result1[0]}
        train_ids_2 = {r["id"] for r in result2[0]}
        assert train_ids_1 == train_ids_2

        # Different seed should give different split
        result3 = stratified_dedup_split(dataset, test_size=0.2, random_state=123)
        train_ids_3 = {r["id"] for r in result3[0]}
        assert train_ids_1 != train_ids_3

    def test_split_record_format(self) -> None:
        """Test that output records have correct format."""
        dataset = LabeledDeduplicationDataset(
            entity_names={"1": "Company A", "2": "Company B"},
            labeled_groups=[],
        )

        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            dataset, test_size=0.5, random_state=42
        )

        # Check record structure
        for record in train_records + test_records:
            assert "id" in record
            assert "name" in record
            assert isinstance(record["id"], str)
            assert isinstance(record["name"], str)
            assert record["name"] == dataset.entity_names[record["id"]]

    def test_split_cluster_format(self) -> None:
        """Test that output clusters are sets of strings."""
        dataset = LabeledDeduplicationDataset(
            entity_names={"1": "A", "2": "A Inc", "3": "B"},
            labeled_groups=[
                LabeledGroup(
                    canonical_name="A",
                    entity_ids=["1", "2"],
                    entity_names=["A", "A Inc"],
                )
            ],
        )

        train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
            dataset, test_size=0.5, random_state=42
        )

        # Check cluster structure
        for cluster in train_clusters + test_clusters:
            assert isinstance(cluster, set)
            for id_ in cluster:
                assert isinstance(id_, str)
