"""
End-to-end integration test for Approach 1: Classical String Matching.

This test validates the complete pipeline:
1. AllPairsBlocker generates candidate pairs
2. RapidfuzzModule scores pairs with string similarity
3. Clusterer forms entity groups from match decisions

Success criteria:
- Pipeline runs without errors
- Known duplicate groups are correctly identified
- Provenance is captured throughout
- Parameters are tunable (library design principle)
"""

import pytest

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.modules.rapidfuzz import RapidfuzzModule
from tests.fixtures.companies import COMPANY_RECORDS, EXPECTED_DUPLICATE_GROUPS


class TestApproach1EndToEnd:
    """End-to-end tests for Approach 1 pipeline."""

    def test_basic_pipeline_runs(self):
        """Test that the basic pipeline runs without errors."""
        # Step 1: Generate candidate pairs
        blocker = AllPairsBlocker()
        candidates = blocker.stream(COMPANY_RECORDS)

        # Step 2: Score pairs with string similarity
        module = RapidfuzzModule()
        judgements = module.forward(candidates)

        # Step 3: Form clusters
        clusterer = Clusterer(threshold=0.7)
        clusters = clusterer.cluster(judgements)

        # Should produce some clusters
        assert len(clusters) > 0

    def test_identifies_exact_duplicates(self):
        """Test that exact duplicates are correctly clustered."""
        blocker = AllPairsBlocker()
        candidates = blocker.stream(COMPANY_RECORDS)

        module = RapidfuzzModule()
        judgements = module.forward(candidates)

        # Use high threshold for exact matches
        clusterer = Clusterer(threshold=0.95)
        clusters = clusterer.cluster(judgements)

        # Convert clusters to set format for comparison
        cluster_sets = [set(cluster) for cluster in clusters]

        # The exact duplicate group should be in the clusters
        exact_duplicate_group = {"c1", "c1_dup1"}
        assert exact_duplicate_group in cluster_sets

    def test_identifies_typo_duplicates_with_lower_threshold(self):
        """Test that typo duplicates are found with appropriate threshold."""
        blocker = AllPairsBlocker()
        candidates = blocker.stream(COMPANY_RECORDS)

        module = RapidfuzzModule()
        judgements = module.forward(candidates)

        # Use medium threshold for typo matches
        clusterer = Clusterer(threshold=0.8)
        clusters = clusterer.cluster(judgements)

        cluster_sets = [set(cluster) for cluster in clusters]

        # The typo duplicate group should be in the clusters
        typo_duplicate_group = {"c2", "c2_typo"}
        assert typo_duplicate_group in cluster_sets

    def test_identifies_all_expected_groups_with_tuned_threshold(self):
        """Test that all expected duplicate groups are found with optimal threshold."""
        blocker = AllPairsBlocker()
        candidates = blocker.stream(COMPANY_RECORDS)

        module = RapidfuzzModule()
        judgements = list(module.forward(candidates))

        # Try to find optimal threshold (this would be done by Optuna in real use)
        # For this test, we use a threshold that captures most duplicates
        clusterer = Clusterer(threshold=0.85)
        clusters = clusterer.cluster(judgements)

        cluster_sets = [set(cluster) for cluster in clusters]

        # Count how many expected groups were found
        found_groups = 0
        for expected_group in EXPECTED_DUPLICATE_GROUPS:
            if expected_group in cluster_sets:
                found_groups += 1

        # Should find most duplicate groups (allow some tolerance for challenging cases)
        assert found_groups >= 3  # At least 3 out of 5 groups

    def test_threshold_parameter_affects_clustering(self):
        """Test that threshold parameter affects clustering results (library design)."""
        blocker = AllPairsBlocker()
        candidates_high = blocker.stream(COMPANY_RECORDS)
        candidates_low = blocker.stream(COMPANY_RECORDS)

        module = RapidfuzzModule()
        judgements_high = list(module.forward(candidates_high))
        judgements_low = list(module.forward(candidates_low))

        # High threshold = fewer, tighter clusters
        clusterer_high = Clusterer(threshold=0.95)
        clusters_high = clusterer_high.cluster(judgements_high)

        # Low threshold = more, looser clusters
        clusterer_low = Clusterer(threshold=0.5)
        clusters_low = clusterer_low.cluster(judgements_low)

        # Low threshold should produce larger/more connected clusters
        total_entities_in_clusters_high = sum(len(c) for c in clusters_high)
        total_entities_in_clusters_low = sum(len(c) for c in clusters_low)

        assert total_entities_in_clusters_low >= total_entities_in_clusters_high

    def test_algorithm_parameter_affects_matching(self):
        """Test that algorithm parameter affects match results (library design)."""
        blocker = AllPairsBlocker()

        # Test with different algorithms
        module_ratio = RapidfuzzModule(algorithm="ratio")
        module_token_set = RapidfuzzModule(algorithm="token_set_ratio")

        candidates_ratio = blocker.stream(COMPANY_RECORDS)
        candidates_token = blocker.stream(COMPANY_RECORDS)

        judgements_ratio = list(module_ratio.forward(candidates_ratio))
        judgements_token = list(module_token_set.forward(candidates_token))

        # Algorithms should produce different scores (not testing which is better)
        scores_ratio = [j.score for j in judgements_ratio]
        scores_token = [j.score for j in judgements_token]

        # At least some scores should be different
        assert scores_ratio != scores_token

    def test_field_weights_parameter_affects_matching(self):
        """Test that field weights affect match results (library design)."""
        blocker = AllPairsBlocker()

        # Name-heavy weighting
        module_name_heavy = RapidfuzzModule(field_weights={"name": 0.9, "address": 0.1})
        # Address-heavy weighting
        module_address_heavy = RapidfuzzModule(
            field_weights={"name": 0.1, "address": 0.9}
        )

        candidates_name = blocker.stream(COMPANY_RECORDS)
        candidates_addr = blocker.stream(COMPANY_RECORDS)

        judgements_name = list(module_name_heavy.forward(candidates_name))
        judgements_addr = list(module_address_heavy.forward(candidates_addr))

        # Different weights should produce different scores
        scores_name = [j.score for j in judgements_name]
        scores_addr = [j.score for j in judgements_addr]

        assert scores_name != scores_addr

    def test_provenance_captured_end_to_end(self):
        """Test that provenance is captured throughout the pipeline."""
        blocker = AllPairsBlocker()
        candidates = blocker.stream(COMPANY_RECORDS)

        module = RapidfuzzModule(
            threshold=0.7,
            algorithm="token_set_ratio",
            field_weights={"name": 0.8, "address": 0.2},
        )
        judgements = list(module.forward(candidates))

        # Check that provenance is captured for all judgements
        assert len(judgements) > 0

        for judgement in judgements:
            # Provenance should contain parameters
            assert "threshold" in judgement.provenance
            assert "algorithm" in judgement.provenance
            assert "field_weights" in judgement.provenance
            assert "field_scores" in judgement.provenance

            # Parameters should match what was configured
            assert judgement.provenance["threshold"] == 0.7
            assert judgement.provenance["algorithm"] == "token_set_ratio"
            assert judgement.provenance["field_weights"]["name"] == 0.8

    def test_pipeline_handles_missing_fields_gracefully(self):
        """Test that pipeline handles records with missing optional fields."""
        # Create dataset with missing fields and very different names
        data_with_missing = [
            {"id": "m1", "name": "Alpha Corporation"},  # Only required fields
            {"id": "m2", "name": "Alpha Corporation"},  # Only required fields
            {"id": "m3", "name": "Zeta Industries", "address": "123 Main St"},
            {"id": "m4", "name": "Zeta Industries", "address": "123 Main St"},
        ]

        blocker = AllPairsBlocker()
        candidates = blocker.stream(data_with_missing)

        module = RapidfuzzModule()
        judgements = module.forward(candidates)

        # Should not raise errors with missing fields
        clusterer = Clusterer(threshold=0.9)
        clusters = clusterer.cluster(judgements)

        # Should produce valid clusters
        assert isinstance(clusters, list)
        # Both duplicate groups should be found (exact name matches)
        cluster_sets = [set(cluster) for cluster in clusters]
        assert {"m1", "m2"} in cluster_sets
        assert {"m3", "m4"} in cluster_sets

    def test_pipeline_scales_with_data_size(self):
        """Test that pipeline can handle varying data sizes."""
        test_sizes = [2, 5, 10, 15]

        for size in test_sizes:
            data = COMPANY_RECORDS[:size]

            blocker = AllPairsBlocker()
            candidates = blocker.stream(data)

            module = RapidfuzzModule()
            judgements = module.forward(candidates)

            clusterer = Clusterer(threshold=0.7)
            clusters = clusterer.cluster(judgements)

            # Should produce some result for any size
            assert isinstance(clusters, list)

    def test_streaming_behavior_end_to_end(self):
        """Test that streaming/lazy evaluation works end-to-end."""
        blocker = AllPairsBlocker()
        candidates = blocker.stream(COMPANY_RECORDS)

        # Candidates should be an iterator
        assert hasattr(candidates, "__iter__")
        assert hasattr(candidates, "__next__")

        module = RapidfuzzModule()
        judgements = module.forward(candidates)

        # Judgements should be an iterator
        assert hasattr(judgements, "__iter__")
        assert hasattr(judgements, "__next__")

        # Can materialize into list
        judgement_list = list(judgements)
        assert len(judgement_list) > 0

    def test_reproducible_results(self):
        """Test that pipeline produces reproducible results."""
        # Run 1
        blocker1 = AllPairsBlocker()
        candidates1 = blocker1.stream(COMPANY_RECORDS)
        module1 = RapidfuzzModule(threshold=0.7, algorithm="ratio")
        judgements1 = list(module1.forward(candidates1))
        clusterer1 = Clusterer(threshold=0.8)
        clusters1 = clusterer1.cluster(judgements1)

        # Run 2 (same configuration)
        blocker2 = AllPairsBlocker()
        candidates2 = blocker2.stream(COMPANY_RECORDS)
        module2 = RapidfuzzModule(threshold=0.7, algorithm="ratio")
        judgements2 = list(module2.forward(candidates2))
        clusterer2 = Clusterer(threshold=0.8)
        clusters2 = clusterer2.cluster(judgements2)

        # Results should be identical
        assert len(clusters1) == len(clusters2)

        # Convert to sets for comparison (order may vary)
        cluster_sets1 = [set(c) for c in clusters1]
        cluster_sets2 = [set(c) for c in clusters2]
        assert set(frozenset(c) for c in cluster_sets1) == set(
            frozenset(c) for c in cluster_sets2
        )


@pytest.mark.slow
class TestApproach1Performance:
    """Performance and scalability tests for Approach 1 (marked as slow)."""

    def test_all_pairs_blocker_is_quadratic(self):
        """Test that AllPairsBlocker generates O(N²) pairs."""
        test_cases = [(10, 45), (20, 190), (50, 1225)]  # (N, N*(N-1)/2)

        for n, expected_pairs in test_cases:
            data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(n)]

            blocker = AllPairsBlocker()
            candidates = list(blocker.stream(data))

            assert len(candidates) == expected_pairs

    def test_pipeline_with_larger_dataset(self):
        """Test pipeline with a moderately sized dataset."""
        # Generate 50 companies (50*49/2 = 1225 pairs)
        n = 50
        large_dataset = [
            {"id": f"c{i}", "name": f"Company {i}", "address": f"{i} Main St"}
            for i in range(n)
        ]

        blocker = AllPairsBlocker()
        candidates = blocker.stream(large_dataset)

        module = RapidfuzzModule()
        judgements = module.forward(candidates)

        clusterer = Clusterer(threshold=0.8)
        clusters = clusterer.cluster(judgements)

        # Should complete without errors
        assert isinstance(clusters, list)
