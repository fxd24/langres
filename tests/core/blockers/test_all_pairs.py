"""
Tests for AllPairsBlocker (Approach 1: Classical String Matching).

This test suite validates:
1. Schema normalization: raw dict → CompanySchema
2. Pair generation: N items → N*(N-1)/2 pairs
3. ERCandidate structure: valid blocker_name, left/right entities
4. Handling of missing optional fields (address, phone, website)
"""

import pytest

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.models import CompanySchema, ERCandidate


class TestAllPairsBlockerSchemaValidation:
    """Test schema normalization from raw dicts to CompanySchema."""

    def test_normalizes_complete_record(self):
        """Test that a complete record is correctly normalized to CompanySchema."""
        blocker = AllPairsBlocker()
        data = [
            {
                "id": "c1",
                "name": "Acme Corp",
                "address": "123 Main St",
                "phone": "+1-555-0100",
                "website": "https://acme.com",
            }
        ]

        # Should not raise any errors
        candidates = list(blocker.stream(data))
        # With 1 record, we expect 0 pairs
        assert len(candidates) == 0

    def test_normalizes_minimal_record(self):
        """Test that a minimal record (only required fields) is normalized correctly."""
        blocker = AllPairsBlocker()
        data = [
            {"id": "c1", "name": "Acme Corp"},
            {"id": "c2", "name": "TechStart Inc"},
        ]

        candidates = list(blocker.stream(data))
        assert len(candidates) == 1  # 2 items -> 1 pair

        # Check that optional fields are None
        candidate = candidates[0]
        assert candidate.left.address is None
        assert candidate.left.phone is None
        assert candidate.left.website is None
        assert candidate.right.address is None
        assert candidate.right.phone is None
        assert candidate.right.website is None

    def test_handles_missing_optional_fields(self):
        """Test that missing optional fields (not in dict) are handled gracefully."""
        blocker = AllPairsBlocker()
        data = [
            {"id": "c1", "name": "Company A", "address": "123 Main St"},
            {"id": "c2", "name": "Company B", "phone": "+1-555-0100"},
            {"id": "c3", "name": "Company C", "website": "https://c.com"},
        ]

        candidates = list(blocker.stream(data))
        assert len(candidates) == 3  # 3 items -> 3 pairs

        # Verify schema normalization worked
        assert all(isinstance(c.left, CompanySchema) for c in candidates)
        assert all(isinstance(c.right, CompanySchema) for c in candidates)

    def test_raises_error_on_missing_required_fields(self):
        """Test that missing required fields (id or name) raise validation errors."""
        blocker = AllPairsBlocker()

        # Missing 'id'
        data_missing_id = [{"name": "Acme Corp"}]
        with pytest.raises(Exception):  # Pydantic will raise ValidationError
            list(blocker.stream(data_missing_id))

        # Missing 'name'
        data_missing_name = [{"id": "c1"}]
        with pytest.raises(Exception):  # Pydantic will raise ValidationError
            list(blocker.stream(data_missing_name))


class TestAllPairsBlockerPairGeneration:
    """Test candidate pair generation logic."""

    def test_generates_zero_pairs_for_empty_data(self):
        """Test that empty data generates no pairs."""
        blocker = AllPairsBlocker()
        data = []

        candidates = list(blocker.stream(data))
        assert len(candidates) == 0

    def test_generates_zero_pairs_for_single_record(self):
        """Test that a single record generates no pairs."""
        blocker = AllPairsBlocker()
        data = [{"id": "c1", "name": "Acme Corp"}]

        candidates = list(blocker.stream(data))
        assert len(candidates) == 0

    def test_generates_one_pair_for_two_records(self):
        """Test that two records generate exactly one pair."""
        blocker = AllPairsBlocker()
        data = [
            {"id": "c1", "name": "Acme Corp"},
            {"id": "c2", "name": "TechStart Inc"},
        ]

        candidates = list(blocker.stream(data))
        assert len(candidates) == 1

        # Verify the pair
        candidate = candidates[0]
        assert candidate.left.id == "c1"
        assert candidate.right.id == "c2"

    def test_generates_correct_number_of_pairs(self):
        """Test that N records generate N*(N-1)/2 pairs."""
        blocker = AllPairsBlocker()

        # Test with varying sizes
        test_cases = [
            (2, 1),  # 2 items -> 1 pair
            (3, 3),  # 3 items -> 3 pairs
            (4, 6),  # 4 items -> 6 pairs
            (5, 10),  # 5 items -> 10 pairs
        ]

        for n, expected_pairs in test_cases:
            data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(n)]
            candidates = list(blocker.stream(data))
            assert len(candidates) == expected_pairs

    def test_generates_unique_pairs_no_duplicates(self):
        """Test that all generated pairs are unique (no duplicates)."""
        blocker = AllPairsBlocker()
        data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(5)]

        candidates = list(blocker.stream(data))

        # Check uniqueness by collecting (left_id, right_id) tuples
        pairs = {(c.left.id, c.right.id) for c in candidates}
        assert len(pairs) == len(candidates)  # No duplicates

    def test_never_pairs_entity_with_itself(self):
        """Test that entities are never paired with themselves."""
        blocker = AllPairsBlocker()
        data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(10)]

        candidates = list(blocker.stream(data))

        # Verify no self-pairs
        for candidate in candidates:
            assert candidate.left.id != candidate.right.id


class TestAllPairsBlockerERCandidateStructure:
    """Test that generated ERCandidate objects have correct structure."""

    def test_yields_valid_er_candidate_objects(self):
        """Test that all yielded objects are valid ERCandidate instances."""
        blocker = AllPairsBlocker()
        data = [
            {"id": "c1", "name": "Acme Corp"},
            {"id": "c2", "name": "TechStart Inc"},
        ]

        candidates = list(blocker.stream(data))

        for candidate in candidates:
            assert isinstance(candidate, ERCandidate)
            assert isinstance(candidate.left, CompanySchema)
            assert isinstance(candidate.right, CompanySchema)

    def test_blocker_name_is_set_correctly(self):
        """Test that blocker_name is set to 'all_pairs'."""
        blocker = AllPairsBlocker()
        data = [
            {"id": "c1", "name": "Acme Corp"},
            {"id": "c2", "name": "TechStart Inc"},
        ]

        candidates = list(blocker.stream(data))

        for candidate in candidates:
            assert candidate.blocker_name == "all_pairs"

    def test_left_and_right_are_different_entities(self):
        """Test that left and right entities in each candidate are different."""
        blocker = AllPairsBlocker()
        data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(5)]

        candidates = list(blocker.stream(data))

        for candidate in candidates:
            # They should be different objects with different IDs
            assert candidate.left.id != candidate.right.id
            assert candidate.left is not candidate.right


class TestAllPairsBlockerStreamingBehavior:
    """Test that blocker.stream() returns a proper generator/iterator."""

    def test_stream_returns_iterator(self):
        """Test that stream() returns an iterator (lazy evaluation)."""
        blocker = AllPairsBlocker()
        data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(3)]

        result = blocker.stream(data)

        # Should be a generator/iterator, not a list
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_stream_can_be_consumed_multiple_times(self):
        """Test that calling stream() multiple times works correctly."""
        blocker = AllPairsBlocker()
        data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(3)]

        # First consumption
        candidates1 = list(blocker.stream(data))
        # Second consumption
        candidates2 = list(blocker.stream(data))

        # Should produce the same results
        assert len(candidates1) == len(candidates2)
        assert len(candidates1) == 3  # 3 items -> 3 pairs


class TestAllPairsBlockerWithRealData:
    """Integration tests with realistic company data from fixtures."""

    def test_works_with_fixture_data(self):
        """Test that blocker works with the real company dataset."""
        from tests.fixtures.companies import COMPANY_RECORDS

        blocker = AllPairsBlocker()
        candidates = list(blocker.stream(COMPANY_RECORDS))

        # 15 companies -> 15*14/2 = 105 pairs
        assert len(candidates) == 105

        # Verify all candidates are valid
        for candidate in candidates:
            assert isinstance(candidate, ERCandidate)
            assert candidate.blocker_name == "all_pairs"
            assert candidate.left.id in [c["id"] for c in COMPANY_RECORDS]
            assert candidate.right.id in [c["id"] for c in COMPANY_RECORDS]
