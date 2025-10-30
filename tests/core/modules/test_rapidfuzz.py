"""
Tests for RapidfuzzModule (Approach 1: Classical String Matching).

This test suite validates:
1. Exact matches return score ≈ 1.0
2. No match returns score ≈ 0.0
3. Threshold parameter affects decisions
4. Different algorithms (ratio, partial_ratio, token_set_ratio)
5. Field weighting works correctly
6. Missing fields (None values) are handled gracefully
7. Provenance captures all parameters
8. Library design: parameters exposed for optimization
"""

import pytest

from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.modules.rapidfuzz import RapidfuzzModule


class TestRapidfuzzModuleExactMatches:
    """Test that exact matches return high scores."""

    def test_exact_match_returns_near_perfect_score(self):
        """Test that identical companies get score ≈ 1.0."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="Acme Corporation")
        right = CompanySchema(id="c2", name="Acme Corporation")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        assert len(judgements) == 1

        judgement = judgements[0]
        assert judgement.score >= 0.99
        assert judgement.score <= 1.0

    def test_exact_match_with_all_fields(self):
        """Test exact match with all fields populated."""
        module = RapidfuzzModule()

        left = CompanySchema(
            id="c1",
            name="TechStart Inc",
            address="123 Main St",
            phone="+1-555-0100",
            website="https://techstart.io",
        )
        right = CompanySchema(
            id="c2",
            name="TechStart Inc",
            address="123 Main St",
            phone="+1-555-0100",
            website="https://techstart.io",
        )
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]
        assert judgement.score >= 0.99


class TestRapidfuzzModuleNoMatches:
    """Test that completely different entities return low scores."""

    def test_no_match_returns_low_score(self):
        """Test that completely different companies get low scores."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="Acme Corporation")
        right = CompanySchema(id="c2", name="Quantum Dynamics")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        # Different companies should have low similarity
        assert judgement.score < 0.5

    def test_completely_different_returns_near_zero(self):
        """Test that very different companies get score ≈ 0.0."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="A")
        right = CompanySchema(id="c2", name="ZZZZZZZZZZZZZZZZZZZZ")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert judgement.score < 0.2


class TestRapidfuzzModuleThreshold:
    """Test that threshold parameter affects behavior (important for optimization)."""

    def test_threshold_parameter_exists(self):
        """Test that threshold can be set via __init__."""
        # This is critical for library design: parameters must be exposed
        module = RapidfuzzModule(threshold=0.75)
        assert module.threshold == 0.75

    def test_default_threshold(self):
        """Test that default threshold is sensible."""
        module = RapidfuzzModule()
        # Default should be in range [0.0, 1.0]
        assert 0.0 <= module.threshold <= 1.0

    def test_threshold_affects_decision_interpretation(self):
        """Test that threshold is captured in provenance for optimization."""
        module = RapidfuzzModule(threshold=0.8)

        left = CompanySchema(id="c1", name="Acme Corp")
        right = CompanySchema(id="c2", name="Acme Corporation")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        # Threshold should be in provenance for future optimization
        assert "threshold" in judgement.provenance
        assert judgement.provenance["threshold"] == 0.8


class TestRapidfuzzModuleAlgorithms:
    """Test different rapidfuzz algorithms (parameter for optimization)."""

    def test_algorithm_parameter_exists(self):
        """Test that algorithm can be set via __init__."""
        module = RapidfuzzModule(algorithm="partial_ratio")
        assert module.algorithm == "partial_ratio"

    def test_default_algorithm(self):
        """Test that default algorithm is set."""
        module = RapidfuzzModule()
        assert module.algorithm in ["ratio", "partial_ratio", "token_set_ratio"]

    def test_ratio_algorithm_works(self):
        """Test that 'ratio' algorithm works."""
        module = RapidfuzzModule(algorithm="ratio")

        left = CompanySchema(id="c1", name="Acme Corp")
        right = CompanySchema(id="c2", name="Acme Corporation")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert "algorithm" in judgement.provenance
        assert judgement.provenance["algorithm"] == "ratio"
        assert 0.0 <= judgement.score <= 1.0

    def test_partial_ratio_algorithm_works(self):
        """Test that 'partial_ratio' algorithm works."""
        module = RapidfuzzModule(algorithm="partial_ratio")

        left = CompanySchema(id="c1", name="Acme")
        right = CompanySchema(id="c2", name="Acme Corporation Inc.")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert judgement.provenance["algorithm"] == "partial_ratio"
        # partial_ratio should handle substring matching well
        assert judgement.score > 0.5

    def test_token_set_ratio_algorithm_works(self):
        """Test that 'token_set_ratio' algorithm works."""
        module = RapidfuzzModule(algorithm="token_set_ratio")

        left = CompanySchema(id="c1", name="Global Systems Inc")
        right = CompanySchema(id="c2", name="Inc Global Systems")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert judgement.provenance["algorithm"] == "token_set_ratio"
        # token_set_ratio should handle word order differences
        assert judgement.score > 0.8


class TestRapidfuzzModuleFieldWeights:
    """Test field weighting for multi-field matching (critical for optimization)."""

    def test_field_weights_parameter_exists(self):
        """Test that field_weights can be set via __init__."""
        module = RapidfuzzModule(
            field_weights={"name": 0.6, "address": 0.3, "phone": 0.1}
        )
        assert module.field_weights["name"] == 0.6
        assert module.field_weights["address"] == 0.3
        assert module.field_weights["phone"] == 0.1

    def test_default_field_weights(self):
        """Test that default field weights are sensible."""
        module = RapidfuzzModule()
        # Should have weights for name (most important)
        assert "name" in module.field_weights
        assert module.field_weights["name"] > 0

    def test_field_weights_sum_to_one(self):
        """Test that field weights are normalized to sum to 1.0."""
        module = RapidfuzzModule(
            field_weights={"name": 0.6, "address": 0.3, "phone": 0.1}
        )
        total_weight = sum(module.field_weights.values())
        assert abs(total_weight - 1.0) < 0.01  # Allow small floating point error

    def test_field_weights_affect_score(self):
        """Test that field weights actually affect the final score."""
        # Name-heavy weighting
        module_name_heavy = RapidfuzzModule(field_weights={"name": 0.9, "address": 0.1})
        # Address-heavy weighting
        module_address_heavy = RapidfuzzModule(
            field_weights={"name": 0.1, "address": 0.9}
        )

        # Same name, different addresses
        left = CompanySchema(
            id="c1", name="Acme Corp", address="123 Main St, San Francisco, CA"
        )
        right = CompanySchema(
            id="c2", name="Acme Corp", address="456 Other Ave, New York, NY"
        )
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgement_name_heavy = list(module_name_heavy.forward([candidate]))[0]
        judgement_address_heavy = list(module_address_heavy.forward([candidate]))[0]

        # Name-heavy should have higher score (name matches perfectly)
        assert judgement_name_heavy.score > judgement_address_heavy.score

    def test_field_scores_captured_in_provenance(self):
        """Test that individual field scores are captured in provenance."""
        module = RapidfuzzModule(
            field_weights={"name": 0.6, "address": 0.3, "phone": 0.1}
        )

        left = CompanySchema(
            id="c1", name="Acme Corp", address="123 Main St", phone="+1-555-0100"
        )
        right = CompanySchema(
            id="c2", name="Acme Corporation", address="123 Main St", phone="+1-555-0100"
        )
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        # Provenance should contain per-field scores
        assert "field_scores" in judgement.provenance
        field_scores = judgement.provenance["field_scores"]
        assert "name" in field_scores
        assert "address" in field_scores
        assert "phone" in field_scores
        # All field scores should be in [0, 1]
        for score in field_scores.values():
            if score is not None:
                assert 0.0 <= score <= 1.0


class TestRapidfuzzModuleMissingFields:
    """Test handling of missing/None fields."""

    def test_handles_none_fields_gracefully(self):
        """Test that None fields don't cause errors."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="Acme Corp", address=None, phone=None)
        right = CompanySchema(id="c2", name="Acme Corp", address=None, phone=None)
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        # Should not raise errors
        judgements = list(module.forward([candidate]))
        assert len(judgements) == 1

    def test_none_fields_ignored_in_scoring(self):
        """Test that None fields are skipped (not compared)."""
        module = RapidfuzzModule(field_weights={"name": 0.5, "address": 0.5})

        left = CompanySchema(id="c1", name="Acme Corp", address=None)
        right = CompanySchema(id="c2", name="Acme Corp", address="123 Main St")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        # Should still get a reasonable score based on name match
        assert judgement.score > 0.5

        # Field scores should indicate None for missing field
        field_scores = judgement.provenance["field_scores"]
        assert field_scores["name"] > 0.9  # Name matches
        assert field_scores["address"] is None  # Address was None

    def test_both_none_returns_none_for_field(self):
        """Test that when both fields are None, field score is None."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="Acme Corp", address=None)
        right = CompanySchema(id="c2", name="Acme Corp", address=None)
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        field_scores = judgement.provenance["field_scores"]
        assert field_scores["address"] is None


class TestRapidfuzzModuleProvenance:
    """Test that provenance captures all necessary metadata for optimization."""

    def test_provenance_contains_threshold(self):
        """Test that provenance includes threshold parameter."""
        module = RapidfuzzModule(threshold=0.75)

        left = CompanySchema(id="c1", name="Acme")
        right = CompanySchema(id="c2", name="Acme")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert "threshold" in judgement.provenance
        assert judgement.provenance["threshold"] == 0.75

    def test_provenance_contains_algorithm(self):
        """Test that provenance includes algorithm parameter."""
        module = RapidfuzzModule(algorithm="token_set_ratio")

        left = CompanySchema(id="c1", name="Acme")
        right = CompanySchema(id="c2", name="Acme")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert "algorithm" in judgement.provenance
        assert judgement.provenance["algorithm"] == "token_set_ratio"

    def test_provenance_contains_field_weights(self):
        """Test that provenance includes field weights."""
        weights = {"name": 0.7, "address": 0.3}
        module = RapidfuzzModule(field_weights=weights)

        left = CompanySchema(id="c1", name="Acme")
        right = CompanySchema(id="c2", name="Acme")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert "field_weights" in judgement.provenance
        assert judgement.provenance["field_weights"]["name"] == 0.7

    def test_provenance_contains_field_scores(self):
        """Test that provenance includes per-field scores."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="Acme", address="123 Main St")
        right = CompanySchema(id="c2", name="Acme Corp", address="123 Main St")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert "field_scores" in judgement.provenance
        assert isinstance(judgement.provenance["field_scores"], dict)


class TestRapidfuzzModulePairwiseJudgement:
    """Test that PairwiseJudgement structure is correct."""

    def test_yields_valid_pairwise_judgement(self):
        """Test that forward() yields valid PairwiseJudgement objects."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="Acme")
        right = CompanySchema(id="c2", name="Acme")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        assert len(judgements) == 1
        assert isinstance(judgements[0], PairwiseJudgement)

    def test_left_id_and_right_id_match_candidates(self):
        """Test that judgement IDs match the candidate IDs."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="Acme")
        right = CompanySchema(id="c2", name="Acme")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert judgement.left_id == "c1"
        assert judgement.right_id == "c2"

    def test_score_in_valid_range(self):
        """Test that score is in [0.0, 1.0]."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="Acme")
        right = CompanySchema(id="c2", name="XYZ Corp")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert 0.0 <= judgement.score <= 1.0

    def test_score_type_is_heuristic(self):
        """Test that score_type is 'heuristic' for rapidfuzz."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="Acme")
        right = CompanySchema(id="c2", name="Acme")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert judgement.score_type == "heuristic"

    def test_decision_step_is_set(self):
        """Test that decision_step captures the algorithm used."""
        module = RapidfuzzModule(algorithm="token_set_ratio")

        left = CompanySchema(id="c1", name="Acme")
        right = CompanySchema(id="c2", name="Acme")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        assert judgement.decision_step is not None
        assert len(judgement.decision_step) > 0


class TestRapidfuzzModuleStreaming:
    """Test streaming/lazy evaluation behavior."""

    def test_forward_returns_iterator(self):
        """Test that forward() returns an iterator (lazy evaluation)."""
        module = RapidfuzzModule()

        left = CompanySchema(id="c1", name="Acme")
        right = CompanySchema(id="c2", name="Acme")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        result = module.forward([candidate])

        # Should be a generator/iterator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_processes_multiple_candidates(self):
        """Test that forward() processes multiple candidates."""
        module = RapidfuzzModule()

        candidates = [
            ERCandidate(
                left=CompanySchema(id="c1", name="Acme"),
                right=CompanySchema(id="c2", name="Acme"),
                blocker_name="test",
            ),
            ERCandidate(
                left=CompanySchema(id="c3", name="TechStart"),
                right=CompanySchema(id="c4", name="TechStart"),
                blocker_name="test",
            ),
            ERCandidate(
                left=CompanySchema(id="c5", name="Global"),
                right=CompanySchema(id="c6", name="Global"),
                blocker_name="test",
            ),
        ]

        judgements = list(module.forward(candidates))
        assert len(judgements) == 3


class TestRapidfuzzModuleWithRealData:
    """Integration tests with realistic data from fixtures."""

    def test_works_with_exact_duplicates(self):
        """Test with exact duplicate company records."""
        from tests.fixtures.companies import COMPANY_RECORDS

        module = RapidfuzzModule()

        # c1 and c1_dup1 are exact duplicates
        left = CompanySchema(**COMPANY_RECORDS[0])
        right = CompanySchema(**COMPANY_RECORDS[1])
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        # Exact duplicates should have very high score
        assert judgement.score > 0.95

    def test_works_with_typo_duplicates(self):
        """Test with typo duplicate company records."""
        from tests.fixtures.companies import COMPANY_RECORDS

        module = RapidfuzzModule()

        # c2 and c2_typo have a typo in the name
        left = CompanySchema(**COMPANY_RECORDS[2])
        right = CompanySchema(**COMPANY_RECORDS[3])
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        # Typo duplicates should have high score (minor typo, all other fields match)
        assert judgement.score > 0.8

    def test_works_with_non_duplicates(self):
        """Test with completely different companies."""
        from tests.fixtures.companies import COMPANY_RECORDS

        module = RapidfuzzModule()

        # c6 and c7 are different companies
        left = CompanySchema(**COMPANY_RECORDS[10])
        right = CompanySchema(**COMPANY_RECORDS[11])
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        # Non-duplicates should have low score
        assert judgement.score < 0.5


class TestRapidfuzzModuleEdgeCases:
    """Test edge cases and error conditions for 100% coverage."""

    def test_invalid_threshold_below_zero_raises_error(self):
        """Test that threshold < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            RapidfuzzModule(threshold=-0.1)

    def test_invalid_threshold_above_one_raises_error(self):
        """Test that threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            RapidfuzzModule(threshold=1.1)

    def test_all_fields_none_returns_zero_score(self):
        """Test that when all fields are None, score is 0.0."""
        # Both entities have only required fields (all optional fields are None)
        left = CompanySchema(id="c1", name="Test")
        right = CompanySchema(id="c2", name="Test")

        # Use field_weights that exclude name to test the all-None case
        # (All other fields are None by default)
        module_no_name = RapidfuzzModule(
            field_weights={"address": 0.5, "phone": 0.3, "website": 0.2}
        )

        candidate = ERCandidate(left=left, right=right, blocker_name="test")
        judgements = list(module_no_name.forward([candidate]))
        judgement = judgements[0]

        # All weighted fields are None, so score should be 0.0
        assert judgement.score == 0.0

    def test_zero_weight_for_all_fields_returns_zero_score(self):
        """Test that when all fields have zero weight, score is 0.0."""
        # This tests the second edge case in _compute_weighted_score
        module = RapidfuzzModule(
            field_weights={"name": 0.0, "address": 0.0, "phone": 0.0, "website": 0.0}
        )

        left = CompanySchema(id="c1", name="Test", address="123 Main St")
        right = CompanySchema(id="c2", name="Test", address="123 Main St")
        candidate = ERCandidate(left=left, right=right, blocker_name="test")

        judgements = list(module.forward([candidate]))
        judgement = judgements[0]

        # All weights are zero, so score should be 0.0
        assert judgement.score == 0.0
