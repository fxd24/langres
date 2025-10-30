"""
Comprehensive tests for langres.core.models data contracts.

This test suite validates:
- ERCandidate[SchemaT]: Generic Pydantic model for normalized pairs
- PairwiseJudgement: Rich decision output with provenance
- CompanySchema: Test domain model for POC

All tests are written BEFORE implementation to ensure correct specification.
"""

import pytest
from pydantic import BaseModel, ValidationError


class TestCompanySchema:
    """Tests for CompanySchema domain model."""

    def test_company_schema_with_all_fields(self):
        """CompanySchema should instantiate with all fields provided."""
        from langres.core.models import CompanySchema

        company = CompanySchema(
            id="c123",
            name="Acme Corporation",
            address="123 Main St, New York, NY 10001",
            phone="+1-555-0100",
            website="https://acme.com",
        )

        assert company.id == "c123"
        assert company.name == "Acme Corporation"
        assert company.address == "123 Main St, New York, NY 10001"
        assert company.phone == "+1-555-0100"
        assert company.website == "https://acme.com"

    def test_company_schema_with_only_required_fields(self):
        """CompanySchema should work with only id and name."""
        from langres.core.models import CompanySchema

        company = CompanySchema(id="c456", name="Beta LLC")

        assert company.id == "c456"
        assert company.name == "Beta LLC"
        assert company.address is None
        assert company.phone is None
        assert company.website is None

    def test_company_schema_partial_optional_fields(self):
        """CompanySchema should support arbitrary combinations of optional fields."""
        from langres.core.models import CompanySchema

        # Only address provided
        c1 = CompanySchema(id="c1", name="Company One", address="123 Main St")
        assert c1.address == "123 Main St"
        assert c1.phone is None
        assert c1.website is None

        # Only phone provided
        c2 = CompanySchema(id="c2", name="Company Two", phone="+1-555-0200")
        assert c2.address is None
        assert c2.phone == "+1-555-0200"
        assert c2.website is None

        # Phone and website provided
        c3 = CompanySchema(
            id="c3", name="Company Three", phone="+1-555-0300", website="https://c3.com"
        )
        assert c3.address is None
        assert c3.phone == "+1-555-0300"
        assert c3.website == "https://c3.com"

    def test_company_schema_missing_required_fields(self):
        """CompanySchema should fail validation if required fields are missing."""
        from langres.core.models import CompanySchema

        # Missing name
        with pytest.raises(ValidationError) as exc_info:
            CompanySchema(id="c789")
        assert "name" in str(exc_info.value)

        # Missing id
        with pytest.raises(ValidationError) as exc_info:
            CompanySchema(name="Missing ID Corp")
        assert "id" in str(exc_info.value)

        # Missing both
        with pytest.raises(ValidationError):
            CompanySchema()

    def test_company_schema_serialization(self):
        """CompanySchema should serialize to dict correctly."""
        from langres.core.models import CompanySchema

        company = CompanySchema(
            id="c100", name="Test Corp", address="456 Oak Ave", phone="+1-555-0400"
        )

        data = company.model_dump()
        assert data == {
            "id": "c100",
            "name": "Test Corp",
            "address": "456 Oak Ave",
            "phone": "+1-555-0400",
            "website": None,
        }

    def test_company_schema_deserialization(self):
        """CompanySchema should deserialize from dict correctly."""
        from langres.core.models import CompanySchema

        data = {
            "id": "c200",
            "name": "Deserialized Inc",
            "address": "789 Pine St",
            "phone": None,
            "website": "https://deserial.com",
        }

        company = CompanySchema.model_validate(data)
        assert company.id == "c200"
        assert company.name == "Deserialized Inc"
        assert company.address == "789 Pine St"
        assert company.phone is None
        assert company.website == "https://deserial.com"


class TestERCandidate:
    """Tests for ERCandidate[SchemaT] generic model."""

    def test_er_candidate_with_company_schema(self):
        """ERCandidate should work with CompanySchema as the generic type."""
        from langres.core.models import CompanySchema, ERCandidate

        left = CompanySchema(id="c1", name="Acme Corp", address="123 Main St")
        right = CompanySchema(id="c2", name="Acme Corporation", address="123 Main Street")

        candidate = ERCandidate[CompanySchema](
            left=left, right=right, blocker_name="rapidfuzz_blocker"
        )

        assert candidate.left.id == "c1"
        assert candidate.left.name == "Acme Corp"
        assert candidate.right.id == "c2"
        assert candidate.right.name == "Acme Corporation"
        assert candidate.blocker_name == "rapidfuzz_blocker"

    def test_er_candidate_with_custom_schema(self):
        """ERCandidate should work with any Pydantic schema."""

        class ProductSchema(BaseModel):
            id: str
            title: str
            price: float

        from langres.core.models import ERCandidate

        left = ProductSchema(id="p1", title="iPhone 15", price=999.99)
        right = ProductSchema(id="p2", title="iPhone 15 Pro", price=1199.99)

        candidate = ERCandidate[ProductSchema](
            left=left, right=right, blocker_name="embedding_blocker"
        )

        assert candidate.left.title == "iPhone 15"
        assert candidate.right.title == "iPhone 15 Pro"
        assert candidate.blocker_name == "embedding_blocker"

    def test_er_candidate_missing_fields(self):
        """ERCandidate should require all fields."""
        from langres.core.models import CompanySchema, ERCandidate

        left = CompanySchema(id="c1", name="Test")
        right = CompanySchema(id="c2", name="Test2")

        # Missing blocker_name
        with pytest.raises(ValidationError) as exc_info:
            ERCandidate[CompanySchema](left=left, right=right)
        assert "blocker_name" in str(exc_info.value)

        # Missing right
        with pytest.raises(ValidationError) as exc_info:
            ERCandidate[CompanySchema](left=left, blocker_name="test_blocker")
        assert "right" in str(exc_info.value)

        # Missing left
        with pytest.raises(ValidationError) as exc_info:
            ERCandidate[CompanySchema](right=right, blocker_name="test_blocker")
        assert "left" in str(exc_info.value)

    def test_er_candidate_serialization(self):
        """ERCandidate should serialize correctly."""
        from langres.core.models import CompanySchema, ERCandidate

        left = CompanySchema(id="c1", name="Test Left")
        right = CompanySchema(id="c2", name="Test Right")
        candidate = ERCandidate[CompanySchema](left=left, right=right, blocker_name="test_blocker")

        data = candidate.model_dump()
        assert data["left"]["id"] == "c1"
        assert data["left"]["name"] == "Test Left"
        assert data["right"]["id"] == "c2"
        assert data["right"]["name"] == "Test Right"
        assert data["blocker_name"] == "test_blocker"

    def test_er_candidate_deserialization(self):
        """ERCandidate should deserialize correctly."""
        from langres.core.models import CompanySchema, ERCandidate

        data = {
            "left": {
                "id": "c1",
                "name": "Deserialized Left",
                "address": None,
                "phone": None,
                "website": None,
            },
            "right": {
                "id": "c2",
                "name": "Deserialized Right",
                "address": None,
                "phone": None,
                "website": None,
            },
            "blocker_name": "vector_blocker",
        }

        candidate = ERCandidate[CompanySchema].model_validate(data)
        assert candidate.left.id == "c1"
        assert candidate.left.name == "Deserialized Left"
        assert candidate.right.id == "c2"
        assert candidate.right.name == "Deserialized Right"
        assert candidate.blocker_name == "vector_blocker"


class TestPairwiseJudgement:
    """Tests for PairwiseJudgement decision output model."""

    def test_pairwise_judgement_with_all_fields(self):
        """PairwiseJudgement should instantiate with all fields."""
        from langres.core.models import PairwiseJudgement

        judgement = PairwiseJudgement(
            left_id="c1",
            right_id="c2",
            score=0.95,
            score_type="calibrated_prob",
            decision_step="llm_judge",
            reasoning="Both companies have matching names and addresses with minor spelling variations.",
            provenance={"model": "gpt-4", "temperature": 0.0, "rapidfuzz_score": 0.88},
        )

        assert judgement.left_id == "c1"
        assert judgement.right_id == "c2"
        assert judgement.score == 0.95
        assert judgement.score_type == "calibrated_prob"
        assert judgement.decision_step == "llm_judge"
        assert "matching names" in judgement.reasoning
        assert judgement.provenance["model"] == "gpt-4"
        assert judgement.provenance["rapidfuzz_score"] == 0.88

    def test_pairwise_judgement_without_optional_fields(self):
        """PairwiseJudgement should work without optional reasoning field."""
        from langres.core.models import PairwiseJudgement

        judgement = PairwiseJudgement(
            left_id="c3",
            right_id="c4",
            score=0.72,
            score_type="sim_cos",
            decision_step="embedding_similarity",
            provenance={"model": "e5-small", "vector_dim": 384},
        )

        assert judgement.left_id == "c3"
        assert judgement.right_id == "c4"
        assert judgement.score == 0.72
        assert judgement.score_type == "sim_cos"
        assert judgement.decision_step == "embedding_similarity"
        assert judgement.reasoning is None
        assert judgement.provenance["model"] == "e5-small"

    def test_pairwise_judgement_all_score_types(self):
        """PairwiseJudgement should accept all valid score_type literals."""
        from langres.core.models import PairwiseJudgement

        score_types = ["sim_cos", "prob_llm", "heuristic", "calibrated_prob"]

        for score_type in score_types:
            judgement = PairwiseJudgement(
                left_id="c1",
                right_id="c2",
                score=0.5,
                score_type=score_type,
                decision_step="test",
                provenance={},
            )
            assert judgement.score_type == score_type

    def test_pairwise_judgement_invalid_score_type(self):
        """PairwiseJudgement should reject invalid score_type values."""
        from langres.core.models import PairwiseJudgement

        with pytest.raises(ValidationError) as exc_info:
            PairwiseJudgement(
                left_id="c1",
                right_id="c2",
                score=0.5,
                score_type="invalid_type",
                decision_step="test",
                provenance={},
            )
        assert "score_type" in str(exc_info.value)

    def test_pairwise_judgement_score_bounds_valid(self):
        """PairwiseJudgement should accept scores in valid range [0.0, 1.0]."""
        from langres.core.models import PairwiseJudgement

        # Boundary values
        j1 = PairwiseJudgement(
            left_id="c1",
            right_id="c2",
            score=0.0,
            score_type="heuristic",
            decision_step="test",
            provenance={},
        )
        assert j1.score == 0.0

        j2 = PairwiseJudgement(
            left_id="c1",
            right_id="c2",
            score=1.0,
            score_type="heuristic",
            decision_step="test",
            provenance={},
        )
        assert j2.score == 1.0

        # Mid-range value
        j3 = PairwiseJudgement(
            left_id="c1",
            right_id="c2",
            score=0.5432,
            score_type="heuristic",
            decision_step="test",
            provenance={},
        )
        assert j3.score == 0.5432

    def test_pairwise_judgement_score_below_zero(self):
        """PairwiseJudgement should reject scores below 0.0."""
        from langres.core.models import PairwiseJudgement

        with pytest.raises(ValidationError) as exc_info:
            PairwiseJudgement(
                left_id="c1",
                right_id="c2",
                score=-0.1,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )
        # Check that validation error mentions the score field
        error_str = str(exc_info.value).lower()
        assert "score" in error_str or "greater" in error_str or "equal" in error_str

    def test_pairwise_judgement_score_above_one(self):
        """PairwiseJudgement should reject scores above 1.0."""
        from langres.core.models import PairwiseJudgement

        with pytest.raises(ValidationError) as exc_info:
            PairwiseJudgement(
                left_id="c1",
                right_id="c2",
                score=1.1,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )
        # Check that validation error mentions the score field
        error_str = str(exc_info.value).lower()
        assert "score" in error_str or "less" in error_str or "equal" in error_str

    def test_pairwise_judgement_missing_required_fields(self):
        """PairwiseJudgement should require all non-optional fields."""
        from langres.core.models import PairwiseJudgement

        # Missing left_id
        with pytest.raises(ValidationError) as exc_info:
            PairwiseJudgement(
                right_id="c2",
                score=0.5,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )
        assert "left_id" in str(exc_info.value)

        # Missing right_id
        with pytest.raises(ValidationError) as exc_info:
            PairwiseJudgement(
                left_id="c1",
                score=0.5,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )
        assert "right_id" in str(exc_info.value)

        # Missing score
        with pytest.raises(ValidationError) as exc_info:
            PairwiseJudgement(
                left_id="c1",
                right_id="c2",
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )
        assert "score" in str(exc_info.value)

        # Missing decision_step
        with pytest.raises(ValidationError) as exc_info:
            PairwiseJudgement(
                left_id="c1",
                right_id="c2",
                score=0.5,
                score_type="heuristic",
                provenance={},
            )
        assert "decision_step" in str(exc_info.value)

        # Missing provenance
        with pytest.raises(ValidationError) as exc_info:
            PairwiseJudgement(
                left_id="c1",
                right_id="c2",
                score=0.5,
                score_type="heuristic",
                decision_step="test",
            )
        assert "provenance" in str(exc_info.value)

    def test_pairwise_judgement_provenance_flexible_dict(self):
        """PairwiseJudgement provenance should accept any dict structure."""
        from langres.core.models import PairwiseJudgement

        # Simple provenance
        j1 = PairwiseJudgement(
            left_id="c1",
            right_id="c2",
            score=0.5,
            score_type="heuristic",
            decision_step="test",
            provenance={"step": 1},
        )
        assert j1.provenance == {"step": 1}

        # Complex nested provenance
        j2 = PairwiseJudgement(
            left_id="c1",
            right_id="c2",
            score=0.5,
            score_type="heuristic",
            decision_step="test",
            provenance={
                "model": "gpt-4",
                "config": {"temperature": 0.0, "max_tokens": 100},
                "scores": [0.1, 0.2, 0.3],
                "metadata": {"run_id": "abc123", "version": "1.0"},
            },
        )
        assert j2.provenance["model"] == "gpt-4"
        assert j2.provenance["config"]["temperature"] == 0.0
        assert j2.provenance["scores"] == [0.1, 0.2, 0.3]

        # Empty provenance
        j3 = PairwiseJudgement(
            left_id="c1",
            right_id="c2",
            score=0.5,
            score_type="heuristic",
            decision_step="test",
            provenance={},
        )
        assert j3.provenance == {}

    def test_pairwise_judgement_serialization(self):
        """PairwiseJudgement should serialize correctly."""
        from langres.core.models import PairwiseJudgement

        judgement = PairwiseJudgement(
            left_id="c1",
            right_id="c2",
            score=0.85,
            score_type="prob_llm",
            decision_step="cascade_llm",
            reasoning="High confidence match",
            provenance={"model": "claude-3", "cost_usd": 0.002},
        )

        data = judgement.model_dump()
        assert data["left_id"] == "c1"
        assert data["right_id"] == "c2"
        assert data["score"] == 0.85
        assert data["score_type"] == "prob_llm"
        assert data["decision_step"] == "cascade_llm"
        assert data["reasoning"] == "High confidence match"
        assert data["provenance"]["model"] == "claude-3"
        assert data["provenance"]["cost_usd"] == 0.002

    def test_pairwise_judgement_deserialization(self):
        """PairwiseJudgement should deserialize correctly."""
        from langres.core.models import PairwiseJudgement

        data = {
            "left_id": "c10",
            "right_id": "c20",
            "score": 0.42,
            "score_type": "sim_cos",
            "decision_step": "embedding_filter",
            "reasoning": None,
            "provenance": {"embedding_model": "bge-large", "dimension": 1024},
        }

        judgement = PairwiseJudgement.model_validate(data)
        assert judgement.left_id == "c10"
        assert judgement.right_id == "c20"
        assert judgement.score == 0.42
        assert judgement.score_type == "sim_cos"
        assert judgement.decision_step == "embedding_filter"
        assert judgement.reasoning is None
        assert judgement.provenance["embedding_model"] == "bge-large"
