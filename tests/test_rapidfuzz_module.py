"""Tests for RapidfuzzModule (Approach 1: Classical String Matching).

The RapidfuzzModule is a schema-agnostic module that computes string similarity
scores using rapidfuzz. It accepts field extractors and weights to work with any
Pydantic schema type.
"""

from pydantic import BaseModel

from langres.core.models import CompanySchema, ERCandidate
from langres.core.modules.rapidfuzz import RapidfuzzModule


# Test schema: Product (demonstrates schema-agnostic design)
class ProductSchema(BaseModel):
    """Product schema for testing schema-agnostic module."""

    id: str
    title: str
    brand: str | None = None
    price: float | None = None


def test_rapidfuzz_module_with_company_schema():
    """Test RapidfuzzModule works with CompanySchema."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 0.7),
            "address": (lambda x: x.address or "", 0.3),
        },
        threshold=0.5,
    )

    # Create test candidates
    left = CompanySchema(id="c1", name="Acme Corporation", address="123 Main St")
    right = CompanySchema(id="c2", name="Acme Corp", address="123 Main Street")

    candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]

    # Run module
    judgements = list(module.forward(iter(candidates)))

    # Verify output
    assert len(judgements) == 1
    judgement = judgements[0]

    assert judgement.left_id == "c1"
    assert judgement.right_id == "c2"
    assert 0.0 <= judgement.score <= 1.0
    assert judgement.score_type == "heuristic"
    assert judgement.decision_step == "rapidfuzz_weighted"

    # Verify provenance
    assert "field_scores" in judgement.provenance
    assert "name" in judgement.provenance["field_scores"]
    assert "address" in judgement.provenance["field_scores"]


def test_rapidfuzz_module_with_product_schema():
    """Test RapidfuzzModule works with different schema (ProductSchema)."""
    module = RapidfuzzModule(
        field_extractors={
            "title": (lambda x: x.title, 0.8),
            "brand": (lambda x: x.brand or "", 0.2),
        },
        threshold=0.6,
    )

    # Create test candidates
    left = ProductSchema(id="p1", title="iPhone 15 Pro", brand="Apple")
    right = ProductSchema(id="p2", title="iPhone 15 Pro Max", brand="Apple")

    candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]

    # Run module
    judgements = list(module.forward(iter(candidates)))

    # Verify output
    assert len(judgements) == 1
    judgement = judgements[0]

    assert judgement.left_id == "p1"
    assert judgement.right_id == "p2"
    assert judgement.score > 0.6  # Should be high similarity
    assert judgement.score_type == "heuristic"

    # Verify provenance has field scores
    assert "title" in judgement.provenance["field_scores"]
    assert "brand" in judgement.provenance["field_scores"]


def test_rapidfuzz_module_exact_match():
    """Test RapidfuzzModule gives perfect score for exact matches."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 1.0),
        },
        threshold=0.5,
    )

    left = CompanySchema(id="c1", name="Acme Corporation")
    right = CompanySchema(id="c2", name="Acme Corporation")

    candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]
    judgements = list(module.forward(iter(candidates)))

    assert len(judgements) == 1
    assert judgements[0].score == 1.0


def test_rapidfuzz_module_completely_different():
    """Test RapidfuzzModule gives low score for completely different strings."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 1.0),
        },
        threshold=0.0,
    )

    left = CompanySchema(id="c1", name="Acme Corporation")
    right = CompanySchema(id="c2", name="XYZ Industries")

    candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]
    judgements = list(module.forward(iter(candidates)))

    assert len(judgements) == 1
    assert judgements[0].score < 0.5  # Should be low similarity


def test_rapidfuzz_module_weighted_fields():
    """Test RapidfuzzModule correctly applies field weights."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 0.8),
            "address": (lambda x: x.address or "", 0.2),
        },
        threshold=0.0,
    )

    # Same name, different address
    left = CompanySchema(id="c1", name="Acme Corporation", address="123 Main St")
    right = CompanySchema(id="c2", name="Acme Corporation", address="456 Oak Ave")

    candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]
    judgements = list(module.forward(iter(candidates)))

    # Score should be high because name (80% weight) is exact match
    assert judgements[0].score > 0.8


def test_rapidfuzz_module_multiple_candidates():
    """Test RapidfuzzModule processes multiple candidates correctly."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 1.0),
        },
        threshold=0.0,
    )

    candidates = [
        ERCandidate(
            left=CompanySchema(id="c1", name="Acme Corp"),
            right=CompanySchema(id="c2", name="Acme Corporation"),
            blocker_name="test",
        ),
        ERCandidate(
            left=CompanySchema(id="c3", name="TechStart"),
            right=CompanySchema(id="c4", name="TechStrat"),
            blocker_name="test",
        ),
        ERCandidate(
            left=CompanySchema(id="c5", name="DataFlow"),
            right=CompanySchema(id="c6", name="DataFlow"),
            blocker_name="test",
        ),
    ]

    judgements = list(module.forward(iter(candidates)))

    # Should produce 3 judgements
    assert len(judgements) == 3

    # Check IDs are preserved
    assert judgements[0].left_id == "c1"
    assert judgements[0].right_id == "c2"
    assert judgements[1].left_id == "c3"
    assert judgements[1].right_id == "c4"
    assert judgements[2].left_id == "c5"
    assert judgements[2].right_id == "c6"

    # Third pair should have perfect score (exact match)
    assert judgements[2].score == 1.0


def test_rapidfuzz_module_streaming_behavior():
    """Test RapidfuzzModule returns a generator (streaming)."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 1.0),
        },
        threshold=0.5,
    )

    candidates = [
        ERCandidate(
            left=CompanySchema(id="c1", name="Acme Corp"),
            right=CompanySchema(id="c2", name="Acme Corporation"),
            blocker_name="test",
        )
    ]

    result = module.forward(iter(candidates))

    # forward() should return a generator
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


def test_rapidfuzz_module_threshold_parameter():
    """Test RapidfuzzModule threshold parameter is stored."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 1.0),
        },
        threshold=0.75,
    )

    assert module.threshold == 0.75


def test_rapidfuzz_module_algorithm_parameter():
    """Test RapidfuzzModule accepts algorithm parameter."""
    # Test with different algorithms
    for algorithm in ["ratio", "token_sort_ratio", "token_set_ratio"]:
        module = RapidfuzzModule(
            field_extractors={
                "name": (lambda x: x.name, 1.0),
            },
            threshold=0.5,
            algorithm=algorithm,
        )

        assert module.algorithm == algorithm

        # Verify it can process candidates
        left = CompanySchema(id="c1", name="Acme Corporation")
        right = CompanySchema(id="c2", name="Acme Corp")
        candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]

        judgements = list(module.forward(iter(candidates)))
        assert len(judgements) == 1


def test_rapidfuzz_module_handles_empty_fields():
    """Test RapidfuzzModule handles None/empty fields gracefully."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 0.6),
            "address": (lambda x: x.address or "", 0.4),
        },
        threshold=0.0,
    )

    # One has address, one doesn't
    left = CompanySchema(id="c1", name="Acme Corp", address="123 Main St")
    right = CompanySchema(id="c2", name="Acme Corp", address=None)

    candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]
    judgements = list(module.forward(iter(candidates)))

    # Should still produce a judgement
    assert len(judgements) == 1
    # Score should be lower than perfect match due to missing address
    assert judgements[0].score < 1.0


def test_rapidfuzz_module_provenance_includes_algorithm():
    """Test RapidfuzzModule includes algorithm in provenance."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 1.0),
        },
        threshold=0.5,
        algorithm="token_sort_ratio",
    )

    left = CompanySchema(id="c1", name="Acme Corporation")
    right = CompanySchema(id="c2", name="Corporation Acme")

    candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]
    judgements = list(module.forward(iter(candidates)))

    assert "algorithm" in judgements[0].provenance
    assert judgements[0].provenance["algorithm"] == "token_sort_ratio"


def test_rapidfuzz_module_typo_detection():
    """Test RapidfuzzModule can detect typos with reasonable similarity."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 1.0),
        },
        threshold=0.5,
    )

    # "TechStart" vs "TechStrat" (one char different)
    left = CompanySchema(id="c1", name="TechStart Industries")
    right = CompanySchema(id="c2", name="TechStrat Industries")

    candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]
    judgements = list(module.forward(iter(candidates)))

    # Should have reasonably high similarity (typo detection)
    assert judgements[0].score > 0.8


def test_rapidfuzz_module_abbreviation_detection():
    """Test RapidfuzzModule handles abbreviations."""
    module = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 1.0),
        },
        threshold=0.5,
    )

    # "Incorporated" vs "Inc."
    left = CompanySchema(id="c1", name="Global Systems Incorporated")
    right = CompanySchema(id="c2", name="Global Systems Inc.")

    candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]
    judgements = list(module.forward(iter(candidates)))

    # Should have high similarity (abbreviation)
    assert judgements[0].score > 0.7


def test_rapidfuzz_module_field_extractors_are_configurable():
    """Test RapidfuzzModule accepts arbitrary field extractors."""

    # Complex extractor: combine multiple fields
    def full_name_extractor(x: CompanySchema) -> str:
        return f"{x.name} {x.address or ''}"

    module = RapidfuzzModule(
        field_extractors={
            "full_name": (full_name_extractor, 1.0),
        },
        threshold=0.5,
    )

    left = CompanySchema(id="c1", name="Acme Corp", address="123 Main St")
    right = CompanySchema(id="c2", name="Acme Corp", address="123 Main Street")

    candidates = [ERCandidate(left=left, right=right, blocker_name="test_blocker")]
    judgements = list(module.forward(iter(candidates)))

    assert len(judgements) == 1
    assert "full_name" in judgements[0].provenance["field_scores"]


def test_rapidfuzz_module_invalid_threshold_raises_error():
    """Test RapidfuzzModule raises ValueError for invalid threshold."""
    import pytest

    # Threshold too low
    with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
        RapidfuzzModule(
            field_extractors={"name": (lambda x: x.name, 1.0)}, threshold=-0.1
        )

    # Threshold too high
    with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
        RapidfuzzModule(
            field_extractors={"name": (lambda x: x.name, 1.0)}, threshold=1.5
        )


def test_rapidfuzz_module_invalid_algorithm_raises_error():
    """Test RapidfuzzModule raises ValueError for unsupported algorithm."""
    import pytest

    with pytest.raises(
        ValueError,
        match="algorithm must be one of: ratio, token_sort_ratio, token_set_ratio",
    ):
        RapidfuzzModule(
            field_extractors={"name": (lambda x: x.name, 1.0)},
            threshold=0.5,
            algorithm="invalid_algorithm",  # type: ignore[arg-type]
        )
