"""
Basic usage example for langres.core.models data contracts.

This example demonstrates how to use the foundational data types:
- CompanySchema: Domain model for company entities
- ERCandidate: Generic container for entity pairs
- PairwiseJudgement: Rich decision output with provenance
"""

from langres import CompanySchema, ERCandidate, PairwiseJudgement


def main() -> None:
    """Demonstrate basic usage of langres data contracts."""
    print("=" * 60)
    print("langres.core.models - Basic Usage Example")
    print("=" * 60)

    # 1. Create company entities
    print("\n1. Creating company entities...")
    company_a = CompanySchema(
        id="c001",
        name="Acme Corp",
        address="123 Main St, New York, NY 10001",
        phone="+1-555-0100",
        website="https://acme.com",
    )
    company_b = CompanySchema(
        id="c002",
        name="Acme Corporation",
        address="123 Main Street, New York, NY 10001",
        phone="+1-555-0100",
    )  # Note: missing website field

    print(f"  Company A: {company_a.name} (ID: {company_a.id})")
    print(f"  Company B: {company_b.name} (ID: {company_b.id})")

    # 2. Create a candidate pair
    print("\n2. Creating ERCandidate pair...")
    candidate = ERCandidate[CompanySchema](
        left=company_a, right=company_b, blocker_name="rapidfuzz_blocker"
    )

    print(f"  Blocker: {candidate.blocker_name}")
    print(f"  Left entity: {candidate.left.name}")
    print(f"  Right entity: {candidate.right.name}")

    # 3. Create a pairwise judgement (simulating a Module's decision)
    print("\n3. Creating PairwiseJudgement...")
    judgement = PairwiseJudgement(
        left_id=candidate.left.id,
        right_id=candidate.right.id,
        score=0.92,
        score_type="calibrated_prob",
        decision_step="string_similarity",
        reasoning="High name similarity (0.95) and exact phone match",
        provenance={
            "name_score": 0.95,
            "address_score": 0.88,
            "phone_match": True,
            "website_match": False,
            "method": "rapidfuzz.fuzz.WRatio",
        },
    )

    print(f"  Match score: {judgement.score}")
    print(f"  Score type: {judgement.score_type}")
    print(f"  Decision step: {judgement.decision_step}")
    print(f"  Reasoning: {judgement.reasoning}")
    print(f"  Provenance keys: {list(judgement.provenance.keys())}")

    # 4. Serialize to dict
    print("\n4. Serialization example...")
    judgement_dict = judgement.model_dump()
    print(f"  Serialized keys: {list(judgement_dict.keys())}")
    print(f"  JSON-ready: {judgement_dict['score']} ({type(judgement_dict['score'])})")

    # 5. Deserialize from dict
    print("\n5. Deserialization example...")
    restored = PairwiseJudgement.model_validate(judgement_dict)
    print(f"  Restored score: {restored.score}")
    print(f"  Type preserved: {type(restored.score)}")
    print(f"  Validation works: {restored.score == judgement.score}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
