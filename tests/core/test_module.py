"""
Comprehensive tests for langres.core.Module base class.

The Module is the abstract base class for all entity comparison logic.
These tests validate:
1. Cannot instantiate Module directly (ABC behavior)
2. Can create concrete subclasses implementing forward()
3. forward() accepts and yields correct types (Iterator[ERCandidate] -> Iterator[PairwiseJudgement])
4. Generic typing works correctly with SchemaT
5. Concrete implementations process real data correctly
"""

from collections.abc import Iterator

import numpy as np
import pytest

from langres.core import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.reports import ScoreInspectionReport


class TestModuleAbstractBehavior:
    """Test that Module is a proper abstract base class."""

    def test_cannot_instantiate_module_directly(self):
        """Module is abstract and cannot be instantiated without implementing forward()."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Module()  # type: ignore[abstract]

    def test_cannot_instantiate_module_with_incomplete_implementation(self):
        """Subclass without forward() implementation cannot be instantiated."""

        class IncompleteModule(Module):
            """Missing forward() implementation."""

            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModule()  # type: ignore[abstract]


class TestModuleConcreteImplementation:
    """Test that valid concrete implementations work correctly."""

    def test_can_create_concrete_module_with_forward_implementation(self):
        """Concrete Module with forward() can be instantiated."""

        class ValidModule(Module):
            """Minimal valid Module implementation."""

            def forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]:
                """Yields a judgement for each candidate."""
                for candidate in candidates:
                    yield PairwiseJudgement(
                        left_id=candidate.left.id,
                        right_id=candidate.right.id,
                        score=0.5,
                        score_type="heuristic",
                        decision_step="test_module",
                        provenance={"method": "test"},
                    )

            def inspect_scores(
                self,
                judgements: list[PairwiseJudgement],
                sample_size: int = 10,
            ) -> ScoreInspectionReport:
                """Minimal test fixture implementation."""
                if not judgements:
                    scores = [0.0]
                else:
                    scores = [j.score for j in judgements]

                return ScoreInspectionReport(
                    total_judgements=len(judgements),
                    score_distribution={
                        "mean": float(np.mean(scores)),
                        "median": float(np.median(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                    },
                    high_scoring_examples=[],
                    low_scoring_examples=[],
                    recommendations=["Test fixture - no recommendations"],
                )

        # Should not raise
        module = ValidModule()
        assert module is not None

    def test_forward_method_signature_accepts_iterator(self):
        """forward() accepts Iterator[ERCandidate] as input."""

        class DummyModule(Module):
            def forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]:
                for candidate in candidates:
                    yield PairwiseJudgement(
                        left_id=candidate.left.id,
                        right_id=candidate.right.id,
                        score=0.0,
                        score_type="heuristic",
                        decision_step="dummy",
                        provenance={},
                    )

            def inspect_scores(
                self,
                judgements: list[PairwiseJudgement],
                sample_size: int = 10,
            ) -> ScoreInspectionReport:
                """Minimal test fixture implementation."""
                if not judgements:
                    scores = [0.0]
                else:
                    scores = [j.score for j in judgements]

                return ScoreInspectionReport(
                    total_judgements=len(judgements),
                    score_distribution={
                        "mean": float(np.mean(scores)),
                        "median": float(np.median(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                    },
                    high_scoring_examples=[],
                    low_scoring_examples=[],
                    recommendations=["Test fixture - no recommendations"],
                )

        module = DummyModule()

        # Create test candidates
        candidates = iter(
            [
                ERCandidate(
                    left=CompanySchema(id="1", name="Company A"),
                    right=CompanySchema(id="2", name="Company B"),
                    blocker_name="test_blocker",
                )
            ]
        )

        # Should accept the iterator
        result = module.forward(candidates)
        assert result is not None

    def test_forward_method_returns_iterator(self):
        """forward() returns an Iterator[PairwiseJudgement]."""

        class DummyModule(Module):
            def forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]:
                for candidate in candidates:
                    yield PairwiseJudgement(
                        left_id=candidate.left.id,
                        right_id=candidate.right.id,
                        score=0.0,
                        score_type="heuristic",
                        decision_step="dummy",
                        provenance={},
                    )

            def inspect_scores(
                self,
                judgements: list[PairwiseJudgement],
                sample_size: int = 10,
            ) -> ScoreInspectionReport:
                """Minimal test fixture implementation."""
                if not judgements:
                    scores = [0.0]
                else:
                    scores = [j.score for j in judgements]

                return ScoreInspectionReport(
                    total_judgements=len(judgements),
                    score_distribution={
                        "mean": float(np.mean(scores)),
                        "median": float(np.median(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                    },
                    high_scoring_examples=[],
                    low_scoring_examples=[],
                    recommendations=["Test fixture - no recommendations"],
                )

        module = DummyModule()

        candidates = iter(
            [
                ERCandidate(
                    left=CompanySchema(id="1", name="Company A"),
                    right=CompanySchema(id="2", name="Company B"),
                    blocker_name="test_blocker",
                )
            ]
        )

        result = module.forward(candidates)

        # Result should be an iterator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")


class TestModuleWithRealData:
    """Test Module implementations with real CompanySchema data."""

    def test_module_processes_single_candidate_pair(self):
        """Module can process a single ERCandidate and yield a PairwiseJudgement."""

        class AlwaysMatchModule(Module):
            """Test module that always returns score=1.0."""

            def forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]:
                for candidate in candidates:
                    yield PairwiseJudgement(
                        left_id=candidate.left.id,
                        right_id=candidate.right.id,
                        score=1.0,
                        score_type="heuristic",
                        decision_step="always_match",
                        provenance={"method": "always_match"},
                    )

            def inspect_scores(
                self,
                judgements: list[PairwiseJudgement],
                sample_size: int = 10,
            ) -> ScoreInspectionReport:
                """Minimal test fixture implementation."""
                if not judgements:
                    scores = [0.0]
                else:
                    scores = [j.score for j in judgements]

                return ScoreInspectionReport(
                    total_judgements=len(judgements),
                    score_distribution={
                        "mean": float(np.mean(scores)),
                        "median": float(np.median(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                    },
                    high_scoring_examples=[],
                    low_scoring_examples=[],
                    recommendations=["Test fixture - no recommendations"],
                )

        module = AlwaysMatchModule()

        # Create a test candidate
        candidate = ERCandidate(
            left=CompanySchema(id="acme_1", name="Acme Corp"),
            right=CompanySchema(id="acme_2", name="ACME Corporation"),
            blocker_name="test_blocker",
        )

        # Process and collect results
        results = list(module.forward(iter([candidate])))

        assert len(results) == 1
        judgement = results[0]

        assert isinstance(judgement, PairwiseJudgement)
        assert judgement.left_id == "acme_1"
        assert judgement.right_id == "acme_2"
        assert judgement.score == 1.0
        assert judgement.score_type == "heuristic"
        assert judgement.decision_step == "always_match"
        assert judgement.provenance == {"method": "always_match"}

    def test_module_processes_multiple_candidate_pairs(self):
        """Module can process multiple candidates and yield multiple judgements."""

        class SequentialScoreModule(Module):
            """Test module that assigns sequential scores."""

            def forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]:
                score = 0.0
                for candidate in candidates:
                    yield PairwiseJudgement(
                        left_id=candidate.left.id,
                        right_id=candidate.right.id,
                        score=score,
                        score_type="heuristic",
                        decision_step="sequential",
                        provenance={"index": int(score * 10)},
                    )
                    score += 0.1

            def inspect_scores(
                self,
                judgements: list[PairwiseJudgement],
                sample_size: int = 10,
            ) -> ScoreInspectionReport:
                """Minimal test fixture implementation."""
                if not judgements:
                    scores = [0.0]
                else:
                    scores = [j.score for j in judgements]

                return ScoreInspectionReport(
                    total_judgements=len(judgements),
                    score_distribution={
                        "mean": float(np.mean(scores)),
                        "median": float(np.median(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                    },
                    high_scoring_examples=[],
                    low_scoring_examples=[],
                    recommendations=["Test fixture - no recommendations"],
                )

        module = SequentialScoreModule()

        # Create multiple test candidates
        candidates = [
            ERCandidate(
                left=CompanySchema(id=f"left_{i}", name=f"Company {i}"),
                right=CompanySchema(id=f"right_{i}", name=f"Firm {i}"),
                blocker_name="test_blocker",
            )
            for i in range(5)
        ]

        results = list(module.forward(iter(candidates)))

        assert len(results) == 5

        for i, judgement in enumerate(results):
            assert judgement.left_id == f"left_{i}"
            assert judgement.right_id == f"right_{i}"
            assert judgement.score == pytest.approx(i * 0.1)
            assert judgement.provenance["index"] == i

    def test_module_handles_empty_candidate_stream(self):
        """Module gracefully handles an empty iterator of candidates."""

        class DummyModule(Module):
            def forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]:
                for candidate in candidates:
                    yield PairwiseJudgement(
                        left_id=candidate.left.id,
                        right_id=candidate.right.id,
                        score=0.5,
                        score_type="heuristic",
                        decision_step="dummy",
                        provenance={},
                    )

            def inspect_scores(
                self,
                judgements: list[PairwiseJudgement],
                sample_size: int = 10,
            ) -> ScoreInspectionReport:
                """Minimal test fixture implementation."""
                if not judgements:
                    scores = [0.0]
                else:
                    scores = [j.score for j in judgements]

                return ScoreInspectionReport(
                    total_judgements=len(judgements),
                    score_distribution={
                        "mean": float(np.mean(scores)),
                        "median": float(np.median(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                    },
                    high_scoring_examples=[],
                    low_scoring_examples=[],
                    recommendations=["Test fixture - no recommendations"],
                )

        module = DummyModule()

        # Empty iterator
        results = list(module.forward(iter([])))

        assert results == []

    def test_module_preserves_candidate_data_integrity(self):
        """Module correctly extracts IDs from nested CompanySchema objects."""

        class IdExtractorModule(Module):
            """Module that just extracts and validates ID pairs."""

            def forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]:
                for candidate in candidates:
                    # Verify we can access nested schema attributes
                    assert isinstance(candidate.left, CompanySchema)
                    assert isinstance(candidate.right, CompanySchema)
                    assert candidate.left.name is not None
                    assert candidate.right.name is not None

                    yield PairwiseJudgement(
                        left_id=candidate.left.id,
                        right_id=candidate.right.id,
                        score=0.5,
                        score_type="heuristic",
                        decision_step="id_extractor",
                        provenance={
                            "left_name": candidate.left.name,
                            "right_name": candidate.right.name,
                        },
                    )

            def inspect_scores(
                self,
                judgements: list[PairwiseJudgement],
                sample_size: int = 10,
            ) -> ScoreInspectionReport:
                """Minimal test fixture implementation."""
                if not judgements:
                    scores = [0.0]
                else:
                    scores = [j.score for j in judgements]

                return ScoreInspectionReport(
                    total_judgements=len(judgements),
                    score_distribution={
                        "mean": float(np.mean(scores)),
                        "median": float(np.median(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                    },
                    high_scoring_examples=[],
                    low_scoring_examples=[],
                    recommendations=["Test fixture - no recommendations"],
                )

        module = IdExtractorModule()

        candidate = ERCandidate(
            left=CompanySchema(
                id="comp_123",
                name="Tech Innovations Inc.",
                address="123 Main St",
                phone="555-0100",
            ),
            right=CompanySchema(
                id="comp_456",
                name="Tech Innovations LLC",
                website="https://techinnovations.com",
            ),
            blocker_name="test_blocker",
        )

        results = list(module.forward(iter([candidate])))

        assert len(results) == 1
        judgement = results[0]

        assert judgement.left_id == "comp_123"
        assert judgement.right_id == "comp_456"
        assert judgement.provenance["left_name"] == "Tech Innovations Inc."
        assert judgement.provenance["right_name"] == "Tech Innovations LLC"


class TestModuleStreamingBehavior:
    """Test that Module supports lazy/streaming evaluation."""

    def test_module_forward_is_lazy_generator(self):
        """forward() is a generator and doesn't process until consumed."""

        class CountingModule(Module):
            """Module that tracks how many candidates it has processed."""

            def __init__(self):
                self.processed_count = 0

            def forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]:
                for candidate in candidates:
                    self.processed_count += 1
                    yield PairwiseJudgement(
                        left_id=candidate.left.id,
                        right_id=candidate.right.id,
                        score=0.5,
                        score_type="heuristic",
                        decision_step="counting",
                        provenance={"count": self.processed_count},
                    )

            def inspect_scores(
                self,
                judgements: list[PairwiseJudgement],
                sample_size: int = 10,
            ) -> ScoreInspectionReport:
                """Minimal test fixture implementation."""
                if not judgements:
                    scores = [0.0]
                else:
                    scores = [j.score for j in judgements]

                return ScoreInspectionReport(
                    total_judgements=len(judgements),
                    score_distribution={
                        "mean": float(np.mean(scores)),
                        "median": float(np.median(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                    },
                    high_scoring_examples=[],
                    low_scoring_examples=[],
                    recommendations=["Test fixture - no recommendations"],
                )

        module = CountingModule()

        candidates = [
            ERCandidate(
                left=CompanySchema(id=f"{i}", name=f"Company {i}"),
                right=CompanySchema(id=f"{i + 1}", name=f"Company {i + 1}"),
                blocker_name="test_blocker",
            )
            for i in range(10)
        ]

        # Call forward() but don't consume the iterator
        result_iterator = module.forward(iter(candidates))

        # Generator should not have processed anything yet
        assert module.processed_count == 0

        # Consume just one item
        first_result = next(result_iterator)
        assert module.processed_count == 1
        assert first_result.provenance["count"] == 1

        # Consume the rest
        remaining = list(result_iterator)
        assert len(remaining) == 9
        assert module.processed_count == 10

    def test_module_supports_partial_consumption(self):
        """Can partially consume the judgement stream without processing all candidates."""

        class DummyModule(Module):
            def forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]:
                for candidate in candidates:
                    yield PairwiseJudgement(
                        left_id=candidate.left.id,
                        right_id=candidate.right.id,
                        score=0.5,
                        score_type="heuristic",
                        decision_step="dummy",
                        provenance={},
                    )

            def inspect_scores(
                self,
                judgements: list[PairwiseJudgement],
                sample_size: int = 10,
            ) -> ScoreInspectionReport:
                """Minimal test fixture implementation."""
                if not judgements:
                    scores = [0.0]
                else:
                    scores = [j.score for j in judgements]

                return ScoreInspectionReport(
                    total_judgements=len(judgements),
                    score_distribution={
                        "mean": float(np.mean(scores)),
                        "median": float(np.median(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                    },
                    high_scoring_examples=[],
                    low_scoring_examples=[],
                    recommendations=["Test fixture - no recommendations"],
                )

        module = DummyModule()

        candidates = [
            ERCandidate(
                left=CompanySchema(id=f"{i}", name=f"Company {i}"),
                right=CompanySchema(id=f"{i + 1}", name=f"Company {i + 1}"),
                blocker_name="test_blocker",
            )
            for i in range(100)
        ]

        result_iterator = module.forward(iter(candidates))

        # Only consume first 3 results
        partial_results = [next(result_iterator) for _ in range(3)]

        assert len(partial_results) == 3
        # The iterator should still be valid and could consume more if needed
