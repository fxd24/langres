"""Tests for Blocker base class."""

from collections.abc import Iterator

import numpy as np
import pytest

from langres.core.blocker import Blocker
from langres.core.groups import ERCandidateGroup
from langres.core.models import CompanySchema, ERCandidate
from langres.core.reports import CandidateInspectionReport
from tests.conftest import pairs_from_candidates, pairs_from_groups


class DummyBlocker(Blocker[CompanySchema]):
    """Test blocker that generates all pairs from a list."""

    def stream(self, data: list[dict[str, str]]) -> Iterator[ERCandidate[CompanySchema]]:
        """Generate all pairs from the data."""
        # Convert raw dicts to CompanySchema
        companies = [
            CompanySchema(id=item["id"], name=item["name"], address=item.get("address"))
            for item in data
        ]

        # Generate all pairs
        for i, left in enumerate(companies):
            for right in companies[i + 1 :]:
                yield ERCandidate(left=left, right=right, blocker_name="dummy_blocker")

    def inspect_candidates(
        self,
        candidates: list[ERCandidate[CompanySchema]],
        entities: list[CompanySchema],
        sample_size: int = 10,
    ) -> CandidateInspectionReport:
        """Minimal test fixture implementation."""
        return CandidateInspectionReport(
            total_candidates=len(candidates),
            avg_candidates_per_entity=len(candidates) / len(entities) if entities else 0.0,
            candidate_distribution={},
            examples=[],
            recommendations=["Test fixture - no recommendations"],
        )


def test_cannot_instantiate_abstract_blocker() -> None:
    """Test that Blocker cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Blocker()  # type: ignore[abstract]


def test_can_create_concrete_blocker() -> None:
    """Test that concrete implementation can be instantiated."""
    blocker = DummyBlocker()
    assert isinstance(blocker, Blocker)


def test_blocker_stream_accepts_list() -> None:
    """Test that stream() accepts a list of raw data."""
    blocker = DummyBlocker()

    data = [
        {"id": "1", "name": "Acme Corp"},
        {"id": "2", "name": "Beta Inc"},
    ]

    result = blocker.stream(data)

    # Result should be an iterator
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


def test_blocker_yields_er_candidates() -> None:
    """Test that stream() yields valid ERCandidate objects."""
    blocker = DummyBlocker()

    data = [
        {"id": "1", "name": "Acme Corp"},
        {"id": "2", "name": "Beta Inc"},
    ]

    candidates = list(blocker.stream(data))

    assert len(candidates) == 1  # Only 1 pair from 2 items

    candidate = candidates[0]
    assert isinstance(candidate, ERCandidate)
    assert candidate.left.id == "1"
    assert candidate.right.id == "2"
    assert candidate.blocker_name == "dummy_blocker"


def test_blocker_generates_multiple_pairs() -> None:
    """Test that blocker generates correct number of pairs."""
    blocker = DummyBlocker()

    data = [
        {"id": "1", "name": "Company A"},
        {"id": "2", "name": "Company B"},
        {"id": "3", "name": "Company C"},
        {"id": "4", "name": "Company D"},
    ]

    candidates = list(blocker.stream(data))

    # Should generate C(4,2) = 6 pairs
    assert len(candidates) == 6

    # All should be ERCandidate objects
    assert all(isinstance(c, ERCandidate) for c in candidates)


def test_blocker_normalizes_schema() -> None:
    """Test that blocker normalizes raw dicts to CompanySchema."""
    blocker = DummyBlocker()

    # Raw data with various fields
    data = [
        {"id": "1", "name": "Acme Corp", "address": "123 Main St"},
        {"id": "2", "name": "Beta Inc"},  # Missing address
    ]

    candidates = list(blocker.stream(data))

    assert len(candidates) == 1

    # Check schema normalization
    candidate = candidates[0]
    assert isinstance(candidate.left, CompanySchema)
    assert isinstance(candidate.right, CompanySchema)
    assert candidate.left.address == "123 Main St"
    assert candidate.right.address is None  # Optional field


def test_blocker_handles_empty_data() -> None:
    """Test that blocker handles empty input gracefully."""
    blocker = DummyBlocker()

    data: list[dict[str, str]] = []
    candidates = list(blocker.stream(data))

    assert candidates == []


def test_blocker_handles_single_record() -> None:
    """Test that blocker handles single record (no pairs)."""
    blocker = DummyBlocker()

    data = [{"id": "1", "name": "Only Company"}]
    candidates = list(blocker.stream(data))

    assert candidates == []  # Can't make pairs from 1 item


def test_blocker_is_lazy_generator() -> None:
    """Test that stream() is a lazy generator."""

    class CountingBlocker(Blocker[CompanySchema]):
        """Blocker that tracks how many pairs it generated."""

        def __init__(self):
            self.pair_count = 0

        def stream(self, data: list[dict[str, str]]) -> Iterator[ERCandidate[CompanySchema]]:
            companies = [CompanySchema(id=item["id"], name=item["name"]) for item in data]

            for i, left in enumerate(companies):
                for right in companies[i + 1 :]:
                    self.pair_count += 1
                    yield ERCandidate(left=left, right=right, blocker_name="counting_blocker")

        def inspect_candidates(
            self,
            candidates: list[ERCandidate[CompanySchema]],
            entities: list[CompanySchema],
            sample_size: int = 10,
        ) -> CandidateInspectionReport:
            """Minimal test fixture implementation."""
            return CandidateInspectionReport(
                total_candidates=len(candidates),
                avg_candidates_per_entity=len(candidates) / len(entities) if entities else 0.0,
                candidate_distribution={},
                examples=[],
                recommendations=["Test fixture - no recommendations"],
            )

    blocker = CountingBlocker()

    data = [{"id": str(i), "name": f"Company {i}"} for i in range(10)]

    # Call stream() but don't consume
    result_iterator = blocker.stream(data)
    assert blocker.pair_count == 0  # Should not process yet

    # Consume just one
    first = next(result_iterator)
    assert blocker.pair_count == 1
    assert first.left.id == "0"

    # Consume the rest
    remaining = list(result_iterator)
    assert len(remaining) == 44  # C(10,2) - 1 = 45 - 1
    assert blocker.pair_count == 45  # C(10,2)


# ---------------------------------------------------------------------------
# stream_groups(): the default, derived-from-stream() implementation (E3).
#
# Documented as BUFFERED and SKEW-PRONE -- not for benchmark use. A blocker
# that needs anchor-fair groups (e.g. VectorBlocker, whose kNN structure is
# already per-anchor) overrides stream_groups() natively instead of relying on
# this default.
# ---------------------------------------------------------------------------


def test_blocker_stream_groups_has_a_concrete_default() -> None:
    """A plain Blocker subclass (only implementing stream()) gets stream_groups() for free."""
    blocker = DummyBlocker()

    data = [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}]

    groups = list(blocker.stream_groups(data))

    assert len(groups) == 1
    assert isinstance(groups[0], ERCandidateGroup)


def test_blocker_stream_groups_default_groups_by_left_id() -> None:
    """The default groups by each pair's left.id (documented left-id skew)."""
    blocker = DummyBlocker()

    data = [
        {"id": "1", "name": "Company 1"},
        {"id": "2", "name": "Company 2"},
        {"id": "3", "name": "Company 3"},
    ]

    groups = list(blocker.stream_groups(data))

    # DummyBlocker's stream() yields i<j pairs: (1,2), (1,3), (2,3).
    # -> "1" anchors {2,3}; "2" anchors {3}; "3" never anchors (all-pairs skew).
    by_anchor = {g.group_id: g for g in groups}
    assert set(by_anchor) == {"1", "2"}
    assert {m.id for m in by_anchor["1"].members} == {"2", "3"}
    assert {m.id for m in by_anchor["2"].members} == {"3"}


def test_blocker_stream_groups_default_handles_empty_data() -> None:
    """Empty input -> no groups."""
    blocker = DummyBlocker()
    assert list(blocker.stream_groups([])) == []


def test_blocker_stream_groups_default_handles_single_record() -> None:
    """A single record produces no pairs and so no groups."""
    blocker = DummyBlocker()
    assert list(blocker.stream_groups([{"id": "1", "name": "Only Company"}])) == []


def test_blocker_stream_groups_default_pairs_equivalence_property() -> None:
    """Property (CEO #14): pairs from stream_groups() == pairs from stream().

    No dupes, no losses -- verified against the SAME blocker/data via the
    default derived implementation.
    """
    blocker = DummyBlocker()
    data = [{"id": str(i), "name": f"Company {i}"} for i in range(6)]

    stream_pairs = pairs_from_candidates(blocker.stream(data))
    group_pairs = pairs_from_groups(blocker.stream_groups(data))

    assert group_pairs == stream_pairs
    assert len(group_pairs) == 15  # C(6, 2)
