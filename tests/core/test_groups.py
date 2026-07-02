"""Tests for ERCandidateGroup and the derived pairwise-to-group helper.

ERCandidateGroup represents "anchor + K candidate members" — the set-wise
input contract that GroupwiseModule.forward_groups() consumes (W1.0
contracts-only; see docs/TECHNICAL_OVERVIEW.md "Group contract"). These tests
are schema-agnostic: both CompanySchema and a local ProductSchema exercise the
model and the derivation helper.
"""

from collections.abc import Iterator

from pydantic import BaseModel

from langres.core.groups import ERCandidateGroup, derive_groups_from_pairs
from langres.core.models import CompanySchema, ERCandidate
from tests.conftest import pairs_from_candidates, pairs_from_groups


class ProductSchema(BaseModel):
    """Second schema for schema-agnostic verification."""

    id: str
    title: str
    price: float | None = None


# ---------------------------------------------------------------------------
# ERCandidateGroup model
# ---------------------------------------------------------------------------


def test_er_candidate_group_holds_anchor_members_and_group_id() -> None:
    """ERCandidateGroup stores anchor, members, and group_id verbatim."""
    anchor = CompanySchema(id="c1", name="Acme Corp")
    members = [CompanySchema(id="c2", name="Acme Corporation"), CompanySchema(id="c3", name="Acme")]

    group = ERCandidateGroup(anchor=anchor, members=members, group_id="c1")

    assert group.anchor == anchor
    assert group.members == members
    assert group.group_id == "c1"


def test_er_candidate_group_is_schema_agnostic_with_product_schema() -> None:
    """ERCandidateGroup works with a second, unrelated schema (ProductSchema)."""
    anchor = ProductSchema(id="p1", title="iPhone 15")
    members = [ProductSchema(id="p2", title="iPhone 15 Pro")]

    group = ERCandidateGroup[ProductSchema](anchor=anchor, members=members, group_id="p1")

    assert isinstance(group.anchor, ProductSchema)
    assert group.members[0].title == "iPhone 15 Pro"


def test_er_candidate_group_allows_empty_members() -> None:
    """A group with zero members is valid (an anchor with no matched candidates)."""
    anchor = CompanySchema(id="c1", name="Acme Corp")

    group = ERCandidateGroup(anchor=anchor, members=[], group_id="c1")

    assert group.members == []


# ---------------------------------------------------------------------------
# derive_groups_from_pairs: the documented buffered/skew-prone default
# ---------------------------------------------------------------------------


def _candidates(pairs: list[tuple[str, str]]) -> Iterator[ERCandidate[CompanySchema]]:
    for left_id, right_id in pairs:
        yield ERCandidate(
            left=CompanySchema(id=left_id, name=f"Company {left_id}"),
            right=CompanySchema(id=right_id, name=f"Company {right_id}"),
            blocker_name="test_blocker",
        )


def test_derive_groups_from_pairs_groups_by_left_id() -> None:
    """One group per unique left.id; members accumulate the paired rights."""
    pairs = [("a", "b"), ("a", "c"), ("d", "e")]

    groups = list(derive_groups_from_pairs(_candidates(pairs)))

    by_anchor = {g.group_id: g for g in groups}
    assert set(by_anchor) == {"a", "d"}
    assert {m.id for m in by_anchor["a"].members} == {"b", "c"}
    assert {m.id for m in by_anchor["d"].members} == {"e"}


def test_derive_groups_from_pairs_handles_empty_stream() -> None:
    """No candidates -> no groups."""
    groups = list(derive_groups_from_pairs(iter([])))
    assert groups == []


def test_derive_groups_from_pairs_preserves_first_seen_order() -> None:
    """Groups are yielded in first-seen anchor order (deterministic)."""
    pairs = [("z", "y"), ("a", "b"), ("z", "x")]

    groups = list(derive_groups_from_pairs(_candidates(pairs)))

    assert [g.group_id for g in groups] == ["z", "a"]


def test_derive_groups_from_pairs_is_schema_agnostic_with_product_schema() -> None:
    """derive_groups_from_pairs works with ProductSchema candidates too."""
    candidates = iter(
        [
            ERCandidate(
                left=ProductSchema(id="p1", title="iPhone"),
                right=ProductSchema(id="p2", title="iPhone Pro"),
                blocker_name="test_blocker",
            )
        ]
    )

    groups = list(derive_groups_from_pairs(candidates))

    assert len(groups) == 1
    assert isinstance(groups[0].anchor, ProductSchema)
    assert groups[0].members[0].title == "iPhone Pro"


def test_derive_groups_from_pairs_pairs_equivalence_property() -> None:
    """Property (CEO #14): pairs recovered from derived groups == pairs from stream.

    No dupes, no losses: flattening the derived groups back into canonical
    (order-independent) pairs must equal the original pair set exactly.
    """
    pairs = [("a", "b"), ("a", "c"), ("d", "e"), ("f", "g"), ("f", "h"), ("f", "i")]
    candidates = list(_candidates(pairs))

    stream_pairs = pairs_from_candidates(candidates)
    groups = list(derive_groups_from_pairs(iter(candidates)))
    group_pairs = pairs_from_groups(groups)

    assert group_pairs == stream_pairs
    assert len(group_pairs) == len(pairs)  # no losses, no dupes
