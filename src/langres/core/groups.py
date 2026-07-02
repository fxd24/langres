"""Set-wise candidate grouping: ERCandidateGroup and the pairwise->group default.

ERCandidateGroup is the set-wise input contract for :class:`~langres.core.module.
GroupwiseModule` (W1.0 contracts; see docs/TECHNICAL_OVERVIEW.md "Group
contract"). It represents "anchor + K candidate members" -- e.g. one LLM call
asking "which of these K candidates match the anchor?" instead of K separate
pairwise calls (the ComEM-style SelectJudge this contract is designed for,
which lands in a later branch).

This module also provides :func:`derive_groups_from_pairs`, the default,
schema-agnostic way to derive groups from an existing pairwise ``ERCandidate``
stream. It is deliberately buffered and anchor-skewed (see its docstring) --
blockers whose native search structure is already per-anchor (e.g.
``VectorBlocker``'s kNN search) should override
``Blocker.stream_groups()`` instead of relying on this derivation.
"""

from collections.abc import Iterator
from typing import Generic, TypeVar

from pydantic import BaseModel

from langres.core.models import ERCandidate

# Generic type variable for schema types (must be a Pydantic model). Defined
# locally (not imported from blocker.py/module.py) to avoid a circular import,
# matching the existing per-module SchemaT convention (see models.py, module.py).
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ERCandidateGroup(BaseModel, Generic[SchemaT]):
    """Anchor + K candidate members: the set-wise input to a GroupwiseModule.

    Attributes:
        anchor: The reference entity all members are compared against.
        members: The K candidate entities being evaluated against the anchor.
        group_id: Identifier for this group, e.g. the anchor's id. Carried
            through to ``PairwiseJudgement.provenance["group_id"]`` by
            :func:`~langres.core.module.stamp_group_cost` so every judgement
            produced from one group call is traceable back to it.
    """

    anchor: SchemaT
    members: list[SchemaT]
    group_id: str


def derive_groups_from_pairs(
    candidates: Iterator[ERCandidate[SchemaT]],
) -> Iterator[ERCandidateGroup[SchemaT]]:
    """Derive per-anchor groups from a pairwise candidate stream.

    BUFFERED AND SKEW-PRONE -- not for benchmark use (E3). Groups pairs by
    their ``left`` entity: each unique ``left.id`` becomes one group's anchor,
    with every paired ``right`` entity as a member. An entity that never
    appears as ``left`` in the stream (e.g. because an upstream blocker
    canonicalizes pair order by id, as ``VectorBlocker.stream()`` and most
    all-pairs blockers do) never becomes its own anchor -- under-representing
    it as a group anchor. This is the "skew" callers must not rely on for
    anchor-fair benchmarking; a blocker with a naturally per-anchor structure
    (e.g. ``VectorBlocker``'s kNN search) should implement
    ``stream_groups()`` natively instead.

    Despite the skew, this derivation is *lossless* over pairs: the set of
    (anchor, member) edges it produces, flattened back to canonical
    order-independent pairs, is exactly the set of pairs in ``candidates`` --
    no dupes, no losses (see the pairs-equivalence property tests in
    ``tests/core/test_groups.py`` / ``tests/core/test_blocker.py``).

    This function fully buffers ``candidates`` into memory (it must see every
    pair sharing an anchor before it can yield that anchor's group), unlike
    the streaming-first pairwise contract elsewhere in langres.

    Args:
        candidates: A pairwise candidate stream, e.g. from ``Blocker.stream()``.

    Yields:
        One ``ERCandidateGroup`` per unique ``left`` entity encountered, in
        first-seen order, with ``group_id`` set to the anchor's ``id``.
    """
    groups: dict[str, tuple[SchemaT, list[SchemaT]]] = {}
    order: list[str] = []
    for candidate in candidates:
        anchor_id = candidate.left.id  # type: ignore[attr-defined]
        if anchor_id not in groups:
            groups[anchor_id] = (candidate.left, [])
            order.append(anchor_id)
        groups[anchor_id][1].append(candidate.right)

    for anchor_id in order:
        anchor, members = groups[anchor_id]
        yield ERCandidateGroup(anchor=anchor, members=members, group_id=anchor_id)
