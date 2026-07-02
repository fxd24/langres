"""Pytest configuration and shared fixtures for langres tests."""

from collections.abc import Iterable
from typing import Any


def pairs_from_candidates(candidates: Iterable[Any]) -> set[frozenset[str]]:
    """Canonical (order-independent) pair-id set from an ``ERCandidate`` stream.

    Shared by the ``stream()`` vs ``stream_groups()`` pairs-equivalence property
    tests (CEO #14) so both the default derived grouping and VectorBlocker's
    native grouping compare against the same extraction logic.
    """
    return {frozenset([c.left.id, c.right.id]) for c in candidates}


def pairs_from_groups(groups: Iterable[Any]) -> set[frozenset[str]]:
    """Canonical (order-independent) pair-id set from an ``ERCandidateGroup`` stream.

    Flattens each group into (anchor, member) edges. Used opposite
    :func:`pairs_from_candidates` in the pairs-equivalence property tests.
    """
    return {frozenset([group.anchor.id, member.id]) for group in groups for member in group.members}


def edge_list_from_groups(groups: Iterable[Any]) -> list[frozenset[str]]:
    """Canonical (anchor, member) edges from an ``ERCandidateGroup`` stream, AS A LIST.

    Unlike :func:`pairs_from_groups` (which returns a ``set`` and so silently
    collapses a pair that appears in two different groups), this preserves
    duplicates -- use it to assert NO duplicate edge exists across groups
    (``len(edges) == len(set(edges))``), not just that the covered pair SET is
    correct.
    """
    return [frozenset([group.anchor.id, member.id]) for group in groups for member in group.members]


# Add shared fixtures here as needed
