"""W1-T2: the ERModel spine carries a :class:`Pairs` internally.

The spine (``_candidates`` -> score -> cluster) now threads a
:class:`~langres.core.pairs.Pairs` carrier and bridges back to the legacy
``ERCandidate`` at the matcher boundary (matchers are unmigrated -- that is W2).
These tests pin the bridge: ``_candidates(records)`` is a ``Pairs`` whose
``.to_candidates()`` is field-for-field the ``ERCandidate`` stream the old
generator produced, and ``compare`` (incl. its "blocking cannot veto" fallback)
still returns the same verdicts through the carrier.
"""

import pytest

from langres.core.models import CompanySchema, ERCandidate
from langres.core.pairs import Pairs
from langres.core.resolver import Resolver

_RECORDS = [
    {"id": "1", "name": "Acme Corporation", "address": "1 Main St"},
    {"id": "2", "name": "Acme Corp", "address": "1 Main Street"},
    {"id": "3", "name": "Totally Unrelated Restaurant"},
]


def _resolver() -> Resolver:
    """A cheap, offline default Resolver (AllPairsBlocker + comparator + string matcher)."""
    return Resolver.from_schema(CompanySchema)


def _pre_refactor_candidates(resolver: Resolver, records: list[dict]) -> list[ERCandidate]:
    """Reconstruct exactly what the pre-carrier ``_candidates`` returned.

    The old generator was ``blocker.stream(records)`` with, when a comparator is
    configured, a per-candidate ``comparison`` attached by ``model_copy`` -- the
    ground truth the new ``Pairs`` bridge must reproduce field-for-field.
    """
    candidates = list(resolver.blocker.stream(records))
    if resolver.comparator is not None:
        comparator = resolver.comparator
        candidates = [
            c.model_copy(update={"comparison": comparator.compare(c.left, c.right)})
            for c in candidates
        ]
    return candidates


def test_private_candidates_returns_a_pairs_carrier() -> None:
    """``_candidates`` now yields a ``Pairs`` of unscored, store-bound rows."""
    resolver = _resolver()
    pairs = resolver._candidates(_RECORDS)

    assert isinstance(pairs, Pairs)
    assert len(pairs) == 3  # AllPairs over 3 records -> C(3,2) = 3
    for row in pairs:
        # A freshly-blocked row is "blocked, not yet scored": no judge output.
        assert row.score_type is None
        assert row.decision is None
        # The comparator (wired by from_schema) attached its vector before build.
        assert row.comparison is not None
        # Store-bound: the typed entities materialize on access.
        assert row.left.id in {"1", "2", "3"}
        assert row.right.id in {"1", "2", "3"}


def test_candidates_bridge_equals_the_pre_refactor_stream_with_comparator() -> None:
    """candidates() (Pairs -> to_candidates) == the legacy compared ERCandidate list."""
    resolver = _resolver()
    assert resolver.comparator is not None  # from_schema wires one by default

    expected = _pre_refactor_candidates(resolver, _RECORDS)
    via_bridge = resolver.candidates(_RECORDS)

    # Full Pydantic value equality: ids, blocker_name, similarity_score, comparison.
    assert via_bridge == expected
    assert all(c.comparison is not None for c in via_bridge)


def test_candidates_bridge_equals_the_pre_refactor_stream_without_comparator() -> None:
    """The AllPairs+String pipeline with NO comparator round-trips losslessly too."""
    resolver = _resolver()
    resolver.comparator = None

    expected = _pre_refactor_candidates(resolver, _RECORDS)
    via_bridge = resolver.candidates(_RECORDS)

    assert via_bridge == expected
    assert all(c.comparison is None for c in via_bridge)


def test_public_candidates_is_the_bridge_of_private_candidates() -> None:
    """``candidates()`` is exactly ``_candidates().to_candidates()`` -- one list, same ids."""
    resolver = _resolver()

    via_public = resolver.candidates(_RECORDS)
    via_bridge = resolver._candidates(_RECORDS).to_candidates()

    assert isinstance(via_public, list)
    assert all(isinstance(c, ERCandidate) for c in via_public)
    assert via_public == via_bridge


def test_compare_returns_correct_verdicts_through_the_carrier() -> None:
    """compare() over the single-row Pairs still matches / rejects the right pairs."""
    resolver = _resolver()

    match = resolver.compare(_RECORDS[0], _RECORDS[1])  # Acme Corporation ~ Acme Corp
    non_match = resolver.compare(_RECORDS[0], _RECORDS[2])  # Acme ~ a restaurant

    assert match.match is True
    assert non_match.match is False


def test_pair_candidate_is_a_single_row_pairs() -> None:
    """``_pair_candidate`` returns a one-row ``Pairs`` for the named pair."""
    resolver = _resolver()
    normalized = resolver._prepare([_RECORDS[0], _RECORDS[1]])

    pair = resolver._pair_candidate(normalized)

    assert isinstance(pair, Pairs)
    assert len(pair) == 1
    candidates = pair.to_candidates()
    assert len(candidates) == 1
    assert {candidates[0].left.id, candidates[0].right.id} == {"1", "2"}


def test_blocking_cannot_veto_the_pair_fallback_still_builds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the blocker yields nothing, ``_pair_candidate`` builds the pair directly.

    Blocking is a recall optimization for batches; a caller naming exactly two
    records has decided the pair is worth judging. So a blocker that filters it
    out must NOT silently veto compare() to a no-match -- the carrier routes the
    fallback exactly as before.
    """
    resolver = _resolver()
    # Force the blocker to emit no candidates for any input.
    monkeypatch.setattr(resolver.blocker, "stream", lambda records: iter([]))

    normalized = resolver._prepare([_RECORDS[0], _RECORDS[1]])
    pair = resolver._pair_candidate(normalized)

    assert isinstance(pair, Pairs)
    assert len(pair) == 1  # built directly, not vetoed to empty
    assert pair.rows[0].blocker_name == "compare"
    assert {pair.rows[0].left.id, pair.rows[0].right.id} == {"1", "2"}

    # And compare() end-to-end still returns a verdict rather than a silent no-match.
    verdict = resolver.compare(_RECORDS[0], _RECORDS[1])
    assert verdict.match is True
