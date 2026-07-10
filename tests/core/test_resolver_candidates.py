"""Tests for ``Resolver.candidates()`` -- the public, materialized-list

counterpart to the private ``_candidates()`` generator.

Regression context: ``evaluate_judge_on_candidates`` (``core/benchmark.py``)
both calls ``len(candidates)`` and iterates the candidate sequence TWICE (once
to judge, once to build the graded candidate pairs). Handing it a raw
generator breaks ``len()`` and silently yields nothing on the second pass --
a plausible-but-wrong empty gold-pair set. ``candidates()`` exists to close
that footgun by materializing a list.
"""

from langres.core.models import CompanySchema
from langres.core.resolver import Resolver

_RECORDS = [
    {"id": "1", "name": "Acme Corporation", "address": "1 Main St"},
    {"id": "2", "name": "Acme Corp", "address": "1 Main Street"},
    {"id": "3", "name": "Totally Unrelated Restaurant"},
]


def _resolver() -> Resolver:
    """A cheap, offline default Resolver (AllPairsBlocker + comparator + string judge)."""
    return Resolver.from_schema(CompanySchema)


def _pair_ids(candidates: list) -> set[frozenset[str]]:
    return {frozenset({c.left.id, c.right.id}) for c in candidates}


def test_candidates_returns_a_materialized_list() -> None:
    resolver = _resolver()
    result = resolver.candidates(_RECORDS)
    assert isinstance(result, list)


def test_candidates_len_works() -> None:
    """A generator has no len() -- this is the whole point of materializing."""
    resolver = _resolver()
    result = resolver.candidates(_RECORDS)
    # 3 records, AllPairsBlocker -> C(3,2) = 3 pairs.
    assert len(result) == 3


def test_candidates_iterating_twice_yields_the_same_nonempty_content() -> None:
    """Regression guard: a raw generator returned here would silently yield
    nothing on a second pass -- exactly the bug this method exists to
    prevent (see module docstring)."""
    resolver = _resolver()
    result = resolver.candidates(_RECORDS)

    first_pass = list(result)
    second_pass = list(result)

    assert first_pass == second_pass
    assert len(first_pass) > 0


def test_candidates_matches_private_candidates_generator_content() -> None:
    """Same blocking + comparison-attachment behavior as _candidates(), just
    eagerly materialized rather than lazily streamed."""
    resolver = _resolver()

    via_public = resolver.candidates(_RECORDS)
    via_private = list(resolver._candidates(_RECORDS))

    assert _pair_ids(via_public) == _pair_ids(via_private)
    assert len(via_public) == len(via_private)


def test_candidates_attaches_comparison_vectors_when_comparator_configured() -> None:
    """Resolver.from_schema wires a comparator by default -- candidates() must
    attach comparison vectors exactly like _candidates() does. A caller that
    instead reaches into e.g. ``bench.build_blocker().stream(records)`` gets
    candidates WITHOUT them, silently changing what a comparison-reading judge
    (e.g. WeightedAverageJudge) sees -- see the method's docstring."""
    resolver = _resolver()
    result = resolver.candidates(_RECORDS)

    assert result
    assert all(c.comparison is not None for c in result)


def test_candidates_no_comparison_attached_without_a_comparator() -> None:
    """When the Resolver has no comparator, candidates() (like _candidates())
    must leave `comparison` unset -- not silently attach one."""
    resolver = _resolver()
    resolver.comparator = None
    result = resolver.candidates(_RECORDS)

    assert result
    assert all(c.comparison is None for c in result)


def test_candidates_empty_input_returns_empty_list() -> None:
    resolver = _resolver()
    result = resolver.candidates([])
    assert result == []
