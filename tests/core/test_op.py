"""Tests for the ``op`` component contract (W2, epic #193).

Covers the keystone contract at the core tier (behavior + edges + errors):

- :class:`Feasible` metadata (exact / algorithm_forced / implied_scope / shape);
- the :class:`Score` (scope + out_space) and :class:`Select` (feasible) roles,
  their validation, and that both stay abstract;
- the ``Select(CLUSTERING)`` guard (refuse without an explicit algorithm, stamp
  ``is_heuristic``) and the exact-feasible path (no algorithm needed);
- :class:`ClusterStage` defaulting its algorithm, marking heuristic, and
  rejecting an unknown one (the same validator ``Select(CLUSTERING)`` uses);
- :meth:`Sequential.check` running at construction — a valid pipeline builds; a
  Select-after-vector-Score, a carrier mismatch and a missing Source all raise a
  problem + fix message.

The stages are trivial test doubles (identity ``forward``\\ s), since this wave
ships the wiring CONTRACT, not the concrete impls or an executor.
"""

import pytest

from langres.core.models import CompanySchema
from langres.core.op import (
    ClusterStage,
    Clusters,
    Feasible,
    Finalize,
    GoldenRecord,
    Op,
    Records,
    Score,
    Select,
    Sequential,
    Source,
)
from langres.core.pairs import Pairs

# --------------------------------------------------------------------------------------
# Test doubles — trivial forwards, one per role/boundary.
# --------------------------------------------------------------------------------------


class _Score(Score[CompanySchema]):
    """A Score whose forward is the identity (same rows, 'new' scores)."""

    def forward(self, pairs: Pairs[CompanySchema]) -> Pairs[CompanySchema]:
        return pairs


class _Select(Select[CompanySchema]):
    """A Select whose forward keeps every row (a trivial subset)."""

    def forward(self, pairs: Pairs[CompanySchema]) -> Pairs[CompanySchema]:
        return pairs


class _Source(Source[CompanySchema]):
    """A Source stage (its forward is never run — check() validates, not executes)."""

    def forward(self, records: Records) -> Pairs[CompanySchema]:
        return Pairs(store={}, rows=[])


class _ClusterStage(ClusterStage[CompanySchema]):
    """A ClusterStage that returns one cluster of every id in the store."""

    def forward(self, pairs: Pairs[CompanySchema]) -> Clusters:
        return [set(pairs.store)]


class _VerifyFinalize(Finalize):
    """A verify Finalize: returns the clusters unchanged."""

    def forward(self, clusters: Clusters) -> Clusters | GoldenRecord:
        return clusters


def _has_problem_and_fix(message: str) -> bool:
    """A wiring error is one message carrying both the problem and a fix hint."""
    return "wiring error" in message.lower() and "fix:" in message.lower()


# --------------------------------------------------------------------------------------
# Feasible — the one parameter that gives selection its many names.
# --------------------------------------------------------------------------------------


def test_clustering_is_inexact_and_forces_an_algorithm():
    assert Feasible.CLUSTERING.exact is False
    assert Feasible.CLUSTERING.algorithm_forced is True
    assert Feasible.CLUSTERING.implied_scope == "global"


@pytest.mark.parametrize(
    "feasible",
    [Feasible.THRESHOLD, Feasible.TOPK, Feasible.LINK, Feasible.ASSIGNMENT],
)
def test_the_four_exact_feasibles_are_exact_and_unforced(feasible):
    assert feasible.exact is True
    assert feasible.algorithm_forced is False


@pytest.mark.parametrize(
    ("feasible", "scope"),
    [
        (Feasible.THRESHOLD, "pair"),
        (Feasible.TOPK, "group"),
        (Feasible.LINK, "group"),
        (Feasible.ASSIGNMENT, "global"),
        (Feasible.CLUSTERING, "global"),
    ],
)
def test_feasible_implies_a_scope_and_carries_a_shape(feasible, scope):
    assert feasible.implied_scope == scope
    assert isinstance(feasible.shape, str) and feasible.shape


# --------------------------------------------------------------------------------------
# Score / Select — the two roles.
# --------------------------------------------------------------------------------------


def test_score_carries_scope_and_out_space():
    score = _Score(scope="pair", out_space="prob_llm")
    assert score.scope == "pair"
    assert score.out_space == "prob_llm"


def test_score_out_space_may_be_vector():
    score = _Score(scope="pair", out_space="vector")
    assert score.out_space == "vector"


@pytest.mark.parametrize("bad_scope", ["record", "PAIR", ""])
def test_score_rejects_an_unknown_scope(bad_scope):
    with pytest.raises(ValueError, match="not a valid scope"):
        _Score(scope=bad_scope, out_space="prob_llm")


@pytest.mark.parametrize("bad_space", ["prob", "cosine", ""])
def test_score_rejects_an_unknown_out_space(bad_space):
    with pytest.raises(ValueError, match="not a known score family"):
        _Score(scope="pair", out_space=bad_space)


def test_select_carries_its_feasible():
    select = _Select(feasible=Feasible.THRESHOLD)
    assert select.feasible is Feasible.THRESHOLD
    assert select.algorithm is None
    assert select.is_heuristic is False


def test_op_score_and_select_stay_abstract():
    # forward is unimplemented on the ABCs, so none is directly instantiable.
    with pytest.raises(TypeError):
        Op()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Score(scope="pair", out_space="prob_llm")  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Select(feasible=Feasible.THRESHOLD)  # type: ignore[abstract]


# --------------------------------------------------------------------------------------
# Select(CLUSTERING) — the escape hatch that forces a named heuristic.
# --------------------------------------------------------------------------------------


def test_select_clustering_without_an_algorithm_refuses_and_names_the_fix():
    with pytest.raises(ValueError) as excinfo:
        _Select(feasible=Feasible.CLUSTERING)
    message = str(excinfo.value)
    assert "not approximable" in message
    assert "transitive_closure" in message  # the fix names a heuristic


def test_select_clustering_with_an_algorithm_constructs_and_is_heuristic():
    select = _Select(feasible=Feasible.CLUSTERING, algorithm="transitive_closure")
    assert select.algorithm == "transitive_closure"
    assert select.is_heuristic is True
    # The heuristic-ness is stamped into the provenance label (THEORY §8).
    assert select.label == {
        "role": "select",
        "feasible": "CLUSTERING",
        "algorithm": "transitive_closure",
        "is_heuristic": True,
    }


def test_select_clustering_rejects_an_unknown_algorithm():
    with pytest.raises(ValueError, match="not a known clustering heuristic"):
        _Select(feasible=Feasible.CLUSTERING, algorithm="kmeans")


def test_select_at_an_exact_feasible_needs_no_algorithm():
    select = _Select(feasible=Feasible.TOPK)
    assert select.algorithm is None
    assert select.is_heuristic is False
    assert select.label["is_heuristic"] is False


# --------------------------------------------------------------------------------------
# ClusterStage — the phase-1 exit, a named heuristic.
# --------------------------------------------------------------------------------------


def test_cluster_stage_defaults_transitive_closure_and_marks_heuristic():
    stage = _ClusterStage()
    assert stage.algorithm == "transitive_closure"
    assert stage.is_heuristic is True
    assert stage.label == {
        "role": "cluster_stage",
        "algorithm": "transitive_closure",
        "is_heuristic": True,
    }


def test_cluster_stage_takes_a_named_algorithm():
    stage = _ClusterStage(algorithm="pivot")
    assert stage.algorithm == "pivot"
    assert stage.is_heuristic is True


def test_cluster_stage_rejects_an_unknown_algorithm():
    # Same validator (and message) a raw Select(CLUSTERING) uses.
    with pytest.raises(ValueError) as excinfo:
        _ClusterStage(algorithm="kmeans")
    message = str(excinfo.value)
    assert "not a known clustering heuristic" in message
    assert "Fix:" in message


# --------------------------------------------------------------------------------------
# Sequential.check() — auto at construction.
# --------------------------------------------------------------------------------------


def test_valid_pipeline_constructs():
    pipeline = Sequential([_Source(), _Score(scope="pair", out_space="prob_llm"), _ClusterStage()])
    assert pipeline.stages  # constructed without raising


def test_valid_pipeline_with_finalize_constructs():
    pipeline = Sequential(
        [
            _Source(),
            _Score(scope="pair", out_space="prob_llm"),
            _ClusterStage(),
            _VerifyFinalize(),
        ]
    )
    assert isinstance(pipeline.stages[-1], Finalize)


def test_select_after_a_vector_score_raises_and_names_the_scalarizer_fix():
    with pytest.raises(ValueError) as excinfo:
        Sequential(
            [
                _Source(),
                _Score(scope="pair", out_space="vector"),
                _Select(feasible=Feasible.TOPK),
                _ClusterStage(),
            ]
        )
    message = str(excinfo.value)
    assert "not orderable" in message
    assert "scalarizer" in message
    assert _has_problem_and_fix(message)


def test_a_scalarizer_score_clears_the_vector_select():
    # vector, then a scalarizer Score (vector -> scalar), then the Select -> ok.
    pipeline = Sequential(
        [
            _Source(),
            _Score(scope="pair", out_space="vector"),
            _Score(scope="pair", out_space="sim_cos"),
            _Select(feasible=Feasible.TOPK),
            _ClusterStage(),
        ]
    )
    assert pipeline.stages


def test_op_after_the_cluster_stage_is_a_carrier_mismatch():
    with pytest.raises(ValueError) as excinfo:
        Sequential(
            [
                _Source(),
                _Score(scope="pair", out_space="prob_llm"),
                _ClusterStage(),
                _Score(scope="pair", out_space="prob_llm"),
            ]
        )
    message = str(excinfo.value)
    assert "out of pipeline order" in message
    assert _has_problem_and_fix(message)


def test_finalize_before_the_cluster_stage_is_a_carrier_mismatch():
    with pytest.raises(ValueError) as excinfo:
        Sequential([_Source(), _VerifyFinalize(), _ClusterStage()])
    assert _has_problem_and_fix(str(excinfo.value))


def test_pipeline_must_start_with_a_source():
    with pytest.raises(ValueError) as excinfo:
        Sequential([_Score(scope="pair", out_space="prob_llm"), _ClusterStage()])
    message = str(excinfo.value)
    assert "must start with a Source" in message
    assert _has_problem_and_fix(message)


def test_an_unknown_stage_type_raises_typeerror():
    class _NotAStage:
        pass

    with pytest.raises(TypeError, match="not a Source, Op"):
        Sequential([_NotAStage()])  # type: ignore[list-item]
