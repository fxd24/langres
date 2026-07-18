"""Tests for the ``op`` component contract (W2, epic #193).

Covers the keystone contract at the core tier (behavior + edges + errors):

- :class:`Feasible` metadata (exact / algorithm_forced / implied_scope / shape);
- the :class:`Score` (scope + out_space) and :class:`Select` (feasible) roles,
  their validation, and that both stay abstract;
- the ``Select(CLUSTERING)`` guard (refuse without an explicit algorithm, stamp
  ``is_heuristic``) and the exact-feasible path (no algorithm needed);
- :class:`ClusterStage` defaulting its algorithm, marking heuristic, and
  rejecting an unknown one (the same validator ``Select(CLUSTERING)`` uses);
- the concrete W3-b Selects — :class:`ThresholdSelect` (keep the
  ``predicted_match`` rows: decision-wins, abstain dropped) and
  :class:`TopKSelect` (keep the k best per ``left_id``);
- the ``"unknown"`` OutSpace sentinel — a scalar-family placeholder a
  :class:`Score` may declare, so a :class:`Select` legally follows it while a
  ``"vector"`` Score still raises;
- :meth:`Sequential.check` running at construction — a valid pipeline builds; a
  Select-after-vector-Score, a carrier mismatch and a missing Source all raise a
  problem + fix message.

The boundary/``Score`` stages are trivial test doubles (identity ``forward``\\ s),
since this contract ships no executor; the two concrete Selects are real and
their ``forward`` selection semantics are exercised on built ``Pairs``.
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
    ThresholdSelect,
    TopKSelect,
)
from langres.core.pairs import PairRow, Pairs

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


# --------------------------------------------------------------------------------------
# Concrete Selects — ThresholdSelect / TopKSelect (W3-b).
# --------------------------------------------------------------------------------------


def _scored(
    left: str, right: str, score: float | None, *, decision: bool | None = None
) -> PairRow[CompanySchema]:
    """A SCORED heuristic-family row — the input a Select consumes (score_type set)."""
    return PairRow(
        left_id=left,
        right_id=right,
        blocker_name="test",
        score=score,
        score_type="heuristic",
        decision=decision,
    )


def _pairs(rows: list[PairRow[CompanySchema]]) -> Pairs[CompanySchema]:
    """A Pairs over a minimal store (a Select reads ids/score/decision, not entities)."""
    ids = {rid for row in rows for rid in (row.left_id, row.right_id)}
    return Pairs(store={i: CompanySchema(id=i, name=i) for i in ids}, rows=rows)


def test_threshold_and_topk_construct_at_their_feasible():
    thr = ThresholdSelect[CompanySchema](0.7)
    assert isinstance(thr, Select) and isinstance(thr, Op)
    assert thr.feasible is Feasible.THRESHOLD
    assert thr.threshold == 0.7
    assert thr.algorithm is None and thr.is_heuristic is False  # exact feasible

    topk = TopKSelect[CompanySchema](3)
    assert isinstance(topk, Select) and isinstance(topk, Op)
    assert topk.feasible is Feasible.TOPK
    assert topk.k == 3
    assert topk.algorithm is None and topk.is_heuristic is False


def test_threshold_select_keeps_exactly_the_predicted_matches():
    """Keep the ``predicted_match(t) is True`` rows: abstain -> None dropped, decision wins."""
    pairs = _pairs(
        [
            _scored("a", "b", 0.9),  # clears the price -> kept
            _scored("a", "c", 0.3),  # below the price -> dropped
            _scored("d", "e", None),  # abstain (no score, no decision) -> dropped, not a "no"
            _scored("f", "g", 0.1, decision=True),  # decision wins over a low score -> kept
            _scored("h", "i", 0.99, decision=False),  # decision wins over a high score -> dropped
        ]
    )
    kept = ThresholdSelect[CompanySchema](0.5).forward(pairs)

    # Exactly the predicted matches, in input order.
    assert [(r.left_id, r.right_id) for r in kept.rows] == [("a", "b"), ("f", "g")]
    assert all(r.predicted_match(0.5) is True for r in kept.rows)
    # Entities are carried by reference (never deep-copied).
    assert all(kept.store[k] is pairs.store[k] for k in pairs.store)


def test_topk_select_keeps_the_k_best_rows_per_left_id():
    """Keep at most k highest-scoring rows per left_id; a smaller group keeps all of it."""
    pairs = _pairs(
        [
            _scored("x", "a", 0.9),  # x: top-2 by score are a(0.9) and c(0.7)
            _scored("x", "b", 0.5),  # dropped (3rd best for x)
            _scored("x", "c", 0.7),
            _scored("x", "d", 0.2),  # dropped (4th best for x)
            _scored("y", "e", 0.4),  # y has one row < k -> kept
        ]
    )
    kept = TopKSelect[CompanySchema](2).forward(pairs)

    # k best per left_id, and input order preserved among the survivors.
    assert [(r.left_id, r.right_id) for r in kept.rows] == [("x", "a"), ("x", "c"), ("y", "e")]


def test_topk_select_ties_keep_input_order():
    """On a score tie the earlier (input-order) row wins the last kept slot."""
    pairs = _pairs(
        [
            _scored("x", "a", 0.5),  # tie with b; a appears first -> kept
            _scored("x", "b", 0.5),  # tie with a; dropped (k=1, a came first)
        ]
    )
    kept = TopKSelect[CompanySchema](1).forward(pairs)
    assert [(r.left_id, r.right_id) for r in kept.rows] == [("x", "a")]


# --------------------------------------------------------------------------------------
# The "unknown" OutSpace sentinel — a scalar family, so a Select may follow it.
# --------------------------------------------------------------------------------------


def test_score_out_space_may_be_the_unknown_sentinel():
    score = _Score(scope="pair", out_space="unknown")
    assert score.out_space == "unknown"


def test_select_after_an_unknown_score_is_legal_but_after_a_vector_still_raises():
    """The sentinel is a scalar: selection is defined on it (unlike a vector)."""
    # Legal: a Select after a scalar "unknown" Score (a concrete TopKSelect composes too).
    ok = Sequential(
        [
            _Source(),
            _Score(scope="pair", out_space="unknown"),
            TopKSelect[CompanySchema](2),
            _ClusterStage(),
        ]
    )
    assert any(isinstance(stage, TopKSelect) for stage in ok.stages)

    # Still illegal: a Select after a "vector" Score (a ComparisonVector is not orderable).
    with pytest.raises(ValueError, match="not orderable"):
        Sequential(
            [
                _Source(),
                _Score(scope="pair", out_space="vector"),
                ThresholdSelect[CompanySchema](0.5),
                _ClusterStage(),
            ]
        )
