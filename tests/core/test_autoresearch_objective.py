"""Tests for the autoresearch :class:`Objective` (the keep-if-better scorer).

Core contract code -> high coverage tier. Covers feasibility gating, goal
normalization, Pareto dominance (dominating / dominated / incomparable / tie),
the ``is_better`` decision against ``None`` and feasible/infeasible incumbents,
the ergonomic constructors, and every fail-loud validation branch.
"""

import pytest

from langres.core.autoresearch.objective import (
    Constraint,
    Goal,
    Objective,
)

# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------


def test_goal_is_frozen() -> None:
    g = Goal("recall", "maximize")
    with pytest.raises((AttributeError, TypeError)):
        g.metric = "precision"  # type: ignore[misc]


def test_goal_bad_direction_raises() -> None:
    with pytest.raises(ValueError, match="direction must be one of"):
        Goal("recall", "maximise")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "threshold", "value", "expected"),
    [
        (">=", 0.9, 0.9, True),
        (">=", 0.9, 0.89, False),
        ("<=", 0.1, 0.1, True),
        ("<=", 0.1, 0.11, False),
        (">", 0.5, 0.6, True),
        (">", 0.5, 0.5, False),
        ("<", 0.5, 0.4, True),
        ("<", 0.5, 0.5, False),
    ],
)
def test_constraint_satisfied_each_op(
    op: str, threshold: float, value: float, expected: bool
) -> None:
    c = Constraint("m", op, threshold)  # type: ignore[arg-type]
    assert c.satisfied({"m": value}) is expected


def test_constraint_bad_op_raises() -> None:
    with pytest.raises(ValueError, match="constraint op must be one of"):
        Constraint("m", ">>", 0.5)  # type: ignore[arg-type]


def test_constraint_missing_metric_raises() -> None:
    c = Constraint("precision", ">=", 0.9)
    with pytest.raises(ValueError, match="metric 'precision' is missing"):
        c.satisfied({"recall": 0.9})


# ---------------------------------------------------------------------------
# Objective construction / value normalization
# ---------------------------------------------------------------------------


def test_objective_requires_a_goal() -> None:
    with pytest.raises(ValueError, match="at least one goal"):
        Objective(goals=())


def test_value_maximize_passthrough() -> None:
    obj = Objective.maximize("recall")
    assert obj.value({"recall": 0.8}) == (0.8,)


def test_value_minimize_is_negated() -> None:
    obj = Objective.minimize("log_loss")
    # Normalized to higher-is-better: a minimize goal is negated.
    assert obj.value({"log_loss": 0.3}) == (-0.3,)


def test_value_orders_by_goals() -> None:
    obj = Objective.pareto([("recall", "maximize"), ("cost", "minimize")])
    assert obj.value({"recall": 0.9, "cost": 2.0}) == (0.9, -2.0)


def test_value_missing_metric_raises() -> None:
    obj = Objective.maximize("recall")
    with pytest.raises(ValueError, match="metric 'recall' is missing"):
        obj.value({"precision": 0.9})


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


def test_is_feasible_no_constraints_is_vacuously_true() -> None:
    obj = Objective.maximize("recall")
    assert obj.is_feasible({"recall": 0.1}) is True


def test_is_feasible_all_satisfied() -> None:
    obj = Objective.maximize("recall", subject_to=[("precision", ">=", 0.9)])
    assert obj.is_feasible({"recall": 0.8, "precision": 0.95}) is True


def test_is_feasible_one_violated() -> None:
    obj = Objective.maximize("recall", subject_to=[("precision", ">=", 0.9)])
    assert obj.is_feasible({"recall": 0.8, "precision": 0.85}) is False


def test_is_feasible_missing_constraint_metric_raises() -> None:
    obj = Objective.maximize("recall", subject_to=[("precision", ">=", 0.9)])
    with pytest.raises(ValueError, match="metric 'precision' is missing"):
        obj.is_feasible({"recall": 0.8})


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------


def test_dominates_strictly_better_on_all() -> None:
    obj = Objective.pareto([("recall", "maximize"), ("precision", "maximize")])
    a = {"recall": 0.9, "precision": 0.9}
    b = {"recall": 0.8, "precision": 0.8}
    assert obj.dominates(a, b) is True
    assert obj.dominates(b, a) is False


def test_dominates_equal_on_one_better_on_other() -> None:
    obj = Objective.pareto([("recall", "maximize"), ("precision", "maximize")])
    a = {"recall": 0.9, "precision": 0.8}
    b = {"recall": 0.8, "precision": 0.8}  # a is >= on both and > on recall
    assert obj.dominates(a, b) is True


def test_dominates_incomparable_tradeoff() -> None:
    obj = Objective.pareto([("recall", "maximize"), ("precision", "maximize")])
    a = {"recall": 0.9, "precision": 0.7}
    b = {"recall": 0.8, "precision": 0.8}  # a better on recall, worse on precision
    assert obj.dominates(a, b) is False
    assert obj.dominates(b, a) is False


def test_dominates_tie_is_not_dominance() -> None:
    obj = Objective.pareto([("recall", "maximize"), ("precision", "maximize")])
    m = {"recall": 0.9, "precision": 0.9}
    assert obj.dominates(m, dict(m)) is False


def test_dominates_respects_minimize_direction() -> None:
    obj = Objective.pareto([("recall", "maximize"), ("cost", "minimize")])
    a = {"recall": 0.9, "cost": 1.0}
    b = {"recall": 0.9, "cost": 2.0}  # same recall, lower cost -> a dominates
    assert obj.dominates(a, b) is True
    assert obj.dominates(b, a) is False


# ---------------------------------------------------------------------------
# is_better — the loop's keep/revert decision
# ---------------------------------------------------------------------------


def test_is_better_feasible_beats_none_incumbent() -> None:
    obj = Objective.maximize("recall")
    assert obj.is_better({"recall": 0.5}, None) is True


def test_is_better_infeasible_candidate_never_better_even_vs_none() -> None:
    obj = Objective.maximize("recall", subject_to=[("precision", ">=", 0.9)])
    assert obj.is_better({"recall": 0.99, "precision": 0.5}, None) is False


def test_is_better_feasible_beats_infeasible_incumbent() -> None:
    obj = Objective.maximize("recall", subject_to=[("precision", ">=", 0.9)])
    candidate = {"recall": 0.5, "precision": 0.95}  # feasible
    incumbent = {"recall": 0.99, "precision": 0.5}  # infeasible
    assert obj.is_better(candidate, incumbent) is True


def test_is_better_single_goal_strict_improvement() -> None:
    obj = Objective.maximize("recall")
    assert obj.is_better({"recall": 0.9}, {"recall": 0.8}) is True
    assert obj.is_better({"recall": 0.8}, {"recall": 0.9}) is False


def test_is_better_single_goal_tie_keeps_incumbent() -> None:
    obj = Objective.maximize("recall")
    assert obj.is_better({"recall": 0.8}, {"recall": 0.8}) is False


def test_is_better_minimize_single_goal() -> None:
    obj = Objective.minimize("log_loss")
    assert obj.is_better({"log_loss": 0.2}, {"log_loss": 0.3}) is True
    assert obj.is_better({"log_loss": 0.4}, {"log_loss": 0.3}) is False


def test_is_better_multi_goal_dominating() -> None:
    obj = Objective.pareto([("recall", "maximize"), ("precision", "maximize")])
    assert obj.is_better(
        {"recall": 0.9, "precision": 0.9}, {"recall": 0.8, "precision": 0.8}
    ) is True


def test_is_better_multi_goal_incomparable_keeps_incumbent() -> None:
    obj = Objective.pareto([("recall", "maximize"), ("precision", "maximize")])
    # Better on recall, worse on precision: incomparable -> not better.
    assert obj.is_better(
        {"recall": 0.9, "precision": 0.7}, {"recall": 0.8, "precision": 0.8}
    ) is False


def test_is_better_missing_metric_raises() -> None:
    obj = Objective.maximize("recall")
    with pytest.raises(ValueError, match="metric 'recall' is missing"):
        obj.is_better({"precision": 0.9}, {"recall": 0.8})


# ---------------------------------------------------------------------------
# Ergonomic constructors
# ---------------------------------------------------------------------------


def test_maximize_builds_single_goal_and_constraints() -> None:
    obj = Objective.maximize("recall", subject_to=[("precision", ">=", 0.9)])
    assert obj.goals == (Goal("recall", "maximize"),)
    assert obj.constraints == (Constraint("precision", ">=", 0.9),)


def test_minimize_builds_single_minimize_goal() -> None:
    obj = Objective.minimize("log_loss")
    assert obj.goals == (Goal("log_loss", "minimize"),)
    assert obj.constraints == ()


def test_pareto_builds_multiple_goals_with_mixed_directions() -> None:
    obj = Objective.pareto(
        [("recall", "maximize"), ("cost", "minimize")],
        subject_to=[("precision", ">=", 0.8)],
    )
    assert obj.goals == (Goal("recall", "maximize"), Goal("cost", "minimize"))
    assert obj.constraints == (Constraint("precision", ">=", 0.8),)


def test_pareto_bad_direction_raises() -> None:
    with pytest.raises(ValueError, match="direction must be one of"):
        Objective.pareto([("recall", "up")])


def test_constructor_bad_constraint_op_raises() -> None:
    with pytest.raises(ValueError, match="constraint op must be one of"):
        Objective.maximize("recall", subject_to=[("precision", "=>", 0.9)])
