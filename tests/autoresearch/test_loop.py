"""Tests for the autoresearch keep-if-better :mod:`~langres.autoresearch.loop`.

Core contract code -> high coverage tier. Every test injects a **fake dict
scorer** (config -> canned metrics), so the loop's incumbent tracking, dedup,
persistence, and failure handling are exercised with zero embeddings / faiss /
benchmark load -- the point of the injected-scorer seam.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from langres.autoresearch.loop import LoopResult, Trial, run_loop
from langres.autoresearch.objective import Objective
from langres.tracking.runs import RunStore


def _scorer(table: dict[str, dict[str, float]]) -> Any:
    """A fake scorer resolving a config's ``id`` to canned metrics."""

    def scorer(config: Mapping[str, Any]) -> dict[str, float]:
        return dict(table[config["id"]])

    return scorer


def _cfg(cid: str, blocker: str = "vector", **extra: Any) -> dict[str, Any]:
    """A minimal config dict tagged with an ``id`` the fake scorer keys on."""
    return {"blocker": blocker, "id": cid, **extra}


# ---------------------------------------------------------------------------
# Incumbent tracking
# ---------------------------------------------------------------------------


def test_strictly_better_config_displaces_the_incumbent() -> None:
    configs = [_cfg("a"), _cfg("b"), _cfg("c")]
    scorer = _scorer({"a": {"r": 0.5}, "b": {"r": 0.9}, "c": {"r": 0.7}})
    result = run_loop(configs, scorer, Objective.maximize("r"), experiment="e", dataset_name="d")

    assert isinstance(result, LoopResult)
    assert result.best_config == _cfg("b")
    assert result.best_metrics == {"r": 0.9}
    assert [t.accepted for t in result.trials] == [True, True, False]
    assert [t.status for t in result.trials] == ["completed", "completed", "completed"]


def test_first_feasible_config_is_accepted_against_a_none_incumbent() -> None:
    result = run_loop(
        [_cfg("only")],
        _scorer({"only": {"r": 0.1}}),
        Objective.maximize("r"),
        experiment="e",
        dataset_name="d",
    )
    assert result.best_config == _cfg("only")
    assert result.trials[0].accepted is True


def test_empty_config_stream_yields_no_incumbent() -> None:
    result = run_loop([], _scorer({}), Objective.maximize("r"), experiment="e", dataset_name="d")
    assert result.best_config is None
    assert result.best_metrics is None
    assert result.trials == ()


def test_infeasible_candidate_is_never_accepted() -> None:
    # A constraint the only config violates -> feasible-gate keeps it out.
    obj = Objective.maximize("r", subject_to=[("r", ">=", 0.8)])
    result = run_loop(
        [_cfg("a")], _scorer({"a": {"r": 0.5}}), obj, experiment="e", dataset_name="d"
    )
    assert result.best_config is None
    assert result.trials[0].accepted is False
    assert result.trials[0].status == "completed"  # scored fine, just not better


def test_incomparable_multiobjective_trade_off_is_kept_out() -> None:
    # Candidate b is better on recall but worse on cost -> neither dominates ->
    # the incumbent (a) is kept.
    obj = Objective.pareto([("recall", "maximize"), ("cost", "minimize")])
    configs = [_cfg("a"), _cfg("b")]
    scorer = _scorer({"a": {"recall": 0.5, "cost": 0.2}, "b": {"recall": 0.9, "cost": 0.9}})
    result = run_loop(configs, scorer, obj, experiment="e", dataset_name="d")

    assert result.best_config == _cfg("a")
    assert [t.accepted for t in result.trials] == [True, False]


# ---------------------------------------------------------------------------
# Persistence (every trial logged; accepted flag; lineage)
# ---------------------------------------------------------------------------


def test_every_trial_is_logged_with_accepted_flag_and_lineage(tmp_path: Any) -> None:
    store_path = tmp_path / "runs.jsonl"
    configs = [_cfg("a"), _cfg("b"), _cfg("c")]
    scorer = _scorer({"a": {"r": 0.5}, "b": {"r": 0.9}, "c": {"r": 0.7}})
    obj = Objective.maximize("r")
    result = run_loop(configs, scorer, obj, experiment="e", dataset_name="d", store=store_path)

    records = RunStore(store_path).read()
    # One terminal record per attempt (running + completed collapse by attempt_id).
    assert len(records) == 3
    by_attempt = {r.attempt_id: r for r in records}
    assert set(by_attempt) == {t.attempt_id for t in result.trials}

    # Records come back in first-seen (loop) order.
    assert [r.metrics["accepted"] for r in records] == [1.0, 1.0, 0.0]
    assert [r.headline_metric for r in records] == [0.5, 0.9, 0.7]
    assert all(r.metric_definition == str(obj) for r in records)
    assert all(r.status == "completed" for r in records)

    # Lineage: each accepted incumbent parents the next trial; the rejected c
    # still points at the incumbent (b) at comparison time.
    a, b, c = records
    assert a.context.parent_run_id is None
    assert b.context.parent_run_id == a.attempt_id
    assert c.context.parent_run_id == b.attempt_id


def test_store_none_writes_nothing(tmp_path: Any) -> None:
    result = run_loop(
        [_cfg("a"), _cfg("b")],
        _scorer({"a": {"r": 0.3}, "b": {"r": 0.6}}),
        Objective.maximize("r"),
        experiment="e",
        dataset_name="d",
        store=None,
    )
    # The loop still ran and tracked an incumbent...
    assert result.best_metrics == {"r": 0.6}
    assert all(t.attempt_id for t in result.trials)  # attempt ids exist even unpersisted
    # ...but nothing was written anywhere.
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_dedup_skips_a_duplicate_recipe(tmp_path: Any) -> None:
    store_path = tmp_path / "runs.jsonl"
    # Two byte-identical configs -> one recipe_id -> the second is skipped.
    dup = _cfg("a", blocker="all_pairs")
    result = run_loop(
        [dict(dup), dict(dup)],
        _scorer({"a": {"r": 0.5}}),
        Objective.maximize("r"),
        experiment="e",
        dataset_name="d",
        store=store_path,
    )
    assert len(result.trials) == 1
    assert len(RunStore(store_path).read()) == 1


def test_dedup_can_be_disabled(tmp_path: Any) -> None:
    store_path = tmp_path / "runs.jsonl"
    dup = _cfg("a", blocker="all_pairs")
    result = run_loop(
        [dict(dup), dict(dup)],
        _scorer({"a": {"r": 0.5}}),
        Objective.maximize("r"),
        experiment="e",
        dataset_name="d",
        store=store_path,
        dedup=False,
    )
    # Both scored; two attempts persisted (distinct attempt_ids, same recipe_id).
    assert len(result.trials) == 2
    records = RunStore(store_path).read()
    assert len(records) == 2
    assert len({r.recipe_id for r in records}) == 1


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_scorer_failure_is_logged_failed_and_loop_continues(tmp_path: Any) -> None:
    store_path = tmp_path / "runs.jsonl"

    def scorer(config: Mapping[str, Any]) -> dict[str, float]:
        if config["id"] == "boom":
            raise RuntimeError("scorer blew up")
        return {"a": {"r": 0.5}, "c": {"r": 0.9}}[config["id"]]

    result = run_loop(
        [_cfg("a"), _cfg("boom"), _cfg("c")],
        scorer,
        Objective.maximize("r"),
        experiment="e",
        dataset_name="d",
        store=store_path,
    )

    # The failure did not abort the sweep: three trials, the middle one failed.
    assert [t.status for t in result.trials] == ["completed", "failed", "completed"]
    failed = result.trials[1]
    assert failed.metrics is None
    assert failed.accepted is False
    assert result.best_config == _cfg("c")  # c still won

    records = {r.attempt_id: r for r in RunStore(store_path).read()}
    assert records[failed.attempt_id].status == "failed"
    assert records[failed.attempt_id].error_type == "RuntimeError"

    # Lineage skips the failed trial: c's parent is a's attempt (the incumbent
    # never advanced through the failure).
    good_a, _boom, good_c = result.trials
    assert records[good_c.attempt_id].context.parent_run_id == good_a.attempt_id


def test_trial_is_a_frozen_dataclass() -> None:
    trial = Trial({"blocker": "vector"}, {"r": 1.0}, True, "rid", "aid", "completed")
    with pytest.raises((AttributeError, TypeError)):
        trial.accepted = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tracker spec (DX: ``tracker="trackio"`` resolved internally via
# ``core.trackers.resolve_tracker`` -- mirrors ``judge="..."`` presets).
# ---------------------------------------------------------------------------


class _SpyTracker:
    """Minimal in-memory ``ExperimentTracker`` double -- records every call it gets."""

    name = "spy"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def start_run(self, context: Any, *, run_name: str | None = None) -> None:
        self.calls.append("start_run")

    def log_params(self, params: Any) -> None:
        self.calls.append("log_params")

    def log_metrics(self, metrics: Any, *, step: int | None = None) -> None:
        self.calls.append("log_metrics")

    def log_artifact(self, key: str, value: str) -> None:
        self.calls.append("log_artifact")

    def set_tags(self, tags: Any) -> None:
        self.calls.append("set_tags")

    def finish(self, *, status: str) -> None:
        self.calls.append("finish")

    @property
    def run_url(self) -> str | None:
        return None

    @property
    def native(self) -> Any:
        return self


def test_tracker_none_default_never_raises() -> None:
    result = run_loop(
        [_cfg("a")],
        _scorer({"a": {"r": 1.0}}),
        Objective.maximize("r"),
        experiment="e",
        dataset_name="d",
        tracker=None,
    )
    assert result.best_config == _cfg("a")


def test_tracker_instance_is_driven_through_the_loop() -> None:
    spy = _SpyTracker()
    run_loop(
        [_cfg("a")],
        _scorer({"a": {"r": 1.0}}),
        Objective.maximize("r"),
        experiment="e",
        dataset_name="d",
        tracker=spy,
    )
    assert spy.calls == ["start_run", "log_metrics", "finish"]


def test_tracker_string_resolves_to_the_named_backend_and_is_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tracker="trackio"`` resolves to a real ``TrackioTracker`` end to end.

    ``trackio`` is mocked exactly like ``tests/test_trackio_tracker.py``'s
    ``fake_trackio`` fixture -- no network, no real HF login -- so this proves
    the spec resolved AND the resulting tracker actually got driven by
    ``capture_run``, not just that ``resolve_tracker`` returns the right type.
    """
    import langres.tracking.trackers.trackio_tracker as trackio_mod

    class _FakeRun:
        def __init__(self) -> None:
            self.log_calls: list[Any] = []

        def log(self, data: Any, step: int | None = None) -> None:
            self.log_calls.append((data, step))

        def finish(self) -> None:
            pass

    class _FakeTrackio:
        def __init__(self) -> None:
            self.init_kwargs: dict[str, Any] | None = None
            self.run = _FakeRun()

        def init(self, **kwargs: Any) -> Any:
            self.init_kwargs = kwargs
            return self.run

    fake = _FakeTrackio()
    monkeypatch.setattr(trackio_mod, "trackio", fake)

    run_loop(
        [_cfg("a")],
        _scorer({"a": {"r": 1.0}}),
        Objective.maximize("r"),
        experiment="e",
        dataset_name="d",
        tracker="trackio",
    )
    assert fake.init_kwargs is not None  # TrackioTracker.start_run really fired


def test_tracker_sequence_fans_out_to_every_child() -> None:
    a, b = _SpyTracker(), _SpyTracker()
    run_loop(
        [_cfg("x")],
        _scorer({"x": {"r": 1.0}}),
        Objective.maximize("r"),
        experiment="e",
        dataset_name="d",
        tracker=[a, b],
    )
    assert a.calls == ["start_run", "log_metrics", "finish"]
    assert b.calls == ["start_run", "log_metrics", "finish"]


def test_tracker_unknown_backend_string_raises_before_any_config_is_scored() -> None:
    scored: list[str] = []

    def scorer(config: Mapping[str, Any]) -> dict[str, float]:
        scored.append(config["id"])
        return {"r": 1.0}

    with pytest.raises(ValueError, match="unknown"):
        run_loop(
            [_cfg("a")],
            scorer,
            Objective.maximize("r"),
            experiment="e",
            dataset_name="d",
            tracker="not-a-backend",
        )
    assert scored == []  # the spec is resolved before the loop scores anything
