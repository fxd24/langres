"""Tests for the W&B ``ExperimentTracker`` adapter (Stream S4).

Behavior/smoke tier: ``wandb`` is fully mocked (a fake module patched over both
``langres.clients.tracking.wandb`` -- used by the reused ``create_wandb_tracker``
init -- and ``langres.core.trackers.wandb_tracker.wandb``), so no test touches the
network or a real W&B login. We assert that start_run -> log -> finish call the
right W&B APIs with the flattened context, that ``run_url``/``native`` derive from
the underlying run, that status maps to an exit code, and that
``resolve_tracker("wandb")`` returns a real :class:`WandbTracker`.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from langres.clients.settings import Settings
from langres.core.runs import RunContext
from langres.core.trackers import ExperimentTracker, resolve_tracker
from langres.core.trackers.wandb_tracker import WandbTracker, _flatten


class _FakeConfig:
    """Stand-in for ``wandb.Run.config`` -- records the merged config dict."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def update(self, values: dict[str, Any], allow_val_change: bool | None = None) -> None:
        self.data.update(values)


class _FakeRun:
    """Stand-in for a ``wandb`` run object -- records ``log``/``finish`` on itself."""

    def __init__(self, *, url: str | None = "https://wandb.ai/acme/langres/runs/abc123") -> None:
        self.config = _FakeConfig()
        self.summary: dict[str, Any] = {}
        self.name: str | None = None
        self.tags: tuple[str, ...] = ()
        self.url = url
        self.log_calls: list[tuple[dict[str, Any], int | None]] = []
        self.finish_calls: list[int | None] = []

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        self.log_calls.append((dict(data), step))

    def finish(self, exit_code: int | None = None) -> None:
        self.finish_calls.append(exit_code)


class _FakeWandb:
    """Minimal fake ``wandb`` module: ``init`` returns the run; ``log``/``finish``.

    The module-global ``log``/``finish`` here are the *tripwire* for M5: the
    adapter must route through ``run.log``/``run.finish`` (recorded on
    :class:`_FakeRun`), so these ``log_calls``/``finish_calls`` must stay empty.
    """

    def __init__(self, run: _FakeRun | None = None) -> None:
        self.run = run if run is not None else _FakeRun()
        self.init_kwargs: dict[str, Any] | None = None
        self.log_calls: list[tuple[dict[str, Any], int | None]] = []
        self.finish_calls: list[int | None] = []

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_kwargs = kwargs
        return self.run

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        self.log_calls.append((data, step))

    def finish(self, exit_code: int | None = None) -> None:
        self.finish_calls.append(exit_code)


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> _FakeWandb:
    """Patch the fake over the ``wandb`` the reused ``create_wandb_tracker`` init uses.

    ``create_wandb_tracker`` now imports ``wandb`` lazily inside its own body (so
    ``langres.clients.tracking`` -- and therefore ``create_trackio_tracker`` -- is
    importable without the ``wandb`` extra), so the ``wandb.init`` seam is patched
    at ``sys.modules["wandb"]``: an in-body ``import wandb`` binds to whatever is
    there. The adapter itself never references a module-global ``wandb`` (log/finish
    route through ``self._run``).
    """
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture
def settings() -> Settings:
    """Explicit settings so ``create_wandb_tracker``'s api-key check passes offline."""
    return Settings(wandb_api_key="test-key", wandb_project="test-proj", wandb_entity="acme")


def _context() -> RunContext:
    return RunContext(
        experiment="dedupe-ag",
        tags={"env": "ci"},
        git_sha="deadbeef",
        llm_model="deepseek/deepseek-chat",
        cascade_band=(0.3, 0.7),
        blocking_k=5,
        dataset_name="amazon-google",
        seeds={"split": 42},
        resolver_config={"type_name": "Resolver", "seed": 7},
    )


class TestFlatten:
    def test_dots_nested_and_drops_none(self) -> None:
        flat = _flatten({"a": {"b": 1}, "c": None, "d": 2})
        assert flat == {"a.b": 1, "d": 2}

    def test_indexes_sequences(self) -> None:
        assert _flatten({"band": [0.3, 0.7]}) == {"band[0]": 0.3, "band[1]": 0.7}

    def test_empty_mapping_yields_nothing(self) -> None:
        assert _flatten({"tags": {}}) == {}


class TestProtocolAndConstruction:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(WandbTracker(), ExperimentTracker)

    def test_name_is_wandb(self) -> None:
        assert WandbTracker().name == "wandb"

    def test_no_run_before_start(self) -> None:
        tracker = WandbTracker()
        assert tracker.native is None
        assert tracker.run_url is None


class TestStartRun:
    def test_inits_with_settings_project_entity_and_job_type(
        self, fake_wandb: _FakeWandb, settings: Settings
    ) -> None:
        tracker = WandbTracker(settings, job_type="evaluation")
        tracker.start_run(_context(), run_name="dedupe-ag")

        assert fake_wandb.init_kwargs == {
            "project": "test-proj",
            "entity": "acme",
            "job_type": "evaluation",
        }
        assert fake_wandb.run.name == "dedupe-ag"

    def test_flattens_context_into_config(self, fake_wandb: _FakeWandb, settings: Settings) -> None:
        tracker = WandbTracker(settings)
        tracker.start_run(_context())

        config = fake_wandb.run.config.data
        assert config["experiment"] == "dedupe-ag"
        assert config["llm_model"] == "deepseek/deepseek-chat"
        assert config["blocking_k"] == 5
        assert config["seeds.split"] == 42
        assert config["tags.env"] == "ci"
        assert config["git_sha"] == "deadbeef"
        assert config["cascade_band[0]"] == 0.3
        assert config["cascade_band[1]"] == 0.7
        assert config["resolver_config.seed"] == 7
        # None-valued context fields are dropped, not surfaced as "None".
        assert "group" not in config


class TestLogging:
    def test_log_metrics_routes_to_run_log_not_module_global(
        self, fake_wandb: _FakeWandb, settings: Settings
    ) -> None:
        tracker = WandbTracker(settings)
        tracker.start_run(_context())
        tracker.log_metrics({"f1": 0.91, "recall": 0.88}, step=3)

        # Routed to THIS run (M5), not the module-global wandb.log.
        assert fake_wandb.run.log_calls == [({"f1": 0.91, "recall": 0.88}, 3)]
        assert fake_wandb.log_calls == []

    def test_log_metrics_defaults_step_to_none(
        self, fake_wandb: _FakeWandb, settings: Settings
    ) -> None:
        tracker = WandbTracker(settings)
        tracker.start_run(_context())
        tracker.log_metrics({"f1": 0.5})

        assert fake_wandb.run.log_calls == [({"f1": 0.5}, None)]

    def test_log_metrics_noop_before_start(self, fake_wandb: _FakeWandb) -> None:
        # No run yet -> nothing logged, and no crash (must not hit the module global).
        WandbTracker().log_metrics({"f1": 0.5})
        assert fake_wandb.log_calls == []

    def test_log_params_merges_into_config(
        self, fake_wandb: _FakeWandb, settings: Settings
    ) -> None:
        tracker = WandbTracker(settings)
        tracker.start_run(_context())
        tracker.log_params({"extra": {"nested": 1}})

        assert fake_wandb.run.config.data["extra.nested"] == 1

    def test_log_params_noop_before_start(self, fake_wandb: _FakeWandb) -> None:
        # No run yet -> nothing to update, and no crash.
        WandbTracker().log_params({"a": 1})

    def test_log_artifact_records_in_summary(
        self, fake_wandb: _FakeWandb, settings: Settings
    ) -> None:
        tracker = WandbTracker(settings)
        tracker.start_run(_context())
        tracker.log_artifact("report_md", "runs/report.md")

        assert fake_wandb.run.summary["report_md"] == "runs/report.md"

    def test_log_artifact_noop_before_start(self, fake_wandb: _FakeWandb) -> None:
        WandbTracker().log_artifact("k", "v")

    def test_set_tags_sets_key_value_labels(
        self, fake_wandb: _FakeWandb, settings: Settings
    ) -> None:
        tracker = WandbTracker(settings)
        tracker.start_run(_context())
        tracker.set_tags({"env": "ci", "team": "er"})

        assert fake_wandb.run.tags == ("env:ci", "team:er")

    def test_set_tags_noop_before_start(self, fake_wandb: _FakeWandb) -> None:
        WandbTracker().set_tags({"a": "b"})


class TestFinish:
    def test_completed_maps_to_exit_code_zero(
        self, fake_wandb: _FakeWandb, settings: Settings
    ) -> None:
        tracker = WandbTracker(settings)
        tracker.start_run(_context())
        tracker.finish(status="completed")

        # Ends THIS run (M5), not the module-global wandb.finish.
        assert fake_wandb.run.finish_calls == [0]
        assert fake_wandb.finish_calls == []
        assert fake_wandb.run.summary["status"] == "completed"

    @pytest.mark.parametrize("status", ["failed", "budget_exceeded"])
    def test_non_completed_maps_to_exit_code_one(
        self, fake_wandb: _FakeWandb, settings: Settings, status: str
    ) -> None:
        tracker = WandbTracker(settings)
        tracker.start_run(_context())
        tracker.finish(status=status)

        assert fake_wandb.run.finish_calls == [1]
        assert fake_wandb.finish_calls == []

    def test_finish_before_start_is_noop(self, fake_wandb: _FakeWandb) -> None:
        # No run was started -> nothing to finish; must not touch the module global.
        WandbTracker().finish(status="completed")
        assert fake_wandb.run.finish_calls == []
        assert fake_wandb.finish_calls == []


class TestRunUrlAndNative:
    def test_run_url_from_underlying_run(self, fake_wandb: _FakeWandb, settings: Settings) -> None:
        tracker = WandbTracker(settings)
        tracker.start_run(_context())
        assert tracker.run_url == "https://wandb.ai/acme/langres/runs/abc123"

    def test_run_url_none_when_run_has_no_url(
        self, monkeypatch: pytest.MonkeyPatch, settings: Settings
    ) -> None:
        fake = _FakeWandb(_FakeRun(url=None))
        monkeypatch.setitem(sys.modules, "wandb", fake)

        tracker = WandbTracker(settings)
        tracker.start_run(_context())
        assert tracker.run_url is None

    def test_native_returns_underlying_run(
        self, fake_wandb: _FakeWandb, settings: Settings
    ) -> None:
        tracker = WandbTracker(settings)
        tracker.start_run(_context())
        assert tracker.native is fake_wandb.run


class TestResolveTracker:
    def test_resolve_tracker_wandb_returns_real_wandb_tracker(self, fake_wandb: _FakeWandb) -> None:
        resolved = resolve_tracker("wandb")
        assert isinstance(resolved, WandbTracker)
        assert resolved.name == "wandb"
