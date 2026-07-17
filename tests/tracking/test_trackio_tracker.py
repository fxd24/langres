"""Tests for the Trackio ``ExperimentTracker`` adapter.

Behavior/smoke tier: ``trackio`` is fully mocked (a fake module patched over
``langres.tracking.trackers.trackio_tracker.trackio``), so no test touches the
network or a real HF login. We assert that start_run -> log -> finish call the
right trackio APIs with the flattened context, that ``run_url``/``native``
derive from the underlying run, that the HF-sync credential guard fires
before ``trackio.init`` (and never fires for a local-only run), and that
``resolve_tracker("trackio")`` returns a real :class:`TrackioTracker`. One
``@pytest.mark.slow`` test drives the REAL, uninstalled-mock trackio against a
local (no ``space_id``) run to prove the adapter works end to end offline.
"""

from __future__ import annotations

import os
from typing import Any

import huggingface_hub
import pytest

from langres.clients.settings import Settings
from langres.tracking.runs import RunContext
from langres.tracking.trackers import ExperimentTracker, resolve_tracker
from langres.tracking.trackers.trackio_tracker import TrackioTracker, _flatten


class _FakeRun:
    """Stand-in for a ``trackio.Run`` -- records ``log``/``finish`` on itself.

    ``config`` is a plain ``dict`` (unlike wandb's special config object), since
    that is what the real ``trackio.Run.config`` is -- ``.update()`` is the
    built-in ``dict`` method.
    """

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.name: str | None = None
        self.log_calls: list[tuple[dict[str, Any], int | None]] = []
        self.finish_calls: int = 0

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        self.log_calls.append((dict(data), step))

    def finish(self) -> None:
        self.finish_calls += 1


class _FakeTrackio:
    """Minimal fake ``trackio`` module: ``init`` records kwargs and returns the run."""

    def __init__(self, run: _FakeRun | None = None) -> None:
        self.run = run if run is not None else _FakeRun()
        self.init_kwargs: dict[str, Any] | None = None

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_kwargs = kwargs
        return self.run


@pytest.fixture
def fake_trackio(monkeypatch: pytest.MonkeyPatch) -> _FakeTrackio:
    """Patch the fake over ``trackio_tracker.trackio`` (the module-level import)."""
    fake = _FakeTrackio()
    import langres.tracking.trackers.trackio_tracker as tracker_mod

    monkeypatch.setattr(tracker_mod, "trackio", fake)
    return fake


@pytest.fixture(autouse=True)
def _no_ambient_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from a real local HF login / env token on the dev machine.

    Every test that touches the credential guard controls
    ``huggingface_hub.get_token`` (or ``settings.hf_token``) explicitly; this
    fixture just guarantees the *ambient* environment can't leak in and flip a
    "no token" test green (or a "has token" test's isolation) depending on
    whoever's machine runs the suite.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


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
        assert isinstance(TrackioTracker(), ExperimentTracker)

    def test_name_is_trackio(self) -> None:
        assert TrackioTracker().name == "trackio"

    def test_no_run_before_start(self) -> None:
        tracker = TrackioTracker()
        assert tracker.native is None
        assert tracker.run_url is None


class TestStartRunLocal:
    def test_local_run_needs_no_token(self, fake_trackio: _FakeTrackio) -> None:
        """The default (no space_id) path never touches the HF-token guard."""
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context(), run_name="dedupe-ag")  # must not raise

        assert fake_trackio.init_kwargs is not None
        assert fake_trackio.init_kwargs["space_id"] is None
        assert fake_trackio.init_kwargs["dataset_id"] is None

    def test_project_defaults_to_experiment(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())

        assert fake_trackio.init_kwargs is not None
        assert fake_trackio.init_kwargs["project"] == "dedupe-ag"

    def test_explicit_project_overrides_experiment(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings(), project="my-project")
        tracker.start_run(_context())

        assert fake_trackio.init_kwargs is not None
        assert fake_trackio.init_kwargs["project"] == "my-project"

    def test_run_name_passed_through(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context(), run_name="dedupe-ag-run")

        assert fake_trackio.init_kwargs is not None
        assert fake_trackio.init_kwargs["name"] == "dedupe-ag-run"

    def test_flattens_context_into_config(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())

        config = fake_trackio.init_kwargs["config"]
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

    def test_extra_config_merged_in(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings(), config={"extra": 1})
        tracker.start_run(_context())

        assert fake_trackio.init_kwargs["config"]["extra"] == 1


class TestHfSyncCredentialGuard:
    def test_space_id_without_token_raises_actionable_value_error(
        self, fake_trackio: _FakeTrackio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
        tracker = TrackioTracker(Settings(), space_id="acme/langres-runs")

        with pytest.raises(ValueError, match=r"acme/langres-runs.*HF_TOKEN"):
            tracker.start_run(_context())

        # The guard fired before trackio.init was ever called.
        assert fake_trackio.init_kwargs is None

    def test_space_id_with_settings_hf_token_succeeds_and_exports_env(
        self, fake_trackio: _FakeTrackio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
        tracker = TrackioTracker(Settings(hf_token="hf_explicit"), space_id="acme/langres-runs")
        tracker.start_run(_context())  # must not raise

        assert fake_trackio.init_kwargs["space_id"] == "acme/langres-runs"
        # Propagated so trackio's own internal huggingface_hub calls see it too.
        assert os.environ["HF_TOKEN"] == "hf_explicit"

    def test_space_id_with_ambient_hf_token_succeeds_without_settings(
        self, fake_trackio: _FakeTrackio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-login-token")
        tracker = TrackioTracker(Settings(), space_id="acme/langres-runs")
        tracker.start_run(_context())  # must not raise

        assert fake_trackio.init_kwargs["space_id"] == "acme/langres-runs"

    def test_space_id_from_settings_fallback(
        self, fake_trackio: _FakeTrackio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A space_id configured only via Settings (not the constructor) also works."""
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-login-token")
        settings = Settings(trackio_space_id="acme/langres-runs", trackio_dataset_id="acme/ds")
        tracker = TrackioTracker(settings)
        tracker.start_run(_context())

        assert fake_trackio.init_kwargs["space_id"] == "acme/langres-runs"
        assert fake_trackio.init_kwargs["dataset_id"] == "acme/ds"

    def test_constructor_space_id_overrides_settings(
        self, fake_trackio: _FakeTrackio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-login-token")
        settings = Settings(trackio_space_id="settings/space")
        tracker = TrackioTracker(settings, space_id="explicit/space")
        tracker.start_run(_context())

        assert fake_trackio.init_kwargs["space_id"] == "explicit/space"

    def test_preexisting_hf_token_env_wins_over_settings_token(
        self, fake_trackio: _FakeTrackio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user-exported HF_TOKEN is never clobbered by settings.hf_token.

        Locks in the ``os.environ.setdefault`` precedence: the explicit ambient
        token wins, and trackio's own huggingface_hub calls keep seeing it.
        """
        monkeypatch.setenv("HF_TOKEN", "preexisting-token")
        tracker = TrackioTracker(Settings(hf_token="hf_explicit"), space_id="acme/langres-runs")
        tracker.start_run(_context())  # must not raise (token present)

        assert os.environ["HF_TOKEN"] == "preexisting-token"

    def test_dataset_id_without_space_id_raises(
        self, fake_trackio: _FakeTrackio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dataset_id with no space_id is a misconfiguration -- fail fast, no init."""
        monkeypatch.delenv("TRACKIO_SPACE_ID", raising=False)
        tracker = TrackioTracker(Settings(), dataset_id="acme/ds")

        with pytest.raises(ValueError, match=r"dataset_id.*requires a space_id"):
            tracker.start_run(_context())

        assert fake_trackio.init_kwargs is None


class TestLogging:
    def test_log_metrics_routes_to_run_log(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        tracker.log_metrics({"f1": 0.91, "recall": 0.88}, step=3)

        assert fake_trackio.run.log_calls == [({"f1": 0.91, "recall": 0.88}, 3)]

    def test_log_metrics_defaults_step_to_none(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        tracker.log_metrics({"f1": 0.5})

        assert fake_trackio.run.log_calls == [({"f1": 0.5}, None)]

    def test_log_metrics_noop_before_start(self, fake_trackio: _FakeTrackio) -> None:
        TrackioTracker().log_metrics({"f1": 0.5})
        assert fake_trackio.run.log_calls == []

    def test_log_params_merges_into_config_and_logs(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        tracker.log_params({"extra": {"nested": 1}})

        # Kept in config (in-memory / pre-first-log flush) AND emitted via run.log
        # under a param. prefix (the durable path -- see below).
        assert fake_trackio.run.config["extra.nested"] == 1
        assert fake_trackio.run.log_calls[-1] == ({"param.extra.nested": 1}, None)

    def test_log_params_persists_after_a_metric(self, fake_trackio: _FakeTrackio) -> None:
        """trackio flushes config only on the first run.log, so a param logged
        AFTER a metric must still reach the store -- via its own run.log entry."""
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        tracker.log_metrics({"f1": 0.9})  # first log latches config
        tracker.log_params({"late": 1})

        assert ({"param.late": 1}, None) in fake_trackio.run.log_calls

    def test_log_params_noop_before_start(self) -> None:
        TrackioTracker().log_params({"a": 1})  # must not raise

    def test_log_artifact_logs_as_metadata_entry(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        tracker.log_artifact("report_md", "runs/report.md")

        assert fake_trackio.run.log_calls[-1] == ({"report_md": "runs/report.md"}, None)

    def test_log_artifact_noop_before_start(self) -> None:
        TrackioTracker().log_artifact("k", "v")  # must not raise

    def test_set_tags_folds_into_config_with_prefix_and_logs(
        self, fake_trackio: _FakeTrackio
    ) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        tracker.set_tags({"env": "ci", "team": "er"})

        assert fake_trackio.run.config["tag.env"] == "ci"
        assert fake_trackio.run.config["tag.team"] == "er"
        assert fake_trackio.run.log_calls[-1] == ({"tag.env": "ci", "tag.team": "er"}, None)

    def test_set_tags_persists_after_a_metric(self, fake_trackio: _FakeTrackio) -> None:
        """Same one-way-config-latch reason as params: a tag set after a metric
        must still persist via run.log, not only the (already-flushed) config."""
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        tracker.log_metrics({"f1": 0.9})
        tracker.set_tags({"env": "ci"})

        assert ({"tag.env": "ci"}, None) in fake_trackio.run.log_calls

    def test_set_tags_noop_before_start(self) -> None:
        TrackioTracker().set_tags({"a": "b"})  # must not raise


class TestFinish:
    def test_finish_logs_status_then_closes_run(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        tracker.finish(status="completed")

        assert fake_trackio.run.log_calls[-1] == ({"langres_status": "completed"}, None)
        assert fake_trackio.run.finish_calls == 1

    @pytest.mark.parametrize("status", ["failed", "budget_exceeded"])
    def test_finish_logs_any_status(self, fake_trackio: _FakeTrackio, status: str) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        tracker.finish(status=status)

        assert fake_trackio.run.log_calls[-1] == ({"langres_status": status}, None)

    def test_finish_before_start_is_noop(self, fake_trackio: _FakeTrackio) -> None:
        TrackioTracker().finish(status="completed")
        assert fake_trackio.run.finish_calls == 0


class TestRunUrlAndNative:
    def test_run_url_none_for_local_run(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        assert tracker.run_url is None

    def test_run_url_deep_links_hf_space(
        self, fake_trackio: _FakeTrackio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-login-token")
        tracker = TrackioTracker(Settings(), space_id="acme/langres-runs")
        tracker.start_run(_context())

        assert tracker.run_url == "https://acme-langres-runs.hf.space/"

    def test_run_url_none_for_namespaceless_space_id(
        self, fake_trackio: _FakeTrackio, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare (no ``/``) space_id can't build a user-space URL -> None, no crash."""
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-login-token")
        tracker = TrackioTracker(Settings(), space_id="justareponame")
        tracker.start_run(_context())

        assert tracker.run_url is None

    def test_native_returns_underlying_run(self, fake_trackio: _FakeTrackio) -> None:
        tracker = TrackioTracker(Settings())
        tracker.start_run(_context())
        assert tracker.native is fake_trackio.run


class TestResolveTracker:
    def test_resolve_tracker_trackio_returns_real_trackio_tracker(
        self, fake_trackio: _FakeTrackio
    ) -> None:
        resolved = resolve_tracker("trackio")
        assert isinstance(resolved, TrackioTracker)
        assert resolved.name == "trackio"

    def test_resolve_tracker_trackio_is_local_first_by_default(
        self, fake_trackio: _FakeTrackio
    ) -> None:
        """No creds/env configured -> resolve_tracker("trackio") still starts a run."""
        tracker = resolve_tracker("trackio")
        tracker.start_run(_context())  # must not raise: no space_id configured
        assert fake_trackio.init_kwargs["space_id"] is None


@pytest.mark.slow
class TestRealLocalSmoke:
    """One real (unmocked) local trackio.init -- proves the adapter works offline.

    No ``space_id`` -> no network, no HF login required. Skips cleanly if a real
    local trackio init cannot complete offline (kept mocked above for everything
    else specifically so this env-dependent edge doesn't gate the fast suite).
    """

    def test_real_local_run_writes_to_local_store(self, tmp_path: Any) -> None:
        import trackio

        from langres.tracking.trackers.trackio_tracker import TrackioTracker as RealTracker

        project = f"langres-test-{os.getpid()}"
        tracker = RealTracker(Settings(), project=project)
        tracker.start_run(_context(), run_name="smoke-run")
        try:
            tracker.log_metrics({"f1": 0.5}, step=0)
            tracker.log_metrics({"f1": 0.6}, step=1)
        finally:
            tracker.finish(status="completed")

        assert tracker.native is not None
        assert tracker.run_url is None  # local run: no HF Space configured
        runs = trackio.SQLiteStorage.get_runs(project)
        assert "smoke-run" in runs
