"""Behavior/smoke tests for the MLflow ``ExperimentTracker`` adapter (Stream S3).

``mlflow`` is an optional extra and is **not installed** in the core test
environment, so the whole file mocks it: a fake ``mlflow`` module is injected
into ``sys.modules`` and the adapter module is (re)imported so its top-level
``import mlflow`` binds to the fake. The tests then assert the adapter drives
the right MLflow APIs -- start/params/tags/metrics/artifacts/finish -- with the
context flattened into dotted params, ``run_url`` derived from an HTTP tracking
server, ``finish(status=...)`` mapped to MLflow's ``RunStatus``, and that
``resolve_tracker("mlflow")`` yields a real ``MlflowTracker`` once the (mocked)
module is importable.

Per the tiered coverage policy these are behavior/smoke tests: adapter bodies
are exercised through the fake, with ``# pragma: no cover`` reserved for
genuinely un-mockable external calls (there are none here -- the fake covers
them).
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from langres.clients.settings import Settings
from langres.core.runs import RunContext

_TRACKER_MODULE = "langres.core.trackers.mlflow_tracker"


class _FakeRun:
    """Stand-in for an MLflow ``ActiveRun`` -- only ``.info`` ids are read."""

    def __init__(self, run_id: str = "run-abc123", experiment_id: str = "exp-1") -> None:
        self.info = SimpleNamespace(run_id=run_id, experiment_id=experiment_id)


class _FakeMlflow:
    """Records every MLflow call the adapter makes, for assertions."""

    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment: str | None = None
        self.run_name: str | None = None
        self.params: dict[str, str] = {}
        self.metrics: list[tuple[dict[str, float], int | None]] = []
        self.tags: dict[str, str] = {}
        self.artifact_files: list[str] = []
        self.artifact_dirs: list[str] = []
        self.ended_status: str | None = None
        self._run = _FakeRun()

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, run_name: str | None = None) -> _FakeRun:
        self.run_name = run_name
        return self._run

    def log_params(self, params: dict[str, str]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step))

    def log_artifact(self, local_path: str) -> None:
        self.artifact_files.append(local_path)

    def log_artifacts(self, local_dir: str) -> None:
        self.artifact_dirs.append(local_dir)

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def end_run(self, status: str | None = None) -> None:
        self.ended_status = status


@pytest.fixture(autouse=True)
def _clean_mlflow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate Settings from any ambient MLflow env so tests are deterministic."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT", raising=False)
    # The file-store maintenance-mode flag (F1) -- cleared so each test observes
    # the adapter's own setdefault, not a leaked value. monkeypatch restores it.
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)


@pytest.fixture
def mlflow_env(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, _FakeMlflow]:
    """Inject a fake ``mlflow`` and (re)import the adapter module against it.

    Returns ``(adapter_module, fake_mlflow)``. monkeypatch restores both
    ``sys.modules["mlflow"]`` and the adapter module entry on teardown, so the
    fake never leaks into other tests.
    """
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)  # type: ignore[misc]
    # Drop any real/prior import so the top-level `import mlflow` rebinds to fake.
    monkeypatch.delitem(sys.modules, _TRACKER_MODULE, raising=False)
    module = importlib.import_module(_TRACKER_MODULE)
    return module, fake


def _settings(**overrides: Any) -> Settings:
    """A Settings that ignores ``.env`` so overrides fully control the config."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def _make_context() -> RunContext:
    """A representative context with nested config, seeds, tuple, and tags."""
    return RunContext(
        experiment="er-poc",
        tags={"team": "er", "phase": "poc"},
        llm_model="deepseek/deepseek-chat",
        blocking_k=5,
        cascade_band=(0.3, 0.7),
        resolver_config={"blocker": {"type_name": "AllPairsBlocker", "random_state": 42}},
        dataset_name="febrl4",
        dataset_fingerprint="fp-deadbeef",
        split_id="test",
        seeds={"split": 7, "blocker.random_state": 42},
    )


# ---------------------------------------------------------------------------
# start_run: store config + flattened params/tags
# ---------------------------------------------------------------------------


def test_start_run_configures_store_and_opens_run(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    module, fake = mlflow_env
    tracker = module.MlflowTracker(
        _settings(mlflow_tracking_uri="https://mlflow.example.com", mlflow_experiment="er-exp")
    )

    tracker.start_run(_make_context(), run_name="er-poc")

    assert fake.tracking_uri == "https://mlflow.example.com"
    assert fake.experiment == "er-exp"
    assert fake.run_name == "er-poc"
    assert tracker.name == "mlflow"


def test_start_run_flattens_context_into_dotted_params(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    module, fake = mlflow_env
    tracker = module.MlflowTracker(_settings(mlflow_tracking_uri="https://mlflow.example.com"))

    tracker.start_run(_make_context())

    # Scalars, nested config, seeds.*, and the tuple all flatten to params.
    assert fake.params["experiment"] == "er-poc"
    assert fake.params["llm_model"] == "deepseek/deepseek-chat"
    assert fake.params["blocking_k"] == "5"
    assert fake.params["dataset_fingerprint"] == "fp-deadbeef"
    assert fake.params["seeds.split"] == "7"
    assert fake.params["seeds.blocker.random_state"] == "42"
    assert fake.params["resolver_config.blocker.type_name"] == "AllPairsBlocker"
    assert fake.params["cascade_band[0]"] == "0.3"
    assert fake.params["cascade_band[1]"] == "0.7"
    # tags are routed to MLflow tags, NOT duplicated as params.
    assert "tags.team" not in fake.params
    assert "tags" not in fake.params


def test_start_run_routes_context_tags_to_mlflow_tags(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    module, fake = mlflow_env
    tracker = module.MlflowTracker(_settings())

    tracker.start_run(_make_context())

    assert fake.tags["team"] == "er"
    assert fake.tags["phase"] == "poc"


def test_start_run_without_tracking_uri_skips_set_tracking_uri(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    """Unset URI -> MLflow's own local ./mlruns default; adapter must not force one."""
    module, fake = mlflow_env
    tracker = module.MlflowTracker(_settings(mlflow_tracking_uri=None))

    tracker.start_run(_make_context())

    assert fake.tracking_uri is None
    assert fake.experiment == "langres"  # the Settings default


# ---------------------------------------------------------------------------
# start_run: MLflow 3.14 local-file-store maintenance-mode flag (F1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri",
    [
        None,  # unset -> file-store fallback (MLflow <3.14's ./mlruns); allow it
        "file:///tmp/mlruns",  # explicit file: URI
        "./mlruns",  # scheme-less local path
    ],
)
def test_start_run_allows_local_file_store_out_of_the_box(
    mlflow_env: tuple[ModuleType, _FakeMlflow], uri: str | None
) -> None:
    """Unset/local-file config -> adapter sets MLFLOW_ALLOW_FILE_STORE.

    So a file-store fallback never trips MLflow 3.14's maintenance-mode guard.
    """
    module, _ = mlflow_env
    tracker = module.MlflowTracker(_settings(mlflow_tracking_uri=uri))

    tracker.start_run(_make_context())

    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


@pytest.mark.parametrize(
    "uri",
    [
        "https://mlflow.example.com",  # HTTP tracking server
        "http://localhost:5000",
        "sqlite:///mlflow.db",  # SQLAlchemy backend
        "postgresql://user@host/db",
    ],
)
def test_start_run_does_not_set_file_store_flag_for_http_or_sql(
    mlflow_env: tuple[ModuleType, _FakeMlflow], uri: str
) -> None:
    """HTTP/SQL backends aren't file stores -> the maintenance-mode flag is never set."""
    module, _ = mlflow_env
    tracker = module.MlflowTracker(_settings(mlflow_tracking_uri=uri))

    tracker.start_run(_make_context())

    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_start_run_preserves_explicit_file_store_flag(
    mlflow_env: tuple[ModuleType, _FakeMlflow], monkeypatch: pytest.MonkeyPatch
) -> None:
    """setdefault semantics: an explicit user value is never clobbered by the adapter."""
    module, _ = mlflow_env
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    tracker = module.MlflowTracker(_settings(mlflow_tracking_uri=None))

    tracker.start_run(_make_context())

    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "false"


# ---------------------------------------------------------------------------
# log_params / log_metrics / log_artifact
# ---------------------------------------------------------------------------


def test_log_params_stringifies_and_skips_empty(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    module, fake = mlflow_env
    tracker = module.MlflowTracker(_settings())

    tracker.log_params({})  # empty -> no MLflow call, no crash
    assert fake.params == {}

    tracker.log_params({"blocking_k": 5, "flag": True})
    assert fake.params == {"blocking_k": "5", "flag": "True"}


def test_log_metrics_forwards_with_step(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    module, fake = mlflow_env
    tracker = module.MlflowTracker(_settings())

    tracker.log_metrics({})  # empty -> skipped
    tracker.log_metrics({"pair_f1": 0.91}, step=3)
    tracker.log_metrics({"bcubed_f1": 0.87})

    assert fake.metrics == [({"pair_f1": 0.91}, 3), ({"bcubed_f1": 0.87}, None)]


def test_log_artifact_dir_file_and_url(
    mlflow_env: tuple[ModuleType, _FakeMlflow], tmp_path: Any
) -> None:
    module, fake = mlflow_env
    tracker = module.MlflowTracker(_settings())

    a_dir = tmp_path / "resolver_save"
    a_dir.mkdir()
    a_file = tmp_path / "report.md"
    a_file.write_text("# report\n")

    tracker.log_artifact("resolver", str(a_dir))
    tracker.log_artifact("report", str(a_file))
    tracker.log_artifact("wandb_run_url", "https://wandb.ai/x/y/runs/z")

    assert fake.artifact_dirs == [str(a_dir)]
    assert fake.artifact_files == [str(a_file)]
    # A URL/reference (not a local path) is recorded as a tag, never uploaded.
    assert fake.tags["wandb_run_url"] == "https://wandb.ai/x/y/runs/z"


# ---------------------------------------------------------------------------
# finish: status mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", "FINISHED"),
        ("failed", "FAILED"),
        ("budget_exceeded", "KILLED"),
        ("running", "RUNNING"),
    ],
)
def test_finish_maps_status_and_stamps_langres_status_tag(
    mlflow_env: tuple[ModuleType, _FakeMlflow], status: str, expected: str
) -> None:
    module, fake = mlflow_env
    tracker = module.MlflowTracker(_settings())
    tracker.start_run(_make_context())

    tracker.finish(status=status)

    assert fake.ended_status == expected
    assert fake.tags["langres.status"] == status


def test_finish_maps_unknown_status_to_finished(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    module, fake = mlflow_env
    tracker = module.MlflowTracker(_settings())
    tracker.start_run(_make_context())

    tracker.finish(status="something-unexpected")

    assert fake.ended_status == "FINISHED"


def test_finish_is_noop_before_start(mlflow_env: tuple[ModuleType, _FakeMlflow]) -> None:
    module, fake = mlflow_env
    tracker = module.MlflowTracker(_settings())

    tracker.finish(status="completed")

    assert fake.ended_status is None
    assert "langres.status" not in fake.tags


# ---------------------------------------------------------------------------
# run_url + native
# ---------------------------------------------------------------------------


def test_run_url_derives_deep_link_from_http_server(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    module, _ = mlflow_env
    tracker = module.MlflowTracker(
        _settings(mlflow_tracking_uri="https://mlflow.example.com/")  # trailing slash
    )
    tracker.start_run(_make_context())

    assert tracker.run_url == "https://mlflow.example.com/#/experiments/exp-1/runs/run-abc123"


def test_run_url_is_none_for_local_file_store(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    module, _ = mlflow_env
    tracker = module.MlflowTracker(_settings(mlflow_tracking_uri=None))
    tracker.start_run(_make_context())

    assert tracker.run_url is None


def test_run_url_is_none_before_start(mlflow_env: tuple[ModuleType, _FakeMlflow]) -> None:
    module, _ = mlflow_env
    tracker = module.MlflowTracker(_settings(mlflow_tracking_uri="https://mlflow.example.com"))

    assert tracker.run_url is None


def test_native_returns_the_underlying_run(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    module, fake = mlflow_env
    tracker = module.MlflowTracker(_settings())

    assert tracker.native is None  # pre-start escape hatch
    tracker.start_run(_make_context())
    assert tracker.native is fake._run


# ---------------------------------------------------------------------------
# resolve_tracker wiring (the S1 seam -> S3 adapter)
# ---------------------------------------------------------------------------


def test_resolve_tracker_returns_real_mlflow_tracker(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    """With the (mocked) module importable, resolve_tracker('mlflow') returns it."""
    module, _ = mlflow_env
    from langres.core.trackers import resolve_tracker

    tracker = resolve_tracker("mlflow")

    assert isinstance(tracker, module.MlflowTracker)
    assert tracker.name == "mlflow"


def test_getattr_exposes_mlflow_tracker_class(
    mlflow_env: tuple[ModuleType, _FakeMlflow],
) -> None:
    """`from langres.core.trackers import MlflowTracker` resolves via lazy __getattr__."""
    module, _ = mlflow_env
    import langres.core.trackers as trackers

    assert trackers.MlflowTracker is module.MlflowTracker
