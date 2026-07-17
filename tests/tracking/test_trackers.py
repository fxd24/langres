"""Tests for the dep-free ``ExperimentTracker`` layer (Stream S1).

Covers the pure logic: the ``NoOpTracker`` null object, ``MultiTracker``
fan-out, ``resolve_tracker`` dispatch (every branch), and the lazy
``MlflowTracker``/``WandbTracker`` module ``__getattr__`` -- which must raise a
helpful ``ImportError`` naming the real extra when the backend/adapter is
absent. S3 landed the ``mlflow`` adapter (and made ``mlflow`` a real dev dep),
so its missing-extra case is now *simulated*; ``wandb``'s adapter is still
absent until S4.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Callable
from typing import Any

import pytest

from langres.tracking.trackers import (
    ExperimentTracker,
    MultiTracker,
    NoOpTracker,
    resolve_tracker,
)


class _SpyTracker:
    """Minimal in-memory tracker implementing the Protocol -- records every call."""

    name = "spy"

    def __init__(self, *, url: str | None = None) -> None:
        self._url = url
        self.calls: list[tuple[str, Any]] = []

    def start_run(self, context: Any, *, run_name: str | None = None) -> None:
        self.calls.append(("start_run", run_name))

    def log_params(self, params: Any) -> None:
        self.calls.append(("log_params", dict(params)))

    def log_metrics(self, metrics: Any, *, step: int | None = None) -> None:
        self.calls.append(("log_metrics", (dict(metrics), step)))

    def log_artifact(self, key: str, value: str) -> None:
        self.calls.append(("log_artifact", (key, value)))

    def set_tags(self, tags: Any) -> None:
        self.calls.append(("set_tags", dict(tags)))

    def finish(self, *, status: str) -> None:
        self.calls.append(("finish", status))

    @property
    def run_url(self) -> str | None:
        return self._url

    @property
    def native(self) -> Any:
        return self


class TestNoOpTracker:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(NoOpTracker(), ExperimentTracker)

    def test_methods_are_noops_and_return_none(self) -> None:
        tracker = NoOpTracker()
        assert tracker.start_run(None, run_name="x") is None
        assert tracker.log_params({"a": 1}) is None
        assert tracker.log_metrics({"m": 1.0}, step=3) is None
        assert tracker.log_artifact("k", "v") is None
        assert tracker.set_tags({"t": "v"}) is None
        assert tracker.finish(status="completed") is None

    def test_run_url_and_native_are_none(self) -> None:
        tracker = NoOpTracker()
        assert tracker.run_url is None
        assert tracker.native is None

    def test_has_a_name(self) -> None:
        assert isinstance(NoOpTracker().name, str)


class _BoomTracker:
    """A Protocol-satisfying tracker that raises on one named method.

    Used to prove :class:`MultiTracker` isolates a raising child so the remaining
    children still get the call and the exception never propagates.
    """

    name = "boom"

    def __init__(self, method: str) -> None:
        self._method = method

    def _maybe_boom(self, method: str) -> None:
        if method == self._method:
            raise RuntimeError(f"boom in {method}")

    def start_run(self, context: Any, *, run_name: str | None = None) -> None:
        self._maybe_boom("start_run")

    def log_params(self, params: Any) -> None:
        self._maybe_boom("log_params")

    def log_metrics(self, metrics: Any, *, step: int | None = None) -> None:
        self._maybe_boom("log_metrics")

    def log_artifact(self, key: str, value: str) -> None:
        self._maybe_boom("log_artifact")

    def set_tags(self, tags: Any) -> None:
        self._maybe_boom("set_tags")

    def finish(self, *, status: str) -> None:
        self._maybe_boom("finish")

    @property
    def run_url(self) -> str | None:
        return None

    @property
    def native(self) -> Any:
        return self


class TestMultiTracker:
    def test_exposes_trackers_list(self) -> None:
        children = [_SpyTracker(), _SpyTracker()]
        multi = MultiTracker(children)
        assert multi.trackers == children
        assert isinstance(multi.trackers, list)

    def test_fans_out_every_call_to_children(self) -> None:
        a, b = _SpyTracker(), _SpyTracker()
        multi = MultiTracker([a, b])
        multi.start_run("ctx", run_name="r")
        multi.log_params({"p": 1})
        multi.log_metrics({"m": 2.0}, step=1)
        multi.log_artifact("k", "v")
        multi.set_tags({"t": "x"})
        multi.finish(status="completed")
        for child in (a, b):
            kinds = [c[0] for c in child.calls]
            assert kinds == [
                "start_run",
                "log_params",
                "log_metrics",
                "log_artifact",
                "set_tags",
                "finish",
            ]

    def test_run_url_returns_first_non_none_child_url(self) -> None:
        multi = MultiTracker([_SpyTracker(url=None), _SpyTracker(url="http://x")])
        assert multi.run_url == "http://x"

    def test_run_url_none_when_no_child_has_one(self) -> None:
        assert MultiTracker([_SpyTracker(), _SpyTracker()]).run_url is None

    def test_satisfies_protocol(self) -> None:
        assert isinstance(MultiTracker([]), ExperimentTracker)

    def test_native_exposes_child_list(self) -> None:
        children = [_SpyTracker(), _SpyTracker()]
        assert MultiTracker(children).native == children


class TestMultiTrackerErrorIsolation:
    """M3: one child raising must not abort the fan-out or propagate."""

    def test_child_raising_on_finish_still_finishes_others_and_does_not_propagate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        boom, spy = _BoomTracker("finish"), _SpyTracker()
        multi = MultiTracker([boom, spy])

        with caplog.at_level(logging.ERROR, logger="langres.tracking.trackers"):
            multi.finish(status="completed")  # must NOT raise

        # The second child still received finish despite the first raising.
        assert ("finish", "completed") in spy.calls
        # The failure was logged (not printed), naming the failing method.
        assert any("finish" in record.getMessage() for record in caplog.records)

    def test_every_fanout_method_isolates_a_raising_child(self) -> None:
        cases: list[tuple[str, Callable[[MultiTracker], None]]] = [
            ("start_run", lambda m: m.start_run("ctx", run_name="r")),
            ("log_params", lambda m: m.log_params({"p": 1})),
            ("log_metrics", lambda m: m.log_metrics({"x": 1.0}, step=1)),
            ("log_artifact", lambda m: m.log_artifact("k", "v")),
            ("set_tags", lambda m: m.set_tags({"t": "v"})),
            ("finish", lambda m: m.finish(status="completed")),
        ]
        for method, call in cases:
            spy = _SpyTracker()
            multi = MultiTracker([_BoomTracker(method), spy])
            call(multi)  # must NOT raise for any method
            assert spy.calls and spy.calls[-1][0] == method


class TestResolveTracker:
    def test_none_returns_noop(self) -> None:
        assert isinstance(resolve_tracker(None), NoOpTracker)

    def test_instance_passed_through(self) -> None:
        spy = _SpyTracker()
        assert resolve_tracker(spy) is spy

    def test_sequence_becomes_multitracker(self) -> None:
        a, b = _SpyTracker(), _SpyTracker()
        resolved = resolve_tracker([a, b])
        assert isinstance(resolved, MultiTracker)
        assert resolved.trackers == [a, b]

    def test_tuple_becomes_multitracker(self) -> None:
        a, b = _SpyTracker(), _SpyTracker()
        resolved = resolve_tracker((a, b))
        assert isinstance(resolved, MultiTracker)
        assert list(resolved.trackers) == [a, b]

    def test_sequence_of_specs_is_resolved_recursively(self) -> None:
        spy = _SpyTracker()
        resolved = resolve_tracker([None, spy])
        assert isinstance(resolved, MultiTracker)
        assert isinstance(resolved.trackers[0], NoOpTracker)
        assert resolved.trackers[1] is spy

    def test_mlflow_string_raises_helpful_import_error_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing ``mlflow`` extra -> a helpful ``langres[mlflow]`` ImportError.

        S3 landed the adapter module and made ``mlflow`` a real dev/CI dependency,
        so genuine absence no longer occurs in this env. Simulate it by forcing the
        adapter import to fail (mirrors ``_fake_import`` below) -- the missing-extra
        translation is what ``resolve_tracker`` must surface either way.
        """
        import langres.tracking.trackers as trackers_mod

        real_import = trackers_mod.importlib.import_module

        def _fail_mlflow_import(path: str, *a: Any, **k: Any) -> Any:
            if path == "langres.tracking.trackers.mlflow_tracker":
                raise ImportError("No module named 'mlflow'")
            return real_import(path, *a, **k)

        monkeypatch.setattr(trackers_mod.importlib, "import_module", _fail_mlflow_import)
        with pytest.raises(ImportError, match=r"langres\[mlflow\]"):
            resolve_tracker("mlflow")

    def test_wandb_string_raises_helpful_import_error_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing ``wandb`` extra -> a helpful ``langres[wandb]`` ImportError.

        S4 landed the adapter module and ``wandb`` is a real dev dependency, so
        genuine absence no longer occurs in this env. Simulate it by forcing the
        adapter import to fail (mirrors the ``mlflow`` twin above) -- the
        missing-extra translation is what ``resolve_tracker`` must surface either way.
        """
        import langres.tracking.trackers as trackers_mod

        real_import = trackers_mod.importlib.import_module

        def _fail_wandb_import(path: str, *a: Any, **k: Any) -> Any:
            if path == "langres.tracking.trackers.wandb_tracker":
                raise ImportError("No module named 'wandb'")
            return real_import(path, *a, **k)

        monkeypatch.setattr(trackers_mod.importlib, "import_module", _fail_wandb_import)
        with pytest.raises(ImportError, match=r"langres\[wandb\]"):
            resolve_tracker("wandb")

    def test_trackio_string_raises_helpful_import_error_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing ``trackio`` extra -> a helpful ``langres[trackio]`` ImportError.

        ``trackio`` is a real dev dependency in this env, so genuine absence
        doesn't occur here. Simulate it by forcing the adapter import to fail
        (mirrors the mlflow/wandb twins above).
        """
        import langres.tracking.trackers as trackers_mod

        real_import = trackers_mod.importlib.import_module

        def _fail_trackio_import(path: str, *a: Any, **k: Any) -> Any:
            if path == "langres.tracking.trackers.trackio_tracker":
                raise ImportError("No module named 'trackio'")
            return real_import(path, *a, **k)

        monkeypatch.setattr(trackers_mod.importlib, "import_module", _fail_trackio_import)
        with pytest.raises(ImportError, match=r"langres\[trackio\]"):
            resolve_tracker("trackio")

    def test_unknown_backend_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            resolve_tracker("not-a-backend")

    def test_non_spec_object_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="cannot resolve tracker"):
            resolve_tracker(42)  # type: ignore[arg-type]

    def test_mlflow_string_instantiates_adapter_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The success path: a present adapter module is imported + instantiated."""
        import langres.tracking.trackers as trackers_mod

        class _FakeMlflow(NoOpTracker):
            name = "fake-mlflow"

        fake_module = types.SimpleNamespace(MlflowTracker=_FakeMlflow)
        real_import = trackers_mod.importlib.import_module

        def _fake_import(path: str, *a: Any, **k: Any) -> Any:
            if path == "langres.tracking.trackers.mlflow_tracker":
                return fake_module
            return real_import(path, *a, **k)

        monkeypatch.setattr(trackers_mod.importlib, "import_module", _fake_import)
        assert isinstance(resolve_tracker("mlflow"), _FakeMlflow)


class TestLazyAdapterGetattr:
    def test_mlflow_tracker_attribute_raises_helpful_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``trackers.MlflowTracker`` -> helpful ImportError when the extra is absent.

        Absence is simulated (see the ``resolve_tracker`` twin above): S3 made the
        adapter module + ``mlflow`` present, so the raw missing case is forced here.
        """
        import langres.tracking.trackers as trackers

        real_import = trackers.importlib.import_module

        def _fail_mlflow_import(path: str, *a: Any, **k: Any) -> Any:
            if path == "langres.tracking.trackers.mlflow_tracker":
                raise ImportError("No module named 'mlflow'")
            return real_import(path, *a, **k)

        monkeypatch.setattr(trackers.importlib, "import_module", _fail_mlflow_import)
        with pytest.raises(ImportError, match=r"langres\[mlflow\]"):
            trackers.MlflowTracker  # noqa: B018

    def test_wandb_tracker_attribute_raises_helpful_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``trackers.WandbTracker`` -> helpful ImportError when the extra is absent.

        Absence is simulated (see the ``resolve_tracker`` twin above): S4 made the
        adapter module + ``wandb`` present, so the raw missing case is forced here.
        """
        import langres.tracking.trackers as trackers

        real_import = trackers.importlib.import_module

        def _fail_wandb_import(path: str, *a: Any, **k: Any) -> Any:
            if path == "langres.tracking.trackers.wandb_tracker":
                raise ImportError("No module named 'wandb'")
            return real_import(path, *a, **k)

        monkeypatch.setattr(trackers.importlib, "import_module", _fail_wandb_import)
        with pytest.raises(ImportError, match=r"langres\[wandb\]"):
            trackers.WandbTracker  # noqa: B018

    def test_trackio_tracker_attribute_raises_helpful_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``trackers.TrackioTracker`` -> helpful ImportError when the extra is absent."""
        import langres.tracking.trackers as trackers

        real_import = trackers.importlib.import_module

        def _fail_trackio_import(path: str, *a: Any, **k: Any) -> Any:
            if path == "langres.tracking.trackers.trackio_tracker":
                raise ImportError("No module named 'trackio'")
            return real_import(path, *a, **k)

        monkeypatch.setattr(trackers.importlib, "import_module", _fail_trackio_import)
        with pytest.raises(ImportError, match=r"langres\[trackio\]"):
            trackers.TrackioTracker  # noqa: B018

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import langres.tracking.trackers as trackers

        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            trackers.not_a_real_attribute  # noqa: B018
