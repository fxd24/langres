"""Tests for the dep-free ``ExperimentTracker`` layer (Stream S1).

Covers the pure logic: the ``NoOpTracker`` null object, ``MultiTracker``
fan-out, ``resolve_tracker`` dispatch (every branch), and the lazy
``MlflowTracker``/``WandbTracker`` module ``__getattr__`` -- which must raise a
helpful ``ImportError`` naming the real extra while the adapter modules are
still absent (S3/S4 add them).
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from langres.core.trackers import (
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

    def test_mlflow_string_raises_helpful_import_error_when_absent(self) -> None:
        with pytest.raises(ImportError, match=r"langres\[mlflow\]"):
            resolve_tracker("mlflow")

    def test_wandb_string_raises_helpful_import_error_when_absent(self) -> None:
        with pytest.raises(ImportError, match=r"langres\[wandb\]"):
            resolve_tracker("wandb")

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
        import langres.core.trackers as trackers_mod

        class _FakeMlflow(NoOpTracker):
            name = "fake-mlflow"

        fake_module = types.SimpleNamespace(MlflowTracker=_FakeMlflow)
        real_import = trackers_mod.importlib.import_module

        def _fake_import(path: str, *a: Any, **k: Any) -> Any:
            if path == "langres.core.trackers.mlflow_tracker":
                return fake_module
            return real_import(path, *a, **k)

        monkeypatch.setattr(trackers_mod.importlib, "import_module", _fake_import)
        assert isinstance(resolve_tracker("mlflow"), _FakeMlflow)


class TestLazyAdapterGetattr:
    def test_mlflow_tracker_attribute_raises_helpful_import_error(self) -> None:
        import langres.core.trackers as trackers

        with pytest.raises(ImportError, match=r"langres\[mlflow\]"):
            trackers.MlflowTracker  # noqa: B018

    def test_wandb_tracker_attribute_raises_helpful_import_error(self) -> None:
        import langres.core.trackers as trackers

        with pytest.raises(ImportError, match=r"langres\[wandb\]"):
            trackers.WandbTracker  # noqa: B018

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import langres.core.trackers as trackers

        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            trackers.not_a_real_attribute  # noqa: B018
