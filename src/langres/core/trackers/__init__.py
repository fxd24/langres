"""Pluggable experiment-tracker layer -- dep-free Protocol + null/fan-out impls.

The tracking layer follows the Accelerate ``GeneralTracker`` shape: langres owns
the run schema (:mod:`langres.core.runs`) and *pushes* it into whatever backend
the caller wired, symmetrically -- there is no privileged default backend. This
module carries only the seam: the :class:`ExperimentTracker` Protocol, the
:class:`NoOpTracker` null object (zero overhead when unconfigured), the
:class:`MultiTracker` fan-out (run MLflow *and* W&B together), and
:func:`resolve_tracker` (``None|"mlflow"|"wandb"|instance|sequence`` dispatch,
mirroring :func:`langres.core.presets.resolve_judge`).

The concrete backend adapters (``MlflowTracker`` in ``mlflow_tracker.py``,
``WandbTracker`` in ``wandb_tracker.py``, ``TrackioTracker`` in
``trackio_tracker.py``) pull their heavy dependency (``mlflow`` / ``wandb`` /
``trackio``) lazily. They are exposed here via a PEP 562 module ``__getattr__``
so ``from langres.core.trackers import MlflowTracker`` works, but importing this
package -- and therefore ``import langres`` -- never pulls
``mlflow``/``wandb``/``trackio`` into ``sys.modules``. When an adapter's extra
isn't installed, accessing the name (or ``resolve_tracker("mlflow")``) raises a
clear :class:`ImportError` naming the ``pip install 'langres[<backend>]'`` extra
to install.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Type-only: the tracker sees a RunContext but never imports runs at runtime
    # (runs imports this module for its NoOpTracker default -- keep it acyclic).
    from langres.core.runs import RunContext

logger = logging.getLogger(__name__)

__all__ = [
    "ExperimentTracker",
    "MultiTracker",
    "NoOpTracker",
    "resolve_tracker",
]

#: ``backend name -> (adapter module, adapter class)`` for the lazily-loaded
#: concrete trackers. Both the module ``__getattr__`` and
#: :func:`resolve_tracker` resolve through :func:`_load_adapter_class` so the
#: missing-extra ``ImportError`` message is identical from either entry point.
_ADAPTERS: dict[str, tuple[str, str]] = {
    "mlflow": ("langres.core.trackers.mlflow_tracker", "MlflowTracker"),
    "wandb": ("langres.core.trackers.wandb_tracker", "WandbTracker"),
    "trackio": ("langres.core.trackers.trackio_tracker", "TrackioTracker"),
}


@runtime_checkable
class ExperimentTracker(Protocol):
    """The one seam a backend implements to receive langres runs.

    Shaped after HuggingFace Accelerate's ``GeneralTracker``: langres flattens
    the :class:`~langres.core.runs.RunContext` into params/tags the backend
    understands and streams metrics/artifacts as the run progresses.
    ``run_url`` deep-links back into the backend UI (threaded into
    :attr:`~langres.core.runs.RunRecord.artifacts`); ``native`` is the escape
    hatch to the underlying backend client for backend-specific calls.
    """

    name: str

    # Interface stubs -- excluded from coverage (declarations, never executed).
    def start_run(  # pragma: no cover
        self, context: RunContext, *, run_name: str | None = None
    ) -> None: ...

    def log_params(self, params: Mapping[str, Any]) -> None: ...  # pragma: no cover

    def log_metrics(  # pragma: no cover
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None: ...

    def log_artifact(self, key: str, value: str) -> None: ...  # pragma: no cover

    def set_tags(self, tags: Mapping[str, str]) -> None: ...  # pragma: no cover

    def finish(self, *, status: str) -> None: ...  # pragma: no cover

    @property
    def run_url(self) -> str | None: ...  # pragma: no cover

    @property
    def native(self) -> Any: ...  # pragma: no cover


class NoOpTracker:
    """The null tracker: every call is a no-op. The zero-overhead default.

    Used whenever no backend is configured (``resolve_tracker(None)``), so the
    tracking seam is always present and callers never branch on ``if tracker``.
    """

    name = "noop"

    def start_run(self, context: RunContext, *, run_name: str | None = None) -> None:
        return None

    def log_params(self, params: Mapping[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        return None

    def log_artifact(self, key: str, value: str) -> None:
        return None

    def set_tags(self, tags: Mapping[str, str]) -> None:
        return None

    def finish(self, *, status: str) -> None:
        return None

    @property
    def run_url(self) -> str | None:
        return None

    @property
    def native(self) -> Any:
        return None


class MultiTracker:
    """Fan-out tracker: forward every call to N children (compose, don't merge).

    Lets a single run land in MLflow *and* W&B at once. Composition, not
    merging: children keep their own identity, reachable via :attr:`trackers`
    so a caller can grab a specific backend (e.g. its ``native`` client) when
    running several together.
    """

    name = "multi"

    def __init__(self, trackers: Sequence[ExperimentTracker]) -> None:
        self.trackers: list[ExperimentTracker] = list(trackers)

    def _fan_out(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call ``method`` on every child, isolating per-child failures.

        A child raising must not abort the fan-out or mask a real exception --
        this matters most for ``finish``, which runs inside ``capture_run``'s
        ``finally`` where a raising child would otherwise swallow the user's own
        error. So each call is guarded: a failure is logged (never printed) and
        the remaining children still get the call.
        """
        for tracker in self.trackers:
            try:
                getattr(tracker, method)(*args, **kwargs)
            except Exception:
                logger.exception(
                    "tracker %r raised in %s(); continuing with the remaining trackers",
                    getattr(tracker, "name", tracker),
                    method,
                )

    def start_run(self, context: RunContext, *, run_name: str | None = None) -> None:
        self._fan_out("start_run", context, run_name=run_name)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._fan_out("log_params", params)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        self._fan_out("log_metrics", metrics, step=step)

    def log_artifact(self, key: str, value: str) -> None:
        self._fan_out("log_artifact", key, value)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._fan_out("set_tags", tags)

    def finish(self, *, status: str) -> None:
        self._fan_out("finish", status=status)

    @property
    def run_url(self) -> str | None:
        """The first child's deep link, if any (a single record holds one URL)."""
        for tracker in self.trackers:
            if tracker.run_url is not None:
                return tracker.run_url
        return None

    @property
    def native(self) -> Any:
        """The child list -- reach a specific backend via :attr:`trackers` instead."""
        return self.trackers


def _load_adapter_class(backend: str) -> type[Any]:
    """Import a backend adapter class, or raise a helpful missing-extra ImportError.

    The adapter modules pull ``mlflow``/``wandb`` at their own module level, so
    a missing extra (or a not-yet-added adapter module) surfaces here as an
    ``ImportError`` naming the exact ``pip install 'langres[<backend>]'`` fix
    instead of a raw ``ModuleNotFoundError`` two frames deep.
    """
    module_path, class_name = _ADAPTERS[backend]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"the {backend!r} tracker requires the {backend!r} extra: "
            f"pip install 'langres[{backend}]' (or uv add 'langres[{backend}]')"
        ) from exc
    return getattr(module, class_name)  # type: ignore[no-any-return]


def resolve_tracker(
    spec: None | str | ExperimentTracker | Sequence[Any],
) -> ExperimentTracker:
    """Resolve a tracker spec to a concrete :class:`ExperimentTracker`.

    Mirrors :func:`langres.core.presets.resolve_judge`:

    * ``None`` -> :class:`NoOpTracker` (no persistence, zero overhead).
    * ``"mlflow"`` / ``"wandb"`` / ``"trackio"`` -> the lazily-loaded backend
      adapter (a helpful :class:`ImportError` when the extra is absent).
      ``"trackio"`` is local-first: no credentials/network unless the caller
      also configures an HF Space (``TRACKIO_SPACE_ID`` / ``TrackioTracker``).
    * an ``ExperimentTracker`` instance -> returned as-is (dependency injection).
    * a sequence of specs -> a :class:`MultiTracker` over the resolved children.

    Raises:
        ImportError: A ``"mlflow"``/``"wandb"``/``"trackio"`` spec whose extra is
            not installed.
        ValueError: An unrecognized backend string.
    """
    if spec is None:
        return NoOpTracker()
    if isinstance(spec, str):
        if spec not in _ADAPTERS:
            raise ValueError(
                f"unknown tracker backend {spec!r}; choose 'mlflow', 'wandb', "
                "'trackio', pass an ExperimentTracker instance, or a sequence of "
                "these"
            )
        return _load_adapter_class(spec)()  # type: ignore[no-any-return]
    if isinstance(spec, ExperimentTracker):
        return spec
    if isinstance(spec, Sequence):
        return MultiTracker([resolve_tracker(child) for child in spec])
    raise TypeError(
        f"cannot resolve tracker from {type(spec).__name__}; expected None, a "
        "backend name, an ExperimentTracker, or a sequence of these"
    )


def __getattr__(name: str) -> Any:
    """PEP 562: expose ``MlflowTracker``/``WandbTracker``/``TrackioTracker`` lazily.

    Keeps ``mlflow``/``wandb``/``trackio`` out of ``sys.modules`` on a bare
    ``import langres`` -- the class object is only resolved on first attribute
    access, and surfaces the missing-extra ``ImportError`` if the backend isn't
    installed.
    """
    if name == "MlflowTracker":
        return _load_adapter_class("mlflow")
    if name == "WandbTracker":
        return _load_adapter_class("wandb")
    if name == "TrackioTracker":
        return _load_adapter_class("trackio")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
