"""MLflow adapter for the langres :class:`ExperimentTracker` seam (Stream S3).

Lazily loaded: ``import mlflow`` sits at this module's top level, but the module
itself is only imported on first attribute access -- via the package
``__getattr__`` / :func:`~langres.tracking.trackers.resolve_tracker` in
``langres.tracking.trackers.__init__`` (see its ``_ADAPTERS`` table). So a bare
``import langres`` never pulls ``mlflow`` into ``sys.modules``; only wiring a
``"mlflow"`` tracker does (guarded by ``tests/test_import_budget.py``). Install
with ``pip install 'langres[mlflow]'``.

The adapter flattens a :class:`~langres.tracking.runs.RunContext` into MLflow params
itself (nested dicts/lists -> dotted keys; ``context.tags`` -> MLflow tags),
streams metrics as the run progresses, uploads local file/dir artifacts (and
records URL/reference artifacts as tags), and derives a deep-link ``run_url``
when the tracking URI is an HTTP server.

Local-store default (the F1 fix): with no ``mlflow_tracking_uri`` configured the
adapter leaves MLflow to its own default backend, which is version-dependent --
MLflow <3.14 uses the local ``./mlruns`` FileStore, MLflow 3.14+ a local
``sqlite:///mlflow.db``. MLflow 3.14 also puts *any* local FileStore in
"maintenance mode" and refuses to open a run against it unless
``MLFLOW_ALLOW_FILE_STORE=true``. So :meth:`MlflowTracker.start_run` sets that
flag (via ``os.environ.setdefault`` -- an explicit user value is never clobbered)
whenever the configured store is a local file store: an unset URI (the old
``./mlruns`` fallback still in play on MLflow <3.14) or an explicit ``file:`` /
local-path ``mlflow_tracking_uri``. It is **not** set for HTTP tracking servers
or SQLAlchemy (``sqlite:``/``postgresql:``/...) backends -- including MLflow
3.14's own sqlite default -- which don't need it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import mlflow

from langres.clients.settings import Settings

if TYPE_CHECKING:
    # Type-only: the adapter receives a RunContext but never imports runs at
    # runtime (runs imports the trackers package for its NoOpTracker default).
    from langres.tracking.runs import RunContext

#: langres run status -> MLflow ``RunStatus`` enum value. MLflow has no
#: "budget capped" state, so ``budget_exceeded`` maps to ``KILLED`` (a
#: deliberate termination); the precise langres status is preserved losslessly
#: as a ``langres.status`` tag in :meth:`MlflowTracker.finish`.
_STATUS_TO_MLFLOW: dict[str, str] = {
    "running": "RUNNING",
    "completed": "FINISHED",
    "failed": "FAILED",
    "budget_exceeded": "KILLED",
}


#: Characters MLflow forbids in a param/metric name. MLflow allows only
#: alphanumerics, ``_ - . space /`` (and ``:`` off Windows); we sanitize to the
#: Windows-safe subset (no colon) so a key validates on every platform.
_MLFLOW_UNSAFE_KEY_CHARS = re.compile(r"[^\w.\-/ ]")


def _sanitize_key(key: str) -> str:
    """Coerce a flattened key into an MLflow-safe param name.

    MLflow rejects any name outside ``[alphanumerics _ - . space /]`` -- a raw
    ``key[0]`` from a sequence field, or a config key carrying a stray symbol,
    would crash ``mlflow.log_params``. Every out-of-charset character is replaced
    with ``_`` so *any* context logs without a crash.
    """
    return _MLFLOW_UNSAFE_KEY_CHARS.sub("_", key)


def _flatten(value: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict/list into ``{dotted.key: str value}`` for MLflow params.

    MLflow params are flat, scalar key/value pairs, so nested config
    (``resolver_config``, ``seeds``) is expanded to dotted keys (``seeds.split``,
    ``resolver_config.blocker.type_name``) and list items to dotted indices
    (``cascade_band.0``). ``.`` is used for indices (not ``[0]``) because ``[``/``]``
    are outside MLflow's allowed param-name charset; :func:`_sanitize_key` is the
    backstop for any remaining out-of-charset character. Leaf scalars are
    stringified; the top-level call is always a mapping, so the empty ``prefix``
    never yields a keyless entry.
    """
    flat: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flat.update(_flatten(child, child_prefix))
    elif prefix:
        flat[prefix] = str(value)
    return flat


def _is_local_file_store(tracking_uri: str | None) -> bool:
    """True when the configured tracking URI selects (or falls back to) a FileStore.

    An unset URI (``None``) is treated as a file store: on MLflow <3.14 it falls
    back to the local ``./mlruns`` FileStore, so we proactively allow it. A
    scheme-less local path or an explicit ``file:`` URI is a FileStore too.
    HTTP(S) tracking servers and SQLAlchemy (``sqlite:``/``postgresql:``/...)
    backends are NOT file stores -- they don't need the maintenance-mode flag.
    """
    if tracking_uri is None:
        return True
    return urlparse(tracking_uri).scheme in ("", "file")


class MlflowTracker:
    """Push langres runs into MLflow -- one adapter instance per :func:`capture_run`.

    Reads its store config from :class:`~langres.clients.settings.Settings`
    (``mlflow_tracking_uri`` -- unset lets MLflow pick its own default backend,
    a local sqlite db on MLflow 3.14 -- and ``mlflow_experiment``; see the module
    docstring for the local-file-store handling). ``start_run`` flattens the
    :class:`~langres.tracking.runs.RunContext` into params/tags itself, since the
    :func:`capture_run` driver only calls ``start_run`` / ``log_metrics`` /
    ``log_artifact`` / ``finish``.
    """

    name = "mlflow"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._tracking_uri = self._settings.mlflow_tracking_uri
        self._run: Any = None
        self._run_id: str | None = None
        self._experiment_id: str | None = None

    def start_run(self, context: RunContext, *, run_name: str | None = None) -> None:
        """Open an MLflow run and stamp the flattened context onto it as params/tags."""
        if _is_local_file_store(self._tracking_uri):
            # MLflow >=3.14 puts the local ./mlruns FileStore in "maintenance
            # mode" and refuses to open a run unless this flag is set. setdefault
            # so an explicit user override wins; never set it for http/sql stores.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        if self._tracking_uri is not None:
            mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._settings.mlflow_experiment)
        # nested=True when a run is already active, so a nested capture_run (e.g. a
        # DSPy compile() run inside an outer benchmark run) does not trip MLflow's
        # "Run ... is already active" guard.
        self._run = mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None)
        self._run_id = self._run.info.run_id
        self._experiment_id = self._run.info.experiment_id

        dumped = context.model_dump(mode="json", exclude_none=True)
        tags = dumped.pop("tags", {})
        self.log_params(_flatten(dumped))
        self.set_tags(tags)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Flatten and log params (keys sanitized, values stringified).

        Keys pass through :func:`_sanitize_key` -- MLflow param names allow only a
        restricted charset, so any out-of-charset character (e.g. from a sequence
        or a config key) is coerced to ``_`` rather than crashing the run.
        """
        flattened = _flatten(params)
        if flattened:
            mlflow.log_params({_sanitize_key(key): value for key, value in flattened.items()})

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Stream numeric metrics, optionally at a training ``step``."""
        if metrics:
            mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, key: str, value: str) -> None:
        """Upload a local file/dir artifact; record a URL/reference as a tag.

        ``value`` may be a local path (``resolver.save`` dir, a report file) or a
        reference (a backend URL). Real local paths are uploaded to the MLflow
        artifact store; anything else (a URL, a missing path) is kept as a tag so
        the reference still surfaces in the run.
        """
        path = Path(value)
        if path.is_dir():
            mlflow.log_artifacts(str(path))
        elif path.is_file():
            mlflow.log_artifact(str(path))
        else:
            mlflow.set_tag(key, value)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Attach searchable tags to the active run."""
        if tags:
            mlflow.set_tags(dict(tags))

    def finish(self, *, status: str) -> None:
        """Close the run, mapping the langres status to MLflow's ``RunStatus``.

        A no-op if no run was ever started. The exact langres status (which can
        be finer-grained than MLflow's enum, e.g. ``budget_exceeded``) is also
        stamped as a ``langres.status`` tag so it stays queryable.
        """
        if self._run is None:
            return
        mlflow.set_tag("langres.status", status)
        mlflow.end_run(status=_STATUS_TO_MLFLOW.get(status, "FINISHED"))

    @property
    def run_url(self) -> str | None:
        """Deep link into the MLflow UI, or ``None`` for a local file store.

        Buildable only against an HTTP tracking server (``http(s)://``); the
        default local ``./mlruns`` store has no browsable URL, so this returns
        ``None`` and no ``run_url`` artifact is threaded into the record.
        """
        if self._run_id is None or self._experiment_id is None:
            return None
        uri = self._tracking_uri
        if uri and (uri.startswith("http://") or uri.startswith("https://")):
            return f"{uri.rstrip('/')}/#/experiments/{self._experiment_id}/runs/{self._run_id}"
        return None

    @property
    def native(self) -> Any:
        """The underlying MLflow run object -- the escape hatch (``None`` pre-start)."""
        return self._run
