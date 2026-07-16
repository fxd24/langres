"""Trackio :class:`ExperimentTracker` adapter -- the local-first, HF-optional backend.

Lazily loaded by :func:`langres.core.trackers.resolve_tracker` and
``from langres.core.trackers import TrackioTracker`` -- ``trackio`` (and its own
transitive ``huggingface_hub`` dependency) is only pulled in when this module is
actually imported, so a bare ``import langres`` never touches either (asserted
by ``tests/test_import_budget.py``). Install with ``pip install 'langres[trackio]'``.

Unlike :class:`~langres.core.trackers.wandb_tracker.WandbTracker` (which
delegates ``wandb.init`` to :func:`langres.clients.tracking.create_wandb_tracker`),
this adapter calls ``trackio.init`` directly, mirroring
:class:`~langres.core.trackers.mlflow_tracker.MlflowTracker`'s self-contained
shape. That keeps ``trackio``'s import fully scoped to *this* module -- the
factory in ``langres.clients.tracking`` (``create_trackio_tracker``) still
exists for settings-driven construction, but only imports this module lazily
inside its own function body, so a ``langres[wandb]``-only install (that file's
existing unconditional ``import wandb``) never needs ``trackio`` importable, and
vice versa.

Local-first by design: with no ``space_id`` configured, ``trackio.init`` writes
to a local SQLite store only -- no credentials, no network, no HF account. A
``space_id`` opts into syncing the run to a Hugging Face Space (and optionally a
persistent Dataset via ``dataset_id`` -- verified against the installed trackio
0.20.2 API; NOT ``bucket_id``, an unconfirmed name floated before installing).
Since HF Space sync requires a *write* token, :func:`_require_hf_token` fails
fast with an actionable :class:`ValueError` naming exactly what to set --
mirroring :func:`~langres.clients.tracking.create_wandb_tracker`'s missing-cred
guard -- rather than letting the (still correct, but less actionable) exception
surface deep inside ``huggingface_hub``. This adapter never deploys or names a
Space itself; the caller opts in explicitly via ``space_id``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import huggingface_hub
import trackio

from langres.clients.settings import Settings

if TYPE_CHECKING:
    # Type-only: the adapter receives a RunContext but never imports runs at
    # runtime (runs imports the trackers package -- keep the dependency acyclic).
    from langres.core.runs import RunContext

__all__ = ["TrackioTracker"]

#: HF Spaces' stable URL convention (matches ``trackio.deploy.SPACE_HOST_URL``,
#: not imported directly since that submodule is deploy-only internals) -- used
#: to build a real deep link, since ``trackio.Run.url`` on a Space run is just
#: the raw ``"user/space"`` id, not a browsable URL.
_SPACE_URL = "https://{user}-{space}.hf.space/"


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping/sequence into dotted-key scalars (dropping ``None``).

    Mirrors :func:`langres.core.trackers.wandb_tracker._flatten` (trackio's
    ``Run.config`` is, like W&B's, a flat scalar dict fed straight to
    ``trackio.init(config=...)``).
    """
    flat: dict[str, Any] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(value, child))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            flat.update(_flatten(value, f"{prefix}[{index}]"))
    elif obj is not None:
        flat[prefix] = obj
    return flat


def _has_hf_token(settings: Settings) -> bool:
    """True if a Hugging Face *write* token is available from any source.

    Checks, in order: an explicit ``settings.hf_token`` (constructor override),
    then ``huggingface_hub.get_token()`` -- which itself checks the ``HF_TOKEN``
    env var, the legacy ``HUGGING_FACE_HUB_TOKEN`` env var, and the token file
    written by ``hf auth login`` (verified against huggingface_hub 0.36).
    """
    return bool(settings.hf_token) or huggingface_hub.get_token() is not None


def _require_hf_token(space_id: str, settings: Settings) -> None:
    """Fail fast with an actionable error when ``space_id`` has no token to use.

    Raised BEFORE ``trackio.init`` is ever called -- ``trackio`` itself would
    eventually raise ``huggingface_hub.errors.LocalTokenNotFoundError`` for the
    same condition, but only after already resolving/creating Space state; this
    guard mirrors ``create_wandb_tracker``'s missing-cred ``ValueError`` so the
    failure is immediate and names the fix.
    """
    if not _has_hf_token(settings):
        raise ValueError(
            f"space_id={space_id!r} requires a Hugging Face token to sync this "
            "Trackio run to the Hub: set HF_TOKEN (or the legacy "
            "HUGGING_FACE_HUB_TOKEN), or run `hf auth login`, before configuring "
            "an HF-synced Trackio run. Leave space_id unset for a local-only run "
            "(no credentials needed)."
        )


class TrackioTracker:
    """Push langres runs into Trackio (the ``"trackio"`` backend) -- local by default.

    One :class:`~langres.core.runs.RunContext` becomes one Trackio run: the
    context is flattened into ``run.config`` (passed to ``trackio.init`` up
    front, so it is captured even for a single-``log`` run), metrics stream via
    ``run.log``, and ``log_artifact``/``set_tags`` do the closest honest mapping
    given trackio 0.20.2 has no dedicated artifact or tag API -- both fold into
    ``run.log``/``run.config`` respectively. :attr:`run_url` deep-links the HF
    Space dashboard when ``space_id`` is configured (``None`` for local runs, which
    have no static browsable URL outside a notebook embed); :attr:`native` is the
    escape hatch to the underlying ``trackio.Run``.
    """

    name = "trackio"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        project: str | None = None,
        space_id: str | None = None,
        dataset_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the adapter; ``trackio.init`` is deferred to :meth:`start_run`.

        Args:
            settings: Source of ``trackio_space_id``/``trackio_dataset_id``/
                ``hf_token`` fallbacks when the matching keyword is ``None``.
                ``None`` -> :class:`~langres.clients.settings.Settings` from the
                environment.
            project: The trackio project name. ``None`` -> the run's
                ``context.experiment`` at :meth:`start_run` time.
            space_id: HF Space to sync to (``"user/space"``), e.g. for durable
                off-laptop persistence. ``None`` (the default) -> a pure local
                run: no credentials, no network. Overrides
                ``settings.trackio_space_id`` when given.
            dataset_id: HF Dataset to additionally sync metrics to (requires
                ``space_id``). Overrides ``settings.trackio_dataset_id``.
            config: Extra config merged with the flattened
                :class:`~langres.core.runs.RunContext` at :meth:`start_run`.
        """
        self._settings = settings
        self._project = project
        self._space_id = space_id
        self._dataset_id = dataset_id
        self._extra_config = dict(config) if config is not None else {}
        self._run: Any = None

    def start_run(self, context: RunContext, *, run_name: str | None = None) -> None:
        """Open a trackio run, seeding its config from the flattened ``context``.

        Resolves ``space_id``/``dataset_id`` (constructor arg, else
        ``settings``), guards HF credentials when ``space_id`` is set (see
        :func:`_require_hf_token`), and defaults ``project`` to
        ``context.experiment``.
        """
        settings = self._settings or Settings()
        space_id = self._space_id if self._space_id is not None else settings.trackio_space_id
        dataset_id = (
            self._dataset_id if self._dataset_id is not None else settings.trackio_dataset_id
        )
        if space_id is not None:
            if settings.hf_token:
                # setdefault: an explicit user-exported HF_TOKEN always wins.
                os.environ.setdefault("HF_TOKEN", settings.hf_token)
            _require_hf_token(space_id, settings)

        config = {**_flatten(context.model_dump(mode="json")), **self._extra_config}
        self._run = trackio.init(
            project=self._project or context.experiment,
            name=run_name,
            config=config,
            space_id=space_id,
            dataset_id=dataset_id,
        )
        self._space_id = space_id

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Merge extra (flattened) params into the run config."""
        if self._run is not None:
            self._run.config.update(_flatten(dict(params)))

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Stream metrics to *this* run (``run.log``). A no-op before :meth:`start_run`."""
        if self._run is not None:
            self._run.log(dict(metrics), step=step)

    def log_artifact(self, key: str, value: str) -> None:
        """Record an artifact path/URL as a log entry (no dedicated artifact API)."""
        if self._run is not None:
            self._run.log({key: value})

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Fold ``key:value`` labels into the run config (no dedicated tags API).

        Prefixed ``tag.<key>`` so a tag never silently overwrites a same-named
        config entry from :meth:`start_run`/:meth:`log_params`.
        """
        if self._run is not None:
            self._run.config.update({f"tag.{key}": value for key, value in tags.items()})

    def finish(self, *, status: str) -> None:
        """Record the terminal ``status`` then close *this* run (``run.finish``).

        Trackio's ``Run.finish()`` takes no status argument, so the langres
        status is stamped as a ``langres_status`` log entry first (mirrors
        MLflow's ``langres.status`` tag). A no-op before :meth:`start_run`.
        """
        if self._run is not None:
            self._run.log({"langres_status": status})
            self._run.finish()

    @property
    def run_url(self) -> str | None:
        """Deep link to the HF Space dashboard, or ``None`` for a local run."""
        if self._run is None or self._space_id is None:
            return None
        user, space = self._space_id.split("/", 1)
        return _SPACE_URL.format(user=user, space=space)

    @property
    def native(self) -> Any:
        """The underlying ``trackio.Run`` object (escape hatch), or ``None``."""
        return self._run
