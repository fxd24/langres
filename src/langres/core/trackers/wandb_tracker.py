"""W&B (Weights & Biases) :class:`ExperimentTracker` adapter (Stream S4).

Lazily loaded by :func:`langres.core.trackers.resolve_tracker` and
``from langres.core.trackers import WandbTracker`` -- it pulls ``wandb`` only
transitively (via :func:`langres.clients.tracking.create_wandb_tracker`), so it
never lands on the eager ``import langres`` path (asserted by
``tests/test_import_budget.py``). The adapter reuses that helper for the
``wandb.init`` (project/entity/api-key resolution from
:class:`~langres.clients.settings.Settings`) and enriches the returned run --
logging, metrics, and finish all go through *that* run object, never the
module-global ``wandb.*`` functions -- with the flattened
:class:`~langres.core.runs.RunContext`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from langres.clients.settings import Settings
from langres.clients.tracking import create_wandb_tracker

if TYPE_CHECKING:
    # Type-only: the adapter receives a RunContext but never imports runs at
    # runtime (runs imports the trackers package -- keep the dependency acyclic).
    from langres.core.runs import RunContext

__all__ = ["WandbTracker"]


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping/sequence into dotted-key scalars (dropping ``None``).

    ``{"seeds": {"split": 1}}`` -> ``{"seeds.split": 1}``; a list/tuple ->
    ``prefix[0]``, ``prefix[1]``, ... . ``None`` values are dropped so the W&B
    config table stays a signal of what was actually set. This keeps the flat
    W&B config aligned with the flat MLflow params for cross-backend comparison.
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


class WandbTracker:
    """Push langres runs into Weights & Biases (the ``"wandb"`` backend).

    One :class:`~langres.core.runs.RunContext` becomes one W&B run: the context
    is flattened into ``run.config``, metrics stream via ``run.log``, and
    artifact pointers land in ``run.summary``. :attr:`run_url` deep-links the W&B
    UI (threaded into :attr:`~langres.core.runs.RunRecord.artifacts`);
    :attr:`native` is the escape hatch to the underlying ``wandb`` run.
    """

    name = "wandb"

    def __init__(self, settings: Settings | None = None, *, job_type: str = "experiment") -> None:
        """Configure the adapter; ``wandb.init`` is deferred to :meth:`start_run`.

        Args:
            settings: Source of ``wandb`` project/entity/api-key. ``None`` ->
                :class:`~langres.clients.settings.Settings` from the environment.
            job_type: W&B ``job_type`` categorization for the run.
        """
        self._settings = settings
        self._job_type = job_type
        self._run: Any = None

    def start_run(self, context: RunContext, *, run_name: str | None = None) -> None:
        """Open a W&B run and seed its config from the flattened ``context``."""
        self._run = create_wandb_tracker(self._settings, job_type=self._job_type)
        if run_name is not None:
            self._run.name = run_name
        self._run.config.update(_flatten(context.model_dump(mode="json")), allow_val_change=True)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Merge extra (flattened) params into the run config."""
        if self._run is not None:
            self._run.config.update(_flatten(dict(params)), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Stream metrics to *this* run (``run.log``), not the module-global run.

        Routed through ``self._run`` so a nested/second ``WandbTracker`` logs to its
        own run instead of whatever run is globally active (possibly the parent's).
        A no-op before :meth:`start_run`.
        """
        if self._run is not None:
            self._run.log(dict(metrics), step=step)

    def log_artifact(self, key: str, value: str) -> None:
        """Record an artifact path/URL as a run-summary entry (an output pointer)."""
        if self._run is not None:
            self._run.summary[key] = value

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Attach ``key:value`` labels to the W&B run (W&B tags are flat labels)."""
        if self._run is not None:
            self._run.tags = tuple(f"{key}:{value}" for key, value in tags.items())

    def finish(self, *, status: str) -> None:
        """Close *this* run (``run.finish``), mapping status -> exit code (``0``/``1``).

        Routed through ``self._run`` so a child's ``finish()`` ends only its own run
        -- ``wandb.finish()`` (module-global) would end whatever run is globally
        active, possibly the parent's. A no-op before :meth:`start_run`.
        """
        if self._run is not None:
            self._run.summary["status"] = status
            self._run.finish(exit_code=0 if status == "completed" else 1)

    @property
    def run_url(self) -> str | None:
        """Deep link to the W&B run UI (``None`` before start / when offline)."""
        if self._run is None:
            return None
        url = getattr(self._run, "url", None)
        return url if isinstance(url, str) else None

    @property
    def native(self) -> Any:
        """The underlying ``wandb`` run object (escape hatch), or ``None``."""
        return self._run
