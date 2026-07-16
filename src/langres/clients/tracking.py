"""Experiment-tracker client factories (wandb, trackio).

``wandb`` and ``trackio`` are each pulled in only by the factory that needs it
(``import wandb`` inside :func:`create_wandb_tracker`, the ``TrackioTracker``
import inside :func:`create_trackio_tracker`), so importing this module -- which
``from langres.clients import create_trackio_tracker`` does -- never requires the
*other* backend's extra to be installed. A trackio-only install can therefore
call ``create_trackio_tracker`` without ``wandb`` present, and vice versa.
"""

import logging
import os
from collections.abc import Mapping
from typing import Any

from langres.clients.settings import Settings

logger = logging.getLogger(__name__)

#: ``WANDB_MODE`` values where wandb never contacts the server, so no API key is
#: needed (offline runs sync later; disabled runs are a full no-op).
_KEYLESS_WANDB_MODES = frozenset({"offline", "disabled"})


def create_wandb_tracker(settings: Settings | None = None, job_type: str = "optimization") -> Any:
    """Initialize wandb tracking for experiment logging.

    This function initializes wandb for experiment tracking, using
    configuration from Settings or environment variables.

    Args:
        settings: Optional Settings object. If None, loads from environment.
        job_type: Type of job for wandb categorization (e.g., "optimization",
            "training", "evaluation"). Default: "optimization".

    Returns:
        wandb run object that can be used to log metrics and artifacts.

    Raises:
        ValueError: If WANDB_API_KEY is not set AND wandb is in an online mode.
            When ``WANDB_MODE`` is ``offline`` or ``disabled`` wandb needs no
            key, so the requirement is skipped (offline/CI use).

    Environment variables required:
        WANDB_API_KEY: Weights & Biases API key (required online; not needed
            when ``WANDB_MODE`` is ``offline``/``disabled``)
        WANDB_PROJECT: W&B project name (optional, defaults to "langres")
        WANDB_ENTITY: W&B entity/team name (optional)
        WANDB_MODE: ``online`` (default), ``offline``, or ``disabled``

    Example:
        # With explicit settings
        settings = Settings()
        run = create_wandb_tracker(settings, job_type="blocker_optimization")
        wandb.log({"metric": 0.85})
        wandb.finish()

        # Without settings (loads from env)
        run = create_wandb_tracker()
        wandb.log({"f1": 0.90})
        wandb.finish()

    Note:
        The wandb API key is read from WANDB_API_KEY environment variable.

    Note:
        To log metrics during optimization, use:
        - wandb.log({"metric_name": value})
        - wandb.log({"trial": trial_num, "f1": f1_score, "cost": cost_usd})
    """
    # Lazy import so this module (and therefore create_trackio_tracker) is
    # importable in a wandb-less install. A missing extra surfaces the same
    # branded fix as resolve_tracker's other backends.
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "the 'wandb' tracker requires the 'wandb' extra: "
            "pip install 'langres[wandb]' (or uv add 'langres[wandb]')"
        ) from exc

    if settings is None:
        settings = Settings()

    # Validate wandb API key is present -- but only for online runs. Offline /
    # disabled modes never contact the W&B server, so demanding a key there
    # would needlessly block offline/CI use of the tracker.
    offline = os.environ.get("WANDB_MODE", "").strip().lower() in _KEYLESS_WANDB_MODES
    if not offline and not settings.wandb_api_key:
        raise ValueError("WANDB_API_KEY environment variable is required")

    run = wandb.init(
        project=settings.wandb_project, entity=settings.wandb_entity, job_type=job_type
    )

    logger.info(
        "wandb tracker initialized (project: %s, entity: %s, job_type: %s)",
        settings.wandb_project,
        settings.wandb_entity,
        job_type,
    )

    return run


def create_trackio_tracker(
    settings: Settings | None = None,
    *,
    project: str | None = None,
    space_id: str | None = None,
    dataset_id: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> Any:
    """Build a (not-yet-started) :class:`TrackioTracker` from settings + overrides.

    Mirrors :func:`create_wandb_tracker`'s settings-driven construction, with
    two differences forced by trackio's shape: ``trackio.init`` is not called
    here (deferred to ``TrackioTracker.start_run``, matching
    :class:`~langres.core.trackers.mlflow_tracker.MlflowTracker`'s
    init-on-start-run pattern -- there is no live "run" object to configure
    before a run starts), so the missing-HF-token guard also fires there, not
    in this factory; and the import of :class:`TrackioTracker` (and therefore
    ``trackio``) is local to this function body, not this module's top level --
    this file already has an unconditional top-level ``import wandb`` (needed
    so ``wandb_tracker.py``'s missing-extra ``ImportError`` surfaces correctly),
    and importing ``trackio`` there too would make a ``langres[wandb]``-only
    install fail merely by loading this module. Calling this factory always
    requires the ``trackio`` extra; ``resolve_tracker("trackio")`` and
    constructing :class:`TrackioTracker` directly do not need this factory at all.

    Args:
        settings: Source of ``trackio_space_id``/``trackio_dataset_id``/
            ``hf_token`` fallbacks when the matching keyword is ``None``.
            ``None`` -> :class:`Settings` from the environment.
        project: The trackio project name. ``None`` -> resolved from the run's
            experiment name at ``start_run`` time.
        space_id: HF Space to sync to. ``None`` (default) -> local-first, no
            credentials/network. Overrides ``settings.trackio_space_id``.
        dataset_id: HF Dataset to additionally sync to (requires ``space_id``).
            Overrides ``settings.trackio_dataset_id``.
        config: Extra config merged with the run context at ``start_run``.

    Returns:
        An unstarted ``TrackioTracker`` -- call ``.start_run(...)`` to open it.
    """
    from langres.core.trackers.trackio_tracker import TrackioTracker

    if settings is None:
        settings = Settings()
    return TrackioTracker(
        settings,
        project=project,
        space_id=space_id if space_id is not None else settings.trackio_space_id,
        dataset_id=dataset_id if dataset_id is not None else settings.trackio_dataset_id,
        config=config,
    )
