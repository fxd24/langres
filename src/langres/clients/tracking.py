"""wandb tracking client factory."""

import logging
import os
from typing import Any

import wandb

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
