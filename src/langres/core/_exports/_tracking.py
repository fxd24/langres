"""Experiment tracking (S1): run identity + persistence + pluggable trackers.

See ``langres.core._exports`` for the fragment contract.
"""

from typing import TYPE_CHECKING

from langres.core.runs import (
    RunContext,
    RunRecord,
    RunStore,
    capture_run,
    compute_recipe_id,
    resolve_store,
)
from langres.core.trackers import (
    ExperimentTracker,
    MultiTracker,
    NoOpTracker,
    resolve_tracker,
)

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy names visible to `mypy --strict`
    # without pulling mlflow/wandb into a bare `import langres`.
    from langres.core.trackers import MlflowTracker, WandbTracker

__all__ = [
    "capture_run",
    "compute_recipe_id",
    "ExperimentTracker",
    "MultiTracker",
    "NoOpTracker",
    "resolve_store",
    "resolve_tracker",
    "RunContext",
    "RunRecord",
    "RunStore",
]

LAZY_SUBMODULES: tuple[str, ...] = ()

#: The backend tracker adapters (S3/S4): the ``trackers`` package's own
#: ``__getattr__`` pulls the concrete adapter -- and its mlflow/wandb
#: dependency -- only on access.
LAZY_SYMBOLS: dict[str, str] = {
    "MlflowTracker": "langres.core.trackers",
    "WandbTracker": "langres.core.trackers",
}

EXTRA_BY_SYMBOL: dict[str, str] = {
    "MlflowTracker": "mlflow",
    "WandbTracker": "wandb",
}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SUBMODULES, *LAZY_SYMBOLS)
