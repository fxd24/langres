"""Back-compat shim: ``langres.core.trackers`` moved to ``langres.tracking.trackers``.

# TEMPORARY: deleted by the W2 sweep

Experiment tracking is observability, not ER modelling, so it now lives in
``langres.tracking`` beside ``core`` rather than inside it.

The three backend adapters stay **lazy** here exactly as they are in the real
module: they are resolved through ``__getattr__`` below, never imported at this
module's top level. A bare ``import langres`` must not pull mlflow/wandb/trackio
(nor trackio's transitive ``huggingface_hub``) into ``sys.modules`` --
``tests/test_import_budget.py`` is the gate that measures it, and re-exporting
the adapters eagerly from this shim would defeat it.
"""

from typing import TYPE_CHECKING, Any

from langres.tracking.trackers import (
    ExperimentTracker,
    MultiTracker,
    NoOpTracker,
    TrackerSpec,
    resolve_tracker,
)

if TYPE_CHECKING:
    from langres.tracking.trackers import MlflowTracker, TrackioTracker, WandbTracker

__all__ = [
    "ExperimentTracker",
    "MultiTracker",
    "NoOpTracker",
    "TrackerSpec",
    "resolve_tracker",
]

#: The lazily-resolved adapter names, mirroring the real module's ``__getattr__``.
_LAZY_ADAPTERS = frozenset({"MlflowTracker", "WandbTracker", "TrackioTracker"})


def __getattr__(name: str) -> Any:
    """Resolve the backend adapters through the new module, keeping them lazy."""
    if name in _LAZY_ADAPTERS:
        import langres.tracking.trackers as _trackers

        return getattr(_trackers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
