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

# NOT `# pragma: no cover`, unlike the sibling W2 shims (see core/harvest.py).
# Those are pure `from X import Y` redirects that own no contract and cannot
# fail in isolation. This one is different in kind: `__getattr__` below carries
# real conditional resolution -- a lazy-adapter branch and an unknown-name
# `AttributeError` branch -- and behavior excluded from the gate is behavior a
# regression can break silently. It is covered by
# tests/core/test_trackers_backcompat_shim.py instead, which also pins
# `_LAZY_ADAPTERS` against the real module's adapter table so the two cannot
# drift apart. Goes away with this file in the W2 sweep.
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
