"""Back-compat shim: moved to ``langres.tracking.trackers.wandb_tracker``.

# TEMPORARY: deleted by the W2 sweep

Experiment tracking is observability, not ER modelling, so it now lives in
``langres.tracking`` beside ``core`` rather than inside it. ``wandb`` is still
imported lazily by the real adapter, never by this shim.
"""

from langres.tracking.trackers.wandb_tracker import WandbTracker

__all__ = ["WandbTracker"]
