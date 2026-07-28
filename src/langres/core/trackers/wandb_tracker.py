"""Back-compat shim: moved to ``langres.tracking.trackers.wandb_tracker``.

# TEMPORARY: deleted by the W2 sweep

Experiment tracking is observability, not ER modelling, so it now lives in
``langres.tracking`` beside ``core`` rather than inside it. ``wandb`` is still
imported lazily by the real adapter, never by this shim.
"""

# `# pragma: no cover`, as on every other W2 shim (see core/harvest.py): a
# re-export owns no contract. The real adapter is itself inside the
# `langres.core` contract coverage gate, so measuring this redirect would book a
# miss against code already covered at its real home. Goes away with this file
# in the W2 sweep.
from langres.tracking.trackers.wandb_tracker import WandbTracker  # pragma: no cover

__all__ = ["WandbTracker"]  # pragma: no cover
