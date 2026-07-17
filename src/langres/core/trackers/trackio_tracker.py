"""Back-compat shim: moved to ``langres.tracking.trackers.trackio_tracker``.

# TEMPORARY: deleted by the W2 sweep

Experiment tracking is observability, not ER modelling, so it now lives in
``langres.tracking`` beside ``core`` rather than inside it. ``trackio`` is still
imported lazily by the real adapter, never by this shim.
"""

from langres.tracking.trackers.trackio_tracker import TrackioTracker

__all__ = ["TrackioTracker"]
