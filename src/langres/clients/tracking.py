"""Back-compat shim: moved to ``langres.tracking.factories``.

# TEMPORARY: deleted by the W2 sweep

The tracker factories are observability, not client plumbing, so they moved into
``langres.tracking`` beside the adapters they build. ``langres.clients`` still
resolves ``create_wandb_tracker``/``create_trackio_tracker`` lazily, so that
public surface is unchanged.
"""

from langres.tracking.factories import create_trackio_tracker, create_wandb_tracker

__all__ = ["create_trackio_tracker", "create_wandb_tracker"]
