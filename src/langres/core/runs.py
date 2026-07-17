"""Back-compat shim: ``langres.core.runs`` moved to ``langres.tracking.runs``.

# TEMPORARY: deleted by the W2 sweep

Run identity/persistence is observability, not ER modelling, so it now lives in
``langres.tracking`` beside ``core`` rather than inside it. Import from
``langres.tracking.runs`` (or the unchanged ``langres.core`` facade, which still
re-exports these names).
"""

from langres.tracking.runs import (
    RunContext,
    RunRecord,
    RunStore,
    RunStoreError,
    capture_run,
    compute_recipe_id,
    current_run,
    dataset_fingerprint,
    git_sha,
    resolve_store,
)

__all__ = [
    "RunContext",
    "RunRecord",
    "RunStore",
    "RunStoreError",
    "capture_run",
    "compute_recipe_id",
    "current_run",
    "dataset_fingerprint",
    "git_sha",
    "resolve_store",
]
