"""Back-compat shim: ``langres.core.runs`` moved to ``langres.tracking.runs``.

# TEMPORARY: deleted by the W2 sweep

Run identity/persistence is observability, not ER modelling, so it now lives in
``langres.tracking`` beside ``core`` rather than inside it. Import from
``langres.tracking.runs`` (or the unchanged ``langres.core`` facade, which still
re-exports these names).
"""

# `# pragma: no cover`, as on every other W2 shim (see core/harvest.py): a
# re-export owns no contract. `langres.tracking.runs` is itself inside the
# `langres.core` contract coverage gate, so measuring this redirect would book a
# miss against code already covered at its real home. Goes away with this file
# in the W2 sweep.
from langres.tracking.runs import (  # pragma: no cover
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

__all__ = [  # pragma: no cover
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
