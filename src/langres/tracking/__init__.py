"""langres.tracking: run identity, judgement logging, and experiment trackers.

Observability, **not** ER modelling -- so this package sits *beside* ``core``,
never inside it. ``core`` answers "what is a match"; ``tracking`` answers "what
did that run do, and what did it cost". The two are different reasons to change.

This ``__init__`` deliberately **exports nothing** and imports nothing at module
scope. That is not laziness, it is the invariant: ``langres.core`` is eagerly
imported by a bare ``import langres``, and it re-exports the tracking primitives
(``RunStore``, ``ExperimentTracker``, ``JudgementLog``, ...) through
``core/_exports/_tracking.py``. If this file eagerly pulled in
``tracking.trackers``, every backend adapter it can reach would be one edge
closer to the eager graph. Import the module that owns the symbol:

    from langres.tracking.runs import RunStore, capture_run
    from langres.tracking.trackers import resolve_tracker
    from langres.tracking.judgement_log import JudgementLog, LoggingMatcher

The tracker *backends* (mlflow / wandb / trackio) stay lazy behind
``tracking.trackers.__getattr__`` and its ``_ADAPTERS`` table -- a bare
``import langres`` must never pull mlflow/wandb/trackio/huggingface_hub into
``sys.modules``. ``tests/test_import_budget.py`` is the gate that measures it.
"""
