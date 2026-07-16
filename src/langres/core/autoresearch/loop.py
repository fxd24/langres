"""The ``propose → run → evaluate → keep-if-better`` driver for autoresearch.

This is the integration seam of the autoresearch loop (epic #145): it walks a
stream of proposed configs, scores each one, and keeps the incumbent that the
:class:`~langres.core.autoresearch.objective.Objective` says is best — logging
*every* trial (accepted and rejected) through the existing run-tracking spine
(``core.runs``) so a run is durable, deduplicated, and lineage-linked.

**Injected-scorer testability seam.** :func:`run_loop` takes the scorer as a
plain ``Callable[[Mapping], Mapping[str, float]]`` (config → metrics), so its
keep/revert + logging logic is unit-testable with a canned dict scorer — no
embeddings, faiss, or benchmark load required. The concrete blocking scorer
(and its index reuse) lives one layer up in :mod:`langres.optimize`; this module
knows nothing about blockers.

**Import-light by design.** Only stdlib + :mod:`~langres.core.autoresearch.objective`
+ :mod:`~langres.core.runs` (+ its ``trackers``) — all already on the bare
``import langres`` path. It imports no factory / data / metrics module, so it
adds no weight and could sit on the public surface if ever needed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from langres.core.autoresearch.objective import Objective
from langres.core.runs import RunContext, capture_run, compute_recipe_id
from langres.core.trackers import TrackerSpec, resolve_tracker

logger = logging.getLogger(__name__)

#: A scorer maps one config dict to a metrics mapping the Objective can read.
Scorer = Callable[[Mapping[str, Any]], Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class Trial:
    """One config's outcome in a :func:`run_loop` pass — the audit trail unit.

    Attributes:
        config: The config dict that was scored (a copy, so mutating the input
            stream can't retroactively rewrite the trail).
        metrics: The scorer's metrics for this config, or ``None`` if the scorer
            raised (a failed trial).
        accepted: Whether this config displaced the incumbent (``objective.is_better``
            returned ``True``). Also written to the run record's logged metrics as
            ``accepted`` so a reader can reconstruct the incumbent timeline from
            the store alone.
        recipe_id: The content-addressed dedup key (``core.runs.compute_recipe_id``)
            for this config's run recipe.
        attempt_id: The run record PK for this trial (``recipe_id``-``started_at``);
            the *next* trial's ``parent_run_id`` when this one is the incumbent,
            giving the store its lineage chain.
        status: ``"completed"`` (scored) or ``"failed"`` (scorer raised).
    """

    config: dict[str, Any]
    metrics: dict[str, float] | None
    accepted: bool
    recipe_id: str
    attempt_id: str
    status: str


@dataclass(frozen=True, slots=True)
class LoopResult:
    """The best incumbent found plus the full trial trail.

    Attributes:
        best_config: The winning config dict, or ``None`` if no config was ever
            accepted (empty input, or every trial infeasible/failed).
        best_metrics: The winning config's metrics, or ``None`` (see above).
        trials: Every trial in evaluation order — accepted, rejected, and failed —
            for inspection and for reconstructing why the incumbent won.
    """

    best_config: dict[str, Any] | None
    best_metrics: dict[str, float] | None
    trials: tuple[Trial, ...]


def run_loop(
    configs: Iterable[Mapping[str, Any]],
    scorer: Scorer,
    objective: Objective,
    *,
    experiment: str,
    dataset_name: str,
    dataset_fingerprint: str | None = None,
    split_id: str | None = None,
    seeds: dict[str, int] | None = None,
    store: str | Any | None = None,
    tracker: TrackerSpec = None,
    dedup: bool = True,
) -> LoopResult:
    """Run the ``propose → run → evaluate → keep-if-better`` loop over ``configs``.

    For each config, in order:

    1. Build a :class:`~langres.core.runs.RunContext` (``resolver_config`` = the
       config, ``method`` = ``config["blocker"]``, ``blocking_k`` =
       ``config["k_neighbors"]``, ``parent_run_id`` = the *current incumbent's*
       ``attempt_id`` so the store records lineage) and compute its ``recipe_id``.
    2. **Dedup** (``dedup=True``): skip a config whose ``recipe_id`` was already
       seen this run — this collapses P-B's redundant degenerate configs (e.g.
       several ``all_pairs`` configs the caller normalized to one recipe).
    3. Score it *inside* :func:`~langres.core.runs.capture_run`, so timing and
       failures are captured. Compute ``better = objective.is_better(metrics,
       incumbent_metrics)`` and log **every** trial's metrics plus an ``accepted``
       flag (``1.0``/``0.0``), with the first goal's metric as the headline and
       ``str(objective)`` as the metric definition.
    4. If ``better``, the config becomes the incumbent (its ``attempt_id`` becomes
       the next trial's parent). Otherwise the incumbent is kept.

    A scorer that raises is logged as a ``failed`` run (``capture_run`` writes the
    ``failed`` terminal line and re-raises; this wraps the ``with`` per-config to
    swallow it) and the loop continues to the next config — one bad config never
    aborts the sweep. The result is deterministic given a fixed ``configs`` order.

    Args:
        configs: The proposed config dicts (e.g. ``SearchSpace.configs()``), in
            evaluation order. Callers that want degenerate configs deduplicated
            should normalize them to a canonical shape first (see
            :mod:`langres.optimize`).
        scorer: ``config -> metrics``. The one seam under test; the metrics it
            returns must contain every goal/constraint metric the ``objective``
            references (a missing one raises inside ``capture_run`` and the trial
            is recorded ``failed``).
        objective: The immutable keep-if-better decision.
        experiment: Experiment label for every run record (organizational, not
            part of the recipe hash beyond its own field).
        dataset_name: Dataset label recorded on every run (part of the recipe).
        dataset_fingerprint: Optional content hash of the loaded data (part of the
            recipe), so a data change mints fresh ids.
        split_id: Optional split label recorded on every run (part of the recipe).
        seeds: Optional seed map recorded on every run (part of the recipe).
        store: Where to persist run records — a path / ``RunStore`` / ``None``.
            ``None`` writes **nothing** (the offline path); the loop still returns
            the same ``LoopResult``.
        tracker: Experiment tracker spec forwarded to ``capture_run`` --
            a backend name (``"trackio"``/``"mlflow"``/``"wandb"``), an
            ``ExperimentTracker`` instance, a sequence of either (fan-out via
            :class:`~langres.core.trackers.MultiTracker`), or ``None``
            (default; resolves to a fresh
            :class:`~langres.core.trackers.NoOpTracker`). Resolved once via
            :func:`~langres.core.trackers.resolve_tracker`.
        dedup: When ``True`` (default), skip a config whose ``recipe_id`` was
            already scored this run.

    Returns:
        A :class:`LoopResult` with the best incumbent and the full trial trail.
    """
    tracker = resolve_tracker(tracker)
    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_attempt_id: str | None = None
    trials: list[Trial] = []
    seen: set[str] = set()
    headline_metric = objective.goals[0].metric
    objective_repr = str(objective)

    for raw_config in configs:
        config = dict(raw_config)
        context = RunContext(
            experiment=experiment,
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            split_id=split_id,
            seeds=dict(seeds) if seeds else {},
            resolver_config=config,
            method=config.get("blocker"),
            blocking_k=config.get("k_neighbors"),
            parent_run_id=best_attempt_id,
        )
        recipe_id = compute_recipe_id(context)
        if dedup and recipe_id in seen:
            logger.debug("autoresearch loop: skipping duplicate recipe %s", recipe_id)
            continue
        seen.add(recipe_id)

        attempt_id = ""
        metrics: dict[str, float] | None = None
        accepted = False
        try:
            with capture_run(context, store=store, tracker=tracker) as run:
                attempt_id = run.attempt_id
                scored = dict(scorer(config))
                accepted = objective.is_better(scored, best_metrics)
                run.log_metrics(
                    {**scored, "accepted": float(accepted)},
                    headline_metric=scored[headline_metric],
                    metric_definition=objective_repr,
                )
                metrics = scored
        except Exception:
            # capture_run already wrote a status="failed" terminal record and
            # re-raised; log the config as a failed trial and keep going so one
            # bad config never aborts the whole sweep.
            logger.warning("autoresearch loop: scorer failed for config %s", config, exc_info=True)
            trials.append(Trial(config, None, False, recipe_id, attempt_id, "failed"))
            continue

        trials.append(Trial(config, metrics, accepted, recipe_id, attempt_id, "completed"))
        if accepted:
            best_config = config
            best_metrics = metrics
            best_attempt_id = attempt_id

    return LoopResult(best_config, best_metrics, tuple(trials))
