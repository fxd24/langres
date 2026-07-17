"""Persist & compare experiment runs -- the tracking layer, end to end.

langres benchmark results are otherwise *ephemeral*: a run scores well, prints,
and is gone -- no run id, no config/data snapshot, no way to compare against a
run from last week. ``capture_run`` fixes that: it gives every run a
content-addressed identity (``recipe_id``), snapshots the config + git sha +
seeds + dataset fingerprint into one JSONL line, and reads back as a list of
frozen ``RunRecord`` models you can diff across sessions.

This demo is offline and free (like ``judgement_log_demo.py``): it runs the
zero-spend ``"string"`` judge -- no API key, no extras -- at two thresholds,
captures each run under ``capture_run(store=)``, then reads the store back to

1. diff the two runs' metrics (the "compare across sessions" move); and
2. show the agent two-liner: "already ran this recipe? / how much have I spent?"

The demo uses ``capture_run`` directly. A later stream wraps the benchmark
harness so ``run_methods(store=)`` captures runs for you; the primitive shown
here is what that wrap is built on.

Run it:
    uv run python examples/research/experiment_tracking_demo.py
    # macOS + numpy may need: KMP_DUPLICATE_LIB_OK=TRUE uv run python ...
"""

from pathlib import Path

from langres.architectures import FuzzyString
from langres.core import RunContext, RunStore, capture_run, compute_recipe_id
from langres.tracking.runs import dataset_fingerprint, git_sha

STORE_PATH = "tmp/tracking_demo/runs.jsonl"
EXPERIMENT = "string-judge-threshold-sweep"
BUDGET_USD = 5.0

# Toy company records with three true duplicate groups (a*/b*/c*) plus one
# string-similar distractor (x1) that a lenient threshold wrongly merges.
records = [
    {"id": "a1", "name": "Acme Corporation"},
    {"id": "a2", "name": "Acme Corp"},
    {"id": "b1", "name": "Globex Incorporated"},
    {"id": "b2", "name": "Globex Inc"},
    {"id": "c1", "name": "Initech"},
    {"id": "c2", "name": "Initech LLC"},
    {"id": "x1", "name": "Acme Consulting"},
]
gold_pairs = {frozenset({"a1", "a2"}), frozenset({"b1", "b2"}), frozenset({"c1", "c2"})}

# One dataset fingerprint over the already-loaded corpus+gold: mutate any record
# and this (and therefore recipe_id) changes -- data identity, not a filename.
DATASET_FP = dataset_fingerprint(records, gold_pairs)


def pair_prf(clusters: list[set[str]], gold: set[frozenset[str]]) -> dict[str, float]:
    """Pairwise precision/recall/F1 of a clustering against gold match pairs."""
    predicted: set[frozenset[str]] = set()
    for cluster in clusters:
        members = sorted(cluster)
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                predicted.add(frozenset({left, right}))
    true_positives = len(predicted & gold)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def sweep_context() -> RunContext:
    """The parent sweep's recipe: an umbrella over the per-threshold children.

    It has no single config, so ``resolver_config`` stays ``None`` -- the
    best-effort snapshot path (a bespoke run with no registered config captures
    cleanly, it does not crash).
    """
    sha, dirty = git_sha()
    return RunContext(
        experiment=EXPERIMENT,
        git_sha=sha,
        git_dirty=dirty,
        method="threshold_sweep",
        dataset_name="toy-companies",
        dataset_fingerprint=DATASET_FP,
        seeds={"split": 13},
    )


def config_context(threshold: float, *, parent_run_id: str | None = None) -> RunContext:
    """The recipe for one threshold config: config + data + seeds + provenance.

    ``resolver_config`` is a minimal hand-built snapshot here; the deferred
    harness wrap fills it from ``Resolver.config_dict()``. ``parent_run_id`` is
    identity-only (not hashed), so the same config under a different sweep keeps
    the same ``recipe_id`` -- idempotency is parent-independent.
    """
    sha, dirty = git_sha()
    return RunContext(
        experiment=EXPERIMENT,
        parent_run_id=parent_run_id,
        git_sha=sha,
        git_dirty=dirty,
        resolver_config={"judge": "string", "threshold": threshold},
        method="dedupe_string",
        dataset_name="toy-companies",
        dataset_fingerprint=DATASET_FP,
        seeds={"split": 13},
    )


def main() -> None:
    store = RunStore(STORE_PATH)
    # Fresh file so this demo's output is deterministic. In a real workflow you
    # would NOT reset -- successive sessions append and the idempotency guard
    # (Part 3) skips recipes already completed.
    Path(store.path).unlink(missing_ok=True)

    # --- Part 1: capture a two-config threshold sweep -----------------------
    # Lineage: the sweep is the PARENT run; each threshold config is a CHILD
    # carrying parent_run_id = <sweep attempt_id>. The same shape models a
    # DSPy-compile run parenting the eval runs that reuse its compiled program.
    print(f"Capturing runs to {store.path}")
    child_f1s: list[float] = []
    with capture_run(sweep_context(), store=store) as sweep:
        print(f"  sweep run {sweep.recipe_id} ({sweep.attempt_id})")
        for threshold in (0.60, 0.70):
            with capture_run(
                config_context(threshold, parent_run_id=sweep.attempt_id),
                store=store,
            ) as run:
                clusters = list(FuzzyString(threshold=threshold).dedupe(records))
                metrics = pair_prf(clusters, gold_pairs)
                run.log_metrics(
                    metrics,
                    metric_definition="pair_f1",
                    headline_metric=metrics["f1"],
                )
                run.record_cost(0.0)  # zero-spend string judge; a paid judge
                #                       records SpendMonitor.spent here instead.
                child_f1s.append(metrics["f1"])
                print(
                    f"    threshold={threshold:.2f}  P={metrics['precision']:.2f} "
                    f"R={metrics['recall']:.2f} F1={metrics['f1']:.2f}"
                )
        sweep.log_metrics({"best_f1": max(child_f1s)}, headline_metric=max(child_f1s))

    # --- Part 2: read the runs back and compare (across sessions) -----------
    runs = RunStore(STORE_PATH).read()  # last-wins-by-attempt_id; frozen models
    children = [
        r
        for r in runs
        if r.context.experiment == EXPERIMENT and r.context.parent_run_id is not None
    ]
    children.sort(key=lambda r: r.context.resolver_config["threshold"])  # type: ignore[index]
    low, high = children
    print(f"\n{len(runs)} run(s) persisted; comparing the {len(children)} sweep children:")
    for record in (low, high):
        config = record.context.resolver_config or {}
        metrics = record.metrics or {}
        print(
            f"  threshold={config['threshold']:.2f}  recipe={record.recipe_id}  "
            f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}"
        )
    delta = (high.headline_metric or 0.0) - (low.headline_metric or 0.0)
    sha = low.context.git_sha
    print(
        f"  -> raising threshold changed pair-F1 by {delta:+.2f}  (git_sha {sha[:8] if sha else 'n/a'})"
    )

    # --- Part 3: the agent two-liner: idempotency + budget ------------------
    # An agent re-running a sweep drops these two lines at the top to skip a
    # recipe it already paid for and to check remaining budget.
    completed = [r for r in store.read() if r.status == "completed"]
    already_ran = compute_recipe_id(config_context(0.60)) in {r.recipe_id for r in completed}
    spent = sum(r.spend_usd for r in completed)
    print(
        f"\nagent check: recipe 'threshold=0.60' already run? {already_ran}  |  "
        f"spent ${spent:.2f} of ${BUDGET_USD:.2f} budget"
    )


if __name__ == "__main__":
    main()
