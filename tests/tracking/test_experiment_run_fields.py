from __future__ import annotations

import json
from pathlib import Path

from langres.tracking.runs import RunContext, RunRecord, RunStore, capture_run


def test_legacy_run_record_without_experiment_fields_still_loads() -> None:
    payload = {
        "attempt_id": "recipe-time",
        "recipe_id": "recipe",
        "context": {"experiment": "legacy", "dataset_name": "dataset"},
        "started_at": "2026-07-18T12:00:00+00:00",
        "status": "completed",
    }

    record = RunRecord.model_validate(payload)

    assert record.evaluation_id is None
    assert record.cache_id is None
    assert record.protocol is None
    assert record.measurements is None


def test_capture_run_persists_experiment_identity_and_measurements(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    context = RunContext(experiment="research", dataset_name="dataset", budget_usd=None)

    with capture_run(
        context,
        store=path,
        evaluation_id="evaluation",
        cache_id="cache",
        protocol={"version": 1},
    ) as handle:
        handle.record_measurements([{"stage_id": "retrieve", "wall_seconds": 0.1}])

    [record] = RunStore(path).read()
    assert record.evaluation_id == "evaluation"
    assert record.cache_id == "cache"
    assert record.protocol == {"version": 1}
    assert record.measurements == [{"stage_id": "retrieve", "wall_seconds": 0.1}]

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["evaluation_id"] == "evaluation"
