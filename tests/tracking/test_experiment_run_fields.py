from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from langres.experiments import compute_recipe_identity
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

    experiment_recipe_id = compute_recipe_identity(context).recipe_id
    with capture_run(
        context,
        store=path,
        recipe_id=experiment_recipe_id,
        evaluation_id="evaluation",
        cache_id="cache",
        protocol={"version": 1},
    ) as handle:
        handle.record_measurements([{"stage_id": "retrieve", "wall_seconds": 0.1}])

    [record] = RunStore(path).read()
    assert record.recipe_id == experiment_recipe_id
    assert record.evaluation_id == "evaluation"
    assert record.cache_id == "cache"
    assert record.protocol == {"version": 1}
    assert record.measurements == ({"stage_id": "retrieve", "wall_seconds": 0.1},)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["evaluation_id"] == "evaluation"


def test_capture_run_reuses_deep_immutable_protocol_and_measurement_snapshots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runs.jsonl"
    context = RunContext(experiment="research", dataset_name="dataset")
    protocol = {"version": 1, "nested": {"thresholds": [0.4, 0.6]}}
    measurements = [{"stage_id": "retrieve", "runtime": {"device": "cpu"}}]

    with capture_run(context, store=path, protocol=protocol) as handle:
        protocol["nested"]["thresholds"].append(0.9)
        handle.record_measurements(measurements)
        measurements[0]["runtime"]["device"] = "gpu"

    [record] = RunStore(path).read()
    assert record.protocol == {"version": 1, "nested": {"thresholds": (0.4, 0.6)}}
    assert record.measurements == ({"stage_id": "retrieve", "runtime": {"device": "cpu"}},)

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["protocol"] == lines[1]["protocol"]
    with pytest.raises(TypeError, match="immutable"):
        record.protocol["nested"]["thresholds"] = (1.0,)  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        record.measurements[0]["runtime"]["device"] = "gpu"  # type: ignore[index]


def test_capture_run_reuses_the_same_validated_protocol_snapshot_object(
    tmp_path: Path,
) -> None:
    records: list[RunRecord] = []

    class RecordingStore(RunStore):
        def append(self, record: RunRecord) -> None:
            records.append(record)

    store = RecordingStore(tmp_path / "unused.jsonl")
    with capture_run(
        RunContext(experiment="research", dataset_name="dataset"),
        store=store,
        protocol={"version": 1, "nested": {"thresholds": [0.5]}},
    ):
        pass

    assert len(records) == 2
    assert records[0].protocol is records[1].protocol


@pytest.mark.parametrize("field", ["protocol", "measurements"])
def test_run_record_snapshots_reject_non_finite_numbers(field: str) -> None:
    values: dict[str, object] = {
        "attempt_id": "attempt",
        "recipe_id": "recipe",
        "context": RunContext(experiment="research", dataset_name="dataset"),
        "started_at": "2026-07-18T12:00:00+00:00",
        "status": "completed",
        field: (
            {"nested": {"value": float("nan")}}
            if field == "protocol"
            else [{"nested": {"value": float("nan")}}]
        ),
    }
    with pytest.raises(ValidationError, match="finite"):
        RunRecord(**values)
