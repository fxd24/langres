from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from langres.core.op import ExecutionCheckpoint
from langres.core.pairs import PairRow
from langres.experiments.cache import (
    ScoreCacheError,
    StageArtifactStore,
    ordered_input_fingerprint,
)


def _checkpoint() -> ExecutionCheckpoint:
    return ExecutionCheckpoint(
        prefix_plan_id="plan",
        cache_id="cache",
        boundary_index=2,
        boundary_stage_id="stage",
        input_fingerprint="input",
        records=({"id": "a"}, {"id": "b"}),
        rows=(
            PairRow(
                left_id="a",
                right_id="b",
                blocker_name="all",
                score=0.9,
                score_type="heuristic",
            ),
        ),
    )


def test_ordered_input_fingerprint_preserves_row_order_not_key_order() -> None:
    first = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]
    key_permuted = [{"name": "A", "id": "a"}, {"name": "B", "id": "b"}]

    assert ordered_input_fingerprint(first) == ordered_input_fingerprint(key_permuted)
    assert ordered_input_fingerprint(first) != ordered_input_fingerprint(list(reversed(first)))


def test_stage_store_round_trips_and_is_idempotently_immutable(tmp_path: Path) -> None:
    store = StageArtifactStore(tmp_path)
    checkpoint = _checkpoint()

    path = store.put(checkpoint)
    assert store.put(checkpoint) == path
    assert (
        store.load(
            "cache",
            prefix_plan_id="plan",
            boundary_index=2,
            input_fingerprint="input",
        )
        == checkpoint
    )

    changed = checkpoint.model_copy(update={"input_fingerprint": "different"})
    with pytest.raises(ScoreCacheError, match="different content"):
        store.put(changed)


def test_concurrent_identical_commit_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StageArtifactStore(tmp_path)
    entry = tmp_path / "cache"

    def competing_commit(source: str | Path, destination: str | Path) -> None:
        shutil.copytree(source, destination)
        raise FileExistsError("another process committed first")

    monkeypatch.setattr("langres.experiments.cache.os.replace", competing_commit)

    assert store.put(_checkpoint()) == entry
    assert entry.is_dir()
    assert not list(tmp_path.glob(".stage-*"))


def test_corrupt_and_identity_mismatched_entries_are_quarantined(tmp_path: Path) -> None:
    store = StageArtifactStore(tmp_path)
    store.put(_checkpoint())
    (tmp_path / "cache" / "checkpoint.json").write_text("{}", encoding="utf-8")

    assert (
        store.load(
            "cache",
            prefix_plan_id="plan",
            boundary_index=2,
            input_fingerprint="input",
        )
        is None
    )
    assert not (tmp_path / "cache").exists()
    assert list((tmp_path / "quarantine").iterdir())

    store.put(_checkpoint())
    assert (
        store.load(
            "cache",
            prefix_plan_id="other",
            boundary_index=2,
            input_fingerprint="input",
        )
        is None
    )
    assert len(list((tmp_path / "quarantine").iterdir())) == 2


def test_duplicate_pair_ids_fail_before_cache_commit(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    duplicate = checkpoint.model_copy(update={"rows": (checkpoint.rows[0],) * 2})

    with pytest.raises(ScoreCacheError, match="duplicate ordered pair ids"):
        StageArtifactStore(tmp_path).put(duplicate)
    assert not (tmp_path / "cache").exists()


def test_cache_id_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ScoreCacheError, match="unsafe path"):
        StageArtifactStore(tmp_path).load(
            "../escape",
            prefix_plan_id="plan",
            boundary_index=2,
            input_fingerprint="input",
        )
