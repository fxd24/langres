"""Tests for the gold-set data contract (``langres.bootstrap``)."""

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from langres.bootstrap import GoldPair, GoldSet


def test_gold_pair_minimal_defaults() -> None:
    pair = GoldPair(left_id="f1", right_id="z2", label=True, source="teacher")
    assert pair.label is True
    assert pair.source == "teacher"
    assert pair.confidence is None
    assert pair.reasoning is None
    assert pair.provenance == {}


def test_gold_pair_full_fields() -> None:
    pair = GoldPair(
        left_id="f1",
        right_id="z2",
        label=False,
        source="human",
        confidence=0.82,
        reasoning="different cities",
        provenance={"reviewer": "alice"},
    )
    assert pair.confidence == 0.82
    assert pair.reasoning == "different cities"
    assert pair.provenance == {"reviewer": "alice"}


def test_gold_pair_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        GoldPair(left_id="f1", right_id="z2", label=True, source="oracle")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_gold_pair_rejects_out_of_range_confidence(bad: float) -> None:
    with pytest.raises(ValidationError):
        GoldPair(left_id="f1", right_id="z2", label=True, source="teacher", confidence=bad)


@pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
def test_gold_pair_accepts_in_range_confidence(ok: float) -> None:
    pair = GoldPair(left_id="f1", right_id="z2", label=True, source="teacher", confidence=ok)
    assert pair.confidence == ok


def test_gold_pair_provenance_is_independent_per_instance() -> None:
    a = GoldPair(left_id="f1", right_id="z2", label=True, source="teacher")
    b = GoldPair(left_id="f3", right_id="z4", label=False, source="teacher")
    a.provenance["x"] = 1
    assert b.provenance == {}


def test_gold_set_defaults() -> None:
    gs = GoldSet(pairs=[])
    assert gs.schema_version == "1"
    assert gs.pairs == []
    assert gs.metadata == {}


def test_gold_set_save_load_roundtrip(tmp_path: Path) -> None:
    gs = GoldSet(
        pairs=[
            GoldPair(left_id="f1", right_id="z2", label=True, source="ground_truth"),
            GoldPair(
                left_id="f3",
                right_id="z4",
                label=False,
                source="teacher",
                confidence=0.4,
                reasoning="name mismatch",
                provenance={"model": "x"},
            ),
        ],
        metadata={"dataset": "fodors_zagat", "total_cost_usd": 0.0, "n_pairs": 2},
    )
    path = tmp_path / "gold.json"
    gs.save(path)

    loaded = GoldSet.load(path)
    assert loaded == gs
    assert loaded.pairs[1].confidence == 0.4
    assert loaded.metadata["dataset"] == "fodors_zagat"


def test_gold_set_save_accepts_str_path(tmp_path: Path) -> None:
    gs = GoldSet(pairs=[GoldPair(left_id="f1", right_id="z2", label=True, source="human")])
    path = str(tmp_path / "gold_str.json")
    gs.save(path)
    assert GoldSet.load(path).pairs[0].left_id == "f1"


def test_gold_set_json_is_indented(tmp_path: Path) -> None:
    gs = GoldSet(pairs=[GoldPair(left_id="f1", right_id="z2", label=True, source="human")])
    path = tmp_path / "gold.json"
    gs.save(path)
    assert "\n  " in path.read_text()


def test_gold_set_fresh_process_reload(tmp_path: Path) -> None:
    """A gold set written here must reload in a fresh interpreter (DoD)."""
    gs = GoldSet(
        pairs=[GoldPair(left_id="f1", right_id="z2", label=True, source="ground_truth")],
        metadata={"dataset": "fodors_zagat"},
    )
    path = tmp_path / "gold.json"
    gs.save(path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from langres.bootstrap import GoldSet; "
                f"gs = GoldSet.load({str(path)!r}); "
                "assert gs.pairs; "
                "assert gs.pairs[0].source == 'ground_truth'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
