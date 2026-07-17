"""Guard the committed M1 gold-set artifact (the Wave 5 paid deliverable).

Deterministic, no network: loads ``data/gold_sets/fodors_zagat/gold_set.json``
and asserts its structural invariants so an accidental truncation/corruption of
the paid artifact is caught in CI.
"""

from __future__ import annotations

from pathlib import Path

from langres.curation import GoldSet

ARTIFACT = (
    Path(__file__).resolve().parents[2] / "data" / "gold_sets" / "fodors_zagat" / "gold_set.json"
)


def test_gold_set_artifact_invariants() -> None:
    gold = GoldSet.load(ARTIFACT)

    # Round-trips through its own serializer.
    assert GoldSet.model_validate_json(gold.model_dump_json()).model_dump() == gold.model_dump()

    assert gold.schema_version == "1"
    # The full Fodors-Zagat cross-source band (hundreds-low-thousands of pairs).
    assert len(gold.pairs) >= 1000

    meta = gold.metadata
    assert meta["labeler"] == "TeacherLabeler"
    assert float(meta["total_cost_usd"]) > 0.0  # a real, paid run
    assert int(meta["matches"]) + int(meta["non_matches"]) == len(gold.pairs)

    for pair in gold.pairs:
        assert pair.source == "teacher"
        assert isinstance(pair.label, bool)
        assert pair.confidence is not None and 0.0 <= pair.confidence <= 1.0
        # label is the thresholded confidence (default 0.5) -> internally consistent.
        assert pair.label == (pair.confidence >= 0.5)
