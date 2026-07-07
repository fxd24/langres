"""Regenerate the committed closed-loop fixtures from the packaged Fodors-Zagat data.

This script produces the two JSONL/JSON fixtures the closed-loop demo
(``examples/flywheel_closed_loop.py``) reads:

* ``records.json``    -- ~80 restaurant records (a deterministic subset of the
  packaged Fodors-Zagat CSVs), each ``{"id", "name", "addr", "city", "phone"}``
  with a stable, source-prefixed id (``f``\\ =Fodor's, ``z``\\ =Zagat). Stable
  ids are load-bearing: the flywheel joins a judgement log back to records by
  id, so positional ids (what a schema-less ``dedupe()`` would assign) could not
  survive a fresh run.
* ``gold_pairs.json`` -- the candidate pairs the loop scores, each
  ``{"left_id", "right_id", "label"}``. ``label`` is the TRUE match label from
  the dataset's ``perfectMapping`` (ground truth), NOT the noisy LLM-teacher
  label carried in ``data/gold_sets/`` -- so the demo's reported P/R/F1 are
  honest. A pair with ``label: true`` is a real duplicate; ``label: false`` is a
  hard non-match the blocking step surfaced.

Everything here is $0 and deterministic: the only "scorer" is rapidfuzz (a
langres core dependency) used to mine hard negatives, the truth is the committed
``perfectMapping``, and the record subset + negative selection are seeded. The
fixtures are committed so the demo and its exit-criteria test read frozen data.
Re-run only to change the scenario::

    uv run python examples/data/flywheel_loop/generate_fixtures.py

Both label classes are guaranteed in the harvested training set
------------------------------------------------------------------
``RandomForestJudge.fit`` (the demo's student) raises an opaque scikit-learn error if it
sees a single label class. The demo trains the student on the harvested labels
of a *seeded half* of ``gold_pairs``; this generator therefore asserts that that
same seeded half carries both a ``true`` and a ``false`` gold pair (and the
whole set does too). Because the demo's simulated teacher is discriminative
(name/addr/phone rapidfuzz separates Fodors-Zagat matches at ~0.99 F1) its
verdicts already span both classes, and the human corrections only add gold
labels on top -- so the guarantee holds by construction, and this assertion is
the tripwire if a future edit breaks it.
"""

from __future__ import annotations

import csv
import json
import random
from importlib import resources
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

#: Scenario parameters (chosen so the closed loop is decisive and non-flaky).
#: ``_SEED`` seeds both the negative mining tie-breaks here and the demo's
#: train/held-out split, so this generator can assert the split is well-formed.
_SEED = 7
#: Number of true ``perfectMapping`` match pairs to seed the record subset from
#: (each contributes one Fodor's + one Zagat record -> ~2x records).
_N_TRUE_PAIRS = 40
#: Hard negatives mined per Fodor's record: its top-N most name/addr/phone-similar
#: Zagat records that are NOT its true match (the pairs a blocker would surface).
_NEG_PER_RECORD = 2
#: Feature weights for mining hard negatives (name-led, mirrors the demo teacher).
_SIGNAL_WEIGHTS = {"name": 0.45, "addr": 0.30, "phone": 0.20, "city": 0.05}

_DATASET_PACKAGE = "langres.data.datasets.fodors_zagat"
_RECORD_FIELDS = ("name", "addr", "city", "phone")
_HERE = Path(__file__).resolve().parent


def _unquote(value: str) -> str:
    """Strip the dataset's wrapping single quotes and unescape inner quotes."""
    value = value.strip()
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value.replace("\\'", "'")


def _load_records() -> dict[str, dict[str, str]]:
    """Load both Fodors-Zagat CSVs into ``{prefixed_id: {field: value}}``."""
    records: dict[str, dict[str, str]] = {}
    for filename, prefix in (("fodors.csv", "f"), ("zagats.csv", "z")):
        text = resources.files(_DATASET_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
        for row in csv.DictReader(text.splitlines()):
            records[f"{prefix}{_unquote(row['id'])}"] = {
                field: _unquote(row.get(field, "")) for field in _RECORD_FIELDS
            }
    return records


def _load_true_pairs() -> list[frozenset[str]]:
    """The perfectMapping as order-independent true-match pairs (deterministic order)."""
    text = (
        resources.files(_DATASET_PACKAGE)
        .joinpath("fodors-zagats_perfectMapping.csv")
        .read_text(encoding="utf-8")
    )
    return sorted(
        (
            frozenset({f"f{_unquote(row['fodors_id'])}", f"z{_unquote(row['zagats_id'])}"})
            for row in csv.DictReader(text.splitlines())
        ),
        key=sorted,
    )


def _signal(left: dict[str, str], right: dict[str, str]) -> float:
    """Name-led rapidfuzz similarity used only to mine plausible hard negatives."""
    total = 0.0
    for field, weight in _SIGNAL_WEIGHTS.items():
        total += weight * (fuzz.token_sort_ratio(left[field], right[field]) / 100.0)
    return total / sum(_SIGNAL_WEIGHTS.values())


def _ordered(pair: frozenset[str]) -> tuple[str, str]:
    """A frozenset pair as a deterministic ``(fodors_id, zagats_id)`` tuple."""
    a, b = sorted(pair)
    return (a, b)


def main() -> None:
    """Regenerate and overwrite the two committed fixtures."""
    records = _load_records()
    true_pairs = _load_true_pairs()

    # --- Select the record subset from the first N true match pairs. ---
    selected_true = true_pairs[:_N_TRUE_PAIRS]
    record_ids: set[str] = set()
    for pair in selected_true:
        record_ids.update(pair)
    fodors_ids = sorted(rid for rid in record_ids if rid.startswith("f"))
    zagat_ids = sorted(rid for rid in record_ids if rid.startswith("z"))
    true_set = set(true_pairs)

    # --- Candidate pairs: the true matches + mined hard negatives. ---
    candidates: set[tuple[str, str]] = {_ordered(pair) for pair in selected_true}
    for fodors_id in fodors_ids:
        scored = sorted(
            (
                (_signal(records[fodors_id], records[zagat_id]), zagat_id)
                for zagat_id in zagat_ids
                if frozenset({fodors_id, zagat_id}) not in true_set
            ),
            reverse=True,
        )
        for _score, zagat_id in scored[:_NEG_PER_RECORD]:
            candidates.add((fodors_id, zagat_id))

    candidate_pairs = [
        {"left_id": left, "right_id": right, "label": frozenset({left, right}) in true_set}
        for left, right in sorted(candidates)
    ]

    _assert_both_classes_in_seeded_split(candidate_pairs)

    # --- records.json ---
    records_payload = [
        {"id": rid, **records[rid]} for rid in sorted(record_ids, key=lambda r: (r[0], int(r[1:])))
    ]
    (_HERE / "records.json").write_text(json.dumps(records_payload, indent=2) + "\n", "utf-8")

    # --- gold_pairs.json ---
    gold_payload: dict[str, Any] = {
        "schema_version": 1,
        "note": (
            "Candidate pairs for examples/flywheel_closed_loop.py. 'label' is the "
            "TRUE Fodors-Zagat perfectMapping label (ground truth), not a judge "
            "verdict. Regenerate with generate_fixtures.py."
        ),
        "candidate_pairs": candidate_pairs,
    }
    (_HERE / "gold_pairs.json").write_text(json.dumps(gold_payload, indent=2) + "\n", "utf-8")

    n_true = sum(1 for pair in candidate_pairs if pair["label"])
    print(
        f"Wrote fixtures: {len(records_payload)} records, {len(candidate_pairs)} candidate "
        f"pairs ({n_true} true / {len(candidate_pairs) - n_true} false), seed={_SEED}."
    )


def _assert_both_classes_in_seeded_split(candidate_pairs: list[dict[str, Any]]) -> None:
    """Guard the RandomForestJudge single-class trap: the demo's train half must span both classes.

    Replays the demo's deterministic 50/50 train/held-out split (same ``_SEED``)
    and asserts the train half carries at least one ``true`` and one ``false``
    gold pair -- the worst-case guarantee that the harvested training labels can
    never be single-class (see the module docstring).
    """
    rng = random.Random(_SEED)
    order = list(range(len(candidate_pairs)))
    rng.shuffle(order)
    train = [candidate_pairs[i] for i in order[::2]]
    train_labels = {pair["label"] for pair in train}
    if train_labels != {True, False}:
        raise AssertionError(
            f"train split does not span both gold classes (labels={train_labels}); "
            "RandomForestJudge.fit would fail on single-class labels -- adjust the scenario "
            "parameters in generate_fixtures.py."
        )


if __name__ == "__main__":
    main()
