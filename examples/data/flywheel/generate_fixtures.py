"""Regenerate the committed flywheel fixtures from Fodors-Zagat + a free scorer.

This script produces the three JSONL fixtures the flywheel demo
(``examples/flywheel_threshold_harvest.py``) reads:

* ``judgements.jsonl``   -- a ``JudgementLog``-format log of a cheap string
  judge scoring the *calibration* split of the Fodors-Zagat candidate pairs,
  with a deliberately BAD verdict threshold (the "magic constant" problem
  ``derive_threshold`` exists to kill).
* ``corrections.jsonl``  -- ``Correction``-format human overrides for the pairs
  a review queue would surface first (verdicts nearest the current threshold
  that disagree with true gold).
* ``heldout_gold.jsonl`` -- the *held-out* split, each pair scored and carrying
  its TRUE gold label -- the set the derived thresholds are judged on but never
  fit on.

Everything here is $0 and deterministic: the scorer is rapidfuzz (a langres core
dependency), the gold is the committed Fodors-Zagat ``perfectMapping``, and the
split is seeded. The fixtures are committed so the demo and its exit-criteria
test read frozen data (no rapidfuzz, no split at run time). Re-run only to change
the scenario::

    uv run python examples/data/flywheel/generate_fixtures.py

The scenario is tuned so the initial (weak-verdict) threshold is measurably
suboptimal on held-out gold and >=25 corrections move the re-derived threshold in
the CORRECT direction (higher held-out F1) -- see the module docstring of the
demo for the schema documentation and the exit-criteria test for the assertion.
"""

from __future__ import annotations

import csv
import json
import random
from importlib import resources
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from langres.curation.harvest import Correction, CorrectionLog
from langres.tracking.judgement_log import JudgementLog

#: Scenario parameters (chosen so the demo is decisive and non-flaky; see module
#: docstring). ``T_BAD`` is the cheap judge's deliberately-miscalibrated verdict
#: cut; the free scorer's true-optimal cut sits near ~0.64, so 0.55 over-merges.
_SEED = 7
_T_BAD = 0.55
_N_CORRECTIONS = 40
_JUDGE_MODEL = "rapidfuzz/name_addr_token_sort"
_DATASET_PACKAGE = "langres.data.datasets.fodors_zagat"
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_GOLD_SET = _REPO_ROOT / "data" / "gold_sets" / "fodors_zagat" / "gold_set.json"


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
                "name": _unquote(row.get("name", "")),
                "addr": _unquote(row.get("addr", "")),
            }
    return records


def _load_true_pairs() -> set[frozenset[str]]:
    """The perfectMapping as order-independent true match pairs."""
    text = (
        resources.files(_DATASET_PACKAGE)
        .joinpath("fodors-zagats_perfectMapping.csv")
        .read_text(encoding="utf-8")
    )
    return {
        frozenset({f"f{_unquote(row['fodors_id'])}", f"z{_unquote(row['zagats_id'])}"})
        for row in csv.DictReader(text.splitlines())
    }


def _score(left: dict[str, str], right: dict[str, str]) -> float:
    """Free, deterministic string similarity: mean of name- and address-sim.

    A cheap judge a user would reach for before paying for an LLM. Discriminative
    on Fodors-Zagat (matches cluster high, non-matches low) but with no built-in
    calibrated threshold -- exactly the setting the flywheel is meant to fix.
    """
    name_sim = fuzz.token_sort_ratio(left["name"], right["name"])
    addr_sim = fuzz.token_sort_ratio(left["addr"], right["addr"])
    return (name_sim + addr_sim) / 200.0


def main() -> None:
    """Regenerate and overwrite the three committed JSONL fixtures."""
    records = _load_records()
    true_pairs = _load_true_pairs()
    gold_set = json.loads(_GOLD_SET.read_text(encoding="utf-8"))

    # Score every mined candidate pair (the real VectorBlocker candidate set).
    scored: list[tuple[str, str, float, bool]] = []
    for pair in gold_set["pairs"]:
        left_id, right_id = pair["left_id"], pair["right_id"]
        if left_id in records and right_id in records:
            score = _score(records[left_id], records[right_id])
            label = frozenset({left_id, right_id}) in true_pairs
            scored.append((left_id, right_id, score, label))

    # Deterministic 50/50 calibration / held-out split.
    rng = random.Random(_SEED)
    order = list(range(len(scored)))
    rng.shuffle(order)
    calibration = [scored[i] for i in order[::2]]
    heldout = [scored[i] for i in order[1::2]]

    # --- judgements.jsonl: the cheap judge scoring calibration at the BAD cut. ---
    judgements_path = _HERE / "judgements.jsonl"
    judgements_path.unlink(missing_ok=True)
    log = JudgementLog(judgements_path)
    for left_id, right_id, score, _label in calibration:
        judgement = _as_judgement(left_id, right_id, score)
        log.append(judgement, verdict=score >= _T_BAD)

    # --- corrections.jsonl: what a review queue surfaces first -- the verdicts
    # nearest the current threshold that disagree with true gold. ---
    wrong = [
        (left_id, right_id, score, label)
        for (left_id, right_id, score, label) in calibration
        if (score >= _T_BAD) != label
    ]
    wrong.sort(key=lambda row: abs(row[2] - _T_BAD))
    corrections_path = _HERE / "corrections.jsonl"
    corrections_path.unlink(missing_ok=True)
    correction_log = CorrectionLog(corrections_path)
    for left_id, right_id, score, label in wrong[:_N_CORRECTIONS]:
        correction_log.append(
            Correction(
                left_id=left_id,
                right_id=right_id,
                label=label,
                original_score=score,
                original_verdict=score >= _T_BAD,
                reviewer="fixture-simulated-reviewer",
            )
        )

    # --- heldout_gold.jsonl: the untouched evaluation split with TRUE labels. ---
    heldout_path = _HERE / "heldout_gold.jsonl"
    with heldout_path.open("w", encoding="utf-8") as fh:
        for left_id, right_id, score, label in heldout:
            row: dict[str, Any] = {
                "left_id": left_id,
                "right_id": right_id,
                "score": score,
                "label": label,
            }
            fh.write(json.dumps(row) + "\n")

    n_corr = min(_N_CORRECTIONS, len(wrong))
    print(
        f"Wrote fixtures: {len(calibration)} judgements, {n_corr} corrections, "
        f"{len(heldout)} held-out gold pairs (seed={_SEED}, T_bad={_T_BAD})."
    )


def _as_judgement(left_id: str, right_id: str, score: float) -> Any:
    """Build a minimal PairwiseJudgement the JudgementLog can log."""
    from langres.core.models import PairwiseJudgement

    return PairwiseJudgement(
        left_id=left_id,
        right_id=right_id,
        score=score,
        score_type="heuristic",
        decision_step="string_judge",
        provenance={"model": _JUDGE_MODEL, "cost_usd": 0.0},
    )


if __name__ == "__main__":
    main()
