"""The data flywheel, end to end: logged verdicts + human corrections -> a better threshold.

langres logs every judge call (:class:`~langres.tracking.judgement_log.JudgementLog`,
the flywheel *inlet*). This demo runs the *harvest* half (W2.4): it turns a
``judgements.jsonl`` log plus a ``corrections.jsonl`` review-queue export into
labeled pairs, and feeds them to
:func:`~langres.core.calibration.derive_threshold` to RE-CALIBRATE a decision
threshold -- ``derive_threshold``'s first production caller.

The scenario (committed fixtures under ``examples/data/flywheel/``, built from
Fodors-Zagat by ``generate_fixtures.py`` at $0):

1. A cheap string judge scored the *calibration* pairs with a deliberately BAD
   verdict cut (``0.55`` -- a hand-set "magic constant" that over-merges).
2. Harvesting the verdicts ALONE and deriving a threshold just recovers that bad
   cut (self-training on your own labels teaches nothing new).
3. A reviewer corrects the 40 pairs the queue surfaces first (verdicts nearest
   the threshold that were actually wrong). Harvesting WITH those corrections and
   re-deriving moves the threshold -- and it moves in the CORRECT direction:
   held-out gold F1 climbs from ~0.56 to ~0.71. That gold split is never used to
   derive the threshold, so the gain is real, not circular.

Run it (needs the ``trained`` extra for scikit-learn, which ``derive_threshold``
uses; no network, no spend)::

    uv run python examples/flywheel_threshold_harvest.py

The two flywheel JSONL schemas
------------------------------

``judgements.jsonl`` -- one line per judge call, written by ``JudgementLog``::

    {"v": 1, "left_id": "f658", "right_id": "z51", "score": 0.378,
     "verdict": false, "model": "rapidfuzz/name_addr_token_sort",
     "cost_usd": 0.0, "decision_step": "string_judge", "timestamp": "..."}

``corrections.jsonl`` -- one line per human override, written by a review queue
(the :class:`~langres.curation.harvest.Correction` contract)::

    {"v": 1, "left_id": "f601", "right_id": "z124", "label": false,
     "original_score": 0.55, "original_verdict": true,
     "reviewer": "...", "timestamp": null}

Only ``left_id``/``right_id``/``label`` are required on a correction; a pair is
matched to its judgement order-independently, so the correction need not repeat
the log's left/right order. ``heldout_gold.jsonl`` (``{left_id, right_id, score,
label}``) is the untouched evaluation split -- true labels the thresholds are
scored on but never fit on.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from langres.curation.harvest import (
    CorrectionLog,
    derive_threshold_from_pairs,
    harvest_labeled_pairs,
)
from langres.tracking.judgement_log import JudgementLog

#: Where the committed fixtures live (regenerate with ``generate_fixtures.py``).
DATA_DIR = Path(__file__).resolve().parent / "data" / "flywheel"


class ThresholdMetrics(BaseModel):
    """Pair-level classification quality of one threshold on the held-out gold set.

    A held-out pair is a predicted match iff ``score >= threshold``; the counts
    compare that prediction to the pair's true gold ``label``.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


class FlywheelResult(BaseModel):
    """The demo's before/after outcome: two thresholds and their held-out metrics."""

    n_judgements: int
    n_corrections: int
    before: ThresholdMetrics
    after: ThresholdMetrics

    @property
    def f1_gain(self) -> float:
        """Held-out F1 improvement from applying the corrections (after - before)."""
        return self.after.f1 - self.before.f1


def _load_heldout_gold(path: Path) -> list[tuple[float, bool]]:
    """Load the held-out evaluation split as ``(score, true_label)`` pairs."""
    pairs: list[tuple[float, bool]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                row = json.loads(stripped)
                pairs.append((float(row["score"]), bool(row["label"])))
    return pairs


def evaluate_threshold(heldout: list[tuple[float, bool]], threshold: float) -> ThresholdMetrics:
    """Score ``threshold`` on the held-out ``(score, label)`` pairs (pairwise P/R/F1).

    Deliberately dependency-light (no ``ranx``): every held-out pair is a labeled
    candidate, so a threshold's quality is just its pairwise precision/recall over
    that fixed set.
    """
    tp = fp = fn = 0
    for score, label in heldout:
        predicted = score >= threshold
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ThresholdMetrics(
        threshold=threshold, precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn
    )


def run_flywheel_harvest(data_dir: Path = DATA_DIR) -> FlywheelResult:
    """Harvest the fixtures, derive before/after thresholds, score both on gold.

    Args:
        data_dir: Directory holding ``judgements.jsonl``, ``corrections.jsonl``
            and ``heldout_gold.jsonl`` (defaults to the committed fixtures).

    Returns:
        A :class:`FlywheelResult` with the weak-verdict threshold and metrics
        (``before``) and the correction-informed threshold and metrics
        (``after``), each scored on the held-out gold split.
    """
    judgement_rows = JudgementLog(data_dir / "judgements.jsonl").read()
    corrections = CorrectionLog(data_dir / "corrections.jsonl").read()
    heldout = _load_heldout_gold(data_dir / "heldout_gold.jsonl")

    # BEFORE: derive from the judge's own verdicts only (weak self-labels).
    before_pairs = harvest_labeled_pairs(judgement_rows, corrections=[])
    before_threshold = derive_threshold_from_pairs(before_pairs)

    # AFTER: derive from verdicts with the human corrections overlaid.
    after_pairs = harvest_labeled_pairs(judgement_rows, corrections)
    after_threshold = derive_threshold_from_pairs(after_pairs)

    return FlywheelResult(
        n_judgements=len(judgement_rows),
        n_corrections=len(corrections),
        before=evaluate_threshold(heldout, before_threshold),
        after=evaluate_threshold(heldout, after_threshold),
    )


def format_report(result: FlywheelResult) -> str:
    """Render a human-readable BEFORE/AFTER report of the flywheel harvest."""

    def row(label: str, m: ThresholdMetrics) -> str:
        return (
            f"  {label:<7} threshold={m.threshold:.4f}  "
            f"F1={m.f1:.4f}  precision={m.precision:.4f}  recall={m.recall:.4f}  "
            f"(tp={m.tp} fp={m.fp} fn={m.fn})"
        )

    lines = [
        "=" * 78,
        "Data flywheel: harvest logged verdicts + human corrections -> new threshold",
        "=" * 78,
        f"Harvested {result.n_judgements} judgements; "
        f"applied {result.n_corrections} human corrections.",
        "",
        "Threshold quality on HELD-OUT gold (never used to derive the threshold):",
        row("BEFORE", result.before),
        row("AFTER", result.after),
        "",
        f"  => corrections moved held-out F1 by {result.f1_gain:+.4f} "
        f"({result.before.f1:.4f} -> {result.after.f1:.4f}).",
        "=" * 78,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(run_flywheel_harvest()))
