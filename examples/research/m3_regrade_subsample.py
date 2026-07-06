"""Re-grade the AG subsample LLM cells against in-scope gold (no new LLM calls).

The first paid run graded the 600-pair Amazon-Google subsample cells against the
*full* 234-pair literature gold. Because only 61 of those 234 gold pairs are in the
600-pair subsample, recall was hard-capped at 61/234 ≈ 0.26 for the LLM judges —
a measurement artifact, not real judge behaviour. ``evaluate_judge_on_candidates``
now restricts gold to candidate-realizable pairs (see ``benchmark.py``); this script
brings the already-committed paid cells in line with that fix.

It needs no new LLM calls: each stored ``pr_curve`` point already carries the
threshold's true/false positives (``tp``/``fp``), which are denominator-independent.
Only recall (``tp / n_positive_in_subsample``) and F1 change. The in-scope positive
count is recovered by rebuilding the deterministic (seed-0) stratified subsample.

Idempotent: cells already carrying ``regrade`` metadata are skipped.

Run: ``uv run python examples/research/m3_regrade_subsample.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from examples.research.m3_race import (  # noqa: E402
    AG_LLM_SUBSAMPLE,
    build_ag_fixed_candidates,
    subsample_stratified,
)

RESULTS_DIR = Path("data/benchmarks/m3/results")
AG_SUBSAMPLE_CELLS = ("agfixed_llm_judge", "agfixed_llm_judge_frontier")


def _ag_subsample_positive_count() -> tuple[int, int]:
    """(positives in the seed-0 600-pair subsample, positives in the full set)."""
    cands, gold = build_ag_fixed_candidates("test")
    sub = subsample_stratified(cands, gold, AG_LLM_SUBSAMPLE, seed=0)
    sub_pairs = {frozenset((c.left.id, c.right.id)) for c in sub}
    return len(gold & sub_pairs), len(gold)


def _regrade_curve(curve: list[dict[str, Any]], n_pos: int) -> list[dict[str, Any]]:
    """Recompute recall/F1/fn against ``n_pos`` in-scope positives (tp/fp unchanged)."""
    out = []
    for m in curve:
        tp, fp = m["tp"], m["fp"]
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / n_pos if n_pos > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        out.append(
            {
                "threshold": m["threshold"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": n_pos - tp,
            }
        )
    return out


def main() -> None:
    n_pos, n_pos_full = _ag_subsample_positive_count()
    for name in AG_SUBSAMPLE_CELLS:
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            print(f"[skip] {name}: no cell")
            continue
        cell = json.loads(path.read_text())
        if "regrade" in cell:
            print(f"[skip] {name}: already regraded")
            continue
        curve = _regrade_curve(cell["eval"]["pair"]["pr_curve"], n_pos)
        best = max(curve, key=lambda m: m["f1"])
        old = cell["eval"]["pair"]
        cell["eval"]["pair"] = {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "pr_curve": curve,
        }
        cell["eval"]["best_threshold"] = best["threshold"]
        cell["regrade"] = {
            "reason": "restrict gold to in-scope (candidate-realizable) pairs",
            "n_positive_in_subsample": n_pos,
            "n_positive_full": n_pos_full,
            "f1_before": old["f1"],
            "f1_after": best["f1"],
        }
        path.write_text(json.dumps(cell, indent=2))
        print(
            f"[regrade] {name}: F1 {old['f1']:.4f} -> {best['f1']:.4f} "
            f"(R {old['recall']:.4f} -> {best['recall']:.4f} @thr{best['threshold']:.2f}, "
            f"in-scope positives {n_pos}/{n_pos_full})"
        )


if __name__ == "__main__":
    main()
