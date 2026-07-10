"""Analyse the logprob-credence probe. Reads committed rows; makes NO API calls ($0).

Answers TWO questions that must never be conflated:

1. ``roc_auc(gold, p_yes)`` -- does credence rank true matches above non-matches?
   This is a *task* question. A judge can ace it and still be useless for review.

2. ``roc_auc(answer_was_correct, credence)`` -- does credence predict the model's
   OWN errors, where ``credence = max(p_yes, 1 - p_yes)`` is how sure it was of
   whatever it said? **This is the one the flywheel needs**, because
   ``select_for_review`` exists to surface the pairs the judge probably got wrong.

Then the operational number: reviewing the K% least-confident pairs, what fraction
of the model's errors do you actually catch? That is `select_for_review`'s job
description, and it is what decides whether a permanent `confidence` field is earned.

Usage:
    uv run python examples/research/analyze_logprob_credence.py <rows.jsonl>
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from langres.core.metrics import average_precision_score, roc_auc_score


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def _expected_calibration_error(conf: list[float], correct: list[int], bins: int = 10) -> float:
    """Standard ECE over equal-width credence bins."""
    total = len(conf)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (c > lo or (b == 0 and c >= lo)) and c <= hi]
        if not idx:
            continue
        acc = statistics.mean(correct[i] for i in idx)
        avg_conf = statistics.mean(conf[i] for i in idx)
        ece += (len(idx) / total) * abs(acc - avg_conf)
    return ece


def _capture_at_k(conf: list[float], correct: list[int], k_frac: float) -> tuple[int, int, float]:
    """Review the k_frac least-confident pairs: how many of the errors are caught?"""
    n_review = max(1, round(k_frac * len(conf)))
    order = sorted(range(len(conf)), key=lambda i: conf[i])  # least confident first
    reviewed = order[:n_review]
    errors_total = sum(1 for c in correct if c == 0)
    errors_caught = sum(1 for i in reviewed if correct[i] == 0)
    return errors_caught, errors_total, n_review / len(conf)


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "")
    rows = _load(path)

    # A bound is a lower bound on p_yes, not a point estimate. Never average one
    # into ECE as if it were exact -- exclude, and report how many were dropped.
    bounds = [r for r in rows if r.get("p_yes_is_bound")]
    usable = [r for r in rows if r.get("p_yes") is not None and not r.get("p_yes_is_bound")]

    print(f"rows                : {len(rows)}")
    print(f"  p_yes is a bound  : {len(bounds)}  (excluded from ECE/Brier)")
    print(f"  p_yes missing     : {sum(1 for r in rows if r.get('p_yes') is None)}")
    print(f"  usable            : {len(usable)}")

    leaked = [r["leaked_mass"] for r in usable if r.get("leaked_mass") is not None]
    if leaked:
        print(f"leaked mass         : mean {statistics.mean(leaked):.2e}  max {max(leaked):.2e}")
        if max(leaked) > 1e-3:
            print("  !! material leaked mass -- the token matcher may be dropping variants")

    # Guard: the text-parsed verdict must agree with argmax(p_yes). If it does not,
    # the token matcher is dropping a casing/whitespace BPE variant -- wrong p_yes,
    # no crash, and every number below would be quietly wrong.
    disagree = [r for r in usable if int(r["p_yes"] >= 0.5) != int(r["verdict"])]
    print(f"argmax(p_yes) != parsed verdict: {len(disagree)}/{len(usable)}")
    if len(disagree) > 0.01 * len(usable):
        print("  !! material disagreement -- SUSPECT THE TOKEN MATCHER, not the model")

    gold = [int(r["gold"]) for r in usable]
    p_yes = [float(r["p_yes"]) for r in usable]
    correct = [int(r["correct"]) for r in usable]
    # How sure the model was of whatever it actually answered.
    credence = [max(p, 1.0 - p) for p in p_yes]

    n_err = sum(1 for c in correct if c == 0)
    print(f"\naccuracy            : {statistics.mean(correct):.4f}  ({n_err} errors)")

    print("\n--- Q1: does credence rank true matches above non-matches? (task question)")
    auc_task = roc_auc_score(gold, p_yes)
    ap_task = average_precision_score(gold, p_yes)
    print(f"roc_auc(gold, p_yes)             = {auc_task:.4f}")
    print(f"average_precision(gold, p_yes)   = {ap_task:.4f}")

    print("\n--- Q2: does credence predict the model's OWN errors? (the flywheel question)")
    if n_err == 0:
        print("roc_auc(correct, credence)       = nan  (no errors -- undefined)")
        auc_self = float("nan")
    else:
        auc_self = roc_auc_score(correct, credence)
        print(f"roc_auc(correct, credence)       = {auc_self:.4f}   (0.5 = worthless)")

    print("\n--- Calibration (bounds excluded)")
    brier = statistics.mean((c - conf) ** 2 for conf, c in zip(credence, correct, strict=True))
    ece = _expected_calibration_error(credence, correct)
    print(f"Brier (credence vs correct)      = {brier:.4f}")
    print(f"ECE  (10 equal-width bins)       = {ece:.4f}")

    print("\n--- Operational: review the K% least-confident pairs")
    print(
        f"{'K':>6}  {'reviewed':>9}  {'errors caught':>14}  {'of':>4}  {'capture':>8}  {'lift':>6}"
    )
    for k in (0.01, 0.02, 0.05, 0.10, 0.20):
        caught, total, actual_k = _capture_at_k(credence, correct, k)
        cap = caught / total if total else float("nan")
        lift = cap / actual_k if actual_k else float("nan")
        print(
            f"{k:>5.0%}  {round(actual_k * len(usable)):>9}  {caught:>14}  {total:>4}  "
            f"{cap:>7.1%}  {lift:>5.1f}x"
        )

    print("\n--- VERDICT")
    if math.isnan(auc_self):
        print("UNDECIDABLE: no errors to predict.")
    elif auc_self < 0.60:
        print(f"roc_auc(correct, credence) = {auc_self:.4f} -- credence does NOT predict the")
        print("model's own errors. DROP `confidence` and `confidence_source` from PR-2.")
        print("Ship `decision` + abstain, which fix the flywheel and the recall bug alone.")
    else:
        print(f"roc_auc(correct, credence) = {auc_self:.4f} -- credence DOES carry signal about")
        print("the model's own errors. `confidence` is earned; PR-2 ships it with this evidence.")
    print("(Q1's AUC is NOT the gate -- a judge can rank matches well and still be")
    print(" uninformative about which of its own answers are wrong.)")


if __name__ == "__main__":
    main()
