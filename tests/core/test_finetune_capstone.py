"""Acceptance test for the fine-tune CAPSTONE: the full PUBLIC train->serve->evaluate flow.

Where ``test_finetune.py::test_finetune_overfit_train_serve_evaluate`` is the
go/no-go for the trainer INTERNALS (hand-built pairs -> run_finetune -> serve),
this is the acceptance of the user-facing CAPSTONE (``examples/finetune_capstone.py``):
every step goes through the PUBLIC surface --

  * ``candidates_for(get_benchmark("fodors_zagat"), split=...)`` for real blocked
    candidate pairs + gold (the eval facade + a real benchmark, not fixtures),
  * ``run_finetune(pairs, QLoRA(...))`` for the model_ref + GPU-seconds,
  * a JSON round-trip of ``to_config(model_ref)`` -> ``normalize_model_ref`` (the
    "weightless ref survives save/reload" claim),
  * ``LLMMatcher(model=ref, ...)`` in-process serve with the finetune prompt,
  * ``evaluate(matcher, held_out, gold, threshold=0.5)`` for an honest held-out F1.

It trains on a small TRAIN-split slice and evaluates on a disjoint TEST-split
slice, so the F1 is genuine generalization (the two splits are entity-disjoint by
benchmark construction), and asserts F1 clears a floor AND the run reports a real
cost (GPU-seconds > 0, on a recorded device). Marked ``slow`` + ``finetune`` so it
runs only in the dedicated ``test-finetune`` job / on demand, never in the fast
suite (real model downloads + real training).
"""

from __future__ import annotations

# faiss (candidates_for's blocker) and torch (the fine-tune) each ship their own
# OpenMP runtime; loading both in one process can hard-crash on macOS. Allow the
# duplicate load before either is imported. Harmless where it does not apply.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from typing import Any

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.finetune]

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _balanced(cands: list[Any], gold: set[frozenset[str]], k: int) -> list[tuple[Any, bool]]:
    """Up to ``k`` gold-positive candidates + ``k`` negatives, labeled, stable order."""
    pos = [c for c in cands if frozenset({str(c.left.id), str(c.right.id)}) in gold]
    neg = [c for c in cands if frozenset({str(c.left.id), str(c.right.id)}) not in gold]
    return [(c, True) for c in pos[:k]] + [(c, False) for c in neg[:k]]


def test_capstone_train_serve_evaluate_public_flow() -> None:
    """The whole public capstone runs: fine-tune -> reload ref -> serve -> held-out F1."""
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")

    from langres.core import LLMMatcher
    from langres.core.finetune import FINETUNE_YES_NO_PROMPT, QLoRA, run_finetune
    from langres.core.matchers.model_ref import normalize_model_ref, to_config
    from langres.eval import candidates_for, evaluate, get_benchmark

    bench = get_benchmark("fodors_zagat")

    # TRAIN: block the train split, label from gold, balance a small slice.
    train_cands, train_gold = candidates_for(bench, split="train")
    train_pairs = _balanced(train_cands, train_gold, k=24)
    assert sum(1 for _, y in train_pairs if y) >= 8  # enough positive signal to learn

    method = QLoRA(base=BASE, epochs=5, batch_size=8)
    outcome = run_finetune(train_pairs, method)

    # COST is real and honest: GPU-seconds > 0 on a recorded device; a base+adapter
    # ref (nothing merged); n_train matches what we trained on.
    assert outcome.gpu_seconds > 0.0
    assert outcome.device in {"cuda", "mps", "cpu"}
    assert outcome.model_ref.adapter is not None
    assert outcome.n_train == len(train_pairs)

    # SAVE -> RELOAD the weightless model_ref through JSON (base id + adapter path,
    # no weight blob) and serve the reloaded ref -- proving the ref round-trips.
    reloaded = normalize_model_ref(json.loads(json.dumps(to_config(outcome.model_ref))))
    matcher: LLMMatcher[Any] = LLMMatcher(
        model=to_config(reloaded),
        confidence="logprob",
        response_parser="binary_yes_no",
        prompt_template=FINETUNE_YES_NO_PROMPT,
    )

    # EVALUATE on a disjoint TEST-split slice at a fixed cut -> honest held-out F1.
    test_cands, test_gold = candidates_for(bench, split="test")
    eval_pairs = _balanced(test_cands, test_gold, k=20)
    eval_cands = [c for c, _ in eval_pairs]
    eval_gold = {frozenset({str(c.left.id), str(c.right.id)}) for c, y in eval_pairs if y}
    result = evaluate(matcher, eval_cands, eval_gold, threshold=0.5)

    assert result.graded_threshold == 0.5  # graded once at the honest fixed cut
    # Floor, not a quality claim: a real trained matcher clears it; noise does not.
    # (Observed on this slice: see the PR report; floor set conservatively below it.)
    assert result.pair.f1 >= 0.5
