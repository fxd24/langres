"""Capstone: fine-tune your own matcher, serve it in-process, and evaluate it.

The training surface end to end, on a real (small) benchmark slice --
**train -> serve -> evaluate** through the public API, reporting F1 *and* the
honest cost (GPU-seconds). This is the script to read to learn "how do I train
my own matcher end to end?"

The flow is five public calls, one per step:

  1. ``candidates_for(bench, split="train")`` -> blocked candidate pairs + gold
  2. label from gold, then balance             -> ``(candidate, is_match)`` pairs
  3. ``run_finetune(pairs, QLoRA(base=...))``   -> a weightless ``model_ref`` + GPU-seconds
  4. ``LLMMatcher(model=model_ref, ...)``        -> serve the fine-tuned model IN-PROCESS
  5. ``evaluate(matcher, test_cands, gold)``     -> honest held-out pair P/R/F1

This is a REAL fine-tune, not a mock. It downloads SmolLM2-135M and a small
embedding model (a few hundred MB total) and runs real peft LoRA + trl SFT.
On a CUDA GPU it is a couple of minutes; on CPU / Apple-Silicon MPS it is SLOW
(~10-20 min) but it DOES run -- the QLoRA trainer falls back to a non-quantized
LoRA fine-tune when there is no CUDA. Local training has **no dollar cost**; the
honest cost fact is GPU-seconds (printed below).

What to expect (the flow is the lesson, not the score): SmolLM2-135M is a
deliberately tiny base, so the held-out F1 *at the default 0.5 cut* is LOW. The
fine-tune reliably finds the true matches (high recall) but over-includes
non-matches (low precision), because the small model's Yes-confidence clusters
near the 0.5 fence. That is not a broken flow -- it is a small, uncalibrated base.
The two levers to raise it, both elsewhere on this same training surface: a
larger base, or a calibrated decision threshold instead of the naive 0.5.

Needs the training + serving + blocking extras:

    uv sync --extra finetune --extra semantic --extra llm
      finetune  -> peft + trl + torch      (the fine-tune itself)
      semantic  -> sentence-transformers + faiss   (candidates_for's blocker)
      llm       -> litellm                 (LLMMatcher serves the model_ref)

Run it:

    # On macOS, KMP_DUPLICATE_LIB_OK avoids a faiss/torch OpenMP double-load crash.
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
        uv run python examples/finetune_capstone.py
"""

import random
from typing import Any

from langres.core.matchers import LLMMatcher
from langres.core.finetune import FINETUNE_YES_NO_PROMPT, QLoRA, run_finetune
from langres.core.matchers.model_ref import to_config
from langres.eval import candidates_for, evaluate, get_benchmark

# SmolLM2-135M is tiny + fast + has a plain chat template -- a good "see it run"
# base. Swap in a larger *same-family* instruct model for more headroom (e.g.
# "HuggingFaceTB/SmolLM2-360M-Instruct"). Avoid "thinking" models (e.g. Qwen3):
# they emit a reasoning turn before the answer, so the first-token Yes/No logprob
# probe this serves with reads the think-tag, not the decision -- a different
# serving contract than this example's.
BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
BENCH = "fodors_zagat"  # a small, real restaurant-matching benchmark


def _is_match(candidate: Any, gold: set[frozenset[str]]) -> bool:
    """Label a candidate pair from the benchmark's gold: is it a true match?"""
    return frozenset({str(candidate.left.id), str(candidate.right.id)}) in gold


def main() -> None:
    bench = get_benchmark(BENCH)

    # 1. BLOCK the TRAIN split into candidate pairs + gold (the public eval seam).
    #    candidates_for uses the benchmark's own vector blocker, so this needs the
    #    [semantic] extra and downloads a small embedding model the first time.
    train_cands, train_gold = candidates_for(bench, split="train")

    # 2. LABEL each candidate from gold, then BALANCE. Blocking output is heavily
    #    negative (~3% positive here); a 1:1 set stops the fine-tune from trivially
    #    learning "always No" and never separating matches from non-matches.
    pos = [(c, True) for c in train_cands if _is_match(c, train_gold)]
    neg = [(c, False) for c in train_cands if not _is_match(c, train_gold)]
    random.Random(0).shuffle(neg)
    train_pairs = pos + neg[: len(pos)]
    random.Random(1).shuffle(train_pairs)
    print(f"train: {len(train_pairs)} balanced pairs ({len(pos)} match / {len(pos)} non-match)")

    # 3. FINE-TUNE SmolLM2-135M with LoRA on the yes/no match prompt. The method
    #    object carries WHICH base + the LoRA/cost knobs; kind="finetune" means the
    #    cost is GPU-seconds (local training = $0). run_finetune returns the honest
    #    digest (ref + GPU-seconds + device); finetune() is the one-liner when you
    #    only want the model_ref.
    method = QLoRA(base=BASE, epochs=3, batch_size=8)
    outcome = run_finetune(train_pairs, method)
    print(
        f"trained on {outcome.n_train} pairs in {outcome.gpu_seconds:.1f} GPU-seconds "
        f"on {outcome.device} (${outcome.dollars:.4f}) -> {outcome.method}\n"
        f"model_ref: {to_config(outcome.model_ref)}"
    )

    # 4. SERVE the fine-tuned model IN-PROCESS -- no server, no API key: LLMMatcher
    #    loads base + LoRA adapter via the transformers backend. Serve with the SAME
    #    prompt the model was trained on (FINETUNE_YES_NO_PROMPT) and read the answer
    #    with the logprob yes/no probe. Asking a differently-worded question than
    #    training taught would score nonsense -- train and serve must match.
    matcher = LLMMatcher(
        model=to_config(outcome.model_ref),
        confidence="logprob",
        response_parser="binary_yes_no",
        prompt_template=FINETUNE_YES_NO_PROMPT,
    )

    # 5. EVALUATE on the held-out TEST split -> honest pair P/R/F1 at a fixed cut.
    #    threshold=0.5 grades ONCE at that cut (no argmax fitted to the test gold),
    #    so the F1 is a real held-out estimate, not an optimistic upper bound.
    test_cands, test_gold = candidates_for(bench, split="test")
    result = evaluate(matcher, test_cands, test_gold, threshold=0.5)
    print(
        f"\nheld-out {BENCH} test ({len(test_cands)} pairs): "
        f"P={result.pair.precision:.3f} R={result.pair.recall:.3f} F1={result.pair.f1:.3f}\n"
        f"honest cost: {outcome.gpu_seconds:.1f} GPU-seconds on {outcome.device} "
        f"(${outcome.dollars:.4f})"
    )
    print(
        "\nRecall-heavy, low precision at the 0.5 cut is expected for a 135M base "
        "(see the module docstring): it finds the matches but over-includes. Raise "
        "it with a larger base or a calibrated threshold -- same training surface."
    )


if __name__ == "__main__":
    main()
