#!/bin/bash
# Resume the one study-A cell the 2026-07-29 OS kill took out.
#
# Why this exists rather than "just re-run run_lfm25.sh": the harness has no
# skip-completed logic. merge_rows() REPLACES re-measured cells, so a full
# re-run would redo ~90 minutes of already-committed work -- and re-measuring
# REFERENCE_MODEL additionally clears every other model's vs_reference_* on the
# benchmarks it touches, because those deltas were computed against per-record
# scores that no longer exist. Re-running the whole study would therefore churn
# results that are already correct.
#
# Why it goes through run_ladder.sh rather than calling embedder_ladder.py
# directly, which is what it used to do: that driver owns three things a resume
# cannot afford to skip, and this script skipped all three.
#   1. The dirty-artifact refusal. A resume writes into rows that a killed sweep
#      may have left holding uncommitted measured cells -- overwriting them is
#      exactly the loss the guard exists for, and this script is reached in
#      precisely that situation.
#   2. The provenance --verify before anything is published.
#   3. The COMMIT. A resume measures an expensive cell; leaving it uncommitted in
#      a worktree is the failure this repo has already paid for once.
# The benchmark narrowing that made a direct call look necessary is now an env
# var on the driver. (Cross-model review.)
#
# LADDER_ALL_MODELS still names all four study-A models even though only one is
# measured: that argument is the report's coverage DENOMINATOR, not the work
# list. Collapsing it to the single measured model is a regression this harness
# has already shipped once.
set -u

export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

STUDY_A_MODELS="intfloat/e5-base-v2 all-MiniLM-L6-v2 BAAI/bge-base-en-v1.5 LiquidAI/LFM2.5-Embedding-350M"

LADDER_ARTIFACT="docs/research/20260729_lfm25_tuned" \
LADDER_REFERENCE_MODEL="intfloat/e5-base-v2" \
LADDER_ALL_MODELS="$STUDY_A_MODELS" \
LADDER_MODELS="LiquidAI/LFM2.5-Embedding-350M" \
LADDER_BENCHMARKS="walmart_amazon" \
  bash examples/research/run_ladder.sh
