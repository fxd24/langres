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
# --ladder-models still names all four study-A models even though only one is
# measured: that argument is the report's coverage DENOMINATOR, not the work
# list. Collapsing it to the single measured model is a regression this harness
# has already shipped once.
set -u

export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

ARTIFACT="docs/research/20260729_lfm25_tuned"

uv run python examples/research/embedder_ladder.py \
  --models "LiquidAI/LFM2.5-Embedding-350M" \
  --benchmarks walmart_amazon \
  --rows "${ARTIFACT}_rows.jsonl" \
  --report "${ARTIFACT}.md" \
  --reference "${ARTIFACT}_reference_recall.json" \
  --ladder-models "intfloat/e5-base-v2" "all-MiniLM-L6-v2" \
    "BAAI/bge-base-en-v1.5" "LiquidAI/LFM2.5-Embedding-350M" \
  --reference-model "intfloat/e5-base-v2"
