#!/bin/sh
# Run the prompt-axis sweep one (model, benchmark) per process.
#
# Two environment traps make this the only shape that finishes on macOS, both
# documented in docs/FRICTION_LOG.md:
#
#   OMP_NUM_THREADS=1  -- without it the Gemma load DEADLOCKS at 0% CPU once
#                         faiss has been imported. KMP_DUPLICATE_LIB_OK alone
#                         suppresses the abort, not the hang.
#   one process per cell -- the MPS allocator caches freed blocks across model
#                         loads, so a single-process sweep OOMs partway through
#                         (measured: row 324 of 400, torch asking 42.44 GiB for
#                         a 0.6B fp16 model).
#
# Splitting costs nothing: embeddings come back from the on-disk cache, and
# --resume skips whole cells that are already recorded.
#
# Usage:  sh examples/research/prompt_axis_sweep.sh <rows.jsonl> <report.md>
set -eu

ROWS=${1:-docs/research/20260728_prompt_axis_rows.jsonl}
REPORT=${2:-docs/research/20260728_prompt_axis.md}

MODELS="sentence-transformers/all-MiniLM-L6-v2 intfloat/e5-base-v2 BAAI/bge-base-en-v1.5 google/embeddinggemma-300m Qwen/Qwen3-Embedding-0.6B"
BENCHMARKS="fodors_zagat abt_buy amazon_google wdc_computers"

for model in $MODELS; do
  for benchmark in $BENCHMARKS; do
    echo "=== $model | $benchmark ==="
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
      uv run python examples/research/prompt_axis.py \
        --models "$model" --benchmarks "$benchmark" \
        --rows "$ROWS" --report "$REPORT" --resume
  done
done

echo "=== sweep complete ==="
