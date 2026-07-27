#!/usr/bin/env bash
# Drive the embedder ladder one model at a time, committing after each.
#
# Why a script and not a shell loop typed by hand:
#
#   - **Durability.** A model's rows must be committed AND pushed the moment they
#     exist, not when someone next looks. A paid run on this repo was already
#     lost to "it was still in the worktree" -- this script exists so that cannot
#     happen again. It is committed BEFORE it is ever run, for the same reason.
#   - **Strict sequencing.** Two concurrent `uv run` in one worktree race on
#     .venv/.coverage and fabricate plausible-but-fake failures (corrupted
#     numpy/faiss errors that have nothing to do with the code). Every step here
#     waits for the previous one to exit. Never parallelise this.
#   - **A failure is a result.** A model that will not load, OOMs, or is killed
#     must leave a visible row, not a gap. The harness writes a failure row when
#     it catches an exception; when the PROCESS itself dies (OOM kill, segfault)
#     there is nothing left to write it, so this script writes it instead and
#     keeps going. A sweep that silently drops what it could not run reads as
#     "these are the models that exist".
#
# Model order is deliberate and is NOT ascending size:
#   1. the reference model first, so the paired-CI sidecar is fresh before any
#      other model is compared against it;
#   2. then the instruction-trained checkpoints (EmbeddingGemma, Qwen3), because
#      they are the only ones that make the query_prompt axis mean anything --
#      every model measured before them was trained without a query-side
#      instruction, so a truncated ascending-size sweep answers a different
#      question than the one asked;
#   3. then the cheap re-runs (warm embedding cache, fast);
#   4. then the tail, largest last, where an OOM is an expected outcome.
#
# Usage:
#   bash examples/research/run_ladder.sh [PID_TO_WAIT_FOR]
#
# Run from the repository root.

set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

ROWS="docs/research/20260727_embedder_ladder_rows.jsonl"
REPORT="docs/research/20260727_embedder_ladder.md"
REFERENCE="docs/research/20260727_embedder_ladder_reference_recall.json"
LOG_DIR="tmp/ladder_logs"
BENCHMARKS="fodors_zagat abt_buy amazon_google wdc_computers walmart_amazon"

mkdir -p "$LOG_DIR"

MODELS=(
  "all-MiniLM-L6-v2"
  "google/embeddinggemma-300m"
  "Qwen/Qwen3-Embedding-0.6B"
  "all-MiniLM-L12-v2"
  "BAAI/bge-small-en-v1.5"
  "all-mpnet-base-v2"
  "BAAI/bge-base-en-v1.5"
  "intfloat/e5-base-v2"
  "Alibaba-NLP/gte-base-en-v1.5"
  "nomic-ai/nomic-embed-text-v1.5"
  "BAAI/bge-large-en-v1.5"
  "mixedbread-ai/mxbai-embed-large-v1"
  "Qwen/Qwen3-Embedding-4B"
  "Qwen/Qwen3-Embedding-8B"
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# 1. Wait for any in-flight sweep. NOT optional: a second `uv run` in this
#    worktree does not fail loudly, it invents failures.
# ---------------------------------------------------------------------------
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  log "waiting for pid $WAIT_PID to exit before touching uv"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 15
  done
  log "pid $WAIT_PID is gone"
  sleep 10
fi

# ---------------------------------------------------------------------------
# 2. Record a process-level death as a real result row.
#    Stdlib python only, and explicitly NOT the venv interpreter -- this runs
#    right after a `uv run` and must not become the second one.
# ---------------------------------------------------------------------------
record_process_failure() {
  local model="$1" code="$2"
  ROWS="$ROWS" FAILED_MODEL="$model" EXIT_CODE="$code" BENCHMARKS="$BENCHMARKS" \
    /usr/bin/python3 <<'PY'
import json
import os

rows_path = os.environ["ROWS"]
model = os.environ["FAILED_MODEL"]
code = os.environ["EXIT_CODE"]

with open(rows_path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

# Replace, never append: a re-run must reproduce the table, not grow it.
rows = [r for r in rows if r["model"] != model]
for benchmark in os.environ["BENCHMARKS"].split():
    rows.append(
        {
            "model": model,
            "benchmark": benchmark,
            "prompt_arm": "-",
            "k": 0,
            "status": "failed",
            "metric_revision": 1,
            "error": (
                f"process exited {code} without writing rows "
                "(killed, OOM, or segfault -- the harness never got to catch it)"
            ),
        }
    )

with open(rows_path, "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\n")
print(f"recorded process-level failure for {model}")
PY
}

# ---------------------------------------------------------------------------
# 3. One model at a time: measure, render, commit, push.
# ---------------------------------------------------------------------------
for model in "${MODELS[@]}"; do
  safe="${model//\//__}"
  log "=== $model ==="

  uv run --env-file .env python examples/research/embedder_ladder.py \
    --models "$model" --device mps > "$LOG_DIR/$safe.log" 2>&1
  code=$?

  if [ $code -ne 0 ]; then
    log "$model exited $code -- recording a failure row and continuing"
    record_process_failure "$model" "$code"
    # Re-render so the failure is visible in the report too.
    uv run python examples/research/embedder_ladder.py --render-only \
      >> "$LOG_DIR/$safe.log" 2>&1
  fi

  git add "$ROWS" "$REPORT" "$REFERENCE" 2>/dev/null
  if git diff --cached --quiet; then
    log "$model produced no change to the tracked artifacts"
  else
    git commit -q -m "results(embedder-ladder): $model" \
      -m "Measured by examples/research/run_ladder.sh, committed as soon as the rows existed. Exit code $code." \
      && log "committed $model"
    git push -q origin HEAD 2>/dev/null && log "pushed" || log "push failed (will retry next model)"
  fi
done

log "ladder complete"
