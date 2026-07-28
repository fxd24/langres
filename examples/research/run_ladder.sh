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
# Environment:
#   LADDER_MODELS                 space-separated subset to sweep
#   LADDER_ARTIFACT               path PREFIX for this study's three tracked
#                                 artifacts (default:
#                                 docs/research/20260727_embedder_ladder), which
#                                 become <prefix>_rows.jsonl, <prefix>.md and
#                                 <prefix>_reference_recall.json. Set it together
#                                 with LADDER_MODELS and LADDER_REFERENCE_MODEL
#                                 to run a separate study without touching the
#                                 standing ladder's files.
#   LADDER_REFERENCE_MODEL        baseline every delta is measured against
#                                 (default: the harness's own, all-MiniLM-L6-v2).
#                                 Changing it REQUIRES a fresh LADDER_ARTIFACT:
#                                 the sidecar is keyed by it and the harness
#                                 refuses to render a file mixing two baselines.
#   LADDER_DEVICE                 pin a torch device (default: let ST choose)
#   LADDER_TRUST_EXISTING_CACHE   set to exactly 1 to forward
#                                 --trust-existing-cache to the harness, vouching
#                                 for an embedding cache that predates the
#                                 integrity canary. Requires LADDER_MODELS to name
#                                 exactly ONE model: vouching is a claim about a
#                                 specific checkpoint's vectors, so a sweep cannot
#                                 make it. An env var and not a flag because this
#                                 script's only positional argument is a wait PID
#                                 -- `--trust-existing-cache` passed here would be
#                                 silently read as one.
#
# Run from the repository root.

set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

# One artifact set per (model list, reference model). Overridable together so a
# study with a different baseline gets its own rows/report/sidecar: the sidecar
# is keyed by the reference model and the recorded deltas are labelled with it,
# so pointing LADDER_REFERENCE_MODEL at a new baseline while writing the default
# files would produce a rows file the harness then refuses to render. Defaults
# reproduce the 2026-07-27 ladder exactly.
ARTIFACT="${LADDER_ARTIFACT:-docs/research/20260727_embedder_ladder}"
ROWS="${ARTIFACT}_rows.jsonl"
REPORT="${ARTIFACT}.md"
REFERENCE="${ARTIFACT}_reference_recall.json"
LOG_DIR="tmp/ladder_logs"
BENCHMARKS="fodors_zagat abt_buy amazon_google wdc_computers walmart_amazon"

# Empty = the harness's own default (the 2026-07-27 baseline, all-MiniLM-L6-v2).
REFERENCE_MODEL_ARG=()
if [ -n "${LADDER_REFERENCE_MODEL:-}" ]; then
  # Caught here rather than 40 minutes later at the first render: a new baseline
  # written into the standing ladder's rows file mixes two baselines in one
  # column, which the harness refuses -- AFTER the sweep has already spent the
  # compute and committed the rows.
  if [ -z "${LADDER_ARTIFACT:-}" ]; then
    log() { echo "[$(date '+%H:%M:%S')] $*"; }
    log "LADDER_REFERENCE_MODEL is set but LADDER_ARTIFACT is not."
    log "  Every vs-reference delta is labelled with the baseline that produced it, and"
    log "  the sidecar is keyed by it, so a new baseline needs its own artifact set."
    log "  Re-run with e.g. LADDER_ARTIFACT=docs/research/<date>_<study>"
    exit 2
  fi
  REFERENCE_MODEL_ARG=(--reference-model "$LADDER_REFERENCE_MODEL")
fi

mkdir -p "$LOG_DIR"

# The full ladder, in the deliberate order documented above. Override with
# LADDER_MODELS (space-separated) to drive a subset -- e.g. re-measuring only the
# models stranded at an older metric revision, which is a real and recurring
# need and otherwise invites editing this list in place and forgetting to
# restore it.
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
if [ -n "${LADDER_MODELS:-}" ]; then
  # shellcheck disable=SC2206  # word splitting is the interface here
  MODELS=(${LADDER_MODELS})
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# `${VAR:+...}` tests only that the variable is NON-EMPTY, so
# LADDER_TRUST_EXISTING_CACHE=0 / false / a typo would all have forwarded
# --trust-existing-cache -- silently adopting unverified vectors on precisely the
# run where the operator was trying to turn adoption OFF. An exact value, checked
# once, and anything else is a hard error rather than a quiet "no": a misspelt
# opt-in that reads as "off" is how a gate stops firing without anyone noticing.
# (Cross-model review.)
TRUST_ARG=()
case "${LADDER_TRUST_EXISTING_CACHE:-}" in
  "") ;;
  1) TRUST_ARG=(--trust-existing-cache)
     # The harness requires exactly one --models per invocation before it will
     # adopt a cache, and this loop hands it exactly one -- per model, N times
     # over. So the Python safeguard passes on every child while the DRIVER
     # blesses the whole ladder: six unverified namespaces from a flag documented
     # as vouching for one. The safeguard has to exist at the level that chooses
     # the model list, which is here. (Cross-model review.)
     if [ ${#MODELS[@]} -ne 1 ]; then
       log "LADDER_TRUST_EXISTING_CACHE=1 vouches for ONE cache, but this run sweeps"
       log "  ${#MODELS[@]} models. Adopting a namespace asserts its vectors came from"
       log "  the checkpoint loaded now -- a claim about a specific model, not one you"
       log "  can make for a whole ladder at once. Re-run as:"
       log "    LADDER_MODELS='<one model>' LADDER_TRUST_EXISTING_CACHE=1 $0"
       exit 2
     fi
     log "LADDER_TRUST_EXISTING_CACHE=1 -- vouching for ${MODELS[0]}'s cache this run" ;;
  *) log "LADDER_TRUST_EXISTING_CACHE must be exactly 1 or unset (got '${LADDER_TRUST_EXISTING_CACHE}')"
     exit 2 ;;
esac

# `.env` is gitignored, so a fresh checkout or a worktree does not have one and
# `uv run --env-file .env` exits 2 before the harness starts. This sweep is $0
# and keyless -- so a missing `.env` must not stop it. (Keyless is not the same as
# offline: with cold `uv` or Hugging Face caches this still needs the network to
# resolve dependencies and download checkpoints. It is offline only once those
# caches are warm.) Pass
# the flag only when the file is actually there.
ENV_FILE_ARG=()
if [ -f .env ]; then
  ENV_FILE_ARG=(--env-file .env)
else
  log "no .env in this checkout -- running without one (this sweep is \$0 and keyless)"
fi

# The OpenMP settings are NOT optional, and making `.env` optional above is
# exactly what exposed that: torch, faiss and scikit-learn each bundle their own
# `libomp.dylib`, and with two runtimes loaded a sweep DEADLOCKS in
# `__kmp_join_barrier` -- the process sits at 0% CPU forever rather than failing,
# so the failure path below never fires and the sweep simply stops. Observed on
# `all-mpnet-base-v2` after three models had already succeeded, which is what
# makes it worth pinning rather than hoping.
#
# `docs/FRICTION_LOG.md` documents these as `.env` contents. Defaulting them
# HERE (only when unset, so `.env` and the caller still win) is what makes the
# script correct on a checkout that has no `.env` -- otherwise "it runs without
# one" is true right up until it hangs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-1}"
log "OMP_NUM_THREADS=$OMP_NUM_THREADS KMP_DUPLICATE_LIB_OK=$KMP_DUPLICATE_LIB_OK"

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

  # No --device: the harness passes None to sentence-transformers, whose
  # get_device_name() picks cuda / mps / npu / xpu / cpu for the host it is on
  # (util/environment.py:51). Pinning a device here would make every run on a
  # host without it exit non-zero -- and the failure path below then DELETES
  # that model's previously measured rows and commits the deletion. Set
  # LADDER_DEVICE only to override deliberately.
  uv run ${ENV_FILE_ARG[@]+"${ENV_FILE_ARG[@]}"} python examples/research/embedder_ladder.py \
    --models "$model" ${LADDER_DEVICE:+--device "$LADDER_DEVICE"} \
    --rows "$ROWS" --report "$REPORT" --reference "$REFERENCE" \
    --ladder-models "${MODELS[@]}" \
    ${REFERENCE_MODEL_ARG[@]+"${REFERENCE_MODEL_ARG[@]}"} \
    ${TRUST_ARG[@]+"${TRUST_ARG[@]}"} \
    > "$LOG_DIR/$safe.log" 2>&1
  code=$?

  # A cache-integrity refusal is NOT a model failure, and must not be recorded as
  # one: record_process_failure deletes every existing row for the model and the
  # commit below ships the deletion. Nothing is wrong with those rows -- the cache
  # is what is suspect -- so stop the sweep and leave the tracked artifacts
  # untouched. The harness reserves this code for exactly that
  # (embedder_ladder.py::EXIT_CACHE_INTEGRITY). (Cross-model review.)
  if [ $code -eq 3 ]; then
    log "$model: cache-integrity refusal (exit 3). NOT recording a failure row --"
    log "  the recorded rows are fine; the embedding cache is not. See $LOG_DIR/$safe.log."
    log "  Delete that model's cache namespace and re-run, or re-run this driver as"
    # LADDER_MODELS is not optional in this suggestion: trust requires exactly one
    # model, and this message normally fires mid-sweep where MODELS is all 14. The
    # earlier version omitted it, so the advertised recovery command hit the
    # single-model guard and exited 2 -- an unusable escape hatch printed at the
    # exact moment it is needed. (Cross-model review.)
    log "    LADDER_MODELS='$model' LADDER_TRUST_EXISTING_CACHE=1 $0"
    log "  -- if you know the cache matches the loaded checkpoint. (The driver takes"
    log "  no such flag: its only positional argument is a wait PID, so"
    log "  --trust-existing-cache would be read as one.)"
    exit 3
  fi

  if [ $code -ne 0 ]; then
    log "$model exited $code -- recording a failure row and continuing"
    record_process_failure "$model" "$code"
    # Re-render so the failure is visible in the report too. The paths and the
    # ladder set must match the measuring call above: rendering the DEFAULT
    # artifact here would rewrite the 2026-07-27 ladder's report from a different
    # study's rows.
    uv run python examples/research/embedder_ladder.py --render-only \
      --rows "$ROWS" --report "$REPORT" --reference "$REFERENCE" \
      --ladder-models "${MODELS[@]}" \
      ${REFERENCE_MODEL_ARG[@]+"${REFERENCE_MODEL_ARG[@]}"} \
      >> "$LOG_DIR/$safe.log" 2>&1
  fi

  git add "$ROWS" "$REPORT" "$REFERENCE" 2>/dev/null
  # Path-scoped, like the commit below: an operator's unrelated staged work
  # would otherwise read as "this model produced results".
  if git diff --cached --quiet -- "$ROWS" "$REPORT" "$REFERENCE"; then
    log "$model produced no change to the tracked artifacts"
  else
    # --only: commit EXACTLY these three paths. A plain `git commit` takes the
    # whole index, so anything the operator had staged for unrelated work would
    # be swept into this commit and pushed by the next line -- publishing
    # in-progress work nobody chose to publish.
    # A failed commit STOPS the sweep. This script exists to make each model's
    # rows durable the moment they exist; if the commit fails (hook, missing
    # identity, disk), the results are merely staged, and carrying on would let
    # the NEXT model's --only commit bundle them under the wrong message. A
    # bare `&& log` would swallow exactly that.
    if ! git commit -q --only "$ROWS" "$REPORT" "$REFERENCE" \
      -m "results($(basename "$ARTIFACT")): $model" \
      -m "Measured by examples/research/run_ladder.sh, committed as soon as the rows existed. Exit code $code."
    then
      log "FATAL: commit failed for $model. Results are staged but NOT durable; stopping."
      exit 1
    fi
    log "committed $model"
    git push -q origin HEAD 2>/dev/null && log "pushed" || log "push failed (will retry next model)"
  fi
done

log "ladder complete"
