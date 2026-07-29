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
#   LADDER_MODELS                 space-separated subset to sweep (the WORK list)
#   LADDER_ALL_MODELS             space-separated set the ARTIFACT is accountable
#                                 for -- the report's coverage denominator, i.e.
#                                 the models whose absence counts as "did not
#                                 run". NOT the same as LADDER_MODELS: a
#                                 single-model resume measures one model while
#                                 remaining accountable for the whole study.
#                                 Defaults to the portfolio ladder below;
#                                 REQUIRED when LADDER_ARTIFACT names a different
#                                 study, which is enforced rather than assumed.
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

# What this ARTIFACT is accountable for -- the denominator of "what did not run",
# which is NOT the same as what this invocation measures. Captured before
# LADDER_MODELS narrows the work list, because the two were the same array and
# the subset was passed as `--ladder-models`: a single-model resume (which is a
# documented recovery path, and this study used one) shrank the coverage
# denominator to that model, so every model still missing rows vanished from the
# missing-grid check and the report could call a one-model ladder complete.
# embedder_ladder.py's own --ladder-models help warns about exactly this.
# Override only to declare a DIFFERENT study's full set. (Cross-model review.)
if [ -n "${LADDER_ALL_MODELS:-}" ]; then
  # shellcheck disable=SC2206  # word splitting is the interface here
  LADDER_ACCOUNTABLE=(${LADDER_ALL_MODELS})
elif [ -n "${LADDER_ARTIFACT:-}" ]; then
  # A separate artifact means a separate study, whose accountable set is NOT the
  # portfolio ladder defined above. Defaulting silently would put 14 portfolio
  # models in a four-model study's missing-grid. Refused up front, in the same
  # shape as the LADDER_REFERENCE_MODEL guard below and for the same reason:
  # cheaper here than after the sweep has spent the compute.
  echo "LADDER_ARTIFACT is set but LADDER_ALL_MODELS is not."
  echo "  --ladder-models is the report's coverage DENOMINATOR -- every model the"
  echo "  artifact is accountable for -- not the work list. A study with its own"
  echo "  artifact must name its own full set, or the missing-grid is measured"
  echo "  against the portfolio ladder this script defines."
  echo "  Re-run with e.g. LADDER_ALL_MODELS=\"\$LADDER_MODELS\""
  exit 2
else
  LADDER_ACCOUNTABLE=("${MODELS[@]}")
fi

if [ -n "${LADDER_MODELS:-}" ]; then
  # shellcheck disable=SC2206  # word splitting is the interface here
  MODELS=(${LADDER_MODELS})
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Push the just-committed results -- but NEVER onto the default branch.
#
# `git push origin HEAD` resolves to whatever branch is checked out. Run from
# `main` (which is where someone reaching for a sweep script most plausibly is),
# that publishes generated results straight to the default branch, bypassing the
# PR-only rule this repo treats as an irreversible-action guardrail. The commits
# are still made -- the durability the sweep depends on is untouched -- only the
# publish step is withheld, with the branch to create printed. (Cross-model
# review.)
push_results() {
  local branch default
  # Provenance is verified per PUBLICATION, not once at the end. --finish runs
  # after every model has run, but this function publishes after EACH one, so a
  # mid-sweep edit to a tracked .py could already be pushed by the time --finish
  # rejects the run. Deliberately blocks the PUSH and not the COMMIT: the rows
  # stay durable (this repo has lost a paid run to an uncommitted result), while
  # artifacts produced by code the open window does not describe are not
  # published. A no-op when no provenance sidecar exists, which is how the
  # standing portfolio ladder runs. (Cross-model review.)
  if ! uv run python examples/research/write_provenance.py --verify 2>&1; then
    log "NOT pushing: measurement code changed mid-sweep (see above)."
    log "  Rows are COMMITTED and safe. Re-measure on a stable tree before publishing."
    return 0
  fi
  branch=$(git rev-parse --abbrev-ref HEAD)
  default=$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null)
  default=${default#origin/}
  default=${default:-main}
  if [ "$branch" = "$default" ] || [ "$branch" = "HEAD" ]; then
    log "NOT pushing: on '$branch'. Results are COMMITTED locally; publish via a PR:"
    log "    git switch -c results/$(basename "$ARTIFACT") && git push -u origin HEAD"
    return 0
  fi
  if git push -q origin HEAD 2>/dev/null; then
    log "pushed"
  else
    log "push failed (will retry next model)"
  fi
}

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
#: $3 (optional) restricts the failure to specific benchmarks. In granular mode a
#: single failed cell used to condemn the whole model: the loop keeps going, later
#: benchmarks write GOOD rows, and this function then deleted every row for the
#: model and committed five failure rows over them. One OOM destroyed four valid
#: measurements and pushed the deletion. Now only the benchmarks that actually
#: failed are replaced. (Cross-model review; data-safety rule.)
record_process_failure() {
  local model="$1" code="$2" benchmarks="${3:-$BENCHMARKS}"
  ROWS="$ROWS" FAILED_MODEL="$model" EXIT_CODE="$code" BENCHMARKS="$benchmarks" \
    /usr/bin/python3 <<'PY'
import json
import os

rows_path = os.environ["ROWS"]
model = os.environ["FAILED_MODEL"]
code = os.environ["EXIT_CODE"]
failed = os.environ["BENCHMARKS"].split()

# A missing rows file is an EMPTY row set, not a crash. When LADDER_ARTIFACT
# names a fresh prefix and the very first model dies before Python creates the
# file, reading it unconditionally made this function -- the last-resort recorder
# of process-level death -- die too, so the failure it exists to preserve went
# unrecorded. (Cross-model review.)
try:
    with open(rows_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
except FileNotFoundError:
    rows = []

# Replace, never append: a re-run must reproduce the table, not grow it. Scoped
# to the FAILED benchmarks so a model's other cells survive.
rows = [r for r in rows if not (r["model"] == model and r["benchmark"] in failed)]
for benchmark in failed:
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
# Used swap in MiB, or empty when it cannot be read. The MPS allocator caches
# across encodes and never releases, so a long sweep can accumulate tens of GiB
# of allocations for a sub-1B model while RSS stays under a gigabyte -- RSS is
# useless here, swap is the observable. Sampled per cell so a monotonic climb is
# caught while there is still a machine to catch it on.
swap_used_mib() {
  sysctl vm.swapusage 2>/dev/null | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p'
}

#: Reclaimable physical memory. This is the observable that actually predicts a
#: kill, and it is here because the rise-counter below did NOT catch the failure
#: it was written for: on 2026-07-29 the sweep was killed by the OS mid-cell with
#: swap at 19892/20480M, while the counter sat at 0-2 rises the whole way up. Used
#: swap dipped between samples often enough to reset the counter, so a guard that
#: watches the *derivative* stayed silent through the exhaustion it existed to
#: prevent -- the repo's recurring "gate decoupled from what it checks" shape.
#: Free swap alone is not enough either: macOS grows the swap file on demand, so
#: `total` moves under you (an earlier sample read used=22903M against total
#: 20480M). Available physical memory does not have that problem.
mem_available_mib() {
  vm_stat 2>/dev/null | awk '
    /page size of/ {ps=$8}
    /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3}
    END {gsub(/\./,"",f); gsub(/\./,"",i); gsub(/\./,"",s);
         if (ps=="") exit; printf "%.0f", (f+i+s)*ps/1048576}'
}

SWAP_BASELINE="$(swap_used_mib)"
SWAP_PREV="$SWAP_BASELINE"
SWAP_RISES=0
#: Consecutive per-cell rises before the sweep stops. Three, not one: swap
#: fluctuates with everything else on the machine, and a single sample is noise.
SWAP_RISE_LIMIT=3
#: Hard floor on reclaimable memory, checked BEFORE each cell as well as after.
#: Below this the next encode is what tips the machine over.
MEM_FLOOR_MIB="${MEM_FLOOR_MIB:-1500}"

check_memory() {
  local now avail
  avail="$(mem_available_mib)"
  if [ -n "$avail" ] && [ "$avail" -lt "$MEM_FLOOR_MIB" ] 2>/dev/null; then
    log "ABORT: only ${avail}MiB reclaimable memory left (floor ${MEM_FLOOR_MIB}MiB)."
    log "  Continuing would get this process killed mid-cell, as happened on 2026-07-29."
    log "  Everything measured so far is committed. Re-run to resume from it."
    return 1
  fi
  now="$(swap_used_mib)"
  [ -z "$now" ] || [ -z "$SWAP_PREV" ] && { SWAP_PREV="$now"; return 0; }
  if awk "BEGIN{exit !($now > $SWAP_PREV)}"; then
    SWAP_RISES=$((SWAP_RISES + 1))
  else
    SWAP_RISES=0
  fi
  log "  swap used ${now}M (baseline ${SWAP_BASELINE}M, rises ${SWAP_RISES}, avail ${avail}MiB)"
  SWAP_PREV="$now"
  if [ "$SWAP_RISES" -ge "$SWAP_RISE_LIMIT" ]; then
    log "ABORT: used swap rose on $SWAP_RISES consecutive cells (${SWAP_BASELINE}M -> ${now}M)."
    log "  That is the allocator-accumulation signature, and continuing risks taking the"
    log "  machine down. Everything measured so far is committed. Re-run to resume."
    return 1
  fi
  return 0
}

#: Back-compat alias -- the call sites below read as "check swap" historically.
check_swap() { check_memory; }

# Position in the sweep, so the POST-cell memory guard can tell "stop before the
# next cell" from "nothing left to stop". Aborting after the final cell of the
# final model commits every row and still exits 9, which makes run_lfm25.sh stop
# before closing provenance and rendering the write-up -- discarding the report
# for a sweep that actually finished. (Cross-model review.)
MODEL_INDEX=0
N_MODELS=${#MODELS[@]}
N_BENCHMARKS=$(echo "$BENCHMARKS" | wc -w | tr -d ' ')

for model in "${MODELS[@]}"; do
  safe="${model//\//__}"
  MODEL_INDEX=$((MODEL_INDEX + 1))
  log "=== $model ==="

  # No --device: the harness passes None to sentence-transformers, whose
  # get_device_name() picks cuda / mps / npu / xpu / cpu for the host it is on
  # (util/environment.py:51). Pinning a device here would make every run on a
  # host without it exit non-zero -- and the failure path below then DELETES
  # that model's previously measured rows and commits the deletion. Set
  # LADDER_DEVICE only to override deliberately.
  measure_cell() {  # $1 = space-separated benchmark list, $2 = log mode (> or >>)
    if [ "$2" = "truncate" ]; then : > "$LOG_DIR/$safe.log"; fi
    # shellcheck disable=SC2086  # word splitting is the interface here
    uv run ${ENV_FILE_ARG[@]+"${ENV_FILE_ARG[@]}"} python examples/research/embedder_ladder.py \
      --models "$model" ${LADDER_DEVICE:+--device "$LADDER_DEVICE"} \
      --benchmarks $1 \
      --rows "$ROWS" --report "$REPORT" --reference "$REFERENCE" \
      --ladder-models "${LADDER_ACCOUNTABLE[@]}" \
      ${REFERENCE_MODEL_ARG[@]+"${REFERENCE_MODEL_ARG[@]}"} \
      ${TRUST_ARG[@]+"${TRUST_ARG[@]}"} \
      >> "$LOG_DIR/$safe.log" 2>&1
  }

  code=0
  # Initialised HERE, before the mode branch, for two reasons. (1) `set -u` is on:
  # the non-granular SUCCESS path never assigned it, so the expansion below exited
  # the script with "unbound variable" *before* the commit block -- turning a clean
  # sweep into lost, uncommitted measurements. Reproduced, not theorised. (2) It
  # must reset per model, or one model's failures would be attributed to the next.
  FAILED_BENCHMARKS=""
  # Set by any path that mutates the rows file. Exactly one re-render is driven
  # off it, so a stale report can never be committed beside edited rows.
  ROWS_EDITED=0
  if [ "${LADDER_BENCHMARK_GRANULAR:-}" = "1" ]; then
    # One BENCHMARK per process, not just one model. The model reload costs ~10s
    # per cell; against a multi-hour sweep that is trivial insurance against the
    # MPS allocator accumulating across benchmarks inside one process, which has
    # already OOM'd this machine once (42.44 GiB of allocations for a 0.6B model
    # at 0.8 GB RSS).
    first=truncate
    BENCH_INDEX=0
    for benchmark in $BENCHMARKS; do
      BENCH_INDEX=$((BENCH_INDEX + 1))
      # Checked BEFORE the encode, not only after it. The 2026-07-29 kill landed
      # *inside* a cell, so a purely post-cell guard never got its turn to run.
      check_memory || { code=9; break; }
      log "  -- $model on $benchmark"
      measure_cell "$benchmark" "$first"
      cell_code=$?
      first=append
      if [ $cell_code -ne 0 ]; then
        code=$cell_code
        # Name the cell that died. Without this the failure path condemns every
        # benchmark for this model, including the ones that measured fine.
        # Exit 3 is excluded on purpose: a cache-integrity refusal means the
        # ROWS are fine and the cache is suspect, so recording it as a death
        # would delete good rows -- the trap the exit-3 branch below exists to
        # avoid.
        [ $cell_code -ne 3 ] && FAILED_BENCHMARKS="$FAILED_BENCHMARKS $benchmark"
      fi
      # A cache-integrity refusal must stop immediately, not after the remaining
      # benchmarks have each recorded their own failure row.
      [ $cell_code -eq 3 ] && break
      # Only guard if something still has to run. After the final cell of the
      # final model there is nothing left to protect, and exiting 9 there costs
      # the write-up for a sweep that completed.
      if [ $BENCH_INDEX -lt "$N_BENCHMARKS" ] || [ $MODEL_INDEX -lt "$N_MODELS" ]; then
        check_swap || { code=9; break; }
      fi
    done
  else
    measure_cell "$BENCHMARKS" truncate
    code=$?
    # Non-granular: one process covers every benchmark, so all of them are lost.
    [ $code -ne 0 ] && FAILED_BENCHMARKS="$BENCHMARKS"
    # Same rule as the granular branch: only abort if another model still runs.
    if [ $MODEL_INDEX -lt "$N_MODELS" ]; then
      check_swap || code=9
    fi
  fi

  # Flush cell deaths BEFORE the terminal branches below, both of which return
  # without reaching the ordinary failure path. `code` gets overwritten by the
  # terminal 9 (memory) or 3 (cache refusal), so a cell that genuinely died
  # earlier in the loop left no trace -- and the next re-run then saw stale
  # SUCCESSFUL rows for a cell that never re-ran. Scoped to the benchmarks that
  # actually failed, and code-3 refusals are already excluded from that list, so
  # this cannot delete a good row. (Cross-model review.)
  if [ -n "${FAILED_BENCHMARKS// /}" ] && { [ $code -eq 9 ] || [ $code -eq 3 ]; }; then
    log "$model: recording cell failures that preceded the stop:$FAILED_BENCHMARKS"
    record_process_failure "$model" "$code" "$FAILED_BENCHMARKS"
    # This EDITED the rows file, so the report beside it is now stale and both
    # terminal branches below must go through the ordinary render+commit path
    # instead of exiting past it. Before this flag, exit 3 returned without
    # staging anything -- the one failure record it had just written died with
    # the worktree -- and exit 9 committed the edited rows under a report
    # rendered before the edit. (Cross-model review.)
    ROWS_EDITED=1
  fi

  if [ $code -eq 9 ]; then
    log "$model: stopping the sweep on memory pressure. Rows measured so far are committed below."
  fi

  # A cache-integrity refusal is NOT a model failure, and must not be recorded as
  # one: record_process_failure deletes every existing row for the model and the
  # commit below ships the deletion. Nothing is wrong with those rows -- the cache
  # is what is suspect -- so this branch records nothing of its own. The harness
  # reserves this code for exactly that (embedder_ladder.py::EXIT_CACHE_INTEGRITY).
  #
  # It does NOT exit here. Earlier granular cells may have died for unrelated
  # reasons and been flushed above, and exiting before the commit block threw that
  # record away -- the refusal branch's own "leave the artifacts untouched"
  # promise had already been broken by the flush that runs before it. The exit
  # moved next to the code-9 exit, after the commit. (Cross-model review.)
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
  fi

  if [ $code -ne 0 ] && [ $code -ne 9 ] && [ $code -ne 3 ]; then
    log "$model exited $code -- recording a failure row for:${FAILED_BENCHMARKS:- $BENCHMARKS}"
    record_process_failure "$model" "$code" "${FAILED_BENCHMARKS:-$BENCHMARKS}"
    ROWS_EDITED=1
  fi

  # ONE re-render, driven by "were the rows edited", not by which code path
  # edited them. Previously only the ordinary failure path rendered, so the
  # terminal paths committed edited rows under a report generated before the
  # edit. The paths and the ladder set must match the measuring call above:
  # rendering the DEFAULT artifact here would rewrite the 2026-07-27 ladder's
  # report from a different study's rows.
  if [ $ROWS_EDITED -eq 1 ]; then
    uv run python examples/research/embedder_ladder.py --render-only \
      --rows "$ROWS" --report "$REPORT" --reference "$REFERENCE" \
      --ladder-models "${LADDER_ACCOUNTABLE[@]}" \
      ${REFERENCE_MODEL_ARG[@]+"${REFERENCE_MODEL_ARG[@]}"} \
      >> "$LOG_DIR/$safe.log" 2>&1
  fi

  # Stage only the artifacts that EXIST. `git add` fails atomically on a
  # pathspec matching nothing, so naming an absent sidecar staged NONE of the
  # three -- and with its status discarded, the next check read "no change" and
  # the run was never committed. On a fresh LADDER_ARTIFACT whose first model
  # dies before Python writes the reference sidecar, that silently threw away
  # the very failure record this path exists to preserve. (Cross-model review.)
  ARTIFACT_PATHS=()
  for artifact_path in "$ROWS" "$REPORT" "$REFERENCE"; do
    [ -e "$artifact_path" ] && ARTIFACT_PATHS+=("$artifact_path")
  done
  if [ ${#ARTIFACT_PATHS[@]} -eq 0 ]; then
    log "FATAL: $model produced none of the tracked artifacts -- nothing to commit; stopping."
    exit 1
  fi
  if ! git add "${ARTIFACT_PATHS[@]}"; then
    log "FATAL: could not stage ${ARTIFACT_PATHS[*]}; stopping rather than losing the result."
    exit 1
  fi
  # Path-scoped, like the commit below: an operator's unrelated staged work
  # would otherwise read as "this model produced results".
  if git diff --cached --quiet -- "${ARTIFACT_PATHS[@]}"; then
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
    if ! git commit -q --only "${ARTIFACT_PATHS[@]}" \
      -m "results($(basename "$ARTIFACT")): $model" \
      -m "Measured by examples/research/run_ladder.sh, committed as soon as the rows existed. Exit code $code."
    then
      log "FATAL: commit failed for $model. Results are staged but NOT durable; stopping."
      exit 1
    fi
    log "committed $model"
    push_results
  fi

  # Committed first, THEN stop: the whole point of aborting on memory pressure or
  # a cache refusal is to keep what was measured, and exiting before the commit
  # above would throw away exactly the rows the abort was protecting.
  if [ $code -eq 9 ]; then
    log "stopping: memory pressure. Re-run this driver to resume from the committed rows."
    exit 9
  fi
  if [ $code -eq 3 ]; then
    log "stopping: cache-integrity refusal (recovery command logged above)."
    exit 3
  fi
done

log "ladder complete"
