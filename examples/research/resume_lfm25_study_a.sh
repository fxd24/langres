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
# ...and why it opens its OWN provenance window. Routing through the driver was
# not enough: a resume runs after the original sweep has closed, and a closed
# window is a deliberate no-op for `--verify` (that is how the standing portfolio
# ladder, which has no provenance at all, is allowed to publish). So the
# re-measured row would have been committed and PUSHED with the only window
# describing it having ended before it was produced. A measurement run owns a
# window; this one opens, requires and closes its own. It is marked --partial
# because it re-measures one cell of one study: every other row predates it, and
# the write-up says so rather than implying this window covers them.
# (Cross-model review.)
#
# LADDER_ALL_MODELS still names all four study-A models even though only one is
# measured: that argument is the report's coverage DENOMINATOR, not the work
# list. Collapsing it to the single measured model is a regression this harness
# has already shipped once.
set -u

export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# A missing or already-closed sidecar must stop publication, not pass silently.
export LADDER_PROVENANCE_REQUIRED=1

STUDY_A_MODELS="intfloat/e5-base-v2 all-MiniLM-L6-v2 BAAI/bge-base-en-v1.5 LiquidAI/LFM2.5-Embedding-350M"
PROVENANCE_JSON="docs/research/20260729_lfm25_provenance.json"

say() { echo "[$(date '+%H:%M:%S')] resume: $*"; }

# --start refuses when the sidecar holds uncommitted work, which is the state a
# killed sweep leaves behind and exactly what must not be overwritten silently.
uv run python examples/research/write_provenance.py --start --studies a || {
  say "FATAL: could not open a provenance window; not measuring."
  exit 1
}
git add "$PROVENANCE_JSON" && git commit --quiet -m "chore(lfm25): open a provenance window for the study-A resume" \
  -m "Captured on the measurement path, before the cell is re-measured. A resume runs after the original window closed, and a closed window cannot describe rows produced after it." || {
  say "FATAL: the open window is not committed; it would die with the worktree."
  exit 1
}

LADDER_ARTIFACT="docs/research/20260729_lfm25_tuned" \
LADDER_REFERENCE_MODEL="intfloat/e5-base-v2" \
LADDER_ALL_MODELS="$STUDY_A_MODELS" \
LADDER_MODELS="LiquidAI/LFM2.5-Embedding-350M" \
LADDER_BENCHMARKS="walmart_amazon" \
  bash examples/research/run_ladder.sh
code=$?

# --partial ALWAYS: this window covers one cell of one study, never the rest.
# Closed even on failure -- an open window committed beside durable rows is the
# state this whole mechanism exists to avoid.
uv run python examples/research/write_provenance.py --finish --partial || \
  say "the provenance window could not be closed cleanly; committing it as it stands"
git add "$PROVENANCE_JSON" && git commit --quiet -m "chore(lfm25): close the study-A resume's provenance window" \
  -m "Marked partial: it describes the re-measured cell only. Every other row in both studies predates it." || \
  say "WARNING: the closed window is not committed"

exit $code
