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

# Shared publication rule (publish_lib.sh), the same one run_ladder.sh and
# run_lfm25.sh use. Four copies of "push, but never onto the default branch" had
# drifted in what they printed and in whether they ran at all.
# shellcheck source=examples/research/publish_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/publish_lib.sh"

export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# A missing or already-closed sidecar must stop publication, not pass silently.
export LADDER_PROVENANCE_REQUIRED=1

STUDY_A_MODELS="intfloat/e5-base-v2 all-MiniLM-L6-v2 BAAI/bge-base-en-v1.5 LiquidAI/LFM2.5-Embedding-350M"
PROVENANCE_JSON="docs/research/20260729_lfm25_provenance.json"
REPORT_MD="docs/research/20260729_lfm25_encoders.md"

say() { echo "[$(date '+%H:%M:%S')] resume: $*"; }

# `--only`, never a bare `git commit`. A bare commit takes the whole INDEX, so
# anything the operator had staged before starting -- unrelated WIP, a secret
# pasted into a scratch file -- rides along inside a "chore(provenance)" commit
# and is then pushed by the child driver. run_ladder.sh and run_lfm25.sh already
# commit this way for exactly that reason; this script was the one site that did
# not. Returns non-zero on failure so callers can decide, rather than warning.
# (Cross-model review.)
commit_only() {
  local message="$1" body="$2"
  shift 2
  git add "$@" || {
    say "could not stage: $*"
    return 1
  }
  git diff --cached --quiet -- "$@" && return 0
  git commit -q --only "$@" -m "$message" -m "$body" || {
    say "commit failed for: $*"
    return 1
  }
}

# Ask the DRIVER whether it would refuse, BEFORE replacing the shared sidecar.
#
# Opening the window first looked harmless -- the driver refuses a moment later
# and nothing is measured. It is not: `--start` REPLACES the sidecar, and the
# window it replaces is the one describing every row already committed. So a
# resume run in exactly the situation this script exists for (a killed sweep,
# dirty rows) swapped real provenance for a no-op window and committed and
# PUBLISHED it, having measured nothing.
#
# LADDER_PREFLIGHT_ONLY runs the driver's own check and stops. Not a copy of it:
# a second implementation of one safety rule is the drift this repo keeps paying
# for. (Cross-model review.)
LADDER_ARTIFACT="docs/research/20260729_lfm25_tuned" \
LADDER_REFERENCE_MODEL="intfloat/e5-base-v2" \
LADDER_ALL_MODELS="$STUDY_A_MODELS" \
LADDER_MODELS="LiquidAI/LFM2.5-Embedding-350M" \
LADDER_BENCHMARKS="walmart_amazon" \
LADDER_PREFLIGHT_ONLY=1 \
  bash examples/research/run_ladder.sh || {
  say "FATAL: the study artifacts are not in a state this resume may overwrite."
  say "  Nothing was measured and the provenance sidecar is UNTOUCHED."
  exit 2
}

# --start refuses when the sidecar holds uncommitted work, which is the state a
# killed sweep leaves behind and exactly what must not be overwritten silently.
uv run python examples/research/write_provenance.py --start --studies a || {
  say "FATAL: could not open a provenance window; not measuring."
  exit 1
}
commit_only "chore(lfm25): open a provenance window for the study-A resume" \
  "Captured on the measurement path, before the cell is re-measured. A resume runs after the original window closed, and a closed window cannot describe rows produced after it." \
  "$PROVENANCE_JSON" || {
  say "FATAL: the open window is not committed; it would die with the worktree."
  exit 1
}

# LADDER_NO_PUSH: this resume owns publication, and it cannot decide until the
# window is closed and the write-up re-rendered. Left to publish for itself, the
# driver pushed the row the moment it was committed -- so a later --finish
# rejection or render failure left origin holding the new row beside an OPEN
# provenance window, with the closing evidence withheld and this script printing
# "everything is committed locally". Suppressing the child's push is what makes
# the single publish at the end actually mean something. (Cross-model review.)
LADDER_ARTIFACT="docs/research/20260729_lfm25_tuned" \
LADDER_REFERENCE_MODEL="intfloat/e5-base-v2" \
LADDER_ALL_MODELS="$STUDY_A_MODELS" \
LADDER_MODELS="LiquidAI/LFM2.5-Embedding-350M" \
LADDER_BENCHMARKS="walmart_amazon" \
LADDER_NO_PUSH=1 \
  bash examples/research/run_ladder.sh
code=$?

# --partial ALWAYS: this window covers one cell of one study, never the rest.
# Closed even on failure -- an open window committed beside durable rows is the
# state this whole mechanism exists to avoid.
#
# The status is KEPT, not swallowed. --finish exits non-zero when measurement
# code moved mid-cell, which is the same condition that makes run_ladder.sh
# withhold its push. Discarding it here re-enabled at the end exactly what the
# child had deliberately refused: the final `git push` published commits that
# provenance verification had rejected. The evidence is still committed -- that
# is why this does not exit immediately. (Cross-model review.)
uv run python examples/research/write_provenance.py --finish --partial || {
  say "provenance --finish REJECTED this run; its evidence will be committed, nothing published"
  [ "$code" -eq 0 ] && code=1
}

# Re-render the COMBINED write-up. run_ladder.sh regenerates only the tuned
# study's own report; 20260729_lfm25_encoders.md states on its face that it is
# generated from both committed rows files, so leaving it untouched beside a
# newly committed row publishes a document contradicting its own source -- and
# the noise-floor table subtracts across the two studies, so a changed study-A
# cell can move it. Rendered AFTER --finish, matching run_lfm25.sh: the report
# quotes the window, so it must read the closed one. (Cross-model review.)
#
# COMMIT_PATHS is why the render result is tracked separately: the writer guard
# refuses to overwrite a hand-edited write-up, and passing REPORT_MD to
# `git commit --only` regardless would have committed that very hand edit under a
# message claiming it was re-rendered -- publishing the bytes the guard had just
# refused to touch. The report is committed only if this process wrote it.
# (Cross-model review.)
COMMIT_PATHS=("$PROVENANCE_JSON")
if [ "$code" -eq 0 ]; then
  say "re-rendering the combined write-up"
  # Same allowance as run_lfm25.sh: --finish wrote the sidecar just above and the
  # commit below carries it with the report.
  if LFM25_PROVENANCE_PENDING=1 uv run python examples/research/lfm25_report.py; then
    COMMIT_PATHS+=("$REPORT_MD")
  else
    say "the write-up did not re-render; committing the closed window WITHOUT it"
    code=4
  fi
fi

# The closed window is the only record of what produced the re-measured row. If
# it cannot be committed it is not durable, and reporting success would tell
# automation the resume completed while the evidence dies with the worktree.
# Previously this only warned and then exited with the child's zero status.
# (Cross-model review.)
commit_only "results(lfm25): close the study-A resume's window" \
  "Marked partial: the window describes the re-measured cell only; every other row in both studies predates it. The combined write-up is included only when this run actually re-rendered it." \
  "${COMMIT_PATHS[@]}" || {
  say "FATAL: the closed window is not durable; failing rather than reporting success."
  exit 1
}

# run_ladder.sh already pushed the re-measured ROW. Without this, a successful
# resume left the remote holding that row beside an OPEN provenance window and a
# stale combined write-up -- the published state contradicting itself -- until
# someone noticed and pushed by hand. Same branch guard as run_ladder.sh's
# push_results: never onto the default branch, where `git push origin HEAD` would
# publish generated results straight past the PR-only rule. (Cross-model review.)
publish() { publish_branch "resume"; }

# Publish only a run that fully succeeded. run_ladder.sh already withholds its own
# push when --verify fails; pushing unconditionally here re-published exactly what
# the child had refused. Everything is COMMITTED either way -- durability is never
# the thing withheld, only publication. (Cross-model review.)
if [ "$code" -eq 0 ]; then
  publish
else
  say "NOT publishing: this resume exited $code. Everything is committed locally."
  say "  Inspect the closed provenance window and the rows, then push deliberately."
fi

exit $code
