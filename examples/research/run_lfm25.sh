#!/usr/bin/env bash
# Measure the LiquidAI LFM2.5 encoders on ER candidate blocking.
#
# Two studies, because they answer two different questions and must not share a
# ranking table:
#
#   A. tuned   -- LFM2.5-Embedding-350M against the models langres actually
#                 ships or shipped: `intfloat/e5-base-v2` (the current
#                 DEFAULT_EMBEDDING_MODEL, and therefore the baseline every
#                 delta is measured against), `BAAI/bge-base-en-v1.5` and
#                 `all-MiniLM-L6-v2`. A like-for-like comparison of
#                 retrieval-tuned embedders.
#
#   B. base    -- LFM2.5-Encoder-350M and -230M, which are base masked-LM
#                 encoders, NOT embedding models. They ship no pooling config,
#                 so sentence-transformers attaches an UNTRAINED mean-pooling
#                 head. Their baseline is LFM2.5-Embedding-350M, which shares the
#                 350M checkpoint's backbone (both declare
#                 base_model: LiquidAI/LFM2.5-350M-Base), so the delta reads as
#                 "what retrieval tuning bought on this backbone" rather than
#                 "which model is better". Never merge these rows into study A.
#
#                 Study B also carries `random-init-control-350M`: the same
#                 architecture with SEEDED RANDOM WEIGHTS. It is the noise floor,
#                 and it is here because a randomly initialised 350M backbone
#                 scored 0.9911 recall / 0.9971 AUC on fodors_zagat. A benchmark
#                 where the control lands near the tuned models cannot separate a
#                 trained retriever from a random feature map, and no embedder
#                 claim may rest on it. Running it on all five turns that from an
#                 anecdote into a per-benchmark verdict on the portfolio.
#
# Both write their own rows/report/sidecar via LADDER_ARTIFACT, and both share
# one embedding cache -- so LFM2.5-Embedding-350M, which appears in both, is
# encoded once and read from cache the second time.
#
# Strictly sequential: two concurrent `uv run` in one worktree race on .venv and
# fabricate failures. run_ladder.sh commits after every model, so a teardown
# mid-sweep loses nothing.
#
# Usage:
#   bash examples/research/run_lfm25.sh [PID_TO_WAIT_FOR]
#
#   LFM25_STUDY=a|b|both   which study to run (default: both)
#
# The selector exists because the harness has NO skip-completed logic:
# merge_rows() REPLACES a re-measured cell, and re-measuring the reference model
# clears every other model's vs_reference_* on the benchmarks it touches. So
# after a partial run, "just run it again" is not a resume -- it is a re-do that
# churns correct results. Naming the unfinished study is.

set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

# Not optional, and not inherited: `.env` is gitignored and absent from a fresh
# worktree. torch, faiss and scikit-learn each bundle their own libomp, and with
# two runtimes loaded a sweep DEADLOCKS in __kmp_join_barrier at 0% CPU with no
# error and no output -- measured at 3.5h of silence before it was noticed.
# KMP_DUPLICATE_LIB_OK alone suppresses the abort but NOT the deadlock;
# OMP_NUM_THREADS=1 is the load-bearing one.
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# One BENCHMARK per subprocess, not just one model. ~10s of model reload per cell
# against a multi-hour sweep, in exchange for the MPS allocator being unable to
# accumulate across benchmarks inside a single process. It has already OOM'd this
# machine once, at 42.44 GiB of allocations for a 0.6B model whose RSS read
# 0.8 GB -- RSS cannot see this, so the driver samples swap per cell instead and
# aborts on a monotonic climb.
export LADDER_BENCHMARK_GRANULAR=1

# The force decision travels to run_ladder.sh, which carries the same guard over
# the artifacts it actually writes. Set once here rather than on each invocation.
[ "${LFM25_FORCE:-0}" = "1" ] && export LADDER_FORCE=1

say() { echo "[$(date '+%H:%M:%S')] lfm25: $*"; }

PROVENANCE_JSON="docs/research/20260729_lfm25_provenance.json"

# Close and COMMIT the provenance window before bailing out.
#
# An expected abort -- memory pressure (9), a cache refusal (3), any child
# failure -- used to `exit` straight from the study branch. By then `--start` had
# already overwritten the sidecar and run_ladder.sh had committed (and possibly
# pushed) partial rows, so the only snapshot describing those now-durable rows was
# an uncommitted file that dies with the worktree, or is silently replaced by the
# next `--start`. That is exactly the "commit before the worktree disappears"
# failure this repo has already paid for once. The window is closed, marked
# partial, and committed on the way out. (Cross-model review.)
commit_provenance() {
  local code="$1" subject="$2"
  if [ -f "$PROVENANCE_JSON" ]; then
    # A WARNING here is not enough. Partial rows are already committed and may be
    # pushed, so this snapshot is the only description of what produced them --
    # and it lives in a gitignored-adjacent working file that dies with the
    # worktree. Warning and exiting with the child's status makes the failure
    # look like the sweep's, and a caller checking only the exit code cannot tell
    # "the sweep aborted" from "the sweep aborted AND its provenance was lost".
    # Exit 10 says the second thing loudly. (Cross-model review.)
    if ! git add "$PROVENANCE_JSON"; then
      say "FATAL: could not stage the run's provenance. It is the ONLY record of"
      say "  what produced the rows already committed. Save it before teardown:"
      say "    cp $PROVENANCE_JSON <somewhere outside this worktree>"
      exit 10
    fi
    if ! git diff --cached --quiet -- "$PROVENANCE_JSON"; then
      if ! git commit -q --only "$PROVENANCE_JSON" \
        -m "$subject" \
        -m "Rows already committed by run_ladder.sh are described by this window."; then
        say "FATAL: the provenance is staged but NOT committed, and it is the"
        say "  ONLY record of what produced the rows already committed. Save it before"
        say "  teardown:  cp $PROVENANCE_JSON <somewhere outside this worktree>"
        exit 10
      fi
    fi
  fi
  exit "$code"
}

abort_with_provenance() {
  local code="$1"
  say "closing the provenance window for the partial run"
  # --finish exits non-zero when measurement code moved mid-sweep; that verdict
  # still belongs in the sidecar, so record it and keep the original exit code.
  # --partial: this window did NOT reach every planned study, so the report must
  # not read `studies_measured` as "every row in both studies".
  uv run python examples/research/write_provenance.py --finish --partial ||
    say "provenance --finish reported a problem; recording it and continuing to commit"
  commit_provenance "$code" "results(lfm25): provenance for a partial sweep (exit $code)"
}

STUDY="${LFM25_STUDY:-both}"
case "$STUDY" in
  a | b | both) ;;
  *)
    say "LFM25_STUDY must be one of: a, b, both (got '$STUDY')"
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# Wait for the in-flight sweep FIRST -- before --start, not inside run_ladder.sh.
#
# The documented invocation is `run_lfm25.sh PID`, and the PID used to be handed
# straight to run_ladder.sh, which waits. But `--start` runs before that: it
# OVERWROTE the shared sidecar while the previous sweep was still measuring, so
# the older process went on to commit and push rows, then verified or closed a
# window describing the replacement run -- rows published under provenance for
# code they never touched. `--start` is also itself a `uv run`, and two of those
# in one worktree do not fail loudly, they invent failures. The wait belongs
# ahead of both. (Cross-model review.)
#
# Still passed down: run_ladder.sh is also run standalone, its own wait is then
# the only one, and a second wait on a PID that has already exited returns at
# once.
# ---------------------------------------------------------------------------
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  say "waiting for pid $WAIT_PID to exit before opening the provenance window"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 15
  done
  say "pid $WAIT_PID is gone"
  sleep 10
fi

# ---------------------------------------------------------------------------
# Provenance is captured HERE, in the measurement path, before a single row is
# written -- not at render time. A provenance line derived from HEAD names
# whatever is checked out when the report is generated, which is not what
# measured the rows; the first commit touching the harness would silently
# reattribute every row to code that never ran. Not stale: false.
# ---------------------------------------------------------------------------
case "$STUDY" in
  both) PROVENANCE_STUDIES="a b" ;;
  *) PROVENANCE_STUDIES="$STUDY" ;;
esac

# ---------------------------------------------------------------------------
# The sidecar is not the only thing a new sweep overwrites.
#
# A granular sweep killed after a benchmark subprocess wrote rows but before the
# model-level commit leaves the ROWS, the REPORT and the REFERENCE sidecar dirty
# while the provenance file is clean -- it was committed at startup. run_ladder.sh
# then rewrites and commits all three under the new window, so measured-but-
# unsaved cells, and any hand edit, are published as this run's or replaced
# outright. Same rule as the provenance guard, same escape. (Cross-model review.)
# ---------------------------------------------------------------------------
study_artifacts() {
  case "$1" in
    a) echo "docs/research/20260729_lfm25_tuned" ;;
    b) echo "docs/research/20260729_lfm25_base_encoders" ;;
  esac
}
DIRTY_ARTIFACTS=""
for study in $PROVENANCE_STUDIES; do
  prefix="$(study_artifacts "$study")"
  for artifact in "${prefix}_rows.jsonl" "${prefix}.md" "${prefix}_reference_recall.json"; do
    # No `-e` guard: git already distinguishes a never-created path (silence)
    # from an uncommitted DELETION (" D"/"D "), and the existence test skipped
    # exactly the second one. A pending deletion of the rows file let a new
    # sweep recreate it from empty and commit a partial replacement over every
    # other expensive result, with no refusal. (Cross-model review.)
    if [ -n "$(git status --porcelain --untracked-files=all -- "$artifact")" ]; then
      DIRTY_ARTIFACTS="$DIRTY_ARTIFACTS $artifact"
    fi
  done
done
if [ -n "${DIRTY_ARTIFACTS// /}" ] && [ "${LFM25_FORCE:-0}" != "1" ]; then
  say "REFUSING to start: these study artifacts have uncommitted changes:"
  for artifact in $DIRTY_ARTIFACTS; do say "    $artifact"; done
  say "  A killed sweep leaves measured cells here that no commit holds. This run would"
  say "  rewrite and commit them under its own provenance. Commit them, or copy them"
  say "  outside this worktree, then re-run. To discard them deliberately:"
  say "    LFM25_FORCE=1 LFM25_STUDY=$STUDY bash $0${1:+ $1}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Provenance is captured HERE, in the measurement path, before a single row is
# written -- not at render time. A provenance line derived from HEAD names
# whatever is checked out when the report is generated, which is not what
# measured the rows; the first commit touching the harness would silently
# reattribute every row to code that never ran. Not stale: false.
# ---------------------------------------------------------------------------
say "opening the provenance window (studies: $STUDY)"
# LFM25_FORCE forwards to --force. Without it the advertised escape was a DEAD
# END: running `write_provenance.py --start --force` by hand opens a window and
# exits, leaving the sidecar dirty, so the very next `run_lfm25.sh` refused the
# file it had just been told to write. The force decision has to travel with the
# run that acts on it. (Cross-model review.)
FORCE_ARG=""
[ "${LFM25_FORCE:-0}" = "1" ] && FORCE_ARG="--force"
# shellcheck disable=SC2086  # word splitting is the interface here
uv run python examples/research/write_provenance.py --start $FORCE_ARG --studies $PROVENANCE_STUDIES || {
  # --start refuses when the existing sidecar has uncommitted changes: a run
  # killed between --finish and its commit leaves the ONLY description of rows
  # that ARE committed sitting in that file, and opening a window would replace
  # it with no way back. The refusal is a stop, not a dead end -- say what the
  # two ways out are rather than leaving a bare exit code.
  say "could not open the provenance window."
  say "  If it refused because $PROVENANCE_JSON has uncommitted changes, those bytes may"
  say "  be the only record of rows a killed run already committed. Commit them, or"
  say "  copy the file outside this worktree, then re-run. To discard them deliberately:"
  say "    LFM25_FORCE=1 LFM25_STUDY=$STUDY bash $0${1:+ $1}"
  exit 1
}

# Committed BEFORE the first measurement, not only at the end.
#
# --start overwrites a TRACKED file. run_ladder.sh then commits, and may push,
# each model's rows as they are produced. If the outer process is killed before
# the graceful finish/abort path -- an OS kill is exactly how this study lost a
# sweep once -- those rows are durable and published while the only snapshot
# describing their measurement window is an uncommitted working file that dies
# with the worktree, leaving the rows paired with the PREVIOUS study's
# provenance. Opening the window durably costs one commit. (Cross-model review.)
git add "$PROVENANCE_JSON" || {
  say "FATAL: could not stage the opened provenance window"
  exit 1
}
if ! git diff --cached --quiet -- "$PROVENANCE_JSON"; then
  git commit -q --only "$PROVENANCE_JSON" \
    -m "results(lfm25): open the provenance window (studies: $STUDY)" \
    -m "Committed before the first measurement so rows published mid-sweep are never paired with a previous window." ||
    {
      say "FATAL: could not commit the opened provenance window"
      exit 1
    }
fi

# ---------------------------------------------------------------------------
# Study A -- the like-for-like comparison.
#
# The reference model MUST run first: the paired-CI sidecar it writes is what
# every later model's interval is computed against, and run_ladder.sh orders the
# list as given.
# ---------------------------------------------------------------------------
if [ "$STUDY" = "a" ] || [ "$STUDY" = "both" ]; then
  say "study A (retrieval-tuned) -- baseline intfloat/e5-base-v2"
  LADDER_ARTIFACT="docs/research/20260729_lfm25_tuned" \
  LADDER_REFERENCE_MODEL="intfloat/e5-base-v2" \
  LADDER_MODELS="intfloat/e5-base-v2 all-MiniLM-L6-v2 BAAI/bge-base-en-v1.5 LiquidAI/LFM2.5-Embedding-350M" \
  LADDER_ALL_MODELS="intfloat/e5-base-v2 all-MiniLM-L6-v2 BAAI/bge-base-en-v1.5 LiquidAI/LFM2.5-Embedding-350M" \
    bash examples/research/run_ladder.sh "${1:-}"
  code=$?
  if [ $code -ne 0 ]; then
    say "study A exited $code -- stopping before study B (its baseline comes from A's checkpoint)"
    abort_with_provenance $code
  fi
fi

# ---------------------------------------------------------------------------
# Study B -- the base encoders, against the tuned model on the same backbone.
# ---------------------------------------------------------------------------
if [ "$STUDY" = "b" ] || [ "$STUDY" = "both" ]; then
  say "study B (base masked-LM encoders) -- baseline LiquidAI/LFM2.5-Embedding-350M"
  LADDER_ARTIFACT="docs/research/20260729_lfm25_base_encoders" \
  LADDER_REFERENCE_MODEL="LiquidAI/LFM2.5-Embedding-350M" \
  LADDER_MODELS="LiquidAI/LFM2.5-Embedding-350M LiquidAI/LFM2.5-Encoder-350M LiquidAI/LFM2.5-Encoder-230M random-init-control-350M" \
  LADDER_ALL_MODELS="LiquidAI/LFM2.5-Embedding-350M LiquidAI/LFM2.5-Encoder-350M LiquidAI/LFM2.5-Encoder-230M random-init-control-350M" \
    bash examples/research/run_ladder.sh "$([ "$STUDY" = "b" ] && echo "${1:-}")"
  code=$?
  if [ $code -ne 0 ]; then
    say "study B exited $code"
    abort_with_provenance $code
  fi
fi

# ---------------------------------------------------------------------------
# Refresh the load probe INSIDE the measurement window, before the render.
#
# The write-up's "was the checkpoint even the thing measured?" section is built
# from this probe: which class transformers instantiates, how many weights fail
# to load, whether the untrusted load collapses every text onto one vector. All
# of that is a property of the INSTALLED library, and a rerun after a dependency
# bump used to commit fresh rows beside loading claims captured under a different
# transformers -- two artifacts, two lifetimes, no link. The documented reproduce
# block made it worse by listing the probe AFTER the render.
#
# Non-fatal: the probe downloads checkpoints and can fail offline or on a rate
# limit, and killing a completed sweep over its explanatory section would trade a
# large loss for a small one. The report cross-checks the recorded transformers
# version against the installed one and says so when they differ, so a probe that
# could not be refreshed is disclosed rather than passed off as current.
# (Cross-model review.)
# ---------------------------------------------------------------------------
say "refreshing the load probe"
LOAD_PROBE_JSON="docs/research/20260729_lfm25_load_probe.json"
if uv run python examples/research/lfm25_load_probe.py; then
  # Committed the moment it exists, not bundled into the final commit: an abort
  # between here and the render would otherwise leave the refreshed probe as
  # uncommitted output, which dies with the worktree.
  git add "$LOAD_PROBE_JSON" || {
    say "FATAL: could not stage the refreshed load probe"
    exit 1
  }
  if ! git diff --cached --quiet -- "$LOAD_PROBE_JSON"; then
    git commit -q --only "$LOAD_PROBE_JSON" \
      -m "results(lfm25): refresh the checkpoint load probe" \
      -m "Captured inside this sweep's measurement window, so the write-up's loading claims describe the same environment as its rows." ||
      {
        say "FATAL: the refreshed load probe is staged but NOT committed; stopping."
        exit 1
      }
  fi
else
  say "the load probe FAILED to refresh; the write-up will disclose it as captured under a different environment"
fi

# ---------------------------------------------------------------------------
# The write-up, generated from both rows files. Never hand-typed: three PRs
# shipped factual errors on one day and every one was in hand-typed prose while
# the generated tables beside them were correct.
# ---------------------------------------------------------------------------
# Close the provenance window BEFORE rendering: --finish refuses if a tracked
# file changed mid-sweep, in which case no single blob describes all the rows and
# the report must not claim one.
say "closing the provenance window"
# NOT `|| exit 1`. --finish writes the finished timestamp and the
# `changed_during_run` evidence and THEN exits non-zero, so bailing out here
# discarded the very record explaining that already-committed rows came from
# mixed code -- the one warning a reader of those rows would need. Routed through
# the close-and-commit path, which persists it. (Cross-model review.)
#
# `commit_provenance`, NOT `abort_with_provenance`: every selected study RAN. The
# abort helper re-invokes `--finish --partial`, which would overwrite the
# `window_complete=true` and the finished timestamp this call just wrote with
# `false` plus the note "the sweep ABORTED before finishing every planned study"
# -- a statement contradicted by the rows sitting beside it. The window is closed
# and complete; only the commit is still owed. (Cross-model review.)
uv run python examples/research/write_provenance.py --finish || {
  say "provenance --finish rejected this sweep; committing its evidence before stopping"
  commit_provenance 1 "results(lfm25): provenance for a sweep whose code moved mid-run"
}

say "rendering the write-up"
# Not a bare exit: --finish has already CLOSED the window and written the
# finished timestamp and stability verdict, and the model rows are committed. A
# render failure here (cohort validation rejecting the artifacts, say) would
# otherwise leave that closed sidecar uncommitted, to be lost on teardown or
# overwritten by the next --start. The rows would then be durable with no record
# of the window that produced them. (Cross-model review.) Same reason as above
# for committing rather than aborting: re-finishing would relabel a complete
# window as partial.
uv run python examples/research/lfm25_report.py || {
  say "rendering failed; committing the closed provenance window before stopping"
  commit_provenance 1 "results(lfm25): provenance for a sweep whose write-up failed to render"
}

REPORT_MD="docs/research/20260729_lfm25_encoders.md"
git add "$REPORT_MD" "$PROVENANCE_JSON" || {
  say "FATAL: could not stage the write-up or its provenance"
  exit 1
}
if ! git diff --cached --quiet -- "$REPORT_MD" "$PROVENANCE_JSON"; then
  # A silent commit failure here is the durability hole that has already cost
  # this repo a paid run: the script would print "complete" and exit 0 with the
  # write-up still uncommitted, and a worktree teardown would take it. Guarded,
  # exactly as run_ladder.sh already guards its own commit.
  if ! git commit -q --only "$REPORT_MD" "$PROVENANCE_JSON" \
    -m "results(lfm25): generated write-up" \
    -m "Rendered by examples/research/lfm25_report.py from the two committed rows files, with the provenance captured across the measurement window."
  then
    say "FATAL: commit failed. The write-up is staged but NOT durable; stopping."
    exit 1
  fi
  # Never onto the default branch: `git push origin HEAD` follows whatever is
  # checked out, so running this from `main` would publish generated results
  # straight there, past the PR-only guardrail. The commit above already made
  # them durable; only the publish is withheld. (Cross-model review.)
  BRANCH=$(git rev-parse --abbrev-ref HEAD)
  DEFAULT=$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null)
  DEFAULT=${DEFAULT#origin/}
  if [ "$BRANCH" = "${DEFAULT:-main}" ] || [ "$BRANCH" = "HEAD" ]; then
    say "NOT pushing: on '$BRANCH'. The write-up is COMMITTED; publish via a PR:"
    say "    git switch -c results/lfm25 && git push -u origin HEAD"
  elif git push -q origin HEAD 2>/dev/null; then
    say "pushed"
  else
    say "push failed"
  fi
fi
say "complete"
