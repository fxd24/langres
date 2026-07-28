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

say() { echo "[$(date '+%H:%M:%S')] lfm25: $*"; }

# ---------------------------------------------------------------------------
# Study A -- the like-for-like comparison.
#
# The reference model MUST run first: the paired-CI sidecar it writes is what
# every later model's interval is computed against, and run_ladder.sh orders the
# list as given.
# ---------------------------------------------------------------------------
say "study A (retrieval-tuned) -- baseline intfloat/e5-base-v2"
LADDER_ARTIFACT="docs/research/20260729_lfm25_tuned" \
LADDER_REFERENCE_MODEL="intfloat/e5-base-v2" \
LADDER_MODELS="intfloat/e5-base-v2 all-MiniLM-L6-v2 BAAI/bge-base-en-v1.5 LiquidAI/LFM2.5-Embedding-350M" \
  bash examples/research/run_ladder.sh "${1:-}"
code=$?
if [ $code -ne 0 ]; then
  say "study A exited $code -- stopping before study B (its baseline comes from A's checkpoint)"
  exit $code
fi

# ---------------------------------------------------------------------------
# Study B -- the base encoders, against the tuned model on the same backbone.
# ---------------------------------------------------------------------------
say "study B (base masked-LM encoders) -- baseline LiquidAI/LFM2.5-Embedding-350M"
LADDER_ARTIFACT="docs/research/20260729_lfm25_base_encoders" \
LADDER_REFERENCE_MODEL="LiquidAI/LFM2.5-Embedding-350M" \
LADDER_MODELS="LiquidAI/LFM2.5-Embedding-350M LiquidAI/LFM2.5-Encoder-350M LiquidAI/LFM2.5-Encoder-230M" \
  bash examples/research/run_ladder.sh
code=$?
if [ $code -ne 0 ]; then
  say "study B exited $code"
  exit $code
fi

# ---------------------------------------------------------------------------
# The write-up, generated from both rows files. Never hand-typed: three PRs
# shipped factual errors on one day and every one was in hand-typed prose while
# the generated tables beside them were correct.
# ---------------------------------------------------------------------------
say "rendering the write-up"
uv run python examples/research/lfm25_report.py || exit 1

git add docs/research/20260729_lfm25_encoders.md
if ! git diff --cached --quiet -- docs/research/20260729_lfm25_encoders.md; then
  git commit -q --only docs/research/20260729_lfm25_encoders.md \
    -m "results(lfm25): generated write-up" \
    -m "Rendered by examples/research/lfm25_report.py from the two committed rows files."
  git push -q origin HEAD 2>/dev/null && say "pushed" || say "push failed"
fi
say "complete"
