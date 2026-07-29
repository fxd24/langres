# One implementation of "publish this branch, never onto the default one".
#
# Sourced, not executed. Three drivers needed this rule -- run_ladder.sh after each
# model, resume_lfm25_study_a.sh after it closes its window, run_lfm25.sh when it
# aborts with provenance -- and for a while each had its own copy or, worse, no copy
# at all. That asymmetry was a real published-state bug twice in this series: rows
# reached origin while the provenance window describing them stayed local, because
# the pusher and the thing that needed pushing lived in different files with
# different rules. One function, one rule.
#
# Why the default-branch check exists: `git push origin HEAD` follows whatever is
# checked out, so running any of these drivers from `main` would publish generated
# results straight there, past the PR-only guardrail this repo works under.
#
# Returns 0 when the branch was pushed, 1 when publication was deliberately
# withheld or failed. Never exits: publication is not durability, and a caller
# whose commits already landed must be free to carry on.

# Where a withheld-publication verdict is recorded, so it outlives the process
# that reached it.
#
# It has to be a FILE, not a shell variable. The condition is a property of the
# BRANCH -- `git push origin HEAD` sends every commit on it -- while the drivers
# that discover it and the drivers that publish are different PROCESSES:
# run_ladder.sh withholds a model's push, returns 0, and its wrapper
# (resume_lfm25_study_a.sh, run_lfm25.sh) then publishes at the end having no way
# to know. A variable cannot cross that boundary; an exit code would have to be
# threaded through two levels and re-interpreted at each. One marker, checked in
# the one function that pushes.
#
# Under tmp/ because it is gitignored: this is worktree state, not a tracked
# artifact. It is deliberately NOT cleared automatically -- the withheld commits
# are still on the branch, so the next sweep's push would publish them too. It
# fails CLOSED and says how to clear it.
_publication_block_file() {
  local root
  root=$(git rev-parse --show-toplevel 2>/dev/null) || return 1
  echo "$root/tmp/.publication-withheld"
}

# Record that nothing further may be published from this branch, and why.
# Idempotent: the first reason is the one that matters, and repeating the banner
# once per model buries it.
block_publication() {
  local reason="$1" marker
  marker=$(_publication_block_file) || {
    echo "[publish] PUBLICATION WITHHELD: $reason"
    echo "[publish]   (could not record it -- not in a git worktree)"
    return 0
  }
  [ -f "$marker" ] && return 0
  mkdir -p "$(dirname "$marker")" 2>/dev/null
  printf '%s\n' "$reason" > "$marker" 2>/dev/null
  echo "[publish] PUBLICATION WITHHELD for the rest of this sweep: $reason"
  echo "[publish]   Every result stays COMMITTED locally. \`git push\` sends the whole"
  echo "[publish]   branch, so a later successful push would republish what was withheld."
  echo "[publish]   Inspect the commits, then push deliberately and remove:"
  echo "[publish]     $marker"
}

publish_branch() {
  local label="${1:-publish}" branch default marker
  marker=$(_publication_block_file 2>/dev/null) || marker=""
  # Checked FIRST, before the branch rule and before any push: this is the one
  # place all three drivers publish from, so it is the only place the verdict has
  # to be honoured. (Cross-model review: the per-call refusal it replaces was
  # scoped to one process and one model, and both wrappers published anyway.)
  if [ -n "$marker" ] && [ -f "$marker" ]; then
    echo "[$label] NOT pushing: publication is withheld for this branch."
    echo "[$label]   $(head -1 "$marker")"
    echo "[$label]   Commits are LOCAL and safe. Clear $marker to re-enable."
    return 1
  fi
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || {
    echo "[$label] NOT pushing: cannot determine the branch."
    return 1
  }
  default=$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null)
  default=${default#origin/}
  default=${default:-main}
  if [ "$branch" = "$default" ] || [ "$branch" = "HEAD" ]; then
    echo "[$label] NOT pushing: on '$branch'. Commits are LOCAL and safe; publish via a PR."
    return 1
  fi
  if git push -q origin HEAD 2>/dev/null; then
    echo "[$label] pushed"
    return 0
  fi
  echo "[$label] push FAILED. Commits are local; push manually before teardown."
  return 1
}
