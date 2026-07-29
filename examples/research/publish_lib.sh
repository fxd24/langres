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

publish_branch() {
  local label="${1:-publish}" branch default
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
