# Parallel Worktree Usage

How the Dave Framework works across multiple git worktrees.

## Model

Each worktree works on a phase (or milestone) independently. There is no central
coordinator, no COORDINATION.yaml, no orchestrator branch. Git is the coordinator.

```
Worktree A (brainsquad-1):  discuss -> research -> plan -> execute -> review -> verify -> push -> reflect
                             Creates branch, does work, creates PR, merges to main.

Worktree B (brainsquad-2):  discuss -> research -> plan -> execute -> review -> verify -> push -> reflect
                             Pulls main (sees A's merged work), creates its own branch, works, merges.
```

## Rules

1. **One milestone/phase per worktree** -- Each worktree works on one thing at a time.
   Do not run two Dave phases in the same worktree simultaneously.

2. **Each phase creates its own branch** -- Use descriptive branch names:
   `feature/phase-3-pdf-extraction`, `feature/milestone-2-phase-1`, etc.

3. **Merge to main when done** -- Each phase pushes a PR and merges independently.
   The next phase (in any worktree) pulls main and sees everything.

4. **STATE.md is per-session** -- STATE.md tracks where THIS session is in the workflow.
   It is committed to the branch and travels with it. Do not worry about conflicts
   between worktrees -- each branch has its own STATE.md reflecting its own progress.
   When merging to main, STATE.md on main reflects the last merged phase.

5. **Phase artifacts are self-contained** -- Each phase writes to its own directory
   under `.state/milestones/{slug}/phases/{N}/`. Different phases use different
   directories, so there are no file conflicts.

6. **Pull main before starting a dependent phase** -- If Phase 5 depends on Phase 3's
   output, make sure Phase 3 is merged to main first. Then pull main in your worktree
   before starting Phase 5.

## Example: Two worktrees, two independent phases

```bash
# Worktree 1: Phase 3 (PDF extraction)
cd brainsquad-1
git fetch origin && git checkout -B feature/phase-3-pdf origin/main
# Run Dave workflow: discuss, research, plan, execute, review, verify, push
# PR merges to main

# Worktree 2: Phase 5 (web search integration) -- independent of Phase 3
cd brainsquad-2
git fetch origin && git checkout -B feature/phase-5-web-search origin/main
# Run Dave workflow in parallel with worktree 1
# PR merges to main independently
```

## Example: Sequential phases across worktrees

```bash
# Worktree 1: Phase 3
cd brainsquad-1
git fetch origin && git checkout -B feature/phase-3 origin/main
# Complete phase 3, merge PR to main

# Worktree 2: Phase 4 (depends on Phase 3)
cd brainsquad-2
git fetch origin && git checkout -B feature/phase-4 origin/main  # Now includes Phase 3
# Phase 4 sees Phase 3's artifacts and code via main
```

## What to watch out for

- **STATE.md on main** reflects the last merged phase, not "the current phase globally."
  Each branch has its own STATE.md that is accurate for that branch's context.

- **Locating phase artifacts** -- Workflows find the current phase via STATE.md on the
  current branch. Make sure STATE.md was updated during your phase's discuss/research/plan
  steps before running execute/review/verify.

- **Do not run /dave:progress on main to check all worktrees** -- Progress shows one
  phase at a time (the one tracked in STATE.md on the current branch). Check each
  worktree separately.
