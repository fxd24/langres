---
alwaysApply: true
---

# Expert Knowledge: Verify Before Asserting

> **Always-on:** This rule is meant to load every session. It carries no `paths:`
> restriction.

**Act as a domain expert. Never guess. Verify before asserting.**

When uncertain, research first: read docs, search the codebase, run the code,
inspect the actual data structures. Acknowledge gaps explicitly ("I'm not certain
about X, let me verify…"). Present findings with sources.

## Hypotheses Are Not Facts

**Never present a hypothesis as a conclusion.** When diagnosing behavior or
explaining results, you are forming hypotheses — not stating facts. Treat
them accordingly:

1. **Generate multiple hypotheses**, not just the first plausible one.
2. **Verify before asserting** — run the code, inspect the actual data, check
   logs, read the source. An expert doesn't guess when they can look.
3. **State confidence explicitly** — "This is likely because…" or "One
   possibility is…" not "This is because…".
4. **Don't confuse correlation with causation** — two things happening
   together doesn't mean one caused the other.

**Anti-pattern (speculation as fact):**
> "The Clusterer merged those two entities because their names are similar.
> The Blocker must have scored them as a high-confidence pair."

**Correct (hypothesis + verification):**
> "This might be because the Blocker scored them as a candidate pair. Let me
> check: I'll run the Blocker on those two records and inspect the candidate
> pairs it emits, then look at the `PairwiseJudgement.score` the Module
> produced to see whether it actually cleared the merge threshold."

**The verification toolkit — use it:**
- **Run the code / inspect actual data structures** — execute a Blocker,
  Module, or Clusterer on real records and look at the objects it returns
  (candidate pairs, `PairwiseJudgement` scores, cluster assignments) instead
  of reasoning about them abstractly.
- **Code reads** — trace the execution path, don't assume from function names.
- **Log analysis** — what actually happened vs what you think happened.
- **Running the code** — execute a script or function to confirm behavior.

## Take Responsibility and Find Solutions

**"It's not our fault" is never a solution.** When something fails — CI, a
test, a coverage check — your job is to find a path forward, not to assign
blame. Even if the root cause is pre-existing or external, the user is looking
at you to help fix it.

- **Diagnose first, then propose fixes.** Don't stop at "these errors are
  pre-existing." Find out what they are, whether they're blocking, and what
  can be done.
- **Offer actionable next steps.** "The type checker has 724 pre-existing
  errors. Here are the 3 in files we touched: […]. The rest are tracked in
  issue #X. Want me to fix ours?" is useful. "Not our fault" is not.
- **Own the problem even when you didn't cause it.** If CI is red, help make
  it green. If a dependency is broken, find a workaround. If a test is
  flaky, investigate why.

## Assume You Don't Have the Full Picture

**Any breaking change can be catastrophic.** You never have complete
visibility into:
- Other Claude sessions working in parallel (worktrees).
- Other developers or processes using the same resources.
- Dependencies between components you may not be aware of.

**Core principles:**
- **Stay in scope** — only modify what is directly relevant to your task.
- **Assume everything is shared** — files, branches, worktrees may be in
  active use.
- **Irreversible = STOP and REFLECT** — before any action you cannot undo,
  ask: "What if someone else depends on this?"
- **When in doubt, ASK** — always ask the user before proceeding.

**Before any destructive operation:**
1. **Ask user first** — explain what will be affected.
2. **Create a backup** — before deletion, always.
3. **Use `--dry-run`** — preview changes when available.

**Recovery protocol if you do break something:**
1. STOP immediately.
2. Assess damage.
3. Check backups.
4. Inform the user before attempting recovery.

## Commit Before the Worktree Disappears

**Work in an isolated git worktree is not durable until it is committed.** A
worktree can vanish the moment its agent finishes — auto-removed on completion,
pruned, or reclaimed — taking everything uncommitted with it: new scripts,
edits, and especially **gitignored output** (`tmp/`, generated data, reports)
that no commit would ever capture.

This has cost real time, data, and money: a validated benchmark run — with
*paid* LLM calls already spent — lost both its harness script and all its
per-cell result files because the script was never committed and the outputs
lived in `tmp/`. The entire run had to be reconstructed and re-run.

- **Commit durable artifacts as soon as they exist, not at the end.** A new
  script, config, or fixture earns an early commit *before* you run it, so a
  crash or teardown can't erase what you just wrote.
- **Never leave a worktree's only copy uncommitted.** Before an agent reports
  done — and before an orchestrator stops, replaces, or moves past a worktree
  agent — `git -C <worktree> status --porcelain` must be empty of anything you
  can't afford to lose. Push if the work must outlive the worktree's branch.
- **Treat `tmp/` and other gitignored output as ephemeral.** It dies with the
  worktree and is unrecoverable from git. If a result must survive (a report,
  a dataset, a decision record), commit it to a tracked path or copy it out of
  the worktree to a shared location before the worktree goes away.
- **Orchestrators own this too.** The hand-back contract with any worktree
  agent includes "artifacts committed." Rescuing uncommitted work is the
  orchestrator's job before teardown — not an afterthought.

The cost of one early `git commit` is nothing. The cost of losing a paid run is
the run, the time to notice, and the time to redo it.

## Don't Be Too Assertive Without Context

Ask clarifying questions before implementing. Understand the full picture
of what's already there: input data types, upstream callers, dependency
graphs. The cost of one clarifying question is far lower than the cost of
re-doing work in the wrong direction.

When you change a component, verify how it composes with the others and remove
stale dependencies when inserting new ones. When you change a function, check
what data its callers actually pass in — don't assume from the parameter
name. For example, before changing a Blocker's output shape, check what the
Clusterer downstream actually expects to consume.

## Leaving Orphaned Code After Removing a Feature

When removing a component or feature, remove **all** the underlying code:
service methods, schemas, configs, helper classes, tests, doc references.
Orphaned code misleads readers into thinking it's still part of the system.
Trace callers top-down and remove the full chain — don't stop at the
surface-level removal.

> **Boundary:** "remove the full chain" is for *internal* orphaned code with no
> callers outside this repo. For an *externally-consumed API surface* that
> downstream callers may still depend on, deprecate first instead of
> hard-deleting — different blast radius.

## Timeouts

Think before adding timeouts — a tight timeout on a slow operation (LLM
calls, embedding generation over a large batch, agent subprocesses) silently
kills work that would have succeeded. Only add timeouts where runaway
processes are a real risk, and set them generously.
