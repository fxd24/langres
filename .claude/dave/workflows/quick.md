<purpose>
Execute a small, well-understood task end-to-end: inline plan, TDD, review, verification. Compresses the front half of the Dave workflow (skip discuss/research/plan) while preserving quality (TDD, code review, verification).

You are the quick task orchestrator. Generate the inline plan directly, then launch specialized agents for TDD, review, and verification. If at any point you discover the task is more complex than expected (4+ files, new table, architectural decisions), STOP and tell the user to use the full workflow.
</purpose>

<downstream_awareness>

| Output | Consumer |
|--------|----------|
| QUICK_PLAN.md | TDD executor, reviewer, verifier |
| Implementation code + tests | Reviewers, verifier |
| Atomic commit(s) | Traceable to quick plan |
| REVIEWS.md | Verification, reflect |
| VERIFICATION.md | Push, reflect |
| Phase KNOWLEDGE.md | Only if deviations occurred |

Quick mode reuses: `tdd-developer`, `practical-verifier`, `dave-change-summarizer`, `code-reviewer`, `security-reviewer`/`database-expert` (if relevant), external models (from config.yaml), `dave-review-aggregator` (if external models participated).
</downstream_awareness>

<process>

## 1. Validate and Load Context

### 1a. Check `.state/project/` exists. If missing, tell user to run `/dave:init`. STOP.

### 1b. Parse Arguments
- **Task description** (positional, required — prompt interactively if missing)
- `--skip-review` — skip review phase
- `--no-commit` — run TDD/review/verify but don't commit

### 1c. Load project knowledge: KNOWLEDGE.md, PATTERNS.md, CONVENTIONS.md (if exists).

<!-- PARALLEL WORKTREE NOTE: STATE.md is per-branch. Each worktree's branch has its
     own STATE.md pointing to its active phase. Quick tasks create state for THIS branch. -->

### 1d. Generate slug from task description, create `PHASE_DIR = .state/milestones/adhoc/phases/{slug}`. Handle slug collisions by appending `-2`, `-3`.

### 1e. Announce quick mode with task, slug, directory, knowledge stats.

## 2. Inline Plan

### 2a. Identify affected files using Grep/Glob — search keywords, find test files, check imports/callers.

### 2b. Complexity Check

If analysis reveals 4+ source files, new table/migration, new service/gateway, or architectural pattern change: warn user and ask "Proceed with quick" or "Switch to full workflow".

### 2c. Generate QUICK_PLAN.md

Follow template from `.claude/dave/templates/quick-plan.md` (adapt for quick scope: fewer tasks, simpler must-haves, deviation rules 1-3 auto-fix only). Consider decomposition into parallel tasks when files don't overlap and no true data dependencies.

Must include all five TDD elements:
1. **Files** — exact paths, create/modify
2. **Action** — specific implementation instructions
3. **Tests** — concrete scenarios with expected behavior
4. **Verify** — runnable commands
5. **Done** — acceptance criteria

Also: Must-Haves (1-3 truths), Deviation Rules (1-3 auto-fix, 4 stop).

Write to `${PHASE_DIR}/QUICK_PLAN.md`.

### 2d. Display concise plan summary (files, test count, must-have, verify command).

## 3. TDD Execution

### 3a. Launch TDD Executor(s)

Read QUICK_PLAN.md. If multiple independent tasks, launch in parallel.

**Dynamic data for each executor prompt:**
- Task description, Files, Action, Tests, Verify, Done (from QUICK_PLAN.md)
- Relevant Tier 1 rules from KNOWLEDGE.md
- Relevant patterns from PATTERNS.md/CONVENTIONS.md

**Plan-specific constraints:**
- You are an EXECUTOR — follow the plan spec mechanically
- Do NOT modify files outside your task's Files list (except Deviation Rule 3)
- Do NOT make architectural changes (Deviation Rule 4 — STOP and report)
- Do NOT invent additional tests or skip specified ones
- Do NOT add features/helpers/abstractions beyond the plan
- Strict RED → GREEN → REFACTOR → VERIFY:
  - **RED:** Write failing tests from plan spec exactly. Verify they fail before implementing.
  - **GREEN:** Write MINIMUM code to pass tests. No gold-plating.
  - **REFACTOR:** Clean up while tests stay green. Minimal changes only.
  - **VERIFY:** Run plan's Verify criteria + `make lint`.
- Report: files, tests, verify results, deviations, concerns

### 3b. Collect results: check deviations, blockers, test results. Rule 4 issues → present to user. After 2 failed attempts → ask user.

### 3c. Practical Verification

Launch verifier per task with: Verify/Done criteria, implementation summary.
Verifier protocol: tests → lint → exercise code path → check side effects → verify Done criteria.

**Verifier output format:**
- Tests: PASS/FAIL (count)
- Lint: PASS/FAIL
- Code execution: PASS/FAIL (what was run)
- Side effects: PASS/FAIL (what was checked)
- Done criteria: each with PASS/FAIL
- Overall: PASS/FAIL (if FAIL: specific failure and what needs fixing)

- **PASS:** proceed to commit.
- **FAIL:** re-launch executor with failure details, max 2 fix attempts.

### 3d. Atomic Commit(s)

**Skip if `--no-commit`.**

Stage ONLY task files + test files (never `git add .`). Commit:
```
{type}(quick): {description}

- {changes}
- Tests: {count} passing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```
Type: fix/feat/refactor/test/chore.

### 3e. Announce TDD results (RED/GREEN/REFACTOR status, verification, commit hash).

## 4. Streamlined Review

**Skip if `--skip-review`.**

### 4a. Compute diff against main:
```bash
MERGE_BASE=$(git merge-base HEAD main 2>/dev/null || git merge-base HEAD origin/main)
```

### 4b. Generate Change Summary

Launch `dave-change-summarizer` with diff, changed files, QUICK_PLAN.md. Write to `${PHASE_DIR}/CHANGE_SUMMARY.md`. Fallback to raw diff if summarizer fails.

### 4c. Select reviewers: always `code-reviewer`. Add `security-reviewer` (auth/input/APIs/files/secrets) or `database-expert` (schema/migrations/queries) if relevant.

### 4d. Select external models from `.state/project/config.yaml` (`phases.review.services` with `code-review` capability). Graceful default if none available.

### 4e. Launch ALL reviewers in parallel (internal + external).

**Internal reviewers** receive: CHANGE_SUMMARY.md (or diff fallback), relevant Tier 1 rules, key patterns, review instructions.

**External models** receive: self-contained prompt built from `.claude/skills/review/SKILL.md` template with feature intent, change summary, knowledge rules, patterns, file excerpts.

### 4f. Triage findings

**If external models participated:** launch `dave-review-aggregator` to cross-reference.
**If internal only:** orchestrator triages directly.

Three categories (no "defer" in quick mode):

| Category | Criteria | Action |
|----------|----------|--------|
| Fix now | Bugs, correctness, Tier 1 violations, security | Scoped TDD fix |
| Note | Valid observation, non-blocking | Record in REVIEWS.md |
| Dismiss | False positive, matches PATTERNS.md, out of scope | Record with reason |

Auto-dismiss: contradicts Tier 1 KNOWLEDGE.md, outside task scope, style matching conventions. High confidence if 2+ reviewers flag same issue.

### 4g. Fix Loop

For each fix-now item: create mini-task (Files, Action, Verify), launch TDD executor, commit fix:
```
fix(quick-review): {finding ID} {description}

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```
Scoped re-review on changed files. Loop until convergence. If oscillating, escalate to user.

### 4h. Write `${PHASE_DIR}/REVIEWS.md`

Follow the template from `.claude/dave/templates/reviews.md`. Include: reviewers, external models, totals, fix-now (resolved), notes, dismissed, fix loop history.

### 4i. Announce review results (reviewers, finding counts).

## 5. Streamlined Verification

**Layer 1 (plan conformance) + Layer 3 (automated functional) only.**

### 5a. Layer 1: Plan Conformance
- Verify each must-have truth against codebase (VERIFIED/FAILED/UNCERTAIN)
- Anti-pattern scan: TODO/FIXME/HACK/PLACEHOLDER, empty implementations, log-only error handlers

### 5b. Layer 3: Automated Functional
- `make test`
- `make lint`
- Plan-specific verify commands from QUICK_PLAN.md

### 5c. Gap Closure
If gaps found: create fix tasks, launch TDD executor, re-verify failed items. If not converging, report to user.

### 5d. Write `${PHASE_DIR}/VERIFICATION.md`

Follow the template from `.claude/dave/templates/verification.md`. Include: status, score, Layer 1 must-haves table, anti-pattern scan, Layer 3 checks, gaps, gap closure history.

### 5e. Announce verification results (layer statuses, must-have score, test/lint status).

## 6. Wrap Up

### 6a. If deviations occurred, write `${PHASE_DIR}/KNOWLEDGE.md` with deviations and decisions. Skip if none.

### 6b. Update `.state/STATE.md`: append quick task record to "Quick Tasks" table, update "Last activity".

### 6c. Increment Tier 2 verification counts for any confirmed patterns.

### 6d. Commit state files (**skip if `--no-commit`**):
```bash
git add ${PHASE_DIR}/QUICK_PLAN.md ${PHASE_DIR}/REVIEWS.md ${PHASE_DIR}/VERIFICATION.md .state/STATE.md
[ -f ${PHASE_DIR}/KNOWLEDGE.md ] && git add ${PHASE_DIR}/KNOWLEDGE.md
```
```
state(quick-{slug}): task complete with review and verification

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

### 6e. Present summary: task, commit, state dir, TDD results, review findings, verification status, files changed, deviations, state files written.

</process>

<edge_cases>
- **No task description:** Prompt interactively. Do not proceed without one.
- **`.state/milestones/adhoc/` missing:** Create it (`mkdir -p`). Normal for first quick task.
- **Slug collision:** Append numeric suffix (`-2`, `-3`).
- **TDD executor returns empty/garbage:** Record failure, don't retry same prompt, ask user.
- **Tests already pass before RED phase:** Behavior may already exist — verify codebase, report "already implemented" if so.
- **Verifier cannot run code (DB unreachable):** Skip side-effects, note limitation, add manual note.
- **Review finds architectural issue:** Present to user — "Fix within quick mode" or "Stop and use full workflow".
- **Context pressure (70%+):** Summarize earlier results, release detailed outputs, prioritize completing.
- **`--no-commit` with review fixes:** Apply fixes to working tree without committing.
</edge_cases>

<success_criteria>
- [ ] QUICK_PLAN.md generated with all five TDD elements (Files, Action, Tests, Verify, Done)
- [ ] TDD executor(s) launched (not the orchestrator writing code)
- [ ] TDD protocol followed: RED → GREEN → REFACTOR → VERIFY
- [ ] Practical verification passed for each task
- [ ] Atomic commit(s) created (specific files staged, never `git add .`)
- [ ] Review ran with internal + external reviewers (unless --skip-review)
- [ ] Fix loop converged
- [ ] Layer 1 + Layer 3 verification passed
- [ ] REVIEWS.md and VERIFICATION.md written to phase directory
- [ ] STATE.md updated with quick task record
</success_criteria>
