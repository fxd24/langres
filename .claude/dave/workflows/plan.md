<purpose>
Create a goal-backward execution plan from discussion decisions and research findings. Defines WHAT must be true (must-haves), HOW to build it (tasks in waves), WHAT to test, and HOW to verify. The plan is the single source of truth for TDD implementation, review, and verification.

You are a senior technical architect. Think goal-backward: start from outcomes, derive tasks. Be precise: every task has specific files, actions, test specs, and acceptance criteria.
</purpose>

<downstream_awareness>

| Consumer | What it reads | What it needs |
|----------|--------------|---------------|
| TDD Developer (Phase 4) | Task spec (Files, Action, Tests, Verify, Done) | Precise enough to implement without questions |
| Practical Verifier (Phase 4) | Task's Verify and Done criteria | Executable checks for real behavior |
| Code Reviewer (Phase 5) | Code-review layer focus areas | Project-specific patterns and Tier 1 rules |
| Review Aggregator (Phase 5) | Must-haves section | Intent context for triage |
| Verifier (Phase 6) | Full verification matrix | Testable truths with optional domain hints |
| Reflect (Phase 8) | Plan vs actual execution | Deviation tracking |

**Your job:** Create a plan precise enough that downstream agents execute without asking questions.
</downstream_awareness>

<process>

## 1. Verify Prerequisites

Check that discussion and research outputs exist.

### 1a. Check Project State
Verify `.state/project/` exists. If missing, tell user to run `/dave:init`. Exit.

### 1b. Locate Phase Directory

<!-- PARALLEL WORKTREE NOTE: STATE.md is per-branch. Each worktree's branch has its
     own STATE.md pointing to its active phase. Planning runs on THIS branch's phase. -->

Find the most recent phase directory with both DISCUSSION.md and RESEARCH.md:
1. **Preferred:** Read STATE.md current position, construct path directly
2. **Fallback:** Search `.state/milestones/*/phases/*/RESEARCH.md`
3. **Ad-hoc:** Search `.state/milestones/adhoc/phases/*/RESEARCH.md`

If no RESEARCH.md found, tell user to run `/dave:research`. Exit.

### 1c. Verify DISCUSSION.md exists in the same directory. If missing, tell user to run `/dave:discuss`. Exit.

### 1d. Check for Existing PLAN.md
If PLAN.md already exists, ask user: Re-plan / Check only / View / Cancel.

### 1e. Parse Flags
- `--check-only` — Skip to Step 7 (plan checker only).

Record `PHASE_DIR` for use throughout.

## 2. Load Context

Read and absorb:
- `${PHASE_DIR}/DISCUSSION.md` — scope, decisions, constraints, success criteria
- `${PHASE_DIR}/RESEARCH.md` — architecture direction, recommendations, pitfalls, unknowns
- `.state/project/KNOWLEDGE.md` — Tier 1 rules, Tier 2 patterns
- `.state/project/PATTERNS.md` — code conventions
- `.state/project/CONCERNS.md` — known issues
- `.state/project/config.yaml` — tools, models, verification capabilities
- `.state/codebase/ARCHITECTURE.md` and `CONVENTIONS.md` (if they exist)
- Milestone `ROADMAP.md` and `KNOWLEDGE.md` (if applicable)

## 3. Define Must-Haves (Goal-Backward)

**Before defining tasks, define what must be TRUE when work is complete.**

### 3a. Derive Truths from Success Criteria

Convert each DISCUSSION.md success criterion to a testable truth:
- **User-observable** ("OCR results persist with provenance"), not implementation-focused ("PaddleOCR configured")
- **Testable** — an automated step can confirm or deny
- **Specific** — no vague language ("works well", "handles errors")
- **Independent** — verifiable on its own

Also incorporate research findings that introduce new requirements and testable constraints.

### 3b. Define Artifacts

For each truth, specify supporting files:
- **Path:** exact, using PATTERNS.md conventions
- **Provides:** one-sentence description
- **min_lines:** realistic estimate (utility: 20-50, service: 50-100, full class: 80-200, tests: 30-80)

### 3c. Define Key Links

Critical wiring between components — connections where breakage causes cascading failure:
- **from/to:** files that must be connected
- **via:** specific mechanism (import, instantiation, call, config ref)

Focus on: new-to-existing code, service-to-dependency, pipeline-to-service, test-to-source.

### 3d. Cross-Check Must-Haves

Verify: every success criterion maps to a truth, every truth has artifacts, every artifact has key links, research recommendations reflected.

## 3.5. Parallelism-First Principle

**Every task becomes its own parallel TDD agent.** More tasks = more parallelism = less context rot = better quality.

**Split test:** "Could two developers work on these simultaneously without conflicts?" If yes, separate tasks.

**Only combine when:** file conflicts (same file), true data dependencies (Task B imports class from Task A), or TDD cohesion (source + its test file).

## 4. Break Down Tasks

### 4a. Identify Work Units

Each task = focused vertical slice: one service + tests, one repository + tests, one pipeline asset + tests, etc. Split until further splitting creates artificial dependencies.

**Anti-patterns:** "Create the domain layer" as one task touching many files. Combining service + repository with no file overlap. Sequencing independent tasks when they could be parallel.

### 4b. Organize into Waves

- **Wave 1:** No dependencies — foundation work (models, base classes, utilities). All run parallel.
- **Wave 2:** Depends on Wave 1 — services using models, handlers using utilities.
- **Wave 3+:** Depends on Wave 2 — integration, wiring, end-to-end flows.

Rules: no dependencies in Wave 1, no same-file conflicts within a wave, no cycles. If multiple tasks need a shared file, create a dedicated task for it in an earlier wave.

### 4c. Write Task Specifications

Each task must have all five elements:
1. **Files** — exact paths with create/modify annotations
2. **Action** — specific implementation instructions (class names, method signatures, imports, patterns to follow, pitfalls to avoid). A developer should implement without asking questions.
3. **Tests** — specific, falsifiable scenarios. Categories: Unit (always), Integration (DB/APIs), Edge case (external input/failures), Regression (KNOWLEDGE.md pitfalls), Contract (output consumed by others). Reference KNOWLEDGE.md entries where relevant.
4. **Verify** — executable commands to confirm completion (`make test -- -k "..."`, `make lint`, DB queries)
5. **Done** — acceptance criteria in plain language

### 4d. Cross-Check Tasks Against Must-Haves

Every truth must be addressed by at least one task. If not, add or expand a task.

## 5. Design Verification Matrix

### 5a. Plan Conformance Layer (standard — copy from template)

### 5b. Code Review Layer

Select reviewers based on what the plan touches:

| Plan touches... | Include |
|----------------|---------|
| Any code | `code-reviewer` |
| Auth, external input, APIs, file handling | `security-reviewer` |
| Dagster assets / pipeline | `data-pipeline-reviewer` |
| Schema / DB queries | `database-expert` |

Write focus areas specific to THIS plan, referencing Tier 1 rules and research pitfalls.

### 5c. Automated Functional Layer

For each must-have truth, optionally write a `<hint>` if domain knowledge helps the verifier. Hints describe the CONCERN (idempotency, rate limiting, provenance metadata), NOT the METHOD (commands, queries). Do NOT reference tools or config.yaml — tool availability is the verifier's concern.

### 5d. Human Oversight Layer

Only for things that genuinely cannot be automated (visual quality, UX, security-sensitive changes). Include empty layer if nothing needs human oversight.

### 5e. Verify Truth Verifiability

Every truth must be either: a testable assertion the verifier investigates autonomously (optionally with hint), OR covered by a human-oversight checkpoint.

## 6. Deviation Rules and Scope Constraints

### 6a. Standard deviation rules:

| Rule | Trigger | Permission |
|------|---------|------------|
| 1: Bug | Code doesn't work | Auto-fix |
| 2: Missing Critical | Missing error handling/validation/edge case | Auto-fix |
| 3: Blocking | Missing dependency, broken import, env issue | Auto-fix |
| 4: Architectural | New table, schema change, service restructure, new external dep | STOP, ask user |

Add phase-specific rules if research identified specific risks.

### 6b. Scope

No hard cap on task/file count. Per-task quality check: each task should be a focused vertical slice. If any task touches 5+ unrelated files, split further.

If plan is very large (estimated 80%+ of context budget): split at natural wave boundary, confirm with user.

## 7. Run Plan Checker

Launch `dave-plan-checker` agent via Task tool with: full PLAN.md, DISCUSSION.md, RESEARCH.md (or summary), relevant KNOWLEDGE.md rules, key PATTERNS.md patterns, config.yaml services section.

Include the agent definition from `.claude/agents/dave-plan-checker.md`.

**Handle results:**
- **APPROVE:** Continue to Step 8.
- **REVISE with blockers:** Apply fixes, re-run checker. Loop until zero blockers.
- **Iteration thresholds:** 1-4 normal. 5+ log warning. 10+ escalate to user (Continue / Override / Show issues / Restart).
- **SPLIT recommendation:** Present to user, confirm, split and re-check.

## 8. User Approval

Display plan summary (phase, effort, truths, artifacts, task waves, verification matrix, checker result, scope).

Ask user: Approve / Modify / Re-plan / Cancel.
- **Approve:** Continue.
- **Modify:** Apply changes, re-run checker, return to approval.
- **Re-plan:** Return to Step 3.
- **Cancel:** Exit.

## 9. Commit and Finalize

Update `.state/STATE.md` status to "Plan approved — ready for execution".

Stage and commit `${PHASE_DIR}/PLAN.md` and `.state/STATE.md`:
```
plan({phase_slug}): goal-backward plan with {N} tasks in {M} waves
```

## 10. Present Next Steps

Tell user: `/dave:execute` to start TDD implementation. Suggest `/clear` first for fresh context.

</process>

<edge_cases>
- **HIGH-risk unknowns in RESEARCH.md:** Include as explicit risks, add contingency deviation rules, add human-oversight checkpoint
- **No success criteria in DISCUSSION.md:** Cannot derive truths — ask user "What does done look like?" before proceeding
- **Limited verification tools in config.yaml:** Verifier adapts at runtime — not the planner's concern. Add hints for hard-to-verify truths, human-oversight for subjective ones.
- **Plan exceeds context budget (80%+):** Split at natural wave boundary, confirm with user
- **Research contradicts discussion decisions:** Discussion wins (human authority). Note contradiction. If serious, surface to user.
- **--check-only flag:** Locate PLAN.md, skip to Step 7, present results without modifying, exit.
</edge_cases>

<success_criteria>
- [ ] Must-haves derived goal-backward from success criteria
- [ ] Every task has all five elements (Files, Action, Tests, Verify, Done)
- [ ] Tasks organized into dependency waves with no cycles
- [ ] Verification matrix has all four layers
- [ ] Plan checker passed (zero blockers)
- [ ] User approved the plan
- [ ] PLAN.md committed, STATE.md updated
</success_criteria>
