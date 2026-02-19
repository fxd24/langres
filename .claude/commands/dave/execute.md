---
name: dave:execute
description: "TDD implementation with wave-based parallelism, deviation handling, atomic commits"
argument-hint: "[--wave N] [--task N.M] [--dry-run]"
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
  - Task
  - AskUserQuestion
---
<objective>
Execute the approved plan using strict TDD with wave-based parallelism. Each task gets its own tdd-developer agent (RED-GREEN-REFACTOR), followed by practical-verifier for real-world validation. Commits atomically per task.

**Requires:**
- PLAN.md approved by user (from `/dave:plan`)
- `.state/project/` files (KNOWLEDGE.md, PATTERNS.md)

**Creates:**
- Implementation code and tests as specified in PLAN.md
- Atomic commits per task (one commit = one task + its tests)
- Phase KNOWLEDGE.md entries for any deviations encountered

**Flags:**
- `--wave N` — Start execution from wave N (skip earlier waves, assumes they are complete)
- `--task N.M` — Execute only a specific task (for re-running a failed task)
- `--dry-run` — Show what would be executed without running anything

**After this command:** Run `/dave:review` for multi-agent code review (or proceed to review manually).
</objective>

<execution_context>
@./.claude/dave/workflows/execute.md
@./.claude/dave/templates/plan.md
</execution_context>

<process>
Execute the execution workflow from @./.claude/dave/workflows/execute.md end-to-end.
Parse $ARGUMENTS for flags:
- If `--wave N` is present, skip waves before N.
- If `--task N.M` is present, execute only that specific task.
- If `--dry-run` is present, display the execution plan without running anything.
Preserve all workflow gates (state checks, PLAN.md existence, deviation handling, verification before commit).
</process>
