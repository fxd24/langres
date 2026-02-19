---
name: dave:quick
description: "Quick task: inline plan, TDD, review, verify — no discussion, research, or planning overhead"
argument-hint: "[task description] [--skip-review] [--no-commit]"
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
Execute a small, well-understood task end-to-end in a single session: inline plan, TDD implementation, streamlined review, and automated verification. Compresses the front half of the Dave workflow (skip discuss, research, plan overhead) while preserving the back half (TDD discipline, code review, verification).

**Requires:**
- `.state/project/` files (KNOWLEDGE.md, PATTERNS.md) — run `/dave:init` first
- A clear task description (one sentence, obvious scope)

**Creates:**
- Implementation code and tests via strict TDD
- Atomic commit(s) per task
- `.state/milestones/adhoc/phases/{slug}/QUICK_PLAN.md` — inline plan
- `.state/milestones/adhoc/phases/{slug}/REVIEWS.md` — review findings
- `.state/milestones/adhoc/phases/{slug}/VERIFICATION.md` — verification results
- `.state/milestones/adhoc/phases/{slug}/KNOWLEDGE.md` — only if deviations occurred

**Flags:**
- `--skip-review` — Skip the review phase entirely (for truly trivial changes like typo fixes)
- `--no-commit` — Run TDD and review/verify but do not commit. Useful for exploration.

**When to use quick vs full workflow:**
- Use quick when scope is obvious, no unknowns, 1-3 files, low risk, self-contained
- Use the full workflow when scope needs clarification, unknowns exist, large change, high risk, or architectural impact

**After this command:** Task is complete. Push manually or run `/dave:push` if needed.
</objective>

<execution_context>
@./.claude/dave/workflows/quick.md
@./.claude/dave/templates/quick-plan.md
</execution_context>

<process>
Execute the quick workflow from @./.claude/dave/workflows/quick.md end-to-end.
Parse $ARGUMENTS for:
- The task description (positional, required — prompt interactively if missing)
- If `--skip-review` is present, skip the review phase entirely.
- If `--no-commit` is present, run TDD and review/verify but do not commit.
Preserve all workflow gates (state checks, TDD protocol, verification before declaring complete).
</process>
