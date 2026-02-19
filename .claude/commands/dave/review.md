---
name: dave:review
description: "Multi-agent code review with intelligent aggregation and fix loop"
argument-hint: "[--fix-only] [--skip-external]"
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
Run parallel code reviews (internal + external models), aggregate findings intelligently, and triage into actionable categories. Produces REVIEWS.md and OPEN_QUESTIONS.md.

**Requires:**
- Implementation complete (from `/dave:execute`)
- PLAN.md with `<verification-matrix><layer name="code-review">` section
- `.state/project/` files (KNOWLEDGE.md, PATTERNS.md)

**Creates:**
- `REVIEWS.md` — Actionable, triaged findings (fix now, defer, dismissed)
- `OPEN_QUESTIONS.md` — Ambiguous items for human review
- Phase KNOWLEDGE.md entries for decisions made during triage

**Flags:**
- `--fix-only` — Skip full review, only re-review files changed by previous fix loop iteration
- `--skip-external` — Skip external model reviews (faster, lower cost)

**After this command:** Address "fix now" items (loops back to scoped TDD), then run `/dave:verify` for multi-layer verification.
</objective>

<execution_context>
@./.claude/dave/workflows/review.md
@./.claude/dave/templates/reviews.md
@./.claude/dave/templates/open-questions.md
@./.claude/dave/references/verification-matrix.md
</execution_context>

<process>
Execute the review workflow from @./.claude/dave/workflows/review.md end-to-end.
Parse $ARGUMENTS for flags:
- If `--fix-only` is present, run scoped re-review on fix-loop changes only.
- If `--skip-external` is present, skip external model reviews.
Preserve all workflow gates (state checks, PLAN.md existence, aggregation quality, fix loop convergence).
</process>
