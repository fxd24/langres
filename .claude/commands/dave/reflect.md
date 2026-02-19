---
name: dave:reflect
description: "Learning loop — extract knowledge, update patterns, create phase summary"
argument-hint: "[--promote-only] [--summary-only]"
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
Run the learning loop after a phase completes. Extracts knowledge from plan vs actual execution, review findings, and verification results. Produces SUMMARY.md, updates phase KNOWLEDGE.md, proposes Tier 2 promotions, and optionally aggregates knowledge upward when a milestone ends.

**Requires:**
- Phase complete (push done, or at least verification complete)
- Phase artifacts exist (PLAN.md, EXECUTION_STATE.md, REVIEWS.md, VERIFICATION.md)

**Creates:**
- `SUMMARY.md` — Phase completion summary with metrics and lessons
- Phase `KNOWLEDGE.md` — Updated with new Tier 2 entries from this phase
- Promotion proposals (if any Tier 2 entries exceed threshold)
- Updated project-level files (PATTERNS.md, CONCERNS.md) if patterns discovered

**Flags:**
- `--promote-only` — Skip extraction, only present pending promotion candidates
- `--summary-only` — Only generate SUMMARY.md, skip knowledge extraction

**After this command:** Phase is complete. Start the next phase or milestone.
</objective>

<execution_context>
@./.claude/dave/workflows/reflect.md
@./.claude/dave/templates/summary.md
@./.claude/dave/templates/knowledge.md
@./.claude/dave/references/knowledge-format.md
</execution_context>

<process>
Execute the reflect workflow from @./.claude/dave/workflows/reflect.md end-to-end.
Parse $ARGUMENTS for flags:
- If `--promote-only` is present, skip knowledge extraction and go straight to promotion proposals.
- If `--summary-only` is present, only generate SUMMARY.md without knowledge extraction or promotion.
Preserve all workflow gates (state checks, artifact existence, promotion approval).
</process>
