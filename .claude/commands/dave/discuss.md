---
name: dave:discuss
description: "Structured discussion to establish phase guardrails, scope, and research topics"
argument-hint: "[phase-description or @context-file]"
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
  - AskUserQuestion
---
<objective>
Conduct a structured discussion to establish guardrails, scope, and research topics for a development phase. Captures decisions that downstream agents (researcher, planner, TDD) need to operate autonomously.

**Accepts:**
- A text description of what to build (ad-hoc phase)
- An @-referenced context file with detailed requirements
- A milestone/phase number if milestones exist in `.state/milestones/`

**Creates:**
- `DISCUSSION.md` — scope, decisions, constraints, success criteria, research topics
  - If milestone exists: `.state/milestones/{slug}/phases/{N}/DISCUSSION.md`
  - If ad-hoc: `.state/milestones/adhoc/phases/{slug}/DISCUSSION.md`

**After this command:** Run `/dave:research` to deep-dive the identified research topics.
</objective>

<execution_context>
@./.claude/dave/workflows/discuss.md
</execution_context>

<process>
Execute the discuss workflow from @./.claude/dave/workflows/discuss.md end-to-end.
Parse $ARGUMENTS for either a phase description, @context-file reference, or milestone/phase number.
Preserve all workflow gates (state checks, user confirmations, scope guardrails).
</process>
