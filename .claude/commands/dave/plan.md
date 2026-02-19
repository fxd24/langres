---
name: dave:plan
description: "Goal-backward planning from RESEARCH.md to PLAN.md with verification matrix"
argument-hint: "[--check-only]"
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
  - Task
  - AskUserQuestion
---
<objective>
Create a goal-backward execution plan from research findings. Combines must-haves (what must be TRUE), prescriptive tasks organized in dependency waves, test specifications, and a verification matrix that defines exactly HOW to verify the work.

**Requires:**
- RESEARCH.md with findings and recommendations (from `/dave:research`)
- DISCUSSION.md with scope, decisions, and success criteria (from `/dave:discuss`)
- `.state/project/` files (KNOWLEDGE.md, PATTERNS.md, CONCERNS.md, config.yaml)

**Creates:**
- `PLAN.md` — must-haves, task waves, test specs, verification matrix, deviation rules
  - If milestone exists: `.state/milestones/{slug}/phases/{N}/PLAN.md`
  - If ad-hoc: `.state/milestones/adhoc/phases/{slug}/PLAN.md`

**Flags:**
- `--check-only` — Run the plan checker on an existing PLAN.md without creating a new plan

**After this command:** Run `/dave:execute` to implement the plan with TDD.
</objective>

<execution_context>
@./.claude/dave/workflows/plan.md
@./.claude/dave/templates/plan.md
@./.claude/dave/references/verification-matrix.md
</execution_context>

<process>
Execute the planning workflow from @./.claude/dave/workflows/plan.md end-to-end.
Parse $ARGUMENTS for flags:
- If `--check-only` is present, locate the existing PLAN.md and run only the plan checker step (Step 7 of the workflow).
Preserve all workflow gates (state checks, RESEARCH.md existence, plan checker validation, user approval).
</process>
