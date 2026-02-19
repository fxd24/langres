---
name: dave:progress
description: "Check milestone progress, show current status, route to next action"
argument-hint: "[--verbose]"
allowed-tools:
  - Read
  - Bash
  - Glob
  - Grep
---
<objective>
Show milestone progress and route to the next action. A quick status check that tells you where the project is and what to do next.

**Requires:**
- `.state/` directory exists (from `/dave:init`)

**Creates:**
- Nothing — read-only status check

**Flags:**
- `--verbose` — Show detailed artifact status and velocity metrics

**After this command:** Run whatever command is suggested as the next action.
</objective>

<execution_context>
@./.claude/dave/workflows/progress.md
</execution_context>

<process>
Execute the progress workflow from @./.claude/dave/workflows/progress.md end-to-end.
Parse $ARGUMENTS for flags:
- If `--verbose` is present, include detailed artifact status and velocity metrics.
This is a read-only command — it does not modify any files.
</process>
