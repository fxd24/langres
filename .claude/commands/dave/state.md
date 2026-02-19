---
name: dave:state
description: Inspect and improve project state (knowledge, config, sync)
argument-hint: "[knowledge|config|sync]"
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
  - Task
---
<objective>
Inspect the Dave Framework `.state/` directory health, report on knowledge coverage, tool availability, and sync status between CLAUDE.md and the state files.

**Modes:**
- No argument: full health check across all state files
- `knowledge`: focus on KNOWLEDGE.md coverage, tier distribution, gap analysis
- `config`: focus on config.yaml, tool detection, model profiles
- `sync`: compare CLAUDE.md with state files, identify drift

**Output:** A structured report with findings and actionable suggestions.
</objective>

<execution_context>
@./.claude/dave/workflows/state.md
</execution_context>

<process>
Execute the state workflow from @./.claude/dave/workflows/state.md end-to-end.
Parse $ARGUMENTS for the optional focus mode (knowledge, config, sync).
Default to full health check if no argument provided.
</process>
