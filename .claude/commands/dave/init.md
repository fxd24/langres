---
name: dave:init
description: Initialize project state from CLAUDE.md and codebase analysis
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
  - AskUserQuestion
---
<objective>
Initialize the Dave Framework `.state/` directory for this project by detecting available tools, extracting knowledge from CLAUDE.md, and seeding all project-level state files.

**Creates:**
- `.state/project/config.yaml` — detected tools, model profiles, verification capabilities
- `.state/project/KNOWLEDGE.md` — Tier 1 rules extracted from CLAUDE.md
- `.state/project/PATTERNS.md` — architecture patterns from CLAUDE.md and project docs
- `.state/project/STACK.md` — tech stack, libraries, versions
- `.state/project/CONCERNS.md` — known issues and tech debt
- `.state/STATE.md` — current position and session continuity
- `.state/codebase/` — empty structure (populated by codebase mapping)
- `.state/milestones/` — empty structure (populated when milestones begin)
- `.state/debug/` — empty structure (populated by debug sessions)

**After this command:** Run `/dave:state` to inspect what was created, or start a milestone.
</objective>

<execution_context>
@./.claude/dave/workflows/init.md
</execution_context>

<process>
Execute the init workflow from @./.claude/dave/workflows/init.md end-to-end.
Preserve all workflow gates (existence checks, user confirmations, knowledge extraction).
</process>
