---
name: dave:research
description: "Parallel research orchestration with specialized agents"
argument-hint: "[--skip-arch] [--topics topic1,topic2]"
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
  - Task
  - WebSearch
  - WebFetch
  - AskUserQuestion
---
<objective>
Conduct parallel research across all topics identified during discussion. The orchestrator analyzes scope and launches specialized agents via Task tool: dave-architect for architecture/design and dave-topic-researcher per topic. Research covers the codebase, official docs, web sources, architectural options, strengths, and weaknesses.

**Requires:**
- DISCUSSION.md with research topics (from `/dave:discuss`)
- `.state/project/` files (KNOWLEDGE.md, PATTERNS.md, STACK.md, CONCERNS.md)

**Creates:**
- `RESEARCH.md` — architecture direction, per-topic findings with confidence levels, cross-cutting concerns, remaining unknowns
  - If milestone exists: `.state/milestones/{slug}/phases/{N}/RESEARCH.md`
  - If ad-hoc: `.state/milestones/adhoc/phases/{slug}/RESEARCH.md`
- Updates milestone-level `RESEARCH.md` with cross-phase findings (if milestone exists)

**Flags:**
- `--skip-arch` — Skip the dave-architect agent (when architecture direction is already clear)
- `--topics topic1,topic2` — Focus on specific topics from DISCUSSION.md instead of all

**After this command:** Run `/dave:plan` to create an execution plan from the research findings.
</objective>

<execution_context>
@./.claude/dave/workflows/research.md
@./.claude/dave/templates/research.md
</execution_context>

<process>
Execute the research workflow from @./.claude/dave/workflows/research.md end-to-end.
Parse $ARGUMENTS for flags:
- If `--skip-arch` is present, skip the dave-architect agent.
- If `--topics` is present, parse the comma-separated list and research only those topics from DISCUSSION.md.
Preserve all workflow gates (state checks, DISCUSSION.md existence, user confirmations on new questions).
</process>
