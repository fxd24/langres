---
name: dave:verify
description: "Multi-layer verification (plan conformance, functional, qualitative, human oversight)"
argument-hint: "[--layer N] [--gaps-only]"
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
  - Task
  - AskUserQuestion
  - mcp__claude-in-chrome__tabs_context_mcp
  - mcp__claude-in-chrome__tabs_create_mcp
  - mcp__claude-in-chrome__navigate
  - mcp__claude-in-chrome__read_page
  - mcp__claude-in-chrome__find
  - mcp__claude-in-chrome__computer
  - mcp__claude-in-chrome__javascript_tool
  - mcp__claude-in-chrome__get_page_text
---
<objective>
Execute the verification matrix from PLAN.md across all four layers. Produces VERIFICATION.md with pass/fail results, evidence, and gap analysis.

**Requires:**
- Review complete (from `/dave:review`) — no "fix now" items remaining
- PLAN.md with `<verification-matrix>` section and `<must_haves>` section
- `.state/project/config.yaml` for available verification tools

**Creates:**
- `VERIFICATION.md` — Multi-layer verification results with evidence
- Gap closure fix plans (if gaps found)

**Flags:**
- `--layer N` — Run only a specific layer (1=plan-conformance, 2=code-review, 3=automated-functional, 4=human-oversight)
- `--gaps-only` — Re-verify only previously failed items (after gap closure fixes)

**After this command:** If all layers pass, run `/dave:push` for PR creation. If gaps found, fix and re-verify.
</objective>

<execution_context>
@./.claude/dave/workflows/verify.md
@./.claude/dave/templates/verification.md
@./.claude/dave/references/verification-matrix.md
</execution_context>

<process>
Execute the verify workflow from @./.claude/dave/workflows/verify.md end-to-end.
Parse $ARGUMENTS for flags:
- If `--layer N` is present, run only that specific layer.
- If `--gaps-only` is present, re-verify only previously failed items from VERIFICATION.md.
Preserve all workflow gates (state checks, PLAN.md existence, layer sequencing, gap closure loop).
</process>
