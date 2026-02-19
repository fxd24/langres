---
name: dave:push
description: "Push branch, create PR with structured description, monitor CI"
argument-hint: "[--wait] [--draft] [--no-pr]"
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
Push the feature branch and create a pull request with a structured description derived from phase artifacts. Optionally monitor CI status.

**Requires:**
- Verification complete (from `/dave:verify`) — VERIFICATION.md exists with passing status
- Feature branch (not main/master)
- Clean working tree (no uncommitted changes)

**Creates:**
- Remote branch push
- Pull request with structured description (summary, must-haves, test plan, review summary)
- STATE.md update with PR URL

**Flags:**
- `--wait` — After creating PR, poll CI checks until they complete (or timeout)
- `--draft` — Create the PR as a draft
- `--no-pr` — Push branch only, skip PR creation

**After this command:** Run `/dave:reflect` for the learning loop, or merge the PR.
</objective>

<execution_context>
@./.claude/dave/workflows/push.md
@./.claude/dave/templates/verification.md
@./.claude/dave/templates/reviews.md
</execution_context>

<process>
Execute the push workflow from @./.claude/dave/workflows/push.md end-to-end.
Parse $ARGUMENTS for flags:
- If `--wait` is present, poll CI checks after PR creation until they complete.
- If `--draft` is present, create the PR as a draft.
- If `--no-pr` is present, push the branch but skip PR creation.
Preserve all workflow gates (state checks, verification gate, clean working tree, branch safety).
</process>
