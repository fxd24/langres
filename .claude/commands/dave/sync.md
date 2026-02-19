---
name: dave:sync
description: "Check framework drift and run the correct sync flow (project → dave-codes push or dave-codes → project sync)"
argument-hint: "[project-path]"
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
  - AskUserQuestion
---
<objective>
Run `dave-codes status` and orchestrate the right direction of sync:

1. If a project has framework changes that differ from dave-codes:
- Show the changed files
- Ask the user whether to push those changes to dave-codes
- If approved, run `dave-codes push <project-path>`, then create a dave-codes branch, commit, and open a PR

2. If dave-codes has framework changes not yet synced to projects:
- Show what changed
- Run `dave-codes sync` (or `dave-codes sync <project-path>`)
- Commit in each affected project repo

3. If everything is aligned:
- Report that projects are up to date

The CLI handles file operations only. Branching, commits, and PRs are handled here.
</objective>

<process>
1. Run `dave-codes status` (optionally scoped to `$ARGUMENTS` project path).
2. Classify drift direction from status output.
3. For project-newer or both-changed cases, ask before pushing to dave-codes.
4. For source-newer or source-only cases, run sync to projects.
5. Keep all mutations gated by explicit user confirmation before PR actions.
</process>
