---
alwaysApply: true
---

# Data Safety (ABSOLUTE RULES)

**Mindset: data is irreplaceable. Every destructive operation needs a safety net.**
Detail and recovery protocol in `.claude/rules/expert-knowledge.md` ("Assume You
Don't Have the Full Picture", "Commit Before the Worktree Disappears").

**Uncommitted changes are sacred.** Never delete or discard them. Work around them.

**Irreversible actions:**

| Category | Examples |
|---|---|
| Files | `rm`, `rm -rf`, overwrite, `mv` to unknown location |
| Git | `reset --hard`, `push --force`, branch/worktree deletion, **push to main** (always PR), `git checkout -- <file>` (discards work) |
| Local data / generated outputs | Losing uncommitted `tmp/` output (gitignored, dies with the worktree, unrecoverable from git) — commit or copy it out before teardown |

**Quick rules:** ask before any irreversible operation; prefer `--dry-run` when
available; back up before deletion; never push to `main` without a PR.
