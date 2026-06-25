---
paths:
  - ".claude/agents/**"
  - ".claude/skills/**"
  - ".claude/commands/**"
---

# Token Efficiency: Agent Cost Discipline

**Every agent call has a cost. Minimize tokens without reducing signal.**

These rules apply to every agent prompt, orchestrator command, and multi-agent pipeline in this repo. Read this before writing or modifying any agent definition under `.claude/agents/`.

## The core leaks

Most token waste in an agent pipeline comes from a small set of repeatable mistakes:

1. **Full-file rewrites when a targeted patch would do.** A downstream fix touching 3 lines triggers a full-file `Write` of a 400-line document.
2. **Full-file reads when a targeted range would do.** A reviewer reads a 700-line sibling module to check whether one symbol appears.
3. **Markdown between agents.** Two agents talk to each other in prose, and the consumer must parse + re-narrate.
4. **Re-running dynamic setup on stable data.** Each invocation re-reads a reference doc or template that hasn't changed in months.
5. **Self-reassurance re-reads.** Agents told to "read the file you just wrote to verify it landed" — the harness already errors on failed writes.
6. **Top-tier reasoning for mechanical work.** Dedup, classification-into-buckets, counting, and schema-filling don't need top-tier reasoning.

## Rules

### R1. Edit over Write when the patch is known

If the consumer gave you exact `from` / `to` strings (as a structured fixes manifest does), use `Edit(file, old_string, new_string)`. Never `Read` then `Write` the whole file.

Rewrites are allowed only for initial composition, not for patching.

### R2. Structured JSON between agents; Markdown only for humans

If the consumer of your output is another agent, emit JSON. If the consumer is a human reading the artifact, emit Markdown. A single file should not serve both — that's how you end up with 150-line Markdown wrappers around 30 lines of structured data.

The narrative fields inside the JSON (e.g. `evidence`, `rationale`) stay as strings — don't squeeze them to save tokens; the value is the narrative.

### R3. Grep-before-Read with a read budget

When reviewing or cross-checking another file:

- Read your **primary** file fully.
- For **sibling** files, Grep for the specific entity or claim you're verifying, then Read the matching line range (±20 lines).
- A sibling file under ~150 lines may be full-read.
- If you want to full-read a 200+ line sibling file, stop and name the specific entity or claim you're verifying. If you can't name it, you don't need the read.

This is a **directed-search discipline**, not a cap. Every finding starts with a claim in your primary file; you can't file a finding about something you can't name.

### R4. Do not re-read files you just wrote

`Write` and `Edit` error on failure. A re-read to "verify the file landed" is pure waste.

Keep logical self-checks (checklists about content coverage). Remove re-reads.

### R5. Inline stable reference docs into agent system prompts

If an agent reads the same reference file every invocation (e.g. a definitions doc or a fixed template), inline it into the agent's system prompt. The reference then benefits from prompt-cache retention across invocations and costs no tool call.

Mark inlined content with a sync marker so drift is detectable — e.g.:

```
<inlined source="docs/SOME_REFERENCE.md" synced="YYYY-MM-DD">
…content…
</inlined>
```

When the source changes, re-sync the inlined copy and bump the `synced` date.

### R6. Keep system prompts static; inject dynamic data via user turn

Domain, working directory, current date, file paths — these go in the user-turn task prompt, never in the agent's frontmatter or system prompt. Mutating the system prompt per invocation defeats prompt caching.

### R7. Reasoning-tier discipline

Use the top reasoning tier only when the agent is genuinely reasoning: classification with edge cases, cross-domain merging, risk judgment.

Use a lower tier for mechanical work: dedup against a schema, classification into a fixed bucket list, counting, targeted patching when `from`/`to` are given.

When in doubt, start lower and raise if output quality regresses.

### R8. Grant each agent the tools it actually needs

> **Scope:** This rule applies to **subagent frontmatter** — the `tools:` line in a `.claude/agents/*.md` definition. It has nothing to do with the main Claude Code session's tool access.

**The goal is minimum-necessary permissions, not minimum permissions.** An agent that can't do its job is not a token-efficient agent — it's a broken agent that gets retried or worked around.

Decide per tool by asking: *does this agent actually use it?*

- Writes/patches files → `Edit`, `Write`
- Reads code, docs, artifacts → `Read`, `Grep`, `Glob`
- Runs shell commands (test runs, git, `uv run`, scripts) → `Bash`
- Fetches URLs → `WebFetch`
- Spawns other agents → `Agent`

**Do not deny tools reflexively.** If you're unsure whether an agent needs `Bash`, check how similar agents in the same pipeline use it. Verification-oriented agents (reviewers, comparators, validators) almost always need `Bash` to run tests or scripts. Denying it forces them to guess instead of checking.

Smaller tool lists do reduce per-invocation prompt size, but only marginally. The real cost of a wrong permission block is an agent that silently can't verify its own output.

### R9. Single-pass review when downstream changes are mechanical

If the fix step is targeted Edits (R1), the output cannot drift in prose the way a full rewrite can. A second semantic review pass adds cost without catching meaningful issues. Replace it with a shell-level sanity check that greps for the expected `to` anchors.

### R10. One read-budget escape hatch, not many

If a rule like R3 feels too tight for a specific case, add a single explicit exception in the agent prompt (e.g. "small cross-section files under 150 lines may be full-read"). Do not add layered exceptions — they get ignored under pressure.

## When you write a new agent

- [ ] Identify the consumer: human or agent? (sets R2)
- [ ] Identify stable references and inline them (R5)
- [ ] Set the reasoning tier to the lowest that produces acceptable output (R7)
- [ ] Grant the tools the agent actually needs; don't deny reflexively (R8). Reviewers/comparators/validators usually need `Bash` to run tests or scripts.
- [ ] No re-read-after-write instructions (R4)
- [ ] Mechanical downstream? Skip the second review pass (R9)

## When you modify an existing agent

Before changing reasoning instructions, check whether a cheaper path exists: is the input too large (R3), is the output unnecessarily prose (R2), is the system prompt doing dynamic work (R6)? A prompt rewrite is more expensive to validate than a structural change.
