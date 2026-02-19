<purpose>
Run parallel code reviews (internal agents + external models), then aggregate findings intelligently using project context. Produces REVIEWS.md (triaged findings) and OPEN_QUESTIONS.md (ambiguous items for human review). Manages the fix loop if "fix now" items exist.

You are the review orchestrator. You launch reviewers in parallel, collect their findings, and hand them to the review aggregator agent. You manage the fix loop if needed. You do not review code yourself.
</purpose>

<downstream_awareness>
**Review produces:**

1. **REVIEWS.md** — Consumed by executor (fix loop), verifier (confirms fixes), reflect (patterns)
2. **OPEN_QUESTIONS.md** — Consumed by human (decisions), reflect (knowledge extraction)
3. **Phase KNOWLEDGE.md entries** — Decisions from open question resolution

**The review aggregator receives:**
- All reviewer findings (internal + external)
- CHANGE_SUMMARY.md (structured diff summary, replaces raw diff)
- PLAN.md (what was built and why)
- project/KNOWLEDGE.md (Tier 1 rules)
- project/PATTERNS.md (project conventions)
- Phase DISCUSSION.md (scope decisions)

**The fix loop feeds back to:**
- Phase 4 (TDD) for targeted fixes
- Scoped re-review (fixes only, not full codebase)
</downstream_awareness>

<required_reading>
Read before starting:
- `.claude/dave/templates/reviews.md` — REVIEWS.md structure
- `.claude/dave/templates/open-questions.md` — OPEN_QUESTIONS.md structure
- `.claude/dave/references/verification-matrix.md` — Layer 2 spec
- CLAUDE.md is auto-loaded by Claude Code
</required_reading>

<process>

## 1. Verify Prerequisites

**MANDATORY FIRST STEP — Check that implementation is complete and a plan exists.**

### 1a. Check Project State

```bash
ls -d .state/project/ 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If `.state/project/` does not exist:**
```
Project state not found.
Run `/dave:init` first to initialize the Dave Framework.
```
STOP HERE.

### 1b. Locate Phase Directory

<!-- PARALLEL WORKTREE NOTE: STATE.md is per-branch. In a parallel worktree setup,
     each branch's STATE.md points to that branch's active phase. This correctly
     resolves to the right phase directory for THIS worktree's session. -->

Find the active milestone and phase:

```bash
cat .state/STATE.md 2>/dev/null
```

Read STATE.md to determine the current milestone slug and phase number. Construct the phase path:
`.state/milestones/{slug}/phases/{N}/`

### 1c. Check PLAN.md Exists

```bash
ls .state/milestones/{slug}/phases/{N}/PLAN.md 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If PLAN.md does not exist:**
```
No plan found for the current phase.
Run `/dave:plan` first.
```
STOP HERE.

### 1d. Parse the Code-Review Layer

Read PLAN.md and extract the `<layer name="code-review">` section from the verification matrix:
- `<agents>` — which internal reviewers to launch
- `<focus>` — what to focus reviews on
- `<external>` — which external models to use
- `<skip_external_if>` — condition for skipping externals

### 1e. Check for --fix-only Flag

If `--fix-only` was passed:
- Read the existing REVIEWS.md to find "fix now" items
- Identify which files were changed as fixes
- Scope the review to ONLY those files
- Skip external reviews (fixes are scoped)
- Jump to Step 3 (run scoped review)

### 1f. Check for --skip-external Flag

If `--skip-external` was passed:
- Set external_models = [] (empty, skip all externals)

---

## 2. Prepare Review Context

### 2a. Determine Changed Files

Identify all files changed during execution. Use git diff against the branch point:

```bash
# Find the merge base with main
MERGE_BASE=$(git merge-base HEAD main 2>/dev/null || git merge-base HEAD origin/main)
git diff --name-only $MERGE_BASE..HEAD
```

Store the list of changed files. This is what reviewers will examine.

### 2b. Read Project Context

Read these files for the aggregator (and pass relevant parts to reviewers):

| File | Purpose |
|------|---------|
| `project/KNOWLEDGE.md` | Tier 1 rules reviewers must know about |
| `project/PATTERNS.md` | Conventions reviewers need context on |
| Phase `PLAN.md` | What was built and why |
| Phase `DISCUSSION.md` | Scope decisions and guardrails |

### 2c. Compute Change Size

Count the total lines changed to determine if external reviews should be skipped:

```bash
git diff --stat $MERGE_BASE..HEAD | tail -1
```

If the `<skip_external_if>` condition is met (e.g., "changes are less than 50 lines"), set external_models = [].

### 2d. Generate the Diff

Prepare the diff for reviewers:

```bash
git diff $MERGE_BASE..HEAD
```

For large diffs, also prepare per-file diffs for focused review.

### 2e. Generate Change Summary

Launch the `dave-change-summarizer` agent via Task tool with `subagent_type: "general-purpose"` to produce a structured summary of the diff. This runs IN PARALLEL with internal reviewers.

**Prompt:**
```
Summarize the following code changes for Phase {N}: {phase name}.

## Changed Files
{list of changed files from Step 2a}

## Diff
{the diff from Step 2d}

## PLAN.md
{content of PLAN.md}

## Output Directory
Write CHANGE_SUMMARY.md to: {phase directory path}
```

The summarizer runs in the background while reviewers work. Its output is consumed by the aggregator in Step 5.

After the summarizer completes (collected alongside reviewer results), read the generated CHANGE_SUMMARY.md. Store its content for use in external reviewer prompts (Steps 4b) and the aggregator (Step 5a).

**Fallback:** If the summarizer fails or CHANGE_SUMMARY.md is not produced, fall back to passing the raw diff directly to external reviewers and the aggregator (original behavior). Log a warning:
```
WARNING: Change summarizer did not produce CHANGE_SUMMARY.md. Falling back to raw diff for reviewers.
```

---

## 3. Launch Internal Reviews (Parallel)

Launch internal reviewer agents in PARALLEL. Each reviewer runs independently. The change summarizer from Step 2e is also running in parallel at this point — internal reviewers do not depend on the change summary (they work from the raw diff or their own file reads). The summarizer result is collected alongside reviewer results for use by the aggregator in Step 5.

### 3a. Determine Which Internal Reviewers

From the `<agents>` field in the code-review layer. Always includes `code-reviewer`. Others are selected based on what the phase touches:

| Agent | When Selected | Subagent Type |
|-------|---------------|---------------|
| `code-reviewer` | Always | `code-reviewer` |
| `security-reviewer` | Plan includes auth, external input, API endpoints, file handling | `security-reviewer` |
| `data-pipeline-reviewer` | Plan touches Dagster assets or pipeline code | `data-pipeline-reviewer` |
| `database-expert` | Plan includes schema changes or new queries | `database-expert` |

### 3b. Launch Each Internal Reviewer

For each reviewer agent, launch as a Task with:

**If CHANGE_SUMMARY.md exists (preferred):**

```
Review the following code changes for {phase name}.

## Change Summary
{CHANGE_SUMMARY.md content}

## Focus Areas (from plan)
{focus items from <focus> in verification matrix}

## Suggested Focus for You
{the reviewer-specific focus section from the Change Summary's "Suggested Review Focus"}

## Project Context
{Tier 1 rules from KNOWLEDGE.md that are relevant}
{Key patterns from PATTERNS.md}

## Instructions
- Review against the focus areas above
- **Read specific files** for any change marked "significant" or "complex" in the summary
- **Read specific files** for any area of concern listed in the summary
- Pay special attention to "unplanned changes" flagged in the summary
- For each finding, provide:
  - File and line range
  - What is wrong
  - Severity (critical / high / medium / low)
  - Category (bug / security / correctness / data-integrity / performance / maintainability / style)
  - Suggested fix
- Be specific — no vague "could be improved" without saying HOW
- If you are unsure about something, flag it anyway with a confidence note
```

**Fallback — if CHANGE_SUMMARY.md does not exist:**

```
Review the following code changes for {phase name}.

## Focus Areas (from plan)
{focus items from <focus> in verification matrix}

## Project Context
{Tier 1 rules from KNOWLEDGE.md that are relevant}
{Key patterns from PATTERNS.md}

## Changed Files
{list of changed files}

## Diff
{the diff}

## Instructions
- Review against the focus areas above
- Flag anything that violates the project rules listed above
- For each finding, provide:
  - File and line range
  - What is wrong
  - Severity (critical / high / medium / low)
  - Category (bug / security / correctness / data-integrity / performance / maintainability / style)
  - Suggested fix
- Be specific — no vague "could be improved" without saying HOW
- If you are unsure about something, flag it anyway with a confidence note
```

Launch ALL internal reviewers in parallel using the Task tool.

### 3c. Collect Internal Results

Wait for all internal reviewers to complete. Collect all findings. If any reviewer failed, include failure notices.

### 3d. Handle Reviewer Failures

If any internal reviewer agent fails (crashes, times out, or returns no findings):

1. **Log the failure:** Record which reviewer failed and why
2. **Continue with remaining reviewers:** Do not block on a single failure
3. **Notify aggregator:** Include a `REVIEWER_FAILED: {agent_name} — {reason}` entry in the findings passed to the aggregator
4. **Reduce confidence:** The aggregator should note that consensus is based on fewer reviewers, which may reduce confidence in findings that lack multi-reviewer corroboration

Do NOT retry failed reviewers — the remaining reviewers provide sufficient coverage. If ALL reviewers fail, STOP and escalate to the user.

---

## 4. Launch External Reviews (Parallel, if applicable)

### 4a. Check External Model Availability

**Service Registry:** Read available services from `.state/project/config.yaml` for this phase.
Check `phases.review.services` for review tools. Group available services by capability:
- `code_reviewers` = services where `code-review` in capabilities (ai-model + code-review types)
- `error_monitors` = services where `error-tracking` in capabilities (e.g., Sentry)
- `tracers` = services where `prompt-tracing` in capabilities (e.g., Langfuse)
Use `phases.review.config.<key>` for phase-specific overrides (e.g., prompt_prefix, review_mode).
If `phases.review.primary` is set and available, prefer that service for code review.
Fall back to the `review_models` section if the `services` registry is not present.

```yaml
# Legacy format (backward compatible):
review_models:
  - name: codex
    command: "codex exec ..."
    available: true
  - name: kimi
    command: "opencode run ..."
    available: true
```

Skip if external_models is empty (--skip-external or size threshold).

### 4b. Build Self-Contained Review Prompts for External Models

**CRITICAL:** External models (codex, opencode) do NOT have subagents. They cannot read files, search the codebase, or ask follow-up questions. The review prompt MUST be completely self-contained — everything the model needs to perform a quality review must be in the prompt itself.

**If CHANGE_SUMMARY.md exists (preferred):**

For each external model, construct a summary-based prompt with targeted file excerpts:

```markdown
# Code Review Request

## What Was Built

{Phase name and description from PLAN.md}
{1-2 paragraph summary of what this feature does, from must-haves}

## Why It Was Built This Way

{Key architectural decisions from DISCUSSION.md and RESEARCH.md}
{Pattern choices and their rationale}

## Change Summary

{CHANGE_SUMMARY.md content}

## Targeted File Excerpts

<!-- Include excerpts ONLY for files rated "significant" or "complex" in the
     Change Summary. For each such file, include the changed regions plus
     ~10 lines of surrounding context. For new files, include full content
     since the diff IS the file. Trivial and straightforward files are
     summary-only — no excerpts needed. -->

### {file_path} (significant)
```{language}
{changed region with surrounding context, from the diff or file read}
```

### {file_path} (complex)
```{language}
{changed region with surrounding context}
```

{...repeat for all significant/complex files...}

## Areas of Concern (from Change Summary)

{Copy the "Areas of Concern" section from CHANGE_SUMMARY.md — these are
 the highest-priority items for review}

## Project Rules (MUST check against these)

These are non-negotiable project conventions. Flag ANY violation.

{Full Tier 1 entries from KNOWLEDGE.md — include rule text, not just IDs}

## Project Patterns (context for review)

These are intentional patterns — do NOT flag these as issues:

{Key patterns from PATTERNS.md that reviewers might otherwise flag}

## Review Focus Areas

{Focus items from <focus> in verification matrix}

## Suggested Focus for External Models

{The "For external models" section from CHANGE_SUMMARY.md's Suggested Review Focus}

## What NOT to Review

- Files rated "trivial" in the Change Summary (unless they appear in Areas of Concern)
- Patterns listed above as intentional project conventions
- Style preferences that contradict the project's established patterns
- Suggestions to add features or capabilities beyond the scope described above

## Expected Output Format

For each finding, provide:

### Finding {N}
- **File:** {path}:{line range}
- **Severity:** critical | high | medium | low
- **Category:** bug | security | correctness | data-integrity | performance | maintainability
- **What is wrong:** {specific description}
- **Suggested fix:** {concrete suggestion}
- **Confidence:** {how sure you are this is a real issue, not a false positive}

If no issues found, state "No issues found" with a brief explanation of what was checked.
```

**Fallback — if CHANGE_SUMMARY.md does not exist:**

Use the original raw-diff-based prompt:

```markdown
# Code Review Request

## What Was Built

{Phase name and description from PLAN.md}
{1-2 paragraph summary of what this feature does, from must-haves}

## Why It Was Built This Way

{Key architectural decisions from DISCUSSION.md and RESEARCH.md}
{Pattern choices and their rationale}

## Files Changed

{List of all changed files with a 1-line description of what changed in each}

## The Diff

{Complete git diff}

## Project Rules (MUST check against these)

These are non-negotiable project conventions. Flag ANY violation.

{Full Tier 1 entries from KNOWLEDGE.md — include rule text, not just IDs}

## Project Patterns (context for review)

These are intentional patterns — do NOT flag these as issues:

{Key patterns from PATTERNS.md that reviewers might otherwise flag}

## Review Focus Areas

{Focus items from <focus> in verification matrix}

## What NOT to Review

- Files not in the changed files list
- Patterns listed above as intentional project conventions
- Style preferences that contradict the project's established patterns
- Suggestions to add features or capabilities beyond the scope described above

## Expected Output Format

For each finding, provide:

### Finding {N}
- **File:** {path}:{line range}
- **Severity:** critical | high | medium | low
- **Category:** bug | security | correctness | data-integrity | performance | maintainability
- **What is wrong:** {specific description}
- **Suggested fix:** {concrete suggestion}
- **Confidence:** {how sure you are this is a real issue, not a false positive}

If no issues found, state "No issues found" with a brief explanation of what was checked.
```

### 4c. Launch External Reviews

For each available external model, write the prompt to a temporary file and execute:

```bash
# Write self-contained prompt
cat > /tmp/dave-review-{model_name}.md << 'PROMPT'
{the complete self-contained prompt from 4b}
PROMPT

# Execute the external model command with the prompt file
{command from config.yaml} "$(cat /tmp/dave-review-{model_name}.md)"
```

**External models are READ-ONLY** — they analyze and suggest, never modify code.

Launch ALL external reviews in parallel using the Task tool (Bash commands).

### 4d. Collect External Results

Wait for all external reviews to complete. Collect all findings.

---

## 5. Aggregate Findings

### 5a. Launch Review Aggregator Agent

Launch the `dave-review-aggregator` agent (from `.claude/agents/dave-review-aggregator.md`) with:

**Prompt:**
```
Aggregate these review findings for Phase {N}: {phase name}.

## Reviewer Findings

### {reviewer 1 name}
{findings from reviewer 1}

### {reviewer 2 name}
{findings from reviewer 2}

### {external model 1 name}
{findings from external model 1}

...

## Change Summary
<!-- If CHANGE_SUMMARY.md exists, include it here instead of the raw diff.
     This gives the aggregator the semantic map of what changed and why,
     which is sufficient for triaging findings. If CHANGE_SUMMARY.md does
     not exist, include the raw diff as a fallback. -->
{CHANGE_SUMMARY.md content OR raw diff if summary unavailable}

## Project Context

### KNOWLEDGE.md (Tier 1 rules)
{content of project KNOWLEDGE.md}

### PATTERNS.md (conventions)
{content of project PATTERNS.md}

### PLAN.md must-haves
{must_haves section from PLAN.md}

### DISCUSSION.md decisions
{key decisions from DISCUSSION.md}

## Output Directory
Write REVIEWS.md and OPEN_QUESTIONS.md to: {phase directory path}

## Templates
Follow the REVIEWS.md template from .claude/dave/templates/reviews.md
Follow the OPEN_QUESTIONS.md template from .claude/dave/templates/open-questions.md
```

### 5b. Validate Aggregation Output

After the aggregator completes, verify:
1. REVIEWS.md exists in the phase directory
2. OPEN_QUESTIONS.md exists in the phase directory
3. Summary counts in REVIEWS.md match actual findings
4. Every finding from every reviewer is accounted for

---

## 6. Present Summary (Non-Blocking)

### 6a. Show Summary

Display the review summary:

```
## Review Results

**Reviewers:** {list}
**Total findings:** {N}

| Category | Count | Action |
|----------|-------|--------|
| Fix now | {N} | Autonomous fix loop (parallel where independent) |
| Defer | {N} | Create GitHub issues |
| Dismissed | {N} | False positives (explained in REVIEWS.md) |
| Open questions | {N} | Collected for human review after fixes complete |
```

### 6b. Collect Open Questions (Non-Blocking)

**IMPORTANT: Open questions do NOT block the fix loop.** If OPEN_QUESTIONS.md has pending items:

1. Record all open questions in OPEN_QUESTIONS.md with the aggregator's best guess
2. Proceed immediately to the fix loop (Step 7)
3. Present open questions to the human AFTER all autonomous fixes are done (Step 8)

This prevents human review from blocking autonomous work that can proceed independently.

---

## 7. Fix Loop (Parallel, Autonomous)

### 7a. Check for Fix Now Items

Read REVIEWS.md. If "fix now" count > 0:

```
## Fix Required

{N} items need fixing before verification can proceed:

{list of fix-now items with IDs}

Launching autonomous fix loop — independent fixes run in parallel.
```

### 7b. Create Focused Fix Plan

For each "fix now" item, create a mini-task:
- **Files:** From the finding location
- **Action:** From the suggested fix
- **Verify:** The specific check that proves the fix works
- **Done:** The finding is resolved

### 7c. Analyze Fix Dependencies

Before launching fixes, determine which are independent and which depend on each other:

- **Independent fixes:** Touch different files, address different concerns → can run in parallel
- **Dependent fixes:** Touch the same file, or fix B depends on fix A's output → must run sequentially

Organize fixes into dependency waves (same pattern as execution waves):

```
Fix Wave 1 (parallel): [Fix A (file1.py), Fix B (file3.py), Fix C (file5.py)]
Fix Wave 2 (sequential): [Fix D (file1.py — same file as Fix A, depends on A)]
```

### 7d. Execute Fixes (Parallel TDD)

**Launch independent fixes in parallel using the Task tool.** Each fix gets its own TDD executor agent.

For each fix task, launch a TDD executor agent with:
- The specific fix task (Files, Action, Verify, Done)
- Project KNOWLEDGE.md and PATTERNS.md
- The original finding for context
- Clear instruction: "This is a SCOPED fix — address ONLY the specific finding. Do not refactor surrounding code."

After each fix, launch a `practical-verifier` to confirm the fix.

Commit each fix atomically:
```
fix({phase}-review): {finding ID} {brief description}
```

**Parallelism within fix waves:**
```
Fix Wave 1 (all independent — launch in parallel):
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ tdd-exec: Fix A  │  │ tdd-exec: Fix B  │  │ tdd-exec: Fix C  │
│ Finding: F001    │  │ Finding: F003    │  │ Finding: F005    │
│ RED → GREEN      │  │ RED → GREEN      │  │ RED → GREEN      │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         ▼                     ▼                     ▼
     verify + commit       verify + commit       verify + commit

Fix Wave 2 (depends on Wave 1):
┌─────────────────┐
│ tdd-exec: Fix D  │
│ Finding: F002    │
│ Depends on: F001 │
└────────┬────────┘
         ▼
     verify + commit
```

### 7e. Scoped Re-Review

After all fixes are committed, run a SCOPED re-review:
- Only review the files changed by the fixes
- Only use internal reviewers (skip external for re-review)
- Only check focus areas related to the fixes

This is equivalent to running the workflow again with `--fix-only`.

### 7f. Check Convergence

After scoped re-review:
- If new "fix now" items = 0 → Fix loop converged. Proceed to Step 8.
- If new "fix now" items > 0 but fewer than before → Continue loop (iteration 2, same parallel pattern)
- If iteration 3+ and still finding new issues → STOP, escalate to user

```
## Fix Loop Not Converging

After {N} iterations, the fix loop is still producing new findings.
This suggests a structural issue, not a simple fix.

Latest findings: {list}

How would you like to proceed?
```

### 7g. Update REVIEWS.md

After the fix loop converges, update REVIEWS.md:
- Update fix loop history section
- Mark fix-now items as resolved
- Record any new findings from re-review iterations

---

## 8. Present Open Questions to Human

**This step runs AFTER the autonomous fix loop completes.** The human is not blocked during fixes.

### 8a. Check for Pending Open Questions

Read OPEN_QUESTIONS.md. If there are pending items:

```
## Open Questions

All autonomous fixes are complete. I need your input on {N} ambiguous findings:

### Q001: {title}
{Finding description}
**My best guess:** {aggregator's guess}
**What I need from you:** {what would resolve it}
```

Use AskUserQuestion for each open question. Record decisions in OPEN_QUESTIONS.md.

### 8b. Record Decisions

For each resolved open question:
1. Update OPEN_QUESTIONS.md with the decision, rationale, and date
2. If the decision reveals a new convention → add to phase KNOWLEDGE.md
3. If the decision resulted in a "fix" → add to the fix list and run one more fix iteration

### 8c. Handle Decision-Triggered Fixes

If any open question resolution requires code changes:
1. Create fix tasks from the decisions
2. Run a focused fix round (same parallel pattern as Step 7)
3. Run scoped re-review on those fixes only
4. Update REVIEWS.md

---

## 9. Gate Check

Before declaring review complete, verify:

1. **All reviews ran:** Internal reviewers + external models (unless skipped per rules)
2. **Findings triaged:** Every finding is classified (fix/defer/dismiss/open-question)
3. **No "fix now" items remain:** All have been addressed via the fix loop
4. **Open questions resolved:** Human has decided on all items in OPEN_QUESTIONS.md
5. **REVIEWS.md complete:** Summary counts match, fix loop history recorded
6. **OPEN_QUESTIONS.md complete:** All decisions recorded with rationale

### Gate Passed

```
## Review Complete

All reviews ran. Findings triaged. Fix loop converged ({N} iterations).
Open questions resolved ({N} decisions made).

**Ready for verification.** Run `/dave:verify` to execute the verification matrix.
```

### Gate Failed

If any gate condition is not met, explain what is missing and what action is needed.

---

## 10. Update STATE.md

Update `.state/STATE.md` with:
- Current phase status: "review complete"
- Review summary (counts, iterations)
- Next action: `/dave:verify`

</process>
