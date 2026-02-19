<purpose>
Execute the verification matrix from PLAN.md across all four layers. Each layer catches different classes of problems. Produces VERIFICATION.md with pass/fail results, evidence, and gap analysis. Manages gap closure if verification fails.

You are the verification orchestrator. You execute each layer systematically, collecting evidence at every step. You do not skip layers. For Layer 3, you operate autonomously: you read the actual implementation, discover available tools, generate verification strategies, and assess confidence per truth. If verification fails, you identify specific gaps and can loop back to Phase 4 for targeted fixes.
</purpose>

<downstream_awareness>
**Verification produces:**

1. **VERIFICATION.md** — Consumed by reflect (what verification caught), human (oversight checkpoints)
2. **Gap closure fix plans** — Fed back to Phase 4 (TDD) for targeted fixes
3. **Phase KNOWLEDGE.md entries** — Verification failures that reveal patterns

**The verification workflow reads:**
- PLAN.md must-haves (truths, artifacts, key links) and verification matrix
- REVIEWS.md (confirms Layer 2 gate — no fix-now items remaining)
- config.yaml (available verification tools)
- All source files modified during the phase
</downstream_awareness>

<required_reading>
Read before starting:
- `.claude/dave/templates/verification.md` — VERIFICATION.md structure
- `.claude/dave/references/verification-matrix.md` — all layer specs
- CLAUDE.md is auto-loaded by Claude Code
</required_reading>

<process>

## 1. Verify Prerequisites

**MANDATORY FIRST STEP — Check that review is complete and plan exists.**

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
     each branch's STATE.md points to that branch's active phase. -->

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

### 1d. Check Review Gate (Layer 2 prerequisite)

```bash
ls .state/milestones/{slug}/phases/{N}/REVIEWS.md 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If REVIEWS.md does not exist:**
```
Review not yet complete.
Run `/dave:review` first.
```
STOP HERE.

Read REVIEWS.md and check that "Fix now" count is 0. If not:
```
Review has {N} unresolved "fix now" items.
Run `/dave:review` to complete the fix loop first.
```
STOP HERE.

### 1e. Parse PLAN.md

Read PLAN.md and extract:
- `<must_haves>` section (truths, artifacts, key links)
- `<verification_matrix>` section (all four layers)

### 1f. Handle Flags

**`--layer N`:** If specified, execute only that layer:
- 1 = plan-conformance
- 2 = code-review (just checks REVIEWS.md gate)
- 3 = automated-functional
- 4 = human-oversight

**`--gaps-only`:** If specified:
- Read existing VERIFICATION.md
- Find all items with status FAILED
- Re-verify only those items
- Skip all passing items

### 1g. Read config.yaml

**Service Registry:** Read available services from `.state/project/config.yaml` for this phase.
Check `phases.verify.services` for verification tools. Build a tool availability map
by grouping available services by type:

```
tool_map = {}
for key in phases.verify.services:
  svc = services[key]
  if svc.available:
    tool_map[svc.type] = tool_map.get(svc.type, []) + [svc]
```

When multiple tools of the same type exist (e.g., chrome_mcp and playwright for browser),
use the one with the lowest `phases.verify.config.<key>.priority` value.
Use `phases.verify.config.<key>` for phase-specific overrides (e.g., sentry.check_mode).
Fall back to the `verification` section if the `services` registry is not present.

```yaml
# Legacy format (backward compatible):
verification:
  chrome_mcp: { available: true, type: browser }
  playwright: { available: false, type: browser }
  bash: { available: true, type: script }
  database: { available: true, type: query }
  docker: { available: true, type: container }
```

Build a tool availability map for Step 4 (automated functional).

---

## 2. Layer 1: Plan Conformance

**Goal:** Check must-haves from PLAN.md to determine whether the goal was actually achieved.

This layer requires NO external tools — only file system access and code analysis.

### 2a. Verify Truths

For each truth in `must_haves.truths`:

1. **Trace the codebase** — Read the relevant source files and determine whether the truth holds
2. **Look for supporting evidence** — Imports, function definitions, call sites, test assertions
3. **Check for contradicting evidence** — Stubs, TODOs, dead code paths, missing error handling

Classify each truth:

| Status | Criteria |
|--------|----------|
| VERIFIED | All supporting code paths exist, are complete, and are wired |
| FAILED | Code path is missing, uses stubs, or has broken wiring |
| UNCERTAIN | Cannot determine programmatically — will escalate to human |

Record evidence for each truth (specific file paths, line numbers, code snippets).

### 2b. Verify Artifacts

For each artifact in `must_haves.artifacts`:

**Level 1 — Exists:**
```bash
[ -f "{path}" ] && echo "EXISTS" || echo "MISSING"
```

**Level 2 — Substantive:**
```bash
wc -l "{path}"
```
Check: line count >= min_lines AND no stub patterns:
- Search for: `TODO`, `FIXME`, `HACK`, `PLACEHOLDER`, `raise NotImplementedError`, `pass` (as sole function body), `return None` (as sole function body)

**Level 3 — Wired:**
Search the codebase for imports/usage of this file:
```
grep -r "from {module}" --include="*.py" | grep -v "__pycache__" | grep -v ".pyc"
grep -r "import {module}" --include="*.py" | grep -v "__pycache__" | grep -v ".pyc"
```

Record: exists (yes/no), substantive (yes/no + line count), wired (yes/no + import locations).

### 2c. Verify Key Links

For each key link in `must_haves.key_links`:

1. Check that the `from` file imports or references the `to` file
2. Check that the connection mechanism described in `via` exists
3. Check that the connection is not commented out or behind dead code

Record: connected (yes/no) + evidence (import location, call site).

### 2d. Anti-Pattern Scan

Search all files modified during this phase for:

```bash
# Get list of modified files
MERGE_BASE=$(git merge-base HEAD main 2>/dev/null || git merge-base HEAD origin/main)
CHANGED_FILES=$(git diff --name-only $MERGE_BASE..HEAD -- '*.py')
```

For each changed file, search for:
- `TODO`, `FIXME`, `HACK`, `PLACEHOLDER`, `XXX`
- Empty implementations: `pass` as sole body, `return None` as sole body, `return {}`, `return []`
- Log-only error handlers: `except` blocks that only log without re-raising or handling
- `raise NotImplementedError`

Record: pattern, file, line number, context.

### 2e. Layer 1 Summary

Compile Layer 1 results:
```
Layer 1: Plan Conformance
- Truths: {N}/{M} verified
- Artifacts: {N}/{M} verified (all 3 levels)
- Key links: {N}/{M} connected
- Anti-patterns: {N} found
- Status: {passed | failed}
```

---

## 3. Layer 2: Code Review (Gate Check)

**Goal:** Confirm that the review phase completed and no "fix now" items remain.

This is NOT a re-review. Layer 2 results live in REVIEWS.md and OPEN_QUESTIONS.md (produced by `/dave:review`). This step simply confirms the review gate passed.

### 3a. Check REVIEWS.md

Read REVIEWS.md:
- Confirm "Fix now" count = 0
- Confirm fix loop converged
- Note defer count and dismissed count

### 3b. Check OPEN_QUESTIONS.md

Read OPEN_QUESTIONS.md:
- Confirm all questions have decisions (no "pending" items)
- Note any decisions that resulted in "fix" (should have been handled in fix loop)

### 3c. Layer 2 Summary

```
Layer 2: Code Review
- Fix now items remaining: 0
- Open questions resolved: {N}/{M}
- Status: see REVIEWS.md
```

---

## 4. Layer 3: Automated Functional (Autonomous Verification)

**Goal:** Autonomously verify each must-have truth using available tools. The verifier decides HOW to verify based on the actual implementation, not pre-scripted steps.

### 4a. Understand — Build Verification Context

Read the plan's must-haves and the actual implementation to understand what was built.

**Read plan context:**
- PLAN.md `<must_haves>` — truths, artifacts, key links (the contract)
- PLAN.md `<verifier_guidance>` hints (if present) — domain context from the planner (advisory only)

**Read implementation context:**

```bash
# Get the actual diff
MERGE_BASE=$(git merge-base HEAD main 2>/dev/null || git merge-base HEAD origin/main)
git diff --name-only $MERGE_BASE..HEAD -- '*.py'
```

For each changed file, read and understand:
- What classes/functions were created or modified
- What external systems they interact with (DB, APIs, file system)
- What error handling exists

**Read execution context:**
- EXECUTION_STATE.md — What tasks were executed, what deviations occurred
- CHANGE_SUMMARY.md (if exists) — Summary of what was actually built

**Map truths to implementation:**

For each truth, build a verification model:
- What code paths support this truth?
- What external systems are involved (database, APIs, file system, UI)?
- What are the likely failure modes?
- Did the implementation deviate from the plan? If so, how?

### 4b. Discover — Build Active Tool Inventory

Read `config.yaml` to discover what tools are available for verification.

**Service registry lookup:**

1. Read `phases.verify.services` for the list of service keys available to verification
2. For each service key, look up its full definition in the `services` section of `config.yaml`
3. Filter to services where `available: true`

**Health checks:**

For each available service that has a `test` field:
```bash
# Run the health check command
{service.test}
```

- If health check passes: add to active tool inventory
- If health check fails: mark as `degraded`, log warning, do NOT use for verification

**Build the active tool inventory:**

```
active_tools:
  - key: bash, type: script, capabilities: [run_command, check_exit_code, file_operations]
  - key: postgresql, type: database, capabilities: [select, count, verify_schema]
  - key: docker, type: container, capabilities: [build, run, compose]
  ...

degraded_tools:
  - key: chrome_mcp, reason: "health check failed — extension not running"
```

**Browser tool priority:** When multiple browser tools are available (chrome_mcp, playwright),
use the one with lowest `phases.verify.config.<key>.priority` value.

**Fallback:** If the services registry (`phases.verify.services`) is not present in `config.yaml`,
fall back to the legacy `verification` section for backward compatibility.

### 4c. Verify — Generate and Execute Strategies

For each truth in `must_haves.truths`, generate a verification strategy, execute it, and assess confidence.

**Strategy generation (for each truth):**

1. Analyze the truth statement
2. Identify what code paths support it (from 4a context)
3. Determine what verification signals are possible:
   - Can we run the relevant unit/integration tests? (bash)
   - Can we query the database for expected state? (database tool)
   - Can we exercise the code path directly? (bash)
   - Can we check the UI for expected behavior? (browser tool)
   - Can we query observability tools for traces/metrics? (langfuse)
4. Match signals to available tools from the active inventory (from 4b)
5. If a `<hint>` exists for this truth, incorporate its domain guidance
6. Generate a concrete strategy: ordered list of checks to execute

**Strategy execution (for each truth):**

1. Execute each check in the strategy
2. Collect evidence (command output, query results, screenshots)
3. Classify each check result:
   - **SUPPORTS** — Evidence confirms the truth
   - **CONTRADICTS** — Evidence refutes the truth
   - **INCONCLUSIVE** — Evidence is ambiguous or incomplete
4. If a check contradicts, investigate further before concluding FAILED
   - Could this be a false negative? (wrong table name, different API path)
   - Read the actual code again to understand the discrepancy

**Confidence assessment (for each truth):**

Based on the evidence chain, assign a confidence level:

| Confidence | Criteria |
|------------|----------|
| **HIGH** | 3+ independent signals, all converging, including at least one runtime signal (test execution, database query, API call) |
| **MEDIUM** | 2+ signals converging, OR 1 runtime signal without cross-validation |
| **LOW** | Only static analysis signals (code looks correct but was not exercised) |
| **UNABLE** | No signals could be gathered (tools unavailable, code unreadable) |

### 4d. Handle Escalations

**For truths with confidence UNABLE:**

Before escalating to human, exhaust alternatives:

1. **Alternative tool** — Can bash serve as a fallback? (e.g., Python one-liner via bash instead of database tool)
2. **Partial verification** — Can we verify a subset of the truth? (e.g., code analysis + unit tests, but not end-to-end pipeline run)
3. **Indirect verification** — Can we verify a consequence of the truth? (e.g., verify retry decorator is applied to the right methods, even if we cannot trigger a real rate limit)

If confidence remains UNABLE after alternatives:
- Document what was tried and why it failed
- Prepare evidence for human escalation
- Create a **dynamic human checkpoint** added to Layer 4:
  ```
  Dynamic Checkpoint: {truth statement}
  - Source: Verifier escalation (Truth {TN})
  - Why human review needed: {what could not be automated}
  - What to review: {specific action for human}
  - Evidence prepared: {what the verifier found}
  - Criteria: {what "good" looks like}
  - Verifier's partial confidence: {LOW or UNABLE}
  ```

**For truths with confidence LOW:**
- Document the limitation
- Include specific recommendations for what would raise confidence
- Flag for user awareness (not a full checkpoint, but a note in the summary)

### 4e. Layer 3 Summary

```
Layer 3: Automated Functional (Autonomous)
- Truths verified: {N}/{M}
- Confidence: {HIGH: N, MEDIUM: N, LOW: N, UNABLE: N}
- Composite confidence: {min of individual confidences, unless LOWs are deferred}
- Tools used: {list of tools actually used}
- Tool health: {all healthy | N degraded}
- Escalations: {N} truths escalated to human
- Status: {passed | degraded | failed}
```

**Status rules:**
- **passed:** All truths HIGH or MEDIUM confidence, no contradictions
- **degraded:** Some truths LOW confidence, no contradictions
- **failed:** Any truth has contradicting evidence, OR all truths UNABLE

---

## 5. Layer 4: Human Oversight

**Goal:** Present human oversight checkpoints — both **static** (from the plan's `<layer name="human-oversight">`) and **dynamic** (created by the verifier during Layer 3 escalations).

### 5a. Parse Checkpoints

**Static checkpoints (from plan):**

Read each `<checkpoint>` from the human-oversight layer. For each:
- Extract `<what>` — what to review
- Extract `<why>` — why it cannot be automated
- Extract `<evidence>` — what to prepare and present
- Extract `<criteria>` — what "good" looks like

**Dynamic checkpoints (from verifier):**

Collect all dynamic checkpoints created during Layer 3 (Step 4d). These are truths the verifier escalated because it could not reach sufficient confidence. Each includes pre-prepared evidence from the verifier's investigation.

### 5b. Prepare Evidence

For each checkpoint, prepare the evidence BEFORE presenting to the human:
- If evidence requires screenshots → take them using browser tools
- If evidence requires data samples → query and format them
- If evidence requires file content → read and present relevant sections
- If evidence requires comparisons → prepare side-by-side views

### 5c. Present Checkpoints

Present each checkpoint to the human with prepared evidence:

```
## Human Review Checkpoint {N}/{M}

**What to review:** {what}
**Why this needs your eyes:** {why}
**Criteria:** {criteria}

### Evidence
{prepared evidence — screenshots, data, comparisons}

Does this pass your review?
```

Use AskUserQuestion for each checkpoint:
- Options: Pass, Fail (with notes), Skip (with reason)

### 5d. Record Results

For each checkpoint, record in VERIFICATION.md:
- Status: PASSED / FAILED / SKIPPED
- Human notes (if any)
- Date

### 5e. Layer 4 Summary

```
Layer 4: Human Oversight
- Static checkpoints: {N}/{M} completed (from plan)
- Dynamic checkpoints: {N}/{M} completed (from verifier escalations)
- Passed: {N}
- Failed: {N}
- Skipped: {N}
- Status: {passed | pending | failed}
```

---

## 6. Compile VERIFICATION.md

### 6a. Write VERIFICATION.md

Combine all layer results into VERIFICATION.md following the template from `.claude/dave/templates/verification.md`.

Determine overall status:
- **passed** — All layers pass, no gaps
- **gaps_found** — One or more layers failed, gaps identified
- **human_needed** — Human checkpoints still pending

Calculate score: {N}/{M} must-haves verified (from Layer 1 truths).

### 6b. Identify Gaps

For any failed items across all layers, create entries in the gaps table:
- Which truth or step failed
- Which layer it failed in
- Why it failed
- Proposed fix

---

## 7. Gap Closure (if gaps found)

### 7a. Present Gaps to User

```
## Verification Gaps Found

{N} items failed verification:

| # | Item | Layer | Reason | Proposed Fix |
|---|------|-------|--------|-------------|
{gap table}

Options:
1. Fix all gaps (loop back to TDD for targeted fixes, then re-verify)
2. Defer specific gaps (mark as accepted with rationale)
3. Investigate further (get more details before deciding)
```

Use AskUserQuestion to determine how to proceed with each gap.

### 7b. Fix Gaps (if user chooses to fix)

For each gap to fix:
1. Create a focused fix task (Files, Action, Verify, Done)
2. Launch tdd-developer for the fix
3. Launch practical-verifier after the fix
4. Commit the fix atomically:
   ```
   fix({phase}-verify): {gap description}
   ```

### 7c. Re-Verify Gaps Only

After fixes are committed, re-verify ONLY the failed items:
- Re-run the specific Layer 1 checks that failed
- Re-run the specific Layer 3 steps that failed
- Update VERIFICATION.md with new results

This is equivalent to running the workflow with `--gaps-only`.

### 7d. Record Gap Closure

Update VERIFICATION.md:
- Gap closure history section
- Updated status for re-verified items
- New overall status

---

## 8. Gate Check

Before declaring verification complete, verify:

1. **All four layers executed:** (or explicitly scoped with `--layer`)
2. **Layer 1 pass:** All truths VERIFIED or UNCERTAIN (no FAILED)
3. **Layer 2 pass:** Review gate confirmed (no fix-now items)
4. **Layer 3 pass:** All truths HIGH or MEDIUM confidence, no contradictions (status: passed or degraded)
5. **Layer 4 pass:** All checkpoints completed — both static (from plan) and dynamic (from verifier escalations) (no pending)
6. **VERIFICATION.md complete:** All sections filled, evidence recorded, gaps resolved or deferred

### Gate Passed

```
## Verification Complete

All {N} must-haves verified. All automated checks pass.
Human checkpoints completed.

**Verification confidence:** {HIGH | MEDIUM | LOW}

**Ready for push.** Run `/dave:push` for PR creation and CI monitoring.
```

### Gate Failed

If any gate condition is not met, explain:
- Which layers failed
- Which specific items failed
- What action is needed (fix gaps, resolve checkpoints, etc.)

---

## 9. Update STATE.md

Update `.state/STATE.md` with:
- Current phase status: "verification complete" or "verification gaps found"
- Verification summary (score, confidence, gaps)
- Next action: `/dave:push` (if passed) or gap closure (if gaps found)

</process>
