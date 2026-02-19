# Verification Matrix Reference

Detailed specification for the verification matrix in PLAN.md and how it is executed during Phase 6 (Verification).

---

## Overview

The verification matrix is an XML structure embedded in PLAN.md that defines exactly HOW to verify the work. It has four layers, each catching different classes of problems. The planner designs it during Phase 3. The verifier executes it during Phase 6.

```
Layer 1: Plan Conformance       -- Did we build what we said we would?
Layer 2: Code Review            -- Is the code correct and well-designed?
Layer 3: Automated Functional   -- Does it actually work when exercised?
Layer 4: Human Oversight        -- Does it meet qualitative standards?
```

Each layer is independent and can pass or fail separately. A feature is not done until all layers pass (or gaps are explicitly deferred with user approval).

---

## Layer 1: Plan Conformance

Checks must-haves from PLAN.md to determine whether the goal was actually achieved. Runs automatically without any tools beyond file system access.

### XML Format

```xml
<layer name="plan-conformance">
  <check>Each truth verified against codebase</check>
  <check>Each artifact exists, is substantive (above min_lines), and is wired</check>
  <check>Each key link connected (import exists, call exists)</check>
  <check>No TODO/FIXME/HACK/PLACEHOLDER in modified files</check>
</layer>
```

### What the Verifier Does

**Truth verification:** For each truth in `must_haves.truths`, the verifier traces the codebase to determine whether the truth holds. This is NOT a test run -- it is a code analysis check. The verifier reads the implementation and reasons about whether the stated behavior is achievable.

| Status | Meaning |
|--------|---------|
| VERIFIED | All supporting code paths exist and are complete |
| FAILED | Code path is missing, broken, or uses stubs |
| UNCERTAIN | Cannot determine programmatically -- escalate to human oversight |

**Artifact verification (three levels):**

| Level | Check | How | What It Catches |
|-------|-------|-----|-----------------|
| 1. Exists | `[ -f "$path" ]` | File exists at the declared path | Missing files, typos in paths |
| 2. Substantive | Line count >= min_lines, no stub patterns | Content is real implementation | Placeholder files, empty modules |
| 3. Wired | grep for imports/usage in other files | Connected to the rest of the system | Orphaned components that exist but are never called |

**Stub detection patterns:**
```
TODO, FIXME, HACK, PLACEHOLDER, XXX
return None, return {}, return [], pass (as sole function body)
raise NotImplementedError
# ... (ellipsis comments)
"not implemented", "coming soon", "add later"
```

**Key link verification:** For each key link, verify:
1. The `from` file imports or references the `to` file
2. The connection mechanism described in `via` actually exists in the code
3. The connection is not commented out or behind a dead code path

**Anti-pattern scan:** Search all files modified during this phase for:
- Stub indicators (above)
- Log-only error handlers (`except: log(...)` with no re-raise or handling)
- Empty except blocks (`except: pass`)
- Hardcoded test values where dynamic values are expected

---

## Layer 2: Code Review

Internal reviewers (always) plus external models (from config.yaml). This layer is about code quality, not functional correctness.

### XML Format

```xml
<layer name="code-review">
  <agents>code-reviewer, security-reviewer</agents>
  <focus>
    - Gateway pattern compliance (all external calls through gateways)
    - 3-phase DB pattern (no connections held during network I/O)
    - Error handling at service boundaries
  </focus>
  <external>codex, kimi</external>
  <skip_external_if>changes are less than 50 lines</skip_external_if>
</layer>
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `<agents>` | Yes | Comma-separated list of internal reviewer agents to invoke. `code-reviewer` is always present. Others are selected based on what the phase touches. |
| `<focus>` | Yes | Bullet list of specific areas reviewers should focus on. These come from the plan's understanding of what was built and what project patterns apply. |
| `<external>` | No | Comma-separated list of external review model names from config.yaml. |
| `<skip_external_if>` | No | Condition under which external reviews are skipped (to save cost on small changes). |

### Available Internal Reviewers

| Agent | When Selected | Focus |
|-------|---------------|-------|
| `code-reviewer` | Always | Architecture, patterns, conventions, cognitive load |
| `security-reviewer` | Plan includes auth, external input, API endpoints, or file handling | OWASP Top 10, secrets, injection, auth/authz |
| `data-pipeline-reviewer` | Plan touches Dagster assets or pipeline code | Asset dependencies, idempotency, failure semantics |
| `database-expert` | Plan includes schema changes or new queries | Schema design, migration safety, query performance |

### External Model Invocation

External models are invoked via the `command` from config.yaml. They receive:
- The diff of all changes in the phase
- The focus areas from the verification matrix
- Project KNOWLEDGE.md (Tier 1 rules)

They return structured findings. The review aggregator triages all findings (internal + external) together.

---

## Layer 3: Automated Functional (Autonomous Verification)

The verifier agent autonomously decides HOW to verify each truth. It reads the actual implementation, discovers available tools from `config.yaml`, and constructs its own verification strategy. This layer actually RUNS the code and checks that it works.

### Why Autonomous?

Pre-scripted verification steps (designed during planning, before code exists) break when the implementation deviates from the plan -- different file paths, different table names, different method signatures. The autonomous verifier reads the actual code and adapts.

### Plan Format

The plan provides optional `<verifier_guidance>` hints, not scripted steps:

```xml
<layer name="automated-functional">
  <verifier_guidance>
    <hint truth="T1">
      This truth involves database persistence. Check that records
      actually exist after the pipeline runs. Provenance metadata
      (provider name, model version) should be non-null.
    </hint>
    <hint truth="T2">
      This truth involves idempotency. Running the same operation
      twice should not create duplicate records.
    </hint>
  </verifier_guidance>
</layer>
```

**Hints describe CONCERNS, not COMMANDS.** They tell the verifier what makes a truth tricky to verify, not what specific tool or query to use. Hints are optional -- the verifier verifies all truths regardless of whether hints exist.

### The 3-Stage Verification Pipeline

The verifier follows: **Understand, Discover, Verify.**

#### Stage 1: Understand

The verifier reads the plan's truths, the actual diff, EXECUTION_STATE.md (for deviations), and the implementation files. For each truth, it builds a verification model:

- What code paths support this truth?
- What external systems are involved (database, APIs, file system, UI)?
- What are the likely failure modes?
- Did the implementation deviate from the plan?

#### Stage 2: Discover Tools

The verifier reads `config.yaml` to build a tool inventory at runtime:

1. Read `phases.verify.services` for service keys available to verification
2. For each service key, look up its definition in the `services` registry
3. If the service has a `test` field, run it as a health check
4. Build an active tool inventory: services that are both `available: true` AND pass their health check

Tool inventory is dynamic. When new tools are added to `config.yaml`, the verifier can use them immediately without any plan changes.

#### Stage 3: Verify

For each truth, the verifier:

1. **Generates a verification strategy** -- What tools to use, what to check, what constitutes evidence. Decided at verification time based on the actual implementation.
2. **Executes the strategy** -- Runs commands, queries databases, navigates UIs. Collects evidence at each step.
3. **Assesses confidence** -- Based on the evidence chain, assigns a confidence level.
4. **Records evidence** -- Every check, command, and result is recorded in VERIFICATION.md.

### Confidence Levels

Instead of binary PASSED/FAILED, the verifier reports graduated confidence per truth:

| Confidence | Criteria |
|------------|----------|
| **HIGH** | 3+ independent signals confirm the truth, including at least one runtime signal (test execution, database query, API call) |
| **MEDIUM** | 2+ signals converging, OR 1 runtime signal without cross-validation |
| **LOW** | Only static analysis signals (code looks correct but was not exercised) |
| **UNABLE** | No signals could be gathered (tools unavailable, code unreadable). Escalated to human. |

**Signal types:**
- **Static analysis** -- Code path exists, imports are wired, logic reads correctly
- **Unit tests** -- Tests pass for the relevant functionality
- **Runtime query** -- Database query, API call, or pipeline execution confirms actual state
- **UI verification** -- Browser tool confirms visual state

### Tool Capability Matching

The verifier matches each truth's needs to available tool capabilities:

| Truth involves... | Needs capability... | Typical tools |
|-------------------|--------------------|--------------|
| Database persistence | select, count, verify_schema | postgresql, database |
| Script/pipeline execution | run_command, check_exit_code | bash |
| UI state verification | navigate, screenshot, read_page | chrome_mcp, playwright |
| HTTP endpoint behavior | run_command (curl) | bash |
| Container operations | build, run, compose | docker |
| LLM call tracing | query traces, check spans | langfuse |
| Error rates | query errors, check rates | sentry |

When no direct tool is available, the verifier tries alternatives (e.g., bash as fallback for database queries via Python one-liners). When no alternative exists, it escalates to human oversight.

### Escalation Protocol

Before escalating a truth to human review, the verifier exhausts alternatives:

1. **Alternative tool** -- Can bash serve as fallback? (e.g., Python one-liner instead of database tool)
2. **Partial verification** -- Can we verify a subset of the truth? (e.g., code analysis + unit tests, but not end-to-end)
3. **Indirect verification** -- Can we verify a consequence? (e.g., retry decorator applied to the right methods, even if we cannot trigger a real rate limit)

If confidence remains UNABLE after alternatives, the verifier creates a **dynamic human checkpoint** with prepared evidence, documenting what was tried and why it failed.

### Dynamic Human Checkpoints

The verifier can create human checkpoints at verification time (not just the static ones from the plan). These are added to Layer 4 output as "Dynamic Checkpoints" and include:

- The truth being escalated
- What verification was attempted
- What evidence was gathered (partial support)
- What the human should check
- Specific criteria for the human's judgment

---

## Layer 4: Human Oversight

Things the human MUST review because they cannot be verified programmatically.

### XML Format

```xml
<layer name="human-oversight">
  <checkpoint>
    <what>Review OCR output for 3 sample PDFs (invoice, report, scan)</what>
    <why>Visual quality of OCR cannot be verified programmatically</why>
    <evidence>Side-by-side comparison: original PDF vs extracted text</evidence>
    <criteria>Text is readable, layout preserved, no garbled output</criteria>
  </checkpoint>

  <checkpoint>
    <what>Review error messages for clarity and helpfulness</what>
    <why>Message quality is subjective and context-dependent</why>
    <evidence>List of all new error messages with triggering conditions</evidence>
    <criteria>Each message tells the user what went wrong and what to do about it</criteria>
  </checkpoint>
</layer>
```

### Checkpoint Fields

| Field | Required | Description |
|-------|----------|-------------|
| `<what>` | Yes | What the human should review. Specific enough to be actionable. |
| `<why>` | Yes | Why this cannot be automated. Justifies the human's time. |
| `<evidence>` | Yes | What to present to the human. The verifier prepares this BEFORE presenting the checkpoint. |
| `<criteria>` | Yes | What "good" looks like. Specific enough for a pass/fail judgment. |

### When to Include Human Oversight

**Always include for:**
- Visual output quality (OCR, generated documents, UI appearance)
- User-facing message quality (error messages, notifications)
- Architectural decisions that emerged during implementation (deviations from plan)
- Security-sensitive changes (auth, access control, data exposure)

**Never include for:**
- Things that can be tested programmatically (use automated-functional instead)
- Style preferences already captured in KNOWLEDGE.md or PATTERNS.md
- Routine code review (that is Layer 2)

### Human Checkpoint Process

1. The verifier prepares all evidence BEFORE presenting the checkpoint
2. Evidence is presented in a clear format (screenshots, data tables, diffs)
3. The human provides a pass/fail judgment with optional notes
4. Results are recorded in VERIFICATION.md

---

## How the Planner Writes Verifier Guidance

During Phase 3 (Plan), the planner writes optional hints for the verifier. The planner does NOT select tools, write commands, or reference `config.yaml` for Layer 3. Tool availability is the verifier's concern.

### What the Planner Does

For each must-have truth, optionally write a `<hint>` if domain knowledge would help the verifier focus:

```
For each must-have truth:
  1. Consider: Does the verifier need domain context to verify this?
     - If the truth involves non-obvious concerns (idempotency, race conditions,
       provenance requirements), write a hint describing the concern
     - If the truth is straightforward (file exists, import works, tests pass),
       no hint is needed -- the verifier will figure it out

  2. Write the hint about the CONCERN, not the METHOD:
     - Good: "This truth involves idempotency -- running twice must not duplicate records"
     - Bad: "Run SELECT COUNT(*) FROM report_extractions before and after"
     - Good: "Provenance metadata (provider name, model version) should be non-null"
     - Bad: "Query the database using uv run --env-file .env python -c ..."
```

### Example: Planner Writing Guidance

Given the must-have truth: "OCR results persist in the database with full provenance metadata"

The planner writes:
```xml
<verifier_guidance>
  <hint truth="T1">
    This truth involves database persistence with provenance. Check that
    records actually exist after the operation. Provider name and model
    version should be populated (non-null) for every extraction record.
  </hint>
</verifier_guidance>
```

The verifier then reads the actual implementation, discovers that the table is `extraction_results` (not `report_extractions` as might have been assumed), finds the `postgresql` tool in `config.yaml`, and constructs the appropriate query at verification time.

---

## How the Verifier Executes Layer 3

During Phase 6 (Verification), the verifier executes Layer 3 autonomously using the 3-stage pipeline described above.

### Execution Flow

```
1. Parse PLAN.md verification matrix

2. Execute Layer 1 (Plan Conformance):
   - Pure code analysis, no external tools needed
   - Check truths, artifacts, key links, anti-patterns
   - Record results

3. Execute Layer 2 (Code Review):
   - Check REVIEWS.md gate (no fix-now items remaining)
   - Record results

4. Execute Layer 3 (Automated Functional — Autonomous):

   4a. Understand:
       - Read truths from PLAN.md <must_haves>
       - Read hints from PLAN.md <verifier_guidance> (if present)
       - Read EXECUTION_STATE.md for deviations
       - Read the actual diff and implementation files
       - Map each truth to its supporting code paths

   4b. Discover:
       - Read config.yaml phases.verify.services
       - For each service: look up in services registry, run health check
       - Build active tool inventory

   4c. Verify (for each truth):
       - Generate a verification strategy using available tools
       - Execute the strategy, collecting evidence
       - Classify each check: SUPPORTS, CONTRADICTS, INCONCLUSIVE
       - Assess confidence: HIGH, MEDIUM, LOW, UNABLE
       - If UNABLE: exhaust alternatives before escalating

   4d. Compile:
       - Per-truth confidence with evidence chains
       - Tool health report
       - Escalations for UNABLE truths

5. Execute Layer 4 (Human Oversight):
   For each <checkpoint> (static from plan + dynamic from verifier):
     a. Prepare evidence (run commands, take screenshots)
     b. Present checkpoint to human
     c. Wait for human judgment
     d. Record result

6. Compile VERIFICATION.md from all layer results
```

---

## VERIFICATION.md Output Format

The verifier compiles all results into VERIFICATION.md. Layer 3 uses a confidence-based model with per-truth evidence chains.

```yaml
---
status: passed | gaps_found | human_needed
score: N/M must-haves verified

plan_conformance:                  # Layer 1
  truths_verified: N/M
  artifacts_verified: N/M
  key_links_verified: N/M
  anti_patterns_found: []

automated_functional:              # Layer 3
  status: passed | degraded | failed
  composite_confidence: HIGH | MEDIUM | LOW
  tools_used: [bash, postgresql, docker]
  tool_health:
    bash: healthy
    postgresql: healthy
    chrome_mcp: degraded (extension not running)
  per_truth:
    - truth: "T1: OCR results persist with provenance"
      confidence: HIGH
      signals: 3
      evidence:
        - check: "Code path analysis"
          tool: "file system"
          result: SUPPORTS
          detail: "ExtractionResult model has provider_name, model_version fields"
        - check: "Unit tests"
          tool: "bash"
          result: SUPPORTS
          detail: "make test -- -k test_extraction_provenance: 3 tests pass"
        - check: "Database query"
          tool: "postgresql"
          result: SUPPORTS
          detail: "5 rows in extraction_results, all provenance fields non-null"
    - truth: "T2: Pipeline is idempotent"
      confidence: MEDIUM
      signals: 2
      note: "Could not run full pipeline in verification environment"
  escalations:
    - truth: "T3: Rate limits handled gracefully"
      confidence: UNABLE
      attempted: ["code analysis (PARTIAL)", "unit test (PARTIAL)", "live trigger (NOT POSSIBLE)"]
      escalated_to: "Layer 4 Dynamic Checkpoint #1"

qualitative:                       # (if applicable)
  evaluated: true | false
  accuracy: "92% on text fields"
  issues: []

human_oversight:                   # Layer 4
  static_checkpoints:              # from plan
    - what: "Review OCR output for 3 sample PDFs"
      status: pending | passed | failed
      notes: ""
  dynamic_checkpoints:             # from verifier escalations
    - source: "Verifier escalation (Truth T3)"
      what: "Verify rate limit handling under real conditions"
      evidence_prepared: "Retry decorator config, unit test results, code snippet"
      status: pending | passed | failed

gaps:
  - truth: "Re-running the pipeline does not create duplicates"
    layer: "automated_functional"
    confidence: LOW
    reason: "Could only verify via unit tests, not end-to-end pipeline run"
    fix: "Run pipeline twice in staging and compare record counts"
---
```

**Status rules for Layer 3:**
- **passed:** All truths HIGH or MEDIUM confidence, no contradictions
- **degraded:** Some truths LOW confidence, no contradictions
- **failed:** Any truth has contradicting evidence, OR all truths UNABLE

Note: Code review results (Layer 2) are captured in REVIEWS.md and OPEN_QUESTIONS.md, not in VERIFICATION.md. This is because review findings go through the aggregation process (Phase 5) before being acted on.
