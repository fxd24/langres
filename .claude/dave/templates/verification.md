# Verification Template

Template for `.state/milestones/{slug}/phases/{N}/VERIFICATION.md` -- multi-layer verification results with evidence and gap analysis.

**Context summary pattern:** Add `<!-- SUMMARY: {PASSED/FAILED}. {N}/{M} must-haves verified. Layer 1: {status}, Layer 2: {status}, Layer 3: {status}, Layer 4: {status}. {gaps count} gaps remaining. -->` after the file header.

**Purpose:** Record the results of executing all four verification layers from the plan's verification matrix. Each layer catches different classes of problems. This file provides the definitive answer to "did we build what we said we would, and does it actually work?"

**Downstream consumers:**
- `executor` (Phase 4 gap closure) -- Reads gaps to create focused fix plans
- `reflect` (Phase 8) -- Analyzes what verification caught for knowledge extraction
- `human` -- Reviews human-oversight checkpoint results

---

## File Template

```markdown
# Phase {N}: {Name} - Verification

**Verified:** {date}
**Status:** {passed | gaps_found | human_needed}
**Score:** {N}/{M} must-haves verified

---

## Layer 1: Plan Conformance

**Status:** {passed | failed}

### Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| T1 | "{truth statement}" | VERIFIED / FAILED / UNCERTAIN | {brief evidence — file path, code reference} |
| T2 | "{truth statement}" | {status} | {evidence} |
| T3 | "{truth statement}" | {status} | {evidence} |

### Artifacts

| # | Path | Exists | Substantive | Wired | Notes |
|---|------|--------|-------------|-------|-------|
| A1 | `{path}` | yes/no | yes/no ({N} lines, min {M}) | yes/no | {import/usage location} |
| A2 | `{path}` | {status} | {status} | {status} | {notes} |

### Key Links

| # | From | To | Via | Connected | Evidence |
|---|------|----|-----|-----------|----------|
| L1 | `{from}` | `{to}` | {mechanism} | yes/no | {import location, call site} |
| L2 | `{from}` | `{to}` | {mechanism} | {status} | {evidence} |

### Anti-Pattern Scan

| Pattern | Files Scanned | Occurrences | Details |
|---------|---------------|-------------|---------|
| TODO/FIXME/HACK | {N} | {N} | {file:line if any} |
| Empty implementations | {N} | {N} | {details} |
| Log-only error handlers | {N} | {N} | {details} |
| Stub patterns | {N} | {N} | {details} |

---

## Layer 2: Code Review

**Status:** see REVIEWS.md
**Fix now items remaining:** {0 — must be 0 to pass}
**Open questions resolved:** {yes/no — see OPEN_QUESTIONS.md}

> Layer 2 results are captured in REVIEWS.md and OPEN_QUESTIONS.md,
> not duplicated here. This section confirms the review gate passed.

---

## Layer 3: Automated Functional

**Status:** {passed | degraded | failed}
**Composite confidence:** {HIGH | MEDIUM | LOW}
**Tools used:** {list of tools actually used}
**Tool health:** {all healthy | N degraded — see Tool Health Report}

### Per-Truth Verification

#### Truth T1: "{truth statement}"

**Confidence:** {HIGH | MEDIUM | LOW | UNABLE}
**Signals:** {N} ({signal types — e.g., code analysis, unit tests, database query})

| # | Check | Tool | Result | Evidence |
|---|-------|------|--------|----------|
| 1 | {what was checked} | {tool used} | SUPPORTS / CONTRADICTS / INCONCLUSIVE | {brief evidence — command output, query results, code reference} |
| 2 | {check} | {tool} | {result} | {evidence} |
| 3 | {check} | {tool} | {result} | {evidence} |

**Deviation from plan:** {note if implementation differed from plan, and how the verifier adapted — or "None"}

#### Truth T2: "{truth statement}"

**Confidence:** {confidence}
**Signals:** {N} ({signal types})

| # | Check | Tool | Result | Evidence |
|---|-------|------|--------|----------|
| 1 | {check} | {tool} | {result} | {evidence} |

**Note:** {explanation if confidence is MEDIUM or LOW — what limited the verification}

#### Truth TN: "{truth statement}"

**Confidence:** UNABLE
**Escalated to:** Layer 4, Dynamic Checkpoint #{N}

**Attempted:**
- {Check 1}: {result} [{PARTIAL or NOT POSSIBLE}]
- {Check 2}: {result} [{PARTIAL or NOT POSSIBLE}]

**Escalation reason:** {why this truth could not be verified automatically}

### Tool Health Report

| Tool | Status | Health Check | Notes |
|------|--------|-------------|-------|
| {tool key} | healthy / degraded / unavailable | {health check command or "--"} | {notes} |

### Escalations

<!-- Truths that could not be verified automatically. Each becomes a dynamic
     human checkpoint in Layer 4. -->

| Truth | Confidence | Reason | Escalated To |
|-------|------------|--------|-------------|
| {TN} | {UNABLE or LOW} | {why verification failed} | {Layer 4 Dynamic Checkpoint #N} |

---

## Layer 4: Human Oversight

**Status:** {passed | pending | failed}
**Static checkpoints:** {N}/{M} completed (from plan)
**Dynamic checkpoints:** {N}/{M} completed (from verifier escalations)

### Static Checkpoints (from plan)

#### Checkpoint 1: {what}
- **Source:** Plan
- **Why human review needed:** {why}
- **Evidence prepared:** {evidence description}
- **Criteria:** {what good looks like}
- **Status:** PASSED / FAILED / PENDING
- **Human notes:** {notes from human review}

#### Checkpoint 2: {what}
...

### Dynamic Checkpoints (from verifier)

<!-- Created during Layer 3 when the verifier could not automatically verify
     a truth. These provide targeted, evidence-rich review tasks. -->

#### Dynamic Checkpoint 1: {truth statement}
- **Source:** Verifier escalation (Truth {TN})
- **Why human review needed:** {what the verifier could not automate}
- **What to review:** {specific action for the human}
- **Evidence prepared:**
  - {evidence item 1 — e.g., retry decorator configuration}
  - {evidence item 2 — e.g., unit test results}
  - {evidence item 3 — e.g., code snippet}
- **Criteria:** {what good looks like}
- **Verifier's partial confidence:** {LOW | UNABLE}
- **Status:** PASSED / FAILED / PENDING
- **Human notes:** {notes from human review}

---

## Qualitative Evaluation

<!-- Only present if the plan specified qualitative evaluation is needed -->

**Evaluated:** {true | false | not-applicable}
**Accuracy:** {metric if applicable}
**Issues:** {list of quality issues found}
**Details:** {detailed evaluation notes}

---

## Gaps

<!-- Items that failed verification. Each gap needs a fix plan or explicit
     deferral with user approval. -->

| # | Truth/Step | Layer | Status | Reason | Fix |
|---|-----------|-------|--------|--------|-----|
| G1 | "{failed truth or step}" | {layer name} | FAILED | {what went wrong} | {proposed fix or "deferred: {reason}"} |
| G2 | "{item}" | {layer} | {status} | {reason} | {fix} |

### Gap Closure History

<!-- Track gap closure iterations. Each should resolve specific gaps. -->

#### Iteration 1
- **Date:** {date}
- **Gaps addressed:** {list}
- **Fix approach:** {brief description}
- **Result:** {all resolved | N remaining}

---

## Summary

**Overall verdict:** {PASSED — all layers pass | GAPS FOUND — see gaps table | HUMAN NEEDED — awaiting checkpoints}

**Verification confidence:** {HIGH | MEDIUM | LOW}
- HIGH: All 4 layers pass, all truths HIGH or MEDIUM confidence, no escalations
- MEDIUM: All layers pass, some truths LOW confidence (degraded), no contradictions
- LOW: Significant gaps, UNABLE truths, or contradicting evidence

---

*Phase: {N}*
*Verification completed: {date}*
*Gap closure iterations: {N}*
*Human checkpoints: {N}/{M} completed*
```

<guidelines>

**Layer execution order:**
1. Plan conformance (always first — no tools needed)
2. Code review results (check REVIEWS.md gate — no fix-now items remaining)
3. Automated functional (run steps using available tools)
4. Human oversight (last — present evidence after all automated checks)

**Evidence quality:**
- Every truth has an evidence chain (what checks were run, what tool produced what result)
- Every SUPPORTS result has specific evidence (command output, query result, code reference)
- Every CONTRADICTS result has investigation notes (was it a false negative?)
- Every UNABLE truth documents what was tried and why it failed
- Screenshots saved to phase directory when browser tools used

**Gap closure rules:**
- Each gap gets a focused fix plan (not a full re-plan)
- Fixes loop back to Phase 4 (scoped TDD)
- Re-verify only the gaps, not the full matrix
- User can explicitly defer gaps with rationale

**VERIFICATION.md completeness:**
- All must-haves from PLAN.md accounted for (VERIFIED, FAILED, or UNCERTAIN in Layer 1)
- All truths have confidence levels and evidence chains in Layer 3
- Tool health report is complete (all tools from inventory reported)
- All human checkpoints presented or marked pending (both static and dynamic)
- Escalations table is complete for any UNABLE truths
- Gaps table is complete (no silent failures)

</guidelines>
