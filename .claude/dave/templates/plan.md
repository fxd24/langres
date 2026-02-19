# Plan Template

Template for `.state/milestones/{slug}/phases/{N}/PLAN.md` -- the execution contract that drives TDD implementation, review, and verification.

**Purpose:** Combine goal-backward must-haves, prescriptive tasks organized in dependency waves, and a verification matrix that defines WHAT must be verified and provides guidance for the autonomous verifier. This is the single source of truth for what gets built and how we prove it works.

**Downstream consumers:**
- `tdd-developer` (Phase 4) -- Reads its specific task (files, action, verify, done)
- `practical-verifier` (Phase 4) -- Reads task verify criteria
- `code-reviewer` (Phase 5) -- Reads the code-review layer focus areas
- `review-aggregator` (Phase 5) -- Reads must-haves to understand intent
- `verifier` (Phase 6) -- Reads truths (contract) and hints (guidance), autonomously verifies each truth
- `reflect` (Phase 8) -- Compares plan vs actual to extract learnings

---

## File Template

```markdown
# Phase {N}: {Name} - Plan

<!-- SUMMARY: {1-2 sentences: number of must-haves, number of tasks in N waves, estimated effort, key verification approach} -->

**Planned:** {date}
**Based on:** DISCUSSION.md ({date}), RESEARCH.md ({date})
**Estimated effort:** {S | M | L}

<must_haves>
## Must-Haves (Goal-Backward)

Before defining tasks, define what must be TRUE when the work is complete.
These prevent the failure mode where all tasks are completed but the goal
is not achieved.

### Truths
<!-- Observable behaviors that must be true from the user's perspective.
     Each truth is verified in Phase 6. Write them as testable assertions.
     Truths are numbered T1, T2, T3... These IDs are referenced by verifier
     guidance hints and the verification report. -->

- **T1:** "{Truth 1 -- user-observable, testable behavior}"
- **T2:** "{Truth 2}"
- **T3:** "{Truth 3}"

### Artifacts
<!-- Files that must exist and be substantive (not stubs or placeholders).
     Each artifact has a path, what it provides, and a minimum complexity
     threshold. -->

- path: "{src/path/to/file.py}"
  provides: "{What this file does in one sentence}"
  min_lines: {N}

- path: "{src/path/to/another_file.py}"
  provides: "{purpose}"
  min_lines: {N}

### Key Links
<!-- Critical wiring between components. Breakage here causes cascading
     failure. The verifier checks that these connections exist. -->

- from: "{src/path/to/caller.py}"
  to: "{src/path/to/callee.py}"
  via: "{How they connect -- e.g., 'import and instantiation in __init__', 'service injection in constructor'}"

- from: "{path}"
  to: "{path}"
  via: "{mechanism}"

</must_haves>

<tasks>
## Task Breakdown

### Wave 1: {wave description}
<!-- Independent tasks that can run in parallel (separate tdd-developer agents). -->

#### Task 1.1: {task name}
- **Files:** {exact paths created or modified}
  - `{src/path/to/file.py}` (create | modify)
  - `{tests/path/to/test_file.py}` (create | modify)
- **Action:** {Specific implementation instructions. What to build, what
  patterns to follow, what to import from where. Enough detail that the
  developer does not need to ask questions.}
- **Verify:** {How to prove the task is complete. Specific commands, checks,
  or conditions. E.g., "`make test` passes for OCR tests, DB records created."}
- **Done:** {Acceptance criteria in plain language. E.g., "Pages are OCR'd,
  results stored with provider metadata, errors logged not raised."}

#### Task 1.2: {task name}
- **Files:** {paths}
- **Action:** {instructions}
- **Verify:** {how to check}
- **Done:** {acceptance criteria}

### Wave 2: {wave description}
<!-- Tasks that depend on Wave 1 completion. -->

#### Task 2.1: {task name}
- **Depends on:** {Task 1.1, Task 1.2, or specific artifacts from Wave 1}
- **Files:** {paths}
- **Action:** {instructions}
- **Verify:** {how to check}
- **Done:** {acceptance criteria}

### Wave 3: {wave description}

#### Task 3.1: {task name}
- **Depends on:** {dependencies}
- **Files:** {paths}
- **Action:** {instructions}
- **Verify:** {how to check}
- **Done:** {acceptance criteria}

</tasks>

<verification_matrix>
## Verification Matrix

<!-- Four layers of verification. Each layer catches different classes of
     problems. Layer 3 is autonomous — the verifier discovers tools from
     config.yaml at verification time and decides how to verify. -->

<layer name="plan-conformance">
  <!-- Goal-backward verifier checks must-haves. Runs automatically. -->
  <check>Each truth verified against codebase</check>
  <check>Each artifact exists, is substantive (above min_lines), and is wired</check>
  <check>Each key link connected (import exists, call exists)</check>
  <check>No TODO/FIXME/HACK/PLACEHOLDER in modified files</check>
</layer>

<layer name="code-review">
  <!-- Internal reviewers (always) + external models (from config.yaml). -->
  <agents>{code-reviewer, security-reviewer, data-pipeline-reviewer -- select based on what the phase touches}</agents>
  <focus>
    - {Focus area 1 -- e.g., "Gateway pattern compliance"}
    - {Focus area 2 -- e.g., "3-phase DB pattern"}
    - {Focus area 3 -- e.g., "Error handling at service boundaries"}
  </focus>
  <external>{codex, kimi -- from config.yaml review_models}</external>
  <skip_external_if>{Condition -- e.g., "changes are less than 50 lines"}</skip_external_if>
</layer>

<layer name="automated-functional">
  <!-- The verifier agent autonomously decides HOW to verify each truth.
       It reads the implementation, discovers available tools from config.yaml,
       and constructs its own verification strategy.

       Hints are OPTIONAL guidance from the planner. They describe the
       CONCERN (what makes this truth tricky to verify), not the METHOD
       (what command to run). The verifier may use them, ignore them, or
       go beyond them. Truths without hints are still verified. -->

  <verifier_guidance>
    <hint truth="{T1}">
      {Describe the CONCERN, not the command. E.g., "This truth involves
      database persistence — check that records actually exist after the
      operation. Provenance metadata (provider name, model version) should
      be non-null." Do NOT reference specific tools, commands, or queries.}
    </hint>

    <hint truth="{T2}">
      {Another concern. E.g., "This truth involves idempotency — running
      the same operation twice should not create duplicate records."}
    </hint>

    <!-- Hints are optional. Omit for truths where the verifier needs
         no domain guidance. The verifier verifies ALL truths regardless
         of whether hints exist. -->
  </verifier_guidance>
</layer>

<layer name="human-oversight">
  <!-- Things the human MUST review. Only include what cannot be
       automated. Each checkpoint specifies what, why, evidence, and
       criteria. -->
  <checkpoint>
    <what>{What the human should review}</what>
    <why>{Why this cannot be automated}</why>
    <evidence>{What to show the human -- screenshots, data, comparisons}</evidence>
    <criteria>{What "good" looks like}</criteria>
  </checkpoint>
</layer>

</verification_matrix>

<deviation_rules>
## Deviation Rules

<!-- Clear rules for what the executor can handle autonomously vs what
     requires stopping and asking the user. -->

| Rule | Trigger | Permission |
|------|---------|------------|
| Rule 1: Bug | Code does not work as intended | Auto-fix |
| Rule 2: Missing Critical | Missing error handling, validation, edge case | Auto-fix |
| Rule 3: Blocking | Missing dependency, broken import, env issue | Auto-fix |
| Rule 4: Architectural | New DB table, schema change, service restructure, new external dependency | STOP, ask user |

</deviation_rules>

<scope_constraints>
## Scope Constraints

<!-- No hard cap on task count or file count. Each task should be a focused
     vertical slice. Split until further splitting would create artificial
     dependencies. More tasks = more parallel agents = better quality.

     The only scope limit is the overall context budget — if the plan is
     very large, split into separate phases at a natural wave boundary. -->

| Constraint | Warning | Split into phases |
|------------|---------|-------------------|
| Per-task scope | Task touches 5+ unrelated files | Split the task further |
| Context budget | ~70% estimated execution context | 80%+ → split into phases |

</scope_constraints>

---

*Phase: {N}*
*Plan created: {date}*
*Plan checker: {passed | N blockers found}*
*Approved by user: {yes | pending}*
```

<guidelines>

**What the planner reads before creating the plan:**

| File | What It Provides |
|------|-----------------|
| `project/KNOWLEDGE.md` | Pitfalls to avoid (Tier 1 > Tier 2) |
| `project/PATTERNS.md` | Conventions to follow |
| `project/CONCERNS.md` | Known issues to watch for |
| `project/config.yaml` | Available tools, models, verification capabilities |
| `codebase/ARCHITECTURE.md` | Where to put code, how layers connect |
| Phase `DISCUSSION.md` | Scope, guardrails, success criteria |
| Phase `RESEARCH.md` | Technical findings, recommendations, pitfalls |

**Must-haves quality:**
- Truths must be user-observable, not implementation details
  - Good: "OCR results persist in the database with full provenance"
  - Bad: "PaddleOCR library is installed"
- Artifacts must have realistic min_lines thresholds
- Key links must identify the actual connection mechanism

**Task quality:**
- Every task has all four elements: Files, Action, Verify, Done
- Action is specific enough that the developer does not need to ask questions
- Verify includes specific commands or conditions, not just "check that it works"
- Files lists exact paths (create vs modify)

**Wave organization:**
- Maximize tasks per wave — more parallel agents = better
- Each task = one vertical slice (test + implementation for one focused responsibility)
- No artificial cap on task count — split until further splitting creates artificial dependencies
- Wave 1 contains all independent tasks (no dependencies)
- Each subsequent wave depends only on previous waves
- No cycles allowed
- Independent tasks within a wave run as parallel tdd-developer agents

**Verification matrix quality:**
- All four layers present
- Every must-have truth is a testable assertion (verifiable by the autonomous verifier or has a human-oversight checkpoint)
- Automated-functional layer uses `<verifier_guidance>` with `<hint>` elements (not scripted steps)
- Hints describe concerns, not commands (no specific tools, queries, or scripts)
- Human oversight only includes what genuinely cannot be automated
- Code review focus areas are specific to this plan, not generic

**Plan checker validates:**
1. Requirement coverage -- every DISCUSSION.md requirement has tasks
2. Task completeness -- every task has Files, Action, Verify, Done
3. Dependency correctness -- no cycles, waves are consistent
4. Key links planned -- artifacts are wired, not isolated
5. Must-haves are user-observable -- not implementation details
6. Verification matrix complete -- all four layers, all truths verifiable
7. Knowledge compliance -- plan does not violate Tier 1 entries
8. Scope sanity -- within context budget

**After creation:**
- File lives at `.state/milestones/{slug}/phases/{N}/PLAN.md`
- Plan checker runs and reports blockers
- User approves before execution begins
- TDD developers receive their specific tasks
- Verifier executes the verification matrix after implementation

</guidelines>
