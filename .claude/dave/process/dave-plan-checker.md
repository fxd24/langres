# dave-plan-checker: Process Guide

## Input Context

You receive from the planning workflow:

| Input | What it tells you |
|-------|------------------|
| PLAN.md | The plan to validate |
| DISCUSSION.md | The scope, decisions, and success criteria the plan must satisfy |
| RESEARCH.md | Technical findings and recommendations the plan should incorporate |
| KNOWLEDGE.md (project) | Tier 1 rules the plan must not violate |
| PATTERNS.md (project) | Conventions the plan should follow |
| config.yaml | Available verification tools |

All inputs are passed inline by the orchestrator. Work only with what you are given.

## Process

### Step 1: Extract Evaluation Criteria

Before checking anything, establish what you are checking against.

#### 1a: Extract Must-Haves from PLAN.md

List every truth, artifact, and key link from the plan's `<must_haves>` section. These are your primary evaluation targets.

#### 1b: Extract Requirements from DISCUSSION.md

List every success criterion, in-scope item, hard constraint, and decision from DISCUSSION.md. These are the requirements the plan must cover.

#### 1c: Extract Tier 1 Rules

List every Tier 1 rule from KNOWLEDGE.md that is relevant to this plan's domain. These are constraints the plan must not violate.

#### 1d: Extract Research Recommendations

List the key recommendations from RESEARCH.md. The plan should incorporate these (or explicitly justify deviations).

### Step 2: Requirement Coverage

For each requirement from DISCUSSION.md:
- Find the task(s) that address it
- Mark as: COVERED (clear mapping), PARTIAL (addressed but incomplete), or MISSING (no task covers this)

**Blocker if:** Any success criterion from DISCUSSION.md has no corresponding task.

### Step 3: Must-Haves Quality

#### 3a: Truth Quality

For each truth in the plan:
- Is it user-observable? (Good: "OCR results persist in the database" / Bad: "PaddleOCR library is installed")
- Is it testable? Can an automated verification step confirm it?
- Is it distinct from other truths? (No redundancy)

**Blocker if:** Any truth is an implementation detail rather than a user-observable behavior.

#### 3b: Artifact Quality

For each artifact:
- Is the path specific? (Not "somewhere in src/services")
- Is min_lines realistic? (Not 5 for a service, not 500 for a utility)
- Does the `provides` description match what the tasks actually build?

**Warning if:** min_lines seems unrealistic (too low or too high for the stated purpose).

#### 3c: Key Link Quality

For each key link:
- Does the `from` file exist or will be created by a task?
- Does the `to` file exist or will be created by a task?
- Is the `via` mechanism specific? (Not "some connection" but "import and instantiation in asset function")
- Is this link actually critical? (Breakage would cause cascading failure)

**Blocker if:** A key link references files that no task creates or modifies.

### Step 4: Task Completeness

For each task in the plan:

#### 4a: Required Fields

Check that every task has all five elements:
- **Files** — Exact paths with (create | modify) annotation
- **Action** — Specific implementation instructions
- **Tests** — Test categories with specific scenarios
- **Verify** — How to prove completion (specific commands or conditions)
- **Done** — Acceptance criteria in plain language

**Blocker if:** Any task is missing a required field.

#### 4b: Action Quality

For each task's Action:
- Is it specific enough that a developer does not need to ask questions?
- Does it reference patterns from PATTERNS.md where relevant?
- Does it include import paths, class names, or method signatures where helpful?
- Does it reference research findings where relevant?

**Warning if:** Action is vague ("implement the service") rather than specific ("create OCRService with `process_pages` method that...").

#### 4c: Verify Quality

For each task's Verify:
- Does it include specific commands to run? (Not "check that it works")
- Can it be executed by an agent without interpretation?
- Does it catch the actual goal, not just test passing?

**Warning if:** Verify is just "tests pass" without specifying which tests or what they prove.

#### 4d: Parallelism Quality

Check whether tasks are as granular as the work allows:

- Is Wave 1 maximizing its parallelism potential? Could any Wave 1 task be split into two independent tasks?
- For each multi-file task: do all files in the task genuinely need to be in the same task (file conflicts or TDD cohesion), or could they be separate parallel tasks?
- Are there tasks that combine unrelated responsibilities (e.g., "Create models AND repository" when models and repository have no file overlap)?

**Warning if:** Tasks look too chunky — suggest specific splits where work could be further decomposed into independent agents.

### Step 5: Test Specification Quality

For each task's Tests section:

#### 5a: Specificity

- Are test scenarios specific and falsifiable? (Not "test it works" but "test OCRService.process_pages returns structured results for valid JPEG images")
- Do edge cases reference relevant KNOWLEDGE.md entries where applicable?
- Do integration tests specify real dependencies (not "use mocks for everything")?

**Blocker if:** Tests are generic ("test the service works") rather than specific scenarios.

#### 5b: Coverage Appropriateness

- Do test categories match the task's nature? (Data processing tasks need edge cases, integration tasks need contract tests)
- Are regression tests included for known pitfalls from KNOWLEDGE.md?
- Is there at least one test that exercises the happy path end-to-end?

**Warning if:** A task that touches external systems has no integration test specified.

### Step 6: Dependency Correctness

#### 6a: Wave Consistency

- Wave 1 tasks have no dependencies (they are the foundation)
- Each wave N+1 task depends only on tasks in waves 1 through N
- No circular dependencies
- Independent tasks within a wave do not share the same files (parallel safety)

**Blocker if:** Circular dependencies exist or Wave 1 tasks have dependencies.

#### 6b: File Conflict Check

- No two tasks in the same wave modify the same file (would cause merge conflicts in parallel execution)
- If two tasks in different waves modify the same file, the later one depends on the earlier one

**Warning if:** Two parallel tasks touch the same file (even if different sections — risk of conflict).

### Step 7: Verification Matrix Completeness

#### 7a: Layer Presence

Check that all four layers are present:
1. `plan-conformance` — Goal-backward checks
2. `code-review` — Agents and focus areas
3. `automated-functional` — Tool-based verification steps
4. `human-oversight` — Checkpoints (can be empty if not needed)

**Blocker if:** `plan-conformance` or `automated-functional` layers are missing.
**Warning if:** `code-review` layer is missing.

#### 7b: Truth Verifiability

For each must-have truth, verify that it is a testable assertion that the autonomous verifier can investigate. The plan does NOT need scripted steps per truth -- the verifier decides how to verify at runtime.

Check each truth for verifiability:
- Is it a concrete, testable assertion? (Not vague like "system works well")
- Could an agent with access to code, tests, database, and bash verify this?
- If a `<hint>` exists for this truth, does the hint describe a CONCERN (good) rather than a COMMAND (bad)?

```
Truth: "X" → Verifiable: yes (testable assertion about database state)
Truth: "Y" → Verifiable: yes (testable assertion about behavior) + hint provides domain context
Truth: "Z" → Verifiable: no (subjective — "user experience is good") → should be human-oversight
```

**Blocker if:** Any truth is not a testable assertion (vague, subjective, or requires human judgment without a human-oversight checkpoint).

**Warning if:** A `<hint>` prescribes specific commands or tools instead of describing concerns. Hints should say "involves idempotency" not "run SELECT COUNT(*)...".

#### 7c: Verifier Guidance Quality

If `<verifier_guidance>` hints are present:
- Hints reference truths by ID (e.g., `truth="T1"`)
- Hints describe concerns, not commands (no specific tools, queries, or scripts)
- Hints do not reference `config.yaml` (tool availability is the verifier's concern)

**Note:** Hints are optional. The absence of hints is NOT a problem -- the verifier verifies all truths regardless. Hints are only needed when domain context would help the verifier focus.

#### 7d: Code Review Focus

- Focus areas are specific to this plan (not generic "check for bugs")
- Relevant Tier 1 rules are referenced in focus areas
- Reviewer selection matches the plan's domain (pipeline code → data-pipeline-reviewer)

**Warning if:** Focus areas are generic or reviewer selection seems mismatched.

### Step 8: Knowledge Compliance

For each relevant Tier 1 rule:
- Does any task's Action violate it?
- Does the plan's architecture contradict it?
- Are there patterns in the plan that ignore known pitfalls?

**Blocker if:** Any task explicitly contradicts a Tier 1 rule.
**Warning if:** A known pitfall from KNOWLEDGE.md is not addressed by any task or test.

### Step 9: Scope and Parallelism Quality

#### 9a: Per-Task Focus

For each task:
- Is it a focused vertical slice (one service, one repository, one processor + tests)?
- Does it touch many unrelated files (5+ files across different layers without justification)?
- Could this task be further split into independent agents without creating artificial dependencies?

**Warning if:** Any task touches 5+ unrelated files without justification (e.g., an atomic migration is justified; "create the domain layer" touching models, enums, utils, and repository is not).

#### 9b: Parallelism Check

For each task:
- Could two developers work on this task's sub-parts simultaneously without file conflicts? If yes → the task should be split.
- Does this task combine a service and a repository that have no file overlap? If yes → split into separate tasks.

**Warning if:** A task contains work that could be further decomposed into independent parallel agents.

#### 9c: Wave Utilization

For tasks across waves:
- Are independent tasks unnecessarily sequenced in different waves?
- Do tasks in Wave 2+ actually depend on earlier wave outputs, or are they independent?
- Could any Wave 2+ task be moved to Wave 1 (no true dependencies)?

**Warning if:** Tasks with no file overlap and no data dependencies are in different waves without a dependency reason.

#### 9d: Context Budget

Check overall plan size:
- Estimate the total execution context (all tasks combined)
- If estimated at 80%+ of a single agent's context budget, recommend splitting into phases

**Warning if:** Total plan size approaches 70% of estimated context budget.
**Blocker if:** Total plan size exceeds 80% of estimated context budget. Recommend splitting into phases at a natural wave boundary.

## Final Step: Verify Output Structure

Before returning your report, verify it matches `.claude/dave/templates/output/plan-checker-output.md`:
1. Validation summary table is present with PASS/FAIL per check
2. Every blocker has a specific fix suggestion
3. Clear PASS/REVISE verdict at the top

## Success Criteria

Plan validation is complete when:

- [ ] All requirements from DISCUSSION.md checked for coverage
- [ ] All must-have truths checked for quality and verifiability
- [ ] All artifacts checked for path specificity and realistic thresholds
- [ ] All key links checked for file references and mechanism specificity
- [ ] All tasks checked for required fields (Files, Action, Tests, Verify, Done)
- [ ] Test specifications checked for specificity and appropriateness
- [ ] Dependencies checked for cycles and file conflicts
- [ ] All four verification matrix layers checked
- [ ] Every truth is a testable assertion (verifiable by the autonomous verifier or has a human-oversight checkpoint)
- [ ] All Tier 1 rules checked for compliance
- [ ] Scope constraints checked
- [ ] Report includes specific fix suggestions for every blocker and warning
