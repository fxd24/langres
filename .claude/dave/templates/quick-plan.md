# Quick Plan Template

Template for `.state/milestones/adhoc/phases/{slug}/QUICK_PLAN.md` -- a lightweight inline plan for quick tasks that drives TDD implementation, review, and verification.

**Purpose:** Provide just enough structure for TDD execution, review, and verification to work without the full PLAN.md ceremony. This is a compact task specification generated directly by the orchestrator (no plan-checker agent, no user approval gate).

**Downstream consumers:**
- `tdd-developer` (TDD execution) -- Reads Files, Action, Tests, Verify, Done
- `practical-verifier` (post-TDD) -- Reads Verify and Done criteria
- `code-reviewer` (review) -- Reads the plan to understand intent
- `verification` (verify) -- Reads Must-Haves and Verify criteria

---

## File Template

```markdown
# Quick Task: {description}

**Created:** {date}
**Mode:** quick
**Slug:** {slug}

## Task

### Files
- `{src/path/to/file.py}` (create | modify)
- `{tests/path/to/test_file.py}` (create | modify)

### Action
{Specific implementation instructions. What to build, what patterns to follow,
what to import from where. Enough detail that a TDD executor can implement
without asking questions.}

### Tests
- {Test scenario 1 — specific, with expected input/output}
- {Test scenario 2}
- {Test scenario 3}

### Verify
{Specific commands to confirm completion. E.g.:}
`make test -- -k "test_relevant_tests"` passes
`make lint` passes

### Done
{Acceptance criteria in plain language. What must be true when the task is complete.}

## Must-Haves

1. {Truth derived from task description — user-observable, testable}
2. {Truth 2 — if applicable}

## Deviation Rules

| Rule | Trigger | Permission |
|------|---------|------------|
| Rule 1: Bug | Code does not work as intended | Auto-fix |
| Rule 2: Missing Critical | Missing error handling, validation, edge case | Auto-fix |
| Rule 3: Blocking | Missing dependency, broken import, env issue | Auto-fix |
| Rule 4: Architectural | New DB table, schema change, service restructure, new external dependency | STOP, ask user |

---

*Mode: quick*
*Plan created: {date}*
*Approved: auto (quick mode — user provided task description)*
```

<guidelines>

**What makes a good quick plan:**
- All five TDD elements present (Files, Action, Tests, Verify, Done)
- Action is specific enough that the TDD executor does not need to ask questions
- Tests are concrete scenarios (not vague "test that it works")
- Verify includes runnable commands
- Must-Haves are user-observable truths (not implementation details)

**What a quick plan does NOT need:**
- Artifacts/Key Links sections (too much ceremony for quick tasks)
- Verification matrix (Layer 1 + Layer 3 are run inline)
- Plan-checker agent validation (the orchestrator validates inline)
- User approval gate
- Estimated effort (quick = small by definition)

**Multi-task plans:**
The plan should actively look for opportunities to parallelize. Parallel execution is a first-class benefit: more code is produced simultaneously, and each agent gets its own context window — preventing context rot from accumulating implementation details. When the work can be split into independent pieces, structure it as waves with multiple tasks (each task gets its own Files/Action/Tests/Verify/Done section with a task ID like `## Task 1.1`, `## Task 1.2`). The orchestrator determines granularity based on the actual work.

**Slug generation:**
- Derive from task description
- Lowercase, hyphens for spaces
- Max 50 characters
- Example: "fix URL normalization for trailing slashes" -> "fix-url-normalization-trailing-slashes"

**Must-Haves quality:**
- Must be user-observable, not implementation details
  - Good: "URL normalization consistently handles trailing slashes"
  - Bad: "Added strip_trailing_slash function"
- Keep to 1-3 truths (quick tasks are small)

</guidelines>
