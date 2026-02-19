---
description: Practical verification of implemented features. Analyzes recent changes, plans real-world test scenarios (happy path, edge cases, integration, performance), executes them using Claude Code tools, and reports a structured verdict. Use after code generation, review, or whenever you want confidence that a feature actually works beyond just passing tests.
user-invocable: true
argument-hint: "<optional: specific feature/files to verify, or 'last commit' to verify the most recent change>"
allowed-tools: Bash, Read, Glob, Grep, Task
---

# Practical Feature Verification

Verify that implemented features actually work in practice by thinking like a user/QA engineer, executing real scenarios, and comparing expected vs actual behavior.

## Philosophy

**Tests prove code correctness. Verification proves the feature works.**

- Tests check units in isolation with mocks. Verification checks the real thing.
- Tests assert return values. Verification compares actual outputs against ground truth.
- Tests run fast with fake data. Verification runs with real data, real services, real files.
- A feature can have 100% test coverage and still not work in production.

**Use every tool at your disposal:**
- Read tool for PDFs, images, notebooks - understand what content SHOULD be extracted
- Bash for running scripts, querying databases, calling APIs
- Grep/Glob for finding test data, config files, related code
- Browser tools if verifying web UI behavior

## Step 1: Understand What Changed

Analyze recent changes to identify what needs verification.

```bash
# If no specific target provided, check recent changes
git diff --stat HEAD~1
git diff HEAD~1 --name-only
git log --oneline -5

# For unstaged/staged changes
git diff --stat
git diff --cached --stat

# Understand the full scope
git diff HEAD~1  # Read the actual code changes
```

**From the diff, determine:**
- What feature or fix was implemented?
- What are the inputs and outputs?
- What external systems are involved (DB, APIs, files, LLMs)?
- What configuration options exist?
- What error handling was added?

## Step 2: Plan Verification Scenarios

Think creatively about how this feature will be used in practice. Generate scenarios across these dimensions:

### Happy Path
The basic flow that should always work.
- What is the simplest end-to-end usage?
- Does the most common input produce correct output?
- Are the defaults sensible?

### Edge Cases
Unusual but valid inputs.
- Empty input, single item, very large input
- Unicode, special characters, unusual formats
- Boundary values (0, -1, max int, empty string)
- Missing optional fields, all optional fields present

### Error Handling
What happens when things go wrong?
- Invalid input - is the error message helpful?
- External service down - does it retry/fail gracefully?
- Partial failure - is the state consistent?
- Timeout - does it respect timeouts?

### Integration
Does it work with the rest of the system?
- Does data flow correctly from upstream?
- Are database records created/updated correctly?
- Do downstream consumers get what they expect?
- Are Langfuse traces / observability hooks working?

### Performance
Is it fast enough for production use?
- How long does a typical operation take?
- How does it scale with input size?
- Are there unnecessary network calls or DB queries?

### Configuration
Do options work as documented?
- Different config values produce expected behavior
- Defaults are sensible
- Invalid config fails with clear error

## Step 3: Execute Verification

For each scenario, follow this pattern:

### 3a. Establish Ground Truth

Before running the code, determine what the correct output should be using independent means:

**For data extraction (OCR, HTML parsing, PDF extraction):**
- Read the source document directly using Claude's Read tool (supports PDF, images)
- Note what content exists: text, headings, tables, images, logos
- This becomes the ground truth to compare against

**For data transformation:**
- Manually trace the expected output from the input
- Check against domain knowledge or reference implementations

**For API integrations:**
- Know what the API should return for given inputs
- Check API documentation for expected response format

### 3b. Run the Code

Execute the actual code path, not just tests:

```bash
# Run scripts
uv run --env-file .env python -c "
from src.module import SomeService
# ... exercise the actual code path
"

# Or run existing scripts
uv run --env-file .env python src/scripts/relevant_script.py --limit 1

# Or run tests that exercise real paths (integration tests)
uv run --env-file .env pytest tests/integration/relevant_test.py -v
```

**Safety guidelines:**
- Use `--limit`, `--dry-run`, or small inputs when testing against production resources
- Prefer test databases/collections when available
- Never run destructive operations without explicit user approval

### 3c. Compare Results

Compare the actual output against the ground truth established in 3a:

- **Text extraction:** Does the extracted text match what's in the document?
- **Data quality:** Are values accurate, complete, properly formatted?
- **Side effects:** Were database records created? Files written? Traces logged?
- **Error messages:** Are they helpful and actionable?

## Step 4: Verification Report

Present findings in this structured format:

```markdown
## Verification Report: [Feature Name]

### What Was Verified
- Feature: [brief description]
- Changes: [files changed, from git diff]
- Scope: [what aspects were tested]

### Environment
- Branch: [branch name]
- Commit: [short hash]
- Config: [relevant configuration used]

### Scenarios Tested

#### 1. [Scenario Name] - [PASS/FAIL/PARTIAL]
- **Input:** [what was provided]
- **Expected:** [what should happen]
- **Actual:** [what actually happened]
- **Evidence:** [command output, DB query result, file contents]
- **Notes:** [any observations]

#### 2. [Scenario Name] - [PASS/FAIL/PARTIAL]
...

### Ground Truth Comparison
[For extraction/transformation features]
- Source: [what the source document contains]
- Output: [what the code produced]
- Accuracy: [percentage or qualitative assessment]
- Missing: [what was missed]
- Extra: [what was hallucinated or incorrectly added]

### Issues Found

| # | Severity | Description | Impact | Suggested Fix |
|---|----------|-------------|--------|---------------|
| 1 | Critical | [blocks functionality] | [what breaks] | [how to fix] |
| 2 | Warning  | [works but concerning] | [risk] | [recommendation] |
| 3 | Info     | [minor observation] | [low] | [optional improvement] |

### Verdict

**Overall: [PASS / FAIL / NEEDS ATTENTION]**

Summary:
- X scenarios passed
- Y scenarios failed
- Z scenarios need attention

Confidence level: [HIGH / MEDIUM / LOW]
- [Reason for confidence level]
```

## Feature-Specific Verification Patterns

### OCR / Document Extraction
1. Read the PDF/image using Claude's Read tool to see what's actually in it
2. Run the OCR/extraction code on the same document
3. Compare extracted text against what Claude read directly
4. Check for: missed content, hallucinated content, formatting issues
5. Test with different document types (text-heavy, image-heavy, tables, mixed)
6. Verify visual elements are identified (logos, charts, signatures)
7. Test different config options (resolution, model, prompt templates)

### Database Migrations
1. Check current schema state
2. Run migration
3. Verify columns/tables exist with correct types
4. Test rollback (downgrade)
5. Verify data integrity if backfill was involved

### API Integrations
1. Make actual API call with known input
2. Verify response structure matches expected schema
3. Test error responses (invalid auth, bad input, rate limits)
4. Check retry behavior
5. Verify observability (traces, logs, metrics)

### Data Pipelines
1. Run pipeline with small sample
2. Verify each stage produced correct intermediate output
3. Check final output in database/storage
4. Test idempotency (run twice, verify no duplicates)
5. Test with previously-failed inputs

### LLM-Based Features
1. Run with known input where expected output is clear
2. Check prompt is well-formed (read the actual prompt sent)
3. Verify structured output parsing works
4. Test with adversarial inputs
5. Check Langfuse/observability traces

## Tips

1. **Start small** - Verify one happy path before testing edge cases. If the basic flow is broken, edge cases don't matter.
2. **Be specific** - "Found 3 records in the table with correct values" not "data exists."
3. **Show evidence** - Include command output, query results, file contents. Don't just assert things work.
4. **Think adversarially** - What would a determined user do to break this? What input would cause confusion?
5. **Check the invisible** - Logs, traces, metrics, database state. The visible output may look correct while invisible side effects are broken.
6. **Use real data when safe** - Sample files from `data/`, `tests/fixtures/`, or `tmp/` are more realistic than synthetic inputs.
7. **Time operations** - If something takes 30 seconds that should take 1 second, that's a finding even if the output is correct.
