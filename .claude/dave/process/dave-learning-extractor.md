# dave-learning-extractor: Process Guide

## Input Context

You receive from the reflect workflow orchestrator:

| Input | What it tells you |
|-------|------------------|
| PLAN.md | What was planned — must-haves, tasks, verification matrix |
| EXECUTION_STATE.md | What actually happened — task status, deviations, commit history |
| REVIEWS.md | What reviewers found — fix-now items, deferred items, dismissed items |
| OPEN_QUESTIONS.md | Ambiguous items and how they were resolved |
| VERIFICATION.md | What verification found — pass/fail per layer, gaps, evidence |
| Phase KNOWLEDGE.md (if exists) | Current phase-level knowledge entries |
| Project KNOWLEDGE.md | Existing project-level Tier 1 and Tier 2 entries |
| Project PATTERNS.md | Existing project patterns and conventions |
| Project CONCERNS.md | Existing project concerns and tech debt |
| Phase directory path | Where to find all phase artifacts |

## Process

### Step 1: Plan vs Actual Analysis

Read PLAN.md and EXECUTION_STATE.md. For each task:

a. **Did it complete as planned?** Check status, deviations, commit messages.
b. **Were there deviations?** What changed and why?
c. **Did the plan's must-haves hold?** Were truths verified, artifacts created, links connected?
d. **Was the task harder or easier than planned?** Look for multiple fix attempts, deviation rules invoked.

**Output:** List of plan-vs-actual gaps with analysis of WHY each gap occurred.

### Step 2: Review Pattern Analysis

Read REVIEWS.md. Analyze across all findings:

a. **Category distribution:** Which categories had the most findings? (bug, security, performance, etc.)
b. **Recurrence:** Did multiple reviewers flag the same CATEGORY of issue (even if different specific findings)?
c. **False positive patterns:** Which dismissed findings share a common theme? (This may indicate reviewer calibration issues OR project conventions that need documentation.)
d. **Fix-now patterns:** What do the fix-now items have in common? (e.g., "all fix-now items were about missing error handling" → this is a pattern)
e. **Defer patterns:** What themes appear in deferred items?

**Output:** List of review patterns with frequency and significance assessment.

### Step 3: Verification Pattern Analysis

Read VERIFICATION.md. Analyze results:

a. **Which layers failed?** Is there a pattern (e.g., Layer 3 always catches the same type of issue)?
b. **Gap closure:** How many gaps were found? Were they plan gaps or execution gaps?
c. **Confidence:** Was the overall confidence HIGH, MEDIUM, or LOW? Why?
d. **Skipped steps:** Were any verification steps skipped? Why? Should tools be added?

**Output:** List of verification patterns with improvement suggestions.

### Step 4: Extract New Knowledge Entries

From the analysis in Steps 1-3, extract Tier 2 entries. For each potential entry:

a. **Specificity check:** Is it specific enough that an agent can act on it? ("Be more careful" = NO. "Always validate batch size against GPU memory before committing to a provider" = YES.)
b. **Durability check:** Will this be relevant in 3 months? (One-off workaround for a specific API bug = NO. Pattern about how to handle API response format changes = YES.)
c. **Generalizability check:** Would this help in a DIFFERENT phase? (If it only applies to this exact scenario, it is a phase-specific detail, not a knowledge entry.)
d. **Novelty check:** Does this already exist in project KNOWLEDGE.md? (If so, increment the Verified count instead of creating a duplicate.)

**Format each entry as:**
```markdown
- [A###] {Rule text — specific, actionable, unambiguous}
  Source: Agent (reflect)
  Added: {YYYY-MM-DD}
  Confidence: {HIGH | MEDIUM | LOW}
  Verified: 1 times
  Promoted: No
  Evidence: {Which phase artifact supports this — e.g., "REVIEWS.md F003, VERIFICATION.md Layer 3 Step 2"}
```

### Step 5: Update Existing Knowledge

Check project KNOWLEDGE.md Tier 2 entries:

a. **Verification updates:** Did this phase independently confirm any existing Tier 2 entry? If so, increment its Verified count.
b. **Contradiction check:** Did this phase contradict any existing Tier 2 entry? If so, note the contradiction for human review.
c. **Promotion candidates:** After updating Verified counts, check if any entries exceed `tier2_promotion_threshold` from config.yaml. Flag these as promotion candidates.

### Step 6: Identify Pattern and Concern Updates

From the analysis, identify:

a. **New patterns for PATTERNS.md:** Conventions that emerged during this phase and should be followed going forward. (e.g., "When adding a new gateway, always implement rate limiting with RPM+TPM, not just semaphore")
b. **New concerns for CONCERNS.md:** Technical debt or risks identified but not addressed. (e.g., "Rate limiters are per-process only — multi-process environments need coordination")
c. **Resolved concerns:** Did this phase address any existing CONCERNS.md items?

### Step 7: Compile Output

Write a structured report with all findings. This report is consumed by the reflect workflow orchestrator, who will write the actual files.

## Final Step: Verify Output Structure

Before returning your output, verify it matches `.claude/dave/templates/output/learning-extractor-output.md`:
1. Every required section is present (plan-vs-actual, review patterns, new entries, promotions)
2. Every new Tier 2 entry has evidence citation and confidence level
3. Quality over quantity: 3 excellent entries beat 10 mediocre ones

## Quality Checks

Before finalizing output:

1. **Every new entry passes the 4 checks:** Specificity, durability, generalizability, novelty.
2. **No duplicate entries:** Cross-referenced against project KNOWLEDGE.md.
3. **Evidence is traceable:** Each entry cites a specific artifact section.
4. **Promotion candidates have sufficient verified count:** Check against config.yaml threshold.
5. **Pattern updates are conventions, not one-offs:** Would you teach this to a new team member?
6. **Concerns are actionable:** Each concern has enough context to be addressed later.
