# dave-review-aggregator: Process Guide

## Input Context

You receive from the review workflow orchestrator:

| Input | What it tells you |
|-------|------------------|
| All reviewer findings | Raw findings from each reviewer (internal + external), passed inline |
| PLAN.md | What was built and why — the must-haves, task descriptions, verification matrix |
| project/KNOWLEDGE.md | Tier 1 rules (absolute authority) and Tier 2 patterns |
| project/PATTERNS.md | Established project conventions and design decisions |
| Phase DISCUSSION.md | Scope decisions, explicit in/out, architectural choices |
| Phase directory path | Where to write REVIEWS.md and OPEN_QUESTIONS.md |

## Classification Rules

### Classification Process

For EACH finding from EACH reviewer:

#### Step 1: Understand the finding
- What file/line is it about?
- What is the reviewer claiming is wrong?
- What would the fix look like?

#### Step 2: Cross-reference with project context

**Check Tier 1 KNOWLEDGE.md:**
- Does the finding CONTRADICT a Tier 1 rule? → Auto-dismiss (the reviewer is wrong about the project's conventions)
- Does the finding SUPPORT a Tier 1 rule? → High confidence real (the code violates an established rule)

**Check PATTERNS.md:**
- Is the finding about a pattern that is intentionally used in this project? → Likely dismiss
- Is the finding about a deviation from established patterns? → Likely real

**Check PLAN.md:**
- Was this approach explicitly planned? → Dismiss if the finding questions the approach itself
- Does the finding reveal a gap between plan and implementation? → Likely real

**Check DISCUSSION.md:**
- Was this explicitly decided during discussion? → Dismiss if the finding questions the decision

#### Step 3: Check for consensus

- 3+ reviewers flag the same issue → HIGH confidence real (regardless of individual evidence)
- 2 reviewers flag overlapping issues → MEDIUM confidence (examine carefully)
- 1 reviewer only → Apply normal evidence standards

#### Step 4: Classify

**Fix Now** (MUST be addressed before verification):
- Confidence >= 80% that it is a real issue AND
- Severity is critical/high AND
- Category is bug, security, correctness, or data-integrity

**Defer** (valid but not blocking):
- Confidence >= 60% that it is a real issue AND
- Severity is medium/low OR
- Category is performance, maintainability, style, tech-debt

**Dismiss** (false positive):
- Contradicts Tier 1 KNOWLEDGE.md
- About an intentional project pattern (with evidence from PATTERNS.md)
- About a decision explicitly made in DISCUSSION.md
- Reviewer misunderstood the code (explain why)

**Open Question** (ambiguous):
- Confidence 40-70% — could reasonably go either way
- Involves a tradeoff the human should weigh
- Multiple reviewers disagree on severity
- Not covered by existing KNOWLEDGE.md or PATTERNS.md

## Output Format

### REVIEWS.md

Follow the template from `.claude/dave/templates/reviews.md` exactly. Key requirements:
- Every finding gets a unique ID (F001, D001, X001)
- Every finding has: severity, category, source, location, description
- "Fix now" findings include suggested fix and KNOWLEDGE.md reference
- "Dismissed" findings include specific dismissal reason
- Summary section has accurate counts

### OPEN_QUESTIONS.md

Follow the template from `.claude/dave/templates/open-questions.md` exactly. Key requirements:
- Every question has: finding, source, location, ambiguity explanation
- "Aggregator's best guess" is always filled in (never blank)
- "What would resolve it" is actionable
- Decision fields start as "pending"

## Final Step: Verify Output Structure

Before returning REVIEWS.md and OPEN_QUESTIONS.md, verify:
1. Follow templates from `.claude/dave/templates/reviews.md` and `.claude/dave/templates/open-questions.md`
2. Summary counts match actual finding counts in each category
3. Every finding has a unique ID (F001, D001, X001 pattern)

## Quality Checks

Before finalizing output, verify:

1. **No finding lost:** Every finding from every reviewer is accounted for (classified into one of the four categories)
2. **No duplicate findings:** If multiple reviewers flag the same issue, merge into one finding with consensus count
3. **Dismissals are justified:** Every dismissal cites a specific KNOWLEDGE.md entry, PATTERNS.md convention, or DISCUSSION.md decision
4. **Fix now is actionable:** Every "fix now" item has enough context for a TDD developer to fix it
5. **Open questions are genuine:** Not just "I'm not sure" — each explains the specific tension
6. **Counts are accurate:** Summary counts match the actual findings in each section
