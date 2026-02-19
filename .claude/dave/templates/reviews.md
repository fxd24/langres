# Reviews Template

Template for `.state/milestones/{slug}/phases/{N}/REVIEWS.md` -- aggregated, triaged review findings from multi-agent code review.

**Context summary pattern:** Add `<!-- SUMMARY: {fix-now count} fixes required, {defer count} deferred, {dismiss count} dismissed. Fix loop: {converged/not converged} in {N} iterations. -->` after the file header.

**Purpose:** Capture all review findings from internal and external reviewers, triaged by the review aggregator into actionable categories. This is the single source of truth for what reviewers found and what to do about it.

**Downstream consumers:**
- `executor` (Phase 4 fix loop) -- Reads "fix now" items to create focused fix plans
- `verifier` (Phase 6) -- Confirms "fix now" items were addressed
- `reflect` (Phase 8) -- Analyzes review patterns for knowledge extraction

---

## File Template

```markdown
# Phase {N}: {Name} - Reviews

**Reviewed:** {date}
**Reviewers:** {list of reviewers that ran}
**External models:** {list of external models, or "skipped"}
**Fix loop iteration:** {1 | 2 | ...}

## Summary

- **Total findings:** {N}
- **Fix now:** {N} (bugs, security, correctness — must fix before verify)
- **Defer:** {N} (valid but not blocking — create GitHub issues)
- **Dismissed:** {N} (false positives, explained below)
- **Open questions:** {N} (ambiguous — sent to OPEN_QUESTIONS.md)

## Fix Now

<!-- Findings that MUST be addressed before verification. These loop back
     to Phase 4 (scoped TDD). Each finding has enough context for the
     TDD developer to fix it without asking questions. -->

### [F001] {Finding title}
- **Severity:** critical | high
- **Category:** bug | security | correctness | data-integrity
- **Source:** {reviewer name or external model}
- **Consensus:** {N}/{M} reviewers flagged this
- **Location:** `{file_path}:{line_range}`
- **Finding:** {What is wrong}
- **Impact:** {What happens if not fixed}
- **Suggested fix:** {How to fix it}
- **KNOWLEDGE.md reference:** {H001, A003, or "none"}

### [F002] {Finding title}
...

## Defer

<!-- Valid findings that are not blocking. Each gets a GitHub issue. -->

### [D001] {Finding title}
- **Severity:** medium | low
- **Category:** performance | maintainability | style | tech-debt
- **Source:** {reviewer}
- **Location:** `{file_path}:{line_range}`
- **Finding:** {What could be improved}
- **Rationale for deferral:** {Why this is not blocking}

### [D002] {Finding title}
...

## Dismissed

<!-- False positives with explanation. Logged for transparency and to
     improve future reviews. -->

### [X001] {Finding title}
- **Source:** {reviewer}
- **Location:** `{file_path}:{line_range}`
- **Finding:** {What the reviewer flagged}
- **Dismissal reason:** {Why this is a false positive — e.g., "contradicts Tier 1 rule H002", "reviewer lacked context about intentional pattern X"}

### [X002] {Finding title}
...

## Fix Loop History

<!-- Track convergence across fix loop iterations. Each iteration should
     be lighter than the previous. If not converging, the problem is in
     triage, not in the code. -->

### Iteration 1
- **Date:** {date}
- **Fix now items:** {N}
- **Items fixed:** {N}
- **New items from re-review:** {N}

### Iteration 2
- **Date:** {date}
- **Fix now items:** {N} (should be fewer)
- **Items fixed:** {N}
- **New items from re-review:** {N} (should be 0 or very few)

---

*Phase: {N}*
*Reviews created: {date}*
*Fix loop converged: {yes | pending}*
*Open questions: see OPEN_QUESTIONS.md*
```

<guidelines>

**Aggregation quality:**
- Every finding has a clear severity, category, and source
- Consensus boost: 3+ reviewers flagging same issue = high confidence
- Convention filter: findings contradicting Tier 1 KNOWLEDGE = auto-dismiss
- No finding without a concrete location (file + line range)

**Fix now criteria:**
- Bugs (code does not work as intended)
- Security vulnerabilities (OWASP Top 10, secrets exposure)
- Correctness issues (wrong output, data corruption risk)
- Data integrity issues (missing constraints, duplicate risk)

**Defer criteria:**
- Performance improvements not affecting correctness
- Style or maintainability improvements
- Tech debt that is not in the current scope

**Dismissal criteria:**
- Contradicts Tier 1 KNOWLEDGE.md entry
- Reviewer lacked context about intentional project pattern
- Finding is about code outside the current phase scope
- Suggestion conflicts with an explicit decision in DISCUSSION.md

**Fix loop convergence:**
- Each iteration should have fewer "fix now" items
- New findings from re-review should approach zero
- If iteration 3+ still produces new findings, escalate to user

</guidelines>
