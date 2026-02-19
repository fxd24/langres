# Learning Extraction Report

## Plan vs Actual

### Gaps Found: {N}

{For each gap:}
#### Gap {N}: {title}
- **Task:** {task ID}
- **Planned:** {what was planned}
- **Actual:** {what happened}
- **Root cause:** {why the gap occurred}
- **Lesson:** {what to do differently — becomes a Tier 2 entry if it passes checks}

### No Gaps
{If no gaps, state: "Plan executed as designed. No deviations."}

## Review Patterns

### Top Categories
| Category | Count | Pattern |
|----------|-------|---------|
| {category} | {N} | {pattern description} |

### Recurring Themes
{List themes that appeared across multiple findings}

### False Positive Themes
{List dismissed findings that share a common cause — may indicate missing documentation}

## Verification Patterns

### Layer Results
| Layer | Status | Notes |
|-------|--------|-------|
| Plan Conformance | {pass/fail} | {notes} |
| Code Review | {pass/fail} | {notes} |
| Automated Functional | {pass/fail} | {notes} |
| Human Oversight | {pass/fail} | {notes} |

### Improvement Suggestions
{Specific suggestions for improving verification in future phases}

## New Knowledge Entries

### Tier 2 Entries to Add

{List of new Tier 2 entries in knowledge format}

### Existing Entries to Update

| Entry | Current Verified | New Verified | Notes |
|-------|-----------------|-------------|-------|
| [A###] | {N} | {N+1} | {evidence from this phase} |

### Promotion Candidates

{List of entries that now exceed the promotion threshold}

## Pattern Updates

### New Patterns for PATTERNS.md
{List of new conventions to add}

### New Concerns for CONCERNS.md
{List of new tech debt or risks}

### Resolved Concerns
{List of CONCERNS.md items addressed by this phase}

## Phase Summary Data

{Key metrics and facts for SUMMARY.md — the orchestrator uses this to write SUMMARY.md}

- Phase name: {name}
- Tasks: {completed}/{planned}
- Commits: {N}
- Review iterations: {N}
- Verification confidence: {level}
- Deviations: {N}
- Key decisions: {list}
- Lessons: {count}
