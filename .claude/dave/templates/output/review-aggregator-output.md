# Review Aggregator Output (two files)

## REVIEWS.md

```markdown
# Phase {N}: {Name} - Reviews

**Reviewed:** {date} | **Reviewers:** {reviewer_list} | **External:** {model_list|skipped}
**Fix loop iteration:** {iteration_number}

## Summary

| Category | Count |
|----------|-------|
| Fix now | {fix_count} |
| Defer | {defer_count} |
| Dismissed | {dismissed_count} |
| Open questions | {question_count} |

## Fix Now

### [{id}] {title}
- **Severity:** {critical|high} | **Category:** {bug|security|correctness|data-integrity}
- **Source:** {reviewer_name} | **Consensus:** {n}/{m} reviewers
- **Location:** `{file_path}:{line_range}`
- **Finding:** {what_is_wrong}
- **Suggested fix:** {how_to_fix}
- **KNOWLEDGE.md ref:** {entry_id|none}

## Defer

### [{id}] {title}
- **Severity:** {medium|low} | **Category:** {performance|maintainability|style|tech-debt}
- **Source:** {reviewer_name} | **Location:** `{file_path}:{line_range}`
- **Finding:** {what_could_be_improved}
- **Rationale for deferral:** {why_not_blocking}

## Dismissed

### [{id}] {title}
- **Source:** {reviewer_name} | **Location:** `{file_path}:{line_range}`
- **Finding:** {what_was_flagged}
- **Dismissal reason:** {reference_to_knowledge_patterns_or_discussion}

## Fix Loop History

| Iteration | Date | Fix Items | Fixed | New from Re-review |
|-----------|------|-----------|-------|--------------------|
| {n} | {date} | {count} | {count} | {count} |

*Phase: {N}* | *Fix loop converged: {yes|pending}*
```

## OPEN_QUESTIONS.md

```markdown
# Phase {N}: {Name} - Open Questions

**Created:** {date} | **Review iteration:** {iteration_number}
**Pending:** {pending_count}/{total_count}

### [{id}] {title}
- **Finding:** {what_reviewer_flagged}
- **Source:** {reviewer_name} | **Location:** `{file_path}:{line_range}`
- **Why ambiguous:** {specific_tension_explanation}
- **Best guess:** {fix|defer|dismiss} -- {reasoning}
- **Would resolve:** {actionable_information_needed}
- **Decision:** {pending|fix|defer|dismiss|not-applicable}

## Decision Summary

| ID | Decision | Rationale | Knowledge Candidate? |
|----|----------|-----------|---------------------|
| {id} | {decision} | {brief_rationale} | {yes|no} |

*Phase: {N}* | *All resolved: {yes|pending}*
```
