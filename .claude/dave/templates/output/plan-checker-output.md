# Plan Validation Report

**Plan:** {phase name}
**Iteration:** {N}
**Result:** {PASS | REVISE (blockers found) | REVISE (warnings only)}

## Executive Summary

- **Blockers:** {N} — {1-line description of most critical, or "none"}
- **Warnings:** {N} — {1-line description of most impactful, or "none"}
- **Coverage:** {N}/{M} requirements covered

## Blockers (Must Fix)

{List each blocker with:}
### {Blocker title}
- **Check:** {which check found this}
- **Issue:** {specific description of the problem}
- **Fix:** {specific suggestion for how to fix it}
- **Affected:** {which task/truth/artifact is affected}

## Warnings (Should Fix)

{List each warning with:}
### {Warning title}
- **Check:** {which check found this}
- **Issue:** {specific description}
- **Suggestion:** {how to improve}

## Notes (Nice to Have)

- {observation or suggestion}

## Validation Summary

| Check | Status | Details |
|-------|--------|---------|
| Requirement coverage | {PASS/FAIL} | {N}/{M} requirements covered |
| Must-haves quality | {PASS/FAIL} | {issues if any} |
| Task completeness | {PASS/FAIL} | {N}/{M} tasks complete |
| Test specification | {PASS/FAIL} | {issues if any} |
| Dependency correctness | {PASS/FAIL} | {N} waves, no cycles |
| Verification matrix | {PASS/FAIL} | {N}/{M} truths verifiable |
| Knowledge compliance | {PASS/FAIL} | {N} Tier 1 rules checked |
| Scope sanity | {PASS/FAIL} | {N} tasks, {M} files |

## Recommendation

{One of:}
- **APPROVE** — Plan is ready for user review. No blockers, warnings are minor.
- **REVISE** — Fix {N} blockers before approval. Specific fixes listed above.
- **SPLIT** — Plan exceeds scope constraints. Recommend splitting into {description of split}.
