# Summary Template

Template for `.state/milestones/{slug}/phases/{N}/SUMMARY.md` -- phase completion summary with metrics, decisions, and lessons learned.

**Purpose:** Provide a concise record of what happened during this phase. Captures the delta between what was planned and what was actually built, key decisions made, and knowledge extracted. This is the historical record for future reference.

**Downstream consumers:**
- `reflect` (milestone aggregation) -- Reads all phase summaries to aggregate milestone-level lessons
- `planner` (future phases) -- Reads past summaries to understand what patterns worked
- `human` -- Quick reference for what a phase accomplished

---

## File Template

```markdown
# Phase {N}: {Name} - Summary

**Completed:** {YYYY-MM-DD}
**Duration:** {time from first commit to push}
**PR:** {PR URL or "not pushed yet"}

## What Was Built

{2-4 sentences describing what this phase delivered, written in past tense.
Derived from PLAN.md must-haves and what was actually implemented.}

### Must-Haves Achieved

| # | Truth | Status |
|---|-------|--------|
| T1 | {truth text} | {Verified | Verified with deviation} |
| T2 | {truth text} | {status} |

### Key Artifacts

| Artifact | Path | Lines | Wired |
|----------|------|-------|-------|
| {name} | `{path}` | {N} | {yes: imported by X | no} |

## Metrics

| Metric | Value |
|--------|-------|
| Tasks planned | {N} |
| Tasks completed | {N} |
| Execution waves | {N} |
| Commits | {N} |
| Tests added | {N} |
| Lines changed | {+N / -N} |
| Review iterations | {N} |
| Fix-now items resolved | {N} |
| Deferred items | {N} |
| Verification confidence | {HIGH | MEDIUM | LOW} |
| Plan deviations | {N} |

## Deviations from Plan

<!-- Only include this section if deviations occurred.
     Each deviation explains WHAT changed and WHY. -->

### DEV-1: {Brief description}
- **Task:** {task ID}
- **Planned:** {what the plan said}
- **Actual:** {what was actually done}
- **Reason:** {why the deviation was necessary}
- **Impact:** {how this affected must-haves or downstream work}

## Key Decisions

<!-- Decisions made during this phase that future phases should know about.
     Sourced from OPEN_QUESTIONS.md resolutions and execution deviations. -->

- **{Decision title}:** {What was decided and why}
  Source: {OPEN_QUESTIONS.md Q001 | Execution deviation DEV-1 | Discussion}

## Lessons Learned

<!-- New knowledge extracted during reflect. These become Tier 2 entries
     in phase KNOWLEDGE.md and may be promoted to Tier 1. -->

- {Lesson 1 — specific, actionable, explains what to do differently}
- {Lesson 2}

## Deferred Work

<!-- Items from REVIEWS.md that were deferred. These are follow-up work
     for future phases or separate tasks. -->

- [D001] {Deferred finding title} — {brief description}
- [D002] {Deferred finding title} — {brief description}

## Open Items

<!-- Anything unresolved that the next phase or milestone should address. -->

- {Open item 1}

---

*Phase: {N}*
*Completed: {YYYY-MM-DD}*
*Reflected: {YYYY-MM-DD}*
```

<guidelines>

**Content quality:**
- Summary should be self-contained — a reader should understand what happened without reading other phase files
- Metrics must be accurate (derived from actual artifacts, not estimated)
- Deviations must explain WHY, not just WHAT
- Lessons must be specific and actionable (see KNOWLEDGE.md quality guidelines)
- Deferred work should link back to REVIEWS.md finding IDs

**When to create:**
- After the phase is complete (at minimum, verification passed)
- Created by the reflect workflow, not manually

**After creation:**
- Phase KNOWLEDGE.md updated with new Tier 2 entries
- STATE.md updated with phase completion
- If this is the last phase in a milestone, knowledge aggregation triggers

**Size target:**
- Keep under 100 lines
- This is a summary, not a detailed log
- Link to other phase files for details

</guidelines>
