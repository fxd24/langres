# Open Questions Template

Template for `.state/milestones/{slug}/phases/{N}/OPEN_QUESTIONS.md` -- ambiguous review findings that require human judgment.

**Purpose:** Capture findings where the review aggregator cannot confidently triage. Each question provides enough context for the human to make a quick decision. Decisions flow into phase KNOWLEDGE.md and may be generalized into project KNOWLEDGE.md during reflect.

**Downstream consumers:**
- `human` -- Makes decisions on each question
- `executor` (Phase 4 fix loop) -- Implements decisions marked as "fix"
- `reflect` (Phase 8) -- Examines decisions for generalizable patterns

---

## File Template

```markdown
# Phase {N}: {Name} - Open Questions

**Created:** {date}
**From review iteration:** {N}
**Total questions:** {N}
**Resolved:** {N}/{N}

## Questions

### [Q001] {Question title}

**Finding:** {What the reviewer flagged}
**Source:** {reviewer name}
**Location:** `{file_path}:{line_range}`

**Why it is ambiguous:**
{Explain why the aggregator could not confidently classify this as fix/defer/dismiss}

**Aggregator's best guess:** {fix | defer | dismiss} — {brief reasoning}

**What would resolve it:**
{What information or decision from the human would settle the question}

**Decision:** {pending | fix | defer | dismiss | not-applicable}
**Decision rationale:** {filled in after human decides}
**Decided by:** {human}
**Date:** {date}

---

### [Q002] {Question title}

**Finding:** {description}
**Source:** {reviewer}
**Location:** `{file_path}:{line_range}`

**Why it is ambiguous:**
{explanation}

**Aggregator's best guess:** {category} — {reasoning}

**What would resolve it:**
{needed information}

**Decision:** {pending}
**Decision rationale:**
**Decided by:**
**Date:**

---

## Decision Summary

<!-- Filled in after human reviews all questions. Used by reflect to
     extract patterns for KNOWLEDGE.md. -->

| ID | Decision | Rationale | Knowledge candidate? |
|----|----------|-----------|---------------------|
| Q001 | {fix/defer/dismiss} | {brief} | {yes/no — would this decision apply to future phases?} |
| Q002 | {decision} | {rationale} | {yes/no} |

---

*Phase: {N}*
*Questions created: {date}*
*All resolved: {yes | pending}*
```

<guidelines>

**When to create an open question:**
- Finding could reasonably be fix OR defer (aggregator confidence < 70%)
- Finding involves a tradeoff the human should weigh
- Finding touches a pattern not covered by KNOWLEDGE.md or PATTERNS.md
- Multiple reviewers disagree on severity or approach

**Question quality:**
- "Why it is ambiguous" must explain the SPECIFIC tension (not "I'm not sure")
- "Aggregator's best guess" forces a stance — never leave it blank
- "What would resolve it" must be actionable — what does the human need to tell us?

**After human review:**
- Decisions that reveal unstated conventions = candidates for KNOWLEDGE.md promotion
- Decisions that match existing KNOWLEDGE.md = validation (increase verified count)
- Decisions that contradict existing Tier 2 = demote or remove the Tier 2 entry

</guidelines>
