# Research Template

Template for `.state/milestones/{slug}/phases/{N}/RESEARCH.md` -- comprehensive research findings synthesized by the research workflow orchestrator from parallel research agents (dave-architect + dave-topic-researchers).

**Purpose:** Document what was researched, what was decided, and what confidence level backs each finding. The planner uses this to make informed technical decisions without re-researching.

**Downstream consumers:**
- `planner` (Phase 3) -- Uses recommendations to select approaches, findings to anticipate pitfalls, and confidence levels to plan fallbacks
- `tdd-developer` (Phase 4) -- References code patterns and pitfall avoidance guidance
- `reflect` (Phase 8) -- Extracts learnings for KNOWLEDGE.md if research predictions proved wrong

---

## File Template

```markdown
# Phase {N}: {Name} - Research

<!-- SUMMARY: {1-2 sentences: recommended approach, overall confidence level, number of topics researched, key risks identified} -->

**Researched:** {date}
**Time budget:** {X}% of estimated implementation time
**Overall confidence:** {HIGH | MEDIUM | LOW}

<architecture_direction>
## Architecture Direction

{High-level approach for this phase. Key architectural decisions with
rationale. How this fits into the existing codebase architecture.}

**Primary approach:** {One-liner summary of the recommended direction}

**Key architectural choices:**
- {Choice 1}: {What and why}
- {Choice 2}: {What and why}

**How it connects to existing code:**
- {Integration point 1}: {How the new code fits}
- {Integration point 2}: {How}

</architecture_direction>

<findings>
## Research Findings

### Topic 1: {topic name}

**Decision:** {What was decided and why. Must be specific and actionable.}

**Recommendation:** {Primary recommendation with rationale. What the planner
should use as the default approach.}

**Alternatives considered:**
| Alternative | Pros | Cons | Why rejected |
|-------------|------|------|--------------|
| {option A} | {pros} | {cons} | {reason} |
| {option B} | {pros} | {cons} | {reason} |

**Findings:**
- [HIGH] {Finding backed by official docs or multiple credible sources}
  Source: {URL or reference}
- [MEDIUM] {Finding from credible source, not independently confirmed}
  Source: {URL or reference}
- [LOW] {Finding needing validation -- single source or training data only}
  Source: {URL or reference}

**Pitfalls:**
- {Common mistake 1}: {What goes wrong and how to avoid it}
- {Common mistake 2}: {What goes wrong and how to avoid it}

**Open questions:**
- {Anything that could not be fully resolved. Will be addressed in planning
  or flagged as explicit risk.}

### Topic 2: {topic name}

**Decision:** {decision}

**Recommendation:** {recommendation}

**Alternatives considered:**
| Alternative | Pros | Cons | Why rejected |
|-------------|------|------|--------------|
| {option} | {pros} | {cons} | {reason} |

**Findings:**
- [{confidence}] {finding}
  Source: {source}

**Pitfalls:**
- {pitfall}

**Open questions:**
- {question}

</findings>

<cross_cutting>
## Cross-Cutting Concerns

<!-- Things that affect multiple research topics. E.g., "all providers have
     rate limits," "all DB operations need the 3-phase pattern," "error
     handling at service boundaries follows the same pattern." -->

- **{Concern 1}:** {What it affects and how to handle it}
- **{Concern 2}:** {What it affects and how to handle it}

</cross_cutting>

<unknowns>
## Remaining Unknowns

<!-- Questions that research could not answer. These become explicit risks
     in the plan. The planner must acknowledge these and either plan
     fallbacks or flag them as acceptable risks. -->

1. **{Unknown 1}**
   - What we know: {partial information}
   - What we do not know: {the gap}
   - Risk level: {HIGH | MEDIUM | LOW}
   - Mitigation: {How the plan should handle this -- fallback, test early, or accept risk}

2. **{Unknown 2}**
   - What we know: {info}
   - What we do not know: {gap}
   - Risk level: {level}
   - Mitigation: {approach}

</unknowns>

<sources>
## Sources

### Primary (HIGH confidence)
- {Official docs URL or library reference} -- {what was checked}
- {Codebase pattern} -- {where in the codebase and what it shows}

### Secondary (MEDIUM confidence)
- {Web search verified against official source} -- {finding + verification}

### Tertiary (LOW confidence -- needs validation)
- {Web search only or training data} -- {finding, flagged for validation}

</sources>

---

*Phase: {N}*
*Research completed: {date}*
*Ready for planning: {yes | no -- and why not}*
```

<guidelines>

**When to create:**
- After Phase 1 (Discussion) produces DISCUSSION.md with identified research topics
- Before Phase 3 (Plan) -- the planner depends on this

**Structure:**
- architecture_direction comes first (sets the frame for all findings)
- Each topic from DISCUSSION.md's research_topics gets its own findings section
- Cross-cutting concerns capture patterns that span topics
- Remaining unknowns become explicit risks in the plan

**Content quality:**
- Decisions must be specific: "Use PaddleOCR v3 with batch size 4" not "Use an OCR provider"
- Findings must have confidence levels and sources
- Pitfalls must be actionable: "X goes wrong because Y -- avoid by doing Z"
- Alternatives must explain WHY they were rejected, not just that they were

**Confidence criteria:**
- HIGH: Official docs, verified library APIs, multiple credible sources agree
- MEDIUM: Web search verified against one official source, credible but unconfirmed
- LOW: Single web source, unverified, training data only

**Source hierarchy (highest to lowest):**
1. Official documentation -- state as fact
2. Codebase patterns -- this is how the project actually works
3. Web search (verified) -- cross-referenced with official sources
4. Web search (single source) -- flag for validation

**Time-boxing:**
- Research is capped at 15-20% of estimated implementation time
- Goal: gather enough to plan well, not encyclopedic coverage
- If a topic cannot be resolved within budget, record it as a remaining unknown

**After creation:**
- File lives at `.state/milestones/{slug}/phases/{N}/RESEARCH.md`
- The planner loads it to make informed technical decisions
- TDD developers reference it for patterns and pitfall avoidance
- Reflect checks whether research predictions were accurate

</guidelines>
