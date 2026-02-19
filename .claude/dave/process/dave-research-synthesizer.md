# dave-research-synthesizer: Process Guide

## Input Context

You receive from the research workflow orchestrator:

| Input | What it tells you |
|-------|------------------|
| Architect output | Architecture direction, codebase exploration, options comparison, recommendation |
| Topic researcher outputs (1 per topic) | Per-topic findings with confidence levels, strengths/weaknesses, pitfalls |
| Phase name and number | Context for the RESEARCH.md header |
| DISCUSSION.md content | Scope, decisions, constraints, success criteria, research topics |
| Project KNOWLEDGE.md | Tier 1 rules (for conflict resolution — Tier 1 always wins) |
| Project PATTERNS.md | Established conventions (for context on intentional patterns) |
| Project STACK.md | Tech stack context |
| RESEARCH.md template path | Structure to follow |
| Phase directory path | Where to write the output |
| Agent failure log | Which agents failed (topics become "Remaining Unknowns") |

## Process

### Step 1: Parse All Inputs

Read and organize all research agent outputs:

1. **Architect output** — Extract: recommended architecture, options considered, integration analysis, concerns
2. **Topic researcher outputs** — For each topic, extract: decision, recommendation, findings (with confidence levels), strengths, weaknesses, pitfalls, open questions
3. **Agent failures** — Note which topics have no research (these become Remaining Unknowns)
4. **Context files** — DISCUSSION.md (scope), KNOWLEDGE.md (Tier 1 rules), PATTERNS.md (conventions)

### Step 2: Validate Confidence Levels

For each finding from each agent, validate the confidence level against the source hierarchy:

| Claimed Level | Required Evidence | Action if Missing |
|---------------|------------------|-------------------|
| HIGH | Official docs URL, verified API, or multiple credible sources cited | Downgrade to MEDIUM with note |
| MEDIUM | At least one credible source cited, cross-referenced | Downgrade to LOW with note |
| LOW | Any source or explicit "from training data" | Keep as-is |

**Downgrade examples:**
- Agent claims [HIGH] but cites only one blog post → Downgrade to [MEDIUM]
- Agent claims [HIGH] but cites no source → Downgrade to [LOW] with "no source provided"
- Agent claims [MEDIUM] but source URL is clearly from training data → Downgrade to [LOW]

Record all downgrades for transparency.

### Step 3: Resolve Contradictions

Check for conflicts between research agents:

#### Architecture vs Topic
- Architect recommends approach X, but topic researcher found X has limitation Y → Reconcile. Does the limitation invalidate the architecture? Or is it manageable?

#### Topic vs Topic
- Two topic agents make conflicting claims about the same technology → Compare evidence quality. The agent with higher-quality sources (official docs > web search) wins.

#### Agent vs Tier 1
- Any agent recommendation that violates a Tier 1 rule → Tier 1 wins. Adjust the recommendation and note the override.

#### Agent vs PATTERNS.md
- Agent recommends a pattern that contradicts established conventions → Note the deviation. If the agent provides strong evidence for the new pattern, flag it as an open question for the user.

For each contradiction resolved, record:
- What conflicted
- What evidence each side had
- How it was resolved
- Confidence in the resolution

### Step 4: Identify Cross-Cutting Concerns

Look for themes that span multiple topics:

- **Shared infrastructure needs** — Multiple topics require the same capability (e.g., "all new services need gateway access")
- **Common error patterns** — Similar failure modes across topics
- **Performance considerations** — Concerns that compound when features interact
- **Testing patterns** — Shared test infrastructure needs
- **Configuration requirements** — Settings that affect multiple components

Cross-cutting concerns often reveal system-level risks that no individual agent would surface.

### Step 5: Compile Remaining Unknowns

Collect all unresolved items:

1. **Open questions from agents** — Collect from each topic researcher's output
2. **Failed agent topics** — Topics where the agent returned empty/error
3. **Unresolved contradictions** — Conflicts that could not be resolved with available evidence
4. **LOW confidence findings** — Items the plan should not depend on without validation

For each unknown:
- **What we know** (partial information)
- **What we don't know** (the gap)
- **Risk level** (HIGH if the plan depends on this, MEDIUM if it affects quality, LOW if it is informational)
- **Mitigation** (fallback, test early, or accept risk)

### Step 6: Identify New Questions for Discussion

Check if synthesis revealed questions that should go back to the user:

A new question qualifies if:
- Research found a constraint not anticipated in DISCUSSION.md
- Two viable approaches exist with significant tradeoff the user should weigh
- A Tier 1 rule conflicts with the most practical approach
- A HIGH-confidence finding contradicts a discussion assumption
- An unknown carries HIGH risk for the plan

List new questions (if any) with context and suggested resolution options.

### Step 7: Write RESEARCH.md

Assemble the document following this structure:

```markdown
# Phase {N}: {Name} - Research

**Researched:** {date}
**Synthesized by:** dave-research-synthesizer
**Overall confidence:** {HIGH if most findings are HIGH, MEDIUM if mixed, LOW if significant unknowns}

<architecture_direction>
## Architecture Direction

{Synthesized from architect output. Primary approach, key choices, integration points.}
{If architect was skipped, note it and use DISCUSSION.md decisions.}

</architecture_direction>

<findings>
## Research Findings

### Topic 1: {name}

**Decision:** {specific, actionable}
**Recommendation:** {with rationale}
**Alternatives considered:**
| Alternative | Pros | Cons | Why rejected |
|...|...|...|...|

**Findings:**
- [HIGH] {finding} Source: {URL}
- [MEDIUM] {finding} Source: {URL}
- [LOW] {finding} Source: {URL}
{Note any confidence downgrades from validation}

**Pitfalls:**
- {pitfall}: {cause and prevention}

**Open questions:**
- {unresolved items}

### Topic 2: {name}
...

</findings>

<cross_cutting>
## Cross-Cutting Concerns

- **{Concern 1}:** {what, why, how to handle}
- **{Concern 2}:** {what, why, how to handle}

</cross_cutting>

<contradictions>
## Contradictions Resolved

- **{Contradiction}:** {what conflicted, evidence on each side, resolution, confidence}

</contradictions>

<unknowns>
## Remaining Unknowns

1. **{Unknown}**
   - What we know: {partial info}
   - What we don't know: {gap}
   - Risk level: {HIGH/MEDIUM/LOW}
   - Mitigation: {approach}

</unknowns>

{If new questions exist:}
## New Questions for Discussion

*Questions that emerged during research synthesis.*

1. **{Question}**
   - Context: {what prompted it}
   - Why it matters: {impact}
   - Options: {what the user can decide}

<sources>
## Sources

### Primary (HIGH confidence)
- {URL} — {what}

### Secondary (MEDIUM confidence)
- {URL} — {what}

### Tertiary (LOW confidence)
- {URL} — {what}

</sources>

---

*Phase: {N}*
*Research completed: {date}*
*Ready for planning: {yes/no — reason if no}*
```

**Quality checks before writing:**
- Every finding has confidence level AND source
- Every recommendation has both strengths and weaknesses
- Alternatives explain WHY they were rejected
- Pitfalls are actionable
- No unresolved contradictions remain without being documented
- Cross-cutting concerns are identified (not just individual topic findings)
- Confidence downgrades are noted for transparency
- "Ready for planning" is "no" if HIGH-risk unknowns remain

## Final Step: Verify Output Structure

Before returning RESEARCH.md, verify it matches the inline template defined above:
1. Executive summary is present and concise (3-5 sentences)
2. Every agent's findings are accounted for (none lost)
3. Contradictions section exists (even if "none found")
4. Readiness assessment is present with clear READY/NOT READY verdict

## Success Criteria

Synthesis is complete when:

- [ ] All agent outputs parsed and organized
- [ ] Confidence levels validated against source hierarchy (downgrades noted)
- [ ] All contradictions identified and resolved (or documented as unresolvable)
- [ ] Cross-cutting concerns identified across topics
- [ ] Remaining unknowns compiled with risk assessments
- [ ] New questions for discussion identified (if any)
- [ ] RESEARCH.md written following the template structure
- [ ] Every finding from every agent is accounted for in the final document
- [ ] "Ready for planning" assessment is honest and justified
- [ ] Document is coherent — reads as a unified analysis, not a concatenation of agent outputs
