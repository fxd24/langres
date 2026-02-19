# Discussion Template

Template for `.state/milestones/{slug}/phases/{N}/DISCUSSION.md` -- captures scope, guardrails, and decisions from the discussion phase.

**Purpose:** Remove ambiguity and establish boundaries within which all subsequent agents operate autonomously. After discussion, the AI should have enough context to research, plan, implement, review, and verify without asking.

**Downstream consumers:**
- `research` (Phase 2) -- Reads identified research topics to know WHAT to investigate
- `planner` (Phase 3) -- Reads scope, decisions, and constraints to create an executable plan
- `tdd-developer` (Phase 4) -- Reads success criteria to understand WHAT "done" looks like
- `code-reviewer` (Phase 5) -- Reads constraints to check compliance
- `verifier` (Phase 6) -- Reads success criteria to verify goal achievement

---

## File Template

```markdown
# Phase {N}: {Name} - Discussion

<!-- SUMMARY: {1-2 sentence executive summary: what is being built, key constraints, and number of research topics identified} -->

**Discussed:** {date}
**Status:** Ready for research

<scope>
## Scope

### In Scope
<!-- What this phase delivers. Be specific -- vague scope causes scope creep. -->
- {Deliverable 1 -- concrete, observable}
- {Deliverable 2}
- {Deliverable 3}

### Out of Scope
<!-- What this phase explicitly does NOT deliver. Include reasoning. -->
- {Exclusion 1} -- {why it is out of scope}
- {Exclusion 2} -- {why}

### Deferred
<!-- Ideas that came up during discussion but belong in later phases. -->
- {Deferred idea 1} -- {which phase or "backlog"}
- {Deferred idea 2} -- {destination}

</scope>

<decisions>
## Architectural Decisions

### {Decision Area 1}
- **Decision:** {What was decided}
- **Rationale:** {Why this choice was made}
- **Alternatives rejected:** {What else was considered and why it was rejected}

### {Decision Area 2}
- **Decision:** {What was decided}
- **Rationale:** {Why}

### Agent Discretion
<!-- Areas where the human explicitly said "you decide." Agents have
     flexibility here during research, planning, and implementation. -->
- {Area 1 -- what the agent can decide freely}
- {Area 2}

</decisions>

<constraints>
## Constraints and Guardrails

### Hard Constraints
<!-- Non-negotiable limits. Violation is a bug. -->
- {Constraint 1 -- e.g., "All external calls must go through gateways"}
- {Constraint 2}

### Soft Constraints
<!-- Preferences that can be overridden with justification. -->
- {Preference 1 -- e.g., "Prefer async over sync unless complexity is disproportionate"}
- {Preference 2}

### Integration Points
<!-- How this phase connects to existing code. Critical for the planner. -->
- {Integration point 1 -- e.g., "Must use existing PDFProcessingService interface"}
- {Integration point 2}

</constraints>

<success>
## Success Criteria

<!-- Observable behaviors that must be true when the phase is complete.
     These become the "truths" in the plan's must-haves section. -->

- [ ] {Criterion 1 -- user-observable, testable}
- [ ] {Criterion 2}
- [ ] {Criterion 3}

**How to demonstrate success:**
{Description of how to show the feature working end-to-end. This guides
the verifier in Phase 6.}

</success>

<research_topics>
## Research Topics

<!-- Topics identified during discussion that need investigation before
     planning. Each topic has a question and why it matters. These are
     consumed by Phase 2 (Research Orchestration). -->

### Topic 1: {topic name}
- **Question:** {What needs to be answered}
- **Why it matters:** {How the answer affects planning or implementation}
- **Known context:** {What we already know, if anything}

### Topic 2: {topic name}
- **Question:** {question}
- **Why it matters:** {impact}

</research_topics>

<open_questions>
## Open Questions

<!-- Questions that could not be resolved during discussion. These are
     tracked and resolved later -- either during research, planning, or
     flagged in OPEN_QUESTIONS.md for human review. -->

1. **{Question}**
   - Context: {What prompted this question}
   - Impact: {What decisions depend on the answer}
   - Proposed resolution: {When and how to resolve -- e.g., "research phase", "ask human during review"}

</open_questions>

---

*Phase: {N}*
*Discussion completed: {date}*
*Ready for research: {yes | no -- and why not}*
```

<guidelines>

**This template captures BOUNDARIES for downstream agents.**

The output should answer:
- "What is the researcher investigating?" (research topics)
- "What choices are locked for the planner?" (decisions, constraints)
- "What does 'done' look like for the verifier?" (success criteria)
- "What is explicitly NOT being built?" (scope exclusions)

**Good content (concrete, actionable):**
- "All OCR results must include provenance metadata (provider name, model version, timestamp)"
- "Re-running the pipeline on the same document must not create duplicates"
- "Out of scope: batch processing of multiple PDFs -- Phase 3"

**Bad content (vague, unactionable):**
- "Should handle errors well"
- "Good performance"
- "Clean code"

**After creation:**
- File lives at `.state/milestones/{slug}/phases/{N}/DISCUSSION.md`
- Phase 2 (Research) reads research_topics to know what to investigate
- Phase 3 (Plan) reads all sections to create an executable plan
- Downstream agents should NOT need to ask the user again about captured decisions

</guidelines>
