# Knowledge Template

Template for `KNOWLEDGE.md` files at every scope: phase, milestone, and project.

---

## File Template

```markdown
# Knowledge: {scope name}

> Pitfalls, rules, and learnings with explicit provenance. Tier 1 entries
> have absolute authority and cannot be overridden by agents. Tier 2 entries
> are valuable but can be questioned or demoted.

## Tier 1 (Human-Provided)

<!-- Rules explicitly stated by the project owner, from CLAUDE.md, from
     corrections during discussion or review, or from decisions on open
     questions. Agents MUST follow these. Only the human can modify or
     remove Tier 1 entries. -->

- [H001] {Rule text -- specific, actionable, unambiguous}
  Source: {Human | Human (CLAUDE.md) | Human (review correction) | Human (open question decision)}
  Added: {YYYY-MM-DD}
  Severity: {Critical | High | Medium | Low}

- [H002] {Rule text}
  Source: {source}
  Added: {YYYY-MM-DD}
  Severity: {severity}

## Tier 2 (Agent-Discovered)

<!-- Patterns and pitfalls found during development. Identified by reflect
     from review findings, verification failures, or implementation issues.
     Agents should follow these but can flag conflicts. Entries can be
     promoted to Tier 1 when the human confirms them. -->

- [A001] {Rule text -- specific, actionable, unambiguous}
  Source: {Agent (reflect) | Agent (code-review finding) | Agent (verification failure) | Agent (implementation issue)}
  Added: {YYYY-MM-DD}
  Confidence: {HIGH | MEDIUM | LOW}
  Verified: {N} times
  Promoted: {Yes | No}

- [A002] {Rule text}
  Source: {source}
  Added: {YYYY-MM-DD}
  Confidence: {confidence}
  Verified: {N} times
  Promoted: {No}
  Promotion candidate: {Yes (reason) | No}
```

<purpose>

KNOWLEDGE.md is the learning system's persistent memory. It captures rules and
pitfalls with explicit provenance so that agents know WHAT to avoid and WHY,
and so that human-provided rules are never overridden by agent-discovered ones.

**Problem it solves:** Without provenance, agent-discovered patterns can
silently override human decisions. Without persistence, the same mistakes are
repeated across phases and milestones.

**Solution:** A tiered system where:
- Tier 1 (human) has absolute authority
- Tier 2 (agent) is valuable but subordinate
- Each entry has source, date, confidence, and verification count
- Entries flow upward through generalization (phase -> milestone -> project)

</purpose>

<lifecycle>

**Creation:**
- Phase KNOWLEDGE.md: Created during Phase 8 (Reflect) after each phase
- Milestone KNOWLEDGE.md: Created at milestone end by aggregating phase entries
- Project KNOWLEDGE.md: Seeded from CLAUDE.md during init, updated at milestone end

**Writing -- Phase level:**
- After each phase, reflect records:
  - Specific decisions made during this phase
  - Mistakes caught by review or verification
  - Deviations from plan and why
  - Open question resolutions (become Tier 1 since human decided)

**Writing -- Milestone level:**
- At milestone end, reflect aggregates across phases:
  - Keeps patterns relevant to the milestone scope
  - Generalizes implementation-specific details
  - Discards one-off details that do not recur

**Writing -- Project level:**
- At milestone end, reflect proposes generalizations:
  - Only lessons useful in UNRELATED future work
  - Tier 1 additions require human approval
  - Tier 2 promotions require human approval

**Reading:**
- Planner reads project + milestone KNOWLEDGE.md to avoid known pitfalls
- TDD Developer reads project KNOWLEDGE.md to follow conventions
- Code Reviewer reads project KNOWLEDGE.md to check compliance
- Aggregator reads project KNOWLEDGE.md to filter false positives
- Verifier reads project KNOWLEDGE.md to verify against known issues

</lifecycle>

<sections>

### Tier 1 (Human-Provided)

Rules with absolute authority. Sources:
- Direct statements from the project owner
- Content from CLAUDE.md (imported during init)
- Corrections given during discussion or review
- Decisions made on OPEN_QUESTIONS.md items

**Required fields:**
- `[H###]` -- ID in range H001-H999
- Rule text -- must be specific and actionable
- Source -- where the rule came from
- Added -- date the rule was added
- Severity -- Critical / High / Medium / Low

**Severity guide:**
- Critical: Violation causes data loss, security breach, or production outage
- High: Violation causes incorrect behavior or broken functionality
- Medium: Violation causes maintainability or performance issues
- Low: Violation causes style or convention inconsistency

### Tier 2 (Agent-Discovered)

Patterns found during development with confidence tracking.

**Required fields:**
- `[A###]` -- ID in range A001-A999
- Rule text -- must be specific and actionable
- Source -- which agent/phase discovered it
- Added -- date the entry was added
- Confidence -- HIGH / MEDIUM / LOW
- Verified -- number of independent confirmations
- Promoted -- whether it has been promoted to Tier 1

**Optional fields:**
- Promotion candidate -- flagged when verified count exceeds threshold

**Confidence criteria:**
- HIGH: Confirmed by multiple independent verification events
- MEDIUM: Confirmed once, consistent with project patterns
- LOW: Observed once, may be context-specific

</sections>

<aggregation>

### How Knowledge Flows Upward

```
Phase level (specific)          Milestone level (scoped)         Project level (general)
"PaddleOCR v3 batch > 4        "OCR providers need batch        "External API providers
 causes OOM on RTX 3070         size testing on target           need batch size testing
 with this config"              hardware before choosing"        on target hardware"
         |                               |                               |
         +------ generalize ------->     +------ generalize ------->     |
```

**Phase -> Milestone:**
- Happens at milestone end
- Keeps: Patterns that recurred across phases or affected multiple areas
- Drops: One-off implementation details, context-specific findings

**Milestone -> Project:**
- Happens at milestone end
- Keeps: Lessons useful in UNRELATED future work
- Drops: Milestone-specific context, technology-specific details (unless the
  tech is used project-wide)
- Requires: Human approval for all Tier 1 additions or promotions

</aggregation>

<good_vs_bad>

### Good Entries

```markdown
- [H001] Never use `get_session()` in tests -- connects to production
  Source: Human | Added: 2025-01-15 | Severity: Critical

- [A001] SQLModel `session.exec()` returns ScalarResult -- never chain `.scalars()`
  Source: Agent (code-review finding) | Added: 2025-02-10 | Confidence: HIGH
  Verified: 5 times | Promoted: No
```

Good because: specific, actionable, explains WHY, includes the mistake AND the
correct approach.

### Bad Entries

```markdown
- [H001] Be careful with the database
  Source: Human | Added: 2025-01-15 | Severity: High

- [A001] SQLModel has some quirks
  Source: Agent | Added: 2025-02-10 | Confidence: MEDIUM
  Verified: 1 times | Promoted: No
```

Bad because: vague, not actionable, does not explain what to do or avoid. An
agent reading "be careful with the database" gains no useful information.

</good_vs_bad>
