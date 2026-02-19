# Knowledge Format Reference

Detailed specification for the KNOWLEDGE.md format used at all three scopes (phase, milestone, project).

---

## Entry Format

### Tier 1 -- Human-Provided

```markdown
- [H###] {Rule text}
  Source: {source type}
  Added: {YYYY-MM-DD}
  Severity: {severity level}
```

**Required fields:**

| Field | Format | Description |
|-------|--------|-------------|
| ID | `[H001]` through `[H999]` | Unique within scope. IDs are never reused even if an entry is removed. |
| Rule text | Free text, single line | Must be specific, actionable, and unambiguous. Should describe WHAT to do or avoid AND WHY. |
| Source | One of the source types below | Where the rule originated. |
| Added | `YYYY-MM-DD` | Date the entry was created. |
| Severity | `Critical`, `High`, `Medium`, or `Low` | Impact of violating this rule. |

**Source types for Tier 1:**

| Source | When used |
|--------|-----------|
| `Human` | Direct statement from the project owner during discussion or review |
| `Human (CLAUDE.md)` | Imported from CLAUDE.md during project initialization |
| `Human (review correction)` | Human corrected an agent's behavior during review |
| `Human (open question decision)` | Human resolved an OPEN_QUESTIONS.md item |
| `Human (discussion)` | Explicit decision made during Phase 1 |

**Severity guide:**

| Level | Criteria | Example |
|-------|----------|---------|
| Critical | Violation causes data loss, security breach, production outage, or irreversible damage | "Never use `get_session()` in tests -- connects to production" |
| High | Violation causes incorrect behavior, broken functionality, or silent data corruption | "All external calls must go through gateways" |
| Medium | Violation causes maintainability issues, performance degradation, or technical debt | "Use async-first design; add sync wrapper only if sync caller exists" |
| Low | Violation causes style inconsistency or minor convention breach | "Use `list[int]` not `List[int]` for type hints" |

### Tier 2 -- Agent-Discovered

```markdown
- [A###] {Rule text}
  Source: {source type}
  Added: {YYYY-MM-DD}
  Confidence: {confidence level}
  Verified: {N} times
  Promoted: {Yes | No}
```

**Required fields:**

| Field | Format | Description |
|-------|--------|-------------|
| ID | `[A001]` through `[A999]` | Unique within scope. IDs are never reused. |
| Rule text | Free text, single line | Must be specific, actionable, and unambiguous. |
| Source | One of the source types below | Which agent and event discovered it. |
| Added | `YYYY-MM-DD` | Date the entry was created. |
| Confidence | `HIGH`, `MEDIUM`, or `LOW` | How certain the finding is. |
| Verified | Non-negative integer | Number of independent verification events. |
| Promoted | `Yes` or `No` | Whether this has been promoted to Tier 1. |

**Optional fields:**

| Field | Format | Description |
|-------|--------|-------------|
| Promotion candidate | `Yes ({reason})` or `No` | Flagged when verified count exceeds `tier2_promotion_threshold` from config.yaml. |

**Source types for Tier 2:**

| Source | When used |
|--------|-----------|
| `Agent (reflect)` | Discovered during Phase 8 reflection |
| `Agent (code-review finding)` | Pattern identified by code reviewer |
| `Agent (verification failure)` | Caught during Phase 6 verification |
| `Agent (implementation issue)` | Discovered during Phase 4 implementation |
| `Agent (research)` | Found during Phase 2 research |

**Confidence criteria:**

| Level | Criteria | What it means for agents |
|-------|----------|--------------------------|
| HIGH | Confirmed by multiple independent verification events, or backed by official documentation | Follow this. Plan can depend on it. |
| MEDIUM | Confirmed once, consistent with project patterns, or from a credible source | Follow this, but have a fallback if it proves wrong. |
| LOW | Observed once, may be context-specific, or unverified | Be aware, but do not plan around this. |

---

## ID Scheme

### Allocation

| Range | Tier | Scope |
|-------|------|-------|
| H001 - H999 | Tier 1 | Human-provided, all scopes |
| A001 - A999 | Tier 2 | Agent-discovered, all scopes |

### Rules

1. **IDs are unique within their scope** (phase, milestone, or project). Different scopes can have the same ID number.
2. **IDs are never reused.** If entry H003 is removed, H003 is retired. The next entry is H004.
3. **IDs are assigned sequentially.** No gaps in assignment (gaps may exist from removals).
4. **Promoted entries get a new ID.** When A005 at the phase level is promoted to Tier 1 at the project level, it becomes H00N (the next available H-ID at project scope). The original A005 is marked `Promoted: Yes`.

---

## Promotion Process

Tier 2 entries can be promoted to Tier 1 when the human confirms them. The process:

### 1. Eligibility

An entry is eligible for promotion when:
- `Verified` count exceeds `tier2_promotion_threshold` from config.yaml (default: 3)
- The pattern has been observed in multiple contexts (not just one edge case)
- Reflect flags it with `Promotion candidate: Yes ({reason})`

### 2. Proposal

During Phase 8 (Reflect), the learning agent:
- Identifies eligible entries
- Generalizes the rule text if needed (phase-specific -> project-general)
- Proposes the promotion as a diff for human review

**Example proposal:**

```markdown
## Proposed Promotion

Tier 2 entry:
  [A005] SQLModel `session.exec()` returns ScalarResult -- never chain `.scalars()`
  Verified: 5 times across 3 phases

Proposed Tier 1 entry:
  [H012] SQLModel `session.exec()` returns ScalarResult directly -- do not chain `.scalars()`
  Source: Human (confirmed promotion from A005)
  Severity: High

Generalization: None needed -- rule is already general.
```

### 3. Human Decision

The human can:
- **Approve** -- Entry is added to project KNOWLEDGE.md as Tier 1. Original entry marked `Promoted: Yes`.
- **Modify and approve** -- Human adjusts the rule text, then approves.
- **Reject** -- Entry stays Tier 2. Promotion candidate flag is removed. Can be re-proposed later if more evidence accumulates.

### 4. After Promotion

- New Tier 1 entry is added to the target scope's KNOWLEDGE.md
- Original Tier 2 entry is marked `Promoted: Yes` with a reference to the new Tier 1 ID
- All agents now treat this as absolute authority (Tier 1)

---

## Aggregation Rules

Knowledge flows upward through generalization, not copying. Each level filters and generalizes.

### Phase -> Milestone (at milestone end)

**What moves up:**
- Patterns that recurred across 2+ phases
- Decisions that affect the milestone's overall architecture
- Mistakes that could bite other phases in the same milestone

**What stays at phase level:**
- One-off implementation details
- Context-specific findings (e.g., "batch size X on this specific hardware")
- Decisions that only affect one phase's internal structure

**Generalization example:**

| Phase entry | Milestone entry |
|-------------|-----------------|
| "PaddleOCR v3 batch > 4 causes OOM on RTX 3070 with 8GB VRAM" | "OCR providers need explicit batch size testing on target hardware before committing to a provider" |
| "The `reports` table unique constraint must include `provider_name` to allow re-processing with different models" | "Database unique constraints for pipeline results must include the processing method/model to support side-by-side comparisons" |

### Milestone -> Project (at milestone end)

**What moves up:**
- Lessons useful in UNRELATED future work
- Patterns that transcend the specific technology or domain
- Rules that would prevent mistakes in any future feature

**What stays at milestone level:**
- Milestone-specific context
- Technology-specific details (unless the tech is used project-wide)
- Decisions that only matter within this milestone's scope

**Generalization example:**

| Milestone entry | Project entry |
|-----------------|---------------|
| "OCR providers need explicit batch size testing on target hardware" | "External API providers need batch size testing on target hardware -- never trust published benchmarks" |
| "DB unique constraints for pipeline results must include processing method" | "Unique constraints for pipeline results must include the method/model -- re-running with a different model must preserve previous results" |

### Aggregation is LOSSY

Not everything flows up. This is intentional. The project KNOWLEDGE.md should contain only high-value, broadly applicable rules. If it grows to 50+ entries, it becomes noise that agents stop reading carefully.

Target sizes:
- Phase: 5-15 entries (specific, detailed)
- Milestone: 5-10 entries (aggregated, scoped)
- Project: 10-30 entries (generalized, durable)

---

## Examples

### Good Entries

```markdown
## Tier 1 (Human-Provided)

- [H001] Never use `get_session()` in tests -- connects to production database
  Source: Human | Added: 2025-01-15 | Severity: Critical

- [H002] All external calls must go through gateways (HttpGateway, LLMGateway, etc.)
  Source: Human (CLAUDE.md) | Added: 2025-01-01 | Severity: Critical

- [H003] Store WHAT ran (model name), not WHERE it ran (hosting platform). "PaddleOCR-VL-0.9B" not "vllm"
  Source: Human (review correction) | Added: 2025-02-01 | Severity: High

## Tier 2 (Agent-Discovered)

- [A001] PaddleOCR batch size > 8 causes OOM on RTX 3070 with 8GB VRAM
  Source: Agent (verification failure) | Added: 2025-02-01 | Confidence: HIGH
  Verified: 3 times | Promoted: No

- [A002] SQLModel `session.exec()` returns ScalarResult -- never use `.scalars()` after it
  Source: Agent (code-review finding) | Added: 2025-02-10 | Confidence: HIGH
  Verified: 5 times | Promoted: No
  Promotion candidate: Yes (consistent across 3 milestones, always correct)
```

**Why these are good:**
- Specific: You know exactly what to do or avoid
- Actionable: An agent reading this can change its behavior immediately
- Include the WHY: Explains the consequence of violation
- Include the correct alternative: Not just "don't do X" but "do Y instead"

### Bad Entries

```markdown
## Tier 1 (Human-Provided)

- [H001] Be careful with the database
  Source: Human | Added: 2025-01-15 | Severity: High

- [H002] Follow best practices
  Source: Human | Added: 2025-01-01 | Severity: Medium

## Tier 2 (Agent-Discovered)

- [A001] SQLModel has some quirks
  Source: Agent | Added: 2025-02-10 | Confidence: MEDIUM
  Verified: 1 times | Promoted: No

- [A002] Things can break if you are not careful
  Source: Agent (reflect) | Added: 2025-02-15 | Confidence: LOW
  Verified: 0 times | Promoted: No
```

**Why these are bad:**
- Vague: "Be careful with the database" provides zero actionable guidance
- Not actionable: An agent reading "follow best practices" cannot change its behavior
- Missing specifics: "SQLModel has some quirks" does not say WHICH quirks or how to handle them
- Missing consequences: No explanation of what goes wrong if violated
- Missing correct approach: Does not say what to do INSTEAD

### Edge Cases

**When an entry applies to multiple categories:**
Pick the most impactful one. If "never use `get_session()` in tests" is both a testing convention and a data safety rule, classify it by its highest-severity impact (data safety = Critical).

**When a Tier 2 entry contradicts a Tier 1 entry:**
Tier 1 always wins. The agent should flag the contradiction for human review but MUST follow the Tier 1 entry until the human resolves it.

**When the same pattern is discovered at phase level and already exists at project level:**
Increment the project-level entry's `Verified` count. Do not create a duplicate.
