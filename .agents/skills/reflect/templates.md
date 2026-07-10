# Reflection Report Templates

Example outputs from the reflect skill. Use these as reference for format and depth.

## Example 1: Data Pipeline Feature Session

```markdown
## Reflection Report

### Session Overview

- **Complexity**: High
- **Main objectives**: Implement funder deduplication pipeline
- **Outcome**: Partial success - dedup works, entity resolution incomplete

### Skills Analysis

| Skill      | Invocations | Effectiveness | Issues                                                      |
| ---------- | ----------- | ------------- | ----------------------------------------------------------- |
| dagster    | 3           | Good          | Invoked correctly for asset design                          |
| migrations | 1           | Poor          | Not invoked when schema needed, user had to explicitly call |

**Recommended changes:**

- [ ] migrations: Update description to include "schema", "table", "column"
- [ ] dagster: Add "backfill" and "partition" to trigger terms

### Subagent Analysis

| Agent           | Spawns | Effectiveness | Issues                                          |
| --------------- | ------ | ------------- | ----------------------------------------------- |
| database-expert | 1      | Poor          | Missing migrations skill, wrote raw DDL instead |
| code-reviewer   | 1      | Good          | Caught missing data validation                  |

**Recommended changes:**

- [ ] database-expert: Add `skills: migrations` to inherit project patterns

### Hook Analysis

No hooks triggered during this session.

### Failure Patterns Detected

1. **SA-001: Missing Skills Inheritance**
   - Occurrences: 1 (database-expert)
   - Impact: High - raw DDL instead of Alembic migration
   - Fix: Add migrations to database-expert skills field

2. **SK-002: Missing Trigger Terms**
   - Occurrences: 1 (migrations)
   - Impact: Medium - user had to manually invoke
   - Fix: Add "schema", "table", "add column" to description

### Preservation Checklist

Before applying changes, verify:

- [x] Existing trigger terms still work (tested: "migration", "alembic")
- [x] No skills/agents become orphaned
- [ ] Tool restrictions don't break current workflows (need to verify database-expert)
- [x] Changes don't conflict with other skills

### Actionable Improvements

1. [ ] Edit `.claude/agents/database-expert.md`: Add `skills: migrations`
2. [ ] Edit `.claude/skills/migrations/SKILL.md`: Add trigger terms to description
```

---

## Example 2: Debugging Session

```markdown
## Reflection Report

### Session Overview

- **Complexity**: Medium
- **Main objectives**: Fix SQLModel query returning wrong results
- **Outcome**: Success after 3 attempts

### Skills Analysis

| Skill | Invocations | Effectiveness | Issues                                                |
| ----- | ----------- | ------------- | ----------------------------------------------------- |
| dagster | 2         | Mixed         | Invoked but issue was SQLModel, not Dagster           |

**Recommended changes:**

- [ ] dagster: Add clarification "NOT for SQLModel/SQLAlchemy query issues"

### Subagent Analysis

| Agent         | Spawns | Effectiveness | Issues                                      |
| ------------- | ------ | ------------- | ------------------------------------------- |
| Explore       | 2      | Good          | Found similar patterns in codebase          |

**Recommended changes:**

None needed.

### Hook Analysis

No hooks triggered.

### Failure Patterns Detected

1. **SK-005: Conflicting Guidance**
   - Occurrences: 1 (dagster vs SQLModel patterns)
   - Impact: Medium - confusion about correct approach
   - Fix: Add explicit boundary between skills

### Actionable Improvements

1. [ ] Edit `.claude/skills/dagster/SKILL.md`: Add "NOT for pure SQLModel queries" clarification
```

---

## Example 3: Migration Session

```markdown
## Reflection Report

### Session Overview

- **Complexity**: Low
- **Main objectives**: Add new column to organizations table
- **Outcome**: Success

### Skills Analysis

| Skill      | Invocations | Effectiveness | Issues                         |
| ---------- | ----------- | ------------- | ------------------------------ |
| migrations | 1           | Good          | Correctly guided Alembic usage |

**Recommended changes:**
None - skills appropriately invoked.

### Subagent Analysis

| Agent           | Spawns | Effectiveness | Issues |
| --------------- | ------ | ------------- | ------ |
| database-expert | 1      | Good          | Reviewed migration correctly |

**Recommended changes:**
None - agent performed as expected.

### Hook Analysis

No hooks triggered.

### Failure Patterns Detected

None - clean session.

### Preservation Checklist

All items verified - no changes needed.

### Actionable Improvements

Session completed successfully. No improvements needed.

**Positive patterns to preserve:**

- migrations skill correctly identified for schema change
- database-expert reviewed migration before apply
- No over-spawning of agents for simple task
```

---

## Example 4: Vector Search Feature

```markdown
## Reflection Report

### Session Overview

- **Complexity**: High
- **Main objectives**: Implement hybrid search with Qdrant
- **Outcome**: Success after iteration

### Skills Analysis

| Skill   | Invocations | Effectiveness | Issues                                         |
| ------- | ----------- | ------------- | ---------------------------------------------- |
| dagster | 4           | Good          | Correctly guided asset definitions             |
| (none)  | -           | -             | Missing Qdrant skill - had to research inline  |

**Recommended changes:**

- [ ] Create new skill: `qdrant` for vector search patterns

### Subagent Analysis

| Agent                    | Spawns | Effectiveness | Issues                                    |
| ------------------------ | ------ | ------------- | ----------------------------------------- |
| Explore                  | 2      | Good          | Found existing vector search code         |
| qualitative-evaluator    | 1      | Good          | Assessed search result quality            |

**Recommended changes:**
None needed.

### Failure Patterns Detected

1. **SK-003: Context Bloat**
   - Occurrences: 1 (dagster loaded 600+ lines)
   - Impact: Low - but could optimize
   - Fix: Move advanced partitioning examples to separate file

### Actionable Improvements

1. [ ] Create `.claude/skills/qdrant/SKILL.md`: Vector search patterns for this project
2. [ ] Edit `.claude/skills/dagster/SKILL.md`: Move advanced examples to supporting file
```

---

## Report Checklist

Before finalizing a reflection report:

- [ ] All skills invoked are listed (check conversation for `/skill` usage)
- [ ] All subagents spawned are listed (check for "Task tool" usage)
- [ ] Hook triggers are counted (if hooks are configured)
- [ ] Patterns are categorized (use IDs from patterns.md)
- [ ] Recommendations are specific (include file paths and exact changes)
- [ ] Preservation risks are assessed
- [ ] Priority is assigned (highest impact, lowest risk first)
