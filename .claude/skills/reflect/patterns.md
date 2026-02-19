# Failure Patterns Catalog

Known failure patterns from skills, subagents, and hooks. Use this to identify issues and apply proven fixes.

## Skill Patterns

### SK-001: Vague Description Syndrome

**Symptoms:**

- Skill invoked for unrelated tasks
- User confused why skill activated
- Context wasted on irrelevant guidance

**Example:**

```yaml
# BAD
description: Helps with database operations
# Invoked for: schema design, migrations, queries, backups, monitoring...

# GOOD
description: PostgreSQL schema design and Alembic migrations. Use when creating tables, adding columns, or writing migration files. NOT for query optimization or backups.
```

**Fix:** Add specific trigger terms AND explicit exclusions.

---

### SK-002: Missing Trigger Terms

**Symptoms:**

- User has to explicitly type `/skill-name`
- Skill not auto-invoked when it should be
- User doesn't know skill exists

**Example:**

```yaml
# BAD
description: Database migration patterns

# GOOD
description: Database migrations knowledge base - Alembic migrations with SQLModel and PostgreSQL. Consult when writing migrations, planning schema changes, deploying to production, handling rollbacks.
```

**Fix:** Include terms users naturally say: "migration", "schema change", "add column", "alter table".

---

### SK-003: Context Bloat

**Symptoms:**

- Skill loads 500+ lines every time
- Slow response times
- Irrelevant sections pollute context

**Example:**

```
# BAD - Everything in SKILL.md
SKILL.md (800 lines with all examples, edge cases, API reference)

# GOOD - Progressive disclosure
SKILL.md (150 lines - overview, common patterns)
reference.md (detailed API docs)
examples.md (comprehensive examples)
edge-cases.md (unusual scenarios)
```

**Fix:** Keep SKILL.md focused. Link to supporting files with "For X, see [file.md](file.md)".

---

### SK-004: Tool Restriction Mismatch

**Symptoms:**

- Skill suggests actions it can't perform
- Claude apologizes for limitations
- User has to invoke different skill/agent

**Example:**

```yaml
# BAD - Read-only skill suggests writes
allowed-tools: Read, Grep, Glob
# But skill content says "create a file with..."

# GOOD - Match capabilities to content
allowed-tools: Read, Grep, Glob
# Skill says "recommend creating..." not "create..."
```

**Fix:** Align skill content with tool restrictions. Use "recommend" not "do".

---

### SK-005: Conflicting Skills

**Symptoms:**

- Two skills provide contradictory guidance
- Claude alternates between approaches
- Inconsistent outputs

**Example:**

```
# Skill A says:
Use session.exec() for SQLModel queries

# Skill B says:
Use session.execute() for SQLAlchemy queries
```

**Fix:** Establish hierarchy or merge overlapping concerns into one skill.

---

## Subagent Patterns

### SA-001: Missing Skills Inheritance

**Symptoms:**

- Subagent doesn't follow project patterns
- Output inconsistent with main conversation
- Subagent "forgets" important context

**Root cause:** Subagents do NOT inherit skills from parent conversation.

**Example:**

```yaml
# BAD - No skills field
name: code-reviewer
tools: Read, Grep, Glob, Bash

# GOOD - Explicit skills
name: code-reviewer
tools: Read, Grep, Glob, Bash
skills: dagster, migrations
```

**Fix:** Always list required skills in `skills:` field.

---

### SA-002: Over-Spawning

**Symptoms:**

- Simple tasks delegated to subagents
- Unnecessary context switches
- Slow overall execution

**Example:**

```
# BAD - Spawning agent for trivial task
User: "Check git status"
Claude: [Spawns git-workflow-manager agent]

# GOOD - Handle inline
User: "Check git status"
Claude: [Runs git status directly]
```

**Fix:** Update agent description with "NOT for trivial X" or adjust trigger terms.

---

### SA-003: Tool Starvation

**Symptoms:**

- Agent starts task, then can't complete it
- "I don't have access to X" messages
- Partial results returned

**Example:**

```yaml
# BAD - Missing needed tool
name: code-validator
tools: Read, Grep, Glob
# Agent tries to run tests but can't use Bash

# GOOD - Complete toolset
name: code-validator
tools: Read, Grep, Glob, Bash
```

**Fix:** Add missing tools or update agent description to clarify limitations.

---

### SA-004: Model Waste

**Symptoms:**

- Slow responses for simple tasks
- High cost for routine operations
- Opus used for grep searches

**Example:**

```yaml
# BAD - Opus for exploration
name: explore
model: opus  # Expensive for search tasks

# GOOD - Haiku for speed
name: explore
model: haiku  # Fast and cheap for exploration
```

**Fix:** Use `haiku` for quick tasks, `sonnet` for balanced, `opus` for complex reasoning.

---

### SA-005: Context Pollution

**Symptoms:**

- Agent receives irrelevant conversation history
- Confused by earlier unrelated discussion
- Makes assumptions from wrong context

**Fix:** Provide clear, focused prompts when spawning. Include only relevant context.

---

## Hook Patterns

### HK-001: Over-Aggressive Validation

**Symptoms:**

- Valid operations blocked
- User has to disable hooks temporarily
- Frequent "hook rejected" messages

**Example:**

```json
// BAD - Blocks all file writes
{
  "pattern": "Write|Edit",
  "action": "reject"
}

// GOOD - Specific dangerous patterns only
{
  "pattern": "Write.*\\.env|Edit.*credentials",
  "action": "reject"
}
```

**Fix:** Narrow pattern scope. Allow valid operations.

---

### HK-002: Unclear Rejection Messages

**Symptoms:**

- Hook blocks operation with generic message
- User doesn't know what triggered block
- Repeated failed attempts

**Example:**

```json
// BAD
{ "message": "Operation not allowed" }

// GOOD
{ "message": "Cannot write to .env files. Use .env.example instead." }
```

**Fix:** Include specific reason and suggested alternative.

---

### HK-003: Performance Hooks

**Symptoms:**

- Noticeable delay on every operation
- Hooks run expensive checks repeatedly
- Same validation runs multiple times

**Fix:** Cache validation results. Skip redundant checks. Use async where possible.

---

### HK-004: Missing Safety Hooks

**Symptoms:**

- Dangerous operations executed without warning
- Production resources modified accidentally
- Secrets committed to git

**Recommended hooks:**

```json
{
  "hooks": {
    "pre-commit": {
      "patterns": ["*.env", "credentials*", "*.pem"],
      "action": "warn",
      "message": "Sensitive file detected in commit"
    }
  }
}
```

---

## Cross-Cutting Patterns

### CC-001: Description Drift

**Symptoms:**

- Skill/agent description doesn't match actual behavior
- Updated content but not description
- Incorrect invocation patterns

**Fix:** Review descriptions whenever content changes.

---

### CC-002: Orphaned Components

**Symptoms:**

- Skill/agent exists but never invoked
- Dead code in configuration
- Confusion about purpose

**Fix:** Remove unused components. Document why remaining ones exist.

---

### CC-003: Version Mismatch

**Symptoms:**

- Skills reference deprecated patterns
- Subagents use outdated APIs
- Hooks check for removed behaviors

**Fix:** Periodic review of all components against current codebase patterns.

---

### CC-004: Cross-Cutting Concern Amnesia

**Symptoms:**

- New gateway compiles, passes happy-path tests, but fails under production load
- Docstrings claim retry/rate-limiting exists when it doesn't
- Code review catches gateway *bypass* but not gateway *incompleteness*
- Missing observability (no Langfuse traces, no prompt provenance)

**Root cause:** Developers copy the *structural* pattern of existing gateways (class shape, factory, lazy init) but miss *behavioral* patterns (retry, rate limiting, tracing, provenance). Documentation describes requirements but nothing enforces them at review time.

**Example:**

```python
# BAD - Looks like a gateway but missing 5 cross-cutting concerns
class NewServiceGateway:
    _semaphore: asyncio.Semaphore  # Only concurrency, no TPM tracking
    # No retry config (or false docstring claiming SDK handles it)
    # No prompt template registration
    # No Langfuse trace_id capture
    # No CancelledError handling for circuit breaker

# GOOD - Full parity with LLMGateway
class NewServiceGateway:
    _executor: AsyncRateLimitedExecutor  # Concurrency + TPM
    # Explicit retry (verified from SDK source)
    # _register_prompt_template() with hash caching
    # trace_id capture via get_langfuse_client()
    # CancelledError -> record_cancelled()
```

**Fix:**
1. Code-reviewer agent has "New Gateway Completeness" checklist
2. DESIGN_PRINCIPLES.md lists all required concerns with verification steps
3. CLAUDE.md common mistakes table warns about this pattern
4. Reference `LLMGateway` as gold standard in all documentation

**First observed:** GeminiGateway (2026-02-11) — 5 missing concerns discovered during code review

---

## Pattern Template

Use this when documenting new patterns:

```markdown
### [ID]: [Pattern Name]

**Symptoms:**

- [Observable behavior 1]
- [Observable behavior 2]

**Root cause:** [Why this happens]

**Example:**
\`\`\`
# BAD
[problematic code/config]

# GOOD
[fixed code/config]
\`\`\`

**Fix:** [Specific action to resolve]
```
