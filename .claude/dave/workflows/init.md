<purpose>
Initialize the Dave Framework `.state/` directory for this project. Detects available tools, extracts Tier 1 knowledge from CLAUDE.md, seeds architecture patterns, tech stack, known concerns, and creates the initial state file. This is the foundation — every subsequent Dave command reads from what init creates.
</purpose>

<required_reading>
Read all files referenced by the invoking prompt's execution_context before starting.
</required_reading>

<process>

## 1. Check Existing State

**MANDATORY FIRST STEP — Check if `.state/` already exists:**

```bash
ls -d .state/ 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If `.state/` exists:**

Use AskUserQuestion:
- header: "Existing State Detected"
- question: "`.state/` already exists. What would you like to do?"
- options:
  - "Reset" — Delete and reinitialize from scratch
  - "Update" — Keep existing state, re-extract from CLAUDE.md and merge
  - "Cancel" — Exit without changes

**If "Reset":**
```bash
rm -rf .state/
```
Continue to Step 2.

**If "Update":** Skip directory creation (Step 2), go directly to Step 4 (tool detection) and Step 5 (knowledge extraction). Merge new entries with existing files — do not overwrite manually added entries. Mark any new entries with `Added: {today's date}`.

**If "Cancel":** Exit.

**If `.state/` does not exist:** Continue to Step 2.

## 2. Create Directory Structure

<!-- PARALLEL WORKTREE NOTE: STATE.md is per-branch. Each worktree's branch has its
     own STATE.md pointing to its active phase. Init creates state for THIS branch. -->

Create the full `.state/` directory tree:

```bash
mkdir -p .state/project
mkdir -p .state/codebase
mkdir -p .state/milestones
mkdir -p .state/debug/resolved
```

Verify the structure was created:

```bash
find .state/ -type d | sort
```

Expected output:
```
.state/
.state/codebase
.state/debug
.state/debug/resolved
.state/milestones
.state/project
```

## 3. Detect Available Tools

Detect what tools and capabilities are available in this environment. Run each check independently:

**Chrome MCP:**
```bash
# Check if Chrome MCP tools are available (will be visible in tool list)
# For now, check if the MCP server config exists
ls ~/.claude/mcp_servers.json 2>/dev/null && echo "MCP_CONFIG_EXISTS" || echo "NO_MCP_CONFIG"
```

**Docker:**
```bash
docker --version 2>/dev/null && echo "DOCKER_AVAILABLE" || echo "DOCKER_MISSING"
docker compose version 2>/dev/null && echo "COMPOSE_AVAILABLE" || echo "COMPOSE_MISSING"
```

**Database:**
```bash
# Check if database connection is configured
grep -q "PGHOST" .env 2>/dev/null && echo "DB_CONFIGURED" || echo "DB_NOT_CONFIGURED"
```

**Build tools:**
```bash
# Check Makefile targets
grep -E "^(test|lint|db-)" Makefile 2>/dev/null | head -10
```

**Python/uv:**
```bash
uv --version 2>/dev/null && echo "UV_AVAILABLE" || echo "UV_MISSING"
python3 --version 2>/dev/null
```

**External review models:**
```bash
which codex 2>/dev/null && echo "CODEX_AVAILABLE" || echo "CODEX_MISSING"
which opencode 2>/dev/null && echo "OPENCODE_AVAILABLE" || echo "OPENCODE_MISSING"
```

**Git:**
```bash
git --version 2>/dev/null && echo "GIT_AVAILABLE" || echo "GIT_MISSING"
```

Record all detection results for use in config.yaml generation.

## 4. Create config.yaml

Write `.state/project/config.yaml` using the detected tools. Use the template from the framework spec (`.agent/README.md` Configuration section) as the base structure.

For each tool category, set `available: true` or `available: false` based on detection results from Step 3.

**Template structure:**

```yaml
# Dave Framework Configuration
# Generated: {today's date}
# Last updated: {today's date}

# Models
models:
  primary: claude-opus-4-6
  profiles:
    quality:   { planner: opus, executor: opus, verifier: sonnet }
    balanced:  { planner: opus, executor: sonnet, verifier: sonnet }
    budget:    { planner: sonnet, executor: sonnet, verifier: haiku }

# External review models
review_models:
  - name: codex
    command: "codex exec -m gpt-5.3-codex -c 'model_reasoning_effort=\"high\"'"
    strengths: "code-focused reasoning, catches logic bugs"
    available: {detected}
  - name: opencode
    command: "opencode run -m opencode/kimi-k2.5-free --variant high"
    strengths: "different reasoning approach, catches design issues"
    available: {detected}

# Build tools
tools:
  test: "{detected test command or 'make test'}"
  lint: "{detected lint command or 'make lint'}"
  run_script: "{detected run command or 'uv run --env-file .env python'}"

# Verification tools
verification:
  chrome_mcp:
    available: {detected}
    type: browser
    capabilities: [navigate, click, screenshot, read_page, form_input]
    notes: "Requires Chrome with extension running"
  bash:
    available: true
    type: script
    capabilities: [run_command, check_exit_code, file_operations]
  database:
    available: {detected}
    type: query
    capabilities: [select, count, verify_schema]
    test_connection: "make db-test"
    query_tool: "uv run --env-file .env python -c"
  docker:
    available: {detected}
    type: container
    capabilities: [build, run, compose]

# Knowledge settings
knowledge:
  tier2_promotion_threshold: 3
```

Fill in `{detected}` values from Step 3 results.

## 5. Extract Tier 1 Knowledge from CLAUDE.md

Read CLAUDE.md from the project root. Extract rules from the "Common Mistakes" table and other explicit rules throughout the file.

**5a. Parse Common Mistakes Table**

Read the "Common Mistakes (Read Before Coding)" section. Each row in the table becomes a Tier 1 KNOWLEDGE.md entry.

Format each entry as:

```markdown
- [H{NNN}] {Mistake description} — {Rule}
  Source: Human (CLAUDE.md) | Added: {today's date} | Severity: {Critical|High|Medium}
```

**Severity mapping:**
- Rules about production data, safety, external calls, DB connections → Critical
- Rules about naming, patterns, architecture → High
- Rules about style, preferences → Medium

**5b. Parse Additional Rules**

Scan these CLAUDE.md sections for additional rules:
- "Data Safety (ABSOLUTE RULES)" — each absolute rule becomes Severity: Critical
- "Code Style" — deprecated patterns become High severity entries
- "Key Commands" — operational rules
- "Critical Import Patterns" — import conventions

**5c. Write KNOWLEDGE.md**

Write `.state/project/KNOWLEDGE.md`:

```markdown
# Project Knowledge

Knowledge entries with provenance. Tier 1 (human-provided) has absolute authority.
See .agent/README.md "Knowledge System" for full specification.

## Tier 1 (Human-Provided)

{extracted entries from 5a and 5b}

## Tier 2 (Agent-Discovered)

(None yet — populated during development phases)
```

Count the total entries extracted and save for the summary.

## 6. Extract Architecture Patterns

Read CLAUDE.md "Architecture" section and project docs for architecture patterns:

**6a. Read CLAUDE.md Architecture**

Extract:
- Clean Architecture (DDD) layer structure
- Orchestrating services pattern
- Gateway pattern for external calls
- 3-phase DB pattern
- Repository pattern

**6b. Read Additional Docs (if they exist)**

Check and read if available:
- `docs/DESIGN_PRINCIPLES.md` — design principles and patterns
- `docs/ARCHITECTURE.md` — detailed architecture

Only read what exists. Do not error on missing files.

**6c. Write PATTERNS.md**

Write `.state/project/PATTERNS.md`:

```markdown
# Architecture Patterns

Conventions and patterns established for this project.
Updated by reflect agent after each phase. Human-confirmed.

## Layer Architecture

{extracted from CLAUDE.md Architecture section}

## Service Patterns

{orchestrating services, domain services, gateway pattern}

## Database Patterns

{3-phase pattern, session management, SQLModel conventions}

## Import Conventions

{from src.module import ..., never relative imports}

## Code Style

{type hints, logging, Python 3.12+ conventions}
```

## 7. Extract Tech Stack

Read CLAUDE.md for technology information.

Write `.state/project/STACK.md`:

```markdown
# Tech Stack

Languages, frameworks, libraries, and infrastructure.

## Languages
- Python 3.12+

## Frameworks & Libraries
{extracted from CLAUDE.md — SQLModel, Dagster, Pydantic, etc.}

## Infrastructure
{PostgreSQL, Qdrant, Azure Blob Storage, Docker}

## Development Tools
{uv, ruff, alembic, make}

## AI/ML
{LLM providers, embedding models, OCR}
```

## 8. Extract Known Concerns

Read CLAUDE.md "Known Issues" reference (`docs/ISSUES.md`) and "Data Safety" section.

Write `.state/project/CONCERNS.md`:

```markdown
# Known Concerns

Tech debt, known issues, and things to watch for.
Updated by reflect agent after each phase.

## Production Safety
{from Data Safety section — production resources, safety rules}

## Known Issues
{reference to docs/ISSUES.md if it exists, extract key items}

## Tech Debt
(None identified yet — populated during development)

## Watch List
{items from CLAUDE.md that need ongoing attention}
```

## 9. Create PROJECT.md

Extract a project summary from CLAUDE.md "Project Overview" section.

Write `.state/project/PROJECT.md`:

```markdown
# Project: {project name}

## What This Project Is

{1-2 paragraph summary from CLAUDE.md "Project Overview" — what the project does, who it serves}

## Current Stage

{Current focus and stage from CLAUDE.md — what is being built right now}

## Value Proposition

{Why this project matters — extracted from the overview or inferred from the project description}

## Key Constraints

{Non-negotiable constraints that shape all decisions — from "Data Safety" section and architecture requirements}
- {Constraint 1}
- {Constraint 2}

## Evolution

{How the project has evolved — from "Evolution" subsection if present}

## Next Steps

{What comes after the current stage — from "Next Step" subsection if present}

---

*Extracted from: CLAUDE.md*
*Generated: {today's date}*
```

**If CLAUDE.md does not have a clear "Project Overview" section:** Write a minimal PROJECT.md with what can be inferred from the rest of CLAUDE.md (folder structure, architecture, key services). Note gaps for the user to fill in.

## 10. Create STATE.md

Write `.state/STATE.md`:

```markdown
# Dave Framework State

## Project Reference

See: .state/project/PROJECT.md (project summary)
See: CLAUDE.md (project rules and conventions)
See: .agent/README.md (framework specification)

**Initialized:** {today's date}
**Current focus:** Not started — run a milestone to begin

## Current Position

Milestone: None active
Phase: N/A
Status: Initialized — ready to start first milestone
Last activity: {today's date} — Project state initialized via /dave:init

## Knowledge Summary

Tier 1 entries: {count from Step 5}
Tier 2 entries: 0
Patterns documented: {count of sections in PATTERNS.md}
Concerns tracked: {count of items in CONCERNS.md}

## Tools Available

{summary list from config.yaml — e.g., "Docker: yes, Database: yes, Chrome MCP: no"}

## Session Continuity

Last session: {today's date and time}
Stopped at: Initialization complete
Resume file: None
```

## 11. Present Summary

Display a structured summary of everything that was created:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DAVE FRAMEWORK ► INITIALIZED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Files Created

  .state/
  ├── project/
  │   ├── PROJECT.md         — project summary and constraints
  │   ├── config.yaml        — {N} tools detected, {M} review models
  │   ├── KNOWLEDGE.md       — {K} Tier 1 entries extracted
  │   ├── PATTERNS.md        — {P} pattern categories
  │   ├── STACK.md           — tech stack documented
  │   └── CONCERNS.md        — {C} concerns tracked
  ├── codebase/              — (empty, populated by codebase mapping)
  ├── milestones/            — (empty, populated when milestones begin)
  ├── debug/                 — (empty, populated by debug sessions)
  └── STATE.md               — session state initialized

## Knowledge Extraction

  Source: CLAUDE.md
  Tier 1 rules extracted: {K}
  - Critical: {count}
  - High: {count}
  - Medium: {count}

## Tool Detection

  Build: {test command} / {lint command}
  Docker: {available/missing}
  Database: {configured/not configured}
  Chrome MCP: {available/missing}
  External review: {codex status} / {opencode status}

## Next Steps

  /dave:state          — Inspect what was created in detail
  /dave:state sync     — Compare CLAUDE.md coverage with extracted knowledge
  Start a milestone    — Begin the first development cycle
```

</process>

<success_criteria>
- [ ] `.state/` directory structure created with all subdirectories
- [ ] `PROJECT.md` captures project summary, constraints, and evolution
- [ ] `config.yaml` reflects actual tool availability (not assumed)
- [ ] `KNOWLEDGE.md` has Tier 1 entries extracted from CLAUDE.md Common Mistakes table
- [ ] `PATTERNS.md` documents the project's architecture patterns
- [ ] `STACK.md` lists the actual tech stack
- [ ] `CONCERNS.md` captures known issues and safety rules
- [ ] `STATE.md` initialized with correct counts and tool summary
- [ ] Summary shown to user with actionable next steps
</success_criteria>
