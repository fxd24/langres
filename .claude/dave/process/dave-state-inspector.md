# dave-state-inspector: Process Guide

## Input Context

You inspect `.state/` directory contents. Mode determines scope: full (everything), knowledge (Tier 1/2 entries), config (tool availability), sync (CLAUDE.md alignment).

## Knowledge System Reference

The Dave Framework knowledge system has four layers. Understand their relationships before inspecting.

```
.state/
├── project/                    # Project context (accumulates over project lifetime)
│   ├── PROJECT.md              # What this project is, constraints, value proposition
│   ├── PATTERNS.md             # Architecture patterns, conventions, design decisions
│   ├── KNOWLEDGE.md            # Pitfalls & rules with provenance tiers
│   ├── STACK.md                # Tech stack, libraries, versions, rationale
│   ├── CONCERNS.md             # Known issues, tech debt, things to watch for
│   └── config.yaml             # External tools, models, verification capabilities
│
├── codebase/                   # Codebase analysis (updated by learning agent)
│   ├── STRUCTURE.md            # Where code lives, directory layout, naming patterns
│   ├── ARCHITECTURE.md         # Layers, data flow, entry points, key abstractions
│   └── CONVENTIONS.md          # Code style, imports, type hints, testing patterns
│
├── milestones/                 # Per-milestone lifecycle state
│   └── {milestone-slug}/
│       ├── ROADMAP.md
│       ├── RESEARCH.md
│       ├── KNOWLEDGE.md
│       └── phases/{N}/...
│
├── STATE.md                    # Current position, velocity, session continuity
│
└── debug/                      # Debug sessions
```

**Knowledge Provenance:**
- **Tier 1 (Human-Provided):** Absolute authority. From CLAUDE.md, human corrections, human decisions. Format: `[H00N]`
- **Tier 2 (Agent-Discovered):** Standard authority. From reflect, review findings, verification failures. Format: `[A00N]`

**Key consumers of state:**
- Planner reads: KNOWLEDGE.md, PATTERNS.md, CONCERNS.md, ARCHITECTURE.md, config.yaml
- TDD Developer reads: KNOWLEDGE.md, PATTERNS.md, CONVENTIONS.md
- Code Reviewer reads: KNOWLEDGE.md, PATTERNS.md, ARCHITECTURE.md
- Verifier reads: KNOWLEDGE.md, PLAN.md, ARCHITECTURE.md
- Reflect reads: Everything in .state/

## Inspection Modes

### Mode: Full Health Check (no arguments)

When invoked without arguments, perform a comprehensive health check across all layers.

#### Step 1: Inventory

Check existence and recency of every expected file:

```
.state/project/PROJECT.md       — exists? last modified?
.state/project/PATTERNS.md      — exists? last modified?
.state/project/KNOWLEDGE.md     — exists? last modified?
.state/project/STACK.md         — exists? last modified?
.state/project/CONCERNS.md      — exists? last modified?
.state/project/config.yaml      — exists? last modified?
.state/codebase/STRUCTURE.md    — exists? last modified?
.state/codebase/ARCHITECTURE.md — exists? last modified?
.state/codebase/CONVENTIONS.md  — exists? last modified?
.state/STATE.md                 — exists? last modified?
```

For each file: record existence, last-modified date, and line count.

#### Step 2: Content Quality

For each existing file, assess:

- **Completeness:** Does it have substantive content or is it a skeleton/template?
- **Recency:** Is the content current relative to recent git activity? Compare last-modified date against recent commits.
- **Structure:** Does it follow expected format (proper headings, IDs for knowledge entries, YAML frontmatter where expected)?

#### Step 3: Knowledge System Health

If KNOWLEDGE.md exists:

- Count Tier 1 entries (pattern: `[H0XX]`)
- Count Tier 2 entries (pattern: `[A0XX]`)
- Identify promotion candidates (Tier 2 entries with `Verified: N times` where N >= threshold from config.yaml, default 3)
- Identify entries without proper IDs
- Identify entries without Source, Added date, or Severity/Confidence tags
- Check for ID gaps or duplicates

#### Step 4: Config Validation

If config.yaml exists:

- Parse YAML structure
- For each tool entry with `available: true`, verify the tool actually exists:
  - `chrome_mcp`: Check if MCP browser tools are accessible
  - `playwright`: `which playwright 2>/dev/null`
  - `bash`: Always available
  - `database`: `make db-test 2>/dev/null` or check env vars
  - `docker`: `which docker 2>/dev/null && docker info 2>/dev/null`
- For each review model with `available: true`, verify the command exists:
  - Check if base command is in PATH
- Flag tools marked `available: false` that are actually installed
- Flag tools marked `available: true` that are not found

#### Step 5: Codebase Understanding Freshness

If codebase/ files exist:

- Compare file paths mentioned in STRUCTURE.md against actual directory layout
- Check if ARCHITECTURE.md references current service/module names
- Check if CONVENTIONS.md patterns match current code style

#### Step 6: Milestone State

If milestones/ directory exists:

- List active milestones
- For each milestone: count phases, check for incomplete phases (PLAN.md without VERIFICATION.md or SUMMARY.md)
- Identify the current active phase from STATE.md

#### Step 7: Cross-Reference with CLAUDE.md

Read CLAUDE.md and check:

- Are critical rules from CLAUDE.md represented as Tier 1 entries in KNOWLEDGE.md?
- Are architecture patterns from CLAUDE.md captured in PATTERNS.md?
- Are known issues from CLAUDE.md reflected in CONCERNS.md?
- Are stack details from CLAUDE.md in STACK.md?

#### Step 8: Agent Reference Validation

Verify that all Dave Framework agent cross-references resolve to existing files:

- For each `dave-*.md` in `.claude/agents/`:
  - Read the `<setup>` section
  - Verify referenced process file exists (`.claude/dave/process/dave-{name}.md`)
  - Verify referenced output template exists (`.claude/dave/templates/output/{name}-output.md`) — unless template is inline
- For each process file in `.claude/dave/process/`:
  - Check if it references a template file — verify that file exists
  - Check for references to shared rules files (`.claude/dave/rules/`) — verify they exist
  - Check for references to shared reference files (`.claude/dave/references/`) — verify they exist
- For each workflow file in `.claude/dave/workflows/`:
  - Check agent references (subagent_type values) — verify matching agent file exists in `.claude/agents/`
  - Check template references — verify template files exist in `.claude/dave/templates/`

Flag broken references as **Critical** severity (agent will fail to load its process guide).

---

### Mode: Knowledge Focus ("knowledge" argument)

Deep dive into the knowledge system only.

#### Step 1: Load All Knowledge Files

Read:
- `.state/project/KNOWLEDGE.md` (project-level)
- `.state/milestones/*/KNOWLEDGE.md` (milestone-level, all milestones)
- `.state/milestones/*/phases/*/KNOWLEDGE.md` (phase-level, all phases)

#### Step 2: Tier 1 Analysis

- List all Tier 1 entries with IDs, severity, and added dates
- Check for entries in CLAUDE.md that should be Tier 1 but are missing
- Verify no Tier 1 entries contradict each other
- Check for Tier 1 entries that reference deprecated patterns or removed code

#### Step 3: Tier 2 Analysis

- List all Tier 2 entries with IDs, confidence, and verification counts
- Identify promotion candidates (verified >= threshold, confidence HIGH)
- Identify stale entries (not verified in a long time, may no longer apply)
- Identify entries that should be demoted or removed (LOW confidence, never verified)

#### Step 4: Knowledge Flow Check

- Phase entries that should have been aggregated to milestone level but were not
- Milestone entries that should have been generalized to project level but were not
- Project entries that are too specific (contain phase/milestone-specific details)

#### Step 5: Coverage Analysis

Compare knowledge entries against:
- CLAUDE.md "Common Mistakes" table -- each row should have a corresponding knowledge entry
- CLAUDE.md "Architecture Anti-Patterns" -- each should have a knowledge entry
- CLAUDE.md deprecated patterns -- each should have a knowledge entry

---

### Mode: Config Focus ("config" argument)

Deep dive into configuration and tool availability.

#### Step 1: Parse config.yaml

Read and parse `.state/project/config.yaml`. If it does not exist, report this as critical and propose creating one from detected tools.

#### Step 2: Tool Detection

Run actual detection for each tool category:

```bash
# Build tools
which make 2>/dev/null && echo "make: available"
which uv 2>/dev/null && echo "uv: available"

# Verification tools
which docker 2>/dev/null && echo "docker: available"
which playwright 2>/dev/null && echo "playwright: available"

# Review models
which codex 2>/dev/null && echo "codex: available"
which opencode 2>/dev/null && echo "opencode: available"

# Database connectivity
echo "Checking PGHOST env var..."
```

#### Step 3: Reconcile

Compare detected tools against config.yaml declarations:

- Tools detected but not in config
- Tools in config as available but not detected
- Tools in config as unavailable but now detected
- Missing tool categories

#### Step 4: Model Configuration

- Verify model profiles make sense (e.g., opus for planning, sonnet for execution)
- Check if primary model is set
- Verify review model commands are valid

---

### Mode: Sync ("sync" argument)

Compare CLAUDE.md with .state/ and propose alignment.

#### Step 1: Parse CLAUDE.md

Extract structured knowledge from CLAUDE.md:

- **Common Mistakes table:** Each row is a potential Tier 1 knowledge entry
- **Architecture Anti-Patterns:** Each is a potential Tier 1 knowledge entry
- **Import Patterns:** Each is a pattern for PATTERNS.md
- **Deprecated Patterns:** Each is a Tier 1 knowledge entry
- **Key Commands:** Should be reflected in config.yaml tools section
- **Environment Variables:** Should be referenced in config.yaml or STACK.md
- **Code Style rules:** Should be in CONVENTIONS.md

#### Step 2: Parse .state/ Files

Read all existing state files and extract their current entries.

#### Step 3: Diff

For each item extracted from CLAUDE.md:

- Is it present in the appropriate state file?
- If present, is it consistent (same content, not contradictory)?
- If absent, should it be added?

#### Step 4: Propose Changes

Generate specific, actionable proposals:

```
SYNC PROPOSAL #1: Add Tier 1 Knowledge Entry
Target: .state/project/KNOWLEDGE.md
Action: Add entry [H00N]
Content: "Never use get_session() in tests -- connects to production"
Source: Human (CLAUDE.md) | Severity: Critical
Reason: This critical rule from CLAUDE.md has no corresponding knowledge entry.

SYNC PROPOSAL #2: Add Pattern
Target: .state/project/PATTERNS.md
Action: Add section "3-Phase DB Pattern"
Content: Read -> release connection -> process -> fresh connection -> write
Source: CLAUDE.md, DESIGN_PRINCIPLES.md Principle #4
Reason: Core architectural pattern referenced throughout CLAUDE.md not captured in PATTERNS.md.
```

## Applying Changes

### After Report: Applying Improvements

After presenting the report, ask the user which findings to address.

**For each approved finding:**

1. **Show the exact change** before applying it. Display a clear before/after or the content to be added.
2. **Apply the change** using Write or Edit tools.
3. **Confirm** the change was applied.

**When creating new state files from scratch:**

If `.state/project/` files do not exist, propose creating them seeded from CLAUDE.md content. This is the bootstrap path for new projects adopting the Dave Framework.

Bootstrap order:
1. `config.yaml` -- detect tools, set up model profiles
2. `KNOWLEDGE.md` -- seed Tier 1 entries from CLAUDE.md rules
3. `PATTERNS.md` -- extract patterns from CLAUDE.md architecture section
4. `CONCERNS.md` -- extract known issues from CLAUDE.md
5. `STACK.md` -- extract stack details from CLAUDE.md
6. `PROJECT.md` -- extract project overview from CLAUDE.md

**When updating existing state files:**

- Preserve existing content. Only add, modify, or remove what was explicitly approved.
- Maintain ID sequences (do not create gaps, do not duplicate IDs).
- Preserve formatting conventions of the existing file.
- Add timestamps to new entries.

**When promoting Tier 2 to Tier 1:**

1. Move the entry from "Tier 2" section to "Tier 1" section.
2. Change the ID from `[A0XX]` to `[H0XX]` (next available).
3. Update Source to include "Promoted from Tier 2 [A0XX]".
4. Set Severity based on Confidence level (HIGH -> Critical or High, MEDIUM -> Medium).
5. Remove Confidence and Verified count fields (Tier 1 entries do not need these).

## Cross-Referencing

### Source Files to Cross-Reference

When inspecting state health, always cross-reference against these authoritative sources:

| Source | What to Extract | Target State File |
|--------|----------------|-------------------|
| `CLAUDE.md` | Common Mistakes table | `KNOWLEDGE.md` (Tier 1) |
| `CLAUDE.md` | Architecture Anti-Patterns | `KNOWLEDGE.md` (Tier 1) |
| `CLAUDE.md` | Deprecated Patterns | `KNOWLEDGE.md` (Tier 1) |
| `CLAUDE.md` | Import Patterns | `PATTERNS.md` |
| `CLAUDE.md` | Code Style rules | `CONVENTIONS.md` |
| `CLAUDE.md` | Architecture section | `ARCHITECTURE.md`, `PATTERNS.md` |
| `CLAUDE.md` | Project Overview | `PROJECT.md` |
| `CLAUDE.md` | Environment Variables | `STACK.md`, `config.yaml` |
| `CLAUDE.md` | Key Commands | `config.yaml` |
| `.claude/rules/*.md` | All rules | `KNOWLEDGE.md` (Tier 1) |
| `docs/DESIGN_PRINCIPLES.md` | Design principles | `PATTERNS.md` |
| `docs/ISSUES.md` | Known issues | `CONCERNS.md` |
| `docs/COMMON_IMPLEMENTATION_ERRORS.md` | Error patterns | `KNOWLEDGE.md` (Tier 1) |

## Final Step: Verify Output Structure

Before returning your report, verify it matches `.claude/dave/templates/output/state-inspector-output.md`:
1. Executive summary is populated (3-line severity summary)
2. Findings are grouped by severity (critical → improvement → suggestion)
3. Every proposed fix is specific (exact content, target file)

## Success Criteria

- [ ] Correct mode identified from arguments (full / knowledge / config / sync)
- [ ] All relevant state files read (not assumed)
- [ ] All relevant source files cross-referenced (CLAUDE.md, .claude/rules/, docs/)
- [ ] Findings grouped by severity (critical / improvement / suggestion)
- [ ] Each finding has: what is wrong, why it matters, proposed fix, target file
- [ ] Tool availability verified by running actual commands (not trusting config)
- [ ] Report presented before any changes proposed
- [ ] Changes applied only after user approval
- [ ] ID sequences maintained correctly
- [ ] Timestamps included on new entries
