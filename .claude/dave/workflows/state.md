<purpose>
Inspect the Dave Framework `.state/` directory health. Reports on knowledge coverage, tool availability, configuration status, and sync between CLAUDE.md and state files. Identifies gaps and suggests improvements. This is the diagnostic tool for the framework's project state.

<!-- PARALLEL WORKTREE NOTE: This inspects the .state/ directory on the CURRENT BRANCH.
     In a parallel worktree setup, each branch may have different state reflecting
     different phases of work. The health check applies to this branch's state only. -->
</purpose>

<required_reading>
Read all files referenced by the invoking prompt's execution_context before starting.
</required_reading>

<process>

## 1. Parse Mode

Parse $ARGUMENTS for the optional focus mode:
- Empty or no args → **full** health check
- `knowledge` → focus on knowledge system
- `config` → focus on configuration and tools
- `sync` → compare CLAUDE.md with state files

If $ARGUMENTS contains an unrecognized value, show:
```
Unknown mode: "{value}"

Available modes:
  /dave:state              — Full health check
  /dave:state knowledge    — Knowledge system analysis
  /dave:state config       — Configuration and tools
  /dave:state sync         — CLAUDE.md sync check
```
Exit.

## 2. Verify State Exists

```bash
ls -d .state/ 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If `.state/` does not exist:**
```
No project state found.

Run /dave:init to initialize the Dave Framework for this project.
```
Exit.

## 3. Load State Files

Read all state files that exist. Track which are missing:

```bash
# Check existence of each expected file
for f in .state/STATE.md .state/project/config.yaml .state/project/KNOWLEDGE.md .state/project/PATTERNS.md .state/project/STACK.md .state/project/CONCERNS.md; do
  if [ -f "$f" ]; then echo "OK: $f"; else echo "MISSING: $f"; fi
done
```

Read each existing file for analysis.

## 4. Route by Mode

### Mode: Full Health Check (default)

Run all three sub-analyses and present a combined report.

**4a. Structure Check**

Verify all expected directories and files exist:

| Path | Status | Notes |
|------|--------|-------|
| `.state/project/` | {OK/MISSING} | |
| `.state/project/config.yaml` | {OK/MISSING} | |
| `.state/project/KNOWLEDGE.md` | {OK/MISSING} | |
| `.state/project/PATTERNS.md` | {OK/MISSING} | |
| `.state/project/STACK.md` | {OK/MISSING} | |
| `.state/project/CONCERNS.md` | {OK/MISSING} | |
| `.state/codebase/` | {OK/MISSING} | |
| `.state/milestones/` | {OK/MISSING} | |
| `.state/debug/` | {OK/MISSING} | |
| `.state/STATE.md` | {OK/MISSING} | |

**4b. Knowledge Analysis**

Read `.state/project/KNOWLEDGE.md`. Count:
- Total Tier 1 entries
- Total Tier 2 entries
- Severity distribution (Critical / High / Medium)
- Tier 2 entries eligible for promotion (verified >= threshold from config.yaml)

**4c. Config Analysis**

Read `.state/project/config.yaml`. For each tool marked `available: true`, verify it is still available:

```bash
# Re-run detection for each tool
docker --version 2>/dev/null && echo "DOCKER_OK" || echo "DOCKER_GONE"
# ... etc for each tool
```

Report any drift between config and reality.

**4d. State Freshness**

Read `.state/STATE.md`. Check:
- Last activity date — how stale is it?
- Current position — is it coherent?
- Session continuity — is there an orphaned resume file?

**4e. Present Full Report**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DAVE FRAMEWORK ► STATE HEALTH CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Structure
  {table from 4a — show only issues, or "All files present" if clean}

## Knowledge ({N} Tier 1, {M} Tier 2)
  Critical: {count}  |  High: {count}  |  Medium: {count}
  Promotion candidates: {count} Tier 2 entries ready for review
  {any gaps or suggestions}

## Configuration
  Tools: {count available} / {count total}
  Drift: {any tools that changed availability since last check}
  Review models: {status}

## State Freshness
  Last activity: {date} ({N days ago})
  Position: {current position from STATE.md}
  {warnings if stale or orphaned resume files}

## Suggestions
  {list of actionable improvements, e.g.:
   - "3 Tier 2 entries are eligible for promotion — review with /dave:state knowledge"
   - "config.yaml shows Docker available but docker is not installed"
   - "CONCERNS.md is empty — consider documenting known tech debt"
  }
```

---

### Mode: Knowledge

Focus exclusively on the knowledge system.

**Read:**
- `.state/project/KNOWLEDGE.md`
- `.state/project/config.yaml` (for promotion threshold)

**Analyze:**
1. Count entries by tier and severity
2. Check for duplicate or overlapping entries
3. Identify Tier 2 entries eligible for Tier 1 promotion
4. Check entry format consistency (all have Source, Added, Severity/Confidence)
5. Look for entries without actionable guidance (too vague to be useful)

**Present:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DAVE FRAMEWORK ► KNOWLEDGE SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Tier 1 (Human-Provided) — {N} entries
  Critical: {list of entry IDs and one-line summaries}
  High: {list}
  Medium: {list}

## Tier 2 (Agent-Discovered) — {M} entries
  {list with confidence and verification count}

## Promotion Candidates
  {Tier 2 entries where verified >= threshold}
  {For each: ID, summary, verification count, recommendation}

## Quality Issues
  {Duplicates, missing fields, vague entries}

## Coverage Gaps
  {Areas of the codebase or workflow not covered by any knowledge entry}
```

---

### Mode: Config

Focus exclusively on configuration and tool availability.

**Read:**
- `.state/project/config.yaml`

**Detect current state of each tool** (re-run detection commands from init).

**Present:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DAVE FRAMEWORK ► CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Models
  Primary: {model}
  Active profile: {profile name}

## Build Tools
  Test: {command} — {available/missing}
  Lint: {command} — {available/missing}
  Run: {command} — {available/missing}

## Verification Tools
  {For each tool: name, type, available status, capabilities}
  {Highlight any drift from config}

## External Review Models
  {For each: name, available status, command}

## Suggestions
  {e.g., "playwright is not installed — install with: npm install -g @anthropic/mcp-playwright"
   or "All tools available and matching config"}
```

Offer to update config.yaml if drift is detected:

Use AskUserQuestion:
- header: "Config Drift Detected"
- question: "Tool availability has changed since last check. Update config.yaml?"
- options:
  - "Update" — Fix config.yaml to match current state
  - "Skip" — Leave config.yaml as-is

---

### Mode: Sync

Compare CLAUDE.md with state files to find drift.

**Read:**
- `CLAUDE.md`
- `.state/project/KNOWLEDGE.md`
- `.state/project/PATTERNS.md`
- `.state/project/STACK.md`
- `.state/project/CONCERNS.md`

**Analyze:**

1. **Knowledge sync:** Parse CLAUDE.md "Common Mistakes" table. For each row, check if a corresponding Tier 1 entry exists in KNOWLEDGE.md. Report:
   - Entries in CLAUDE.md not in KNOWLEDGE.md (missing extractions)
   - Entries in KNOWLEDGE.md not traceable to CLAUDE.md (orphaned or from other sources)

2. **Pattern sync:** Compare CLAUDE.md "Architecture" section with PATTERNS.md. Identify:
   - New patterns in CLAUDE.md not in PATTERNS.md
   - Patterns in PATTERNS.md not in CLAUDE.md (agent-discovered or evolved)

3. **Stack sync:** Compare CLAUDE.md tech references with STACK.md. Identify:
   - New technologies mentioned in CLAUDE.md not in STACK.md
   - Technologies in STACK.md not referenced in CLAUDE.md

**Present:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DAVE FRAMEWORK ► SYNC CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## CLAUDE.md → KNOWLEDGE.md
  Rules in CLAUDE.md: {N}
  Matched in KNOWLEDGE.md: {M}
  Missing: {N-M}
  {list of missing rules}

## CLAUDE.md → PATTERNS.md
  Patterns in CLAUDE.md: {N}
  Matched in PATTERNS.md: {M}
  New in CLAUDE.md: {list}
  Agent-added in PATTERNS.md: {list}

## CLAUDE.md → STACK.md
  Technologies in CLAUDE.md: {N}
  Matched in STACK.md: {M}
  Missing from STACK.md: {list}

## Recommendation
  {If drift detected: "Run /dave:init --update to re-extract from CLAUDE.md"
   If clean: "State files are in sync with CLAUDE.md"}
```

</process>

<success_criteria>
- [ ] Mode correctly parsed from arguments
- [ ] Missing `.state/` handled gracefully with init suggestion
- [ ] All state files read and analyzed
- [ ] Report is structured and actionable
- [ ] Tool detection re-runs to catch drift
- [ ] Sync mode identifies gaps between CLAUDE.md and state files
- [ ] Suggestions are specific and actionable (not generic)
</success_criteria>
