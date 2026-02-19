<purpose>
Display the complete Dave Framework command reference. Output ONLY the reference content. Do NOT add project-specific analysis, git status, next-step suggestions, or any commentary beyond the reference.
</purpose>

<reference>
# Dave Framework Command Reference

**Dave Framework** is a multi-agent development workflow combining deep planning, parallel research, strict TDD, multi-model review, tool-agnostic verification, and a learning loop that improves with every feature.

## Quick Start

1. `/dave:init` — Initialize project state from CLAUDE.md
2. Start a milestone with discussion, research, planning
3. Execute phases with TDD, review, and verification

## When to Use Quick vs Full Workflow

| Use `/dave:quick` when... | Use the full workflow when... |
|--------------------------|------------------------------|
| Task touches 1-3 files | Task touches 4+ files |
| No schema/migration changes | New tables, columns, or migrations |
| No new services or gateways | New service, gateway, or major component |
| No architectural decisions needed | Design choices between multiple approaches |
| Well-understood, small scope | Unclear scope or requirements |
| Bug fix, small feature, refactor | New feature, pipeline stage, integration |

**Quick mode flags:**
- `--skip-review` — Skip code review (for trivial changes)
- `--no-commit` — Run TDD/review/verify without committing (exploration mode)
- `--skip-review --no-commit` — Minimal mode: just implement and verify

When in doubt, start with the full workflow. You can always skip phases if they're unnecessary.

## Architecture

```
.agent/          — The system (portable, project-agnostic)
.state/          — Project state (knowledge, config, milestones, codebase)
```

## Available Commands

### Project Initialization

**`/dave:init`**
Initialize the Dave Framework for this project.

- Detects available tools (Docker, database, Chrome MCP, external review models)
- Extracts Tier 1 knowledge from CLAUDE.md (rules, patterns, pitfalls)
- Seeds PATTERNS.md, STACK.md, CONCERNS.md from project docs
- Creates config.yaml with detected tools and model profiles
- Creates STATE.md for session continuity

Creates the full `.state/` directory:
- `project/` — config, knowledge, patterns, stack, concerns
- `codebase/` — code analysis (populated by codebase mapping)
- `milestones/` — lifecycle state (populated when milestones begin)
- `debug/` — debug sessions (persistent across context resets)

Usage: `/dave:init`

### State Inspection

**`/dave:state [focus]`**
Inspect and improve project state health.

Three modes:
- No args: Full health check (structure, knowledge, config, freshness)
- `knowledge`: Deep analysis of KNOWLEDGE.md (tiers, promotions, gaps)
- `config`: Tool detection and config drift analysis
- `sync`: Compare CLAUDE.md with state files, identify missing extractions

Usage:
```
/dave:state              # Full health check
/dave:state knowledge    # Knowledge system analysis
/dave:state config       # Configuration and tool status
/dave:state sync         # CLAUDE.md sync check
```

### Help

**`/dave:help`**
Show this command reference.

Usage: `/dave:help`

### Quick Mode

**`/dave:quick "task description" [--skip-review] [--no-commit]`**
Execute a small, well-understood task end-to-end: inline plan, TDD, review, verification. Compresses the front half of the Dave workflow (skip discuss/research/plan) while preserving quality.

- Generates inline plan (QUICK_PLAN.md) from task description
- Launches TDD executor(s) with strict RED/GREEN/REFACTOR
- Runs practical verification per task
- Runs streamlined review (internal + external reviewers)
- Runs Layer 1 + Layer 3 verification
- Commits atomically per task
- If complexity exceeds quick scope (4+ files, new table, architectural decisions), warns and offers to switch to full workflow

Flags:
- `--skip-review` — Skip code review (for trivial changes)
- `--no-commit` — Run TDD/review/verify without committing (exploration mode)

Combine flags for minimal mode:
```bash
/dave:quick "fix typo in README" --skip-review --no-commit  # Implement + verify only
```

Usage:
```
/dave:quick "add validation to email field"
/dave:quick "fix off-by-one in pagination" --skip-review
/dave:quick "explore new retry logic" --no-commit
/dave:quick "fix typo in README" --skip-review --no-commit
```

### Discussion

**`/dave:discuss [phase-description or @context-file]`**
Structured discussion to establish phase guardrails, scope, and research topics.

- Identifies gray areas specific to the phase
- Lets you choose which areas to discuss
- Captures decisions for downstream agents (researcher, planner, TDD)
- Identifies research topics for Phase 2
- Defines success criteria that become plan must-haves

Usage:
```
/dave:discuss "Add OCR provider selection"
/dave:discuss @feature-idea.md
```

### Research

**`/dave:research [--skip-arch] [--topics topic1,topic2]`**
Parallel research orchestration with specialized agents.

- Launches architecture/design research agent + topic research agents in parallel
- Each topic researched with expert lens (official docs, GitHub issues, codebase patterns)
- Identifies strengths AND weaknesses of each option
- Can loop back to discussion if new questions emerge
- Writes RESEARCH.md with confidence-tagged findings

Flags:
- `--skip-arch` — Skip architecture/design research (for phases with clear direction)
- `--topics` — Focus on specific topics from DISCUSSION.md

Usage:
```
/dave:research
/dave:research --skip-arch
/dave:research --topics "rate-limiting,schema-design"
```

### Planning

**`/dave:plan [--check-only]`**
Goal-backward planning from research findings to execution plan.

- Derives must-haves (truths, artifacts, key links) from success criteria
- Breaks work into tasks organized in dependency waves
- Specifies test scenarios per task (consumed by TDD developer)
- Designs verification matrix with four layers (plan conformance, code review, automated functional, human oversight)
- Runs plan checker agent to validate before user approval
- Defines deviation rules for executor autonomy

Flags:
- `--check-only` — Run the plan checker on an existing PLAN.md without creating a new plan

Usage:
```
/dave:plan
/dave:plan --check-only
```

### Execution

**`/dave:execute [--wave N] [--task N.M] [--dry-run]`**
TDD implementation with wave-based parallelism and atomic commits.

- Executes tasks from the approved plan using strict TDD (RED → GREEN → REFACTOR)
- Each task gets its own tdd-developer agent
- Tasks within a wave run in parallel, waves run sequentially
- Practical-verifier validates each task before committing
- Handles deviations per plan deviation rules (auto-fix vs stop-and-ask)
- Commits atomically per task (one commit = one task + tests)
- Tracks execution state for session continuity

Flags:
- `--wave N` — Start from wave N (skip earlier waves)
- `--task N.M` — Execute only a specific task
- `--dry-run` — Show execution plan without running

Usage:
```
/dave:execute
/dave:execute --wave 2
/dave:execute --task 1.2
/dave:execute --dry-run
```

### Review

**`/dave:review [--fix-only] [--skip-external]`**
Multi-agent code review with intelligent aggregation and fix loop.

- Launches parallel reviews (internal agents + external models from config.yaml)
- Review aggregator triages findings using project context (KNOWLEDGE.md, PATTERNS.md)
- Classifies findings: fix now (bugs, security) / defer (create issues) / dismiss (false positives)
- Ambiguous findings go to OPEN_QUESTIONS.md for human decision
- Fix loop: "fix now" items loop back to scoped TDD, then scoped re-review until converged
- Consensus boost: 3+ reviewers flagging same issue = high confidence

Flags:
- `--fix-only` — Re-review only files changed by fix loop (scoped, faster)
- `--skip-external` — Skip external model reviews (faster, lower cost)

Usage:
```
/dave:review
/dave:review --skip-external
/dave:review --fix-only
```

### Verification

**`/dave:verify [--layer N] [--gaps-only]`**
Multi-layer verification from the plan's verification matrix.

Four layers, each catching different problem classes:
- **Layer 1: Plan Conformance** — Are must-haves achieved? Truths verified, artifacts substantive and wired, no anti-patterns
- **Layer 2: Code Review** — Gate check (confirms review complete, no fix-now items)
- **Layer 3: Automated Functional** — Runs verification steps using available tools (bash, browser, database, API)
- **Layer 4: Human Oversight** — Presents checkpoints with prepared evidence for human judgment

Gap closure: failed items get focused fixes (TDD), then re-verify only the gaps.

Flags:
- `--layer N` — Run only a specific layer (1-4)
- `--gaps-only` — Re-verify only previously failed items

Usage:
```
/dave:verify
/dave:verify --layer 3
/dave:verify --gaps-only
```

### Push

**`/dave:push [--wait] [--draft] [--no-pr]`**
Push the feature branch and create a pull request with a structured description derived from phase artifacts.

- Generates PR title and body from PLAN.md, REVIEWS.md, VERIFICATION.md
- PR body includes: must-haves summary, change stats, code quality metrics, test plan
- Deferred review items listed as follow-up work
- Optionally monitors CI checks until completion
- Handles existing PRs (update or skip)

Flags:
- `--wait` — Poll CI checks after PR creation until they complete (up to 10 min)
- `--draft` — Create the PR as a draft
- `--no-pr` — Push branch only, skip PR creation

Usage:
```
/dave:push
/dave:push --draft
/dave:push --wait
/dave:push --no-pr
```

### Reflect

**`/dave:reflect [--promote-only] [--summary-only]`**
Learning loop — extract knowledge from completed phase, update patterns, create summary.

- Launches learning extractor agent to analyze plan vs actual, review findings, verification results
- Produces SUMMARY.md with metrics, deviations, key decisions, lessons learned
- Extracts new Tier 2 knowledge entries with evidence from phase artifacts
- Updates verification counts for existing Tier 2 entries independently confirmed
- Proposes Tier 2 → Tier 1 promotions when entries exceed verification threshold
- Updates PATTERNS.md and CONCERNS.md with new discoveries
- At milestone boundaries, aggregates knowledge upward (phase → milestone → project)

Flags:
- `--promote-only` — Skip extraction, only present pending promotion candidates
- `--summary-only` — Only generate SUMMARY.md, skip knowledge extraction

Usage:
```
/dave:reflect
/dave:reflect --summary-only
/dave:reflect --promote-only
```

### Progress

**`/dave:progress [--verbose]`**
Check milestone progress and route to the next action.

- Reads STATE.md and phase artifacts to show current position
- Determines next command based on which artifacts exist
- Shows phase-by-phase progress for the active milestone
- Handles mid-execution and mid-review states (suggests resume flags)

Flags:
- `--verbose` — Include detailed artifact metrics, velocity stats, and knowledge summary

Usage:
```
/dave:progress
/dave:progress --verbose
```

## Knowledge System

Dave uses a tiered knowledge system with explicit provenance:

**Tier 1 (Human-Provided)** — Absolute authority. From CLAUDE.md, human corrections, review decisions. Agents MUST follow. Only humans can modify.

**Tier 2 (Agent-Discovered)** — Standard authority. From reflect findings, verification failures, implementation patterns. Agents should follow but can flag conflicts. Can be promoted to Tier 1 after human confirmation.

Knowledge flows upward through generalization:
```
Phase (specific) → Milestone (aggregated) → Project (generalized)
```

## Configuration

All tool and model configuration lives in `.state/project/config.yaml`.

Key sections:
- `models` — Primary model and profiles (quality/balanced/budget)
- `review_models` — External review models (codex, opencode, etc.)
- `tools` — Build commands (test, lint, run)
- `verification` — Available verification tools and capabilities
- `knowledge` — Tier 2 promotion threshold, auto-apply settings

## Files & Structure

```
.state/
├── STATE.md                          # Current position, session continuity
├── project/
│   ├── config.yaml                   # Tools, models, verification capabilities
│   ├── KNOWLEDGE.md                  # Pitfalls & rules with provenance tiers
│   ├── PATTERNS.md                   # Architecture patterns, conventions
│   ├── STACK.md                      # Tech stack, libraries, versions
│   └── CONCERNS.md                   # Known issues, tech debt, watch list
├── codebase/
│   ├── STRUCTURE.md                  # Directory layout, naming patterns
│   ├── ARCHITECTURE.md               # Layers, data flow, key abstractions
│   └── CONVENTIONS.md                # Code style, imports, type hints
├── milestones/
│   └── {milestone-slug}/
│       ├── ROADMAP.md                # Phase breakdown
│       ├── RESEARCH.md               # Milestone-level research
│       ├── KNOWLEDGE.md              # Milestone-level decisions
│       └── phases/
│           └── {N}/
│               ├── DISCUSSION.md     # Scope, guardrails, decisions
│               ├── RESEARCH.md       # Phase-level research
│               ├── PLAN.md           # Execution plan + verification matrix
│               ├── KNOWLEDGE.md      # Phase-level decisions & mistakes
│               ├── REVIEWS.md        # Aggregated review findings
│               ├── OPEN_QUESTIONS.md # Ambiguous items for human review
│               ├── VERIFICATION.md   # Multi-layer verification results
│               └── SUMMARY.md        # Post-completion summary
└── debug/
    ├── {slug}.md                     # Active debug sessions
    └── resolved/                     # Archived resolved issues
```

## Autonomous Mode (Planned)

**`/dave:auto`** *(not yet implemented)*
Chains all phases autonomously after discussion.

- Runs: research → plan → execute → review → verify → push
- Pauses only for: post-research questions, Rule 4 deviations, open questions (after fixes), human verification checkpoints
- Uses Ralph Loop integration for session continuation across gates
- Manual commands still work independently when you want control

## Design Principles

1. **Multi-agent parallelism** — Specialized agents run concurrently where possible
2. **Autonomous within guardrails** — Discussion sets boundaries, agents operate freely within them
3. **Verification is first-class** — Every plan defines HOW to verify
4. **Knowledge has provenance** — Human rules > agent-discovered patterns
5. **Learning accumulates** — Each phase feeds the milestone, each milestone feeds the project
6. **Tool-agnostic** — config.yaml declares capabilities, agents adapt
7. **Portable** — `.agent/` works across projects, `.state/` is project-specific

## Getting Help

- Read `.agent/README.md` for the full framework specification
- Read `.state/project/KNOWLEDGE.md` for project-specific rules
- Run `/dave:state` for a health check of your project state
- Run `/dave:state sync` to check if state files are current
</reference>
