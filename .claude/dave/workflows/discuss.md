<purpose>
Conduct a structured discussion to remove ambiguity and establish guardrails for a development phase. Identify gray areas, let the user choose what to discuss, deep-dive each selected area, then capture everything downstream agents need.

You are a thinking partner, not an interviewer. The user is the visionary — you are the builder. Your job is to capture decisions that will guide research, planning, and implementation. You do not figure out implementation yourself.
</purpose>

<downstream_awareness>
**DISCUSSION.md feeds into:**

1. **Researcher (Phase 2)** — Reads DISCUSSION.md to know WHAT to research
   - "User wants event-driven architecture" -> researcher investigates event bus patterns in the codebase
   - "Must integrate with existing gateway pattern" -> researcher examines gateway implementations
   - Research Topics section is consumed directly by the researcher

2. **Planner (Phase 3)** — Reads DISCUSSION.md to know WHAT decisions are locked
   - "Async-only, no sync wrappers" -> planner designs all tasks as async
   - "Agent Discretion: error message format" -> planner can decide approach
   - Success Criteria become must-have truths in the plan

3. **TDD Developer (Phase 4)** — Reads DISCUSSION.md for constraints
   - "Hard Constraint: no new tables" -> developer works within existing schema
   - "Integration Point: must use BlobCleanupTracker" -> developer includes in implementation

**Your job:** Capture decisions clearly enough that downstream agents can act on them without asking the user again.

**Not your job:** Figure out HOW to implement. That is what research and planning do with the decisions you capture.
</downstream_awareness>

<philosophy>
**User = visionary. Claude = builder.**

The user knows:
- What the phase should accomplish
- What constraints and priorities exist
- What success looks like
- Specific behaviors, references, or preferences they have in mind

The user does not know (and should not be asked):
- Codebase internals (researcher reads the code)
- Technical risks (researcher identifies these)
- Implementation approach (planner figures this out)
- Task decomposition (planner handles this)

Ask about vision, priorities, constraints, and success criteria. Capture decisions for downstream agents.
</philosophy>

<scope_guardrail>
**CRITICAL: No scope creep.**

Discussion clarifies HOW to implement what is scoped, never WHETHER to add new capabilities. The phase boundary is established by the user's description or milestone roadmap and is FIXED.

**Allowed (clarifying ambiguity):**
- "How should extraction results be stored?" (storage decisions within scope)
- "What happens when the provider fails?" (error handling within scope)
- "Should this run per-org or in batch?" (execution model within scope)

**Not allowed (scope creep):**
- "Should we also add a dashboard?" (new capability)
- "What about supporting a new data source?" (new capability)
- "Maybe include export functionality?" (new capability)

**The heuristic:** Does this clarify how we implement what is already in the phase, or does it add a new capability that could be its own phase?

**When user suggests scope creep:**
```
"[Feature X] would be a new capability — that is its own phase.
Want me to note it for the backlog?

For now, let's focus on [phase domain]."
```

Capture the idea in "Deferred Ideas". Do not lose it, do not act on it.
</scope_guardrail>

<gray_area_identification>
Gray areas are **decisions the user cares about** — things that could go multiple ways and would change the result.

**How to identify gray areas:**

1. **Read the phase goal** from the user's description, context file, or milestone roadmap
2. **Understand the domain** — What kind of thing is being built?
   - Something that PROCESSES data -> input handling, output format, error recovery, idempotency matter
   - Something that INTEGRATES systems -> API contracts, data mapping, failure modes matter
   - Something that RESTRUCTURES code -> migration strategy, backward compatibility, rollout matter
   - Something users SEE -> visual presentation, interactions, states matter
   - Something that ORCHESTRATES work -> sequencing, parallelism, retry policy matter
   - Something that STORES data -> schema design, constraints, migration path matter
3. **Cross-reference with project knowledge** — Read KNOWLEDGE.md, PATTERNS.md, CONCERNS.md for project-specific gray areas (e.g., known issues that affect this phase)
4. **Generate phase-specific gray areas** — Not generic categories, but concrete decisions for THIS phase

**Examples by domain:**

```
Phase: "Add OCR provider fallback"
-> Fallback trigger conditions, Provider priority order, Result reconciliation, Partial failure handling

Phase: "Migrate service to gateway pattern"
-> Migration sequencing, Backward compatibility period, Test strategy for dual paths, Rollback approach

Phase: "Extract structured data from PDFs"
-> Schema for extracted fields, Confidence thresholds, Handling ambiguous pages, Output format
```

**The key question:** What decisions would change the outcome that the user should weigh in on?

**Claude handles these (do not ask):**
- Technical implementation details
- Architecture patterns (unless user has strong opinions)
- Performance optimization specifics
- Code organization
</gray_area_identification>

<required_reading>
Read all files referenced by the invoking prompt's execution_context before starting.
</required_reading>

<process>

<step name="load_context" priority="first">
## 1. Load Context

**MANDATORY FIRST STEP — Load project state and parse arguments.**

### 1a. Check Prerequisites

Verify `.state/` exists and has been initialized:

```bash
ls -d .state/project/KNOWLEDGE.md 2>/dev/null && echo "STATE_OK" || echo "STATE_MISSING"
```

**If STATE_MISSING:**
```
Dave Framework state not initialized.

Run /dave:init first to set up project state.
```
Exit workflow.

### 1b. Load Project Knowledge

Read these files to understand the project context. Absorb their content — do not summarize them to the user.

<!-- PARALLEL WORKTREE NOTE: These state files exist on the current branch.
     In a parallel worktree setup, each branch carries its own .state/ directory.
     The content here reflects THIS session's context, not a global state. -->

```
.state/project/KNOWLEDGE.md   — Rules and pitfalls (Tier 1 and Tier 2)
.state/project/PATTERNS.md    — Architecture patterns and conventions
.state/project/CONCERNS.md    — Known issues and tech debt
.state/project/STACK.md       — Tech stack and libraries
.state/STATE.md               — Current position and session state (per-branch)
```

Read each file that exists. Do not error on missing files — some may not exist yet.

### 1c. Parse Arguments

The user provides one of three input types. Determine which:

**Type A — Milestone/Phase Reference:**
If the argument looks like a phase number (e.g., "1", "phase 2", "P3") or milestone reference:
- Check `.state/milestones/` for existing milestones
- If milestone exists, read its `ROADMAP.md` to find the phase
- Set `phase_source = "milestone"`
- Set `phase_dir` to the milestone phase directory

**Type B — Context File Reference:**
If the argument starts with `@` or references a file path:
- Read the referenced file
- Extract the phase description from its content
- Set `phase_source = "context-file"`

**Type C — Text Description:**
If the argument is a plain text description:
- Use it directly as the phase description
- Set `phase_source = "adhoc"`

**If no argument provided:**

Use AskUserQuestion:
- header: "What should we discuss?"
- question: "Describe the phase or feature you want to discuss. What are you trying to build or change?"
- (free text response)

### 1d. Determine Output Path

Based on the phase source:

**If milestone phase:**
```
phase_dir = .state/milestones/{milestone-slug}/phases/{N}
```

**If ad-hoc (text description or context file):**
Generate a slug from the description (lowercase, hyphens, max 40 chars).
```
phase_dir = .state/milestones/adhoc/phases/{slug}
```

Store `phase_dir` for use in later steps.
</step>

<step name="check_existing">
## 2. Check for Existing Discussion

Check if DISCUSSION.md already exists at the target path:

```bash
ls ${phase_dir}/DISCUSSION.md 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If EXISTS:**

Use AskUserQuestion:
- header: "Existing Discussion"
- question: "This phase already has a DISCUSSION.md. What would you like to do?"
- options:
  - "Revise" — Review existing decisions and update
  - "View" — Show me what is there
  - "Restart" — Start fresh, replace existing
  - "Cancel" — Keep existing as-is

**If "Revise":** Read existing DISCUSSION.md, load its decisions as starting context, continue to analyze_scope. Mark any changed decisions with `(Revised: {date})`.
**If "View":** Display DISCUSSION.md content, then offer Revise/Cancel.
**If "Restart":** Continue to analyze_scope with no prior context.
**If "Cancel":** Exit workflow.

**If MISSING:** Continue to analyze_scope.
</step>

<step name="analyze_scope">
## 3. Analyze Scope

Analyze the phase description to identify its domain boundary and gray areas.

### 3a. Determine Domain Boundary

From the phase description (and milestone roadmap if applicable), determine:

1. **What this phase delivers** — One clear sentence describing the capability or change.
2. **What kind of work this is** — Processing, integration, restructuring, UI, orchestration, data modeling, etc.
3. **What project context is relevant** — Cross-reference with KNOWLEDGE.md, PATTERNS.md, CONCERNS.md for items that affect this phase.

### 3b. Identify Gray Areas

Generate 3-6 phase-specific gray areas. Each gray area is a concrete decision point, not a generic category.

**Process:**
- Read the phase description carefully
- Consider the "kind of work" to identify domain-appropriate gray areas
- Check project KNOWLEDGE.md for relevant rules or pitfalls that create decision points
- Check CONCERNS.md for known issues that intersect with this phase

**Each gray area must have:**
- A specific label (not generic like "Architecture" or "Testing")
- 1-2 concrete questions it covers
- Why it matters for this phase

**Quality check — reject gray areas that are:**
- Generic enough to apply to any phase ("Error handling", "Testing strategy")
- Implementation details Claude should decide ("Which design pattern to use")
- Already answered by project KNOWLEDGE.md or PATTERNS.md

### 3c. Assess Discussion Necessity

If after analysis you find fewer than 2 meaningful gray areas, the phase may not need discussion:

Use AskUserQuestion:
- header: "Low Ambiguity Phase"
- question: "This phase seems straightforward with few decision points. Want to discuss anyway, or skip to research?"
- options:
  - "Discuss anyway" — I have things to clarify
  - "Skip to research" — Proceed with defaults

**If "Skip to research":** Write a minimal DISCUSSION.md with scope and defaults, commit, exit.
**If "Discuss anyway":** Continue to present_gray_areas.
</step>

<step name="present_gray_areas">
## 4. Present Gray Areas

Present the domain boundary and gray areas to the user for selection.

### 4a. State the Boundary

```
Phase: {phase name or description}
Domain: {what this phase delivers — one sentence}

We will clarify HOW to implement this.
New capabilities belong in separate phases.
```

If project knowledge has relevant context, mention it briefly:
```
Project context: {1-2 relevant items from KNOWLEDGE.md or CONCERNS.md}
```

### 4b. Present Selectable Gray Areas

Use AskUserQuestion (multiSelect: true):
- header: "Discussion Areas"
- question: "Which areas do you want to discuss?"
- options: 3-6 phase-specific gray areas, each formatted as:
  - "[Specific area label]" — concrete questions this covers

**Do NOT include a "skip" or "none" option.** The user ran this command to discuss — give them real choices.

**Do NOT include a "select all" option.** Let the user pick what matters to them.

**Examples by domain type:**

For a data processing phase:
```
- "Input validation" — What input states to handle? Reject vs repair vs skip?
- "Output schema" — What fields to extract? Required vs optional? Confidence scores?
- "Failure recovery" — Retry on transient errors? Skip and log? Require manual review?
- "Idempotency" — Re-running on same input: overwrite, skip, or version?
```

For a refactoring phase:
```
- "Migration sequencing" — Big bang or incremental? Which components first?
- "Backward compatibility" — How long do old paths stay active? Deprecation warnings?
- "Test coverage" — Existing tests: update in place or write new? Coverage threshold?
- "Rollback approach" — If migration fails midway, what is the recovery plan?
```

For an integration phase:
```
- "Data mapping" — How do external fields map to our schema? Handling mismatches?
- "Auth and access" — API keys, rate limits, credential management?
- "Sync strategy" — Real-time, batch, or event-driven? Frequency?
- "Conflict resolution" — When external data contradicts local data, who wins?
```

Continue to discuss_areas with selected areas.
</step>

<step name="discuss_areas">
## 5. Discuss Selected Areas

For each selected area, conduct a focused discussion loop.

### Philosophy: 2-4 questions, then check.

Ask 2-4 questions per round before offering to continue or move on. Each answer often reveals the next question. Adapt follow-up questions based on what the user says — do not ask pre-scripted questions.

### For Each Selected Area:

**1. Announce the area:**
```
Let's talk about {Area}.
```

**2. Ask 2-4 questions using AskUserQuestion:**

Each question should:
- Have a specific header matching the area
- Offer 2-4 concrete choices (AskUserQuestion adds "Other" automatically)
- Include "You decide" as an option when reasonable — this captures agent discretion
- Be informed by the user's previous answers in this discussion

**Question design principles:**
- Options should be concrete, not abstract ("Overwrite previous results" not "Option A")
- Each answer should inform the next question
- If user picks "Other", receive their input, reflect it back, confirm understanding
- Reference project patterns when relevant ("PATTERNS.md says we use X — does that apply here?")

**3. After 2-4 questions, check continuation:**

Use AskUserQuestion:
- header: "{Area}"
- question: "Anything else about {area}, or move to the next topic?"
- options:
  - "More questions" — I have more to clarify
  - "Next topic" — This is clear enough

**If "More questions":** Ask 2-4 more, then check again.
**If "Next topic":** Proceed to next selected area.

**4. Scope creep handling:**

If the user mentions something outside the phase domain:
```
"{Feature} sounds like a new capability — that belongs in its own phase.
I will note it as a deferred idea.

Back to {current area}: {return to the current question}"
```

Track deferred ideas internally. Capture them with enough context to be useful later.

### After All Areas Complete:

Use AskUserQuestion:
- header: "Discussion Complete"
- question: "That covers {list area names}. Anything else before I capture this?"
- options:
  - "Capture decisions" — Write DISCUSSION.md
  - "Revisit an area" — Go back to a topic
  - "Add a new topic" — Something we missed

**If "Revisit an area":** Ask which area, return to that area's discussion loop.
**If "Add a new topic":** Ask for the topic, conduct a focused discussion, then return to this checkpoint.
**If "Capture decisions":** Continue to identify_research_topics.
</step>

<step name="identify_research_topics">
## 6. Identify Research Topics

Based on the decisions made during discussion, identify topics that need deep research before planning.

**Research topics are questions that cannot be answered by discussion alone.** They require:
- Reading codebase to understand existing patterns
- Investigating library capabilities or limitations
- Analyzing existing data/schema to understand constraints
- Exploring approaches that need technical evaluation

### For Each Research Topic, Capture:

```markdown
### {Topic Title}

**Question:** What specifically needs to be researched?
**Why it matters:** How this affects the phase outcome.
**Known context:** What the user already decided or knows about this.
**Suggested approach:** Where to start looking (codebase paths, docs, external resources).
```

### Research Topic Identification Rules:

**Good research topics:**
- "How does the existing gateway pattern handle retry for this provider?" (codebase investigation)
- "What schema constraints exist on the target table?" (data investigation)
- "Does library X support the batch processing mode we decided on?" (capability investigation)

**Not research topics (these are already decided):**
- Anything the user answered definitively in discussion
- Anything covered by project KNOWLEDGE.md or PATTERNS.md
- Pure implementation questions (planner handles these)

### Present Research Topics to User:

Display the identified research topics briefly:
```
Based on our discussion, here are topics for deep research:

1. {Topic 1} — {one-line summary}
2. {Topic 2} — {one-line summary}
3. {Topic 3} — {one-line summary}

These will be investigated in /dave:research before planning begins.
```

Use AskUserQuestion:
- header: "Research Topics"
- question: "Do these research topics look right?"
- options:
  - "Looks good" — Proceed to write DISCUSSION.md
  - "Add a topic" — I want something else researched
  - "Remove a topic" — One of these is not needed
  - "Adjust a topic" — Modify the scope of a topic

Handle adjustments, then continue to write_discussion.
</step>

<step name="write_discussion">
## 7. Write DISCUSSION.md

Create the phase directory if it does not exist, then write DISCUSSION.md.

### 7a. Create Directory

```bash
mkdir -p ${phase_dir}
```

### 7b. Determine Success Criteria

Synthesize success criteria from the discussion. These are observable conditions that downstream agents treat as must-have truths. They answer: "How do we know this phase is done?"

**Good success criteria:**
- "All organization PDFs are processed through the new extraction pipeline"
- "Gateway handles provider failover without data loss"
- "Existing tests pass without modification"

**Bad success criteria (too vague):**
- "Code is clean"
- "System works well"
- "Performance is good"

### 7c. Write the File

Write to `${phase_dir}/DISCUSSION.md`:

```markdown
# Phase Discussion: {phase name}

**Gathered:** {today's date}
**Status:** Ready for research
**Source:** {milestone phase N | context file path | ad-hoc description}

## Scope

### In Scope
{Bullet list of what this phase delivers. Be specific.}

### Out of Scope
{Bullet list of what is explicitly excluded, with reasoning for each.}
- {Item} — {Why it is out of scope}

### Deferred
{Ideas that came up during discussion but belong elsewhere.}
- {Idea} — Deferred to: {destination or "backlog"}

## Decisions

### {Category 1 — matches a discussed area}
- {Decision or preference captured}
- {Another decision if applicable}

### {Category 2 — matches a discussed area}
- {Decision or preference captured}

### {Additional categories as needed}

### Agent Discretion
{Areas where the user said "you decide" — note that downstream agents have flexibility here.
Be specific about WHAT is discretionary so agents know their freedom.}
- {Area}: {What Claude can decide}

## Constraints

### Hard Constraints
{Non-negotiable requirements. Things that MUST be true.}
- {Constraint with reasoning}

### Soft Constraints
{Preferences that can bend if necessary. Note the threshold for bending.}
- {Preference} — Can bend if: {condition}

### Integration Points
{Other systems, services, or components this phase touches.}
- {System/component}: {How it connects, what contract exists}

## Success Criteria
{Observable criteria that become must-have truths in the plan.
Each criterion should be verifiable — an agent can check whether it is met.}
- [ ] {Criterion 1}
- [ ] {Criterion 2}
- [ ] {Criterion 3}

## Research Topics

{Topics for Phase 2 research. Each topic has a clear question, why it matters,
and what context the user already provided.}

### {Research Topic 1}
**Question:** {What specifically needs to be researched}
**Why it matters:** {How this affects the phase outcome}
**Known context:** {What was decided or is already known}
**Suggested approach:** {Where to start looking}

### {Research Topic 2}
**Question:** {What specifically needs to be researched}
**Why it matters:** {How this affects the phase outcome}
**Known context:** {What was decided or is already known}
**Suggested approach:** {Where to start looking}

## Open Questions
{Unresolved items that need human input later. These are NOT research topics —
they are decisions that could not be made during discussion and need more context.}

- {Question}: {Context, impact, when it should be resolved}

---

*Phase discussion gathered: {today's date}*
*Source: {phase source description}*
```

**Writing rules:**
- Decisions must reflect ACTUAL answers from the discussion, not assumed defaults
- Agent Discretion must list specific items the user delegated, not a blanket statement
- Research Topics must have enough context that a researcher can start without re-asking
- Success Criteria must be observable and verifiable
- If no items exist for a section, write "None" rather than omitting the section
</step>

<step name="update_state">
## 8. Update State

<!-- PARALLEL WORKTREE NOTE: STATE.md is per-branch, not global. Each worktree
     has its own branch with its own STATE.md. When this workflow updates STATE.md,
     it updates the copy on THIS branch only. Other worktrees on other branches
     are unaffected. See parallel-usage.md for details. -->

Update `.state/STATE.md` to reflect the current position:

Read the existing STATE.md, then update:
- **Current focus** to the phase being discussed
- **Status** to "Discussion complete — ready for research"
- **Last activity** to "{today's date} — Phase discussion captured via /dave:discuss"

If the phase is within a milestone, also update the milestone reference.

Write the updated STATE.md.
</step>

<step name="git_commit">
## 9. Commit

Stage and commit the DISCUSSION.md file and updated STATE.md.

```bash
git add ${phase_dir}/DISCUSSION.md .state/STATE.md
git commit -m "docs(discuss): capture phase discussion for {phase name}"
```

If the commit fails, report the error but do not block the workflow. The file is written regardless.
</step>

<step name="present_summary">
## 10. Present Summary

Display a structured summary:

```
Created: ${phase_dir}/DISCUSSION.md

## Decisions Captured

### {Category}
- {Key decision}

### {Category}
- {Key decision}

{If agent discretion items exist:}
### Agent Discretion
- {What Claude can decide}

{If deferred ideas exist:}
## Noted for Later
- {Deferred idea} — {destination}

## Research Topics Identified

1. {Topic 1} — {one-line summary}
2. {Topic 2} — {one-line summary}

## Success Criteria

- {Criterion 1}
- {Criterion 2}

---

Next step: /dave:research
Research will investigate the {N} topics identified above.

/clear first for a fresh context window.

Also available:
- Review/edit DISCUSSION.md before continuing
- /dave:state to check overall project state
```
</step>

</process>

<success_criteria>
- [ ] Project state loaded (KNOWLEDGE.md, PATTERNS.md, CONCERNS.md read)
- [ ] Phase description parsed from argument (milestone, context file, or ad-hoc)
- [ ] Gray areas identified through intelligent analysis specific to THIS phase
- [ ] User selected which areas to discuss
- [ ] Each selected area explored with concrete questions until user satisfied
- [ ] Scope creep redirected to deferred ideas (not lost)
- [ ] Research topics identified with enough context for the researcher
- [ ] Success criteria are observable and verifiable
- [ ] DISCUSSION.md captures actual decisions, not vague vision
- [ ] STATE.md updated with current position
- [ ] User knows next steps (/dave:research)
</success_criteria>
