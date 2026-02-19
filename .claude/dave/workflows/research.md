<purpose>
Orchestrate parallel research across all topics identified during discussion. The orchestrator analyzes the phase scope, identifies what domain experts would investigate, launches specialized research agents in parallel via Task tool (dave-architect for architecture/design + dave-topic-researcher per topic), synthesizes their findings, and writes RESEARCH.md. Research is broad — it covers codebase patterns, official documentation, web sources, architectural options, strengths, weaknesses, and pitfalls.
</purpose>

<required_reading>
Read before starting:
- `.claude/dave/templates/research.md` — RESEARCH.md structure
- CLAUDE.md is auto-loaded by Claude Code
</required_reading>

<process>

## 1. Verify Prerequisites

**MANDATORY FIRST STEP — Check that the project state and discussion output exist.**

### 1a. Check Project State

```bash
ls -d .state/project/ 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If `.state/project/` does not exist:**
```
Project state not found.

Run /dave:init to initialize the Dave Framework before researching.
```
Exit.

### 1b. Locate DISCUSSION.md

<!-- PARALLEL WORKTREE NOTE: In a parallel worktree setup, each branch has its own
     .state/ directory reflecting that branch's phase. Prefer reading STATE.md to
     locate the phase directory directly rather than relying on glob ordering, which
     could pick the wrong phase if multiple exist on this branch. -->

Search for the most recent DISCUSSION.md. Check in order:

1. **Preferred:** Read STATE.md current position and construct the path directly:
   `.state/milestones/{slug}/phases/{N}/DISCUSSION.md`

2. Active milestone phase (fallback):
   ```bash
   ls .state/milestones/*/phases/*/DISCUSSION.md 2>/dev/null | tail -1
   ```

3. Ad-hoc phases:
   ```bash
   ls .state/milestones/adhoc/phases/*/DISCUSSION.md 2>/dev/null | tail -1
   ```

**If no DISCUSSION.md found:**
```
No DISCUSSION.md found.

Run /dave:discuss first to establish scope and identify research topics.
The research phase needs identified topics to investigate.
```
Exit.

### 1c. Check for Existing RESEARCH.md

Check if RESEARCH.md already exists in the same directory as DISCUSSION.md:

```bash
PHASE_DIR=$(dirname "{path_to_discussion_md}")
ls "$PHASE_DIR/RESEARCH.md" 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If RESEARCH.md already exists:**

Use AskUserQuestion:
- header: "Existing Research Found"
- question: "RESEARCH.md already exists in this phase directory. What would you like to do?"
- options:
  - "Re-research" — Delete and run fresh research from scratch
  - "Supplement" — Keep existing research, add new findings for missing or incomplete topics
  - "Cancel" — Exit without changes

**If "Re-research":** Delete the existing RESEARCH.md. Continue to Step 2.
**If "Supplement":** Read the existing RESEARCH.md. In Step 3, skip topics that already have HIGH confidence findings. Focus on topics marked LOW or with open questions. Continue to Step 2.
**If "Cancel":** Exit.

Record the `PHASE_DIR` for use throughout the workflow.

## 2. Load Context

Load all context needed for the architectural analysis. Read these files:

### 2a. Read DISCUSSION.md

Read the full DISCUSSION.md. Extract:
- **Phase name** from the title (`# Phase Discussion: {phase name}`)
- **Scope** (in scope, out of scope, deferred) from the `## Scope` section
- **Architectural decisions** from the `## Decisions` section
- **Constraints and guardrails** from the `## Constraints` section
- **Success criteria** from the `## Success Criteria` section
- **Research topics** from the `## Research Topics` section — each topic has: name, question, why it matters, known context
- **Open questions** from the `## Open Questions` section

**If no `## Research Topics` section exists or it is empty:**
```
DISCUSSION.md does not contain research topics.

This can happen if:
- The discussion concluded that no research is needed (proceed directly to /dave:plan)
- The research topics section was accidentally omitted

Would you like to:
1. Proceed to /dave:plan (skip research)
2. Re-run /dave:discuss to identify research topics
```
Exit with guidance.

### 2b. Read Project State Files

Read all project state files that exist. Do not error on missing files — use what is available:

- `.state/project/KNOWLEDGE.md` — Tier 1 rules that constrain research recommendations
- `.state/project/PATTERNS.md` — architecture patterns that new work must follow
- `.state/project/STACK.md` — tech stack, libraries, versions
- `.state/project/CONCERNS.md` — known issues and tech debt
- `.state/project/config.yaml` — available tools, model profiles

### 2c. Read Milestone Context (if applicable)

If this phase belongs to a milestone (not ad-hoc):

- `.state/milestones/{slug}/ROADMAP.md` — understand where this phase fits
- `.state/milestones/{slug}/RESEARCH.md` — milestone-level research (if exists, avoid re-researching what is already known)

### 2d. Parse Flags

Parse $ARGUMENTS for:
- `--skip-arch` — If present, set `SKIP_ARCH = true`. The architecture/design research agent will not be launched.
- `--topics topic1,topic2` — If present, parse the comma-separated topic names. Only research topics whose names match (case-insensitive partial match). If a `--topics` value does not match any topic in DISCUSSION.md, warn the user and list available topics.

**If `--topics` is specified but no matches found:**
```
No matching research topics found for: {provided topics}

Available topics from DISCUSSION.md:
- {topic 1 name}
- {topic 2 name}
- ...

Check spelling or run without --topics to research all.
```
Exit.

## 3. Analyze Scope and Design Research Briefs

The orchestrator analyzes the phase scope and prepares briefs for parallel research agents.

### 3a. Analyze Phase Scope

Review the scope, decisions, and constraints from DISCUSSION.md. Identify:

1. **The core problem** — What is this phase trying to accomplish?
2. **The technical domain** — What area of the codebase and external systems does this touch?
3. **The architecture direction** — Based on decisions and constraints, what is the high-level approach? (This informs the architect agent.)
4. **Integration points** — What existing code must the new work connect to?

### 3b. Design Research Briefs

For each research topic from DISCUSSION.md (or the filtered set from `--topics`), create a research brief that a specialized agent will execute. Think like a domain expert — "What would a {relevant expert} research about this?"

For each topic, determine:

1. **Specific questions to answer** — Go beyond the surface question in DISCUSSION.md. Break it into sub-questions that a domain expert would investigate. Include:
   - How does this work technically? (mechanics)
   - What are the limitations? (boundaries)
   - What goes wrong in practice? (pitfalls)
   - What alternatives exist? (options)
   - How does this integrate with our codebase? (compatibility)

2. **Expert lens** — Frame the research as a specific expert would approach it:
   - "What would a database architect research about this migration strategy?"
   - "What would an API integration expert investigate about this provider?"
   - "What would a security engineer check about this authentication flow?"

3. **Source priorities** — Based on the topic, which sources matter most:
   - Official documentation URLs (if known)
   - Codebase files/patterns to examine
   - GitHub issues or discussions to search for
   - Web searches to perform
   - Known community resources

4. **Complexity assessment** — Rate each topic:
   - **Simple lookup** — Answer can be found in one authoritative source (e.g., "What is the rate limit for X API?")
   - **Multi-source research** — Requires cross-referencing several sources (e.g., "What are the tradeoffs between approach A and B?")
   - **Deep investigation** — Requires codebase exploration + external research + synthesis (e.g., "How should we architect the new service layer?")

### 3c. Determine Design/Architecture Research Need

**If `--skip-arch` is set:** Skip this step. Record that design/architecture research was skipped by flag.

**If `--skip-arch` is NOT set:** Evaluate whether design/architecture research is needed:

- If DISCUSSION.md contains architectural decisions that are already firm and specific (not just "use existing patterns"), design research may be light.
- If the phase introduces new patterns, services, or significant structural changes, design research is essential.
- If the phase is purely additive (adding fields, extending existing flows), design research may be skippable.

Decide: Launch dave-architect agent (yes/no) and define its scope.

If design/architecture research IS needed, prepare the architect brief:
- Phase scope (what is being built — 1-2 sentences)
- Key architectural questions (what design decisions need to be made)
- Integration points with existing code (specific services, modules)
- Tier 1 constraints that MUST be satisfied (from KNOWLEDGE.md)
- Known patterns from PATTERNS.md to follow
- Focus areas (specific codebase areas and design decisions to investigate)

## 4. Launch Parallel Research Agents

Launch all research agents in parallel using the Task tool. Each agent runs independently and returns structured findings.

**CRITICAL: All Task tool invocations in this step should be launched in a SINGLE parallel batch.** Do not wait for one agent before launching the next — they are independent.

**Service Registry:** Read available services from `.state/project/config.yaml` for this phase.
Check `phases.research.services` for external AI models with `research` in their capabilities
that can be launched as additional research agents alongside internal agents. For each available
external model, build a self-contained research prompt (same pattern as review external
prompts) and execute via the service's `invoke.command`. Use `phases.research.config.<key>`
for phase-specific overrides (e.g., prompt_prefix). External research agents are
supplementary — internal agents always run. If no external AI services are available,
skip external agent launches silently.

**Agent types launched:**
- **dave-architect** — Design and architecture specialist (1 instance, unless `--skip-arch`)
- **dave-topic-researcher** — Topic research specialist (1 per research topic)
- **External AI models** — From service registry `phases.research.services` (optional, parallel with above)

### 4a. Design/Architecture Research Agent (dave-architect)

**Skip if `--skip-arch` flag is set or Step 3c determined design/architecture research is not needed.**

Launch via Task tool with `subagent_type: "general-purpose"` and include the architect brief from Step 3c:

```
You are a design and architecture research specialist (dave-architect agent).

## Your Task

Research the architectural approach for: {phase name and core problem from 3a}

## Research Brief

Phase scope: {1-2 sentence summary}
Key architectural questions:
{from Step 3c}
Integration points with existing code:
{specific services, modules, file paths}
Tier 1 constraints that MUST be satisfied:
{relevant Tier 1 rules from KNOWLEDGE.md — include full rule text}
Known patterns from PATTERNS.md to follow:
{key patterns — layer architecture, service patterns, gateway pattern}
Focus areas:
{specific codebase areas and design decisions to investigate}

## Context

Architecture direction from discussion:
{architectural decisions from DISCUSSION.md}

Known constraints:
{hard constraints from DISCUSSION.md}

Tech stack:
{relevant items from STACK.md}

## Instructions

1. Explore the existing codebase — read real files, trace patterns, understand integration points
2. Propose 2-3 concrete architectural options with file paths, class names, method signatures
3. Evaluate each option against Tier 1 constraints (compliance check is mandatory)
4. Compare options and recommend one with evidence-based rationale
5. Flag concerns that could affect the plan

Return findings as structured markdown with: Codebase Exploration Summary, Tier 1 Constraints, Architectural Options (2-3), Comparison table, Recommendation with confidence level, Concerns, Integration Analysis.
```

### 4b. Topic Research Agents

For each research topic (from Step 3b), launch a Task tool agent with this prompt structure:

```
You are a research agent investigating a specific technical topic. Think like a domain expert — be thorough, verify claims, and identify both strengths and weaknesses.

## Your Task

Research topic: {topic name}
Expert lens: {expert lens from 3b — e.g., "Think like a database architect"}

## Questions to Answer

{numbered list of specific questions from 3b}

## Known Context

From the discussion:
{known context from DISCUSSION.md for this topic}

Project constraints:
{relevant constraints from DISCUSSION.md}
{relevant Tier 1 rules from KNOWLEDGE.md}

Current codebase patterns:
{relevant patterns from PATTERNS.md}

## How to Research

Research these sources in priority order:

1. **Official documentation** (HIGHEST priority):
   {specific URLs or search queries for official docs}
   Use WebFetch to read official documentation pages.
   Use WebSearch to find official docs if URLs are not known.

2. **Codebase patterns** (HIGH priority):
   {specific files or patterns to look for}
   Use Read to examine specific files.
   Use Grep to find patterns across the codebase.

3. **Web research** (MEDIUM priority):
   {specific search queries}
   Use WebSearch to find practical experience, GitHub issues, known limitations.
   Cross-reference web findings against official docs.

4. **Community knowledge** (LOW priority):
   {what to look for in forums, Stack Overflow, etc.}
   Flag these findings for validation.

## Output Format

Return your findings as structured markdown:

### {topic name}

**Summary:** {One-paragraph summary of what was found}

**Answers to research questions:**
1. {question}: {answer with confidence level}
2. {question}: {answer with confidence level}
...

**Recommendation:** {What to do, based on findings}

**Alternatives considered:**
| Alternative | Pros | Cons | Why rejected/accepted |
|-------------|------|------|----------------------|
| {option} | {pros} | {cons} | {rationale} |

**Strengths of recommended approach:**
- {strength 1}
- {strength 2}

**Weaknesses of recommended approach:**
- {weakness 1}
- {weakness 2}

**Findings with confidence levels:**
- [HIGH] {finding} — Source: {URL or reference}
- [MEDIUM] {finding} — Source: {URL or reference}
- [LOW] {finding} — Source: {URL or reference}

**Pitfalls to avoid:**
- {pitfall 1}: {what goes wrong and how to avoid}
- {pitfall 2}: {what goes wrong and how to avoid}

**Open questions:**
- {anything that could not be resolved}

**Sources consulted:**
- {source 1 — URL or file path}
- {source 2}
```

**Complexity-based agent guidance:**

- **Simple lookup topics:** Direct the agent to focus on one authoritative source. Keep the research brief. The agent should find the answer and verify it, not explore alternatives.
- **Multi-source research topics:** Direct the agent to cross-reference at least 2-3 sources. Emphasize finding contradictions between sources.
- **Deep investigation topics:** Direct the agent to spend more time on codebase exploration. Emphasize integration with existing patterns and propose concrete approaches.

### 4c. Handle Agent Failures

After launching all agents, collect their results. If any agent fails (Task tool returns an error or empty result):

1. **Log the failure** — Record which agent failed and what error occurred.
2. **Do not retry automatically** — One failure does not block other agents.
3. **Record as unknown** — The failed topic becomes a "Remaining Unknown" in RESEARCH.md with a note that research failed and manual investigation is needed.
4. **Inform the user** in the summary (Step 9) about which agents failed.

## 5. Synthesize Results (Dedicated Agent)

After all research agents return, launch a **dedicated synthesis agent** (`dave-research-synthesizer`) to combine findings into RESEARCH.md. This gives the synthesizer a clean context window focused purely on combining, validating, and resolving — not researching.

### 5a. Prepare Synthesis Input

Collect all agent outputs and prepare the synthesis prompt:

1. **Architect output** (if launched) — full output text
2. **Topic researcher outputs** — full output text from each agent
3. **Agent failures** — list of topics where agents failed (from Step 4c)
4. **Context for synthesis:**
   - DISCUSSION.md content (scope, decisions, constraints, success criteria)
   - KNOWLEDGE.md Tier 1 rules (for conflict resolution)
   - PATTERNS.md conventions (for context on intentional patterns)
   - STACK.md tech stack (for technology context)
   - RESEARCH.md template path (`.claude/dave/templates/research.md`)
   - Phase directory path (`{PHASE_DIR}`)
   - Phase name and number

### 5b. Launch Synthesis Agent

Launch `dave-research-synthesizer` via Task tool with `subagent_type: "general-purpose"`:

```
You are the dave-research-synthesizer agent. Synthesize the following parallel research findings into a unified RESEARCH.md.

## Phase Context

Phase {N}: {phase name}
Phase directory: {PHASE_DIR}

## Architect Findings

{Full architect output, or "Skipped (--skip-arch flag / deemed unnecessary). Use DISCUSSION.md decisions for architecture direction."}

## Topic Research Findings

### Topic: {topic 1 name}
{Full topic researcher 1 output}

### Topic: {topic 2 name}
{Full topic researcher 2 output}

...

## Agent Failures

{List of failed agents and their topics, or "None — all agents completed successfully."}

## Project Context

### DISCUSSION.md
{Full DISCUSSION.md content — scope, decisions, constraints, success criteria}

### KNOWLEDGE.md (Tier 1 Rules)
{Tier 1 entries from project KNOWLEDGE.md}

### PATTERNS.md
{Key patterns from project PATTERNS.md}

### STACK.md
{Relevant tech stack info}

## Instructions

Follow your agent specification in .claude/agents/dave-research-synthesizer.md.
Follow the RESEARCH.md template structure from .claude/dave/templates/research.md.
Write RESEARCH.md to: {PHASE_DIR}/RESEARCH.md
```

### 5c. Validate Synthesis Output

After the synthesizer completes:

1. **Check RESEARCH.md exists** in the phase directory
2. **Check structure** — architecture direction, findings, cross-cutting concerns, unknowns, sources sections present
3. **Check completeness** — every research topic has a corresponding findings section
4. **Check "Ready for planning"** assessment — if "no", note the reason for the user

## 6. Research-to-Discussion Loop

Check if research revealed significant new questions that were NOT anticipated during discussion.

### 6a. Identify New Questions

A "new question" is one where:
- Research found a constraint or limitation not mentioned in DISCUSSION.md
- Two viable approaches exist and the choice has significant implications the user should weigh in on
- A Tier 1 knowledge rule conflicts with the most practical approach
- A finding at HIGH confidence contradicts an assumption made during discussion
- An agent discovered a risk that could change the scope

**If no new questions emerged:** Skip to Step 7.

### 6b. Present New Questions

If new questions were identified, present them to the user.

Use AskUserQuestion:
- header: "Research Revealed New Questions"
- question: |
    Research uncovered questions not addressed during discussion:

    {numbered list of new questions with brief context for each}

    How would you like to handle these?
- options:
  - "Answer now" — Provide answers inline (the user will respond to each question)
  - "Run follow-up discussion" — Launch a focused /dave:discuss round for these questions
  - "Proceed with best judgment" — The orchestrator makes its best call and notes the assumptions

**If "Answer now":**
- Present each question one at a time using AskUserQuestion.
- Record each answer.
- Append to DISCUSSION.md under a new section:
  ```markdown
  ## Post-Research Additions

  *Added after research revealed new questions ({date})*

  ### {Question topic}
  - **Question:** {the question}
  - **Answer:** {user's answer}
  - **Context:** Discovered during research of {topic name}
  ```

**If "Run follow-up discussion":**
- Inform the user to run `/dave:discuss` with the new questions as context.
- Write the new questions to a temporary file: `{PHASE_DIR}/RESEARCH_QUESTIONS.md`
- Exit with message:
  ```
  New questions written to RESEARCH_QUESTIONS.md.
  Run /dave:discuss to address them, then re-run /dave:research --topics {affected topics}
  ```

**If "Proceed with best judgment":**
- For each question, record the orchestrator's best-judgment answer.
- Mark these as `[best-judgment]` in RESEARCH.md so the planner knows they are assumptions, not confirmed decisions.
- Add them to the "Remaining Unknowns" section with risk assessments.

## 7. Write RESEARCH.md

Write the research output to `{PHASE_DIR}/RESEARCH.md` using the template structure from `.claude/dave/templates/research.md`.

### 7a. Assemble the Document

Build the RESEARCH.md following this exact structure:

```markdown
# Phase {N}: {Name} - Research

**Researched:** {today's date}
**Time budget:** Research phase (parallel agents)
**Overall confidence:** {aggregate confidence — HIGH if most findings are HIGH, MEDIUM if mixed, LOW if significant unknowns remain}
```

### 7b. Write Architecture Direction

```markdown
<architecture_direction>
## Architecture Direction

{High-level approach synthesized from dave-architect agent or DISCUSSION.md decisions}

**Primary approach:** {One-liner summary}

**Key architectural choices:**
- {Choice 1}: {What and why — from architect recommendation or discussion decisions}
- {Choice 2}: {What and why}

**How it connects to existing code:**
- {Integration point 1}: {How the new code fits — from architect's codebase analysis}
- {Integration point 2}: {How}

{If architecture research was skipped, note: "Architecture research skipped (--skip-arch flag / deemed unnecessary). Direction from discussion phase."}

</architecture_direction>
```

### 7c. Write Research Findings

For each topic, write a findings section:

```markdown
<findings>
## Research Findings

### Topic 1: {topic name}

**Decision:** {What was decided and why — specific and actionable}

**Recommendation:** {Primary recommendation with rationale}

**Alternatives considered:**
| Alternative | Pros | Cons | Why rejected |
|-------------|------|------|--------------|
| {option A} | {pros} | {cons} | {reason} |
| {option B} | {pros} | {cons} | {reason} |

**Findings:**
- [HIGH] {finding backed by official docs or multiple credible sources}
  Source: {URL or reference}
- [MEDIUM] {finding from credible source, not independently confirmed}
  Source: {URL or reference}
- [LOW] {finding needing validation}
  Source: {URL or reference}

**Pitfalls:**
- {pitfall 1}: {What goes wrong and how to avoid it}
- {pitfall 2}: {What goes wrong and how to avoid it}

**Open questions:**
- {Anything not fully resolved — becomes risk in plan}

### Topic 2: {topic name}
...

</findings>
```

**Quality checks before writing each topic:**
- Decision is specific ("Use PaddleOCR v3 with batch size 4") not vague ("Use an OCR provider")
- Every finding has a confidence level AND a source
- Both strengths (in recommendation rationale) and weaknesses (in pitfalls) are present
- Alternatives explain WHY they were rejected
- Pitfalls are actionable ("X goes wrong because Y — avoid by doing Z")

### 7d. Write Cross-Cutting Concerns

```markdown
<cross_cutting>
## Cross-Cutting Concerns

- **{Concern 1}:** {What it affects and how to handle it}
- **{Concern 2}:** {What it affects and how to handle it}

</cross_cutting>
```

### 7e. Write Remaining Unknowns

```markdown
<unknowns>
## Remaining Unknowns

1. **{Unknown 1}**
   - What we know: {partial information}
   - What we do not know: {the gap}
   - Risk level: {HIGH | MEDIUM | LOW}
   - Mitigation: {fallback, test early, or accept risk}

2. **{Unknown 2}**
   ...

</unknowns>
```

### 7f. Write New Questions Section (if applicable)

Only include this section if Step 6 identified new questions:

```markdown
## New Questions for Discussion

*These questions emerged during research and were not anticipated in the original discussion.*

{If answered by user:}
### {Question 1} [resolved]
- **Question:** {question}
- **Answer:** {user's answer}
- **Impact:** {how this affects the plan}

{If handled by best judgment:}
### {Question 2} [best-judgment]
- **Question:** {question}
- **Best judgment:** {orchestrator's answer}
- **Confidence:** {LOW — this is an assumption}
- **Risk if wrong:** {what happens if the assumption is incorrect}
```

### 7g. Write Sources

```markdown
<sources>
## Sources

### Primary (HIGH confidence)
- {Official doc URL} — {what was checked}
- {Codebase file path} — {what pattern it shows}

### Secondary (MEDIUM confidence)
- {Web source verified against official} — {finding + verification}

### Tertiary (LOW confidence — needs validation)
- {Web search only} — {finding, flagged for validation}

</sources>

---

*Phase: {N}*
*Research completed: {today's date}*
*Ready for planning: {yes | no — and why not}*
```

**Set "Ready for planning" to "no" if:**
- Any HIGH-risk unknown remains unresolved
- The research-to-discussion loop resulted in "Run follow-up discussion" (user needs to discuss first)
- A critical agent failed and the topic is essential for planning
Otherwise set to "yes."

## 8. Update Milestone Research (if applicable)

If this phase belongs to a milestone (not ad-hoc), update the milestone-level RESEARCH.md.

### 8a. Check for Milestone RESEARCH.md

```bash
MILESTONE_DIR=$(dirname "$(dirname "$PHASE_DIR")")
ls "$MILESTONE_DIR/RESEARCH.md" 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

### 8b. Create or Update

**If milestone RESEARCH.md does not exist:** Create it with a synthesis header and the current phase's key findings.

```markdown
# Milestone Research: {milestone name}

Cross-phase research synthesis. Updated as each phase completes research.

## Phase {N}: {name}
**Date:** {today}
**Key findings:**
- {Most important finding 1}
- {Most important finding 2}
**Architecture direction:** {one-liner}
**Remaining unknowns carried forward:** {list or "none"}
```

**If milestone RESEARCH.md already exists:** Append a new phase section. Check if any findings from THIS phase contradict or update findings from PREVIOUS phases. If so, add a note:

```markdown
**Cross-phase update:** Phase {N} research updated the understanding of {topic} from Phase {M}. Previous finding: {old}. Updated finding: {new}.
```

## 9. Commit Research

Stage and commit the research artifacts.

```bash
git add {PHASE_DIR}/RESEARCH.md
```

If milestone RESEARCH.md was created or updated:
```bash
git add {MILESTONE_DIR}/RESEARCH.md
```

If DISCUSSION.md was updated with post-research additions (Step 6b):
```bash
git add {PHASE_DIR}/DISCUSSION.md
```

If RESEARCH_QUESTIONS.md was created (Step 6b follow-up discussion path):
```bash
git add {PHASE_DIR}/RESEARCH_QUESTIONS.md
```

Commit with message:
```
research({phase_slug}): parallel research with {N} topic agents

Topics: {comma-separated topic names}
Architecture research: {yes/no/skipped}
Overall confidence: {HIGH/MEDIUM/LOW}
```

## 10. Present Summary

Display a structured summary of the research findings:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DAVE FRAMEWORK ► RESEARCH COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Phase: {N} — {name}
## Overall Confidence: {HIGH | MEDIUM | LOW}

## Architecture Direction

  {One-liner summary of the primary approach}

## Research Findings

  {For each topic:}
  ### {topic name}
    Decision: {one-liner}
    Confidence: {aggregate for this topic}
    Key finding: {most important finding}
    {If pitfalls:} Watch out: {most critical pitfall}

## Agent Summary

  Architecture agent: {completed / skipped / failed}
  Topic agents: {N completed} / {M total} ({any failures noted})

## Confidence Breakdown

  HIGH findings:   {count}
  MEDIUM findings: {count}
  LOW findings:    {count}

## Remaining Unknowns

  {count} unknowns — {list the HIGH-risk ones if any}

{If new questions emerged:}
## New Questions

  {count} new questions — {resolution: answered / best-judgment / pending discussion}

## Files Written

  {PHASE_DIR}/RESEARCH.md
  {MILESTONE_DIR}/RESEARCH.md (if updated)
  {PHASE_DIR}/DISCUSSION.md (if updated with post-research answers)

## Next Steps

  /dave:plan     — Create execution plan from research findings
  {If not ready for planning:}
  /dave:discuss   — Address unresolved questions before planning
```

</process>

<edge_cases>

## Edge Case: No Research Topics in DISCUSSION.md

Handled in Step 2a. If `## Research Topics` is empty or missing, exit with guidance to either proceed to planning or re-run discussion.

## Edge Case: All Topics Are Simple Lookups

If all topics are rated "simple lookup" in Step 3b, the research phase will be fast. This is fine — launch agents anyway. Simple lookups still benefit from verification against official docs. Do not skip the research phase just because topics seem simple.

## Edge Case: Agent Returns Empty or Garbage

Handled in Step 4c. If a Task agent returns an empty result, malformed output, or an error:
1. Record the topic as a "Remaining Unknown" with note: "Research agent failed — manual investigation needed."
2. Continue with other agents' results.
3. Report the failure in the summary.

## Edge Case: --topics Flag Matches No Topics

Handled in Step 2d. List available topics and exit.

## Edge Case: --skip-arch With Architecture-Heavy Phase

If `--skip-arch` is set but the phase clearly needs design/architecture research (introduces new services, layers, or patterns), the orchestrator should warn:
```
Note: --skip-arch flag set, but this phase appears to introduce significant
architectural changes. Architecture direction will be based solely on
DISCUSSION.md decisions. Consider re-running without --skip-arch if the plan
encounters architectural uncertainty.
```
Proceed with the flag honored — do not override the user's explicit choice.

## Edge Case: Supplement Mode (Existing RESEARCH.md)

When supplementing (Step 1c):
- Read existing RESEARCH.md
- Identify topics with LOW confidence or open questions
- Only launch agents for those topics
- Merge new findings into existing RESEARCH.md (do not overwrite HIGH confidence findings)
- Update the "Research completed" date and confidence levels

## Edge Case: DISCUSSION.md Has Post-Research Additions Already

If DISCUSSION.md already has a `## Post-Research Additions` section (from a previous research run), read it as additional context but do not re-surface those questions.

## Edge Case: Milestone RESEARCH.md Has Conflicting Findings

If milestone-level RESEARCH.md from a previous phase contradicts this phase's findings, note the contradiction in both the phase RESEARCH.md and update the milestone RESEARCH.md with a cross-phase reconciliation note.

</edge_cases>

<success_criteria>
- [ ] DISCUSSION.md was found and research topics were extracted
- [ ] Project state files were loaded for context
- [ ] Architectural analysis produced research briefs for each topic
- [ ] Research agents were launched in parallel (not sequentially)
- [ ] Architecture research was performed (or explicitly skipped with rationale)
- [ ] All agent results were synthesized with validated confidence levels
- [ ] Contradictions between sources were identified and resolved
- [ ] Cross-cutting concerns were identified
- [ ] Remaining unknowns have risk levels and mitigation strategies
- [ ] New questions (if any) were surfaced to the user
- [ ] RESEARCH.md follows the template structure exactly
- [ ] Every finding has a confidence level and source
- [ ] Every recommendation has both strengths and weaknesses documented
- [ ] Milestone RESEARCH.md updated (if applicable)
- [ ] Changes committed with descriptive message
- [ ] Summary presented with clear next steps
</success_criteria>
