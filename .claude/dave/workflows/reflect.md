<purpose>
Run the learning loop after a phase completes. Extracts knowledge from the gap between plan and actual execution, review findings, and verification results. Produces SUMMARY.md, updates phase KNOWLEDGE.md with new Tier 2 entries, proposes promotions for entries exceeding the threshold, and optionally aggregates knowledge upward at milestone boundaries.

You are the reflect orchestrator. You collect phase artifacts, launch the learning extractor agent for analysis, then apply the extracted knowledge to the appropriate files. You manage the promotion process (presenting candidates to the human for approval). You write SUMMARY.md and update state.
</purpose>

<downstream_awareness>
**Reflect produces:**

1. **SUMMARY.md** — Phase completion record. Consumed by future planners and milestone aggregation.
2. **Phase KNOWLEDGE.md updates** — New Tier 2 entries with evidence. Consumed by all future agents.
3. **Project KNOWLEDGE.md updates** — Verification count increments and promotions. Consumed by all agents.
4. **PATTERNS.md updates** — New conventions discovered. Consumed by planners, reviewers, TDD developers.
5. **CONCERNS.md updates** — New tech debt or risks. Consumed by planners and milestone-end review.

**Reflect reads:**
- PLAN.md (what was planned)
- EXECUTION_STATE.md (what was actually done)
- REVIEWS.md (what reviews found)
- OPEN_QUESTIONS.md (ambiguous items and decisions)
- VERIFICATION.md (verification results)
- Phase KNOWLEDGE.md (existing phase-level entries)
- Project KNOWLEDGE.md, PATTERNS.md, CONCERNS.md (existing project-level state)
- config.yaml (tier2_promotion_threshold)
</downstream_awareness>

<required_reading>
Read before starting:
- `.claude/dave/templates/summary.md` — SUMMARY.md structure
- `.claude/dave/templates/knowledge.md` — KNOWLEDGE.md format
- `.claude/dave/references/knowledge-format.md` — entry format, promotion, aggregation
- CLAUDE.md is auto-loaded by Claude Code
</required_reading>

<process>

## 1. Verify Prerequisites

**MANDATORY FIRST STEP — Check that the phase has artifacts to reflect on.**

### 1a. Check Project State

```bash
ls -d .state/project/ 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If `.state/project/` does not exist:**
```
Project state not found.
Run `/dave:init` first to initialize the Dave Framework.
```
STOP HERE.

### 1b. Locate Phase Directory

<!-- PARALLEL WORKTREE NOTE: STATE.md is per-branch. Each worktree's branch has its
     own STATE.md pointing to its active phase. Reflect runs on THIS branch's phase. -->

Find the active milestone and phase:

```bash
cat .state/STATE.md 2>/dev/null
```

Read STATE.md to determine the current milestone slug and phase number. Construct the phase path:
`.state/milestones/{slug}/phases/{N}/`

### 1c. Check Phase Artifacts Exist

At minimum, PLAN.md and VERIFICATION.md should exist:

```bash
ls .state/milestones/{slug}/phases/{N}/PLAN.md 2>/dev/null && echo "PLAN EXISTS" || echo "PLAN MISSING"
ls .state/milestones/{slug}/phases/{N}/VERIFICATION.md 2>/dev/null && echo "VERIFICATION EXISTS" || echo "VERIFICATION MISSING"
```

**If PLAN.md does not exist:**
```
No plan found for the current phase. Nothing to reflect on.
```
STOP HERE.

**If VERIFICATION.md does not exist:**
```
Verification not yet complete. Reflect works best after verification.
Continue anyway? (Reflect will have less data to analyze.)
```
Use AskUserQuestion:
- "Continue" — Proceed with available artifacts
- "Wait for verification" — STOP HERE

### 1d. Parse Flags

- `--promote-only` — Set `PROMOTE_ONLY = true`. Skip to Step 5.
- `--summary-only` — Set `SUMMARY_ONLY = true`. Skip to Step 4.

### 1e. Read config.yaml

Read `.state/project/config.yaml` for:
- `knowledge.tier2_promotion_threshold` (default: 3)

---

## 2. Collect Phase Data

### 2a. Read All Phase Artifacts

Read all available files from the phase directory:

| File | Required | Contains |
|------|----------|----------|
| `PLAN.md` | Yes | Must-haves, tasks, verification matrix |
| `EXECUTION_STATE.md` | No | Task status, deviations, commit hashes |
| `REVIEWS.md` | No | Triaged review findings |
| `OPEN_QUESTIONS.md` | No | Ambiguous items and decisions |
| `VERIFICATION.md` | No | Layer results, gaps, evidence |
| `KNOWLEDGE.md` | No | Existing phase-level entries |
| `DISCUSSION.md` | No | Scope, decisions, guardrails |

### 2b. Read Project-Level State

| File | Contains |
|------|----------|
| `project/KNOWLEDGE.md` | Tier 1 and Tier 2 entries |
| `project/PATTERNS.md` | Conventions |
| `project/CONCERNS.md` | Tech debt and risks |

### 2c. Compute Metrics

```bash
MERGE_BASE=$(git merge-base HEAD main 2>/dev/null || git merge-base HEAD origin/main)
git log --oneline $MERGE_BASE..HEAD | wc -l
git diff --stat $MERGE_BASE..HEAD | tail -1
```

Count: commits, lines changed, files changed.

---

## 3. Launch Learning Extractor

### 3a. Launch Agent

Launch the `dave-learning-extractor` agent (from `.claude/agents/dave-learning-extractor.md`) via the Task tool:

**Prompt:**
```
Extract learnings from this completed phase.

## Phase: {N} — {name}

## Phase Artifacts

### PLAN.md
{content of PLAN.md — focus on must-haves, task specifications}

### EXECUTION_STATE.md
{content of EXECUTION_STATE.md — task status, deviations}

### REVIEWS.md
{content of REVIEWS.md — all findings and classifications}

### OPEN_QUESTIONS.md
{content of OPEN_QUESTIONS.md — items and decisions}

### VERIFICATION.md
{content of VERIFICATION.md — layer results, gaps}

### Phase KNOWLEDGE.md
{content of phase KNOWLEDGE.md if exists, or "No existing phase knowledge"}

## Project Context

### Project KNOWLEDGE.md
{content of project KNOWLEDGE.md — for dedup and verification count checks}

### Project PATTERNS.md
{content of project PATTERNS.md — for pattern novelty checks}

### Project CONCERNS.md
{content of project CONCERNS.md — for resolved concern checks}

## Configuration
- Tier 2 promotion threshold: {threshold from config.yaml}

## Phase Directory
{phase directory path}
```

### 3b. Collect Results

The learning extractor returns a structured report with:
- Plan vs actual gaps
- Review patterns
- Verification patterns
- New Tier 2 entries
- Existing entries to update
- Promotion candidates
- Pattern and concern updates

---

## 4. Write SUMMARY.md

### 4a. Generate Summary

Using the learning extractor's report and the phase artifacts, write SUMMARY.md following the template from `.claude/dave/templates/summary.md`.

Key content sources:
- **What was built:** PLAN.md must-haves + VERIFICATION.md Layer 1 results
- **Metrics:** Git stats + EXECUTION_STATE.md + REVIEWS.md + VERIFICATION.md
- **Deviations:** EXECUTION_STATE.md deviations section
- **Key decisions:** OPEN_QUESTIONS.md decisions + execution deviations
- **Lessons learned:** Learning extractor's new Tier 2 entries (summarized)
- **Deferred work:** REVIEWS.md deferred items
- **Open items:** Any unresolved items

### 4b. Write File

Write SUMMARY.md to `{phase directory}/SUMMARY.md`.

**If `--summary-only` was set:** Skip to Step 7 (State Update).

---

## 5. Apply Knowledge Updates

### 5a. Update Phase KNOWLEDGE.md

From the learning extractor's report, add new Tier 2 entries to the phase KNOWLEDGE.md:

- If phase KNOWLEDGE.md already exists (from execution deviations), append to the Tier 2 section
- If it does not exist, create it following the knowledge template

Also add any human decisions from OPEN_QUESTIONS.md as Tier 1 entries:
- Each resolved open question where the human made a decision becomes a Tier 1 entry
- Source: `Human (open question decision)`

### 5b. Update Project KNOWLEDGE.md

From the learning extractor's report:

**Verification count updates:**
- For each existing Tier 2 entry that was independently confirmed by this phase, increment the Verified count
- Update the entry in project KNOWLEDGE.md

**Do NOT add phase-specific entries to project level.** Only milestone aggregation promotes entries upward.

### 5c. Update PATTERNS.md (if new patterns found)

From the learning extractor's report, if new conventions were identified:
- Read current PATTERNS.md
- Append new patterns in the appropriate section
- Include evidence (which phase discovered this pattern)

### 5d. Update CONCERNS.md (if new concerns found)

From the learning extractor's report, if new tech debt or risks were identified:
- Read current CONCERNS.md
- Append new concerns
- Mark any resolved concerns

---

## 6. Promotion Process

### 6a. Identify Promotion Candidates

From the learning extractor's report, collect entries where:
- Verified count exceeds `tier2_promotion_threshold`
- OR the extractor explicitly flagged as a promotion candidate

### 6b. Present Candidates to User

If there are promotion candidates:

```
## Knowledge Promotion Candidates

{N} Tier 2 entries have been verified enough times to be considered for promotion to Tier 1 (absolute authority).

### Candidate 1: [A###]

**Current entry:**
  {entry text}
  Verified: {N} times across {M} phases
  Confidence: {level}

**Proposed Tier 1 entry:**
  [H###] {generalized rule text}
  Source: Human (confirmed promotion from A###)
  Severity: {recommended severity}

**Evidence:**
  {list of phases where this was independently confirmed}
```

Use AskUserQuestion for each candidate:
- header: "Promote to Tier 1?"
- question: "Should this pattern become an absolute rule?"
- options:
  - "Approve" — Promote as-is
  - "Modify and approve" — Let user edit the rule text first
  - "Reject" — Keep as Tier 2, remove promotion candidate flag

### 6c. Apply Approved Promotions

For each approved promotion:
1. Add the new Tier 1 entry to project KNOWLEDGE.md with the next available H-ID
2. Mark the original Tier 2 entry as `Promoted: Yes`
3. Record the promotion in the phase KNOWLEDGE.md

For "Modify and approve":
- Present the entry text for user editing via AskUserQuestion
- Apply the modified text

### 6d. Handle Rejections

For rejected promotions:
- Remove the `Promotion candidate` flag from the Tier 2 entry
- The entry stays at Tier 2 and can be re-proposed if more evidence accumulates

---

## 7. Milestone Boundary Check

### 7a. Is This the Last Phase?

Read the milestone ROADMAP.md to check if this is the last phase:

```bash
cat .state/milestones/{slug}/ROADMAP.md 2>/dev/null
```

**If this is NOT the last phase:** Skip to Step 8.

### 7b. Milestone Knowledge Aggregation

If this is the last phase in the milestone:

1. Read ALL phase KNOWLEDGE.md files from this milestone
2. Read ALL phase SUMMARY.md files from this milestone
3. Identify patterns that recurred across 2+ phases
4. Generalize phase-specific entries into milestone-level entries
5. Write milestone KNOWLEDGE.md to `.state/milestones/{slug}/KNOWLEDGE.md`

**Aggregation rules (from knowledge-format.md):**
- Keep patterns that recurred across phases or affected multiple areas
- Drop one-off implementation details and context-specific findings
- Generalize: "PaddleOCR v3 batch > 4 causes OOM on RTX 3070" → "OCR providers need batch size testing on target hardware"

### 7c. Project Knowledge Aggregation

From the milestone KNOWLEDGE.md, identify entries useful in UNRELATED future work:
1. Generalize further for project-level applicability
2. Propose project KNOWLEDGE.md additions (Tier 2)
3. Present to user for approval before adding

**Aggregation rules:**
- Only lessons useful in UNRELATED future work
- Tier 1 additions ALWAYS require human approval
- Generalize technology-specific details unless the tech is used project-wide

### 7d. Milestone Summary

If this is the last phase, also generate a brief milestone summary:

```
## Milestone Complete: {milestone-name}

Phases: {N} completed
Duration: {total time}
Key outcomes: {1-3 bullet points}
Knowledge entries: {N} Tier 2 created, {N} promoted to Tier 1
```

---

## 8. Gate Check

Before declaring reflect complete, verify:

1. **SUMMARY.md written:** Phase summary exists with accurate metrics
2. **Phase KNOWLEDGE.md updated:** New entries added (or "no new entries" noted)
3. **Project KNOWLEDGE.md updated:** Verification counts incremented where applicable
4. **Promotions handled:** All candidates presented and decisions applied
5. **PATTERNS.md updated:** If new patterns were identified
6. **CONCERNS.md updated:** If new concerns were identified or resolved
7. **Milestone aggregation done:** If this was the last phase

### Gate Passed

```
## Reflect Complete

### Phase Summary
{brief summary — 2-3 sentences}

### Knowledge Extracted
- New Tier 2 entries: {N}
- Verification updates: {N} existing entries confirmed
- Promotions: {N} approved, {N} rejected
- New patterns: {N}
- New concerns: {N}

### Files Updated
- {list of files written or updated}

**Phase {N} is complete.** Start the next phase or milestone.
```

### Gate Failed

If any required file could not be written, explain what failed and why.

---

## 9. Update STATE.md

Update `.state/STATE.md` with:
- Current phase status: "complete" (reflected)
- Performance metrics update (phase duration, velocity)
- Next action: Start next phase, or complete milestone
- Phase artifact checklist: all items marked "exists"

### 9a. Commit State Files

<!-- PARALLEL WORKTREE NOTE: Stage specific files rather than all of .state/ to avoid
     accidentally including state from other phases that may have been merged in. -->

```bash
git add ${PHASE_DIR}/SUMMARY.md ${PHASE_DIR}/KNOWLEDGE.md .state/STATE.md .state/project/KNOWLEDGE.md .state/project/PATTERNS.md .state/project/CONCERNS.md
git commit -m "state({phase_slug}): reflect complete — {N} entries extracted, SUMMARY.md written

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

Stage only the files that were actually modified during reflect. If milestone KNOWLEDGE.md was created, add that too.

</process>

<edge_cases>

## Edge Case: No Deviations, No Review Findings, Clean Verification

If the phase executed perfectly:
- SUMMARY.md still gets written (metrics, must-haves, etc.)
- KNOWLEDGE.md may have zero new entries — this is fine
- State: "No significant learnings — plan executed as designed"

## Edge Case: Phase Was Skipped or Partial

If some phase artifacts are missing (e.g., review was skipped):
- Extract from whatever is available
- Note the gaps in SUMMARY.md
- Flag missing artifacts: "Review was skipped — limited learning data"

## Edge Case: --promote-only With No Candidates

If `--promote-only` was set but no entries exceed the threshold:
```
No Tier 2 entries currently meet the promotion threshold ({N} verifications required).

Closest candidates:
- [A###] {text} — Verified: {N}/{threshold}
```

## Edge Case: Milestone Boundary With Missing Phase Summaries

If this is the last phase but earlier phases have no SUMMARY.md:
- Warn the user
- Aggregate from whatever KNOWLEDGE.md files exist
- Note the gaps in milestone aggregation

## Edge Case: Large Number of Findings

If the learning extractor returns many entries (>10):
- Present them in groups of 5 for readability
- Ask the user if they want to review all entries or trust the extraction
- Only truly novel, specific entries should make it through

## Edge Case: Reflect Run Multiple Times

If SUMMARY.md already exists (reflect was run before):
- Ask the user: "SUMMARY.md already exists. Overwrite or skip?"
- If overwrite: Replace with new content
- If skip: Only process knowledge updates and promotions

</edge_cases>

<success_criteria>
- [ ] Phase artifacts read (PLAN.md + available artifacts)
- [ ] Learning extractor agent launched and completed
- [ ] SUMMARY.md written with accurate metrics and lessons
- [ ] Phase KNOWLEDGE.md updated with new Tier 2 entries (or noted as empty)
- [ ] Project KNOWLEDGE.md verification counts updated
- [ ] Promotion candidates presented to user (if any)
- [ ] Approved promotions applied
- [ ] PATTERNS.md and CONCERNS.md updated (if applicable)
- [ ] Milestone aggregation done (if last phase)
- [ ] STATE.md updated with phase completion
- [ ] State files committed
- [ ] User knows next steps
</success_criteria>
