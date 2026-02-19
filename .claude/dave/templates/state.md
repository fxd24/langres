# State Template

Template for `.state/STATE.md` -- the project's living memory across milestones and sessions.

---

## File Template

```markdown
# Project State

## Project Reference

See: .state/project/PROJECT.md (updated {date})

**Core value:** {One-liner from PROJECT.md}
**Current focus:** {Current milestone and phase}

## Current Position

Milestone: {milestone-slug} -- {milestone name}
Phase: {N} of {M} ({phase name})
Status: {Discussion | Research | Planning | Implementing | Reviewing | Verifying | Reflecting | Complete}
Last activity: {YYYY-MM-DD} -- {What happened}

Progress: [{filled}{empty}] {pct}%

## Performance Metrics

**Velocity:**
- Phases completed: {N}
- Average phase duration: {X} min
- Total execution time: {X.X} hours

**By Milestone:**

| Milestone | Phases | Duration | Avg/Phase |
|-----------|--------|----------|-----------|
| - | - | - | - |

**Recent Trend:**
- Last 3 phases: {durations}
- Trend: {Improving | Stable | Degrading}

*Updated after each phase completion*

## Active Milestone

**Milestone:** {milestone-slug}
**Roadmap:** .state/milestones/{milestone-slug}/ROADMAP.md

| Phase | Name | Status |
|-------|------|--------|
| 1 | {name} | {Not started | In progress | Complete} |
| 2 | {name} | {status} |
| 3 | {name} | {status} |

**Current phase artifacts:**
- DISCUSSION.md: {exists | pending}
- RESEARCH.md: {exists | pending}
- PLAN.md: {exists | pending}
- KNOWLEDGE.md: {exists | pending}
- VERIFICATION.md: {exists | pending}

## Session Continuity

Last session: {YYYY-MM-DD HH:MM}
Stopped at: {Description of last completed action}
Next action: {What should happen next}
Blockers: {Any blockers, or "None"}
```

<purpose>

STATE.md is the project's short-term memory spanning all milestones and sessions.

**Problem it solves:** Each session starts without context. Agents do not know
where the project is, what happened last, or what is next. Without STATE.md,
every session begins with expensive codebase exploration.

**Solution:** A single, small file that is:
- Read first in every workflow
- Updated after every significant action
- Contains a digest of current position and velocity
- Enables instant session restoration

</purpose>

<lifecycle>

**Creation:** After the first milestone roadmap is created.
- Reference PROJECT.md for core value and focus
- Initialize empty metrics and milestone tracking
- Set position to "Phase 1, Discussion"

**Reading:** First step of every workflow.
- Agents read STATE.md to know where the project is
- The orchestrator reads it to decide what to do next
- A new session reads it to restore context immediately

**Writing:** After every significant action.
- Phase transitions: Update position, phase statuses, artifact checklist
- Phase completion: Update metrics (velocity, duration, trend)
- Milestone transitions: Update milestone table, reset phase tracking
- Session end: Update session continuity section

</lifecycle>

<sections>

### Project Reference
Points to PROJECT.md for full context. Includes:
- Core value (the ONE thing that matters)
- Current focus (which milestone and phase)
- Last update date (triggers re-read if stale)

### Current Position
Where the project is right now:
- Which milestone
- Which phase within the milestone
- Current status within the phase (8 possible values matching the 8 workflow phases)
- Last activity with date
- Visual progress bar

Progress calculation: (completed phases across all milestones) / (total phases) x 100%

### Performance Metrics
Track velocity to understand execution patterns:
- Total phases completed
- Average phase duration
- Per-milestone breakdown
- Recent trend (improving/stable/degrading)

Updated after each phase completion.

### Active Milestone
The currently active milestone with:
- Link to its ROADMAP.md
- Phase status table (overview of all phases in the milestone)
- Current phase artifact checklist (which files exist vs pending)

This tells the orchestrator exactly what state the current phase is in.

### Session Continuity
Enables instant resumption:
- When the last session occurred
- What was last completed
- What should happen next
- Any blockers preventing progress

</sections>

<size_constraint>

Keep STATE.md under 100 lines.

It is a DIGEST, not an archive. If it grows too large:
- Keep only the active milestone in the milestone table
- Summarize completed milestones as a single line
- Keep only 3 most recent phases in the trend

The goal is "read once, know where we are." If it is too long, that fails.

</size_constraint>
