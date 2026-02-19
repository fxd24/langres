<purpose>
Show milestone progress and route to the next action. This is a quick, read-only status check that reads STATE.md and phase artifacts to determine where the project is and what command to run next.

You are the progress reporter. You read state files and present a clear summary. You do not modify anything. Output ONLY the progress report — no additional commentary or analysis.

<!-- PARALLEL WORKTREE NOTE: This shows progress for the phase tracked in STATE.md
     on the CURRENT BRANCH. In a parallel worktree setup, each worktree has its own
     branch with its own STATE.md. To check all worktrees, run /dave:progress in
     each worktree separately. This is a per-session view, not a global dashboard. -->
</purpose>

<process>

## 1. Check Project State

```bash
ls -d .state/project/ 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If `.state/project/` does not exist:**
```
Dave Framework not initialized.

Run `/dave:init` to get started.
```
STOP HERE.

## 2. Read State

```bash
cat .state/STATE.md 2>/dev/null
```

Extract from STATE.md:
- Current milestone slug and name
- Current phase number and name
- Current status
- Last activity
- Next action (if recorded)

## 3. Read Milestone ROADMAP

```bash
cat .state/milestones/{slug}/ROADMAP.md 2>/dev/null
```

Extract the phase list with statuses.

## 4. Check Current Phase Artifacts

Check which artifacts exist for the current phase:

```bash
for f in DISCUSSION.md RESEARCH.md PLAN.md EXECUTION_STATE.md REVIEWS.md OPEN_QUESTIONS.md VERIFICATION.md SUMMARY.md KNOWLEDGE.md; do
  [ -f ".state/milestones/{slug}/phases/{N}/$f" ] && echo "$f: exists" || echo "$f: pending"
done
```

## 5. Determine Next Action

Based on artifact existence and STATE.md status, determine which command to run next:

| Condition | Next Action |
|-----------|-------------|
| No DISCUSSION.md | `/dave:discuss` |
| DISCUSSION.md exists, no RESEARCH.md | `/dave:research` |
| RESEARCH.md exists, no PLAN.md | `/dave:plan` |
| PLAN.md exists, no EXECUTION_STATE.md | `/dave:execute` |
| EXECUTION_STATE.md exists, no REVIEWS.md | `/dave:review` |
| REVIEWS.md exists, no VERIFICATION.md | `/dave:verify` |
| VERIFICATION.md exists (passed), no PR pushed | `/dave:push` |
| Push complete, no SUMMARY.md | `/dave:reflect` |
| SUMMARY.md exists | Phase complete — start next phase or milestone |

If the phase is mid-execution (EXECUTION_STATE.md shows incomplete tasks):
- Next action: `/dave:execute` (will resume from interrupted state)

If the review has unresolved fix-now items:
- Next action: `/dave:review --fix-only`

If verification has gaps:
- Next action: `/dave:verify --gaps-only`

## 6. Present Progress Report

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DAVE FRAMEWORK ► PROGRESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Milestone: {milestone-name}
  Phase:     {N}/{M} — {phase name}
  Status:    {status}

  Progress: [{filled}{empty}] {pct}%

  Phases:
    {for each phase in roadmap:}
    {icon} Phase {N}: {name} — {status}
    {icon legend: ✓ = complete, ► = current, · = not started}

  Current Phase Artifacts:
    {icon} DISCUSSION.md    {icon} RESEARCH.md
    {icon} PLAN.md          {icon} EXECUTION_STATE.md
    {icon} REVIEWS.md       {icon} VERIFICATION.md
    {icon} SUMMARY.md
    {icon legend: ✓ = exists, · = pending}

  Last activity: {date} — {description}

  ► Next: {next action command}
    {brief explanation of what it does}
```

## 7. Verbose Mode (if --verbose)

If `--verbose` was set, also show:

### 7a. Detailed Artifact Status

For each existing artifact, show key metrics:

```
  Artifact Details:
    PLAN.md         — {N} tasks in {M} waves, {N} must-haves
    EXECUTION_STATE — {N}/{M} tasks complete, {N} deviations
    REVIEWS.md      — {N} findings ({N} fixed, {N} deferred, {N} dismissed)
    VERIFICATION.md — {confidence} confidence, {N}/{M} layers passed
```

### 7b. Velocity Metrics

From STATE.md performance metrics section:

```
  Velocity:
    Phases completed:    {N}
    Avg phase duration:  {X} min
    Total time:          {X.X} hours
    Recent trend:        {Improving | Stable | Degrading}
```

### 7c. Knowledge Summary

```
  Knowledge:
    Project Tier 1: {N} entries
    Project Tier 2: {N} entries ({N} promotion candidates)
    Phase entries:  {N}
```

</process>

<edge_cases>

## Edge Case: No Active Milestone

If STATE.md exists but has no active milestone:
```
No active milestone.

Run `/dave:discuss` with a feature description to start a new phase,
or set up a milestone first.
```

## Edge Case: STATE.md Corrupted or Missing Sections

If STATE.md is incomplete:
- Show what is available
- Flag missing sections
- Suggest `/dave:state sync` to repair

## Edge Case: Multiple Milestones

If completed milestones exist:
- Show only the active milestone in the main view
- In `--verbose` mode, show a one-line summary of completed milestones

## Edge Case: Phase Has No Standard Artifacts

If the current phase was done outside the Dave workflow (manual work):
- Show that artifacts are missing
- Suggest starting with `/dave:discuss` for the next phase

</edge_cases>

<success_criteria>
- [ ] STATE.md read successfully
- [ ] Current milestone and phase identified
- [ ] Phase artifact status checked
- [ ] Next action determined correctly based on artifact state
- [ ] Progress report displayed clearly
- [ ] No files modified (read-only)
</success_criteria>
