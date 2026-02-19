<purpose>
Push the feature branch to remote, create a pull request with a structured description derived from phase artifacts, and optionally monitor CI checks. The PR description is generated from PLAN.md (what was built), REVIEWS.md (code quality summary), and VERIFICATION.md (verification results).

You are the push orchestrator. You prepare the PR, push the branch, create the PR, and optionally wait for CI. You do not modify code — if CI fails, you diagnose and report, but the user decides what to do.
</purpose>

<downstream_awareness>
**Push produces:**

1. **Remote branch** — Available for PR review and CI
2. **Pull request** — With structured description from phase artifacts
3. **CI status** — If `--wait`, the workflow reports pass/fail

**Push reads:**
- PLAN.md must-haves (what was built — becomes PR summary)
- REVIEWS.md summary (code quality — becomes PR quality section)
- VERIFICATION.md results (verification — becomes PR test plan)
- EXECUTION_STATE.md (deviations — noted in PR if any)
- OPEN_QUESTIONS.md decisions (key decisions — noted in PR)
</downstream_awareness>

<required_reading>
Read all files referenced by the invoking prompt's execution_context before starting.
In particular:
- The verification template (.claude/dave/templates/verification.md) — understand verification output
- The reviews template (.claude/dave/templates/reviews.md) — understand review output
- CLAUDE.md — project rules (especially git safety rules, PR conventions)
</required_reading>

<process>

## 1. Verify Prerequisites

**MANDATORY FIRST STEP — Check that verification passed and we are on a feature branch.**

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
     own STATE.md pointing to its active phase. The push workflow pushes THIS branch's
     work, which correctly scopes to this worktree's phase. Each phase gets its own
     branch and its own PR -- there is no single feature branch assumption. -->

Find the active milestone and phase:

```bash
cat .state/STATE.md 2>/dev/null
```

Read STATE.md to determine the current milestone slug and phase number. Construct the phase path:
`.state/milestones/{slug}/phases/{N}/`

### 1c. Check Verification Gate

```bash
ls .state/milestones/{slug}/phases/{N}/VERIFICATION.md 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

**If VERIFICATION.md does not exist:**
```
Verification not yet complete.
Run `/dave:verify` first.
```
STOP HERE.

Read VERIFICATION.md and check the overall status. If status is not "passed":
```
Verification did not pass (status: {status}).
Resolve verification gaps before pushing.
```
STOP HERE.

### 1d. Check Branch Safety

```bash
git branch --show-current
```

**If on `main` or `master`:**
```
Cannot push from main/master.
Create a feature branch first: `git checkout -b feature/{phase-name}`
```
STOP HERE.

### 1e. Check Working Tree

```bash
git status --porcelain
```

**If there are uncommitted changes:**
```
Working tree is not clean.

Uncommitted changes:
{list}

Commit or stash changes before pushing.
```
STOP HERE.

### 1f. Parse Flags

- `--wait` — Set `WAIT_FOR_CI = true`
- `--draft` — Set `CREATE_DRAFT = true`
- `--no-pr` — Set `SKIP_PR = true`

---

## 2. Prepare PR Description

### 2a. Read Phase Artifacts

Read the following files and extract relevant content:

| File | Extract |
|------|---------|
| PLAN.md | `<must_haves>` section (truths, artifacts, key links) |
| PLAN.md | Phase name and description |
| REVIEWS.md | Summary section (counts: fix now, defer, dismissed) |
| REVIEWS.md | Deferred items (these become follow-up work) |
| VERIFICATION.md | Layer summaries and overall confidence |
| EXECUTION_STATE.md | Deviation count and descriptions (if any) |
| DISCUSSION.md | Phase scope and key decisions |

### 2b. Compute Change Stats

```bash
MERGE_BASE=$(git merge-base HEAD main 2>/dev/null || git merge-base HEAD origin/main)
git diff --stat $MERGE_BASE..HEAD | tail -1
git diff --name-only $MERGE_BASE..HEAD | wc -l
git log --oneline $MERGE_BASE..HEAD | wc -l
```

Store: total lines changed, files changed, commit count.

### 2c. Generate PR Title

Format: `feat({phase-slug}): {phase name}`

Keep under 70 characters. Use the phase name from PLAN.md or DISCUSSION.md.

### 2d. Generate PR Body

Build the PR body from phase artifacts:

```markdown
## Summary

{1-3 bullet points describing what this phase delivers, derived from PLAN.md must-haves}

## Must-Haves Verified

| # | Truth | Status |
|---|-------|--------|
{truths from VERIFICATION.md Layer 1, each with VERIFIED/FAILED status}

## Changes

- **Files changed:** {N}
- **Lines changed:** {+N / -N}
- **Commits:** {N}
{if deviations:}
- **Plan deviations:** {N} (see details below)

## Code Quality

- **Review iterations:** {N} (from REVIEWS.md fix loop history)
- **Findings fixed:** {N} (fix-now items resolved)
- **Deferred items:** {N} (tracked for follow-up)
- **Verification confidence:** {HIGH | MEDIUM | LOW}

{if deferred items:}
### Deferred Items

{list deferred items from REVIEWS.md with brief descriptions — these are follow-up work}

{if deviations:}
### Plan Deviations

{list deviations from EXECUTION_STATE.md with brief descriptions}

## Test Plan

{verification steps from VERIFICATION.md Layer 3 — what was tested and results}

- [ ] All automated tests pass
- [ ] Lint passes
- [ ] Layer 1 (plan conformance): {status}
- [ ] Layer 2 (code review): {status}
- [ ] Layer 3 (automated functional): {status}
- [ ] Layer 4 (human oversight): {status}

🤖 Generated with [Claude Code](https://claude.com/claude-code) via Dave Framework
```

---

## 3. Push Branch

### 3a. Check Remote Tracking

```bash
git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null
```

**If no upstream set:** Push with `-u` flag.
**If upstream exists:** Push normally.

### 3b. Push

```bash
git push -u origin $(git branch --show-current)
```

**If push fails:**
- Check if remote rejects (protected branch, force push needed, etc.)
- Report the error and ask the user how to proceed
- Do NOT force push

### 3c. Handle --no-pr

If `--no-pr` was set:
```
Branch pushed to remote: {branch-name}

Skipping PR creation (--no-pr flag).
Create manually: gh pr create
```
Skip to Step 5 (State Update).

---

## 4. Create Pull Request

### 4a. Check for Existing PR

```bash
gh pr list --head $(git branch --show-current) --json number,title,url --jq '.[0]'
```

**If a PR already exists:**
```
A pull request already exists for this branch:
  #{number}: {title}
  {url}

Would you like to update it or skip PR creation?
```

Use AskUserQuestion:
- "Update existing PR" — Update the PR description with the new body
- "Skip" — Leave existing PR as-is

**If updating:**
```bash
gh pr edit {number} --body "{new body}"
```

### 4b. Create PR

Build the `gh pr create` command:

```bash
gh pr create --title "{title}" --body "$(cat <<'EOF'
{PR body from step 2d}
EOF
)"
```

Add `--draft` if `CREATE_DRAFT` is true.

### 4c. Capture PR URL

Parse the output of `gh pr create` for the PR URL. Store it for state update.

```
Pull request created: {url}
```

---

## 5. CI Monitoring (if --wait)

### 5a. Check if --wait Was Set

If `WAIT_FOR_CI` is not true, skip to Step 6.

### 5b. Get PR Number

```bash
gh pr view --json number --jq '.number'
```

### 5c. Poll CI Checks

Poll every 30 seconds, up to 10 minutes:

```bash
gh pr checks $(gh pr view --json number --jq '.number') --json name,state,conclusion
```

**While any check has state "QUEUED" or "IN_PROGRESS":**
```
CI in progress... ({completed}/{total} checks complete)
```
Wait 30 seconds, then re-poll.

### 5d. Report CI Results

After all checks complete (or timeout):

**If all pass:**
```
CI passed! All {N} checks green.
```

**If any fail:**
```
CI failed.

| Check | Status | Details |
|-------|--------|---------|
{check table}

Failed checks:
{for each failed check, show the name and any available log snippet}
```

Do NOT attempt to fix CI failures automatically. Report them to the user.

### 5e. Handle CI Timeout

If polling exceeds 10 minutes:
```
CI monitoring timed out after 10 minutes.
{N}/{M} checks still pending.

Check status manually: gh pr checks {number}
```

---

## 6. Gate Check

Before declaring push complete:

1. **Branch pushed:** Remote has the latest commits
2. **PR created:** (unless `--no-pr`) PR exists with structured description
3. **CI status known:** (if `--wait`) All checks reported

### Gate Passed

```
## Push Complete

Branch: {branch-name}
PR: {url} {if draft: "(draft)"}
{if --wait: "CI: {passed | failed | timeout}"}
Commits: {N}
Files: {N} changed

**Next steps:**
- Review and merge the PR
- Run `/dave:reflect` for the learning loop
```

### Gate Failed

If push or PR creation failed, explain the error and what action is needed.

---

## 7. Update STATE.md

Update `.state/STATE.md` with:
- Current phase status: "push complete"
- PR URL
- CI status (if --wait was used)
- Next action: `/dave:reflect` or merge PR

</process>

<edge_cases>

## Edge Case: No Commits Since Merge Base

If `git diff $MERGE_BASE..HEAD` shows no changes:
```
No changes to push. The branch is up-to-date with main.
```
STOP HERE.

## Edge Case: Branch Already Pushed

If the branch already has an upstream and all commits are pushed:
```
Branch is already up-to-date with remote.
```
Skip the push step and proceed to PR creation (if needed).

## Edge Case: gh CLI Not Available

If `gh` is not installed:
```
GitHub CLI (gh) is not available.

Push was successful. Create a PR manually:
  1. Go to the repository on GitHub
  2. Create a PR from branch: {branch-name}
  3. Use this description:

{PR body}
```

## Edge Case: PR Template Exists

If the repository has a `.github/pull_request_template.md`:
- Read the template
- Map phase artifacts to template sections where they align
- Fill in any template sections that do not map to phase artifacts

## Edge Case: Large Diff

If the diff exceeds 5000 lines:
- Summarize changes by directory/module instead of listing individual files
- Note the size in the PR description
- Suggest the reviewer use per-commit review

## Edge Case: Multiple Phase Push

If multiple phases have been executed since last push:
- Include all phase summaries in the PR description
- List must-haves from all phases
- Note this in the summary: "This PR covers phases {N}-{M}"

</edge_cases>

<success_criteria>
- [ ] Prerequisites checked (verification passed, feature branch, clean tree)
- [ ] PR description generated from phase artifacts (PLAN.md, REVIEWS.md, VERIFICATION.md)
- [ ] Branch pushed to remote
- [ ] PR created with structured description (unless --no-pr)
- [ ] PR is NOT a force push
- [ ] CI status reported (if --wait)
- [ ] STATE.md updated with PR URL
- [ ] User knows next steps (/dave:reflect or merge)
</success_criteria>
