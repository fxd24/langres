# dave-change-summarizer: Process Guide

## Input Context

You receive from the review workflow orchestrator:

| Input | What it tells you |
|-------|------------------|
| Changed files list | Which files were modified, added, or deleted |
| Git diff | The complete diff (all hunks, all files) |
| PLAN.md | What was supposed to be built and why |
| Phase directory path | Where to write CHANGE_SUMMARY.md |

You also have tool access to read any file in the codebase, which you
should use to understand the context around changes when the diff alone
is ambiguous.

## Process

### Step 1: Parse Inputs

#### 1a: Catalog changed files

From the changed files list, categorize each file:

| Category | Examples |
|----------|----------|
| New files | Files that did not exist before (created) |
| Modified files | Files with changes to existing code |
| Deleted files | Files that were removed |
| Test files | Files in tests/ directories |
| Config/infra files | pyproject.toml, alembic/*, Makefile, etc. |

#### 1b: Extract plan task map

From PLAN.md, extract:
- Each task ID and description
- The files each task is supposed to touch
- The must-haves (truths, artifacts, key links)

Build a task-to-file mapping:
```
Task 1: [file_a.py, file_b.py]
Task 2: [file_c.py, file_d.py]
```

### Step 2: Analyze Each File's Changes

For each changed file, produce a semantic description:

#### 2a: Identify the nature of the change

- **New class/function:** What does it do? What is its interface (params, return type)?
- **Modified class/function:** What changed in behavior? What stayed the same?
- **Structural change:** File moved, renamed, split, or merged?
- **Import changes:** New dependencies added? Old ones removed?
- **Configuration change:** What setting changed and to what?

#### 2b: Map to plan task

For each file:
- Which plan task does this change serve? (Use the task-to-file mapping)
- If a file was changed but is NOT in any task's file list, flag it as
  "unplanned change" with your best guess at why it was touched

#### 2c: Assess change complexity

Rate each file's changes:
- **Trivial:** Import additions, type hint fixes, docstring updates
- **Straightforward:** New method following established pattern, config value change
- **Significant:** New class, changed method signatures, altered control flow
- **Complex:** Concurrent logic, state management changes, cross-cutting modifications

#### 2d: Note areas of concern

Flag anything that should get extra reviewer attention:
- Changes to error handling or exception paths
- Changes to database queries or session management
- Changes to external API calls or gateway usage
- Changes to security-sensitive code (auth, input validation)
- Deletion of code (especially tests or safety checks)
- Changes that touch multiple architectural layers
- Any change that seems unrelated to the plan

### Step 3: Produce the Change Summary

Write CHANGE_SUMMARY.md to the phase directory using this format:

```markdown
# Phase {N}: {Name} - Change Summary

**Generated:** {date}
**Total files changed:** {N} ({N} new, {N} modified, {N} deleted)
**Total lines changed:** +{N} -{N}
**Diff compression:** {summary_lines}/{diff_lines} ({percentage}%)

## Plan Task Mapping

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| T1 | {task description} | {file_a.py}, {file_b.py} | All files changed as planned |
| T2 | {task description} | {file_c.py} | file_d.py planned but not changed |
| -- | Unplanned | {file_e.py} | Not in any task (see concerns) |

## Changes by File

### {file_path} (new | modified | deleted) -- Task {N}
**Complexity:** trivial | straightforward | significant | complex
**What changed:**
- {Semantic description of change 1}
- {Semantic description of change 2}
**Key interfaces added/changed:**
- `async def method_name(param: Type) -> ReturnType` -- {what it does}
**Lines of interest:** {line ranges where the most significant changes are}

### {file_path} (modified) -- Task {N}
**Complexity:** trivial
**What changed:**
- Added import for {new_dependency}

{...repeat for all files...}

## Unplanned Changes

<!-- Files changed that do not map to any plan task. These deserve
     extra scrutiny -- they may be legitimate supporting changes or
     accidental scope creep. -->

### {file_path}
**What changed:** {description}
**Likely reason:** {best guess -- e.g., "import needed by new service in Task 2"}
**Risk:** low | medium | high

## Areas of Concern

<!-- Specific things reviewers should look at closely. Each entry
     points to a specific file and describes what to check. -->

1. **{Concern title}** -- `{file_path}:{line_range}`
   {What to check and why}

2. **{Concern title}** -- `{file_path}:{line_range}`
   {What to check and why}

## Suggested Review Focus

<!-- Guidance for each reviewer type on what is most relevant to them.
     This allows reviewers to prioritize their time. -->

### For code-reviewer
- {Focus area 1 with file references}
- {Focus area 2 with file references}

### For security-reviewer (if applicable)
- {Focus area 1}

### For data-pipeline-reviewer (if applicable)
- {Focus area 1}

### For database-expert (if applicable)
- {Focus area 1}

### For external models
- {Key areas to examine, with enough context since they cannot read files}

---

*Phase: {N}*
*Summary generated: {date}*
*From diff: {merge_base_sha}..{head_sha}*
```

### Step 4: Self-Validate

Before finishing, verify:
- [ ] Every changed file appears in the summary
- [ ] Every file is mapped to a plan task (or flagged as unplanned)
- [ ] No diff hunk was skipped or ignored
- [ ] The summary is shorter than the raw diff
- [ ] Areas of concern are specific (file + what to look for), not vague

Cross-check: count the files in the "Changed Files" input and count the
files in your summary. If the counts do not match, you missed a file.
Go back and find it.

## Final Step: Verify Output Structure

Before returning your output, verify it matches the inline CHANGE_SUMMARY.md structure defined above:
1. Every changed file is listed (compare count against `git diff --name-only`)
2. Plan task mapping column has no blanks (use "unplanned" if no mapping)
3. Suggested review focus section is populated for each reviewer type

## Success Criteria

Change summary is complete when:

- [ ] Every changed file is cataloged with category and plan task mapping
- [ ] Every significant change has a semantic description (what changed in behavior)
- [ ] Every file has a complexity rating
- [ ] Unplanned changes are explicitly flagged
- [ ] Areas of concern are listed with specific file + line guidance
- [ ] Suggested review focus areas are provided per reviewer type
- [ ] The summary is meaningfully shorter than the raw diff
- [ ] No information was lost -- a reviewer reading only the summary
      knows everything they need to decide which files to inspect
