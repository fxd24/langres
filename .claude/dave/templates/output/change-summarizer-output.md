# Phase {N}: {Name} - Change Summary

**Generated:** {date}
**Total files changed:** {total_files} ({new_count} new, {modified_count} modified, {deleted_count} deleted)
**Total lines changed:** +{lines_added} -{lines_removed}
**Plan tasks:** {plan_task_count}

## Plan Task Mapping

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| {task_id} | {task_description} | {file_list} | {all changed | partial | not started} |
| -- | Unplanned | {file_list} | Not in any task |

## Changes by File

| File | Change Type | Task | Complexity | Description |
|------|-------------|------|------------|-------------|
| `{file_path}` | {new\|modified\|deleted} | {task_id} | {trivial\|straightforward\|significant\|complex} | {semantic_description} |

### Key Interfaces Added/Changed

- `{file_path}`: `{signature}` -- {purpose}

### Lines of Interest

- `{file_path}:{line_range}` -- {why_notable}

## Unplanned Changes

| File | What Changed | Likely Reason | Risk |
|------|-------------|---------------|------|
| `{file_path}` | {description} | {rationale} | {low\|medium\|high} |

## Areas of Concern

1. **{concern_title}** -- `{file_path}:{line_range}`
   {what_to_check_and_why}

## Suggested Review Focus

### For code-reviewer
- {focus_area_with_file_references}

### For security-reviewer
- {focus_area}

### For data-pipeline-reviewer
- {focus_area}

### For database-expert
- {focus_area}

### For external models
- {key_areas_with_context}

---

*Phase: {N}*
*Summary generated: {date}*
*From diff: {merge_base_sha}..{head_sha}*
