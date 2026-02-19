---
description: Comprehensive multi-agent code review. Launches parallel reviews for architecture, security, code quality, and gets external second opinions from different AI models (Codex, OpenCode). Use after implementing features or when you want thorough code analysis.
user-invocable: true
argument-hint: "<files or description of what to review> [--quick] [--no-external]"
allowed-tools: Bash, Read, Glob, Grep, Task
---

# Comprehensive Code Review

Launch multiple parallel code reviews covering different aspects, plus get second opinions from external AI models with different reasoning approaches.

## Philosophy

**Review like an expert:**
- Think in terms of best practices and established patterns
- Consider architecture and design holistically
- Reduce cognitive load - simpler is better
- Avoid unnecessary abstractions (we've learned this the hard way with thin wrappers)
- Every service/repository should add clear value

**Different models, different insights:**
- Claude excels at certain patterns
- Kimi K2.5 has strong reasoning capabilities
- GLM 4.7 brings a different perspective
- Codex is optimized for code understanding (latest: gpt-5.3-codex)

## Review Dimensions

### 1. Architecture & Design (code-reviewer agent)
- Clean architecture principles
- Separation of concerns
- Repository/service layer violations
- Unnecessary abstractions (thin wrappers that add no value)
- Cognitive load of the solution

### 2. Security (security-reviewer agent)
- OWASP Top 10 compliance
- Input validation
- Secret handling
- SQL injection risks

### 3. Data Pipeline (data-pipeline-reviewer agent)
- Dagster asset patterns
- Idempotency
- Error handling
- Connection management

### 4. External Perspectives (second-opinion skill)
- Codex (gpt-5.3-codex) - OpenAI's latest code-focused model
- Codex (gpt-5.2-codex) - OpenAI's previous code-focused model
- Kimi K2.5 - Strong reasoning, different approach
- GLM 4.7 - Alternative perspective
- MiniMax M2.1 - Yet another viewpoint

## Usage

### Standard Review (recommended)
```bash
/review src/services/organization_service.py
```

Launches:
- code-reviewer agent (architecture, patterns, quality)
- security-reviewer agent (if auth/input handling detected)
- data-pipeline-reviewer agent (if Dagster assets detected)
- External reviews via Codex (gpt-5.3-codex) + 2-3 OpenCode models

### Quick Review (skip external)
```bash
/review --quick src/services/
```

Only runs internal Claude agents, skips external AI calls.

### Without External AI
```bash
/review --no-external src/pipelines/
```

Runs all Claude agents but no Codex/OpenCode.

## Implementation

When `/review` is invoked:

### Step 1: Analyze Target

Determine what's being reviewed:
```bash
# Check if files exist
# Detect if it's pipeline code (Dagster assets)
# Detect if it has security-sensitive code (auth, passwords, tokens)
```

### Step 2: Launch Internal Agents (Parallel)

Use the Task tool to spawn appropriate reviewer agents:

```
Task: code-reviewer
"Review {files} for architecture, design patterns, code quality. Focus on:
- Clean architecture violations
- Unnecessary abstractions (thin wrappers)
- Cognitive load
- Repository/service patterns
Read CLAUDE.md and docs/COMMON_IMPLEMENTATION_ERRORS.md first."

Task: security-reviewer (if security-relevant)
"Security review of {files}. Focus on OWASP Top 10."

Task: data-pipeline-reviewer (if pipeline code)
"Review {files} for pipeline best practices, idempotency, error handling."
```

### Step 3: Launch External Reviews (Parallel)

**Important:** External agents must NOT modify code. Always include this instruction:

> "You are reviewing code in READ-ONLY mode. Provide analysis and suggestions only. Do NOT make any changes, do NOT use write tools, do NOT edit files."

Launch via Bash (these run independently). Use temp file pattern to avoid shell escaping issues with large prompts:

```bash
# Codex - high reasoning (latest model)
cat > /tmp/dave-review-codex.md << 'PROMPT'
{complete self-contained prompt from External Review Prompt Template below}
PROMPT
codex exec -m gpt-5.3-codex -c 'model_reasoning_effort="high"' \
  "$(cat /tmp/dave-review-codex.md)"

# Kimi K2.5 - different perspective
cat > /tmp/dave-review-kimi.md << 'PROMPT'
{complete self-contained prompt from External Review Prompt Template below}
PROMPT
opencode run -m opencode/kimi-k2.5-free --variant high \
  "$(cat /tmp/dave-review-kimi.md)"

# GLM 4.7 - another viewpoint
cat > /tmp/dave-review-glm.md << 'PROMPT'
{complete self-contained prompt from External Review Prompt Template below}
PROMPT
opencode run -m opencode/glm-4.7-free --variant high \
  "$(cat /tmp/dave-review-glm.md)"
```

**Note on models:** The model list may not always be current. If a model fails:
1. Try the next one on the list
2. Check `opencode models` for current availability
3. Fall back to Codex which is most stable

### Step 4: Synthesize Results

After all reviews complete:

1. **Collect findings** from all sources
2. **Deduplicate** similar issues found by multiple reviewers
3. **Prioritize** by severity (Critical > High > Medium > Low)
4. **Highlight consensus** - issues found by multiple models are likely real
5. **Note divergent opinions** - different models may have valid different perspectives

### Step 5: Triage Assessment

**This is the most important step.** Without triage, every finding feels equally urgent and review fatigue sets in. The goal is to give the developer a clear action plan, not an overwhelming list.

After synthesizing, assess **each deduplicated finding** and categorize it into exactly one of three buckets:

| Category | Meaning | Action |
|----------|---------|--------|
| **False positive** | Not actually an issue in this context. The reviewer misunderstood the code, the pattern is intentional, or the concern doesn't apply to this project. | Dismiss with brief explanation of why. |
| **Fix now** | Important enough to address before merging. Bugs, security issues, correctness problems, or significant design flaws. | Must be resolved in this PR/changeset. |
| **Defer to issue** | Valid concern but not blocking. Refactoring suggestions, minor improvements, tech debt, or enhancements that are better tracked separately. | Create a GitHub issue to track for later. |

**Triage criteria:**

- **Fix now** if: it could cause a bug in production, it's a security vulnerability, it violates a critical project convention (e.g., holding DB connections during LLM calls), or it would make the code significantly harder to maintain.
- **Defer to issue** if: it's a valid improvement but the code works correctly without it, it's a refactoring suggestion that would touch many files, it's an enhancement to error handling that isn't urgent, or it's about patterns that are already tracked as known tech debt.
- **False positive** if: the reviewer didn't have full context (e.g., flagging a pattern that's intentional), the concern is about code outside the review scope, or the suggestion conflicts with an established project decision.

**Consensus boosts severity:** If 3+ reviewers flag the same issue, it should almost never be a false positive. Treat consensus findings as strong candidates for "Fix now".

**Present the triage table to the user** and let them override any categorization before proceeding.

## Output Format

```markdown
## Comprehensive Code Review: {target}

### Review Sources
- [x] code-reviewer (Claude)
- [x] security-reviewer (Claude)
- [x] Codex gpt-5.3-codex
- [x] OpenCode kimi-k2.5-free
- [x] OpenCode glm-4.7-free

### Critical Issues (consensus)
[Issues found by multiple reviewers]

### High Priority
[Significant issues]

### Architecture & Design
[From code-reviewer + external perspectives]

### Security Concerns
[From security-reviewer + external perspectives]

### Code Quality
[Patterns, maintainability, cognitive load]

### Divergent Opinions
[Where reviewers disagreed - worth considering both views]

### Triage Assessment

| # | Finding | Source(s) | Severity | Assessment | Rationale |
|---|---------|-----------|----------|------------|-----------|
| 1 | [Brief description] | code-reviewer, Codex | Critical | **Fix now** | [Why this needs immediate attention] |
| 2 | [Brief description] | security-reviewer | High | **Fix now** | [Why this is blocking] |
| 3 | [Brief description] | Codex, Kimi | Medium | **Defer to issue** | [Valid but not blocking; suggest issue title] |
| 4 | [Brief description] | GLM | Low | **False positive** | [Why this doesn't apply] |
| ... | ... | ... | ... | ... | ... |

**Summary:**
- **Fix now:** X findings (must address before merging)
- **Defer to issue:** Y findings (create GitHub issues)
- **False positives:** Z findings (dismissed)

> Review the triage above. Let me know if you want to reclassify any findings before I proceed with fixes or issue creation.

### Recommended Actions
1. [Highest priority fix]
2. [Next priority]
...
```

## External Review Prompt Template

**CRITICAL:** External models (codex, opencode) do NOT have subagents. They cannot read files, search the codebase, or ask follow-up questions. The review prompt MUST be completely self-contained — everything the model needs to perform a quality review must be in the prompt itself.

Use this comprehensive prompt structure for external AI reviews. The prompt uses temp file pattern to avoid shell escaping:

```markdown
# Code Review Request

READ-ONLY CODE REVIEW - DO NOT MODIFY ANY FILES

You are an expert software architect reviewing Python code. Think like a senior engineer who values simplicity over cleverness, explicit over implicit, and fewer abstractions when possible.

## What Was Built

{Feature/task description from plan must-haves — 1-2 paragraph summary of what this feature does}

## Why It Was Built This Way

{Key architectural decisions — pattern choices and their rationale}

## Change Summary

{CHANGE_SUMMARY.md content if available — structured summary of what changed, mapped to plan tasks, with complexity ratings and areas of concern}

{If CHANGE_SUMMARY.md not available, include the complete git diff instead}

## Targeted File Excerpts

<!-- Include excerpts ONLY for files rated "significant" or "complex" in the
     Change Summary. For each such file, include the changed regions plus
     ~10 lines of surrounding context. For new files, include full content
     since the diff IS the file. Trivial and straightforward files are
     summary-only — no excerpts needed. -->

### {file_path} (significant)
```{language}
{changed region with surrounding context, from the diff or file read}
```

### {file_path} (complex)
```{language}
{changed region with surrounding context}
```

{...repeat for all significant/complex files...}

## Areas of Concern (from Change Summary)

{Copy the "Areas of Concern" section from CHANGE_SUMMARY.md — these are
 the highest-priority items for review. If no Change Summary, list key
 areas the orchestrator identified during planning.}

## Project Rules (MUST check against these)

These are non-negotiable project conventions. Flag ANY violation.

{Full Tier 1 entries from KNOWLEDGE.md — include complete rule text, not just IDs}

## Project Patterns (context for review)

These are intentional patterns — do NOT flag these as issues:

{Key patterns from PATTERNS.md that reviewers might otherwise flag}

## Review Focus Areas

1. BUGS & EDGE CASES - What could fail? What's not handled?
2. ERROR HANDLING - Is it comprehensive? Appropriate?
3. ARCHITECTURE - Does this follow clean architecture? Any layer violations?
4. UNNECESSARY COMPLEXITY - Are there thin wrappers that add no value? Over-engineering?
5. COGNITIVE LOAD - How hard is this to understand? Could it be simpler?
6. SECURITY - Any obvious vulnerabilities?
7. BEST PRACTICES - Does it follow Python/project conventions?

## What NOT to Review

- Files rated "trivial" in the Change Summary (unless they appear in Areas of Concern)
- Patterns listed above as intentional project conventions
- Style preferences that contradict the project's established patterns
- Suggestions to add features or capabilities beyond the scope described above

## Expected Output Format

For each finding, provide:

### Finding {N}
- **File:** {path}:{line range}
- **Severity:** critical | high | medium | low
- **Category:** bug | security | correctness | data-integrity | performance | maintainability
- **What is wrong:** {specific description}
- **Suggested fix:** {concrete suggestion}
- **Confidence:** {how sure you are this is a real issue, not a false positive}

If no issues found, state "No issues found" with a brief explanation of what was checked.
```

## Configuration

Reasoning effort based on review scope:

| Scope | Codex Effort | OpenCode Variant |
|-------|--------------|------------------|
| Single function | `medium` | `medium` |
| Single file | `high` | `high` |
| Multiple files | `high` | `high` |
| Architecture review | `xhigh` | `max` |

## Tips

1. **Large files**: For files > 500 lines, review in sections
2. **Context matters**: Include imports and related files for better analysis
3. **Be patient**: External AI calls take time, but diverse perspectives are valuable
4. **Trust consensus**: If 3+ reviewers flag the same issue, prioritize it
5. **Value disagreement**: Different opinions often reveal nuance worth exploring
