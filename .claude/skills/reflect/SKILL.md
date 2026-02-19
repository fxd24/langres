---
description: Analyze conversations to identify pitfalls from skills, subagents, and hooks. Generates improvement recommendations without modifying files. Use after complex sessions, when troubleshooting repeated failures, or during retrospectives.
user-invocable: true
argument-hint: <optional: specific area to analyze - skills, subagents, hooks, or all>
allowed-tools: Read, Grep, Glob
---

# Conversation Reflection & Improvement

Analyze completed Claude Code sessions to identify issues and improve skills, subagents, and hooks for better future performance.

<when_to_use>
- After completing complex multi-step tasks
- When you encountered repeated failures or confusion
- During team retrospectives on AI-assisted development
- Before committing improvements to skills/subagents/hooks
- When a skill or subagent behaves unexpectedly
</when_to_use>

## Analysis Framework

### Step 1: Gather Context

Read the relevant configuration files:

```bash
# Skills
.claude/skills/*/SKILL.md

# Subagents
.claude/agents/*.md

# Hooks (in settings)
.claude/settings.json
~/.claude/settings.json
```

### Step 2: Analyze by Category

#### Skills Analysis

| Question                   | What to Look For                                            |
| -------------------------- | ----------------------------------------------------------- |
| **Invocation accuracy**    | Was the skill triggered when needed? Too often? Not enough? |
| **Description match**      | Does the description match actual usage patterns?           |
| **Tool restrictions**      | Did `allowed-tools` prevent necessary actions?              |
| **Progressive disclosure** | Is essential info in SKILL.md, details in supporting files? |
| **Size efficiency**        | Is SKILL.md under 500 lines? Could it be more focused?      |

**Common skill issues:**

- Description too vague -> skill invoked incorrectly
- Description too narrow -> skill not invoked when needed
- Too much content -> context waste
- Missing trigger terms -> user has to explicitly invoke

#### Subagent Analysis

| Question               | What to Look For                                                     |
| ---------------------- | -------------------------------------------------------------------- |
| **Spawn timing**       | Was it spawned at the right moment? Too early? Too late?             |
| **Tool restrictions**  | Did missing tools cause the agent to fail?                           |
| **Skills inheritance** | Did the agent have the skills it needed? (Subagents don't inherit!)  |
| **Context pollution**  | Did unrelated context confuse the agent?                             |
| **Model choice**       | Was the model appropriate (haiku for quick tasks, opus for complex)? |

**Common subagent issues:**

- Missing skills in `skills:` field
- Wrong tool restrictions (e.g., needs Bash but only has Read)
- Spawned for tasks that could be done inline
- Model too expensive for simple tasks

#### Hook Analysis

| Question               | What to Look For                              |
| ---------------------- | --------------------------------------------- |
| **Trigger frequency**  | How often did hooks fire? Were any excessive? |
| **Blocking behavior**  | Did hooks prevent necessary operations?       |
| **Error messages**     | Were hook rejections clear and actionable?    |
| **Performance impact** | Did hooks add significant latency?            |

**Common hook issues:**

- Overly aggressive validation blocking valid operations
- Unclear error messages when hooks reject
- Hooks that fire on every action (context waste)
- Missing hooks for operations that should be validated

### Step 3: Identify Patterns

Look for recurring failure modes:

1. **Over-triggering** - Skill/agent invoked when not needed
2. **Under-triggering** - Skill/agent not invoked when it should be
3. **Context confusion** - AI confused by conflicting guidance
4. **Tool gaps** - Agent couldn't complete task due to missing tools
5. **Skill conflicts** - Multiple skills providing contradictory guidance
6. **Verbose waste** - Skills loading too much irrelevant context

### Step 4: Generate Recommendations

For each issue found, provide:

1. **What went wrong** - Specific description of the failure
2. **Root cause** - Why it happened (description, tools, missing info)
3. **Recommendation** - Exact change to make
4. **Risk assessment** - Could this break existing behavior?

## Output Format

```markdown
## Reflection Report

### Session Overview

- **Complexity**: [low/medium/high]
- **Main objectives**: [what was attempted]
- **Outcome**: [success/partial/failed]

### Skills Analysis

| Skill  | Invocations | Effectiveness     | Issues        |
| ------ | ----------- | ----------------- | ------------- |
| [name] | [count]     | [good/mixed/poor] | [description] |

**Recommended changes:**

- [ ] [skill]: Update description from "[old]" to "[new]"
- [ ] [skill]: Add trigger term "[term]"
- [ ] [skill]: Move [section] to supporting file (reduce size)

### Subagent Analysis

| Agent  | Spawns  | Effectiveness     | Issues        |
| ------ | ------- | ----------------- | ------------- |
| [name] | [count] | [good/mixed/poor] | [description] |

**Recommended changes:**

- [ ] [agent]: Add skill "[skill]" to skills field
- [ ] [agent]: Add tool "[tool]" to tools field
- [ ] [agent]: Change model from [old] to [new]

### Hook Analysis

| Hook   | Triggers | Blocks  | Issues        |
| ------ | -------- | ------- | ------------- |
| [name] | [count]  | [count] | [description] |

**Recommended changes:**

- [ ] [hook]: Adjust pattern from "[old]" to "[new]"
- [ ] [hook]: Improve error message to "[message]"

### Failure Patterns Detected

1. **[Pattern name]**
   - Occurrences: [count]
   - Impact: [high/medium/low]
   - Fix: [specific recommendation]

### Preservation Checklist

Before applying any changes, verify:

- [ ] Existing trigger terms still work
- [ ] No skills/agents become orphaned
- [ ] Tool restrictions don't break current workflows
- [ ] Changes don't conflict with other skills

### Actionable Improvements

Priority order (highest impact, lowest risk first):

1. [ ] [Specific change with file path]
2. [ ] [Specific change with file path]
3. [ ] [Specific change with file path]
```

<best_practices>
## Best Practices

### Preserve What Works

- **Test trigger terms** - Ensure existing invocation patterns still work
- **Preserve working descriptions** - Only modify problematic parts
- **Keep backward compatibility** - Retain tool access that might be needed
- **Document changes** - Note why each change was made

### Progressive Improvement

- **One change at a time** - Rewriting everything at once causes regressions
- **Validate before expanding** - Fix issues before adding features
- **Monitor after changes** - Watch for regressions in next sessions
</best_practices>

### Common Fixes

| Problem                 | Solution                                         |
| ----------------------- | ------------------------------------------------ |
| Skill invoked too often | Narrow description, add "NOT for X" section      |
| Skill not invoked       | Add common trigger terms to description          |
| Subagent fails on tasks | Check skills/tools fields, ensure they're listed |
| Hook blocks valid ops   | Adjust regex pattern, add exceptions             |
| Too much context loaded | Move details to supporting files                 |

## Supporting Files

- [patterns.md](patterns.md) - Catalog of known failure patterns
- [templates.md](templates.md) - Example reflection reports

## Integration

After generating a reflection report:

1. **Review recommendations** with the team
2. **Apply changes incrementally** (one at a time)
3. **Test in next session** to verify improvements
4. **Update patterns.md** with any new failure modes discovered
