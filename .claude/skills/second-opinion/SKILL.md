---
description: Get a second opinion from external AI coding agents (Codex, OpenCode). Use when you want different models to analyze code, suggest improvements, or spot issues that Claude might miss. Different models think differently about code.
user-invocable: true
argument-hint: "<prompt to send to external AI> [--tool codex|opencode] [--model <model>] [--effort low|medium|high|max]"
allowed-tools: Bash, Read, Glob, Grep
---

# Second Opinion - External AI Code Review

Run Codex or OpenCode in headless mode to get analysis from different AI models. Different models have different strengths and may catch things others miss.

## Available Tools & Models

### Codex (OpenAI ChatGPT account)

```bash
codex exec -m <model> -c 'model_reasoning_effort="<effort>"' "<prompt>"
```

| Model | Reasoning Levels | Notes |
|-------|------------------|-------|
| `gpt-5.3-codex` | low/medium/high/xhigh | **Latest, most capable** |
| `gpt-5.2-codex` | low/medium/high/xhigh | Previous generation |
| `gpt-5.1-codex` | low/medium/high | Older version |
| `gpt-5-codex` | low/medium/high | Oldest codex model |

### OpenCode (Multi-provider)

```bash
opencode run -m <model> --variant <effort> "<prompt>"
```

**OpenAI Models (via subscription):**
| Model | Notes |
|-------|-------|
| `openai/gpt-5.3-codex` | Latest Codex |
| `openai/gpt-5.2-codex` | Previous Codex |
| `openai/gpt-5.2` | Base GPT-5.2 |
| `openai/gpt-5.1-codex` | Older codex |
| `openai/gpt-5.1-codex-max` | Max reasoning |
| `openai/gpt-5.1-codex-mini` | Faster, cheaper |

**Free Models (diverse perspectives):**
| Model | Notes |
|-------|-------|
| `opencode/kimi-k2.5-free` | Moonshot AI - strong reasoning |
| `opencode/glm-4.7-free` | Zhipu AI - good for Chinese context |
| `opencode/minimax-m2.1-free` | MiniMax - different approach |
| `opencode/big-pickle` | Alternative perspective |
| `opencode/gpt-5-nano` | Quick checks |
| `opencode/trinity-large-preview-free` | Another option |

**Note:** Model availability may change. If a model fails, try another or check `opencode models` for current list.

## Usage Examples

### Basic Usage

```bash
# Default: Codex with gpt-5.3-codex, high reasoning
/second-opinion "Review src/services/org_service.py for edge cases and error handling"

# Specific tool
/second-opinion --tool opencode "Analyze this code architecture"

# Specific model
/second-opinion --model opencode/kimi-k2.5-free "Suggest refactoring for the repository pattern"

# Higher reasoning effort
/second-opinion --effort xhigh "Find potential security issues in auth flow"
```

### Getting Diverse Perspectives

For comprehensive review, get opinions from multiple models:

```bash
# Codex (OpenAI reasoning)
codex exec -m gpt-5.3-codex -c 'model_reasoning_effort="high"' "Review for bugs: $(cat src/file.py)"

# Kimi (Moonshot AI)
opencode run -m opencode/kimi-k2.5-free --variant high "Review for bugs: $(cat src/file.py)"

# GLM (Zhipu AI)
opencode run -m opencode/glm-4.7-free --variant high "Review for bugs: $(cat src/file.py)"
```

## Implementation

When this skill is invoked:

1. **Parse arguments** to determine tool, model, and effort level
2. **Default to Codex** with `gpt-5.3-codex` and `high` reasoning if not specified
3. **Run the command** and capture output
4. **Present findings** without modifying any code

### Argument Parsing

| Flag | Values | Default |
|------|--------|---------|
| `--tool` | `codex`, `opencode` | `codex` |
| `--model` | See tables above | `gpt-5.3-codex` (codex) or `opencode/kimi-k2.5-free` (opencode) |
| `--effort` | `low`, `medium`, `high`, `xhigh`/`max` | `high` |

### Command Construction

**For Codex:**
```bash
codex exec -m {model} -c 'model_reasoning_effort="{effort}"' "{prompt}"
```

**For OpenCode:**
```bash
opencode run -m {model} --variant {effort} "{prompt}"
```

## Important Guidelines

1. **Read-only**: External agents should NEVER modify code. Add explicit instruction:
   > "Analyze and provide suggestions only. Do not make any changes."

2. **Include context**: When reviewing specific files, include the file content in the prompt:
   ```bash
   codex exec ... "Review this code:\n$(cat src/path/to/file.py)"
   ```

3. **Be specific**: Vague prompts get vague answers. Include what aspects to focus on.

4. **Model failures**: If a model fails, try:
   - Different model from same provider
   - Different provider entirely
   - Check `opencode models` for current availability

## Recommended Prompts for Code Review

```
"Review this code for:
1. Potential bugs and edge cases
2. Error handling completeness
3. Architecture and design issues
4. Performance concerns
5. Security vulnerabilities

Provide specific, actionable suggestions. Do not modify any code.

Code to review:
{file_content}"
```
