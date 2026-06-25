---
name: prompting-claude-4
description: Expert guidance for prompting Claude 4.x models (Opus, Sonnet, Haiku tiers). Use when writing system prompts, optimizing model behavior, fixing behavioral issues like overtriggering or over-engineering, or migrating from older Claude models. Includes XML patterns and behavioral fixes.
---

# Prompting Claude 4.x Models

This skill provides expert guidance for writing effective prompts for Claude 4.x
models across the Opus, Sonnet, and Haiku tiers. It stays version-agnostic: the
tiers trade intelligence for latency and cost — pick per task, and consult the
live Anthropic model docs for current IDs and benchmarks.

## Core Principles

### 1. Be Explicit with Instructions

Claude 4.x models are trained for **precise instruction following**. They take instructions literally and do exactly what you ask—nothing more.

**Less effective:**

```
Create an analytics dashboard
```

**More effective:**

```
Create an analytics dashboard. Include as many relevant features and interactions as possible. Go beyond the basics to create a fully-featured implementation.
```

### 2. Add Context for Why

Explaining motivation helps Claude understand your goals:

**Less effective:**

```
NEVER use ellipses
```

**More effective:**

```
Your response will be read aloud by a text-to-speech engine, so never use ellipses since the text-to-speech engine will not know how to pronounce them.
```

### 3. Tell Claude What TO Do (Not What NOT to Do)

Format guidance is more effective when positive:

- Instead of: "Do not use markdown in your response"
- Try: "Write in smoothly flowing prose paragraphs."

### 4. Use XML Tags for Structure

Claude is trained to recognize XML tags for organizing prompts:

```xml
<behavior_instructions>
Your tone should be professional and direct.
</behavior_instructions>

<coding_guidelines>
Follow existing patterns in the codebase.
Always read files before editing.
</coding_guidelines>
```

## Behavioral Tendencies to Address

Claude 4.x models share tendencies worth steering with explicit prompt guidance.
See [prompt-snippets.md](prompt-snippets.md) for copy-paste fixes.

- **Over-engineering**: add explicit guidance to keep solutions minimal.
- **Tool overtriggering**: use moderate language instead of aggressive directives.
- **Conservative code exploration**: explicitly instruct to read files first.
- **"Think" sensitivity**: when extended thinking is disabled, replace "think"
  with "consider/evaluate/believe".
- **Aggressive parallel tool calling**: powerful, but can bottleneck a system —
  scope it when concurrency is costly (see the parallel-tool-calls pattern below).
- **Concise-by-default communication**: models may skip post-tool summaries; ask
  explicitly when you want them (see Communication Style Adjustments below).

Higher tiers (Opus) also expose finer output-effort control and preserve thinking
blocks across turns; consult the live model docs for the current per-model feature
matrix rather than pinning it here.

## Essential Prompt Patterns

### Parallel Tool Calling

Boost parallel tool usage to ~100%:

```xml
<use_parallel_tool_calls>
If you intend to call multiple tools and there are no dependencies between the tool calls, make all of the independent calls in parallel. Prioritize calling tools simultaneously whenever the actions can be done in parallel rather than sequentially. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. Maximize use of parallel tool calls where possible to increase speed and efficiency. However, if some tool calls depend on previous calls to inform dependent values like the parameters, do NOT call these tools in parallel and instead call them sequentially. Never use placeholders or guess missing parameters in tool calls.
</use_parallel_tool_calls>
```

### Default to Action (for Proactive Behavior)

```xml
<default_to_action>
By default, implement changes rather than only suggesting them. If the user's intent is unclear, infer the most useful likely action and proceed, using tools to discover any missing details instead of guessing. Try to infer the user's intent about whether a tool call (e.g., file edit or read) is intended or not, and act accordingly.
</default_to_action>
```

### Conservative Action (for Hesitant Behavior)

```xml
<do_not_act_before_instructions>
Do not jump into implementation or change files unless clearly instructed to make changes. When the user's intent is ambiguous, default to providing information, doing research, and providing recommendations rather than taking action. Only proceed with edits, modifications, or implementations when the user explicitly requests them.
</do_not_act_before_instructions>
```

### Code Exploration

```xml
<investigate_before_answering>
ALWAYS read and understand relevant files before proposing code edits. Do not speculate about code you have not inspected. If the user references a specific file/path, you MUST open and inspect it before explaining or proposing fixes. Be rigorous and persistent in searching code for key facts. Thoroughly review the style, conventions, and abstractions of the codebase before implementing new features or abstractions.
</investigate_before_answering>
```

### Minimize Markdown/Bullet Points

```xml
<avoid_excessive_markdown_and_bullet_points>
When writing reports, documents, technical explanations, analyses, or any long-form content, write in clear, flowing prose using complete paragraphs and sentences. Use standard paragraph breaks for organization and reserve markdown primarily for `inline code`, code blocks, and simple headings. Avoid using **bold** and *italics*.

DO NOT use ordered lists or unordered lists unless: a) you're presenting truly discrete items where a list format is the best option, or b) the user explicitly requests a list or ranking.

Instead of listing items with bullets or numbers, incorporate them naturally into sentences. This guidance applies especially to technical writing.
</avoid_excessive_markdown_and_bullet_points>
```

## Extended Thinking

Enable for complex coding and reasoning tasks:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",  # or your target model
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[...]
)
```

**Budget recommendations:**

- Start with 10k-16k tokens
- Use 32k+ for very complex tasks (consider batch processing)
- Minimum budget is 1,024 tokens

**Interleaved thinking** (beta): Enables reasoning between tool calls:

```
Beta header: interleaved-thinking-2025-05-14
```

## Context Window Management

Claude 4.x models track their remaining context budget. For long-running tasks:

```
Your context window will be automatically compacted as it approaches its limit, allowing you to continue working indefinitely from where you left off. Therefore, do not stop tasks early due to token budget concerns. As you approach your token budget limit, save your current progress and state to memory before the context window refreshes. Always be as persistent and autonomous as possible and complete tasks fully.
```

## Communication Style Adjustments

Claude 4.x is more concise by default. To get more verbose output:

```
After completing a task that involves tool use, provide a quick summary of the work you've done.
```

## Quick Fixes Reference

See [prompt-snippets.md](prompt-snippets.md) for detailed behavioral fixes including:

- Tool overtriggering → Soften aggressive language
- Over-engineering → Add minimalism instructions
- Code exploration issues → Add explicit read-first requirements
- Frontend "AI slop" aesthetic → Add design guidance
- "Think" word sensitivity → Replace with alternatives

## Additional Resources

- [Extended thinking tips](extended-thinking-tips.md) - Detailed guidance for thinking mode
- [Model comparison](model-comparison.md) - Detailed feature comparison

---

**Sources:**

- [Prompting best practices - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices)
- [What's new in Claude 4.5](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-5)
- [Migrating to Claude 4.5](https://platform.claude.com/docs/en/about-claude/models/migrating-to-claude-4)
- [Claude Opus 4.5 Migration Plugin](https://github.com/anthropics/claude-code/tree/main/plugins/claude-opus-4-5-migration)
