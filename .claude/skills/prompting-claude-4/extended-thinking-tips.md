# Extended Thinking Tips

Guidance for using extended thinking effectively with Claude 4.x models.

## When to Enable Extended Thinking

Enable for:

- Complex multi-step reasoning
- Advanced coding tasks
- Mathematical problems
- Analysis requiring careful deliberation
- Tasks where you want to see Claude's reasoning process

Skip for:

- Simple queries
- Real-time/low-latency requirements
- High-volume processing where cost is critical

## Basic Configuration

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Minimum: 1,024
    },
    messages=[...]
)
```

## Budget Recommendations

| Use Case         | Budget        | Notes                |
| ---------------- | ------------- | -------------------- |
| Simple reasoning | 1,024-4,000   | Start minimal        |
| Standard coding  | 8,000-16,000  | Good balance         |
| Complex analysis | 16,000-32,000 | Thorough reasoning   |
| Very complex     | 32,000+       | Use batch processing |

**Key insight:** Claude may not use the entire budget. Start lower and increase only if quality isn't sufficient.

## Interleaved Thinking (Beta)

Enables Claude to think between tool calls:

```python
# Add beta header
headers = {
    "anthropic-beta": "interleaved-thinking-2025-05-14"
}

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[...],
    extra_headers=headers
)
```

**Benefits:**

- Reason about tool results before next action
- Chain multiple tool calls with reasoning between
- Make more nuanced decisions based on intermediate results

## Guiding Thinking

Add to your prompt:

```
After receiving tool results, carefully reflect on their quality and determine optimal next steps before proceeding. Use your thinking to plan and iterate based on this new information, and then take the best next action.
```

## Thinking Block Preservation

**Claude Opus 4.5:** Automatically preserves thinking blocks across turns (unique feature).

**Other models:** Thinking blocks from previous turns are stripped. You must:

1. Pass thinking blocks back to API during tool use
2. Include complete, unmodified blocks
3. API will filter/use as needed

## Caching Considerations

**System prompts:** Remain cached when thinking parameters change.

**Messages:** Cache invalidated when:

- Thinking enabled/disabled changes
- Budget_tokens value changes
- Non-tool-result content added (strips previous thinking)

**Tool use:** Thinking blocks are cached during tool use loops.

**Tip:** Use 1-hour cache duration for long thinking sessions:

```python
cache_control={"type": "ephemeral", "ttl": "1h"}
```

## Summarized Thinking (Claude 4 models)

Claude 4 models return **summarized** thinking, not full output:

- You're billed for full thinking tokens
- Visible token count won't match billed count
- First few lines are more detailed (useful for prompt engineering)
- No extra charge for summarization process

**Claude Sonnet 3.7:** Returns full thinking output (no summarization).

## Handling Redacted Thinking

Safety systems may encrypt some thinking:

```python
for block in response.content:
    if block.type == "redacted_thinking":
        # Encrypted, still usable in subsequent requests
        # Pass back to API unmodified
        pass
    elif block.type == "thinking":
        # Normal thinking, visible
        print(block.thinking)
```

**Important:** Pass redacted blocks back unmodified to maintain reasoning continuity.

## Tool Use Limitations

- Only works with `tool_choice: {"type": "auto"}` (default) or `"none"`
- Cannot use `"any"` or specific tool forcing
- Cannot toggle thinking mid-turn during tool use loops

## Feature Compatibility

**Not compatible with:**

- `temperature` or `top_k` modifications
- Forced tool use
- Response prefilling

**Can set:** `top_p` between 0.95 and 1.0

## Performance Tips

1. **Budget optimization:** Start at minimum (1,024), increase incrementally
2. **Streaming:** Required when `max_tokens` > 21,333
3. **Batch processing:** Recommended for budgets > 32k tokens
4. **Monitor usage:** Track thinking tokens to optimize costs

## Context Window Impact

```
Effective context =
  (current input - previous thinking) +
  (thinking + encrypted thinking + text output)
```

Previous turn thinking blocks don't accumulate (stripped from context).

Exception: Tool use requires preserving thinking blocks.
