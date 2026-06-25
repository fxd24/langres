# Claude 4.5 Model Comparison

## Quick Selection Guide

| Need                 | Model          | Why                                                    |
| -------------------- | -------------- | ------------------------------------------------------ |
| Maximum intelligence | **Opus 4.5**   | Best reasoning, handles ambiguity, fewer output tokens |
| Coding & Agents      | **Sonnet 4.5** | SWE-bench leader, 30+ hour focus, parallel tools       |
| Speed & Cost         | **Haiku 4.5**  | 2x faster, 1/3 cost, near-frontier quality             |
| Sub-agents           | **Haiku 4.5**  | Fast + intelligent for multi-agent systems             |
| Real-time apps       | **Haiku 4.5**  | Low latency with high quality                          |

## Detailed Comparison

### Intelligence & Capabilities

| Feature            | Opus 4.5                     | Sonnet 4.5    | Haiku 4.5                  |
| ------------------ | ---------------------------- | ------------- | -------------------------- |
| Intelligence tier  | Maximum                      | Best-in-class | Near-frontier (≈ Sonnet 4) |
| Complex reasoning  | Excellent                    | Excellent     | Very good                  |
| Ambiguity handling | Best (less prompting needed) | Good          | Good                       |
| First-try success  | Highest                      | High          | Good                       |

### Coding Performance

| Feature              | Opus 4.5  | Sonnet 4.5       | Haiku 4.5 |
| -------------------- | --------- | ---------------- | --------- |
| SWE-bench Verified   | 80.9%     | State-of-the-art | Strong    |
| Multi-file changes   | Excellent | Excellent        | Good      |
| Security engineering | Excellent | Excellent        | Good      |
| Planning & design    | Excellent | Excellent        | Good      |

### Agent Capabilities

| Feature                | Opus 4.5  | Sonnet 4.5                   | Haiku 4.5 |
| ---------------------- | --------- | ---------------------------- | --------- |
| Long-running tasks     | Excellent | 30+ hours focus              | Good      |
| Parallel tool calling  | Good      | Aggressive (can bottleneck!) | Good      |
| Context awareness      | Yes       | Yes                          | Yes       |
| Subagent orchestration | Excellent | Excellent (proactive)        | Good      |

### Unique Features

| Feature               | Opus 4.5      | Sonnet 4.5 | Haiku 4.5               |
| --------------------- | ------------- | ---------- | ----------------------- |
| Effort parameter      | **Yes**       | No         | No                      |
| Thinking preservation | **Automatic** | Manual     | Manual                  |
| Computer use zoom     | **Yes**       | No         | No                      |
| Extended thinking     | Yes           | Yes        | **First Haiku with it** |
| 1M context (beta)     | No            | **Yes**    | No                      |

### Performance & Cost

| Metric            | Opus 4.5                     | Sonnet 4.5     | Haiku 4.5      |
| ----------------- | ---------------------------- | -------------- | -------------- |
| Input tokens      | $5/M                         | $3/M           | $1/M           |
| Output tokens     | $25/M                        | $15/M          | $5/M           |
| Speed             | Moderate                     | Fast           | **2x+ faster** |
| Output efficiency | 76% fewer tokens than Sonnet | Baseline       | Fast output    |
| Context window    | 200K                         | 200K (1M beta) | 200K           |
| Max output        | 64K                          | 64K            | 64K            |

## Prompting Differences

### Opus 4.5

**Needs less hand-holding:**

- Handles ambiguity well without explicit guidance
- More likely to "figure it out" with less detailed prompts
- Uses 76% fewer output tokens for similar tasks

**Watch out for:**

- Over-engineering (add minimalism guidance)
- Tool overtriggering (soften aggressive language)
- Conservative code exploration (add explicit read-first instructions)
- "Think" word sensitivity when thinking is disabled

### Sonnet 4.5

**Optimal for agents:**

- Aggressive parallel tool use (may need to dial back)
- Excellent at long autonomous sessions
- More concise communication (may skip summaries)

**Prompting notes:**

- Be explicit about actions vs suggestions
- May need more detailed prompts than Opus for intricate tasks
- Benefits significantly from extended thinking for coding

### Haiku 4.5

**Cost-effective intelligence:**

- Near Sonnet 4 quality at lower cost
- Great for high-volume processing
- Ideal as sub-agent in multi-agent systems

**Prompting notes:**

- May benefit from slightly more detailed prompts
- Same core 4.x principles apply
- Enable extended thinking for complex tasks

## When to Switch Models

**Use Opus 4.5 when:**

- Task is highly complex or specialized
- You need maximum first-try success
- Working on professional software engineering
- Ambiguity is high and you can't provide detailed prompts

**Use Sonnet 4.5 when:**

- Building coding agents or autonomous systems
- Tasks run for extended periods
- Need balance of intelligence and cost
- High volume of moderately complex tasks

**Use Haiku 4.5 when:**

- Real-time response is critical
- Processing high volumes
- Budget is constrained
- Using as sub-agent/helper
- Task complexity is moderate

## Extended Thinking Recommendations

| Model      | Default Recommendation              |
| ---------- | ----------------------------------- |
| Opus 4.5   | Enable for complex tasks            |
| Sonnet 4.5 | **Strongly recommended for coding** |
| Haiku 4.5  | Enable for complex problem-solving  |

**Trade-off:** Extended thinking impacts prompt caching efficiency.
