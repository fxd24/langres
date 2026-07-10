# Claude 4.x Prompt Snippets

Copy-paste snippets to fix specific behavioral issues. Apply only when needed—don't add all snippets by default.

## 1. Tool Overtriggering

**Problem:** Prompts designed for older models cause Opus 4.5 to call tools too frequently.

**Solution:** Replace aggressive language with moderate phrasing.

| Before                                      | After                                |
| ------------------------------------------- | ------------------------------------ |
| `CRITICAL: You MUST use this tool when...`  | `Use this tool when...`              |
| `ALWAYS call the search function before...` | `Call the search function before...` |
| `You are REQUIRED to...`                    | `You should...`                      |
| `NEVER skip this step`                      | `Don't skip this step`               |
| `It is VERY IMPORTANT that you...`          | `Please...`                          |

---

## 2. Over-Engineering Prevention

**Problem:** Opus 4.5 creates extra files, adds unnecessary abstractions, or builds unrequested flexibility.

**When to apply:** User reports unwanted files, excessive abstraction, or unrequested features.

**Snippet:**

```
- Avoid over-engineering. Only make changes that are directly requested or clearly necessary. Keep solutions simple and focused.
- Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability.
- Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use backwards-compatibility shims when you can just change the code.
- Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task. Reuse existing abstractions where possible and follow the DRY principle.
```

---

## 3. Code Exploration

**Problem:** Opus 4.5 proposes solutions without reading code or makes assumptions about unread files.

**When to apply:** User reports model proposing fixes without inspecting relevant code.

**Snippet:**

```
ALWAYS read and understand relevant files before proposing code edits. Do not speculate about code you have not inspected. If the user references a specific file/path, you MUST open and inspect it before explaining or proposing fixes. Be rigorous and persistent in searching code for key facts. Thoroughly review the style, conventions, and abstractions of the codebase before implementing new features or abstractions.
```

---

## 4. Frontend Design Quality

**Problem:** Default frontend outputs look generic ("AI slop" aesthetic).

**When to apply:** User requests improved frontend design quality or reports generic-looking outputs.

**Snippet (wrap in XML tags):**

```xml
<frontend_aesthetics>
You tend to converge toward generic, "on distribution" outputs. In frontend design, this creates what users call the "AI slop" aesthetic. Avoid this: make creative, distinctive frontends that surprise and delight.

Focus on:
- Typography: Choose fonts that are beautiful, unique, and interesting. Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics.
- Color & Theme: Commit to a cohesive aesthetic. Use CSS variables for consistency. Dominant colors with sharp accents outperform timid, evenly-distributed palettes. Draw from IDE themes and cultural aesthetics for inspiration.
- Motion: Use animations for effects and micro-interactions. Prioritize CSS-only solutions for HTML. Use Motion library for React when available. Focus on high-impact moments: one well-orchestrated page load with staggered reveals (animation-delay) creates more delight than scattered micro-interactions.
- Backgrounds: Create atmosphere and depth rather than defaulting to solid colors. Layer CSS gradients, use geometric patterns, or add contextual effects that match the overall aesthetic.

Avoid generic AI-generated aesthetics:
- Overused font families (Inter, Roboto, Arial, system fonts)
- Clichéd color schemes (particularly purple gradients on white backgrounds)
- Predictable layouts and component patterns
- Cookie-cutter design that lacks context-specific character

Interpret creatively and make unexpected choices that feel genuinely designed for the context. Vary between light and dark themes, different fonts, different aesthetics. You still tend to converge on common choices (Space Grotesk, for example) across generations. Avoid this: it is critical that you think outside the box!
</frontend_aesthetics>
```

---

## 5. Thinking Word Sensitivity

**Problem:** When extended thinking is **not** enabled (the default), Opus 4.5 is particularly sensitive to "think" and variants.

**When to apply:** User reports issues when extended thinking is disabled (no `thinking` parameter in API request).

**Solution:** Replace "think" with alternatives:

| Before            | After                       |
| ----------------- | --------------------------- |
| `think about`     | `consider`                  |
| `think through`   | `evaluate`                  |
| `I think`         | `I believe`                 |
| `think carefully` | `consider carefully`        |
| `thinking`        | `reasoning` / `considering` |

**Or enable extended thinking:**

```python
thinking={
    "type": "enabled",
    "budget_tokens": 10000
}
```

---

## 6. Test-Focused / Hard-Coding Prevention

**Problem:** Claude focuses too heavily on making tests pass at expense of general solutions, or uses workarounds.

**Snippet:**

```
Please write a high-quality, general-purpose solution using the standard tools available. Do not create helper scripts or workarounds to accomplish the task more efficiently. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please inform me rather than working around them.
```

---

## 7. Minimize Hallucinations in Agentic Coding

**Snippet:**

```xml
<investigate_before_answering>
Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.
</investigate_before_answering>
```

---

## 8. Reduce File Creation

**Problem:** Claude creates temporary files during iteration.

**Snippet:**

```
If you create any temporary new files, scripts, or helper files for iteration, clean up these files by removing them at the end of the task.
```

---

## 9. Verbosity Control (Get More Updates)

**Problem:** Claude is too concise, skips summaries after tool calls.

**Snippet:**

```
After completing a task that involves tool use, provide a quick summary of the work you've done.
```

---

## Integration Guidelines

When adding snippets to existing prompts:

1. **Use XML tags** to organize additions (e.g., `<coding_guidelines>`, `<tool_behavior>`)
2. **Match the style** of the existing prompt
3. **Place logically** - put coding snippets near other coding instructions
4. **Don't just append** - integrate thoughtfully
5. **Preserve existing content** - insert without removing functional content
