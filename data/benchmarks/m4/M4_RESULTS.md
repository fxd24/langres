# M4 Wave 2 — paid DSPy-scorer benchmark on Amazon-Google

The falsifiable M4 question: can a **precision-tuned / compiled cheap judge** (GLM-5.2) approach frontier quality (gpt-4o) at materially lower cost, versus its precision-collapsed zero-shot? Evaluated on the deterministic 600-pair Amazon-Google literature band (seed 0), pairwise-F1 (judged once) at the best-F1 grid threshold.

## Resolved OpenRouter model ids

| role | id | status |
| --- | --- | --- |
| cheap judge (primary) | `openrouter/z-ai/glm-5.2` | used (PROVEN in M3) |
| cheapest candidate | `openrouter/deepseek/deepseek-v4-flash` | resolved, not spent |
| teacher/judge candidate | `openrouter/moonshotai/kimi-k2.6` | resolved, not spent |
| frontier ceiling | `openrouter/openai/gpt-4o` | reused from M3 (not re-run) |

## Results (600-pair AG band, pairwise P/R/F1)

| cell | judge | P | R | F1 | thr | USD | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| — | glm-5.2 zero-shot (M3, DEFAULT_PROMPT) | 0.264 | 0.902 | 0.409 | 0.90 | 0.4729 | M3 (reused) |
| ag600_dspy_glm_zeroshot | GLM-5.2 DSPyJudge UNCOMPILED (precision-tuned signature) | 0.671 | 0.869 | 0.757 | 0.90 | 0.6788 | **this run** |
| — | gpt-4o frontier zero-shot (M3, the ceiling) | 0.541 | 0.869 | 0.667 | 0.85 | 0.9114 | M3 (reused, the ceiling) |

## C7 gate — can a precision-tuned/compiled cheap judge approach frontier?

- Cheap-judge **precision**: zero-shot 0.264 (M3) -> precision-tuned DSPy baseline **0.671** (Δ +0.407); frontier ceiling 0.541 (gap -0.130).

## Spend

- Cumulative committed spend: **$0.6788 / $5.00**.
  - `ag600_dspy_glm_zeroshot`: $0.6788
