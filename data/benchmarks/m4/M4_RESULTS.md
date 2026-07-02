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
| ag600_dspy_glm_compiled | GLM-5.2 DSPyJudge MIPROv2-compiled | 0.654 | 0.869 | 0.746 | 0.75 | 1.6313 | **this run** |
| — | gpt-4o frontier zero-shot (M3, the ceiling) | 0.541 | 0.869 | 0.667 | 0.85 | 0.9114 | M3 (reused, the ceiling) |

## C7 gate — can a precision-tuned/compiled cheap judge approach frontier?

- Cheap-judge **precision**: zero-shot 0.264 (M3) -> precision-tuned DSPy baseline **0.671** (Δ +0.407); frontier ceiling 0.541 (gap -0.130).

## Verdict

1. **The lever is the precision-tuned DSPy *signature*, not compilation.** A hand-written,
   hard-negative signature (via `ChainOfThought`) lifts the cheap GLM-5.2 judge from
   F1 **0.409 → 0.757** (precision 0.264 → 0.671) and **beats the frontier gpt-4o
   zero-shot ceiling (0.667) at lower cost** ($0.68 vs $0.91 on this band) — **uncompiled**.
   The M3 precision collapse was a prompt/signature problem, and the DSPy seam fixes it.
2. **MIPROv2 compilation did NOT help** — it slightly regressed the uncompiled baseline
   (0.757 → **0.746**) for an extra **$1.63**. MIPRO scored 100% on its 40-example
   bootstrap metric but held-out pairwise-F1 went *down* — overfitting the compile metric
   on a small ER trainset. This **empirically confirms the OpenSanctions caveat**
   (`docs/research/20260701_er_seam_audit.md`: MIPRO ~1–2 F1, in-context examples
   neutral-to-negative) on our data. **The C7 gate's deeper answer: distillation/compilation
   is not the lever here — cut it; invest in the signature.**
3. **The plumbing works end-to-end paid:** honest per-pair cost is recorded (compile cell
   $1.6313 from 1,028,300 prompt + 218,123 completion tokens), the compiled `Resolver`
   artifact serializes to `data/benchmarks/m4/compiled_resolver/`, and the whole run
   stayed resumable + per-cell-committed under the $5 cap.

## Spend

- Cumulative committed spend: **$2.3100 / $5.00**.
  - `ag600_dspy_glm_compiled`: $1.6313
  - `ag600_dspy_glm_zeroshot`: $0.6788
