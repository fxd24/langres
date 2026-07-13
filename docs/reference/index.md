# API Reference

The public surface of langres, generated from docstrings. It is layered:

- **[`langres`](langres.md)** — the two-verb DX layer (`link` / `dedupe`) plus
  the root exports a front-door user touches: `Resolver`, `JudgementLog`,
  `ReviewQueue` / `select_for_review`, and the exceptions to catch.
- **[`langres.eval`](eval.md)** — the curated evaluation facade: `evaluate`,
  benchmark discovery (`list_benchmarks` / `get_benchmark`), and the
  entity-resolution metrics.
- **`langres.core`** — the low-level primitives custom pipelines compose:
    - [Resolver](resolver.md) — declarative pipeline: `from_schema` /
      `resolve` / `save` / `load`.
    - [Blockers](blockers.md) — candidate generation (`AllPairsBlocker`,
      `VectorBlocker`, ...).
    - [Comparators](comparators.md) — field-wise similarity features.
    - [Judges](judges.md) — pairwise match/no-match scoring (`Module` ABC,
      `LLMJudge`, `CascadeJudge`, ...).
    - [Clusterers](clusterers.md) — judgements to entity clusters.
    - [Flywheel](flywheel.md) — `JudgementLog`, review selection, correction
      harvesting, threshold calibration.
    - [EvalReport](eval-report.md) — the self-contained HTML evaluation
      tearsheet.

!!! note "Lazy imports and optional extras"
    Heavy optional dependencies (torch, litellm, faiss, scikit-learn, ranx)
    resolve lazily: `import langres` never pulls them in. Components that
    need them (e.g. `VectorBlocker`, `LLMJudge`, `RandomForestJudge`) require
    the matching extra — `langres[semantic]`, `[llm]`, `[trained]`, `[eval]`.
