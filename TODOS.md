# TODOS — deferred but real

Backlog items that are real work, deliberately deferred out of the current phase.
This is the durable home for "we decided not to do this now" so it doesn't only
live in a planning doc. Each item points to its tracking issue or milestone.

Detail lives in the M4.5/M5 plan, the ROADMAP, and the seam-audit epic
([#20](https://github.com/raisesquad/langres/issues/20)) with method-delta
backlog in [#55](https://github.com/raisesquad/langres/issues/55).

## Distribution & licensing

- **PyPI publish** — decide whether/when to publish a wheel; not published today
  (install from source via `uv sync`). Gated on a distribution decision. (M6 / David's call)
- **License choice (TBD)** — pick MIT vs Apache-2.0 vs stay-private; an unlicensed
  library is unadoptable regardless of DX. `pyproject.toml` says License TBD. (David's call)

## Method families & extensibility

- **Splink adapter as a Fellegi–Sunter feature-store row** — wrap Splink behind the
  seam instead of only the native FS-EM judge. ([#55](https://github.com/raisesquad/langres/issues/55))
- **Full C1 six-dataset replication portfolio** — only FZ/AG (+ Abt-Buy in M4.5) are
  in scope now; the rest of the benchmark portfolio is deferred. ([#55](https://github.com/raisesquad/langres/issues/55))
- **Public method-registration API** — a supported, documented way for third parties
  to register a new judge/method (beyond editing `methods.py:_make_module_builder`
  in-tree). ([#55](https://github.com/raisesquad/langres/issues/55))

## Hardening

- **PII / audit hardening** — redaction hooks, audit trail, prompt-injection mitigation
  beyond the current documented known-limitation. (M6 — pre-1.0, no external users yet)

## Big bets (earned-by-need)

- **Collective / graph resolution** — stateful, graph-native inference (UC7); out of
  the current pairwise+clustering architecture. (big-bet tier)
- **Active learning** — harvest `JudgementLog` verdicts + corrections into labels that
  retune thresholds / `fit()`; the flywheel's learning loop. (M5 flywheel groundwork; full loop later)

## Post-distribution / consumer-side

- **Hosted demo / notebooks / CLI** — deferred until after the distribution decision.
- **Human correction UX** — langres owns the `corrections.jsonl` contract + harvest
  only; the review-queue UI stays brainsquad-side.
