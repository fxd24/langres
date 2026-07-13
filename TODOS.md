# TODOS — deferred but real

Backlog items that are real work, deliberately deferred out of the current phase.
This is the durable home for "we decided not to do this now" so it doesn't only
live in a planning doc. Each item points to its tracking issue or milestone.

Detail lives in the M4.5/M5 plan, the ROADMAP, and the seam-audit epic
([#20](https://github.com/fxd24/langres/issues/20)) with method-delta
backlog in [#55](https://github.com/fxd24/langres/issues/55).

## Flywheel loop follow-ons (deferred from the closed-loop phase, 0.2.0)

- **Blocker-recall diagnostics (P2)** — the loop can only review/label pairs the blocker
  emitted; pairs it never proposed are silently unrecoverable. Surface a recall estimate
  / missed-pair diagnostic so users can tell a blocking gap from a judging gap. (separate
  seam from the judge loop)
- **`methods.py` → `CascadeJudge` migration (P3)** — the benchmark path still constructs
  the deprecated `CascadeModule`; migrate it to `CascadeJudge` and drop the deprecation
  shim. (`DeprecationWarning` already lands in 0.2.0.)
- **Label-Studio / Argilla export (P3)** — CSV round-trip covers v1 labeling; an export
  adapter for a real annotation tool is the next rung when a dataset outgrows a spreadsheet.
- **Stratified-audit knob (P3)** — `select_for_review`'s audit slice is a uniform random
  sample; a stratified variant (by score band / cluster size) would sharpen the
  confident-false-merge catch. (uniform sampling is already unbiased.)
- **`langres select` subcommand (P3)** — today the queue is created in Python
  (`select_for_review` → `ReviewQueue`); a CLI subcommand would close the last non-Python
  step. Deferred while the CLI surface (UC2) settles.
- **Update-aware `import-csv` de-dup (P3)** — `import-csv` appends every labeled row, so
  re-importing the same CSV duplicates `Correction`s. Non-corrupting (`harvest_labeled_pairs`
  is last-write-wins by pair) and append-always is *intentional* today — it lets a re-import
  **update** a label. A refined guard would skip rows whose label matches the
  already-recorded one while still letting changed labels through. (claude-review #79)
- **CLI/queue durability polish (P3)** — `ReviewQueue.write` truncates in place (fine: the
  queue is a regeneratable snapshot; source-of-truth durability lives on the append-only
  logs) — an atomic temp-then-`os.replace` would harden it. Also add a test that exercises
  the packaged `langres` console-script entry point (CLI tests call `main()` in-process, so
  a typo'd `[project.scripts]` path wouldn't be caught). (claude-review #79)

## Distribution & licensing

- **PyPI publish** — decide whether/when to publish a wheel; not published today
  (install from source via `uv sync`). Gated on a distribution decision. (M6 / David's call)
- **License choice (TBD)** — pick MIT vs Apache-2.0 vs stay-private; an unlicensed
  library is unadoptable regardless of DX. `pyproject.toml` says License TBD. (David's call)

## Method families & extensibility

- **Splink adapter as a Fellegi–Sunter feature-store row** — wrap Splink behind the
  seam instead of only the native FS-EM judge. ([#55](https://github.com/fxd24/langres/issues/55))
- **Full C1 six-dataset replication portfolio** — only FZ/AG (+ Abt-Buy in M4.5) are
  in scope now; the rest of the benchmark portfolio is deferred. ([#55](https://github.com/fxd24/langres/issues/55))
- **Public method-registration API** — a supported, documented way for third parties
  to register a new judge/method (beyond editing `methods.py:_make_module_builder`
  in-tree). ([#55](https://github.com/fxd24/langres/issues/55))

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
  only; the review-queue UI stays consumer-side.
