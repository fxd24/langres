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

## Method families & extensibility

- **Splink adapter as a Fellegi–Sunter feature-store row** — wrap Splink behind the
  seam instead of only the native FS-EM judge. ([#55](https://github.com/fxd24/langres/issues/55))
- **Full C1 six-dataset replication portfolio** — only FZ/AG (+ Abt-Buy in M4.5) are
  in scope now; the rest of the benchmark portfolio is deferred. ([#55](https://github.com/fxd24/langres/issues/55))
- **Public method-registration API** — the in-tree half shipped in the v0.3
  model-identity slice: one `MethodSpec` registry (`core/method_registry.py`,
  `register_method`/`get_method`) that all three dispatch sites resolve
  through, with `/` reserved for `author/method` namespacing. Remaining for
  v0.4: the `langres.methods` entry-points group so a *pip-installed* package
  can register a method without editing langres.
  ([#55](https://github.com/fxd24/langres/issues/55))

## Hardening

- **PII / audit hardening** — redaction hooks, audit trail, prompt-injection mitigation
  beyond the current documented known-limitation. (M6 — pre-1.0, no external users yet)

## Deferred cleanup — surveyed, not executed

Three cleanups were scoped during the docs truth-pass and deliberately **not**
executed: each has an unbounded blast radius and no measurable exit criterion,
so they need their own change with its own tests. The counts below were
**measured** on `main` @ `94df55e`, not estimated — start the follow-up from
this evidence rather than re-surveying.

### 1. The 21 `# TEMPORARY: deleted by the W2 sweep` back-compat shims

Twenty under `src/langres/core/`, one under `src/langres/clients/`. Each is a
pure re-export whose docstring says the W2 sweep deletes it:

`core/analysis.py`, `core/anchor_store.py`, `core/benchmark.py`,
`core/calibration.py`, `core/canonicalizer.py`, `core/debugging.py`,
`core/diagnostics.py`, `core/finetune.py`, `core/fit_report.py`,
`core/harvest.py`, `core/judgement_log.py`, `core/methods_calibrate.py`,
`core/methods_prompt.py`, `core/metrics.py`, `core/review.py`, `core/runs.py`,
`core/trackers/__init__.py`, `core/trackers/mlflow_tracker.py`,
`core/trackers/trackio_tracker.py`, `core/trackers/wandb_tracker.py`,
`clients/tracking.py`

Two things to know before starting:

- **Ownership is unconfirmed.** These shims were assumed to be covered by epic
  [#193](https://github.com/fxd24/langres/issues/193). They are not, as far as
  can be verified: #193's body never mentions shims, back-compat or the W2
  sweep; no open issue matches a search for "shim" or "W2 sweep"; and none of
  the 21 files carries an issue reference. Treat the deletion as **untracked**
  and open an issue for it, or confirm the scope on #193 first.
- **The docs still teach three of these paths**, so deleting the files breaks
  published snippets verbatim. At minimum: `docs/GETTING_STARTED.md` imports
  from `langres.core.harvest`; `docs/TUTORIAL_YOUR_OWN_CSV.md` names
  `langres.core.harvest`; `docs/TECHNICAL_OVERVIEW.md` imports from
  `langres.core.canonicalizer` and `langres.core.review` and cites
  `langres.core.harvest.align_pairs`. The canonical homes
  (`langres.curation.*`, or the root `langres` re-exports) already exist, so
  the docs can be repointed *before* the deletion, as a safe standalone step.

### 2. The 71 `examples/` scripts

`git ls-files` counts **71** tracked `.py` files under `examples/` (21 at the
top level, the rest under `examples/research/` and `examples/data/`).

**19 carry a closed-milestone prefix** and are the historical one-offs:

`m1_bootstrap_fodors_zagat`, `m2_walking_skeleton_fodors_zagat`, `m3_race`,
`m3_regrade_subsample`, `m3_report`, `m3_zero_spend_race`, `m4_calibration`,
`m4_dspy_judge`, `m4_experiment_loop`, `m4_race`, `phase1_blocker_optimization`,
`phase1_llm_placement`, `phase1_rf_floor`, `phase2_full_pipeline`,
`w1_blocking_algebra`, `w1_select_judge_benchmark`, `w1_trained_family_race`,
`w2_person_benchmark`, `w3_paid_smoke` (all under `examples/research/`).

**None of the 71 is broken.** Every `langres` import in every example was
resolved against a `--all-extras` install — module imports and `from ... import
<symbol>` names alike — and all 71 resolved cleanly. So "prune the ones that no
longer work" is not the argument for this cleanup; "prune the ones that no
longer earn their place" is, and that is a judgement call about which
historical runs stay reproducible, not a correctness fix.

### 3. `src/langres/bootstrap/` — not actually a repo item

Reported as an empty package to remove. It is **not in git**: `git ls-files
'src/langres/bootstrap*'` returns zero entries, and the directory does not
exist in a fresh checkout. What exists is a stale `__pycache__` left in one
long-lived working copy from before the package was dissolved into
`langres.curation`. Nothing to delete in the repo; a local `git clean` in that
checkout is the whole fix.

## Big bets (earned-by-need)

- **Collective / graph resolution** — stateful, graph-native inference (UC7); out of
  the current pairwise+clustering architecture. (big-bet tier)
- **Active learning** — harvest `JudgementLog` verdicts + corrections into labels that
  retune thresholds / `fit()`; the flywheel's learning loop. (M5 flywheel groundwork; full loop later)

## Post-distribution / consumer-side

- **Hosted demo / notebooks / CLI** — deferred until after the distribution decision.
- **Human correction UX** — langres owns the `corrections.jsonl` contract + harvest
  only; the review-queue UI stays in the consuming application.
