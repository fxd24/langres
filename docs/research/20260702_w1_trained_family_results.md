# W1.2 trained-family replication: FellegiSunterJudge + RandomForestJudge

**Date:** 2026-07-02
**Branch:** `feat/w1-trained-family`
**Cost:** $0 (CPU-only: MiniLM blocking + classical EM + sklearn RandomForest,
no LLM calls)
**Script:** `examples/research/w1_trained_family_race.py` (`uv run python
examples/research/w1_trained_family_race.py`)

## What this measures

W1.2 shipped the first two members of the "trained family" — judges that need
a `fit()` step before they can score (`docs/EXPERIMENTS.md` § "The fit seam"):

- **`FellegiSunterJudge`** (`fellegi_sunter`) — classical Fellegi-Sunter,
  fit via **EM with no labels** (`UnsupervisedFitMixin.fit_unlabeled`). The
  first "learn with no labels" proof in langres.
- **`RandomForestJudge`** (`random_forest`) — a Magellan-style `sklearn.RandomForestClassifier`
  over `ComparisonVector.similarities`, fit **with labels**
  (`SupervisedFitMixin.fit`).

Both ride the class-default hyperparameters exactly as wired into
`methods.py:_make_module_builder` — no dataset-specific tuning. For each of
the three $0 replication datasets (CEO #9):

1. `benchmark.split(seed=0)` — a leakage-free, whole-cluster train/test split.
2. Fit on the train split's own blocked-and-compared candidates: FS via
   `resolver.fit(train_records)` (no labels — u estimated from random pairs of
   entities the fit stream saw, m/prior via log-space EM); RF via
   `resolver.fit(train_records, labels=derived_from_gold)`.
3. Grade the TEST split's blocked candidates **once** via
   `evaluate_judge_on_candidates` (best-F1 threshold over the dataset's grid) —
   the same judged-once pairwise-F1 surface the M4 DSPy probe used, so numbers
   are comparable across judges.

This is **pairwise F1 on the blocked candidate band**, not BCubed post-cluster
— directly comparable to the literature's pair-F1 tables (no blocking-recall
ceiling, no clustering amplification).

## Results

| dataset | method | precision | recall | f1 | n_candidates (test) | best_threshold |
| --- | --- | --- | --- | --- | --- | --- |
| fodors_zagat | fellegi_sunter | 0.218 | 0.970 | **0.356** | 1,023 | 0.60 |
| fodors_zagat | random_forest | 0.971 | 1.000 | **0.985** | 1,023 | 0.40 |
| amazon_google | fellegi_sunter | 0.057 | 0.900 | **0.107** | 48,854 | 0.70 |
| amazon_google | random_forest | 0.306 | 0.176 | **0.224** | 48,854 | 0.30 |
| abt_buy | fellegi_sunter | 0.117 | 0.897 | **0.207** | 9,004 | 0.60 |
| abt_buy | random_forest | 0.457 | 0.329 | **0.383** | 9,004 | 0.30 |

## Versus published bands (`docs/research/20260701_er_seam_audit.md`)

| Method family | Published band | langres result | In the band? |
| --- | --- | --- | --- |
| FS / Splink | "High-0.9s on clean multi-field identity data; weak on single-text-blob (AG)" | FZ 0.356 (clean/multi-field); AG 0.107 (single-text-blob) | **No on FZ** (expected high-0.9s, got low-0.4s); directionally right that AG is FS's worst case, but the absolute AG number is also below the ZeroER cousin's 0.48 |
| Magellan RF | AG pair-F1 49.1 (i.e. ≈0.491); structured-dataset avg 88.8 | AG 0.224; FZ 0.985 (FZ is langres's "structured" analog) | **FZ is in the structured band** (0.985 vs 0.888 avg — actually exceeds it, expected since FZ is near-saturated); **AG misses** (0.224 vs 0.491) |
| ZeroER (FS-EM cousin) | AG 0.48; FZ 1.00; avg 0.76 ≈ supervised RF | FS: AG 0.107, FZ 0.356 | **No** — both datasets well below ZeroER's numbers |
| Abt-Buy | No RF/FS band in the audit doc (Ditto ~0.89 is a fine-tuned transformer, not comparable) | FS 0.207; RF 0.383 | No established classical-family band to compare against — reported as a new textual-hard data point |

**Honest verdict: RF clears the band on the easy/structured dataset (FZ) but
misses on the harder AG; FS misses its literature band on every dataset.**
This is reported as-is, not tuned to close the gap — see the diagnostic below
for *why*, which is itself the most useful finding of this replication.

## Diagnostic: FellegiSunterJudge's EM does not converge at real-benchmark scale

Every one of the three fits above hit `FellegiSunterJudge`'s **non-convergence
fallback** (`FellegiSunterJudge EM did not converge within max_em_iter=20`), so
the reported FS scores use the *safe initial priors* (`prior=0.5`, `m=0.9` for
every feature) rather than data-learned m-probabilities. `max_em_iter=20` is
the class default — validated in
`tests/core/judges/test_fellegi_sunter_judge.py` against small synthetic
fixtures (which the unit test itself fits with `max_em_iter=50`, not the
default) — not a promise of convergence on hundreds/thousands of real
candidate patterns.

**Widening the iteration budget does not fix it — and on Fodors-Zagat makes it
worse.** Re-running FS on Fodors-Zagat with `max_em_iter=200, tol=1e-6` (a
quick diagnostic, not shipped in the example script) reaches `converged=True`
with real, per-feature-differentiated m-probabilities
(`m_prob={'name': 0.168, 'addr': 0.528, 'city': 0.955, 'phone': 0.801,
'type': 0.356}`, `prior=0.817`) — but pairwise F1 **drops** to 0.099 (P=0.052,
R=1.000): the "converged" fit classifies nearly every blocked candidate as a
match. This is a known failure mode of unsupervised 2-component
record-linkage EM on class-imbalanced, blocked candidate data (the blocked
band is match-enriched relative to the full corpus, and unlike Splink's
production implementation, this EM has no possible-match band, no
value-frequency (TF) adjustment, and no multi-start/init diversity to escape a
degenerate "everything agrees enough to match" local optimum). The
max-iteration safety fallback is doing its documented job (never returning a
diverged partial result) — it is the honest *default* result that is weak,
not a design flaw in having a fallback at all.

**Implication for the roadmap:** this is exactly the C3/S2 gap the seam audit
already flagged (`docs/research/20260701_er_seam_audit.md` rows C3/S2) —
value-frequency-aware FS (Splink-style per-value TF adjustment) and/or a
possible-match band are the standard production fixes, not deeper iteration.
Tracked, not fixed, in this branch (out of scope for W1.2's "does the fit seam
work end-to-end" proof).

### Fix: the fallback itself DID have a real bug (Codex P2, fixed)

PR review caught a genuine correctness bug in the fallback path above: it
returned the raw `init_m=0.9` for every feature **without** applying the same
`m >= u` guard the converged path always gets from `_m_step`. For any feature
whose random-pair `u_prob` exceeds the fixed 0.9 — plausible for a low-entropy
field (e.g. a near-constant category column) — `forward()`'s
`log(m_prob[name] / u_prob[name])` goes negative, so *agreeing* on that
feature would lower the match score instead of raising it (inverted
evidence). Since the fallback fires on every dataset in this replication (see
above), this was a live path, not a corner case.

Fixed by clamping the fallback's `m` against `u_prob` the same way `_m_step`
does (`tests/core/judges/test_fellegi_sunter_judge.py::TestEMFallbackGuard`
has the regression tests — both fail against the pre-fix code with a
constructed low-entropy feature). **Re-running this replication after the fix
produced byte-identical numbers to the table above**: none of the three real
datasets' features happen to have a `u_prob` above 0.9 (checked directly —
max observed `u_prob` was 0.376 on Amazon-Google's `price` field), so the
guard was a latent bug on these particular benchmarks, not one that changed
the reported precision/recall/F1 here. It is a correctness fix for the
general case (any schema, any feature distribution), not a benchmark-number
change.

## What this replication does prove

- **The fit seam works end-to-end for both directions.** `resolver.fit(...)`
  correctly dispatches to `fit_unlabeled` (FS, no labels) and `fit` (RF,
  labels required) per `langres.core.fit`'s runtime-checkable protocols, on
  three different real datasets, with zero code changes per dataset.
- **RF is a credible supervised baseline** — 0.985 F1 on the easy/structured
  dataset (beating the Magellan structured-average band), reasonable-but-band-
  missing on the two harder textual datasets, consistent with RF's own
  reputation as sensitive to weak/noisy feature vectors rather than a text
  specialist.
- **FS's honest weakness is now measured, not assumed** — the EM
  non-convergence-at-scale finding is new, actionable signal for any future
  FS investment (point straight at value-frequency adjustment / possible-match
  bands, not raw iteration budget).

## Reproduce

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run python examples/research/w1_trained_family_race.py
```

No network calls, no API key, ~2 minutes on CPU (three MiniLM embedding passes
+ two EM fits + two RandomForest fits).
