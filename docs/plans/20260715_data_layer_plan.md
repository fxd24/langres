# Data Layer — Composable Profile Report + Mining Seam (design & plan)

> **Status:** Design + execution plan (2026-07-15). Turns the #86 data-prep survey
> (`docs/research/20260707_data_prep_hard_case_mining_survey.md`) and the
> training-surface design (`docs/research/20260714_training_surface_design.md` §5.1)
> into a buildable, **composable** data-layer arc. Precedes any code — for review.

---

## 0. Why now, and what we're building

We are about to start experiments (the autoresearch loop #145; the replication /
training-loop targets T1 silver→student, **T2 AnyMatch generalist**). Before that we
want the **instrumentation the pipeline is missing on the *data* side**. Two confirmed
gaps (exploration 2026-07-15):

1. **No dataset profiler.** Every report today is keyed to *pipeline output* or to
   *gold vs. prediction* (`CandidateInspectionReport`, `ScoreInspectionReport`,
   `ClusterInspectionReport`, `EvalReport`, the blocker metrics). Nothing takes raw
   records / a gold pair-set / an embedding matrix and profiles **the data itself** —
   class balance, per-field stats, gold cluster-size distribution, embedding
   distribution. This is the "QuantStats-style tearsheet, but for ER data" idea.
2. **No "mine the training set first" step.** The training surface landed
   (`Resolver.fit(pairs=…, method=QLoRA|MIPRO|Platt)`, `finetune()`), but it *assumes
   labeled pairs are handed in*. The AnyMatch-style miners were designed with the
   maintainer (§5.1 of the training-surface doc) but never built.

**Decision (maintainer, 2026-07-15):** build these as **one coherent data-layer arc**
(shared substrate + the first two miners + the tearsheet together), with the **full
metric battery including embeddings**, and — the load-bearing constraint —
**flexible / composable / graceful**, not a monolith.

---

## 1. Design principles — the composability contract (load-bearing)

This is the part that must be right. The report is **not** one function that computes
everything and errors if a piece is absent. It is a set of **independent metric blocks
you compose**.

1. **Independent profiler blocks (SRP).** Each metric family — label structure, field
   stats, separability, embeddings, mining-readiness — is a self-contained profiler
   producing a self-contained section. One responsibility each; no cross-talk.
2. **Compose the subset you want.** "Sometimes I only care about a few metrics" is a
   first-class path: build only the sections you want and hand them to the report.
   There is no all-or-nothing entry point.
3. **Graceful degradation — never block.** A missing input (no embeddings, no
   judgement log, no gold) means *that section is simply absent* — never an exception,
   never a blocked report. A profiler validates its own inputs and no-ops when they're
   missing.
4. **Multiple embeddings are first-class.** You pass N embedding models
   (`{model_name: matrix}`) and get **one section per model** plus a **comparison
   panel** (separability side-by-side). Adding an embedder is passing one more entry.
5. **Reuse the `$0` tearsheet infra.** Inline-SVG (`core/_svg.py`), self-contained
   HTML, no CDN/matplotlib, light/dark — the exact pattern `EvalReport` already proves.

The miners obey the same ethos: **plain composable functions** producing `LabeledPair`s
(as already decided in §5.1), routed by the consumer — not a monolithic strategy engine.

---

## 2. Architecture — the profiler seam

### 2.1 `ProfileSection` — the composable unit

A small Pydantic base every metric block subclasses. This is the seam that makes the
report a *bag of sections* instead of a monolith.

```python
# core/data_profile.py  (leaf module — same layering rules as eval_report.py)
class ProfileSection(BaseModel):                 # frozen
    title: str
    def to_dict(self) -> dict: ...               # model_dump
    def to_markdown(self) -> str: ...            # this block's markdown
    @property
    def summary(self) -> dict: ...               # numeric headline
    def panels(self) -> list[str]: ...           # inline-SVG/HTML <section>s (reuses _svg)
```

Concrete sections (each `ProfileSection`):

| Section | Built from | Reports |
|---|---|---|
| `LabelStructureSection` | `gold_clusters`, `gold_pairs` (+ optional splits) | positive prevalence (pos:neg), cluster-size distribution, singleton rate, linkage-vs-dedup shape, train/valid/test balance + **leakage check** |
| `CorpusFieldSection` | `corpus` (records `.model_dump()`), per source | per-field null/missing rate, cardinality/entropy, value- & token-length distribution, source lopsidedness |
| `SeparabilitySection` | a pair-set + a **pluggable signal** (string sim default; or an embedding cosine) | score histogram **positives vs negatives**, overlap/AUC — "how hard is this dataset" |
| `EmbeddingSection` (one **per model**) | an embedding matrix + `model_name` (+ `gold_pairs` for the labeled view) | model name/dim (provenance), vector-norm dist, **cosine positives-vs-negatives**, recall@k of true match in embedding space |
| `EmbeddingComparisonSection` | ≥2 `EmbeddingSection`s | separability by model side-by-side — "which embedder suits this data" |
| `MiningReadinessSection` | a mined `LabeledPair` set (+ optional judgement log) | hard-pos/neg counts, class balance after mining, difficulty histogram (EL2N \|1−p\|), **label-noise estimate** |

### 2.2 `DataProfileReport` — a pure container over sections

```python
class DataProfileReport(BaseModel):              # frozen
    sections: list[ProfileSection]
    def to_html(self, *, title=...) -> str: ...  # renders present sections' panels only
    def to_markdown(self) -> str: ...
    def to_dict(self) -> dict: ...
    @property
    def summary(self) -> dict: ...
```

The report **holds whatever sections you gave it** and renders exactly those. It does
not know how to compute anything — composition lives at the edges. Adding a new metric
family = a new `ProfileSection` subclass + a profiler function; the report is untouched
(open/closed).

### 2.3 The two ways to build it

**(a) Compose explicitly — the "only the metrics I want" path:**

```python
label = profile_label_structure(gold_clusters, gold_pairs)
fields = profile_fields(corpus)
emb    = profile_embeddings({"all-MiniLM-L6-v2": m1, "bge-large-en": m2},
                            gold_pairs=gold_pairs)   # -> [Section, Section, ComparisonSection]
DataProfileReport([label, fields, *emb]).to_html()   # renders only these
```

**(b) Convenience with sensible defaults — everything optional, nothing blocks:**

```python
DataProfileReport.from_benchmark(
    get_benchmark("abt_buy"),
    embeddings={"all-MiniLM-L6-v2": m1},   # OMIT -> no embedding section, no error
    include={"labels", "fields", "separability", "embeddings"},  # optional subset selector
)
```

Omitting `embeddings=` yields a report without the embedding sections — *no exception*.
`include=` narrows the default set. Both paths return the same `DataProfileReport`.

### 2.4 Shared HTML scaffold (non-invasive)

`EvalReport` keeps its `_CSS` + `_panel_*` idiom *inside* the class. To share without
churning it, lift the ~20-line CSS + a `section(title, body)` helper into a tiny
`core/_report_html.py`; `DataProfileReport` uses it, `EvalReport` is **left untouched**
(optional later migration, not in this arc — surgical-changes rule). `core/_svg.py`
chart primitives are already fully shareable as-is.

---

## 3. The metric battery, grounded in real data sources

Every metric maps to data already in the codebase — nothing needs new pipeline plumbing:

- **Label structure** ← `gold_clusters: list[set[str]]` + `gold_pairs: set[frozenset[str]]`
  (the `Benchmark.load()` contract, `core/benchmark.py:396`); splits from
  `_deepmatcher_loader.load_pair_splits` / `fixed_split_pair_benchmark`. Leakage check =
  entity-disjointness across splits (same union-find as `harvest._entity_disjoint_split`).
- **Field stats** ← `record.model_dump()` over the corpus; "source" from the record
  (e.g. `AbtBuySchema.source`). Null = the same emptiness rule `StringComparator` uses
  (`core/comparator.py:247`).
- **Separability** ← any pair-set + a pluggable per-pair signal: default a cheap
  `StringComparator` similarity (core-dep only), or an embedding cosine when a matrix is
  present. Reuses the pos/non-pos split logic already in `EvalReport._histogram`.
- **Embeddings** ← a caller-supplied matrix + `model_name` (numpy only). We **consume**
  vectors; we do not force generation (see §5). `embeddings.py` already knows
  `model_name`/`embedding_dim`.
- **Mining readiness** ← a mined `LabeledPair` set (§4) + optional `JudgementLog` rows
  for the EL2N difficulty view.

## 4. The mining seam (first two miners + the required rail)

Plain functions producing `LabeledPair` (`core/harvest.py:113`), consistent with §5.1.
This arc builds the **first two + the denoise rail**; the rest are staged (§7).

- `mine_misclassified(candidates, labels, *, classifier=…, cap=…) -> list[LabeledPair]`
  — AnyMatch **hard-positive** mining: train a fast tabular classifier on the
  comparison vectors, keep the **positives it misclassifies** (the hard boundary
  cases). **Reuse the existing sklearn `[trained]` extra** (`RandomForestMatcher`
  already depends on it) — no AutoGluon; the paper's signal is just "positives a quick
  classifier gets wrong."
- `sample_negatives(dataset, *, ratio=2.0, seed=0) -> list[LabeledPair]` — AnyMatch's
  2:1 **random** negatives. AnyMatch is deliberately hard-*positive* + *random*-negative;
  we keep "mine hard positives" and "sample/mine negatives" as **separate** functions
  (opposite failure modes — survey §1.1).
- `denoise_pairs(pairs, *, method="confident_learning") -> tuple[clean, flagged]` — the
  **required label-noise rail** (survey §4): any "model got it wrong" signal can't tell
  *genuinely hard* from *mislabeled*. Ship a light built-in (cross-validated
  confident-joint using the sklearn we already have); **Cleanlab optional** later, not a
  new hard dep now.

The report's `MiningReadinessSection` then profiles the miner output — so the tearsheet
and the miners close a loop (mine → profile the mined set → decide) and the report is
never rework.

## 5. Dependencies to front-load

| Need | Dep | Status | Where |
|---|---|---|---|
| Numeric profiling (norms, histograms, cosine) | numpy | **core** (already) | all sections |
| Profile a *given* embedding matrix | numpy only | **core** | `EmbeddingSection` consumes vectors |
| *Generate* embeddings (optional convenience) | sentence-transformers | `[semantic]`, **lazy** | optional `from_embedder` helper only |
| Hard-positive classifier + confident-learning | scikit-learn | `[trained]` (already) | `mine_misclassified`, `denoise_pairs` |
| Packaged label-noise (later) | cleanlab | **new, optional, deferred** | Wave 3+ if the built-in isn't enough |

Import-budget rule (`tests/test_import_budget.py`): `core/data_profile.py` is a **leaf**
that imports only `core.{metrics,benchmark,models,_svg,_report_html}` + numpy. Profiling
a matrix must **not** pull sentence-transformers; only the optional *generator* helper
does, behind the lazy `[semantic]` gate.

## 6. Testing & layering

- Leaf module; nothing in `reports.py`/`module.py` imports it; import-budget test guards
  the bare-`import langres` budget.
- Tiered coverage: this is `core` contract → **95–100%**. Each profiler function and
  section gets unit tests incl. the **graceful-degradation cases** (missing embeddings,
  empty gold, single-source corpus → section absent, *no raise*) and the
  **multi-embedding** case (2 models → 2 sections + comparison).

## 7. Build sequence (each wave in its own worktree)

- **Wave 1 — seam + tabular sections.** `ProfileSection` base, `DataProfileReport`
  container, `core/_report_html.py` scaffold, `LabelStructureSection`,
  `CorpusFieldSection`, `.to_html()` shell. *Exit:* profile a benchmark's labels+fields
  to a self-contained HTML tearsheet at `$0`; composing a subset works; omitting inputs
  never raises.
- **Wave 2 — separability + embeddings (multi-model).** `SeparabilitySection`,
  `EmbeddingSection`, `EmbeddingComparisonSection`, `profile_embeddings({...})`, optional
  `from_embedder` behind `[semantic]`. *Exit:* pass 2 embedders → 2 sections + comparison
  in the tearsheet; no embedder → report still renders.
- **Wave 3 — mining seam + readiness.** `mine_misclassified`, `sample_negatives`,
  `denoise_pairs`, `MiningReadinessSection`. *Exit:* mine hard positives + random
  negatives on one benchmark, denoise, profile the mined set in the tearsheet.
- **Stage 3 (north-star, separate epic):** `attribute_examples` + `flipped` + `ColVal`
  schema-free serializer → `TrainedLMMatcher` → the **leave-one-out generalist harness**
  (pool mined pairs across the 10-dataset registry, hold one out, zero-shot eval) — the
  AnyMatch "one ER model across all benchmarks" payoff (F1 81.96 reference). Consumes
  Waves 1–3; scoped in its own plan.

## 8. Public surface

`DataProfileReport` + the profiler functions export via `langres.eval` alongside
`EvalReport` (data-and-eval reporting live together), following the import-light eager
rule. The miners export via the training/`core` surface next to `harvest`/`align_pairs`.
(Alternative: a dedicated `langres.profile` namespace — flagged as an open decision.)

## 9. Open design decisions (for maintainer review)

1. **Public namespace:** `langres.eval.DataProfileReport` (report lives with `EvalReport`)
   vs. a dedicated `langres.profile` / `langres.data` facade. *Rec: reuse `langres.eval`*
   — fewer surfaces, and it's the reporting home already.
2. **Label-noise rail dep:** built-in confident-learning (sklearn, no new dep) now, with
   Cleanlab as an optional extra later. *Rec: built-in first.*
3. **Separability default signal:** cheap `StringComparator` sim (core-only, always
   available) as the default "how hard" signal, embeddings when supplied. *Rec: yes —
   keeps the section dependency-free by default.*
