# Data Layer — Composable Profile Report + Mining Seam (design & plan)

> **Status:** Design + execution plan (2026-07-15). Turns the #86 data-prep survey
> (`docs/research/20260707_data_prep_hard_case_mining_survey.md`) and the
> training-surface design (`docs/research/20260714_training_surface_design.md` §5.1)
> into a buildable, **composable** data-layer arc.
>
> **Build status (2026-07-15):**
> - **PR-1 — the profile report: SHIPPED.** The `DataProfileReport` seam + the full
>   section battery (`HeroSection`, `LabelStructureSection`, `CorpusFieldSection`,
>   `SeparabilitySection`, `EmbeddingSection`, `EmbeddingComparisonSection`), the
>   memory-efficient consumed-only `EmbeddingSource` (`ArraySource` / `NpySource`),
>   the `from_benchmark` / `from_records` builders (+ the `[semantic]`-gated
>   `from_embedder` on-ramp), the `langres.eval` public surface, the offline
>   `examples/quickstart_profile.py`, and the `docs/reference/data-profile.md`
>   reference all landed (Waves 0–2 of §7).
> - **PR-2 — the mining seam + mining-readiness: not started** (§4 + the
>   `MiningReadinessSection` of §2.1; Wave 3 of §7). Unchanged from this plan.

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
metric battery including embeddings**, and — the load-bearing constraints —
**flexible / composable / graceful, not a monolith**; **embeddings precomputed &
consumed-only** (never generated; multiple first-class, one default); and **text-first,
HTML optional** (the object is the source of truth, rendering is a thin layer).

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
4. **Embeddings are consumed, never generated; multiple are first-class.** The report
   takes **precomputed** embeddings — by in-memory matrix *or* by location (a `.npy`
   path, parquet, a vector-store handle) — keyed by model name. It never loads an
   embedder or generates vectors, so it carries **no `[semantic]` dependency** (profiling
   a given matrix needs only numpy). **Good default = one embedding model (+ one blocker
   view)**; a user who wants more passes N sources → **one section per model + a
   comparison panel**. Adding an embedder is one more entry.
5. **Text-first; HTML is a nice-to-have renderer.** The computed stats object is the
   source of truth; rendering is a thin layer over it. `print`/markdown/dict are the
   **primary** surfaces (many users live in a terminal or notebook and never open HTML);
   the `$0` HTML tearsheet is **optional** (§2.5). This mirrors QuantStats
   (`reports.metrics()` text vs `reports.html()`), statsmodels (`.summary()` vs
   `.as_html()`), ydata-profiling (`description_set` vs `.to_file`) and sklearn
   (`classification_report(output_dict=…)`).
6. **Reuse the `$0` tearsheet infra.** Inline-SVG (`core/_svg.py`), self-contained
   HTML, no CDN/matplotlib, light/dark — the exact pattern `EvalReport` already proves.

The miners obey the same ethos: **plain composable functions** producing `LabeledPair`s
(as already decided in §5.1), routed by the consumer — not a monolithic strategy engine.

---

## 2. Architecture — the profiler seam

### 2.1 `ProfileSection` — the composable unit

A small Pydantic base every metric block subclasses. This is the seam that makes the
report a *bag of sections* instead of a monolith.

```python
# data/data_profile.py  (leaf module — same layering rules as eval_report.py)
class ProfileSection(BaseModel):                 # frozen
    title: str
    def to_dict(self) -> dict: ...               # model_dump (machine / JSON)
    def to_markdown(self) -> str: ...            # this block's markdown
    def __repr__(self) -> str: ...               # -> to_markdown; prints cleanly in a REPL/notebook
    @property
    def summary(self) -> dict: ...               # numeric headline
    def rows(self) -> list[dict]: ...            # tabular rows -> pd.DataFrame(section.rows()), no pandas dep
    def panels(self) -> list[str]: ...           # inline-SVG/HTML <section>s (reuses _svg) — HTML only
```

Concrete sections (each `ProfileSection`):

| Section | Built from | Reports |
|---|---|---|
| `LabelStructureSection` | `gold_clusters`, `gold_pairs` (+ optional splits) | positive prevalence (pos:neg), cluster-size distribution, singleton rate, linkage-vs-dedup shape, train/valid/test balance + **leakage check** |
| `CorpusFieldSection` | `corpus` (records `.model_dump()`), per source | per-field null/missing rate, cardinality/entropy, value- & token-length distribution, source lopsidedness |
| `SeparabilitySection` | a pair-set + a **pluggable signal** (string sim default; or an embedding cosine) | score histogram **positives vs negatives**, overlap/AUC — "how hard is this dataset" |
| `BlockingSection` (one **per blocker**, default one) | a blocker's candidate pairs + `gold_pairs` | candidate count, **reduction ratio**, **pair-completeness** (recall ceiling), candidates-per-record skew — "how blockable is this data" |
| `EmbeddingSection` (one **per model**, default one) | a **precomputed** `EmbeddingSource` (matrix *or* location) + `model_name` (+ `gold_pairs` for the labeled view) | model name/dim (provenance), vector-norm dist, **cosine positives-vs-negatives**, recall@k of true match in embedding space |
| `EmbeddingComparisonSection` | ≥2 `EmbeddingSection`s | separability by model side-by-side — "which embedder suits this data" |
| `MiningReadinessSection` | a mined `LabeledPair` set (+ optional judgement log) | hard-pos/neg counts, class balance after mining, difficulty histogram (EL2N \|1−p\|), **label-noise estimate** |

### 2.2 `DataProfileReport` — a pure container over sections

```python
class DataProfileReport(BaseModel):              # frozen
    sections: list[ProfileSection]
    # primary surfaces (text-first):
    def to_markdown(self) -> str: ...            # concatenate sections' markdown
    def __repr__(self) -> str: ...               # -> to_markdown; `print(report)` just works
    def to_dict(self) -> dict: ...               # {section.title: section.to_dict()}
    @property
    def summary(self) -> dict: ...               # flattened headline numbers
    def __getitem__(self, title: str) -> ProfileSection: ...   # pull one section out
    # optional renderer (nice-to-have):
    def to_html(self, *, title=...) -> str: ...  # renders present sections' panels only
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
emb    = profile_embeddings(                            # precomputed sources, matrix OR location
    [ArraySource("all-MiniLM-L6-v2", ids, m1),         #   in-memory matrix
     NpySource("bge-large-en", "vecs/bge.npy", ids)],  #   or a path on disk
    gold_pairs=gold_pairs)                              # -> [Section, Section, ComparisonSection]

report = DataProfileReport([label, fields, *emb])
print(report)                    # text-first: markdown table straight to the terminal
report.summary                   # {"pos_prevalence": 0.0012, "n_clusters": 1076, ...}
report.to_html("report.html")    # optional nice-to-have tearsheet
```

**(b) Convenience with sensible defaults — everything optional, nothing blocks:**

```python
DataProfileReport.from_benchmark(
    get_benchmark("abt_buy"),
    embeddings=[ArraySource("all-MiniLM-L6-v2", ids, m1)],  # OMIT -> no embedding section, no error
    blocker=VectorBlocker(...),           # OMIT -> no blocking section, no error
    include={"labels", "fields", "separability"},           # optional subset selector
)
```

**Default set** = labels + fields + separability + (blocking if a blocker is passed) +
(one embedding section per source passed). Omitting `embeddings=`/`blocker=` drops those
sections — *no exception*. `include=` narrows further. Both paths return the same object.

### 2.4 Shared HTML scaffold (non-invasive)

`EvalReport` keeps its `_CSS` + `_panel_*` idiom *inside* the class. To share without
churning it, lift the ~20-line CSS + a `section(title, body)` helper into a tiny
`core/_report_html.py`; `DataProfileReport` uses it, `EvalReport` is **left untouched**
(optional later migration, not in this arc — surgical-changes rule). `core/_svg.py`
chart primitives are already fully shareable as-is.

### 2.5 `EmbeddingSource` — precomputed, matrix or location

The report never generates embeddings. It consumes them through a one-method value so
matrices, files, and vector stores are interchangeable and new backends need no profiler
change:

```python
class EmbeddingSource(Protocol):
    name: str                                    # model id, for provenance + comparison
    def vectors_for(self, ids: Sequence[str]) -> np.ndarray: ...   # aligned to record ids
```

Concrete now: `ArraySource(name, ids, matrix)` (in-memory) and `NpySource(name, path,
ids)` (a `.npy` on disk). Later, trivially: a parquet source, a Qdrant/faiss handle —
each just implements `vectors_for`. Profiling only touches numpy; no `[semantic]` dep.

### 2.6 Render targets — text-first, HTML optional (matches the field)

Every section and the report expose the same ladder, so **HTML is never required**:

| Surface | Method | Use | Precedent |
|---|---|---|---|
| **Print / terminal / notebook** | `print(report)` / `__repr__` / `to_markdown()` | the default way to read it | statsmodels `.summary()`, QuantStats `reports.metrics()`, pandas `df.info()` |
| **Headline numbers** | `.summary` (dict) | log it, assert on it, glance | sklearn `classification_report(output_dict=True)` |
| **Machine / JSON** | `.to_dict()` | persist, diff across runs, feed a dashboard | ydata-profiling `.to_json()` |
| **Tabular** | `section.rows()` → `pd.DataFrame(...)` | analysis, no pandas dep forced | pandas `describe()` returns a frame |
| **Tearsheet (nice-to-have)** | `.to_html()` | shareable `$0` self-contained page | QuantStats `reports.html()`, statsmodels `.as_html()` |

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
- **Embeddings** ← a caller-supplied **precomputed** `EmbeddingSource` (matrix or a
  location) + `model_name` (numpy only). We **consume** vectors; we never generate them
  (§2.5). `embeddings.py` already knows `model_name`/`embedding_dim` for anyone producing
  the matrix upstream.
- **Blocking** ← a blocker's candidate pairs vs `gold_pairs`; reduction ratio +
  pair-completeness reuse `core/metrics.reduction_ratio` and the blocker-recall path
  already in `core/analysis.py`.
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
| Profile a *given* embedding matrix/source | numpy only | **core** | `EmbeddingSection` consumes precomputed vectors |
| Hard-positive classifier + confident-learning | scikit-learn | `[trained]` (already) | `mine_misclassified`, `denoise_pairs` |
| Packaged label-noise (later) | cleanlab | **new, optional, deferred** | Wave 3+ if the built-in isn't enough |
| ~~Generate embeddings~~ | ~~sentence-transformers~~ | **out of scope** | the report never generates — consumed only (§2.5) |

The report pulls **no embedding dependency at all** — a decisive simplification from the
first draft. Import-budget rule (`tests/test_import_budget.py`): `data/data_profile.py`
is a **leaf** that imports only `core.{metrics,benchmark,models,_svg,_report_html}` +
numpy, and pulls **no** sentence-transformers / torch / pandas into a bare `import
langres`.

## 6. Testing & layering

- Leaf module; nothing in `reports.py`/`module.py` imports it; import-budget test guards
  the bare-`import langres` budget.
- Tiered coverage: this is `core` contract → **95–100%**. Each profiler function and
  section gets unit tests incl. the **graceful-degradation cases** (missing embeddings,
  empty gold, single-source corpus → section absent, *no raise*) and the
  **multi-embedding** case (2 models → 2 sections + comparison).

## 7. Build sequence (each wave in its own worktree)

- **Wave 1 — seam + tabular sections. ✅ SHIPPED (PR-1).** `ProfileSection` base,
  `DataProfileReport` container, `core/_report_html.py` scaffold, `LabelStructureSection`,
  `CorpusFieldSection`, `.to_html()` shell. *Exit met:* profile a benchmark's labels+fields
  to a self-contained HTML tearsheet at `$0`; composing a subset works; omitting inputs
  never raises.
- **Wave 2 — separability + embeddings (multi-model) + builders + hero. ✅ SHIPPED (PR-1).**
  `SeparabilitySection`, `EmbeddingSection`, `EmbeddingComparisonSection`,
  `profile_embedding`/`profile_embedding_comparison`, `HeroSection`, the
  `from_benchmark`/`from_records` builders, and the optional `from_embedder` behind
  `[semantic]`. *Exit met:* pass 2 embedders → 2 sections + comparison in the tearsheet;
  no embedder → report still renders.
- **Wave 3 — mining seam + readiness. ← PR-2 (not started).** `mine_misclassified`, `sample_negatives`,
  `denoise_pairs`, `MiningReadinessSection`. *Exit:* mine hard positives + random
  negatives on one benchmark, denoise, profile the mined set in the tearsheet.
- **Stage 3 (north-star, separate epic):** `attribute_examples` + `flipped` + `ColVal`
  schema-free serializer → the existing `finetune()`/serve seam (an unbuilt classification-head trainer variant for AnyMatch fidelity) → the **leave-one-out generalist harness**
  (pool mined pairs across the 10-dataset registry, hold one out, zero-shot eval) — the
  AnyMatch "one ER model across all benchmarks" payoff (F1 81.96 reference). Consumes
  Waves 1–3; scoped in its own plan.

## 8. Public surface

`DataProfileReport` + the profiler functions export via `langres.eval` alongside
`EvalReport` (data-and-eval reporting live together), following the import-light eager
rule. The miners export via the training/`core` surface next to `harvest`/`align_pairs`.
(Alternative: a dedicated `langres.profile` namespace — flagged as an open decision.)

## 9. Decisions

**Settled (maintainer, 2026-07-15):**
- **Embeddings are consumed-only** — precomputed, by matrix *or* location, multiple
  first-class, one default. The report generates nothing and carries no `[semantic]` dep.
- **Text-first, HTML optional** — `print`/markdown/dict are the primary surfaces (field
  standard: QuantStats/statsmodels/ydata/sklearn); the tearsheet is a nice-to-have.
- **Good defaults, extensible** — default = labels + fields + separability + one blocker
  + one embedding; power users add N blockers/embedders; `include=` narrows.

**Still open (recommendation given, will proceed unless redirected):**
1. **Public namespace:** `langres.eval.DataProfileReport` (report lives with `EvalReport`)
   vs. a dedicated `langres.profile` / `langres.data` facade. *Rec: reuse `langres.eval`*
   — fewer surfaces, and it's the reporting home already.
2. **Label-noise rail dep:** built-in confident-learning (sklearn, no new dep) now, with
   Cleanlab as an optional extra later. *Rec: built-in first.*
3. **pandas interop:** expose `section.rows()` (list[dict]) so `pd.DataFrame(...)` works,
   *without* adding a pandas dependency. *Rec: yes — DX win, zero dep cost.*
