# langres — Vision & Roadmap

> **Status:** Direction-setting document (2026-06-25). Supersedes the pure-POC
> framing in `POC.md` for *where we are going*; `POC.md` remains the record of
> the original validation plan. This doc defines the vision, the architectural
> spine, and **verifiable milestones** so progress is measurable at each step.

---

## 1. Vision — the PyTorch of Entity Resolution

langres is the **composable seam** for entity resolution: a framework where you
**compose, benchmark, and tune** different ER methods behind common interfaces,
**bootstrap** the labels you don't have, and **ship a versioned model artifact**
that runs at inference elsewhere.

Three stacked definitions of "usable", all true at once:

1. **Seam** — every method (rapidfuzz, embeddings, LLM judge, DSPy-distilled
   judge, GLinker, GLiNER extraction) is wrapped behind one `Blocker`/`Module`
   interface, so you can swap and **benchmark them head-to-head on your data**.
2. **Training framework** — you `fit` / `optimize` / `distill` a pipeline and it
   produces a **versioned artifact** (extractors + blocking funnel + judge +
   thresholds + compiled prompt).
3. **Inference** — a consumer (brainsquad) `load`s that artifact and runs it on
   its own infra. *Engine intelligence in langres; data, persistence, visibility
   in the consumer.*

**General by design, proven by one real use case.** brainsquad
([raisesquad/brainsquad#1051](https://github.com/raisesquad/brainsquad/issues/1051))
is our first and only real consumer — Person (hard, multilingual), Program/Project
(easy), Geography (external authority), Grant (later). We prove the framework on
it, while keeping every abstraction entity-agnostic.

---

## 2. The architectural spine

### 2.1 Entity = a typed **feature bag** (the core model)

An entity is a set of typed, possibly multi-valued, possibly missing features
(name, aliases, role, org, location, dates, identifiers…). Matching compares
feature bags that may be **lopsided** — one side rich, one side sparse. This is
the Fellegi-Sunter model, modernised.

**Design rules that make asymmetric matching work:**

- **Comparison vector, not a single string compare.** Each candidate pair yields
  a per-feature comparison *level*: `exact / fuzzy-high / fuzzy-low / mismatch /
  missing`.
- **`missing` is a neutral level — it never penalises.** Score on shared
  features; stay silent on the rest. This is what lets "many features vs. one
  feature" resolve correctly.
- **The judge weights features by discriminativeness** — learned from the gold
  set, or reasoned by the LLM (which degrades gracefully under partial info).
- **Anchors decide alone.** A shared unique identifier (UID, domain, ORCID) →
  deterministic match, no judge.

**Features unify blocking and matching:** each feature is both a *blocking key*
(union of per-feature keys → high recall; a one-feature entity is still
blockable) and a *comparison signal*. Extraction (incl. GLiNER) populates the bag.

```
raw → [enrich: extractors / GLiNER / (later) web-search] → feature bag
    → [union of per-feature blockers]            → candidate pairs   (HIGH RECALL)
    → [per-feature comparison vector, missing-aware]
    → [judge: heuristic | embedding | LLM | DSPy-distilled | GLinker] → PairwiseJudgement
    → [Clusterer: pos + neg judgements → connected components]        → clusters / golden labels
```

### 2.2 Components (interfaces — the seam)

| Component | Role | Today |
|---|---|---|
| **Enricher / feature source** (extractors incl. `GLiNERExtractor`; later web-search/Exa, registries) | populate & **augment** the feature bag from text or external sources | new (extraction); web-search **deferred, slotted** |
| **Blocker** | feature bag → candidate pairs (high recall) | exists (Vector/AllPairs); add funnel + per-feature union |
| **Comparator** | pair → per-feature comparison vector (missing-aware) | seed in `RapidfuzzModule.field_extractors`; generalise |
| **Module (judge)** | comparison/raw → `PairwiseJudgement` | exists (rapidfuzz/LLM/cascade); add DSPy + GLinker adapters |
| **Clusterer** | judgements (pos+neg) → clusters | exists (connected components); add cannot-link |
| **Bootstrapper** | entities → gold set (sample→label→mine→report) | new — **critical path** |
| **Benchmark** | race methods on gold set → metric table | ~70% (eval exists); add harness |
| **Resolver** | the composed, fitted, **serializable** pipeline | new — **the artifact** |

### 2.3 Two operating modes of a Resolver

- **Dedup / batch** — resolve a dataset against itself → clusters. (Have the parts.)
- **Incremental / linking** — probe a *new* record against an existing entity
  store → matched entity id or "new". (`stream_against`; brainsquad's runtime op.)

### 2.4 The flagship loop — build a golden label, then grow it over time

The core brainsquad pattern (and the most common real ER loop): we first *see* a
sparse mention (often just a name — a person extracted from a document, an org
name), optionally **enrich** it (e.g. add a LinkedIn URL, a registry id, web-search
content), and mint it as a **golden label**. Then every future appearance of that
entity must **link** back to the golden label — and each link **enriches** the
golden record with whatever new features that mention carried.

```
new mention (sparse: maybe just a name)
   → [enrich?]  (extraction now; web-search/Exa later — optional)
   → link against existing golden labels  (incremental, asymmetric: sparse ↔ rich)
        ├─ match  → merge features into the golden record (it grows)  → canonicalize
        └─ no match → mint a new golden label
```

Two properties make this work and must be designed in:

- **Asymmetric matching (sparse ↔ rich).** A new one-feature mention links against
  a feature-rich golden record. The missing-aware Comparator (§2.1) scores on the
  shared feature; the golden record's accumulated **aliases + popularity prior**
  (à la GLinker) *help* rather than hurt. We do **not** require symmetric features.
- **Progressive enrichment = canonicalization in a loop.** The golden record is the
  survivorship-merge of all its linked mentions' features. Each link feeds back. So
  the entity store is **growing and self-enriching**, not a fixed target.

This is **UC2 (Entity Linking) ⊕ UC4 (Master Data Creation), run incrementally** —
see §2.5. langres provides the *brain* (the linker + the canonicalization rules,
as a versioned artifact); brainsquad provides the *body* (the persistent golden
store, the event log, reversibility — exactly its #1051 design).

### 2.5 Use-case coverage — the compass

We track direction against the documented taxonomy in `USE_CASES.md`. Each use
case must end up either **filled by a milestone** or **deliberately delegated to
the consumer** (brainsquad owns the stateful "body"; langres owns the "brain").

| Use case (`USE_CASES.md`) | Where it lands | Status |
|---|---|---|
| **UC1 Deduplication** (batch → clusters) | M2 | on path |
| **UC2 Entity Linking** (link to a target store) | M5 (incremental `stream_against`) | on path |
| **UC10 Fuzzy FK** (special case of UC2) | via M5 | on path |
| **UC4 Master Data Creation** (golden records / survivorship) | M5 — **promoted** (brainsquad needs golden labels *now*, not V1.1) | on path |
| ⭐ **Flagship: incremental linking + progressive golden-record enrichment** (§2.4) | M5 (= UC2 ⊕ UC4 + Enricher loop) | **explicitly captured** — was only implicit across UC2/UC4/UC5 in `USE_CASES.md` |
| **UC9 Negative constraints** (cannot-link) | M6 (constrained clustering) | on path |
| **Human-in-the-loop** labeling | M1 (bootstrapper labeler — human option) | on path |
| **Optimization** (Optuna + DSPy) | M3 / M4 / M6 | on path |
| **Data generation / cold-start** | M1 (bootstrapper) | on path |
| **UC3 Record Linkage** (multi-source) | post-M5, config | deferred (V1.1) |
| **Enrichment via web-search / Exa** | Enricher plug-in (§2.2) | **deferred, slotted** — out of scope now |
| **UC8 PPRL** (privacy-preserving) | consumer strips private features *before* langres sees them (§5) | delegated |
| **UC7 Collective / graph** | langres = pairwise brain; brainsquad builds the graph/network layer on resolved nodes | delegated |
| **UC5 Streaming** | langres compiles the artifact (brain); brainsquad runs it real-time (body) — matches #1051 | delegated |
| **UC6 Temporal evolution** | langres emits reversible judgements; brainsquad owns the temporal store + event log (#1051) | delegated |

The "delegated" rows are not gaps — they are the **clean brain/body seam** the
#1051 design already assumes. *Follow-up: once direction is locked, formalize the
flagship loop (§2.4) as a named use case in `USE_CASES.md` and promote UC4 there.*

---

## 3. The seam in practice — methods we compose & compare

| Method | Wrapped as | Use |
|---|---|---|
| rapidfuzz | Comparator + Module | cheap baseline |
| embedding ANN (FAISS/Qdrant) | Blocker | high-recall candidate gen |
| GLiNER / GLiNER2 | Extractor | feature extraction → blocking keys + comparison features |
| LLM judge (litellm) | Module | strong judge / teacher |
| **DSPy-distilled judge** | Module | cheap student compiled from the teacher (the differentiator) |
| **GLinker** (`gliner-linker`, pg_trgm retrieval) | Blocker + Module | candidate — fit for record↔record is **unverified** (trained on mention↔description; see §8), benchmarked as one option in M3 |
| Fellegi-Sunter / logistic | Module | learned weighted comparison |

**Benchmark on two axes:** (a) brainsquad's own Person/Program gold sets
(dogfood + real validity), and (b) standard ER benchmarks (DBLP-ACM, Abt-Buy,
etc.) for external validity and method sanity-checks.

---

## 4. Cold-start labelling (LLM-teacher first)

We have no gold labels and they gate everything (DSPy, benchmark, optimisation).
The bootstrapper is the unlock.

- **Sampler:** uncertainty + hard-negative mining (embedding-kNN near-misses),
  small random tail for the easy-negative class. *Not* random pairs.
- **Labeler (chosen): LLM-teacher** — GPT-class model labels the ambiguous band
  with rationales; validate against a ~100–200 pair human spot-check; calibrate;
  route low-confidence to a human. (Pluggable with human active-learning / weak
  supervision later.)
- **Output:** `GoldPair` set + **coverage report** (blocking Pair-Completeness,
  label-count-vs-F1 curve, teacher calibration/ECE).
- **Reusable strategy interface:** swappable `sampler` / `labeler` /
  `hard_negative_source` behind one `Bootstrapper.build(entities)`.

---

## 5. The artifact (brainsquad integration contract)

A `Resolver` serialises to a **versioned artifact**: feature extractors +
blocking funnel config + judge (incl. compiled DSPy prompt / model ref) +
thresholds + metric provenance. brainsquad `Resolver.load("person_v1")` and calls
`.resolve(records)` (batch) or `.link(record)` (incremental). langres owns engine
intelligence; brainsquad owns persistence, visibility (public/private feature
stripping happens consumer-side before features reach langres), and the cluster
store.

---

## 6. Milestones (verifiable)

Each milestone has a **measurable exit criterion**. Targets inherit the POC bars:
**blocking recall ≥ 0.95**, **BCubed F1 ≥ 0.85** for the hybrid judge.

### M0 — Contract & spine
Build the `Resolver` container (compose Blocker+Comparator+Module+Clusterer;
`save`/`load`) and the entity feature-bag + missing-aware Comparator abstraction.
- **Exit:** existing example runs through `Resolver.save/load` round-trip and
  produces identical clusters before/after; one external adapter stub compiles.

### M1 — Person gold set (cold-start, LLM-teacher)
Bootstrapper: hard-negative mining from the Blocker + LLM-teacher labeler +
coverage report. Produce the brainsquad **People (board-members) gold set**.
- **Exit:** a Person gold set exists with measured teacher-vs-human agreement on
  a spot-check sample; blocking **Pair-Completeness reported** (target ≥ 0.95).

### M2 — Walking skeleton end-to-end + baseline
feature bag → block → baseline judge → cluster → eval → **artifact**. The shipped
baseline judge is the zero-spend `WeightedAverageJudge` over the feature bag;
richer judges (embedding cascade, `gliner-linker`, LLM) are raced in M3. Shipped on
Fodors-Zagat (Person artifact + brainsquad load-and-run is M5, see below).
- **Exit (SHIPPED):** **BCubed F1 baseline reported** on held-out gold; the saved
  artifact runs a brainsquad-style **`.resolve()`** call end-to-end in a fresh
  process (identical clusters). Measured on Fodors-Zagat (seed=0, threshold 0.8):
  held-out BCubed P/R/F1 = 0.991/0.969/0.980 vs merge-nothing floor 0.932,
  Pair-Completeness 1.0. The M2 consumption contract is `resolve()`-only;
  incremental `.link()` / `.stream_against()` are M5 stubs (below).

### M3 — The seam: multi-method benchmark
**First task (carried from the M2 post-merge audit):** extract the general
evaluation/split/threshold-tuning machinery (`evaluate_resolver_bcubed`,
`BCubedEvalResult`, `complete_partition`, `tune_threshold_on_train`) out of the
dataset-specific `data/er_benchmarks.py` (now a ~620-line god-module) into a
reusable `core` benchmark/eval module, and collapse `split_restaurant_corpus`
back onto a generalised `data/splitting.py`. The harness is a first-class
component (§2.2), not benchmark glue — extract-then-extend so each new dataset
reuses it instead of re-duplicating. Also report **pairwise F1 on true matches**
alongside BCubed (BCubed is inflated on singleton-heavy corpora — see the M2
sanity-floor caveat).

Then wrap ≥3 methods (rapidfuzz, embedding cascade, LLM judge, **GLinker**) behind
the interfaces; the harness emits BCubed / recall / cost / latency on the
Person gold set (+ ≥1 standard benchmark dataset). These method families *are* the
three `POC.md` approaches — 1 (classical/rapidfuzz), 2 (embedding ANN), 3 (hybrid
LLM) — now raced head-to-head behind one interface instead of run in sequence.
- **Exit (SHIPPED):** a reproducible **method-comparison table** with a real-money
  race (total **$2.18** / $15 cap) over an *easy* (Fodors-Zagat) and a *hard*
  (Amazon-Google) dataset. Full results
  [`data/benchmarks/m3/M3_RESULTS.md`](../data/benchmarks/m3/M3_RESULTS.md);
  decision [`docs/M3_DIRECTION_MEMO.md`](M3_DIRECTION_MEMO.md). Headline (AG hard,
  pair-F1): **gpt-4o `llm_judge` 0.667** (SOTA band, beats free embedding 0.471) >
  embedding 0.471 > **GLM-5.2 `llm_judge` 0.409** (high-recall/low-precision,
  *below* free) > weighted 0.288 > rapidfuzz 0.271. On easy FZ, free embedding wins
  (0.816) and the GLM judge degenerates. **The finding reshapes M4:** the LLM judges
  are high-recall but the cheap OSS one over-accepts — so M4 is *make a precise judge
  cheap* (prompt-optimize, calibrate+tune the deferred cascade against the real
  embedding-score distribution, distill frontier-quality labels, stronger embedder),
  not "bolt on a judge." Cascade + frontier-FZ deferred (memo §6).

### M4 — langres is the seam: a working DSPy experimentation foundation
**Reframed (2026-07-01) from a distillation-metric chase to "build the seam we're
happy to use."** M3 showed the cheap OSS judge's *precision* collapses (GLM-5.2 0.409
pair-F1 on AG, *below* free embedding) — the signature of a generic prompt + hand-set
thresholds. M4's job is the **plumbing** that lets us fix that data-drivenly: a clean,
composable scorer seam (DSPy judge, learned thresholds, an experiment facade, honest
cost) that serves experimentation now and deployment later (`Resolver.save/load` is
the bridge). **KISS is a first-class constraint** — the smallest seam that proves the
plumbing and yields a first honest signal; composability is *earned* by real reuse,
not accreted up front.

**Delivered (all validated zero-spend):**
- `DSPyJudge` — import-safe (`import langres.core` never imports `dspy`),
  `compile(bootstrap|mipro)`, honest per-pair cost, serializable — behind the `Module`
  contract.
- `derive_threshold(scores, labels)` (Youden / percentile) — kills the "thresholds
  set by hand, not from the data" sin (M3's cascade `0.3/0.9`).
- `run_methods(...) -> BenchmarkTable` experiment facade + `langres.clients.openrouter`
  (price-pinning + `SpendMonitor` cumulative-spend guard).
- **Proven end-to-end at $0** (DummyLM): a *compiled* DSPyJudge runs through
  `evaluate_judge_on_candidates` (judged-once, pairwise-F1, SOTA-comparable — the right
  surface for a compiled/paid judge; `run_methods` is the cheap-method race). See
  [`docs/EXPERIMENTS.md`](EXPERIMENTS.md), `examples/m4_experiment_loop.py`.

**Paid first signal (monitored, ≤$5):** a manual precision probe + one small MIPROv2
compile on Amazon-Google — **gated behind a frontier-zero-shot null baseline
(delta C7):** if a compiled cheap student can't beat "just call the frontier model" on
cost at equal quality, *cut it*. This de-risks a measured caveat — on OpenSanctions
Pairs, DSPy MIPROv2 lifted only ~1–2 F1 and in-context examples were
neutral-to-negative — so the distillation upside on messy multilingual data is
**uncertain and must be measured, not assumed**.

**Paid result (2026-07-02, $2.31/$5 on the 600-pair AG band — `data/benchmarks/m4/M4_RESULTS.md`):**
a precision-tuned DSPy **signature** lifts the cheap GLM-5.2 judge from pair-F1 **0.409
→ 0.757** (precision 0.264 → 0.671), **beating the frontier gpt-4o ceiling (0.667) at
lower cost — uncompiled**. **MIPROv2 compilation did *not* help** (0.757 → 0.746 for
+$1.63): it overfit its 40-example bootstrap metric, confirming the OpenSanctions caveat
on our data. **C7 verdict: the lever is the signature, not compilation — cut distillation.**
- **Exit (met):** the DSPy experimentation loop is real and reproducible (compile →
  evaluate → serialize; compiled `Resolver` artifact saved), a first honest paid signal
  on AG is recorded with its F1/$ frontier, and we have a read on the DX. **Not** a fixed
  "student ≥ margin of teacher BCubed" metric — the C7 gate said don't chase the compile.

*(Research input: [`docs/research/20260701_er_seam_audit.md`](research/20260701_er_seam_audit.md);
delta backlog tracked in issue #55.)*

### M4.5 — restore "any combination" against SOTA (research-driven)
The seam fully expresses the pairwise pretrained/prompted family, but the ER field has
moved to two shapes it does **not** yet express. Deferred here — each additive,
backward-compatible, and *earned* by a real experiment (not built speculatively):
- **S1 (highest-leverage): a set-wise judgement contract** — `SetJudgement` /
  `ERCandidateGroup` + a groupwise judge that still yields `PairwiseJudgement`
  (downstream untouched). Unlocks ComEM Select (+16 F1 at ~⅓ cost) and in-context
  clustering — the field's biggest cost *and* quality lever.
- **Blocking pair-set algebra** — `KeyBlocker` + `CompositeBlocker`
  (union / intersection / difference) + embedder sweep; recall-first composition.
- **S2: a `fit()` / `fit_unlabeled()` Module hook** + a `compile(student, trainset,
  metric) → Module` Optimizer shape — homes the trained-judge family (Magellan RF,
  Fellegi–Sunter EM, ZeroER, Snorkel, DSPy) the current forward-only Module can't
  express.
- Value-frequency-aware `FSJudge`; merge-resistant clusterer default. (Full
  C / S / B delta table in the research doc + #55.)

### M5 — Generalise + incremental + golden-record loop
Program/Project via **config-only** change; Geography via external authority
(GeoNames) adapter; `stream_against` incremental linking against an entity store;
**Canonicalizer** survivorship so a matched link **merges its features into the
golden record** (§2.4 flagship loop).
- **Exit:** a second entity type resolved with no new core code (config only);
  incremental `.link(new_record)` returns the correct existing entity or "new";
  **a sparse new mention correctly links to a feature-rich golden record, and the
  golden record gains the mention's new features** (enrichment verified).
- **Exit (north-star measurability — carried from the M2 audit):** **Person
  resolution is *measurable*.** M0–M2 validate the machinery on Fodors-Zagat
  restaurants because brainsquad Persons have ~0 duplicates and no ground truth;
  M1's chartered Person gold set shipped as a restaurant one. Before M6 can gate
  on "Person BCubed ≥ 0.85," M5 must establish an evaluable Person target —
  either a real (even small) Person gold set via the bootstrapper, or a formally
  defined measurable proxy (e.g. synthetic name-variant linking). Until this
  exit is met, "validated on Fodors-Zagat" is explicitly *not* "validated on
  Person."

### M6 — Hardening (post-proof)
Blocking-funnel optimisation (recall-first Optuna objective), score calibration,
model/version registry, production/operability guidance, cannot-link clustering.
- **Exit:** Person resolver meets **BCubed F1 ≥ 0.85**; documented deploy/rollback
  path; reproducible artifact versioning.

---

## 7. Mapping to brainsquad (#1051)

| brainsquad task (#1051) | langres milestone |
|---|---|
| Build People gold set (teacher set) | **M1** |
| Walking skeleton: features → blocking → judge → clusters → persist | **M2** (langres half) |
| Iterate: multi-filter blocking + DSPy distillation | **M3 + M4 + M6** |
| Generalise to Program/Project + Geography | **M5** |
| Versioned model artifact consumed at inference | **M2 artifact, hardened in M6** |

---

## 8. Open questions / things to learn

- **Comparator design for heterogeneous features** — how rich to make the
  comparison-level taxonomy; learned vs. LLM-reasoned combiner; how anchors
  short-circuit. (We have the conceptual model; validate empirically in M2–M3.)
- **GLinker fit** — is `gliner-linker` competitive on *record↔record* (it was
  trained on mention↔description)? Answer empirically in M3, don't assume.
- **DSPy distillation cost/quality** on our messy multilingual Person data vs.
  the clean POI case in the Overture talk. **Partial answer (research, 2026-07-01):**
  on OpenSanctions Pairs, MIPROv2 lifted only ~1–2 F1 and in-context examples were
  neutral-to-negative — so the upside is *uncertain*. M4 therefore measures it behind
  a frontier-zero-shot null-baseline gate (delta C7) rather than assuming it. See
  [`docs/research/20260701_er_seam_audit.md`](research/20260701_er_seam_audit.md).
- **Set-wise vs pairwise judging** — the field's strongest *and* cheapest LLM methods
  (ComEM Select, LLM-CER) score an anchor against a candidate *set*, which the current
  `pair → PairwiseJudgement` contract cannot receive. How much of the cost/quality
  frontier does the set-wise contract (M4.5 · S1) actually recover on our data?
