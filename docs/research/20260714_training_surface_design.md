# The training surface: vocabulary, API shape, and seam extensions (decision record)

Date: 2026-07-14
Status: **decided with maintainer** (2026-07-13/14 session) — vocabulary + API shape + build order.
Companion to `docs/plans/20260713_training_loop_plan.md` (amends its §3 framework deltas; targets/waves unchanged) and `docs/research/20260713_model_identity_and_hub.md` (consumes its artifact/lineage contract).

## 0. Decisions (TL;DR)

1. **Vocabulary: the pairwise scoring component is a `Matcher`** (was: judge / `Module`). Role renamed; the *record* vocabulary (`PairwiseJudgement`, `JudgementLog`, `LinkVerdict`, `predicted_match`) is kept. Breaking rename, done now while the surface has no external users and before hub method-ids ossify names.
2. **API shape: A + B as two rungs of one ladder; C rejected.** A = explicit components (PyTorch mode — construct each trainable, call *its* `fit`). B = declare-then-fit (`Resolver.fit()` runs the canonical sequence over *declared* trainables) — admissible only with three honesty devices (§3.3). C (a `resolver.training.*` staged namespace, Splink-style) is rejected: a third dialect, +6–8 methods, reintroduces ordering footguns.
3. **One verb: `fit`.** No `fit`/`optimize` split (the "paid or GPU-heavy" criterion did not survive review — a cross-encoder `fit` is GPU-heavy too). sklearn discipline: config in the constructor, data in `fit`, `clone()` to keep an original, hyperparameter search as a meta-object, **the recipe/lineage is an output** (FitReport/RunRecord), never an up-front config file.
4. **No new top-level orchestration noun.** `TrainingRecipe` (an earlier draft) is dead — its composition intent is served by `Resolver`, its reproducibility intent by captured lineage (the G6 lesson from the DX audit: an intent already served by an existing layer under a better name does not get a new noun).
5. **Labels are pairs.** The unit of ER supervision is `(left_id, right_id, bool)` — `LabeledPair`s or a corrections.jsonl path. Alignment to candidates (id-join + split discipline) is the library's job, never the user's.
6. **Seam extensions before sugar** (§4): pairs→candidates bridge; fit protocols beyond the matcher (trainable Blocker); calibrator slot + banded three-way decision; `budget_usd` on every paid fit.
7. **AnyMatch-first build order** (§7), per maintainer: the T2 target leads once Wave-0 plumbing exists.

## 1. Vocabulary: Matcher

### 1.1 The finding

Functionally, everything in the slot — `RandomForestJudge`, `FellegiSunterJudge`, `LLMJudge` — is the same component: candidate pair in, match score and/or decision out. The ER literature calls this the **matcher** (Magellan `MLMatcher`, Ditto "a matcher", DeepMatcher, MatchGPT, AnyMatch); ML calls it a pairwise classifier. The LLM is a classifier too — a zero-shot one. The distinction that matters downstream is **not** LLM-vs-classical but the contract families the code already draws: *ranker* (emits `score`; threshold decides) vs *decider* (emits binary `decision`) vs *abstain*, plus the `score_type` family tag.

The real inconsistencies "judge" was hiding:
- **Two role names for one slot**: verbs say `judge=`, `Resolver` constructor says `module=`, the ABC is `Module` (a vestigial PyTorch borrow).
- **Two directories for one kind of thing**: `core/judges/` vs `core/modules/`.
- **A false association**: in 2026 "LLM judge" connotes LLM-as-judge *evaluation of another model's outputs*; this component performs the primary task.

### 1.2 The rule: components named by role, records named by what they are

| component (renamed) | emits (kept) |
|---|---|
| `Blocker` | candidates |
| `Comparator` | comparison vectors |
| **`Matcher`** (was judge / `Module`) | **`PairwiseJudgement`** |
| `Clusterer` | clusters → `LinkVerdict` |

"A matcher renders a judgement." The judgement contract (score / decision / **abstain** — PR #106) is langres's distinctive honesty asset; it, `JudgementLog`, `LinkVerdict`, harvest/review vocabulary, and `predicted_match()` are unchanged.

### 1.3 Rename scope (one mechanical PR, no deprecation shims — no external users)

- ABC: `Module` → `Matcher`; `GroupwiseModule` → `GroupwiseMatcher`. `core/judges/` + `core/modules/` merge into `core/matchers/`.
- Classes: `LLMMatcher`, `DSPyMatcher`, `RandomForestMatcher`, `FellegiSunterMatcher`, `WeightedAverageMatcher`, `EmbeddingScoreMatcher`, `CascadeMatcher`, `SelectMatcher`, `ScriptedMatcher` (testing).
- Kwargs/fields: verbs + `Resolver.from_schema` take `matcher=` (preset values `"auto" | "string" | "embedding" | "zero_shot_llm"` unchanged — they name mechanisms); `Resolver(matcher=…)` replaces `module=`; results report `matcher_used`; `NoJudgeAvailableError` → `NoMatcherAvailableError`.
- Registry `type_name`s: `llm_judge` → `llm_matcher`, etc. (artifact `_check_versions` already hard-fails cross-version; no artifacts in the wild to migrate).
- The three dispatch sites (`methods.py:_make_module_builder`, `resolver.py:_build_module_for_judge`, `presets.py:build_judge`) rename accordingly — and remain the three sites until the hub design's single `MethodSpec` registry lands (its v0.3 step).
- Deprecated `CascadeModule` (`cascade.py`): delete in the same PR if `methods.py` can move to `CascadeMatcher`; otherwise leave and delete separately.
- Docs/examples/CLAUDE.md/rules sweep in the same PR (docs-in-sync rule).

**Timing rationale**: before the hub identity work publishes method ids, and before the training program adds new component classes. Every later week raises the price.

## 2. Evidence base (what this design is judged against)

Three investigations, 2026-07-13 (full reports in the session record; compressed here):

**(a) Peer-framework training DX** — 5 ER frameworks (Splink v4, dedupe, Zingg, py_entitymatching, recordlinkage) + 6 ML frameworks (sklearn, HF Trainer/TRL, Lightning, spaCy v3, DSPy, AutoGluon):
- Concept count *is* the DX (range observed: 6 → 20 nouns for the happy path); light frameworks reuse borrowed mental models.
- A memorized call sequence is a facade waiting to be built (Splink's prior→u→EM×2 and its "model not fully trained" forum traffic).
- Labeling loops win as: actively-selected + resumable plain artifact + one call (dedupe's `console_label` = sugar over `uncertain_pairs()`/`mark_pairs()`).
- Learned blocking: invisible by default, inspectable on demand (dedupe) beats hand-assembled (Magellan) and fully opaque (Zingg).
- A zero-label on-ramp is a distinct asset (Splink; recordlinkage `ECMClassifier.fit_predict`).
- Estimator-owns-fit (sklearn) wins where training is cheap-to-moderate and composition matters; external-Trainer wins only where the loop is the expensive shared engineering (not ER); config-file-as-entry (spaCy) is a wall; compile-returns-new-object (DSPy) contributes budget-as-a-word and the two sins to avoid (invisible trials, invisible cost).

**(b) The use-case matrix** — 24 learning/optimization use cases across the docs + `.agent/genalysis` taxonomy. Four classical axes the LLM framing under-served, now design requirements:
1. The fit seam was **matcher-only** — learned blocking (dedupe/Zingg hallmark; contrastive encoders) architecturally inexpressible.
2. `fit()` is **single-shot batch** — the iterative active-learning loop had no home.
3. Labels were **positionally aligned bools** — id-keyed pairs (harvest output) had no bridge; vote matrices / corpus stats can't pass through.
4. A **threshold cut is not calibration** — no Platt/isotonic transform, no three-band (auto-match / clerical-review / non-match) decision.

**(c) AnyMatch ground truth** (arXiv:2409.04073, verified from paper + repo): GPT-2-124M with a **binary classification head**, **full fine-tune (no LoRA)**, trained on 8-of-9 pooled benchmarks, zero-shot on the 9th. The lever is the data recipe: AutoML **hard-positive** mining (misclassified positives; negatives random at 1:2), attribute-level augmentation (largest ablation, −3.14 F1), flipped pairs, `COL/VAL` serialization. No teacher, no distillation. Its repo has no LICENSE — reimplement, never fork.

## 3. The surface: A + B, one verb

### 3.1 Shape A — explicit components (the ground truth; mostly exists)

Construct each trainable component; call *its* `fit`; assemble the `Resolver` yourself. PyTorch property by construction: everything is in your hands.

```python
matcher = RandomForestMatcher(feature_specs=comparator.feature_specs)
train_c, valid_c, y_tr, y_va = align_pairs(cands, pairs, split=0.2)   # NEW: the bridge
matcher.fit(train_c, y_tr)
tau = derive_threshold(scores(matcher, valid_c), y_va)
resolver = Resolver(blocker, comparator, matcher, Clusterer(threshold=tau))
```

### 3.2 Shape B — declare, then fit (thin sugar over A; the front door)

Trainables are **declared at construction**; one `resolver.fit()` runs the canonical sequence — block → (fit blocker if trainable) → fit matcher → fit calibrator → derive threshold/bands — over exactly the declared trainables, literally calling A's seams in order. Opening the hood is scrolling down, not learning a dialect.

```python
resolver = Resolver.from_schema(Company, matcher="random_forest")
report = resolver.fit(records, pairs="corrections.jsonl")
```

### 3.3 B's three honesty devices (admissibility conditions, not options)

The maintainer's criterion: over-simplification that hides what happens underneath *adds* cognitive load. PyTorch's virtue — you know what you're training — is restored inside a one-call API by:

1. **Declared-at-construction**: nothing trains that you didn't name. `TrainedLMMatcher(base="Qwen/Qwen3-1.7B", method=QLoRA(r=16), serializer=ColVal(mode=1))` states the architecture, the method, and the rank in the user's own code.
2. **`resolver.describe()`** — pre-fit: which slots *would* train (and how), which are frozen.
3. **`FitReport`** — post-fit, returned from every `fit`: per-slot what trained (mechanism, learned-param summary), on how much data (train/valid split sizes, label provenance incl. silver-teacher identity), at what cost (tokens/$ for paid fits, GPU time for local), thresholds/bands derived, and validation metrics. Doubles as the lineage record (feeds RunRecord and the hub model card: base model, data recipe, license chain, eval provenance).

### 3.4 Cross-cutting semantics

- **One verb `fit`**; config in constructor, data in fit ("no data in `__init__`").
- **Mutation is explicit**: `fit` mutates in place (sklearn semantics); `resolver.clone()` first to keep an original. No hidden copies.
- **Labels are pairs** (`LabeledPair` list or corrections.jsonl path); alignment is the library's job.
- **Search is a meta-object** (`GridSearchCV` pattern): wraps a component/resolver factory and is itself fittable. `BlockerOptimizer` remains this layer; not a fourth verb.
- **Budgets on paid fits**: `budget_usd=` (compile/fine-tune constructors), enforced through the existing SpendMonitor seam; spend reported in FitReport. Closes DSPy's documented cost-opacity sin at birth.
- **The recipe is an output**: lineage captured from what ran (FitReport → RunRecord `recipe_id`/`attempt_id`), never declared up front.

### 3.5 Rejected shapes (recorded so they stay rejected)

| shape | why rejected |
|---|---|
| `TrainingRecipe` orchestration object | new noun parallel to `Resolver`; reproducibility-as-input is the spaCy config wall; fails the DX-audit G6 test |
| C — `resolver.training.*` staged namespace | +6–8 methods = a third dialect between verbs and core; user owns ordering (Splink's "not fully trained" footgun class). Its one good idea — `derive_bands(auto_precision=…, review_budget=…)` vocabulary — is stolen into config |
| External `Trainer` object (HF/Lightning style) | ER has no thousand-GPU shared loop to amortize; would import `TrainingArguments`-sprawl + inversion-of-control debugging for nothing |
| Config-file entry point (spaCy style) | reproducibility is delivered as *output* artifacts (`resolver.save()`, FitReport) without a wall in front of first success |
| fit/optimize verb split by "paid or GPU-heavy" | criterion incoherent (a cross-encoder `fit` is GPU-heavy); replaced by one verb + explicit `clone()` |

## 4. Seam extensions (shape-independent; unblock the classical axes)

1. **`align_pairs` — the pairs→candidates bridge** (~60–100 LOC): id-join on `frozenset({left_id, right_id})` from `LabeledPair`s/corrections.jsonl to aligned `(candidates, labels)`, with split discipline (train/valid) owned by the library. Unblocks every trained matcher and closes the flywheel loop (harvest output flows into `fit` in one call). This is the single most load-bearing new piece — identical in every shape.
2. **Fit protocols beyond the matcher**: `SupervisedFitMixin`/`UnsupervisedFitMixin` stop being matcher-only; a `Blocker` may opt in structurally (`TrainableVectorBlocker(base=…, method=Contrastive(…)).fit(records, pairs)`). Makes learned blocking *expressible* now, implementable later — invisible by default, inspectable via `describe()`.
3. **Calibrator slot + banded decision**: a `Calibrator` (Platt/isotonic; `fit(scores, labels)`, `transform(scores)`) as a declared pipeline slot fitted during the canonical sequence; `decision=Bands(auto=…, review=…)` yields the classical three-way partition — `result.matches / result.review_queue / result.rejects` — making the review queue a first-class decision output rather than a post-hoc selection. Band derivation accepts target vocabulary (`auto_precision=`, `review_budget=`).
4. **`budget_usd` on every paid fit** (see §3.4).

Deferred, doors kept open (nothing in the surface forecloses them): `partial_fit`/online + drift; weak-supervision vote-matrix label models (a labels-side generalization, not a `fit` signature change); collective/fused clustering (inverts control; own design when reached).

## 5. A-layer components for the training program

The training-loop plan's targets ride these; note **UC-AnyMatch never touches the Resolver** — it is a matcher + eval harness, so this A-layer *is* the AnyMatch tooling gap:

1. **Miners as plain functions**, not strategy objects: `mine_misclassified(dataset, miner=…, cap=…)` (AnyMatch hard-positives), `sample_negatives(dataset, ratio=…, seed=…)`, `attribute_examples(pool, cap=…)`, `flipped(pairs)`, `mine_hard_negatives(blocker_log, …)` (the #86 blocking-derived strategy). All produce `LabeledPair`s — printable, inspectable data prep (the label-noise inspection the #86 survey demands). Hard-positive and hard-negative mining stay **separate functions** (opposite failure modes), routed by consumer (fine-tune wants hard; DSPy demo pools want easy-correct).
2. **`TrainedLMMatcher`** — the trainable-LM matcher: `base=` (HF id), `method=` (`QLoRA(r, alpha)` | `FullFinetune()` — AnyMatch fidelity needs full-FT GPT-2), `head=` (`"classification"` | generative), `serializer=`, epochs/lr. Implements the fit protocol; checkpoint to `state_dir` sidecar (safetensors, no pickle); lazy heavy imports; training-only deps (unsloth/trl/peft) stay dev-group-only.
3. **`ColVal` serializer** — one Ditto-style `COL <val>…` textualizer (AnyMatch modes 1–4; `N/A` for missing), shared by trained matchers, LLM prompts, and the replication.
4. **`CrossEncoderMatcher`** — the Ditto-method-class baseline (sentence-transformers), same fit protocol; completeness yardstick per the training plan's LLM-native framing.
5. **Local student serving** — per the training plan §3.3 Option A: fine-tuned checkpoints served behind `LLMMatcher` via a local OpenAI-compatible `api_base` (same seam as the paid teacher), with the in-process matcher as fallback.

## 6. Acceptance examples (the spec)

Each gap is done when its example is writable as shown. Condensed; `# NEW` marks not-yet-existing seams.

**Zero-label on-ramp (FS EM)** — `resolver.fit(records)` with no labels takes the unsupervised path; report states m/u × features and the percentile threshold.

**Classical supervised (bread-and-butter)**
```python
resolver = Resolver.from_schema(Company, matcher="random_forest")
report = resolver.fit(records, pairs="corrections.jsonl")        # NEW (fit ext + bridge)
```

**Active-learning flywheel (per round)**
```python
todo = select_for_review(resolver.judgements, n=50)               # exists
# … label via `langres review` CLI …                             # exists
report = resolver.fit(records, pairs="corrections.jsonl")        # NEW: re-enterable round
```
The loop stays in user code (dedupe/Prodigy/modAL evidence: primitives + sugar beat a loop object).

**DSPy prompt-tune on teacher silver** — teacher→silver is explicit data prep (harvest), never hidden inside fit:
```python
student = DSPyMatcher(model="local/qwen3-1.7b", optimizer="mipro",
                      metric="pair_f1", budget_usd=5.0)           # NEW: budget
resolver = Resolver.from_schema(Product, matcher=student)
report = resolver.fit(records, pairs=silver)                      # report: demos, $ spent, teacher id
```

**QLoRA fine-tune**
```python
student = TrainedLMMatcher(base="Qwen/Qwen3-1.7B", method=QLoRA(r=16),
                           serializer=ColVal(mode=1), epochs=3)   # NEW component
student.fit(train_cands, silver)                                  # GPU; safetensors sidecar
```

**AnyMatch replication (core layer; no Resolver)**
```python
pool  = [get_benchmark(n) for n in NINE if n != holdout]
pairs = concat(mine_misclassified(d, miner="random_forest", cap=400)   # NEW miners
               + sample_negatives(d, ratio=2, seed=42) for d in pool)
pairs += attribute_examples(pool, cap=600) + flipped(pairs)
matcher = TrainedLMMatcher(base="gpt2", head="classification", serializer=ColVal(mode=1))
matcher.fit(pairs)
f1 = evaluate(matcher, get_benchmark(holdout).candidates("test"), threshold=0.5)
```

**Learned blocking**
```python
blocker = TrainableVectorBlocker(base="qwen3-embedding-0.6b", method=Contrastive())  # NEW
resolver = Resolver.from_schema(Company, blocker=blocker, matcher="random_forest")
resolver.describe()          # NEW: blocker trainable (contrastive) · matcher trainable · clusterer frozen
report = resolver.fit(records, pairs)
```

**Calibration + three-band decision**
```python
resolver = Resolver.from_schema(Person, matcher="fellegi_sunter",
                                calibrate="platt", decision=Bands(auto=.90, review=.60))  # NEW
report = resolver.fit(records, pairs)
result = resolver.resolve(records)         # result.matches / result.review_queue / result.rejects
```

## 7. Build order

| wave | content | notes |
|---|---|---|
| **0a — rename** | the Matcher rename (§1.3), one mechanical PR incl. docs sweep | before hub ids + new components ossify "judge" |
| **0b — seams** | `align_pairs` bridge; fit protocols beyond the matcher; `FitReport` + `describe()` + `Resolver.fit` canonical sequence; `budget_usd` plumbing | serves every shape and use case |
| **0c — training components** | miners (misclassified, random-neg, attribute, flipped, blocking-derived hard-neg); `ColVal`; `TrainedLMMatcher` scaffold (CI dry-run on `tiny_fixture`); GPU smoke (Unsloth QLoRA Qwen3-0.6B overfits 100 pairs on the 3070) | = training plan Wave 0, re-expressed |
| **1 — AnyMatch (T2)** | LOO harness over the registry pool; native run (Qwen3-0.6B QLoRA + sklearn miner) first, paper-faithful (GPT-2 full-FT, AutoGluon) as ablation if numbers disappoint | maintainer decision 2026-07-13: AnyMatch leads |
| **2+** | T1 gold ladder → teacher tune + silver (≤$5 gate) → silver students → T4-H1 blocking probes | per the training plan; unchanged |
| later | calibrator slot + Bands; `TrainableVectorBlocker` (T4-H2 material); `CrossEncoderMatcher` baseline | classical-gap items, sequenced behind the program |

## 8. Open items

- `resolve()` return shape when `decision=Bands(…)` is declared (three-way result object) — settle at implementation.
- The hub `MethodSpec` registry (identity doc v0.3) should collapse the three dispatch sites in the same motion or immediately after the rename — coordinate to avoid renaming the same three sites twice.
- Steiner/Peeters/Bizer 2024 must be read before any silver-harvest design (standing flag from the #86 survey and training plan §8).
