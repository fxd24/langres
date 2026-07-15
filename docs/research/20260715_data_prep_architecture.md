# Data Preparation — Design & Architecture

> **Status:** Design overview / living note (2026-07-15). This is the **architecture
> spine** for langres's data-preparation layer: the organizing taxonomy of method
> *shapes*, the failure-mode-driven curation loop that ties profiling → mining →
> training, a map of how existing code makes each family cheap to add, the
> AnyMatch-first build order, and a backlog of methods to try.
>
> **Reads with:**
> - `docs/research/20260707_data_prep_hard_case_mining_survey.md` — the **literature**
>   (what methods exist; the S1–S6 taxonomy, label-noise trap, augmentation, LLM-era
>   selection, the blocking/embedding architecture). This doc is the *design* over that
>   survey.
> - `docs/plans/20260715_data_layer_plan.md` — the **buildable seam** (the composable
>   `DataProfileReport` — shipped as PR-1 — and the PR-2 mining seam). This doc gives that
>   plan its conceptual frame and its backlog.
> - `docs/research/20260714_training_surface_design.md` §5 — the **agreed miner contract**
>   (plain functions → `LabeledPair`) this design builds on.

---

## 0. Framing — from "let the model write the prompt" to "let the data shape the model"

The inspiration for prompt-optimization in langres is Drew Breunig's talk **"Let the
Model Write the Prompt"** ([dbreunig.com, 2025-06-10](https://www.dbreunig.com/2025/06/10/let-the-model-write-the-prompt.html)).
Its thesis — *"Don't program your prompt. Program your program."* — is that hand-written
production prompts are brittle, model-dependent, and mostly formatting boilerplate (his
dissection of the OpenAI SWE-Bench prompt: 1% task definition, 19% CoT, 32% formatting);
you should instead specify a task (a DSPy signature + metric) and let an optimizer
(MIPROv2) write the prompt. His **entity-resolution example** is the anchor: ~1,000
labeled place pairs (Alameda County restaurant inspections × Overture Maps), tiny models
(Qwen3-0.6B, Llama-3.2-1B, Phi-4-Mini-3.8B), binary-match accuracy — **Qwen3-0.6B goes
60.7% → 82%** after MIPROv2, with *no model change*.

The deeper claim in that talk is the one that matters here: **"your evaluation data is
your most valuable AI asset."** Prompt optimization is really *data-centric* optimization
wearing a prompt-shaped hat — the optimizer is only as good as the labeled pairs and the
metric you hand it. langres already ships the prompt half (`DSPyMatcher` wrapping
MIPROv2). This document is about the generalization:

> **The training/curation data is the asset across *all three* optimization targets** —
> the prompt (DSPy), the fine-tuned matcher (AnyMatch), and the blocker's embedding
> model — and it deserves a first-class, composable **data-preparation layer**, designed
> as deliberately as the model layer.

AnyMatch is the proof the bet pays: its model is a 124M GPT-2; every win comes from the
**data recipe** (hard-positive mining, attribute augmentation — its single largest
ablation at −3.14 F1 — label balancing, serialization). The lever is the data. So we
invest the architecture there.

---

## 1. The organizing idea — data prep is a small set of operation *shapes*

The survey lists ~30 methods. A pile of 30 miners is unmaintainable (the exact
single-use-code trap the project guards against). The insight that makes this a *system*:
**every data-prep method is one of a handful of operation shapes over the same
`LabeledPair` currency.** Design the shapes once; each method is then a small pure
function slotting into a known shape, and the profiler and training surface never change
(open/closed, exactly like the shipped `ProfileSection` seam).

| Shape | Signature (conceptual) | What it does | Method families (§3) |
|---|---|---|---|
| **profile** | `data → Report` | *measure* the data; surface failure modes, imbalance, separability, coverage | `DataProfileReport` (shipped PR-1) |
| **select** | `pairs → pairs` (subset) | keep the *informative* existing pairs | hard-positive, blocking-derived hard-neg, difficulty (EL2N), coreset/coverage |
| **acquire** | `pairs → pairs` (rank for labeling) | pick which pairs a human/teacher labels next (active learning) | uncertainty, committee (shipped), **BADGE** |
| **generate** | `pairs → pairs` (new) | *synthesize* new pairs | Ditto/AnyMatch operators, LLM entity-variant synthesis |
| **relabel** | `pairs → pairs` (labels transformed) | *produce or clean* labels | denoise (confident-learning), weak-supervision label model, **teacher-margin soft labels** |
| **mix** | `sources → weights` | weight/select *across source datasets* | DoReMi-style reweighting (generalist only) |

`acquire` is formally a `select` that ranks rather than filters, but it earns its own row
because it feeds a *human/teacher* loop, not a training set directly. `profile` is not a
producer of pairs — it is the **sensor** the whole loop is steered by (§2).

**The currency is `LabeledPair`** (`core/harvest.py:113`) — printable, inspectable,
order-independent. Every shape except `profile`/`mix` consumes and produces it. This is
already the agreed miner contract (training-surface §5.1: "miners as plain functions, not
strategy objects"). This doc's contribution is naming the *shapes* so the menu of methods
has a home and additions are mechanical.

---

## 2. The spine — the failure-mode-driven curation loop

The shapes are ingredients; the **loop** is the system. The most valuable data prep isn't
"mine hard pairs" in the abstract — it's **find where the model fails, ask how those
failures differ from where it succeeds, and close that specific gap.** This is active
learning lifted from the *per-pair* level to the *slice/failure-mode* level, and it's
what turns a bag of miners into a data engine.

```
        ┌──────────────────────────────────────────────────────────────┐
        │                                                                │
        ▼                                                                │
   ┌─────────┐   run current      ┌──────────────────┐   compare        │
   │ PROFILE │──  matcher, ─────► │  DISCOVER        │──  failures ─────►│
   │ (data)  │   diff vs gold     │  failure modes   │   vs successes   │
   └─────────┘                    └──────────────────┘        │         │
        ▲                          (slice the errors)         ▼         │
        │                                          ┌────────────────────┐│
        │  re-profile: did the                     │  ANALYZE           ││
        │  gap actually close?                     │  distribution      ││
        │                                          │  → find imbalance  ││
        │                                          └────────────────────┘│
        │                                                     │          │
   ┌─────────────────┐   feed training     ┌─────────────────▼─────────┐│
   │  TRAIN          │◄──  (fine-tune /     │  FIX the gap:             ││
   │  matcher/       │     demos /          │  select │ generate │      ││
   │  embedder/      │     embedder)        │  relabel │ acquire        ││
   │  prompt         │                      └───────────────────────────┘│
   └─────────────────┘                                                   │
        └────────────────────────────────────────────────────────────────┘
```

**Concretely, in langres seams:**

1. **PROFILE** — `DataProfileReport` (shipped) measures label structure, per-field stats,
   separability (positives vs negatives), embedding distribution. This is the baseline
   *sensor*.
2. **DISCOVER failure modes** — run the current matcher, harvest its errors from
   `JudgementLog` against gold (false negatives = missed matches, false positives = wrong
   merges), then **slice** those errors: by field-emptiness, by source, by gold
   cluster-size, by string-distance band, by embedding neighborhood. *(New capability: a
   `FailureModeSection` profiler — see §4.)*
3. **ANALYZE distribution** — compare the *error* distribution to the *success* / overall
   distribution across those slices. The signal is **imbalance**: e.g. "70% of false
   negatives are abbreviation/acronym cases, but those are 4% of the training pairs," or
   "the blocker misses all cross-language pairs." This is the data-centric diagnosis the
   maintainer described: *see how the failure modes distribute w.r.t. the ones that
   succeed, and find the fixable imbalance.*
4. **FIX the gap** — pick the shape that matches the diagnosis:
   - imbalance of a *real, present-but-rare* slice → **select** more of it (mine).
   - a slice that's *absent/too sparse to mine* → **generate** it (augment: synthesize
     abbreviation variants, transliterations, dropped-field versions).
   - the slice is actually **mislabeled** (the label-noise trap, survey §4) → **relabel**
     (denoise) rather than train on it.
   - not enough labels to even diagnose → **acquire** (send that slice to the review
     queue / teacher).
5. **TRAIN** — feed the rebalanced set to the matcher fine-tune, the DSPy demo pool, or
   the blocker-embedder contrastive fine-tune.
6. **RE-PROFILE** — run the loop again; did the targeted gap actually close, and did it
   cost recall/precision elsewhere? (The survey's §13 rule: **evaluate every change on
   both stages — blocking recall *and* matching F1 — not just the one it targets.**)

This loop is the reason the profiler (PR-1) and the miners (PR-2) were planned as *one
arc*: the profiler is not a reporting nicety, it is the **control signal** for curation.
The loop reuses PR-1 (sensor) + PR-2 (actuators) + the augmentation operators; the only
genuinely new piece it needs is the **error-slicing view** (§4).

---

## 3. The method menu, mapped to shapes

The survey (§3–§10, §11–§12) is the authoritative literature; this is the **index by
shape**, folding in the newly-verified deltas (2026-07-15 research pass). Cost tags:
`◇` no training loop **and** no paid LLM (cheapest), `◆` needs a training loop or
gradients, `$` needs a paid/strong LLM. Confidence flags where a claim wasn't fully
verified.

### 3.1 `select` — keep the informative existing pairs

| Method | Signal | Cost | langres reuse | Ref |
|---|---|---|---|---|
| **AnyMatch hard-positive** | positives a cheap classifier misclassifies | ◇ | `RandomForestMatcher` (`[trained]`) | 2409.04073 |
| **Blocking-derived hard-neg** | non-match that clears the blocker threshold | ◇ | `Blocker`/`VectorBlocker` scores + `JudgementLog` | survey §S3 |
| **EL2N difficulty** | `\|1 − p(gold)\|` from any probabilistic judge | ◇ | `JudgementLog.score` — no extra training | 2107.07075 |
| **Coreset / herding / moderate-coreset** | geometric coverage near the median | ◇ | reuse `VectorBlocker` embeddings | 1708.00489 |
| **SemDeDup** | drop embedding near-duplicates | ◇ | reuse `VectorBlocker` embeddings | 2303.09540 |
| **Prototypicality pruning** | distance to SSL cluster centroids; *pruning beats power-law* | ◆ | reuse embeddings + k-means | 2206.14486 (high) |
| **Cartography (ambiguous region)** | high cross-epoch *variability* (not just low conf) | ◆ | needs per-epoch logging in the fine-tune loop | survey §S4 |

### 3.2 `acquire` — active learning (which to label next)

| Method | Signal | Cost | langres reuse | Ref |
|---|---|---|---|---|
| **Uncertainty / margin** | low `\|score − threshold\|` | ◇ | ✅ `_select_uncertainty` (`core/review.py`) | — |
| **Query-by-committee** | judge disagreement | ◇ | ✅ `_select_disagreement` | DIAL 2104.03986 |
| **Confident-merge audit** | governance slice over confident merges | ◇ | ✅ `select_for_review` audit | — |
| **BADGE** | gradient-embedding magnitude (uncertainty) × k-means++ (diversity) — **one** acquisition | ◆ | **margin already computed + embeddings already available** → add k-means++ seeding | 1906.03671 (high) |
| **Cluster-Margin** | low-margin points, cluster, round-robin for diversity (scales to 100M) | ◇ | margin + a clustering pass | 2107.14263 (med) |

BADGE is the headline "we already have most of the code" case — see §4.

### 3.3 `generate` — synthesize new pairs (augmentation)

| Method | Mechanism | Cost | langres reuse | Ref |
|---|---|---|---|---|
| **Ditto operators** | `span/attr_del`, `shuffle`, `entry_swap`; MixDA | ◇ | perturb `Comparator`/`ColVal` inputs | 2004.00584 |
| **AnyMatch attr / flip / permute** | single-attr pairs; L↔R swap; attr reorder | ◇ | `attribute_examples`, `flipped` (agreed miners) | 2409.04073 |
| **LLM entity-variant synthesis** | LLM writes abbreviations/typos/transliterations of a record; matcher confirms as consistency filter | $ | `LLMMatcher` is the natural filter | InPars/Promptagator 2202.05144 / 2209.11755 (high) |
| **SyNeg / Syntriever** | LLM self-reflection writes hard *negatives* | $ | densify scarce hard non-matches | 2412.17250 / 2502.03824 |

### 3.4 `relabel` — produce or clean labels

| Method | Mechanism | Cost | langres reuse | Ref |
|---|---|---|---|---|
| **Confident Learning / Cleanlab** | per-class-calibrated confident joint over CV probs; ranks likely label errors | ◇ | `denoise_pairs` built-in (sklearn); Cleanlab optional | 1911.00068 |
| **Weak supervision / data programming (Snorkel)** | many noisy labeling functions → learned label model → probabilistic labels | ◇ | combine `StringComparator` sims, blocking keys, cheap judge | 1605.07723 / 1711.10160 (high) |
| **Teacher-margin soft labels (Margin-MSE / TAS-B)** | a strong cross-encoder scores pairs; those margins *are* the training labels for a cheap bi-encoder | ◆ | **`CascadeMatcher`/`LLMMatcher` margins in `JudgementLog` → contrastive fine-tune the `VectorBlocker` embedder** | 2010.02666 / 2104.06967 (high) |
| **LLM-as-judge silver filtering** | confidence/self-consistency/agreement gate on silver labels | $ | ✅ already an in-repo instance: `CascadeMatcher` + `JudgementLog` | AlpaGasus 2307.08701 |

`harvest_labeled_pairs` (`core/harvest.py`) already does the single-judge version of
`relabel`; Snorkel is the principled *multi-labeler* generalization.

### 3.5 `mix` — weight across source datasets (the generalist question)

AnyMatch's whole claim is **leave-one-dataset-out zero-shot**: train on 8, generalize to
a held-out 9th. The curation lever there is *which sources and at what ratio* — a question
the survey's *within-dataset* coverage work never touches.

| Method | Mechanism | Cost | Relevance | Ref |
|---|---|---|---|---|
| **DoReMi** | proxy model + group-DRO learns per-domain mixture weights | ◆ | which of the 8 EM sources to upweight for the held-out target | 2305.10429 (high) |
| **DoGE** | gradient-alignment: score each source by how its gradient helps the target domain | ◆ | closest to LODO source-selection | 2310.15393 (med) |
| **Data Mixing Laws** | fit loss-vs-mixture on small runs, extrapolate the optimal mixture | ◆ | predict LODO generalization for an unseen ratio | 2403.16952 (med) |
| **Unicorn** | one MoE encoder across many matching datasets; explicit zero-shot to a new task | ◆ | the EM-native precedent/baseline for a generalist matcher | SIGMOD 2023 (med, no arXiv) |

> **Open research gap (both verifiers flagged it):** nobody does *pairwise
> dataset-similarity-ranked source selection* for zero-shot EM LODO, and there's no
> EM-specific DoReMi. AnyMatch itself does difficulty-filtering + label-balancing, **not**
> mixture reweighting. This is the one item that is both *on AnyMatch's critical path* and
> a potential publishable contribution — a research bet, not a build item (§5).

### 3.6 Training-time data strategies (not a shape — a knob on `train`)

Curriculum ordering isn't a data *transform*, it's an *ordering* of the fed set:
**competence-based curriculum** (Platanios et al., `1903.09848`, med) admits pairs only
when model competence ≥ their difficulty — a natural consumer of the EL2N/cartography
difficulty scores. Cheap to try once a fine-tune loop exists; parked until it does.

---

## 4. "We already have most of the code" — the reuse map

The point of the shapes taxonomy is that new methods are *small*. For each high-value
candidate, here is the existing seam it plugs into and the **only** delta to write:

| Method | Existing seam it reuses | The small delta to write |
|---|---|---|
| **BADGE** (`acquire`) | `select_for_review` already ranks by margin; `VectorBlocker` already produces per-record embeddings | pair-embedding features + **k-means++ seeding** over (margin-scaled) embeddings. ≈ one function; no new dep (numpy) |
| **Blocking-derived hard-neg** (`select`) | `Blocker.candidates` + their scores; `JudgementLog` verdicts | filter: `score ≥ threshold ∧ label = non-match`. Pure reuse |
| **EL2N difficulty** (`select`) | `JudgementLog.score` is `p(match)` | compute `\|1 − p(gold)\|`, sort. No training |
| **Matcher→blocker distillation** (`relabel`→embedder) | `CascadeMatcher`/`LLMMatcher` already score pairs into `JudgementLog`; `[semantic]` has sentence-transformers | wrap those margins as `MarginMSELoss` targets; a `TrainedEmbedder` sibling to `TrainedLMMatcher` |
| **Confident-learning denoise** (`relabel`) | `RandomForestMatcher` / sklearn already in `[trained]` | k-fold CV → confident joint. The planned `denoise_pairs` |
| **Snorkel label model** (`relabel`) | `StringComparator` sims, blocking keys, cheap judge are all cheap labeling functions | a light label-model aggregator (or FlyingSquid closed-form) over their votes |
| **Coreset / SemDeDup / prototypicality** (`select`) | `VectorBlocker` embeddings | k-center-greedy / near-dup / centroid-distance over existing vectors |
| **FailureModeSection** (`profile`) | `EvalReport` already slices + histograms; `DataProfileReport` is the composable seam | a new `ProfileSection` that groups errors by slice and diffs error-vs-success distributions |

**BADGE, specifically** (the maintainer's example): the two ingredients BADGE combines —
*uncertainty* (it uses last-layer gradient magnitude; our margin `|score − threshold|` is
a serviceable proxy) and *diversity* (k-means++ over those gradient embeddings) — are both
already in the codebase. `_select_uncertainty` gives the uncertainty ranking; the
`VectorBlocker` embeddings give the diversity space. BADGE ≈ *seed k-means++ over
margin-weighted pair-embeddings*, a single acquisition function added next to the existing
`_select_*` strategies in `core/review.py`. That's the payoff of designing the shapes: the
literature's "advanced" method is a ~30-line addition because the substrate is already
there.

---

## 5. Build order — AnyMatch first, architecture-aware from the start

**Near-term concrete target: reproduce AnyMatch.** That needs exactly the planned Wave 3
(PR-2) miners plus the serializer, nothing from §3.5/§3.6:

- `mine_misclassified` (hard-positive) · `sample_negatives` (2:1 random) ·
  `attribute_examples` · `flipped` · `denoise_pairs` (the required label-noise rail) ·
  `ColVal` serializer · `MiningReadinessSection` → then **Stage 3**: the leave-one-out
  generalist harness over the 10-dataset registry (F1 81.96 reference).

Everything else in §3 is a **layered improvement**. The design guarantee that makes
"AnyMatch first, rest later" safe:

> **Every method in §3 is a new pure function of a known shape (§1). Adding one never
> touches the profiler, the training surface, or the other miners.** The taxonomy + the
> `LabeledPair` contract + the failure-mode loop are the *only* things that must be right
> up front; the menu fills in incrementally without refactor.

Suggested sequence after AnyMatch (cheapest reusable first — the survey's bottom-left
quadrant), each gated on "measure the lift, don't assume it":

1. **`◇` cheap wins, no new deps:** blocking-derived hard-neg; EL2N difficulty;
   `FailureModeSection` (unlocks the §2 loop); SemDeDup/coreset over existing embeddings.
2. **`◇/◆` acquisition + coverage:** BADGE (§4); Snorkel-style label model.
3. **`◆` the blocker-embedding target:** matcher→blocker margin distillation +
   `TrainedEmbedder` — the langres-native data engine that serves the third training
   target with one new seam.
4. **research bets (separate epic, autoresearch #145 / #85):** the `mix`/generalist gap
   (DoReMi-style source reweighting; the open similarity-ranked-source-selection
   contribution); curriculum ordering.

---

## 6. Backlog — things we could implement and try

A checkable menu. Each item: the shape, the cost tag, and **what to measure** (never ship
a mining strategy without a metric — the survey's discipline). Ordered roughly by
value-per-effort.

**Cheap, no new deps (`◇`)**
- [ ] `FailureModeSection` profiler — slice matcher errors, diff error-vs-success
      distributions. *Measure:* does it surface a real imbalance on Abt-Buy / Amazon-Google?
- [ ] Blocking-derived hard-negatives miner (`mine_hard_negatives`). *Measure:* matcher F1
      vs. random negatives at equal count.
- [ ] EL2N difficulty scoring on the review queue. *Measure:* correlation with human
      "hard" labels; overlap with margin.
- [ ] SemDeDup / moderate-coreset over `VectorBlocker` embeddings. *Measure:* F1 at 50%
      data retained vs. full.
- [ ] The §2 loop end-to-end on one benchmark (profile → slice → generate the missing
      slice → re-train → re-profile). *Measure:* did the targeted slice's error rate drop
      without hurting others?

**Medium (`◆` / `$`)**
- [ ] BADGE acquisition next to `_select_uncertainty`. *Measure:* labels-to-target-F1 vs.
      pure uncertainty and vs. committee.
- [ ] Snorkel/FlyingSquid label model over cheap labeling functions. *Measure:* silver-label
      accuracy vs. single cheap judge, on gold.
- [ ] Matcher→blocker margin distillation (`TrainedEmbedder` + `MarginMSELoss`).
      *Measure:* blocking Pair-Completeness of the distilled embedder vs. off-the-shelf,
      **and** that matching F1 is unaffected (transfer check, survey §13).
- [ ] NV-Retriever positive-aware hard-neg mining for the embedder (drop negatives scoring
      within ~95% of the positive). *Measure:* false-negative rate in mined negatives.
- [ ] LLM entity-variant augmentation with matcher-as-consistency-filter. *Measure:* F1 on
      the targeted failure slice; watch for synthetic-artifact bias.
- [ ] Curriculum ordering by difficulty in the fine-tune loop. *Measure:* convergence
      speed + final F1 vs. shuffled.

**Research bets (separate epic)**
- [ ] DoReMi/DoGE-style source reweighting for the LOO generalist. *Measure:* held-out
      F1 vs. AnyMatch's uniform pooling.
- [ ] **Novel:** similarity-ranked source selection for zero-shot EM LODO (no prior work
      found). *Measure:* does target-similar source weighting beat uniform on the held-out
      dataset?
- [ ] GISTEmbed-style guided in-batch negative filtering (matcher as guide) for the
      embedder. *Measure:* embedder recall vs. unfiltered in-batch.

---

## 7. Open design decisions

1. **Where the failure-mode loop lives.** A thin `curate` / `diagnose` facade that wires
   profile→slice→fix, or just loose composable functions the user chains? *Lean:* start as
   loose functions (matches the miner ethos), add a facade only once the loop's shape is
   proven — don't pre-abstract (simplicity-first rule).
2. **Miner namespace.** The plan flagged `langres.eval` (with reports) vs. a dedicated
   `langres.data` / training surface. *Lean:* miners on the training/`core` surface next
   to `harvest`/`align_pairs`; the profiler on `langres.eval`. Revisit if a `langres.data`
   facade earns itself.
3. **Is `mix`/generalist in scope for the seam, or a separate harness?** *Lean:* separate
   — it operates on *sources*, not `LabeledPair`s, so it's a different currency and a
   different epic (Stage-3 LOO harness / autoresearch).
4. **The blocker-embedding training home.** A `TrainedEmbedder` sibling to
   `TrainedLMMatcher` (same fit protocol), fed by the distillation `relabel` step. Confirm
   the fit-protocol shape covers both a classification head (matcher) and a contrastive
   objective (embedder).
5. **Cleanlab as a dep.** Built-in confident-learning first (sklearn, no new dep); Cleanlab
   only if the built-in underperforms (plan §9, already leaning built-in).

---

## 8. References

Primary sources are cited inline by arXiv ID; the consolidated bibliography and the
per-theme reference spine live in `docs/research/README.md`. The load-bearing anchors for
*this* doc:

- **Breunig 2025** — *"Let the Model Write the Prompt"*, [dbreunig.com](https://www.dbreunig.com/2025/06/10/let-the-model-write-the-prompt.html)
  (the DSPy-in-ER inspiration; the "data is the asset" framing).
- **AnyMatch** — Zhang et al. 2024, [2409.04073](https://arxiv.org/abs/2409.04073) (the
  near-term reproduction target).
- **The #86 survey** — `docs/research/20260707_data_prep_hard_case_mining_survey.md` (the
  full literature; S1–S6 taxonomy, label-noise trap §4, blocking/embedding architecture
  §11–12).
- **The data-layer plan** — `docs/plans/20260715_data_layer_plan.md` (the buildable
  profiler + mining seam).
- New-delta anchors (verified 2026-07-15): Margin-MSE/TAS-B [2010.02666](https://arxiv.org/abs/2010.02666)/[2104.06967](https://arxiv.org/abs/2104.06967);
  NV-Retriever [2407.15831](https://arxiv.org/abs/2407.15831); BADGE [1906.03671](https://arxiv.org/abs/1906.03671);
  Snorkel [1711.10160](https://arxiv.org/abs/1711.10160); DoReMi [2305.10429](https://arxiv.org/abs/2305.10429);
  prototypicality pruning [2206.14486](https://arxiv.org/abs/2206.14486).
