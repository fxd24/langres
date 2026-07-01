<!-- Generated 2026-07-01 by the er-research-map ultracode workflow (18 scouted targets, 17 method deep-dives + 3 cross-cutting lenses; 1 deep-dive dropped on a structured-output retry cap). Forward-looking strategic input; not a commitment. -->

# Entity Resolution Research Landscape — A Seam Audit for langres

*Synthesis of 17 method deep-dives + 3 cross-cutting analyses. Anchored on langres's core claim: because langres is a composable seam (blocker × comparator × judge × clusterer × optimizer), any combination should be expressible. This report finds exactly where that holds and where it breaks against SOTA.*

---

## 1. Executive Summary

**State of the field.** Entity resolution has passed through four eras that still coexist in every benchmark table: (1) probabilistic Fellegi–Sunter linkage (Splink, Dedupe, Zingg, Senzing — the production workhorses); (2) classical feature-vector ML (Magellan random forest); (3) fine-tuned deep/transformer matchers (DeepMatcher → Ditto → HierGAT → contrastive Sudowoodo/SC-Block — the supervised accuracy ceiling); and (4) prompted/instruction-tuned LLMs (Peeters & Bizer, Jellyfish, ComEM, LLM-CER — the current zero-shot frontier). The center of gravity has moved decisively to **pretrained representations**: GPT-4-class judges hit 86–99 F1 zero-shot on the classic suite with no task labels and generalize under distribution shift where fine-tuned specialists collapse 22–61 F1 (arXiv:2310.11244). Simultaneously, the *unit of matching* is shifting from the pair to the **set** — listwise "Select" beats pairwise "Match" by +16 F1 *and* costs ~⅓ as much (ComEM, arXiv:2405.16884), and in-context clustering fuses block+match+cluster into one call (LLM-CER, +150% accuracy / 5× fewer calls).

**Honest verdict on the seam claim.** langres's seam **fully expresses the pairwise, pretrained/prompted family** — this is a real and defensible position, and its own M3 result (gpt-4o pair-F1 0.667 on hard Amazon-Google, beating a free embedder at 0.471) sits squarely in the literature band. But "any combination is expressible" is **false against five SOTA shapes**, in rough order of how central they are to the field:

1. **Trainable judges & encoders** — the single most-cited SOTA family (Magellan RF, DeepMatcher, Ditto, Sudowoodo, SC-Block, DIAL, and even label-free ZeroER/Fellegi–Sunter-EM) all need a `fit()` loop. langres's Module is forward-only; the Optimizer is Optuna-over-blocker-hyperparams only. **This is the biggest honest gap.**
2. **Set-wise / groupwise judging** — the strongest *and cheapest* current LLM methods (ComEM Select/Compare, LLM-CER) score an anchor against a candidate *set*. The `ERCandidate = frozenset{left,right}` + `Module.forward: pair → PairwiseJudgement` contract structurally cannot receive them. Notably the *output* still decomposes to PairwiseJudgements, so the fix is additive.
3. **Iterative / stateful control flow** — collective relational ER (Bhattacharya–Getoor), belief-refinement (BoostER), and active-learning loops (DIAL, risk sampling) need a feedback loop the single-pass DAG has no home for.
4. **Incremental, value-frequency-aware production matching** — Senzing/Zingg/Splink weight matches by *live corpus rarity* ("Smith" vs a rare surname) and support merge/split with stable IDs. langres's judge is blind to store statistics and its clusterer emits fresh connected-components per run.
5. **Merge-resistant clustering** — transitive-closure over-merges (langres's own M3 saw −0.63 BCubed from one chained false edge). SOTA uses correlation/constraint-aware clustering.

**Top 5 actions (detail in §4 and §8):**

1. **Generalize the judgement contract** to `PairwiseJudgement | SetJudgement/PartitionJudgement`. Highest-leverage single change — unlocks ComEM Select, LLM-CER, and the field's biggest cost lever (set-batched calls) at once. *(cheap, high-leverage)*
2. **Add an optional `fit()` / `fit_unlabeled()` hook to Module + a second Optimizer shape `compile(student, trainset, metric) → Module`.** Turns the seam from "compose configured scorers" into "train a scorer" — unlocks Magellan RF, Fellegi–Sunter EM, ZeroER, DSPy/MIPRO, Snorkel label models. *(cheap–medium)*
3. **Ship a value-frequency-aware Fellegi–Sunter judge** (m/u weights + term-frequency adjustment). Highest-ROI accuracy change for the multilingual-Person north star; answers ROADMAP open-question §8.1 with the evidence-backed answer (value-level, *not* richer comparison levels). *(cheap)*
4. **Make the Clusterer pluggable + default to merge-resistant**, and make the harness emit **per-method scaling curves** (label→F1 power-law fit) + GMD + reduction-ratio. Both are ~small lifts over what M3 already produces. *(cheap)*
5. **Adopt a 6-dataset external portfolio + OpenSanctions Pairs** as the Person benchmark, and **enrich the LLM judge** with a demonstrations slot + example selector + rules block + richer provenance. *(cheap–medium)*

**One strategic reframing:** stop selling "any combination possible" unqualified. langres is *the composable seam over the pairwise ER family, plus (once the contract generalizes) the set-wise family, plus a serializable artifact.* That is honest, defensible, and something Splink/Dedupe/Zingg (all training-required, all pairwise-transitive) cannot say — because langres's zero-shot LLM-judge → cheap-distilled-student path is a genuine strategic edge.

---

## 2. The Field Map

Six directions organize the landscape. The unit of work (pair vs set vs block), whether the method *trains*, and whether it needs *labels* are the axes that matter for the seam.

| Method / family | Era | How it's built & deployed | Accuracy band (Amazon-Google F1 unless noted) | Cost | Data needs |
|---|---|---|---|---|---|
| **Fellegi–Sunter / Splink** | 1969 / 2020s prod | Discrete comparison levels; per-level weight ω=log₂(m/u); m by EM, u by random sampling — **label-free**; TF adjustment; connected-components at threshold; SQL/Spark | High-0.9s on clean multi-field identity data; weak on single-text-blob (AG) | Near-zero; CPU; 1M rec/min laptop, 100M+ Spark | **No labels**; needs multiple low-correlation columns |
| **Magellan (py_entitymatching)** | 2016 | Auto multi-metric-per-attribute feature vector → sklearn RF (default); pickled estimator | AG 49.1; structured avg **88.8** (beats early DL) | CPU, 1–4 min train | 100s–10Ks labeled pairs |
| **ZeroER** | 2020 | 2-component GMM (match/non-match) over similarity vector, fit by EM; ER-specific regularization + transitivity | AG **0.48**; FZ 1.00, DBLP-ACM 0.96; avg 0.76 ≈ supervised RF | CPU, 10–100× faster than DL | **Zero labels** |
| **DeepMatcher** | 2018 | 4-module DL template (embed→summarize {SIF/RNN/Attn/Hybrid}→compare→classify), fastText/GloVe | AG **69.3**; wins big on dirty (+19.4) / textual (+19.2) | GPU 1–11h (~100–400× Magellan) | 450–28.7K labeled pairs, 3:1:1 |
| **Ditto** | 2020 | RoBERTa cross-encoder fine-tune + augmentation + domain-knowledge injection | AG **~75**; Abt-Buy ~89; SOTA at ≤½ the labels | GPU minutes–hours; cheaper than DeepMatcher | ~20% of ER-Magellan labels |
| **HierGAT** | 2022 | BERT + hierarchical graph-attention (token→attr→entity); "global interdependence" | AG ~75–79; best on dirty; +8.7 F1 vs Ditto | GPU ~hours; per-pair fwd at inference | 100s–10K labeled; data-efficient |
| **Sudowoodo** | 2023 | **One** contrastive RoBERTa shared by blocker (kNN) + matcher (fine-tune); label-light | AG 59.3 @500 labels / 59.1 @**0 labels**; avg 78.3 @500 (=Rotom @750) | 1-time contrastive pretrain; cheap offline inference | 500 or **0** labels |
| **Jellyfish** | 2024 | Instruction-tune local 7B/8B/13B; knowledge injection + reasoning distillation; vLLM serve | AG **81.3** (beats GPT-4 74.2); unseen Abt-Buy 89.6 | LoRA 24–40 A100-h once; ~0.1s/pair local, $0/call | ~38K instruction examples (one-time); English tabular only |
| **LLM prompting (Peeters & Bizer)** | 2023 | Serialize pair → forced MATCH/NO_MATCH; demo selection + rule injection | GPT-4 zero-shot mean **86.8**; +7.85 from related-demo selection | 79× cost span; 6-shot 5.3×, rules 1.8× tokens | **Zero–few labels** |
| **ComEM (Match/Compare/Select)** | 2025 | Listwise Select over candidate set; compound cheap-filter→strong-select funnel | Select mean **81.6** vs Match 64.0; ComEM 85.6 | **Cheapest** ($0.92 vs Match $4.52) — quality & cost both win | Zero-shot |
| **LLM-CER (in-context clustering)** | 2025 | Pack ~9 records/prompt, LLM partitions → fuses block+match+cluster | +150% vs pairwise-batching baseline | 5× fewer API calls | Zero-shot |
| **DeepBlocker** | 2021 | Self-supervised tuple embedding (AE/CTT/Hybrid) via token-dropout positives; FAISS top-K | Recall ~0.90–1.0 at small candidate sets | Cheap; shallow MLP + fastText, minutes, **no labels** | **Zero labels** |
| **SC-Block / NLSHBlock / UniBlocker** | 2023–24 | Supervised-contrastive RoBERTa encoder (or LSH-loss / universal dense) + ANN | 99.5% recall @k=5 Abt-Buy; ~10–15% over prior dense | ~5 min encoder fine-tune; cuts WDC pipeline 30h→8h | Labeled pairs (reused from matcher) / self-sup |
| **DIAL / risk sampling** | 2021–22 | Co-trained blocker+matcher, FAISS-committee, uncertainty/BADGE/risk AL loop | AG 82.1, multilingual En-De 74.3 (all-pairs F1) | GPU, train-per-round; 5–10× fewer labels | ~1280 labels via AL loop |
| **Collective relational (CR-ER)** | 2007 | Greedy agglomerative; sim = (1−α)·simA + α·simR over **current cluster labels**; joint inference | CiteSeer 0.995, BioBase 0.818 (pairwise F1) | Unsupervised, in-memory, modest constant factor | **No labels**; needs relational graph |
| **BoostER** | 2024 (demo) | Belief distribution over partitions; entropy-optimal LLM queries; Bayesian update w/ oracle-accuracy Θ | *No eval section — architectural contribution only* | Budget-bounded LLM spend; no training | Base matcher + LLM key |
| **Weak supervision (Snorkel + ER label model)** | 2018–23 | Noisy labeling functions (w/ ABSTAIN) → generative denoiser → prob labels → distill end model | Within ~3.6% of hand-labeled; ER-transitivity model +9% F1 | Near-zero, API-free, CPU seconds | **No gold labels** (LF authoring) |

**Benchmarks that define the tables:** Magellan/DeepMatcher suite (Structured/Dirty/Textual — DBLP-ACM, DBLP-Scholar, Amazon-Google, Walmart-Amazon, Abt-Buy, Beer, iTunes-Amazon, Fodors-Zagat); **WDC Products** (27 variants over corner-case% × unseen-entity% × dev-size — the modern generalization benchmark); Alaska (heterogeneity); Machamp (7 GEM tasks); **OpenSanctions Pairs** (755K multilingual Person/Org pairs — near-purpose-built for langres's north star).

---

## 3. langres Coverage & Gap Matrix

`can_replicate` = whether the seam can express the method *as the paper builds it* (not merely wrap it as an opaque adapter). Ranked by importance to the field.

| Method | Replicate? | Via which primitives | Core gap | Minimal change to accommodate |
|---|---|---|---|---|
| **LLM prompting judge** | **partial (inference yes, levers no)** | LLMJudge (ships zero-shot Match); Bootstrapper gold as demo pool; benchmark harness | No `{demonstrations}` slot, no example-selector, no `{matching_rules}` block; provenance can't audit the *input* prompt (only output reasoning) | Add demonstrations + rules slots, an `ExampleSelector` strategy, and prompt_variant/demo-ids/rules-hash to provenance |
| **ComEM Select/Compare** | **no (input side)** | Match = native; Clusterer consumes decomposed output unchanged | `ERCandidate` is strictly binary; no `(anchor,[candidates])` type; `Module.forward` is hardwired pairwise | Add `ERCandidateGroup` + a `GroupwiseModule` that still yields `PairwiseJudgement`; Blocker `stream_groups()` |
| **LLM-CER in-context clustering** | **no** | — | Fuses block+match+cluster; needs a partition output the clusterer merely stitches | `PartitionJudgement` + a CollapsedResolver method |
| **Ditto / DeepMatcher / HierGAT** | **partial (wrap-only)** | Module can host a torch model as opaque adapter; Blocker/Clusterer unchanged | No trainable cross-encoder judge; no `fit()`; config-registry artifact can't carry weights; DL design-space not native | Trainable Module contract + weight-reference in artifact; accept these are wrapped, not reproduced |
| **Magellan RF** | **partial** | Comparator (per-attr), Module slot, harness pair-track maps 1:1 to fixed splits | No multi-metric-per-attribute Comparator; no fittable sklearn classifier judge; FeatureSpec can't carry similarity-fn list | Multi-metric Comparator + `SklearnJudge.fit(gold)` serialized via config |
| **Fellegi–Sunter / Splink** | **partial** | Comparator emits discrete levels (1:1 with FS levels); Clusterer=connected-components; artifact≈JSON settings | No weight-learning: weights are static author-set averages, not additive log₂(m/u); no EM; no TF adjustment; no possible-match band | `fit()` hook + `FSJudge` (EM + random-sampling u) + per-value freq table + two-threshold clusterer |
| **ZeroER** | **partial** | New `ZeroERJudge` holds GMM; Comparator vector; Clusterer for transitivity | No **unsupervised** judge-fit lifecycle (everything gates on gold); Module is per-pair, ZeroER is corpus-global batch; missing-neutrality conflicts with GMM-over-complete-vector | `fit_unlabeled(candidates)` hook (default no-op) + continuous-vector Comparator mode |
| **Sudowoodo** | **partial** | Point VectorBlocker + embedding-Module at same checkpoint (inference topology) | No self-supervised contrastive training stage; no shared-encoder object; no cross-encoder judge | `EncoderModel` shared-reference object; delegate contrastive training to sentence-transformers via adapter |
| **DeepBlocker** | **partial** | VectorBlocker + FAISS = ExactTopKVectorPairing; EmbeddingProvider is the plug point | No `fit()` on Blocker/EmbeddingProvider; synthetic-pair generator has no home; no learned-weight serialization | `TrainableEmbeddingProvider.fit(texts)` called by `create_index()`; SerializableState embedder |
| **SC-Block / NLSH / UniBlocker** | **partial** | QdrantHybridIndex (dense+sparse RRF/DBSF), reranking (ColBERT), serializable embedder | Can host but not *produce* the contrastive/LSH encoder; no union/ensemble Blocker; no BlockingPy adapter | `ContrastiveEncoderTrainer` (SupCon/MNRL over gold) + `UnionBlocker` + BlockingPy adapter |
| **DSPy / MIPROv2** | **partial** | Module ABC hosts a `DSPyJudge`; SerializableState (FAISS-style) persists program.json | Optimizer is Optuna-shaped (`objective→dict`), cannot express `compile(student,trainset,metric)→Module` | Second Optimizer shape `CompileOptimizer` + `DSPyJudge` + gold→dspy.Example / metric adapters |
| **Jellyfish (local LLM)** | **partial** | Host via LLMJudge pointed at local vLLM endpoint | No `api_base` in serialized config (env-only, breaks self-contained artifact); can't train; cost reports $0 | Add `api_base` field + per-Module cost override + jellyfish serializer template |
| **Weak supervision (Snorkel)** | **partial** | `Bootstrapper.labeler` receives full candidate list → can build vote matrix | No LabelingFunction (with ABSTAIN); no label-model primitive (needs joint matrix, not per-pair); GoldPairSource lacks `weak_supervision` | Tiny `LabelingFunction` protocol + `LabelModelLabeler` (~40-line numpy MeTaL) + provenance value |
| **DIAL / risk active learning** | **partial** | Bootstrapper (block→mine→label) single pass; Miner/Labeler ABCs | No iterative loop; Miner can't see matcher signals (only static blocker score); no fittable blocker; no risk/core-set selector | Widen sampler contract to see per-pair matcher scores + `build_active(rounds,budget)` loop |
| **Collective relational (CR-ER)** | **no (headline method)** | Can express static "naive-relational" baseline only | Needs feedback loop: one pair's score depends on neighbors' *resolved* cluster labels mid-clustering | `CollectiveClusterer` that owns the merge loop and calls back a `scorer(ci,cj,state)` + relational FeatureSpec |
| **BoostER** | **no** | Blocker→pairs; LLM judge = scarce oracle; budgeted runner ≈ budget | No belief-state, no acquisition policy, no iterative loop, no Bayesian update / Θ | Per-pair marginal store + `Acquisitor` policy + iterative Resolver loop + posterior-update step |
| **Senzing / Zingg / Dedupe (incremental)** | **partial** | Dedupe ≈ fully expressible; Zingg batch+AL largely | **Stateless judge** blind to live corpus stats; no retroactive re-resolution; no stable-ID merge/split delta; batch-shaped AL | Optional corpus-stats provider on judge; `link(record, anchors)→ClusterDelta` contract; interactive AL loop |

**Genuinely inexpressible today (structural, not adapter, gaps):** ComEM Select/Compare and LLM-CER (set-wise *input*); CR-ER collective inference and BoostER (iterative/stateful *control flow*); Senzing-style live-frequency weighting + retroactive re-resolution (stateful store coupling). Everything else is a `fit()`-hook or a config/serialization addition away.

---

## 4. Flexibility Recommendations (smallest set to make "any combination" true)

Prioritized. Each is additive and backward-compatible; the common pairwise/frozen case is untouched.

**Tier 1 — cheap, unlocks the most SOTA:**

1. **Generalize the judgement contract:** `PairwiseJudgement | SetJudgement/PartitionJudgement`. Add `ERCandidateGroup{anchor, candidates, blocker_name}` beside `ERCandidate`, and a `GroupwiseModule.forward(Iterator[ERCandidateGroup]) → Iterator[PairwiseJudgement]` (anchor↔selected = high, anchor↔rejected = low). **The output contract and the entire downstream — Clusterer, Resolver, harness — stay untouched; only the judge input shape is new.** Unlocks ComEM Select (+16 F1, ⅓ cost), Compare, and the field's biggest cost lever (set-batched calls). Give the Blocker a `stream_groups()` emitter that un-flattens the per-anchor kNN it already computes.

2. **Add an optional `fit(gold, hard_negatives)` and `fit_unlabeled(candidates)` hook on Module** (default no-op). This single seam is what turns "compose configured scorers" into "train a scorer" — and it is *not* method-specific: it simultaneously homes Magellan RF, Fellegi–Sunter EM, ZeroER (unlabeled), Snorkel label models, and the DSPy compile step. Pair-serialization + augmentation live in the Comparator.

3. **Add a second Optimizer shape** `compile(student: Module, trainset, metric, valset) → Module` beside `BlockerOptimizer`. Do **not** force MIPRO/contrastive training through the Optuna `objective→dict` interface. One new ABC + a `MIPROCompiler` (delegates to `dspy.MIPROv2`) + a `ContrastiveEncoderTrainer` (delegates to sentence-transformers SupCon/MNRL).

4. **Ship a value-frequency-aware `FSJudge`** (m/u weights, additive log₂ evidence, posterior 2^M/(1+2^M), per-value TF table). Highest-ROI accuracy change for multilingual Person; the evidence (Splink, Senzing, OpenSanctions) says the win is *value-level rarity*, not more comparison levels — so **keep the Comparator thin** and reject enriching the comparison-level taxonomy (ROADMAP §8.1).

5. **Make the Clusterer pluggable + default to merge-resistant.** Replace naive transitive closure as default with correlation-clustering or edge-weight-thresholded components + a max-diameter guard (M3 saw −0.63 BCubed from single-edge over-merge). Add an optional `(lower, upper)` two-threshold band to surface a "possible-match" clerical-review set.

**Tier 2 — medium, targeted:**

6. **LLM-judge levers:** `{demonstrations}` slot + `rules: list[str]` block + a pluggable `ExampleSelector` (static/random/related, "related" reusing existing embeddings/rapidfuzz), and extend `PairwiseJudgement.provenance` with `prompt_variant`, `demo_ids`, `rules_hash`. Make prompt-variant a first-class benchmarkable field.

7. **`TrainableEmbeddingProvider.fit(texts)`** called by `VectorIndex.create_index()` when present + SerializableState embedder weights. Frozen encoders simply omit `fit()`. Unlocks DeepBlocker, SC-Block, Sudowoodo, DIAL's index. Add a `UnionBlocker[Blocker...]` (dedup ERCandidate frozensets) to realize the roadmap's per-feature union.

8. **`api_base` field on LLMJudge** + per-Module cost override (fixes $0-for-local misreporting). Smallest unlock for Jellyfish/local-vLLM as a self-contained artifact.

**Tier 3 — bigger bets, only when a real target demands:**

9. **Iterative control-flow layer** (the deepest gap): `Bootstrapper.build_active(rounds, budget)` loop + an `Acquisitor`/QuerySelector policy object (generalizes today's fixed-band cascade router). Enables active learning (DIAL/risk), BoostER-style belief refinement (per-pair marginals + posterior update, avoiding the exponential possible-worlds enumeration), and is the label-efficient engine for agent autoresearch.

10. **`CollectiveClusterer`** (inverts control: clusterer owns the merge loop, calls back a scorer with mutable `ClusterContext`) for relational/collective ER — a genuinely new stateful primitive, plus relational `FeatureSpec` fields.

11. **Incremental `link(record, existing_anchors) → {matched_id | new} + ClusterDelta`** with stable-ID merge/split, and an optional live-corpus-stats provider on the judge, so brainsquad's living-cluster loop and Senzing-style frequency weighting are expressible without langres owning the store.

**Explicitly out of scope / delegate:** schema-matching/attribute-alignment (Alaska — data integration, upstream of the seam); scale-out blocking execution (SQL/Spark pushdown — delegate to the body); owning GPU training loops for heavy transformers (wrap/delegate, don't reimplement the DL design space).

---

## 5. Experimentation Methodology & Scaling Proxies

The LLM-scaling mindset transfers directly, and langres already has ~80% of the machinery in the wrong shape (a point, not a curve).

**Borrow the scaling-law loop.** Generalization error follows a power law in training-set size: `s(m) = a·m^b + c` (small-data plateau → power-law decline → irreducible floor). Fit it on a handful of cheap subset runs (F1 at 50/100/200/400/800 labels) and extrapolate to decide *before paying* whether to keep labeling. **Method selection = compare label→F1 *curves*, not single-budget F1** — the method that wins at 100 labels often loses at 2000 (Ditto's augmentation shifts the whole curve left). Trust near-extrapolation (2–4×) far more than 10×; always report the confidence band. This is the ROADMAP §4 "label-count-vs-F1 curve" promise made first-class.

**Data-subset selection as the reliable proxy.** Random samples are poor proxies because ER candidate sets are dominated by trivial non-matches (~zero signal). Per Sorscher et al. (NeurIPS 2022), keeping **hard examples near the decision boundary** beats power-law scaling. langres's **bootstrapper hard-negative-mined set is one artifact with three roles**: (a) cold-start gold set, (b) reliable cheap proxy subset for scaling curves, (c) the hard band on which pairwise-F1 is the honest metric (BCubed is inflated on singleton-heavy corpora — langres's own M2/M3 caveat). Don't build separate samplers.

**Exploit pretrained models as free scale.** Start from a strong zero-label baseline (pretrained embedder for blocking + LLM judge for matching — the ceiling), then spend labels/compute *only where a fitted curve says a cheaper student or bigger embedder pays off*. langres's M3→M4 arc (frontier LLM = ceiling; DSPy-distill the student where cost justifies) *is* this thesis — it just needs naming. In-context demonstration selection is itself a tunable the Optimizer should own (related/nearest for frontier models, random for weak ones).

**The autoresearch loop as an explicit gate machine** (agent-drivable, cheap logged runs — not a heavyweight DAG):
1. **Blocking gate** — measure Pair-Completeness on the proxy first; if < target, scale the *blocker* (bigger embedder, more per-feature keys, funnel), never the judge. No judge recovers a pair the blocker dropped.
2. **Zero-label baseline** — run pretrained embedder + LLM judge; record ceiling; use the LLM as teacher to label the hard band.
3. **Curve gate** — fit per-method label→F1 curves on the hard band; if already in the plateau, more labels won't help — change method/features.
4. **Scale decision** — curve steepness + cost frontier picks *one* lever: more labels, bigger embedder, stronger teacher, or distill.
5. **Distill** — DSPy-compile the cheap student when curve + cost justify (M4).

**Cost is a curve axis, not a cap.** The 79× cost span across the same task and "fine-tuned-mini rivals GPT-4" mean the winner is a *frontier point on an F1-per-dollar Pareto*, not a max-F1 point. langres's budgeted runner already has the data; report F1/$ and F1/ms frontiers.

**Calibration is a prerequisite.** Comparing a cosine score, a rapidfuzz ratio, and an LLM prob on one threshold axis compares incommensurable scales; LLM judges are systematically miscalibrated (arXiv:2509.19557). Pull a lightweight per-method calibrator (isotonic/Platt on the gold band) forward from M6 into the harness so curves and cascade bands are commensurable.

**Simplification:** don't build a bespoke scaling subsystem — an ER scaling proxy is two curves langres already half-produces (Pair-Completeness-vs-blocking-effort, F1-vs-#labels). Ship them as one small `ScalingCurve` output. Reuse `tune_threshold_on_train` as the curve engine (run it at N budgets, collect points). Don't chase all 13 datasets — one easy + one hard controls difficulty; Fodors-Zagat + Amazon-Google is the correct minimal pair, add a third (Person) only when a real gap demands.

---

## 6. Simplification Opportunities (KISS-first — be direct)

**Modern LLMs genuinely collapse components — lean into it where it's simpler *and* cheaper.**

- **The minimal composable core is three verbs + a container:** **Retrieve** (candidate source: pairs OR sets) → **Judge** (scorer: pairwise OR set-wise OR collapsed block+match) → **Resolve** (cluster/partition builder). Everything else — Comparator, Optimizer, Canonicalizer, Bootstrapper, and the `tasks.*`/`flows.*`/`blockers.*` high-level layer — is an orthogonal *add-on*, not a peer pillar. This is fewer concepts than the current five-pillar + tasks/flows surface (much of which is unbuilt) and natively admits the collapsed methods.

- **Set-batched LLM calls beat cheap-per-pair students as the primary cost strategy.** The literature's largest savings (LLM-CER 5×, ComEM $0.92 vs $4.52, multi-agent 61% fewer calls) come from *batching records into set-wise prompts* — which langres's strictly pairwise contract *forbids*. Collapsing done right is simultaneously simpler and cheaper. Generalizing the judgement contract (§4.1) is thus a *simplification*, not just a feature.

- **Demote the Comparator from pillar to optional input-transform.** The per-feature comparison vector is a Fellegi–Sunter artifact that heuristic/logistic/FS judges need — but LLM and embedding judges reason over the raw feature bag directly, making the taxonomy dead weight (ROADMAP §8.1 admits it's unvalidated). Let a judge declare whether it consumes a comparison vector or the raw bag. Removes a mandatory stage from the hot path without losing the FS path.

- **In-context clustering (LLM-CER) is one component that replaces blocker+judge+clusterer** for small/medium blocks. Offer a `CollapsedResolver` method so simplicity-seekers get block+match+cluster in one call, while power users still compose the three verbs — same interface, both altitudes.

- **A single universal dense retriever (UniBlocker-style)** as the default Blocker removes most per-dataset Optuna blocker-HPO — collapsing "Blocker + blocker-tuning" into one configured component for the common case.

**Where langres is over-complicated:**

- **Collapse the speculative `tasks.*` / `flows.*` layer into Resolver configs.** A parallel, mostly-unbuilt class hierarchy (DeduplicationTask, EntityLinkingTask, CompanyFlow, ProductFlow) will drift from the Resolver spine (already the real artifact). One configurable Resolver with dedup/linking modes (ROADMAP §2.3) is lower-maintenance.
- **Consolidate the metrics stack** to one module (the M3-extracted `evaluate_resolver_bcubed`), dropping the sklearn + er-metrics + pytrec_eval + scipy sprawl.
- **Fix the clusterer *default*, don't add a second mediocre knob.** Two clustering paths (networkx `connected_components` + scipy `hierarchical`) is more surface than one merge-resistant default. The M3 over-merge data argues for correcting the default.
- **Treat the whole optimization stack (Optuna/DSPy/PyTorch/bootstrapper) as an opt-in "training" module**, gated behind a demonstrated cost win over frontier zero-shot — not part of the core inference spine.

**Unifying method worth adopting:** the **generalized judgement contract + set-wise judge** is the one change that both simplifies (enables batching) and closes the biggest SOTA gaps (ComEM, LLM-CER). Adopt it before investing further in per-pair distillation.

---

## 7. Benchmarks, Datasets & Production-Deployment Gaps

**Datasets to standardize on.** langres's current two (Fodors-Zagat easy, Amazon-Google hard) is too thin to defend the seam — it has no textual-hard case, no dirty/missing-value case (the exact thing the missing-aware Comparator claims to solve), and no unseen-entity generalization case (which a shipped artifact must survive). Adopt a **6-dataset external portfolio**, all DeepMatcher-split (one loader each):

- Keep **Fodors-Zagat** (saturation sanity) + **Amazon-Google** (hard structured).
- Add **Abt-Buy** (textual-hard — proves the embedding/LLM judge earns its cost), **Walmart-Amazon DIRTY** (proves the missing-aware Comparator), **DBLP-ACM** (~99 ceiling — regression guard), and **WDC Products** small + 80%-corner-case + unseen-entity variant (proves artifact generalization — its headline finding is that *every* SOTA matcher degrades badly on unseen entities).

**Make OpenSanctions Pairs the north-star external benchmark.** 755,540 labeled Person/Company/Org pairs, 293 sources, 31 countries, multilingual + multi-script, up to 132 fields, set-valued and time-dependent attributes — feature-bag-shaped, missing-aware, asymmetric. Published baselines to race: rule-based 91.33, GPT-4o **98.95**, DeepSeek-R1-Distill-Qwen-14B **98.23** (open, local), Llama-3.1-8B 95.94. **Sobering caveat langres must heed: DSPy MIPROv2 gave only ~1–2 F1 there and in-context examples were neutral-to-negative** — so M4's distillation upside on messy multilingual Person data is empirically *uncertain and must be measured*, not assumed (ROADMAP §8.3). Failure modes: cross-script transliteration and off-by-one date/id mismatches.

**Metrics to standardize.** Pair-F1 (isolates the scorer — use for method *selection* on the hard band) + BCubed (cluster integrity — always print the all-singletons floor beside it) are necessary but biased. Add as first-class: **Generalized Merge Distance** (cost-based, models merge/split asymmetry) and **separated blocking metrics — Pairs-Completeness AND Reduction-Ratio** (not PC alone). Keep the **widened threshold grid** (0.99, not 0.80-capped) as default — M3 showed a capped grid crushed score-based judges (embedding F1 0.10→0.82 after widening).

**Production-deployment gaps** (production peers converge on three things langres under-builds):

1. **Value-frequency-aware matching** (Splink TF-adjusted m/u; Senzing frequency/exclusivity/stability). langres's WeightedAverageJudge treats discriminativeness as *feature-level*, not *value-level* — the wrong resolution for person names. **Highest-ROI judge upgrade for the north star (§4.4).**
2. **Incremental cluster lifecycle** (Zingg calls merge/split/reassignment/ID-stability "the hardest part"; Senzing self-corrects in real time). langres's transitive-closure Clusterer is batch and *cannot split a cluster*. This is the production credibility gate, M5-stub today — a design decision to make *now* (correlation clustering vs incremental re-cluster), not an M5 afterthought.
3. **Active learning** (Dedupe/Zingg reach production from 30–40 labeled pairs via uncertainty sampling). langres's ReviewQueue/AL loop is documented-but-unbuilt.

**Blocking scale is an unanswered production question.** In-memory FAISS + O(n²) AllPairs won't survive a real person store (millions+). Peers push blocking to SQL/Spark (Splink) or a purpose-built engine (Senzing/Zingg). langres needs at least a funnel with reduction-ratio budgeting + a documented SQL/BlockingPy pushdown path before "deployment later" is credible.

**Position honestly:** langres is *the composable pairwise (+ set-wise) seam + serializable artifact* — not a Senzing/Splink replacement for graph/streaming/billion-scale. Its zero-shot-LLM-judge → cheap-distilled-student story is a defensible edge that training-required OSS cannot tell.

---

## 8. Prioritized Roadmap Deltas

Sequenced and mapped to milestones, separating cheap high-leverage wins from big bets.

### Cheap, high-leverage (do first — M3.5 / M4)

| # | Delta | Milestone | Unlocks |
|---|---|---|---|
| C1 | **Scaling-curve harness output** (power-law fit + extrapolation + band); GMD + Reduction-Ratio metrics; widened grid + per-threshold curve as default | M3.5 | Autoresearch loop; honest method selection; §5, §7 |
| C2 | **Per-pair slice tags + sliced aggregation** in `evaluate_judge_on_candidates` + a `FixedSplitPairBenchmark` adapter | M3.5 | WDC degradation curves; the 6-dataset portfolio; §7 |
| C3 | **Value-frequency-aware `FSJudge`** (m/u + EM + random-sampling u + TF table) behind the PairwiseJudgement contract | M4 | Splink parity; person-name matching; answers §8.1 |
| C4 | **LLM-judge levers**: demonstrations slot + `ExampleSelector` + rules block + richer provenance; prompt-variant as benchmarkable field | M4 | Peeters & Bizer +7.85 F1; auditability |
| C5 | **`api_base` + per-Module cost override** | M4 | Jellyfish/local-vLLM self-contained artifact; honest cost axis |
| C6 | **Merge-resistant Clusterer default + pluggable interface** (correlation/edge-weighted) | M4/M5 | Removes documented over-merge footgun; §7 |
| C7 | **Adopt OpenSanctions Pairs + 6-dataset portfolio; run the frontier-zero-shot null baseline** as a hard gate before further distillation investment | M4 | De-risks M5 "Person is measurable"; validates DSPy ROI |

### Structural but bounded (M4 → M5)

| # | Delta | Milestone | Unlocks |
|---|---|---|---|
| S1 | **Generalize judgement contract** (`SetJudgement`/`PartitionJudgement`) + `ERCandidateGroup` + `GroupwiseModule` + Blocker `stream_groups()`; ship a `SelectJudge` and in-context-clustering method, benchmarked head-to-head on Amazon-Google | M4 | ComEM Select (+16 F1, ⅓ cost), LLM-CER, set-batched cost savings — **the highest-leverage single move** |
| S2 | **`fit()` / `fit_unlabeled()` Module hook** + second Optimizer shape `compile(student,trainset,metric)→Module` | M4 | Trainable-judge family: Magellan RF, FS-EM, ZeroER, DSPy, Snorkel label model |
| S3 | **`DSPyJudge` + MIPROCompiler + gold/metric adapters** (state via SerializableState) | M4 | The M4 cheap-precise-judge target as designed |
| S4 | **`LabelModelLabeler` (~40-line numpy MeTaL) + LabelingFunction protocol** | M4/M5 | Weak-supervision cold-start, API-free; combine-LFs-with-LLM hypothesis |
| S5 | **`TrainableEmbeddingProvider.fit()` + SerializableState embedder + `UnionBlocker` + BlockingPy adapter + `ContrastiveEncoderTrainer`** | M5 | DeepBlocker, SC-Block, Sudowoodo blocker, per-feature union |
| S6 | **Incremental `link(record, anchors) → ClusterDelta`** with stable-ID merge/split | M5 | brainsquad living-cluster loop; production credibility |

### Big bets (M5 → M6, only when a real target demands)

| # | Delta | Milestone | Unlocks |
|---|---|---|---|
| B1 | **Iterative/active control-flow layer**: `build_active(rounds,budget)` + `Acquisitor` policy + belief-state (per-pair marginals + posterior update) | M5/M6 | Active learning (DIAL/risk), BoostER; the label-efficient autoresearch engine |
| B2 | **`CollectiveClusterer`** (owns merge loop, callback scorer, mutable ClusterContext) + relational FeatureSpec | M6 | Collective relational ER (CR-ER); GNN-lineage methods |
| B3 | **Live-corpus-stats provider on the judge** + scale-out blocking (SQL/Spark pushdown, funnel with RR budgeting) | M6 | Senzing-style frequency weighting; billion-scale deployment |
| B4 | **Trainable cross-encoder Module + wired `Optimizer.finetune`** (or documented external-train→load-weights path + weight-reference in artifact) | M6 | Contend with Ditto/HierGAT/R-SupCon on their own terms (else wrap as baseline rows) |

**Sequencing logic.** C1–C7 are the M3-harness-extraction work plus small additive fields — they make the seam *measurable and honest* and cost little. **S1 (set-wise contract) and S2 (fit-hook) are the two structural keystones**: S1 restores "any combination possible" against the set-wise SOTA that is currently the cheapest *and* most accurate, and S2 restores it against the most-cited trained family. Everything in the big-bet tier should be *earned* by a demonstrated gap on a real target (multilingual Person, production scale) — not built speculatively. Above all, gate M4's distillation machinery behind the **frontier-zero-shot null baseline (C7)**: if a distilled student can't beat "just call the frontier model" on cost at equal quality, cut it.

---

*Sources cited inline throughout. Primary references: Mudgal et al. SIGMOD'18 (DeepMatcher); Li et al. VLDB'21 (Ditto, arXiv:2004.00584); Peeters & Bizer arXiv:2305.03423 / 2310.11244 (LLM EM); Wang et al. COLING'25 arXiv:2405.16884 (ComEM); LLM-CER arXiv:2506.02509; Yao et al. SIGMOD'22 (HierGAT); Wang/Li/Wang ICDE'23 (Sudowoodo, arXiv:2207.04122); Thirumuruganathan et al. PVLDB'21 (DeepBlocker); Wu et al. SIGMOD'20 (ZeroER, arXiv:1908.06049); Bhattacharya & Getoor TKDD'07 (CR-ER); Li et al. WWW'24 (BoostER, arXiv:2403.06434); Zhang et al. EMNLP'24 (Jellyfish, arXiv:2312.01678); Opsahl-Ong et al. (MIPRO, arXiv:2406.11695); Splink/MoJ; ZeroER, Snorkel (Ratner et al. VLDB'18), Wu et al. SIGMOD'23 (ER label model, arXiv:2211.06975); SC-Block arXiv:2303.03132; NLSHBlock arXiv:2401.18064; UniBlocker arXiv:2404.14831; DIAL VLDB'22; WDC Products arXiv:2301.09521; OpenSanctions Pairs arXiv:2603.11051; Sorscher et al. NeurIPS'22 arXiv:2206.14486; learning-curve arXiv:2303.01598; calibration arXiv:2509.19557. langres repo: docs/ROADMAP.md, docs/TECHNICAL_OVERVIEW.md, docs/USE_CASES.md, src/langres/core/*.*