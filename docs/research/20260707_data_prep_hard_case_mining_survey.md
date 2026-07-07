# Data Preparation & Hard-Case Mining for Entity Matching — a survey

**Date:** 2026-07-07
**Author:** langres research (orchestrated multi-agent survey; primary-source verified)
**Issues:** Task 0 of [#86](https://github.com/raisesquad/langres/issues/86) (reusable hard-case / informative-pair mining seam) · feeds [#83](https://github.com/raisesquad/langres/issues/83) (AnyMatch) · epic [#85](https://github.com/raisesquad/langres/issues/85)
**Scope of this doc:** state-of-the-art synthesis **+ langres seam mapping**. It deliberately stops short of a seam interface / API / function design — that is a separate, later step (per the research-only decision on 2026-07-07).

---

## 0. Why this survey exists

AnyMatch's headline — a **124M-param GPT-2 landing within ~4% of GPT-4 on entity matching at ~3,900× lower cost**, and generalizing zero-shot to unseen datasets — comes overwhelmingly from **data preparation**, not model size. The bet behind #86 is that *the data recipe is the transferable lever*: get hard-case / informative-pair mining right once, expose it as a reusable seam, and every downstream matcher improves — the cheap judge, the fine-tuned student, the flywheel — across entity types, not just on Amazon-Google / Abt-Buy.

A mined/curated pair set has **three consumers** in langres, and a good seam serves all three:

1. **Fine-tuning** a small judge (`SupervisedFitMixin.fit()`; the Qwen3 QLoRA work in #81/#83).
2. **DSPy prompt optimization** — hard pairs as the bootstrap / few-shot **demo pool** for a `DSPyJudge` (MIPROv2).
3. **Active-learning review** — the existing CLI labeling loop (`select_for_review` / `ReviewQueue`), which is already one consumer of this seam.

This doc maps the field so #86's strategy menu is grounded in what actually works, what is cheap, and what langres already has.

---

## 1. TL;DR — the findings that should shape the seam

1. **AnyMatch is hard-*positive* mining + *random* negatives — NOT hard-negative mining.** Verified from their `automl_filter` code: the "difficult" set is **positives the AutoGluon model gets wrong (false negatives)**; negatives are sampled **uniformly at random** at a 2:1 ratio. This is a specific, cheap, portable idea — and it is a *different strategy* from the dense-retrieval hard-**negative** mining lineage (ANCE/DPR/RocketQA). The seam should treat "mine hard positives" and "mine hard negatives" as **two distinct pluggable strategies**, never one conflated "hard-pair" knob.

2. **The label-noise trap is the central risk, and it has a named fix.** Any signal of the form "the model got this wrong / is unsure" cannot, on its own, distinguish a **genuinely hard** pair from a **mislabeled** one — they look identical on a raw difficulty score. "Misclassified positive" (AnyMatch's own signal) is exactly this ambiguity. **Confident Learning / Cleanlab** (per-class-calibrated confident joint) is the disambiguator to run first; **Dataset Cartography's "ambiguous" region** (high *variability* across epochs, not merely low confidence) is the literature's cleanest notion of "informative." A mining seam that skips this will amplify noise.

3. **langres already implements the active-learning core.** `select_for_review` already mines **uncertainty margin** (`|score − threshold|`), **judge-disagreement** (committee), and a **confident-merge audit slice**. Those three are *done*. The new value is everything else: model-error mining, blocking-derived hard negatives, difficulty scoring, coverage/dedup, and label-noise filtering.

4. **The two cheapest high-value *new* strategies need no training loop and no paid LLM:**
   - **Blocking-derived hard negatives** — near-threshold non-matches that survive the `Blocker` are, by construction, the informative negatives. langres already computes the similarity; the strategy is just "keep the near-misses instead of discarding them." Essentially free.
   - **EL2N-style difficulty** — `|1 − p(gold_label)|` from *any* probabilistic judge already in `JudgementLog`. No extra training.

5. **Mining (selection) and augmentation (generation) are different capabilities that both belong under "data prep."** AnyMatch does both: `automl_filter` *selects* hard positives; its attribute / flip / permute operators *generate* new examples. Ditto is pure generation (rule-based operators). The seam is primarily a *selection* device; augmentation is a sibling worth naming so they aren't conflated.

6. **Diversity is a required guardrail, not a nicety.** Hard-only selection collapses onto one narrow failure mode (e.g. all name-typo pairs). Coreset / representativeness (moderate-coreset, herding) and near-duplicate removal (SemDeDup) exist precisely to keep a mined set *covering* the space. Combine hardness × coverage, don't pick hardness alone.

7. **DSPy caveat:** MIPROv2 already does demo *selection* over whatever pool you hand it, but it does **not mine**, and its bootstrap step **cannot surface demos the teacher itself fails** (which is precisely what hard pairs are). So feeding hard cases to DSPy is plausibly additive — via (a) enriching the validation/candidate pool and (b) supplying **labeled** hard-demo seeds bootstrapping would never produce — but the lift is unproven and must be measured, not assumed.

---

## 2. AnyMatch, precisely (the #83 anchor)

**Paper:** Zhang, Groth, Calixto, Schelter — *"AnyMatch — Efficient Zero-Shot Entity Matching with a Small Language Model"*, [arXiv:2409.04073](https://arxiv.org/abs/2409.04073) (2024; AAAI-25 GOOD-DATA workshop). Model = fine-tuned **GPT-2 (124M)**. Code: `github.com/Jantory/anymatch` (no LICENSE → all-rights-reserved; read for understanding only, **do not fork**).

### Verified recipe (from `utils/data_utils.py` + paper §method)

| Step | What it actually does | Precision note |
|---|---|---|
| **Hard-pair mining** (`automl_filter`) | Train **AutoGluon `TabularPredictor`** on the raw `*_l`/`*_r` attribute columns. Keep `train_pos_wrong_preds = df[(pred != label) & (label == 1)]` — i.e. **positives the model misclassifies (false negatives)**. Keep `min(400, #pos)`; top up with correctly-predicted positives if fewer than 400. | **Positives only.** Negatives are **NOT** hard-mined. |
| **Negatives** | `2 × #positives` (≤ 800) sampled **uniformly at random** from all negatives. | Random, not hard. Mining is *asymmetric*. |
| **Label balancing** (`one_pos_two_neg`) | Enforce **2:1 neg:pos** by downsampling negatives. | Uncredited heuristic (no citation in paper). |
| **Attribute augmentation** (`read_multi_attr_data`) | Add **single-attribute pairs** `(r_l[a], r_r[a])` with the pair's label, both classes, balanced per-attribute, capped 800. | Generation, not selection. |
| **Structure augmentation** (`automl_filter_flip` / `_permute`) | **flip** swaps left↔right; **permute** randomizes attribute order. Label preserved, both classes, deduped. Fixes the table↔text "no attribute order" mismatch. | Whether all are on in the final config vs. ablations was not confirmable from code alone. |
| **Serialization** (`df_serializer`, `mode1`) | Ditto-style `COL <val>, COL <val>`, wrapped `Record A is <p>…</p>. Record B is <p>…</p>. … are they the same?` Missing → `N/A`. | Special tokens `<p>…</p>`. |
| **Protocol** | 9 Magellan + WDC benchmarks; **leave-one-dataset-out** (train on 8, test the held-out) = the zero-shot claim. | — |

### Lineage (what AnyMatch is standing on)

- **AutoGluon** for the mining classifier → Erickson et al. 2020 *(high confidence — the tool they call).*
- **"Curate difficulty because EM benchmarks are trivially solvable"** → Mudgal et al. 2018 (DeepMatcher/Magellan); Papadakis et al. 2024; Leone et al. 2022 *(medium-high; motivational cites read from the arXiv HTML render, treat co-author/year as medium).*
- **Data augmentation + the `COL…VAL` serialization** → **Ditto** (Li et al. 2020) *(high).*
- **Tabular-structure augmentation** → Badaro et al. 2023 (transformers-for-tabular survey) *(medium).*
- **2:1 balancing** → genuinely **uncredited** design heuristic.
- **LLM-EM framing** → Narayan et al. 2022; Peeters & Bizer 2023 (MatchGPT).

**Takeaway:** the portable AnyMatch idea for the seam is *"positives a cheap tabular model gets wrong"* — a **false-negative mining** strategy — kept strictly separate from hard-negative mining and paired with a label-noise filter (see §4).

---

## 3. A taxonomy of hard-case / informative-pair mining

Organized by **the signal that flags a pair as worth keeping** — i.e. by candidate *strategy*. For each: the signal, whether it needs a trained model / paid LLM, cost, and the EM mapping.

### S1 — Model-error signals ("keep what the current model gets wrong")

The oldest and most robust idea: an example a competent model still gets wrong carries the most signal.

| Method | Signal | Needs | Cost | EM mapping |
|---|---|---|---|---|
| **AnyMatch false-negative mining** (AutoGluon) | positives a cheap tabular model misclassifies | a cheap AutoML classifier | low | direct — the #83 strategy; **positive-side only** |
| **Bootstrapping / hard-negative mining** (Sung & Poggio 1994; Felzenszwalb et al. PAMI 2010) | classifier's false positives, added iteratively | partially-trained model, repeated inference | low-med | non-match pairs the current matcher scores as matches |
| **OHEM** (Shrivastava et al. CVPR 2016, [1604.03540](https://arxiv.org/abs/1604.03540)) | per-example **loss**; backprop only top-N | in-training model | ~free (extra fwd pass) | within a batch, train only on worst-scored pairs |
| **Focal Loss** (Lin et al. ICCV 2017) | soft down-weight of easy examples `(1−p_t)^γ` | none (loss change) | free | soft-weight every pair by `1−confidence` instead of hard-selecting |
| **DPR** (Karpukhin et al. EMNLP 2020) | BM25 top lexical hit that is a non-answer | **no model** (fixed lexical scorer) | low | static: string-similar pair that is a true non-match |
| **ANCE** (Xiong et al. ICLR 2021, [2007.00808](https://arxiv.org/abs/2007.00808)) | model's own current top-ranked wrong answers, index refreshed during training | iterative trained model + ANN re-index | **high** | re-run the judge over the blocked pool each round, harvest confident wrong merges |
| **RocketQA** (Qu et al. NAACL 2021) | cross-batch negatives + **denoised** hard negatives (a cross-encoder drops false negatives) | strong teacher judge | med-high | **the denoise step is the EM safety rail** (see §4) |
| **TAS-B / NGAME** | topic-balanced / smart-batch in-batch negatives (cheaper than per-step ANN) | teacher / smart batching | med | scalable approximations of the above |

**Key distinction the seam must preserve:** *positive-side* error mining (AnyMatch: false negatives = matches the model misses) vs. *negative-side* error mining (ANCE/DPR: false positives = non-matches the model is tempted to merge). AnyMatch does **only** the former; the retrieval lineage does the latter. They are separate strategies with opposite failure modes.

### S2 — Uncertainty & disagreement signals (active learning) — **already in langres**

| Method | Signal | Status in langres |
|---|---|---|
| **Uncertainty / margin / entropy sampling** | low `|score − threshold|`, high entropy | ✅ `_select_uncertainty` (`core/review.py:370-388`) |
| **Query-by-committee** (DIAL Index-By-Committee, Jain et al. VLDB 2022, [2104.03986](https://arxiv.org/abs/2104.03986); ALMSER-GB graph-boosted, Primpeli & Bizer 2021) | ensemble/judge **disagreement** | ✅ `_select_disagreement` across two judgement logs (`core/review.py:391-418`) |
| **Confident-merge auditing** | governance sample over confident merges | ✅ audit slice, `audit_fraction` default 0.1 (`core/review.py:307-323`) |
| **DTAL** (Kasai et al. ACL 2019) | transfer + max-entropy; F1 97.73 on DBLP-ACM with ~300 labels | pattern applies; not wired |

DIAL's real contribution beyond langres today: **separate objectives for blocker (recall) and matcher (precision)**, avoiding naive O(n²) active learning — worth remembering when the seam feeds both a blocker and a judge.

### S3 — Boundary-from-blocking signals (cheapest *new* strategy)

**Blocking-derived hard negatives** (DeepMatcher/Magellan lineage, Mudgal et al. SIGMOD 2018): a candidate pair that **clears the blocker's similarity threshold but is a confirmed non-match** is near the decision boundary by construction — syntactically close, semantically distinct — far more informative than random cross-product negatives. **Signal:** `blocker_score ≥ threshold AND label = non-match`. **Cost:** essentially free — a byproduct of the existing `Blocker` / `StringComparator` / `VectorBlocker`; retain near-threshold non-matches instead of discarding them. This is the negative-side complement to AnyMatch's positive-side mining, at zero model cost.

### S4 — Training-dynamics difficulty signals (need a training loop)

| Method | Signal | Needs | EM mapping |
|---|---|---|---|
| **EL2N / GraNd** (Paul et al. NeurIPS 2021, [2107.07075](https://arxiv.org/abs/2107.07075)) | EL2N = `‖softmax − onehot‖`; GraNd = expected grad-norm | early-checkpoint model (few epochs) | **EL2N ≈ `|1 − p(match)|`** — computable from any probabilistic judge with **no extra training** |
| **Forgetting events** (Toneva et al. ICLR 2019, [1812.05159](https://arxiv.org/abs/1812.05159)) | # of correct→incorrect flips across epochs | full multi-epoch run + per-example logging | only if langres trains with per-epoch checkpointing |
| **Dataset Cartography** (Swayamdipta et al. EMNLP 2020) | per-example **confidence × variability** → easy / hard / **ambiguous** regions | one training run w/ per-epoch confidence logging | **the "ambiguous" (high-variability) region is the literature's cleanest "informative pair"** — distinct from "hard-to-learn," which conflates difficulty with label noise |

Cartography's separation of **ambiguous** (keep — most informative) from **hard-to-learn** (often mislabeled) is the conceptual backbone of the label-noise argument in §4.

### S5 — Representativeness / coverage / dedup signals (the diversity guardrail)

Hardness alone over-samples one failure mode; these keep a mined set covering the space.

| Method | Signal | Needs | Note |
|---|---|---|---|
| **k-Center Greedy / Core-Set** (Sener & Savarese ICLR 2018, [1708.00489](https://arxiv.org/abs/1708.00489)) | geometric coverage in feature space | an embedder (not the target model) | run on pair-embeddings for diverse coverage |
| **Moderate Coreset** (Xia et al. ICLR 2023) / **Herding** (Welling 2009) | points near the *median* feature-distance (avoid trivial-easy and outlier-hard) | pretrained embedder only | **cheapest coverage method — no training** |
| **CRAIG / GradMatch** (Mirzasoleiman et al. ICML 2020; [2103.00123](https://arxiv.org/abs/2103.00123)) | subset gradient ≈ full-dataset gradient | per-example gradients | high cost; defer to large fine-tune loops |
| **SemDeDup** (Abbas et al. 2023, [2303.09540](https://arxiv.org/abs/2303.09540)) | embedding-space near-duplication | embedder | **reuses `VectorBlocker` embeddings** to drop near-dup pairs before fine-tuning |

### S6 — Label-noise disambiguation (the required filter, not a mining strategy)

**Confident Learning / Cleanlab** (Northcutt et al. JAIR 2021, [1911.00068](https://arxiv.org/abs/1911.00068)): builds a per-class-calibrated **confident joint** from out-of-fold predicted probabilities, ranking likely label errors by the off-diagonal mass — **distinct from "just low confidence."** This is the mechanism that answers "is this misclassified positive *hard* or *mislabeled*?": a pair the model is **confidently wrong** about (calibrated) flags likely **noise**; a pair it is **uncertain/high-variability** about flags genuine **difficulty**. **Run this before trusting any S1/S4 hardness score.** Needs k-fold CV probabilities from any probabilistic classifier (moderate, one CV pass).

---

## 4. The label-noise trap (cross-cutting — read before designing any error-based strategy)

Three independent slices of this survey converged on the same failure mode:

- **hard-negative lineage:** in EM, string/embedding-mined "hard negatives" are *precisely the pairs most likely to be true duplicates* — RocketQA added a cross-encoder denoiser for exactly this.
- **difficulty scoring:** a confidently-wrong example is *indistinguishable from a genuinely-hard one* on a raw score — Cleanlab (calibrated) and Cartography (variability axis) separate them.
- **AnyMatch:** its signal *is* "misclassified positive," which is the ambiguous case by definition.

**Design consequence (analysis, for the later seam design):** any error-based strategy (S1) or raw-difficulty strategy (S4) needs a **confirmation / denoise step** before mined pairs enter *training* as ground truth — a stronger judge (langres's `CascadeJudge` already escalates uncertain pairs), a Cleanlab-style filter, or human review (the existing `ReviewQueue`). langres's cascade + review queue are already the natural home for this rail.

---

## 5. Augmentation — generating hard cases (sibling to mining)

Selection picks from existing pairs; **augmentation synthesizes new ones.** Both are "data prep."

| Method | Mechanism | Cost | langres fit |
|---|---|---|---|
| **Ditto operators** (Li et al. VLDB 2020, [2004.00584](https://arxiv.org/abs/2004.00584)) | `span/attr_del`, `span/attr_shuffle`, `entry_swap`; **MixDA** blends hidden states (λ~Beta); TF-IDF summarization; domain-knowledge spans | **trivial** (string ops, no new labels) | rule-based perturbation of `Comparator`/`ComparisonVector` inputs; `attr_del`/`span_del` force invariance = synthetic hard examples; best operator is dataset-dependent |
| **AnyMatch attr / flip / permute** | single-attribute pairs; left↔right swap; attribute reorder | trivial | already characterized in §2 |
| **Rotom** (Miao et al. SIGMOD 2021) | **InvDA** seq2seq operator + **meta-learned policy** that combines operators as weak labelers (Snorkel-style) | med-high (meta-learning loop) | defer; the "weak-label combination" idea overlaps `CascadeJudge` |
| **Sudowoodo** (Wang et al. ICDE 2023, [2207.04122](https://arxiv.org/abs/2207.04122)) | contrastive self-supervision: two augmented views of the same record = positive, other records = negative (InfoNCE), zero labels | expensive one-time pretrain, amortized | maps onto a label-free `VectorBlocker` pretrain; bigger infra lift |

---

## 6. LLM-era data selection & the LLM-as-data-engine (the modern angle)

### 6a — Selecting which examples to keep for fine-tuning / demos

| Method | Mechanism | Needs strong (paid) LLM? | EM fit |
|---|---|---|---|
| **LIMA** (Zhou et al. 2023, [2305.11206](https://arxiv.org/abs/2305.11206)) | 1k hand-curated diverse examples > large noisy set | no (human curation) | thesis: **cap and curate** the pool, don't maximize pair volume |
| **AlpaGasus** (Chen et al. 2023, [2307.08701](https://arxiv.org/abs/2307.08701)) | strong LLM **scores** each example; keep top subset (52k→9k) | **yes** | score `judgement_log` silver pairs, drop low-scorers |
| **DEITA** (Liu et al. 2024, [2312.15685](https://arxiv.org/abs/2312.15685)) | complexity × quality × **diversity** (embedding distance) | strong LLM trains scorers **once**, then cheap | **best-matching** — score pairs by ambiguity × label-cleanliness × spread; langres already logs the needed fields |
| **IFD / Superfiltering** (Li et al. NAACL 2024 [2308.12032](https://arxiv.org/abs/2308.12032); ACL 2024 [2402.00530](https://arxiv.org/abs/2402.00530)) | `ppl(response|instruction) / ppl(response)`; a **GPT-2-size proxy transfers the ranking** | **no** — any small LM's perplexity | cheap no-paid-LLM difficulty score for the review queue |
| **LESS** (Xia et al. ICML 2024, [2402.04333](https://arxiv.org/abs/2402.04333)) | LoRA-gradient features; select data whose gradient matches a **target task's** validation gradient | no LLM grader, but white-box gradients | most targeted if optimizing one customer/domain's held-out F1; heavier engineering |
| **SemDeDup** (see §S5) | embedding near-dup removal | no | reuse `VectorBlocker` embeddings |

### 6b — LLMs *doing* the data prep

- **Narayan et al. 2022**, *"Can Foundation Models Wrangle Your Data?"* (VLDB, [2205.09911](https://arxiv.org/abs/2205.09911)) — few-shot GPT on wrangling incl. EM; the precedent for langres's own zero-shot `LLMJudge`. *(needs strong LLM)*
- **Jellyfish** ([2312.01678](https://arxiv.org/abs/2312.01678), VLDB 2024) — Llama-2-13B LoRA-tuned on mixed data-prep tasks; precedent that a 13B open model is a viable cheap local judge. *(whether its instruction data was distilled from a stronger teacher is **unverified** — flag)*
- **Steiner, Peeters & Bizer 2024**, *"Fine-tuning LLMs for Entity Matching"* ([2409.08185](https://arxiv.org/abs/2409.08185)) — **the one EM-native paper on LLM-generated training data**: tests (a) adding LLM explanations to training examples and (b) LLM selection/generation of EM training examples — **mixed results** (helps Llama-3.1-8B, hurts GPT-4o-mini). **Read in full before designing a fine-tune-selection strategy** (abstract-only left the concrete method unverified).
- **LLM-synthesized hard negatives** — SyNeg ([2412.17250](https://arxiv.org/abs/2412.17250)), Syntriever ([2502.03824](https://arxiv.org/abs/2502.03824)): a strong LLM writes hard negatives via self-reflection for contrastive/retrieval training; the same recipe could **densify scarce hard non-match pairs** for the `CascadeJudge` — with synthetic-artifact-bias risk. *(needs strong LLM)*
- **LLM-as-judge filtering of silver labels** — no EM-anchored canonical citation found; the mechanism (confidence / self-consistency / inter-judge agreement gates, cf. AlpaGasus) is **already implemented in-repo** by `CascadeJudge` + `JudgementLog`/`LoggingModule` (cheap judge defers uncertain pairs). It's a validated in-repo instance, not something to import.

**Cost split that matters if budget is binding:** **IFD/Superfiltering, LESS, and SemDeDup need no strong LLM at all.** AlpaGasus, DEITA's scorer training, Narayan/Jellyfish-style generation, and LLM hard-negative synthesis all need a strong (paid) LLM in the loop.

---

## 7. The DSPy consumer — does pre-filtering to hard demos help MIPRO?

*(Verified against DSPy optimizer docs + [arXiv:2406.11695](https://arxiv.org/abs/2406.11695).)*

How DSPy picks demos:
- **LabeledFewShot** — random `k` from the trainset (no selection).
- **BootstrapFewShot** — run a teacher over inputs; **the metric filters traces** — only passing completions become demos (default 4). Strict metric → few high-quality demos.
- **BootstrapFewShotWithRandomSearch** — repeat + random search over demo sets, keep best on validation.
- **KNNFewShot** — per-input nearest-neighbor demo retrieval (dynamic, not a fixed set).
- **MIPROv2** — (1) bootstrap demo **candidates**, (2) propose instruction candidates, (3) **Bayesian (TPE) search** over {instruction, demo-set} on a validation set. Guidance: BootstrapFewShot ~10 examples, RandomSearch ~50+, MIPROv2 ~200+.
- **BootstrapFinetune** — bootstrapped demos → **weight updates** (distillation), not prompt demos.

**Does pre-filtering the pool to hard pairs add value over MIPRO's own search?** *(hypothesis — must be measured, not assumed):* MIPRO does demo *selection* but not demo *mining* — it searches over demos bootstrapped from **whatever pool you hand it**, scored by *aggregate* validation metric. Two reasons pre-filtering plausibly still helps: **(1) coverage** — TPE optimizes average validation score and is biased toward demos that lift the easy majority, so minority hard/boundary pairs can be washed out; seeding the pool raises their sampling probability. **(2) bootstrap feasibility** — BootstrapFewShot only keeps traces the *teacher gets right*, so on genuinely hard pairs the teacher fails the metric and those **never become demos regardless of search budget**; pre-mined **labeled** hard pairs can be injected as LabeledFewShot-style seeds that bootstrapping alone would never surface. Countervailing risk: too-hard demos can be noisy/atypical and *hurt* the metric, so MIPRO may down-select them anyway. **Net:** pre-filtering is most credibly additive as **validation-pool enrichment + labeled hard-demo seeds**, least so as a replacement for MIPRO's combinatorial search.

---

## 8. Mapping to langres seams — what's covered, what's new

*(Part B verified against repo code; `file:line` from `dspy-langres-map`.)*

### Already implemented (generalize, don't rebuild)

| Signal | Location | Criterion |
|---|---|---|
| Uncertainty margin | `core/review.py:370-388` (`_select_uncertainty`) | keep `|score − threshold| ≤ margin`, most-uncertain first; needs `threshold=` |
| Judge disagreement (committee) | `core/review.py:391-418` (`_select_disagreement`) | verdict differs between two judgement logs; sort by largest score gap; needs `against=` |
| Confident-merge audit | `core/review.py:307-323` | seeded-random governance slice over all judged pairs; catches confident false merges; `audit_fraction` default 0.1 |
| Already-answered exclusion | `select_for_review(corrections=…)` `:288-289` | corrected pairs never re-asked; exhausted → `[]` stop signal |
| Per-call signal logged | `core/judgement_log.py:68-86` | JSONL: `v, left_id, right_id, score, verdict, model, cost_usd, decision_step, timestamp`; `features=True` adds reasoning/provenance (PII risk) |
| Verdicts → silver labels | `core/harvest.py:166-231` (`harvest_labeled_pairs`) | verdict = weak label; a `Correction` (order-independent `frozenset` match) overrides; provenance kept |
| Data-driven threshold | `core/harvest.py:234-291` → `core/calibration.py:31-120` | Youden-J / percentile cut; **warns on silver-only calibration as circular** — insists on human corrections |
| Closed loop | `examples/flywheel_closed_loop.py` | bootstrap → `select_for_review(uncertainty)` → corrections → harvest → train RF student → `CascadeJudge` → `select_for_review(disagreement)` → audit trust metric |

### Not present today (candidate *new* strategies surfaced by this survey)

- **Model-error / AutoML mining** (S1 — AnyMatch false-negative; #83's first new strategy).
- **Blocking-derived hard negatives** (S3 — near-threshold non-matches; cheapest, pure reuse of `Blocker` scores).
- **Difficulty scoring beyond `|score − threshold|`** (S4 — EL2N `|1−p|`, Cartography ambiguity).
- **Coverage / representativeness / dedup** (S5 — coreset, herding, SemDeDup on `VectorBlocker` embeddings).
- **Label-noise disambiguation** (S6 — Cleanlab-style confident joint; the safety rail).
- **Per-input KNN demo retrieval** (DSPy KNNFewShot-style).
- **Augmentation operators** (§5 — Ditto/AnyMatch generation).

Infra notes: `RandomForestJudge` / `derive_threshold` need the `[trained]` (scikit-learn) extra; there is **no DSPyJudge/MIPRO wiring in-repo yet** (that consumer is prospective).

---

## 9. Cost / value quadrant (research characterization, not a build order)

| | **No training loop** | **Needs training loop / gradients** |
|---|---|---|
| **No paid LLM** | Blocking-derived hard negatives (S3); EL2N `|1−p|` (S4); moderate-coreset/herding/SemDeDup (S5); IFD/Superfiltering; margin/disagreement (**already built**); Ditto operators (§5); AnyMatch AutoML mining (S1, cheap classifier) | Forgetting events, Cartography, CRAIG/GradMatch, LESS (S4/S5); QLoRA fine-tune itself |
| **Needs paid LLM** | AlpaGasus scoring; LLM hard-neg synthesis; Narayan-style LLM cleaning; DEITA scoring (after one-time scorer train) | LLM-distilled fine-tuning (Steiner/Peeters/Bizer) |

The bottom-left cell is where the cheapest, most reusable *new* langres strategies live.

---

## 10. Confidence & unverified flags

- **AnyMatch recipe:** verified from code (`automl_filter`, `df_serializer`, `one_pos_two_neg`) — **high**. *Motivational* citation co-authors/years (Leone, Papadakis, Badaro) read from the arXiv HTML render — **medium**; the technique→family mapping is solid. Whether flip/permute augmentations are all-on in the final config vs. ablations — **unconfirmed from code alone**.
- **Jellyfish** instruction-data distillation source — **unverified** (flag).
- **LLM-as-judge silver-label filtering for EM** — no EM-anchored citation found; extrapolated from AlpaGasus + general LLM-judge-calibration literature.
- **Steiner/Peeters/Bizer 2024** — read only abstract-level; the concrete generation/selection method is **unverified** and should be read in full before it informs the fine-tune-selection design.
- **DSPy pre-filtering value (§7)** — explicitly a **hypothesis**; measure, don't assume.
- All other method mechanisms cited to primary sources (arXiv IDs in-line).

---

## 11. References

Inline citations with links appear throughout the body; this is the consolidated list. Every entry carries a resolvable link where one was verified by the source agents; entries without a link are cited by author / venue / year (no URL was fabricated). Links marked with the source they were verified against.

### Entity matching / data integration
- **AnyMatch** — Zhang, Groth, Calixto, Schelter 2024, "Efficient Zero-Shot Entity Matching with a Small Language Model" — [arXiv:2409.04073](https://arxiv.org/abs/2409.04073) · code: [github.com/Jantory/anymatch](https://github.com/Jantory/anymatch)
- **Ditto** — Li, Li, Suhara, Doan, Tan, VLDB 2020, "Deep Entity Matching with Pre-trained Language Models" — [arXiv:2004.00584](https://arxiv.org/abs/2004.00584)
- **Rotom** — Miao, Li, Wang, SIGMOD 2021 — [miaozhengjie.com/assets/pdf/rotom-sigmod21.pdf](https://miaozhengjie.com/assets/pdf/rotom-sigmod21.pdf)
- **Sudowoodo** — Wang, Li, Wang, ICDE 2023 — [arXiv:2207.04122](https://arxiv.org/abs/2207.04122)
- **DeepMatcher / Magellan** — Mudgal et al., SIGMOD 2018, "Deep Learning for Entity Matching" — [pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf](https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)
- **DIAL** — Jain, Sarawagi, Sen, VLDB 2022 — [arXiv:2104.03986](https://arxiv.org/abs/2104.03986)
- **ALMSER-GB** — Primpeli & Bizer, ISWC 2021 (graph-boosted active learning for multi-source ER)
- **DTAL** — Kasai, Qian, Gurajada, Li, Popa, ACL 2019 — [aclanthology.org/P19-1586](https://aclanthology.org/P19-1586)
- **Fine-tuning LLMs for Entity Matching** — Steiner, Peeters, Bizer 2024 — [arXiv:2409.08185](https://arxiv.org/abs/2409.08185)
- **MatchGPT** — Peeters & Bizer 2023, "Using ChatGPT for Entity Matching" (cited by AnyMatch; no URL verified this pass)
- **EM benchmark critique** — Papadakis et al. 2024 (critical re-evaluation of EM benchmark datasets); Leone et al. 2022 (cited by AnyMatch as difficulty-curation motivation; no URL verified)
- **Tabular-representation survey** — Badaro, Saeed, Papotti 2023, "Transformers for Tabular Data Representation: A Survey," TACL

### Hard-example / hard-negative mining
- **Bootstrapping (origin)** — Sung & Poggio 1994 (face detection); Dalal & Triggs 2005 (HOG); formalized in **DPM** — Felzenszwalb et al., PAMI 2010 — [cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf](https://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf)
- **OHEM** — Shrivastava, Gupta, Girshick, CVPR 2016 — [arXiv:1604.03540](https://arxiv.org/abs/1604.03540)
- **Focal Loss** — Lin, Goyal, Girshick, He, Dollár, ICCV 2017 — [openaccess.thecvf.com/…/Lin_Focal_Loss_for_ICCV_2017_paper.html](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)
- **FaceNet** (semi-hard triplet mining) — Schroff, Kalenichenko, Philbin, CVPR 2015 — [arXiv:1503.03832](https://arxiv.org/abs/1503.03832)
- **N-pairs loss** — Sohn, NeurIPS 2016 — [proceedings.neurips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html](https://proceedings.neurips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html)
- **Lifted Structured Loss** — Song, Xiang, Jegelka, Savarese, CVPR 2016 — [dspace.mit.edu/handle/1721.1/113397](https://dspace.mit.edu/handle/1721.1/113397)
- **DPR** — Karpukhin et al., EMNLP 2020 — [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- **ANCE** — Xiong et al., ICLR 2021 — [arXiv:2007.00808](https://arxiv.org/abs/2007.00808)
- **RocketQA** — Qu et al., NAACL 2021 — [arXiv:2010.08191](https://arxiv.org/abs/2010.08191)
- **TAS-B** — Hofstätter et al., SIGIR 2021 — [arXiv:2104.06967](https://arxiv.org/abs/2104.06967)
- **NGAME** — Dahiya et al., WSDM 2023 — [arXiv:2207.04452](https://arxiv.org/abs/2207.04452)
- **Debiased Contrastive Learning** — Chuang, Robinson, Lin, Torralba, Jegelka, NeurIPS 2020 — [papers.nips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html](https://papers.nips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)
- **Contrastive Learning with Hard Negative Samples** — Robinson, Chuang, Sra, Jegelka, ICLR 2021 — [arXiv:2010.04592](https://arxiv.org/abs/2010.04592)

### Difficulty / coreset / label-noise / curriculum
- **EL2N / GraNd** ("Deep Learning on a Data Diet") — Paul, Ganguli, Dziugaite, NeurIPS 2021 — [arXiv:2107.07075](https://arxiv.org/abs/2107.07075)
- **Forgetting events** — Toneva et al., ICLR 2019 — [arXiv:1812.05159](https://arxiv.org/abs/1812.05159)
- **Dataset Cartography** — Swayamdipta et al., EMNLP 2020 — [aclanthology.org/2020.emnlp-main.746](https://aclanthology.org/2020.emnlp-main.746) · code: [github.com/allenai/cartography](https://github.com/allenai/cartography)
- **Core-Set / k-center** — Sener & Savarese, ICLR 2018 — [arXiv:1708.00489](https://arxiv.org/abs/1708.00489)
- **CRAIG** — Mirzasoleiman, Bilmes, Leskovec, ICML 2020 — [cs.stanford.edu/people/jure/pubs/craig-icml20.pdf](https://cs.stanford.edu/people/jure/pubs/craig-icml20.pdf)
- **GradMatch** — Killamsetty et al., ICML 2021 — [arXiv:2103.00123](https://arxiv.org/abs/2103.00123)
- **Moderate Coreset** — Xia et al., ICLR 2023 — code: [github.com/tmllab/2023_ICLR_Moderate-DS](https://github.com/tmllab/2023_ICLR_Moderate-DS)
- **Herding** — Welling, ICML 2009
- **Confident Learning / cleanlab** — Northcutt, Jiang, Chuang, JAIR 2021 — [arXiv:1911.00068](https://arxiv.org/abs/1911.00068)
- **Curriculum Learning** — Bengio, Louradour, Collobert, Weston, ICML 2009
- **Self-Paced (Curriculum) Learning** — Kumar et al. NeurIPS 2010 / Jiang et al. AAAI 2015 — [cdn.aaai.org/ojs/9608/9608-13-13136-1-2-20201228.pdf](https://cdn.aaai.org/ojs/9608/9608-13-13136-1-2-20201228.pdf)

### LLM-era selection / data-engine
- **LIMA** — Zhou et al. 2023 — [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- **AlpaGasus** — Chen et al. 2023 — [arXiv:2307.08701](https://arxiv.org/abs/2307.08701)
- **DEITA** — Liu et al. 2024 — [arXiv:2312.15685](https://arxiv.org/abs/2312.15685)
- **IFD / Cherry LLM** — Li et al., NAACL 2024 — [arXiv:2308.12032](https://arxiv.org/abs/2308.12032) · **Superfiltering** — Li et al., ACL 2024 — [arXiv:2402.00530](https://arxiv.org/abs/2402.00530)
- **LESS** — Xia et al., ICML 2024 — [arXiv:2402.04333](https://arxiv.org/abs/2402.04333)
- **SemDeDup** — Abbas et al. 2023 — [arXiv:2303.09540](https://arxiv.org/abs/2303.09540)
- **Can Foundation Models Wrangle Your Data?** — Narayan et al., VLDB 2022 — [arXiv:2205.09911](https://arxiv.org/abs/2205.09911)
- **Jellyfish** — VLDB 2024 — [arXiv:2312.01678](https://arxiv.org/abs/2312.01678)
- **Table-GPT** — 2023 — [arXiv:2307.08674](https://arxiv.org/abs/2307.08674)
- **SyNeg** — 2024 — [arXiv:2412.17250](https://arxiv.org/abs/2412.17250) · **Syntriever** — 2025 — [arXiv:2502.03824](https://arxiv.org/abs/2502.03824)
- **LLM-judge overconfidence** (mechanism for silver-label filtering) — [arXiv:2508.06225](https://arxiv.org/abs/2508.06225)

### Tooling
- **AutoGluon-Tabular** — Erickson et al. 2020 — [arXiv:2003.06505](https://arxiv.org/abs/2003.06505)
- **DSPy** — optimizer paper (MIPROv2), Opsahl-Ong et al. — [arXiv:2406.11695](https://arxiv.org/abs/2406.11695) · optimizers doc: [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md)
