# The Training Loop — replication targets, phasing, and framework deltas (decision-ready plan)

Date: 2026-07-13
Branch: `docs/training-loop-plan` (plan only — no execution, no training, no paid calls)
Status: proposed (for David's decision on targets + first-wave budget)
Amended 2026-07-13 (`docs/training-plan-llm-native`, maintainer feedback): the
ladder is reframed **LLM-native** — the gap-closing students are prompt-tuned
and fine-tuned small **LLMs**; Ditto/Magellan-class methods and the RF floor
are yardstick baselines kept for completeness, not methods to emulate — and a
new blocking-side experiment (**T4**: instruction transfer to the embedder) is
added and ranked.

## Context — why this, why now

The Peeters replication (`examples/research/peeters_llm_em_replication.py`)
proved langres can reproduce a published LLM-EM protocol exactly — F1 90.71 at
99.25% per-pair archive agreement, $0.28 real (docs/CHANGELOG.md). That is a
*prompt* replication: valuable as an honesty instrument, but not game-changing —
anyone with an API key can prompt.

The compelling story is the **full economics loop**:

> An expensive-but-accurate LLM harvests labels → we **train** a cheap model on
> them → **quality holds, cost collapses.**

Proving that requires replicating research where **training is the method** —
Ditto-class fine-tuned matchers (#80 Phase 2), AnyMatch's data-curation recipe
(#83), Jellyfish-class instruct-tuning (#81) — through langres's own seams, not
beside them. Epic #85 already carries the shared foundations (honest protocol,
the Qwen3 QLoRA ladder on the 8GB RTX 3070 via Unsloth, license hygiene); this
plan turns those issues into a ranked, budgeted execution order.

### Where we stand (verified against committed artifacts)

`data/benchmarks/phase1/` (PR #82, #80 Phase 1 — full standard test splits,
threshold never tuned on test):

| honest pairwise F1 | Amazon-Google | Abt-Buy | real cost |
|---|---|---|---|
| RandomForestJudge floor ($0, single-metric) | 0.360 | 0.404 | $0 |
| `deepseek-v4-flash` zero-shot LLM | 0.575 | 0.680 | $0.19 + $0.18 |
| `deepseek-v4-pro` zero-shot LLM | 0.614 | 0.737 | $2.56 + $2.27 |
| **Ditto (published, fine-tuned)** | **0.756** | **0.893** | — |

The zero-shot LLM beats the $0 floor by +0.21…+0.33 but sits **0.14–0.21 below
Ditto's band**. The open gap belongs to *training* — exactly the loop this plan
builds. Two levers we have already measured: (a) M4 showed a precision-tuned
DSPy **signature** lifts a cheap model 0.409 → 0.757 on AG with no compile spend;
(b) the flywheel (#79) showed noisy silver caps a student below its teacher on
hard data — so label-noise handling is a first-class requirement, not a nicety.

### The bet, stated falsifiably

*A smaller LLM — first DSPy-prompt-tuned, then QLoRA-fine-tuned, on labels
harvested by a tuned LLM teacher (silver) — retains ≥90% of the F1 the same
recipe reaches on gold labels, beats its own teacher's zero-shot F1, and serves
at ~$0 marginal cost per pair (vs the teacher's ~$8×10⁻⁵–1.2×10⁻³/pair).* If
false, we publish the gap and name why (precedent: the honest DBLP-Scholar PC
finding).

**The strategic frame (maintainer, 2026-07-13): the ladder is LLM-native.**
The "cheaper model" is not a from-scratch classifier. It is, in order: **(a) a
DSPy prompt-tuned smaller LLM** — teacher model + harvested labels drive the
prompt optimization of the small LLM — then **(b) a fine-tuned small LM** for
the task. The point is to *reuse the knowledge already encoded in LLMs and push
it further*, not to re-train 2018–2020-era architectures. Ditto's published
band remains the **yardstick**; Ditto/Magellan-class methods and the RF floor
are **baselines kept for completeness**, not methods to emulate.

## 1. Candidate replication targets, ranked

Ranking axis: **prove the framework carries training-based methods** ×
**announcement value**, then cost/risk. Summary first, detail below.

| rank | target | claim to replicate / test | paid $ | GPU h (est.) | new langres LOC (est.) | headline if it works |
|---|---|---|---|---|---|---|
| **1 — T1** | **Silver→student: close the Ditto gap, LLM-native** (#80 Phase 2 + flywheel thesis) | prompt-tuned → fine-tuned small LLM reaches the Ditto yardstick; silver-trained variant holds quality at collapsed cost | ~$2–5 (teacher tune + harvest) | ~4–10 | ~500–800 + tests | "An LLM labeled it; a smaller LLM learned it: Ditto-class quality at ~0 $/pair" |
| **2 — T2** | **AnyMatch recipe** (#83 / #86) | 124M-class tiny LM, curated via hard-positive mining, generalizes zero-shot (mean ~82 F1) | $0 | ~2–6 | ~250–400 + tests (mining seam) | "The data recipe is the lever — replicated through a reusable curation seam" |
| **3 — T4** | **Instruction transfer to blocking** (new; recall side) | matcher-side optimized instructions lift a same-family decoder embedder's blocking recall (PC@k) — an unpublished gap (#86 survey §13) | $0 first rung | ~0–2 (H1); H2 gated | ~0 core (knob exists) + harness | "One small-LLM family, both stages: the matcher's optimized prompt improves the blocker's recall" |
| **4 — T3** | **Jellyfish-style instruct-tune** (#81 Task 2) | Qwen3-1.7B/4B on Jellyfish-Instruct (CC-BY-4.0) vs Jellyfish-13B's 0.813 AG | $0 | ~6–20 | ~50–150 (reuses T1 seams) | "A 1.7B student beats a 13B specialist" (with seen-data asterisk) |
| — | Geo/POI conflation (#84) | out of scope this program — new entity type + new blocker, orthogonal to the training loop | — | — | — | — |

**Re-ranked in the LLM-native amendment:** T4 slots third — its first rung is
pure $0 reuse of existing seams and it tests the "one small-LLM base family for
both stages" strategic bet; T3 drops to fourth (largest GPU bill, most caveated
claim). Target labels are stable; the rank column is the order.

GPU-hour figures are **estimates to be verified by the Wave-0 smoke** (epic #85
rule: tooling maturity is confirmed empirically, never assumed). The training
targets share one training/eval substrate (Waves 0–1 below), which is why T3
costs almost no new code once T1 lands.

### T1 — Silver→student: close the Ditto gap, LLM-native (recommended headline target)

- **Papers.** Ditto (Li et al., VLDB 2020, arXiv:2004.00584) for the band
  (0.756 AG / 0.893 Abt-Buy, pairwise F1 on the canonical DeepMatcher splits);
  Steiner, Peeters & Bizer 2024 (arXiv:2409.08185) as the closest published
  precedent for LLM-generated EM training data (mixed results — read in full
  before the silver design; flagged unread in the #86 survey).
- **Exact claims — two rungs, reported separately:**
  - **Rung A (gold, replication-class):** the LLM-native ladder — prompt-tuned,
    then fine-tuned on the *gold* train split — reaches the Ditto yardstick band
    on the full test split under the honest protocol. A hosted trained matcher
    at that band completes #80 Phase 2 (the cross-encoder baseline below is the
    like-for-like Ditto-method anchor for the record).
  - **Rung B (silver, the economics claim):** the *same recipe* trained only on
    **teacher-harvested silver labels** (teacher never sees test) retains ≥90%
    of Rung A's F1 and beats the teacher's own zero-shot F1 — with the
    quality-vs-$/pair table as the deliverable.
- **Datasets.** `amazon_google` + `abt_buy` from the registry (vendored,
  canonical 3:1:1 splits: AG ≈ 6.9k/2293/2293 pairs, AB ≈ 5.7k/1916/1916).
- **Teacher.** `deepseek-v4-flash` behind a **precision-tuned DSPy signature**
  (the M4 lever; the measured Phase-1 profile is high-recall/low-precision, so
  untuned silver would be noise-heavy — the #79 trap). Pro-class teacher is a
  paid ablation, not the default (~$1.1–1.2×10⁻³/pair vs Flash's ~0.8–1.0×10⁻⁴).
- **Students — the LLM-native ladder (the gap-closing candidates), plus one
  baseline kept for completeness:**
  - **Rung (a) — DSPy prompt-tuned small LLM:** Qwen3-0.6B→1.7B served locally,
    signature/instructions optimized against the harvested (or gold) labels.
    Measured precedents: the M4 lever (precision-tuned signature took a cheap
    model 0.409 → 0.757 on AG with no compile spend) and Breunig's Qwen3-0.6B
    60.7% → 82% post-DSPy (epic #85). $0/call once local.
  - **Rung (b) — QLoRA fine-tuned small LM:** Qwen3-1.7B primary (0.6B smoke,
    4B stretch), Unsloth on the 3070 per the #85 ladder — pushes past what
    prompt-tuning alone reaches.
  - **Baseline (completeness, not the story):** a `sentence-transformers`
    `CrossEncoder` (roberta-base-class) trained on the same labels — the
    Ditto/Magellan-method reference point next to the RF floor, and the
    in-process `SupervisedFitMixin.fit(candidates, labels)` reference
    implementation. Already inside the `[semantic]` extra; ~1 GPU-h. *(Verify:
    the pinned sentence-transformers ≥5.x trains cross-encoders via
    `CrossEncoderTrainer`, not the old `.fit()` — API to confirm in Wave 0.)*
- **Cost.** Teacher harvest over train+valid only: ≈ $0.76 (AG) + $0.74 (AB) at
  measured Flash per-pair cost; signature tuning ≈ $1–2 → **$5 cap** is
  comfortable. GPU ≈ 1 h (cross-encoder) + 3–8 h (Qwen ladder).
- **Risk of an unflattering-but-honest result.** Medium. Rung A missing the
  band (e.g. 0.70 vs 0.756) is survivable — Ditto's own AG number varies
  0.66–0.80 across papers (protocol variance, epic #85), so "within the
  published band's spread" is the honest framing. Rung B is the real bet: if
  tuned-teacher silver still caps the student well below gold, the honest
  finding is "the flywheel's ceiling is the teacher's precision" — publishable,
  and it prices the next teacher rung.
- **Headline earned.** The maintainer's economics story, end-to-end, with real
  dollar columns on both sides: teacher $/pair vs student $/pair (~10³–10⁴×
  collapse if quality holds).

### T2 — AnyMatch: the data recipe through our harness

- **Paper.** AnyMatch (Zhang et al., arXiv:2409.04073, AAAI-25 GOOD-DATA):
  GPT-2-124M within ~4% of GPT-4 MatchGPT at ~3,900× lower cost; zero-shot mean
  ~82 F1 across 9 datasets (AG ~55, Abt-Buy ~86), leave-one-dataset-out.
- **Exact claim.** Their *recipe*, reimplemented from the paper (their repo has
  no LICENSE — never fork): hard-**positive** mining (misclassified positives
  from a cheap tabular classifier) + random negatives at 2:1 + attribute/
  structure augmentation, trained on N−1 datasets, evaluated zero-shot on the
  held-out one. Target: mean within ~5 F1 of theirs on the datasets we share.
- **Datasets.** 8 loadable registry sets (`fodors_zagat`, `amazon_google`,
  `abt_buy`, `dblp_acm`, `dblp_scholar`, `walmart_amazon`, `wdc_computers`,
  `febrl_person`) — close to but not identical to their 9; deviation named.
- **Models.** Student = Qwen3-0.6B (or GPT-2-124M for size fidelity — decision
  at run time; 0.6B aligns with the #85 ladder). No paid teacher: the miner is
  a **local classifier**. Fidelity deviation: they used AutoGluon; we start
  with the in-repo sklearn stack (`RandomForestJudge`'s family) as the miner
  and add `autogluon` as a dev-only dep **only if** numbers disappoint.
- **Cost.** $0 paid. GPU ≈ 1–3 h per held-out fold; run 2 headline folds (AG,
  AB) rather than all 8.
- **LOC.** This target *is* the #86 mining seam: generalize `select_for_review`
  into pluggable strategies and land `classifier-misclassification` +
  `blocking-derived hard negatives` (the survey's two cheapest new strategies),
  plus the Ditto-style `COL/VAL` serialization utility shared with T1/T3.
- **Risk.** Medium-high on exact numbers (different base model, miner, and
  dataset pool → "replication-inspired"), **low on framework value**: the
  mining seam is the most transferable artifact of the whole program (survey
  §0: three consumers — fit, DSPy demo pool, review queue).
- **Headline earned.** "A 124M-class model near GPT-4 on EM — because of data
  curation, and that curation is now a reusable langres seam."

### T4 — Instruction transfer to blocking: one small-LLM family for both stages (new)

- **The idea (maintainer, 2026-07-13).** Modern embedding models are
  decoder-LLM-based: **Qwen3-Embedding** (arXiv:2506.05176; Apache-2.0;
  0.6B/4B/8B; MTEB v2 Eng 70.70/74.60/75.22) is *"built upon the Qwen3
  foundation models"* (verified from the paper abstract + HF cards, #86 survey
  §12.3) and is **instruction-aware** — a prepended task instruction steers the
  embedding space. So use the SAME small-LLM base family (Qwen3-class) for both
  the DSPy-prompt-tuned matcher (T1 rung a) and the embedder, and test whether
  matcher-side prompt optimization **transfers** to the embedding side — prompt
  iteration is normally impossible for embedders (nothing is generated), but a
  shared base family + instruction-aware embeddings may let it reach blocking
  recall.
- **Honest mechanics correction (survey §12.4):** Qwen3-Embedding is a
  *separately released, contrastively-tuned checkpoint*, not the chat weights
  doing double duty — "one model" realistically means **one base family, two
  fine-tunes**. So the experiment splits into two hypotheses:
  - **H1 — text-level instruction transfer ($0):** does the DSPy-optimized
    *matching* instruction, reused as the embedding instruction, lift blocking
    recall on the off-the-shelf Qwen3-Embedding-0.6B? Instruction wording is a
    measured lever in the literature (E5-mistral: MTEB 64.5 with instructions
    vs 60.3 without; INSTRUCTOR +3.4%) — but both "optimize an embedder
    instruction for recall" and "a shared retrieve+match instruction" are
    **unpublished gaps** (survey §13.4): novel, genuinely unproven.
  - **H2 — genuinely shared weights (gated):** two LoRA adapters on one Qwen3
    base (contrastive embedding adapter + matching adapter, PEFT-swapped), or a
    GritLM-style joint probe — **no published result exists at 0.6–4B/QLoRA
    scale** (survey §12.2 caveat); a research bet, decided after Waves 1–4.
- **What's measurable, with what's already built:** Pair Completeness
  (recall) + Reduction Ratio via `core.metrics.evaluate_blocking`, plus
  MRR/NDCG via the `[eval]` extra — against the current encoder-baseline
  `VectorBlocker` at fixed k on registry benchmarks (AG, Abt-Buy,
  `wdc_computers`). **Zero new architecture for H1:** `VectorBlocker` already
  persists a `query_prompt` instruction knob in its config
  (`core/blockers/vector.py`), the embedder `encode()` already takes `prompt=`,
  and `examples/research/blocking_evaluation_with_instructions.py` /
  `instruction_embeddings_demo.py` already run Qwen3-family embeddings with
  instruction prompts.
- **$0-first steps, in order:** (1) **gating check** (survey §13.0): does
  off-the-shelf Qwen3-Embedding-0.6B behind `VectorBlocker` even beat the
  current encoder baseline on PC? The **UniBlocker counter-signal**
  (arXiv:2404.14831 — encoder backbones deliberately chosen for structured
  records) is the null hypothesis; if the decoder embedder loses, publish the
  negative and keep an encoder for blocking. (2) **instruction sweep**: none vs
  generic vs task-specific vs hand-varied instructions, reported as a
  *distribution* (wording is high-variance — "One prompt is not enough",
  arXiv:2605.22544 — never a single cherry-picked prompt). (3) **the transfer
  probe**: the matcher's DSPy-optimized instruction text as `query_prompt`,
  once Wave 1/2 produces it. Then, gated: (4) direct optimization of the
  embedding instruction against recall@k — **not** stock DSPy (MIPROv2 only
  tunes LM-completion modules; an embedder emits vectors, so
  `program.predictors()` finds nothing) but the standalone `gepa-ai/gepa`
  library + a custom adapter (survey §13.4 mechanism note); (5) H2 shared-base
  variants (GPU, budget gate).
- **Cost.** H1 entirely $0 (local embeddings, vendored data; steps 1–2 need no
  GPU beyond embedding throughput). H2: GPU-hours + possibly a small paid gate
  if GEPA's reflection LM isn't run locally — decided at the Wave-6 gate.
- **Risk.** Real chance H1 is flat or negative (UniBlocker counter-signal;
  instruction variance may swamp the transfer effect). Both outcomes are
  findings: a lift opens "prompt optimization reaches the recall side"; a null
  kills the shared-instruction story early and cheaply, before H2 spends
  anything. Evaluate *both* stages either way — the survey's transfer rule:
  measure PC *and* F1, so capability transfer vs interference is visible.
- **Headline earned.** "One small-LLM family, both stages: the matcher's
  optimized prompt improves the blocker's recall" — plus first-mover on an
  unpublished experiment either way.

### T3 — Jellyfish-style instruct-tune (defer behind a gate)

- **Paper.** Jellyfish (arXiv:2312.01678, VLDB 2024); training data
  `NECOUDBFM/Jellyfish-Instruct` is CC-BY-4.0 (usable; the 13B weights are
  CC-BY-NC — reference target only).
- **Exact claim.** Qwen3-1.7B (stretch 4B) fine-tuned on the Jellyfish-Instruct
  EM slice matches/beats Jellyfish-13B's 0.813 AG at ~8× smaller.
- **Honesty caveat that shapes the announcement.** Per the verified #85 facts:
  AG was **seen** in Jellyfish's training mix; their zero-shot number is on
  held-out sets (Abt-Buy-class). So "beats 0.813 on AG" is seen-vs-seen; the
  strong claim requires holding Abt-Buy out of our fine-tune too. (#81 Task 0 —
  the SOTA review — re-verifies the exact seen/unseen table before any run.)
- **Cost.** $0 paid; the largest GPU bill (instruction set size unverified —
  the ~422 MB dataset spans four task families; the EM slice must be measured
  first). 4B on 8 GB is tight even 4-bit — verify empirically, never assume.
- **Risk.** Highest: a 13B→1.7B size drop may simply lose, and the seen-data
  asterisk weakens the headline even on a win. Ranked third; becomes cheap once
  T1's training substrate exists (~50–150 new LOC: a dataset adapter).
- **Headline earned.** The strongest *David-vs-Goliath* line if it works —
  but only with the zero-shot framing done right.

## 2. Phasing — $0 first, explicit paid gates

Historically approved cap per paid wave: $5–$10. This program asks for **two
paid gates totaling ≤$10**; everything else is $0 (local GPU + vendored data).

| wave | content | spend | gate |
|---|---|---|---|
| **0 — foundations ($0)** | (a) harvest→fit **id-alignment bridge** (`LabeledPair` id-keyed → positionally-aligned `(candidates, labels)` — the missing glue named in #80); (b) **mining seam v1**: pluggable strategy interface generalized from `select_for_review` + `blocking-derived hard negatives` + `classifier-misclassification` strategies; (c) `COL/VAL` serialization utility; (d) **CrossEncoderJudge scaffold** (fit/forward/save_state/load_state/config) dry-run-trained on `tiny_fixture` in CI (5 steps, CPU); (e) **GPU smoke**: Unsloth + Qwen3-0.6B QLoRA overfits 100 pairs on the 3070 — confirms tooling + calibrates real GPU-h; (f) eval wiring is **already done** (`FixedSplitPairBenchmark` + `evaluate_fixed_split_honest`) — reuse, don't build | $0 | none |
| **1 — T1 Rung A: gold LLM-native ladder ($0)** | DSPy prompt-tune Qwen3-0.6B→1.7B locally against AG/AB **gold train/valid** labels (rung a), then QLoRA fine-tune (rung b); train the cross-encoder baseline alongside for completeness (completes #80 Phase 2); evaluate all on full test, threshold from train/valid; report next to floor / zero-shot / Ditto-yardstick rows | $0 (GPU) | drop a ladder rung here if one clearly dominates |
| **1b — T4-H1: blocking instruction probes ($0)** | Gating check (Qwen3-Embedding-0.6B via `VectorBlocker.query_prompt` vs the encoder baseline, PC/RR at fixed k); instruction sweep reported as a distribution; the transfer probe runs as soon as Wave 1/2 yields a DSPy-optimized instruction. Existing harness: `blocking_evaluation_with_instructions.py`, `evaluate_blocking` | $0 | if the gating check is negative, publish it and drop T4-H2 |
| **2 — teacher tune + silver harvest (PAID gate 1)** | Precision-tune the Flash-class teacher's DSPy signature on train/valid (M4 lever, no MIPROv2 compile — it didn't pay at M4); harvest silver over **train+valid only** for AG+AB; run the label-noise rail (below) on the silver set | **≤$5 cap** | ask David; abort criterion: tuned-teacher valid-split F1 must beat its untuned 0.575/0.680, else stop before harvesting |
| **3 — T1 Rung B: silver students + economics tearsheet ($0)** | Retrain Wave-1 recipes on silver; grade on the same held-out gold test; produce the quality-vs-$/pair table (the `EvalReport` HTML tearsheet from PR #107 is the natural renderer) | $0 (GPU) | none |
| **4 — T2 AnyMatch folds ($0)** | Mining seam as consumer; 2 leave-one-out headline folds | $0 (GPU) | none |
| **5 — comparison anchors (PAID gate 2, optional)** | Either a Pro-class-teacher silver ablation on ONE dataset (teacher-quality curve) or a GPT-4o MatchGPT row via the existing Peeters harness (anchors T2's "% of GPT-4" claim) | **≤$5 cap** | ask David; skip if Waves 2–4 already tell the story |
| **6 — T3 / T4-H2 gate** | Decide the Jellyfish-style run (#81 Task 0 SOTA review first) and the T4-H2 shared-base variants (two-adapter Qwen3 base; GEPA embedding-instruction optimization; GritLM-style joint probe) from Wave 1–4 evidence | $0 paid | David decision |

**Label-noise rail (Wave 2/3, from the #86 survey — required, not optional):**
parse failures are already excluded by the judgement contract (PR #106: `None`/
`None` = abstain, never a fabricated 0.5 — this is what makes silver harvesting
safe at all); additionally (i) drop/flag abstains and low-confidence teacher
calls, (ii) Cleanlab-style confident-joint filter over out-of-fold student
probabilities before pairs enter training as ground truth, (iii) report silver
noise rate against gold on the *valid* split (never test) so the student's
ceiling is explained, not mysterious.

## 3. Framework deltas — what langres itself gains

1. **Harvest→fit bridge** (`core/harvest.py` ↔ `SupervisedFitMixin`) — re-block
   + join on `frozenset({left_id, right_id})` → aligned labels. ~60–100 LOC.
   Unblocks *every* trained judge, not just this program.
2. **A trained-transformer judge (the baseline seam)** — `CrossEncoderJudge`
   implementing `SupervisedFitMixin`, `save_state`/`load_state` writing a
   `safetensors` checkpoint into `state_dir` (the `FAISSIndex` binary-sidecar
   pattern; no pickle), `config`/`from_config`, wired at **all three
   judge-dispatch sites** + `@register` + lazy import (torch stays out of bare
   `import langres`) + `PairwiseJudgement.score_type` `"prob_deep"`. Replaces
   the dead `GLinkerAdapter` stub as the in-process reference deep judge —
   under the LLM-native reframe it is the *baseline* trained judge and the
   `SupervisedFitMixin` reference implementation, while the headline students
   ride the LLM seams (below). Inference dep rides the existing `[semantic]`
   extra (sentence-transformers/torch already there); **training-only deps
   (unsloth, trl/peft, optionally autogluon) stay dev-group-only** — never
   production extras.
3. **Local-LLM student serving** — decision, not code: Option A (recommended)
   wrap the fine-tuned checkpoint behind the existing `LLMJudge` via a local
   OpenAI-compatible server (`api_base`) — zero core changes, evaluates through
   the *identical* seam as the paid teacher; Option B, a small in-process
   transformers judge (~150 LOC) if serving on the 3070 box proves awkward.
4. **The mining seam (#86 design step)** — `mine(candidates | judgement_log,
   strategy, …) -> ranked pairs` with pluggable strategies, `select_for_review`
   refactored to consume it (review queue unchanged), first two new strategies
   landed (S3 blocking-derived hard negatives, S1 classifier-misclassification),
   outputs feeding fit / DSPy demo pool / review.
5. **Serialization utility** — one Ditto-style `COL/VAL` textualizer shared by
   trained students, LLM prompts, and the AnyMatch recipe (today each judge
   serializes ad hoc).
   5b. **Decoder-LLM embedder for blocking (T4) — no new architecture.**
   `VectorBlocker` already takes a pluggable embedder and already persists a
   `query_prompt` instruction knob in its config (`core/blockers/vector.py`);
   the embedder protocol's `encode(texts, prompt=...)` carries the instruction.
   T4-H1 is therefore harness work, not core work (the research examples
   already run Qwen3-family embeddings this way). Only T4-H2 would add code: an
   adapter-swapped shared-base embedder backend, trained dev-side.
6. **Trained artifacts get an identity** — cross-reference the parallel design
   doc `docs/research/20260713_model_identity_and_hub.md` (branch
   `docs/hub-model-identity`, PR #108): a checkpoint is a first-class artifact
   whose identity must capture base model, data recipe (**including
   silver-teacher identity and license chain**), and eval provenance. This plan
   supplies the hub design's first real customers (T1/T2 checkpoints); the hub
   doc supplies where they live beyond `state_dir` and how they're named — its
   model-card schema carries these four requirements (cross-references
   reconciled both directions, 2026-07-13). **Vocabulary alignment (agreed with
   the hub design):** silver-teacher identity on harvested pairs reuses the hub
   doc's v0.3 **method-id + model-id stamp** on `JudgementLog`/RunRecord — one
   lineage vocabulary from harvest to published artifact, not a parallel field.
   Training runs should also record the `recipe_id`/`attempt_id` identity from
   the experiment-tracking plan
   (`docs/plans/20260708_tracking_observability_plan.md`) — coordinate, don't
   duplicate.

## 4. Success criteria & honest-protocol rules

**Protocol (inherited from #80/#85, binding for every run):**
- Full standard test split, never subsampled; threshold pinned from train/valid
  (`derive_threshold`/`evaluate_fixed_split_honest`); argmax-on-test reported
  only as the labeled leakage delta.
- **Silver labels come from train+valid only. The teacher never judges test for
  any purpose that feeds training.** In particular the committed Phase-1 LLM
  judgements are **test-split** logs — they must never enter any training set.
- Held-out gold test is graded once per final recipe, not iterated against.
- Seen-data caveats stated in every comparison (Jellyfish AG = seen); cross-paper
  F1 is not apples-to-apples (Ditto AG 0.66–0.80 across papers) — name the
  protocol next to every number.
- Fidelity deviations from a replicated paper are enumerated per run (base
  model, miner, dataset pool, serialization).
- Licenses named per artifact (Qwen3 Apache-2.0; Jellyfish-Instruct CC-BY-4.0;
  Jellyfish weights CC-BY-NC → reference only; AnyMatch code unlicensed → paper
  reimplementation only; OpenSanctions CC-BY-NC → untouched by this program).
- Every economics claim carries both sides: teacher $/pair (measured, not
  listed prices) and student $/pair (~$0 marginal; GPU amortization named).
- **Blocking-side (T4) runs:** report PC + RR (and MRR/NDCG where `[eval]` is
  present) at fixed k against the encoder baseline; instruction results are
  reported as a **distribution over instruction variants**, never a single
  cherry-picked prompt; the UniBlocker counter-signal is the stated null
  hypothesis, so a negative is a finding; every T4 variant is evaluated on
  *both* stages (PC and downstream F1) to expose transfer vs interference.

**Success bars:**
- **T1-A:** honest F1 within the published Ditto spread (≥ ~0.70 AG / ≥ ~0.85
  AB) ⇒ "the seam hosts a trained SOTA-class matcher."
- **T1-B:** silver student ≥ teacher's zero-shot honest F1 **and** ≥ 90% of the
  gold student's F1 ⇒ the economics loop closes. Below that: publish the gap +
  the silver noise rate that explains it.
- **T2:** leave-one-out F1 within ~5 pts of AnyMatch on shared datasets, with
  deviations named ⇒ the curation seam carries a published recipe.
- **T4-H1:** the transferred (or swept-best) instruction lifts PC@k over both
  the no-instruction and generic-instruction baselines on ≥2 registry
  benchmarks at unchanged k/RR ⇒ prompt optimization reaches the recall side.
  Gating-check failure (decoder embedder ≤ encoder baseline) ⇒ keep the
  encoder for blocking, publish the negative, drop H2.
- Any outcome is reported — the honest-but-unflattering result is a finding,
  not a failure (precedent: DBLP-Scholar PC 0.39, the M4 "MIPROv2 didn't help").

## 5. Recommended first wave (decision requested)

**Recommendation: Wave 0 + Wave 1 + Wave 1b now ($0), with Wave 2 pre-approved
at a $5 cap** so the silver harvest starts the moment the gold ladder exists.
Wave 1b (the T4 gating check + instruction sweep) is free, independent, and can
run on the dev machine in parallel. That is one budget ask: **$5** (second $5
gate deferred until Wave 5 is justified).

Alternatives, left live:
- **B — AnyMatch-first (T2 before T1):** $0 all the way to a publishable
  replication + lands the #86 seam earliest. Costs the economics headline a
  delay; the maintainer's framing points at T1, so this is the fallback if the
  Wave-0 GPU smoke surfaces QLoRA tooling friction (T2's 0.6B/GPT-2 student is
  the least demanding).
- **C — baseline-first T1 (cross-encoder only in Wave 1):** fastest to a
  Ditto-method-class number (~1 GPU-h), smallest surface — but under the
  LLM-native reframe it delivers only the yardstick baseline, not the story.
  Right call only if the 3070 is unavailable (the cross-encoder trains
  anywhere).
- **D — teacher-first (run Wave 2 before Wave 1):** proves the silver pipeline
  earlier but spends money before the $0 gold baseline exists to compare
  against — weaker experiment design; not recommended.

## 6. Access / front-loaded needs (rule #1)

- **8GB RTX 3070 box**: confirm availability + CUDA/driver state before Wave 0e;
  everything else in Waves 0–1 runs on the dev machine ($0, CPU-tolerable for
  the cross-encoder dry-run).
- **OPENROUTER_API_KEY** only at Wave 2/5 gates (sandbox off for those runs);
  one `asyncio.run` per process; per-call persistence before the next paid call
  (paid-run durability rule).
- New dev-only deps at Wave 0: `unsloth` (+ transformers/trl/peft pins), later
  optionally `autogluon` — dev group, never extras.
- HuggingFace reachable (already allowlisted) for base checkpoints +
  Jellyfish-Instruct (T3 only).

## 7. Out of scope

- #84 geo/POI conflation (new entity type + spatial blocker — own program).
- DSPy `BootstrapFinetune` distillation (#81 Task 3) — revisit after T1-B; it
  competes with, not precedes, the direct QLoRA path.
- Any OpenSanctions data (CC-BY-NC), anything requiring cloud GPUs (out of the
  $10 envelope; re-decide only if 4B/T3 justifies it), PyPI/publishing (#55).
- The general `Optimizer`, synthetic data generation.

## 8. Unverified — must be checked before the relevant wave

- sentence-transformers ≥5.x cross-encoder training API (`CrossEncoderTrainer`)
  and 8GB-fit of Qwen3-4B QLoRA — Wave 0 smoke.
- Real GPU-hours (all figures above are estimates).
- Jellyfish-Instruct EM-slice row count + exact seen/unseen table — #81 Task 0.
- Steiner/Peeters/Bizer 2024's concrete method (read in full before Wave 2's
  silver design — survey flag).
- Whether the tuned-teacher lift (M4's 0.409→0.757 was on GLM) transfers to
  DeepSeek-Flash — Wave 2's abort criterion exists precisely for this.
- **T4 facts, provenance:** Qwen3-Embedding "built upon the Qwen3 foundation
  models", 0.6B/4B/8B, Apache-2.0, instruction-aware — verified from the
  arXiv:2506.05176 abstract (this amendment) + the #86 survey's HF-card check
  (2026-07-07). NOT verified: that our pinned sentence-transformers loads
  Qwen3-Embedding-0.6B on this exact code path (the research examples run a
  Qwen3-family embedder — confirm it is the same loader in Wave 1b); GEPA
  custom-adapter glue for direct embedding-instruction optimization (survey
  mechanism note — stock DSPy cannot do it); GritLM-style joint training at
  0.6–4B/QLoRA scale (no published result — survey §12.2).
- **T4-H1's core hypothesis — that matcher-optimized instruction text
  transfers to embedding recall — is unpublished and unproven**; the survey's
  "shared identity-core" framing (§13.4) is the thing being tested, not
  assumed.
