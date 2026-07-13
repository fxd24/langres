# The Training Loop — replication targets, phasing, and framework deltas (decision-ready plan)

Date: 2026-07-13
Branch: `docs/training-loop-plan` (plan only — no execution, no training, no paid calls)
Status: proposed (for David's decision on targets + first-wave budget)

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

*A small model fine-tuned on labels harvested by a tuned LLM teacher (silver)
retains ≥90% of the F1 the same model reaches on gold labels, beats its own
teacher's zero-shot F1, and serves at ~$0 marginal cost per pair (vs the
teacher's ~$8×10⁻⁵–1.2×10⁻³/pair).* If false, we publish the gap and name why
(precedent: the honest DBLP-Scholar PC finding).

## 1. Candidate replication targets, ranked

Ranking axis: **prove the framework carries training-based methods** ×
**announcement value**, then cost/risk. Summary first, detail below.

| rank | target | claim to replicate | paid $ | GPU h (est.) | new langres LOC (est.) | headline if it works |
|---|---|---|---|---|---|---|
| **T1** | **Silver→student: close the Ditto gap** (#80 Phase 2 + flywheel thesis) | trained student reaches Ditto band; silver-trained variant holds quality at collapsed cost | ~$2–5 (teacher tune + harvest) | ~4–10 | ~500–800 + tests | "An LLM labeled it; a small local model learned it: Ditto-class quality at ~0 $/pair" |
| **T2** | **AnyMatch recipe** (#83 / #86) | 124M-class tiny LM, curated via hard-positive mining, generalizes zero-shot (mean ~82 F1) | $0 | ~2–6 | ~250–400 + tests (mining seam) | "The data recipe is the lever — replicated through a reusable curation seam" |
| **T3** | **Jellyfish-style instruct-tune** (#81 Task 2) | Qwen3-1.7B/4B on Jellyfish-Instruct (CC-BY-4.0) vs Jellyfish-13B's 0.813 AG | $0 | ~6–20 | ~50–150 (reuses T1 seams) | "A 1.7B student beats a 13B specialist" (with seen-data asterisk) |
| — | Geo/POI conflation (#84) | out of scope this program — new entity type + new blocker, orthogonal to the training loop | — | — | — | — |

GPU-hour figures are **estimates to be verified by the Wave-0 smoke** (epic #85
rule: tooling maturity is confirmed empirically, never assumed). All three
targets share one training/eval substrate (Waves 0–1 below), which is why T3
costs almost no new code once T1 lands.

### T1 — Silver→student: close the Ditto gap (recommended headline target)

- **Papers.** Ditto (Li et al., VLDB 2020, arXiv:2004.00584) for the band
  (0.756 AG / 0.893 Abt-Buy, pairwise F1 on the canonical DeepMatcher splits);
  Steiner, Peeters & Bizer 2024 (arXiv:2409.08185) as the closest published
  precedent for LLM-generated EM training data (mixed results — read in full
  before the silver design; flagged unread in the #86 survey).
- **Exact claims — two rungs, reported separately:**
  - **Rung A (gold, replication-class):** a student fine-tuned on the *gold*
    train split reaches the Ditto band on the full test split under the honest
    protocol. This alone completes #80 Phase 2.
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
- **Students — two shapes, both planned, one gate to drop either:**
  - **Cross-encoder** (Ditto-fidelity anchor): `sentence-transformers`
    `CrossEncoder` (roberta-base-class) — already inside the `[semantic]` extra,
    trains in well under an hour on the 3070, and implements
    `SupervisedFitMixin.fit(candidates, labels)` naturally. *(Verify: the
    pinned sentence-transformers ≥5.x trains cross-encoders via
    `CrossEncoderTrainer`, not the old `.fit()` — API to confirm in Wave 0.)*
  - **Qwen3-1.7B QLoRA** (forward-looking, the #85 ladder): 0.6B smoke → 1.7B
    primary → 4B stretch, Unsloth on the 3070.
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
| **1 — T1 Rung A: gold students ($0)** | Train cross-encoder + Qwen3-0.6B→1.7B on AG/AB **gold train**; evaluate on full test, threshold from train/valid; report next to floor/zero-shot/Ditto rows. Completes #80 Phase 2. | $0 (GPU) | drop the weaker student shape here if one clearly dominates |
| **2 — teacher tune + silver harvest (PAID gate 1)** | Precision-tune the Flash-class teacher's DSPy signature on train/valid (M4 lever, no MIPROv2 compile — it didn't pay at M4); harvest silver over **train+valid only** for AG+AB; run the label-noise rail (below) on the silver set | **≤$5 cap** | ask David; abort criterion: tuned-teacher valid-split F1 must beat its untuned 0.575/0.680, else stop before harvesting |
| **3 — T1 Rung B: silver students + economics tearsheet ($0)** | Retrain Wave-1 recipes on silver; grade on the same held-out gold test; produce the quality-vs-$/pair table (the `EvalReport` HTML tearsheet from PR #107 is the natural renderer) | $0 (GPU) | none |
| **4 — T2 AnyMatch folds ($0)** | Mining seam as consumer; 2 leave-one-out headline folds | $0 (GPU) | none |
| **5 — comparison anchors (PAID gate 2, optional)** | Either a Pro-class-teacher silver ablation on ONE dataset (teacher-quality curve) or a GPT-4o MatchGPT row via the existing Peeters harness (anchors T2's "% of GPT-4" claim) | **≤$5 cap** | ask David; skip if Waves 2–4 already tell the story |
| **6 — T3 gate** | Decide Jellyfish-style run from Wave 1–4 evidence (#81 Task 0 SOTA review first) | $0 paid | David decision |

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
2. **A trained-transformer judge** — `CrossEncoderJudge` implementing
   `SupervisedFitMixin`, `save_state`/`load_state` writing a `safetensors`
   checkpoint into `state_dir` (the `FAISSIndex` binary-sidecar pattern; no
   pickle), `config`/`from_config`, wired at **all three judge-dispatch sites**
   + `@register` + lazy import (torch stays out of bare `import langres`) +
   `PairwiseJudgement.score_type` `"prob_deep"`. Replaces the dead
   `GLinkerAdapter` stub as the reference deep judge. Inference dep rides the
   existing `[semantic]` extra (sentence-transformers/torch already there);
   **training-only deps (unsloth, trl/peft, optionally autogluon) stay
   dev-group-only** — never production extras.
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
6. **Trained artifacts get an identity** — cross-reference the parallel design
   doc `docs/research/20260713_model_identity_and_hub.md` (branch
   `docs/hub-model-identity`, being written now): a checkpoint is a first-class
   artifact whose identity must capture base model, data recipe (**including
   silver-teacher identity and license chain**), and eval provenance. This plan
   supplies the hub design's first real customers (T1/T2 checkpoints); the hub
   doc supplies where they live beyond `state_dir` and how they're named.
   Reconcile cross-references when both docs land. Training runs should also
   record the `recipe_id`/`attempt_id` identity from the experiment-tracking
   plan (`docs/plans/20260708_tracking_observability_plan.md`) — coordinate,
   don't duplicate.

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

**Success bars:**
- **T1-A:** honest F1 within the published Ditto spread (≥ ~0.70 AG / ≥ ~0.85
  AB) ⇒ "the seam hosts a trained SOTA-class matcher."
- **T1-B:** silver student ≥ teacher's zero-shot honest F1 **and** ≥ 90% of the
  gold student's F1 ⇒ the economics loop closes. Below that: publish the gap +
  the silver noise rate that explains it.
- **T2:** leave-one-out F1 within ~5 pts of AnyMatch on shared datasets, with
  deviations named ⇒ the curation seam carries a published recipe.
- Any outcome is reported — the honest-but-unflattering result is a finding,
  not a failure (precedent: DBLP-Scholar PC 0.39, the M4 "MIPROv2 didn't help").

## 5. Recommended first wave (decision requested)

**Recommendation: Wave 0 + Wave 1 now ($0), with Wave 2 pre-approved at a $5
cap** so the silver harvest starts the moment the gold students exist. That is
one budget ask: **$5** (second $5 gate deferred until Wave 5 is justified).

Alternatives, left live:
- **B — AnyMatch-first (T2 before T1):** $0 all the way to a publishable
  replication + lands the #86 seam earliest. Costs the economics headline a
  delay; the maintainer's framing points at T1, so this is the fallback if the
  Wave-0 GPU smoke surfaces QLoRA tooling friction (T2's 0.6B/GPT-2 student is
  the least demanding).
- **C — cross-encoder-only T1 (skip the Qwen ladder in Wave 1):** fastest to a
  Ditto-band number (~1 GPU-h), smallest surface; gives up the small-LM
  narrative until T3. Right call if 3070 access is the bottleneck.
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
