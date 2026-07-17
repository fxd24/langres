# Research Agenda — the experiments the framework unlocks

> **Status:** Agenda / living note (2026-07-17). This is a **backlog of questions**, not
> a plan and not a result. It records what is worth *measuring* once the core framework
> is ready, why each question matters to langres, what is already known (with citations),
> what would settle it, and what it would cost.
>
> **Scope boundary.** Framework/build work is **out of scope** here. `docs/ROADMAP.md`
> owns the milestones and `docs/plans/` owns executable build plans. This doc is the
> layer *above* both: what the framework is *for*. An item here becoming buildable is a
> signal to write a plan, not to expand this doc.
>
> **Reads with:**
> - `docs/THEORY.md` — the mathematical foundation. **⚠ Not on `main` as of this
>   writing**: it lives on the unmerged branch `docs/theory-foundation`, and every
>   reference below is pinned to **`7a106ce`** ("docs(theory): ground THEORY.md in the
>   literature; retract four claims") so the citation does not dangle. Items A1–A3 are
>   the experiments its §6.5 explicitly asks for; this doc does **not** re-derive it.
> - `docs/research/20260707_data_prep_hard_case_mining_survey.md` — the mining/selection
>   literature backbone. Items C4/C5 are its retrieval-training thread, made measurable.
> - `docs/research/20260715_data_prep_architecture.md` §6 — the data-prep *backlog*. This
>   doc is the wider agenda; that one owns data-prep method choice.
> - `docs/research/20260701_er_seam_audit.md` — the SOTA seam audit (issue #55).

---

## 0. How to read this

Every item has the same five fields:

| Field | Means |
|---|---|
| **The question** | Stated so it can come out *no*. If it cannot fail, it is not here. |
| **Why it matters to langres** | The decision it changes. An item that changes no decision is deleted. |
| **What's already known** | Cited. Primary sources only. Confidence flagged. |
| **What would settle it** | The concrete measurement — not "investigate". |
| **Cost & prerequisites** | $0 / CPU / GPU / paid-API, and what must land first. |

**Method discipline** (this project has been burned — see `.claude/rules/expert-knowledge.md`):
every number below is transcribed from a primary source **read directly**, or is marked
`[unverified]`. A hypothesis is labelled a hypothesis. Novelty is not a goal — the
owner's standard is *"We don't care how much we are contributing with novelty. We care
that what we have is good and correct."* Several items below exist to find out that we
are **wrong**, which is the cheapest possible outcome.

---

## 1. The board

*(priority/cost table — filled in below as items are drafted)*

---

## 3. Thread B — $0 diagnostics on data we already have

Both items here are **free**. Neither needs a GPU, a paid API, or a new component. Both
have been sitting one script away from an answer for months. If anything on this board
gets run, it should be these.

### B1 — The closure diagnostic: does any output cluster contain a pair we scored *below* threshold?

**The question.** Exactly as stated. Run the pipeline, take the output clusters, and ask
whether any two records in the same cluster were **judged and rejected**. If yes: how
often, and does it track the over-merge we already measured?

**Why it matters to langres.** `Clusterer` (transitive closure) is the **default**, and
per `THEORY.md`@`7a106ce` §7 it is correlation clustering with **$+\infty$ on observed
positive edges and $0$ on everything else** — observed negatives *and* unobserved pairs
alike. So the default **discards evidence we already paid for**. Every rejected pair
inside an output cluster is a judgement — sometimes a *paid LLM* judgement — that the
clusterer threw away. The diagnostic turns a theoretical objection into a count, and that
count decides a default.

**What's already known.** Enough to make this cheap and to know what "bad" looks like:

- **The theory is settled, and it is not "closure is bad".** `THEORY.md` §7 (via Bansal
  et al.): *"**Closure is optimal iff the pairwise labels are consistent.** Its failure
  mode is precisely inconsistency."* And §7.1 sharpens the critique to
  **threshold-fragility**, not badness. `THEORY.md` §7.2 already names this exact
  diagnostic and prices it: *"the edges that were candidates and scored **below** $t$ are
  already in the `JudgementLog`. The diagnostic "does this cluster contain a pair we
  scored below threshold?" costs **$0** and directly measures (b)"* — where (b) is
  *whether observed negatives are priced at all*. **This item is that sentence, promoted
  to a task.**
- **The failure is threshold-fragile, with a measured collapse.** Hassanzadeh et al.
  (via §7.1): Partitioning at **F1 0.850** (θ=0.4, beating MinCut) collapses to
  **0.177** at θ=0.2, merging **500 true clusters into 51**. That is a 4.8× cluster-count
  collapse from a threshold move of 0.2 — evidence the failure is a *cliff*, not a slope.
- **We have already been bitten, and already built the fix.** `CorrelationClusterer`
  (`src/langres/core/clusterers/correlation.py`) — the **Ailon–Charikar–Newman pivot
  algorithm** — is implemented, registered, and a **drop-in** (same ctor/config, inherits
  `evaluate`/`inspect_clusters`). A node joins a cluster only via a **direct** edge to
  that cluster's pivot, so it is structurally resistant to chaining. Its docstring names
  the motivating measurement: the **M3 over-merge failure mode, −0.63 BCubed**. **It is
  not the default.** `[verified — read the source]`
- **The instrument for the follow-through exists too.** `core/metrics.py` has
  `PairMetrics` / `classify_pairs`, which exist *precisely* to separate the scorer's
  quality from the clusterer's amplification — the docstring's words: *"transitive
  closure can chain one false-positive edge into many false-positive pairs."*

**What would settle it.** Rebuild the clusterer's edge set from the log, run connected
components, and scan for same-component pairs that were logged as non-matches. **Two
implementation traps** — both discovered by reading the source, and either one silently
invalidates the result:

1. **Do not scan `score < threshold`.** The `JudgementLog` row schema (v3) has **no
   `threshold` column** — the threshold is consumed at write time to compute the `verdict`
   field and is never persisted. Rebuild edges from **`verdict == True`**, which is the
   *same* `predicted_match(judgement, threshold)` predicate `Clusterer.cluster()` uses,
   so the reconstruction is exact.
2. **`predicted_match` gives `decision` precedence over `score`.** For a *decider* judge,
   `verdict` can be `True` with a `score` below threshold (or `None`). A naive
   score-vs-threshold scan **mis-flags exactly those rows** — it would manufacture the
   finding it is looking for. This is the trap that makes the naive version of this
   experiment worse than not running it.

   *(If the raw threshold number is needed, it is on the `RunRecord`'s `resolver_config`,
   joined on `run_id` — not in the JSONL.)*

Report: **(a)** the count and rate of below-threshold pairs inside output clusters;
**(b)** their distribution over cluster size — *hypothesis:* they concentrate in the
largest clusters, which is what chaining predicts; **(c)** the same run under
`CorrelationClusterer`, which prices those negatives. The comparison is the decision:
**if (a) is ~0, closure is fine on our data and §7 is operationally uninteresting** —
a genuinely useful negative that would retire this thread. If (a) is large and (c) fixes
it, the default should change.

**Cost & prerequisites.** **$0.** No new component: `JudgementLog` (with `log=` on
`ERModel.dedupe()`), `CorrelationClusterer`, and `PairMetrics` all ship today. One
script over an existing log; re-running a pipeline to *produce* a log costs whatever the
matcher costs (**$0** on the `rapidfuzz`/`embedding_cosine` path). **Blocked by nothing.**
This is the single cheapest item on the board and it can be run this afternoon.

---

### B2 — What *is* the distribution of the standard ER benchmarks?

**The question.** For each registered benchmark: class balance, pair-difficulty
distribution, duplicate-cluster-size distribution, field sparsity/missingness,
string-length and vocabulary overlap. Then the two that actually matter: **which are
saturated** (everyone scores ≥0.98; no headroom to measure anything) and **which are
unrepresentative** (a structural artifact makes them measure something other than ER)?

**Why it matters to langres.** Every claim langres makes is measured on this portfolio.
If the portfolio is saturated or skewed, **every number we report is about the
benchmarks, not the methods** — including the numbers on this board. A method that wins
on a saturated set has demonstrated nothing. This item is the calibration of our own
instrument, and it gates the interpretation of C1–C5.

**What's already known.** **Most of the machinery is already built** — this item is
mostly *running* it, which is why it is cheap. Checked against the source:

- **`DataProfileReport`** (`src/langres/data/data_profile/`) already computes almost the
  whole list. It is a container of `ProfileSection`s; the relevant ones and their actual
  fields: `[verified — read the models]`
  - `LabelStructureSection` → `n_clusters`, `n_singletons`, `max_cluster_size`,
    `mean_cluster_size`, `positive_pairs`, `total_pairs`, **`prevalence`**,
    **`imbalance_ratio`**, `entropy_bits`, **`size_distribution`** → *class balance and
    cluster-size distribution, done*.
  - `CorpusFieldSection` → per-field `non_null_rate`, `n_distinct`, `uniqueness`,
    `mean_len`, `median_len`, `len_hist`, `all_null` → *field sparsity and string length,
    done*.
  - `SeparabilitySection` → `auc` + per-class histograms → *a difficulty proxy, done*.
  - `HeroSection` lifts `prevalence`, `imbalance_ratio`, `separability_auc` to the top.
  - **Gap:** **vocabulary overlap is not computed by any section.** It is the one genuinely
    new measurement this item needs.
- **The portfolio is 10 registered benchmarks, not 7** (`langres/data/registry.py`) — every
  dataset named in the brief is present, plus `dblp_acm` and `tiny_fixture`:
  `abt_buy`, `amazon_google`, `dblp_acm`, `dblp_scholar`, `walmart_amazon`,
  `wdc_computers`, `fodors_zagat`, `febrl_person`, `tiny_fixture`, and `opensanctions`
  (**external-only, CC-BY-NC-4.0 — never vendored**, incompatible with Apache-2.0).
  **Every registered entry is `task="linkage"`; `"dedup"` is reserved and unused.** That
  is itself a portfolio finding: we have no registered dedup benchmark.
- **Sizes are already pinned in the loaders**, and two are already known to be odd:
  `[verified]`

  | Benchmark | Corpus | Positives → gold pairs | Note |
  |---|---|---|---|
  | `fodors_zagat` | 533 + 331 | **112 gold pairs** | Tiny. Saturation suspect #1. |
  | `febrl_person` | 500/side | 500, **1:1** | Synthetic; 1:1 by construction. |
  | `abt_buy` | 1081 + 1092 | 1028 of 9575 pairs | |
  | `amazon_google` | 1363 + 3226 | 1167 of 11460 pairs | |
  | `walmart_amazon` | 24628 | 962 → **846 components → 1092 pairs** | Closure *adds* pairs. |
  | `dblp_scholar` | **66879** | 5347 → **2351 clusters, 13763 pairs**; largest component **37** | |

  The `walmart_amazon` and `dblp_scholar` rows are the tell: **962 positives collapse to
  846 components but expand to 1092 gold pairs**, and DBLP-Scholar's largest component is
  **37 records**. Project history already records DBLP-Scholar's **PC 0.39 as a
  many-to-many artifact**. A 37-record component in a bibliographic set is a
  *transitive-closure* artifact of the labeling, not 37 genuinely identical papers — which
  means **B2 and B1 are measuring the same phenomenon from opposite ends**, one in the gold
  labels and one in our output.
- **The saturation concern is the field's, not ours.** DeepMatcher/Magellan (Mudgal et al.,
  SIGMOD 2018) is on file in `docs/research/README.md` as *"the 'benchmarks are trivially
  solvable' motivation"*. Fodors-Zagat at 112 gold pairs is the obvious suspect. `[the
  general concern is verified as the field's stated motivation; per-dataset saturation on
  our portfolio is unmeasured — that is this item]`

**What would settle it.** Run `DataProfileReport.from_benchmark()` across all 10, publish
one table, and add the one missing measurement (vocabulary overlap between the two sources
— for a linkage set, the fraction of shared tokens across A and B, which is the thing a
string comparator actually keys on). Then the two verdicts, each with a stated rule
decided *before* looking:

- **Saturated** := the best cheap $0 method (`rapidfuzz`) already scores within ~0.02 of
  the published SOTA. If a free method ties the literature, the set cannot rank methods.
- **Unrepresentative** := the gold labels have a structural artifact (many-to-many
  closure, 1:1 by construction, degenerate cluster-size distribution) that makes the
  metric measure the labeling rather than the task.

Retire nothing on this basis — **annotate**. A saturated set is still a regression test;
it just cannot be evidence for a method. The deliverable is a `README`-linked table plus
per-benchmark caveats in `docs/BENCHMARKS.md`.

**Cost & prerequisites.** **$0** — CPU only; `rapidfuzz` and the profiler are core deps,
no extras. Prerequisites: none (the loaders and the profiler ship today). The only new
code is the vocabulary-overlap measure and the table script. **Blocked by nothing; blocks
the interpretation of C1–C5** — a ladder measured on saturated sets is a ladder measured
on noise, so this should run **before** any GPU item.

---

## 5. Thread D — prompt tuning's ceiling, and the handoff to fine-tuning

### D1 — Where does DSPy prompt tuning plateau, and what takes over?

**The question.** Two questions, and the second is the one nobody has answered:

1. **Where is the ceiling?** As a function of *what* — model size, task difficulty,
   labeled-pair count? Prompt optimization is not free lift forever; where does it stop?
2. **What is the handoff?** Once prompting plateaus, fine-tuning is the next lever. What
   *carries over* — the optimized instruction? the selected demos? the labeled pairs the
   optimizer scored? Or nothing, and you start clean?

**Why it matters to langres.** This is the project's stated cost ladder
(`docs/ROADMAP.md`; the "LLM-native cost ladder" position): *cheaper = a DSPy
prompt-tuned smaller LLM, then a fine-tuned small LM*. The ladder has **two rungs and no
documented step between them**. langres ships both halves — `DSPyMatcher` (MIPROv2) and
the `fit()`/`finetune()` training surface (`20260714_training_surface_design.md`) — and
they currently do not talk to each other. If the handoff is real, it is a *feature* of
the training surface. If it is not, the ladder is two disconnected rungs and we should
say so.

**What's already known.** Unusually much, and it is **mixed** — which is why this is an
open question and not a build task:

- **The signature is the lever; the compiler was not.** (Verified, ours, M4.) A
  **precision-tuned DSPy signature** moved a cheap model **0.409 → 0.757 F1** and beat an
  *uncompiled frontier* model. But **MIPROv2 compilation did not help** — M4 cut
  distillation on that evidence. The lift came from *task specification*, not from
  optimizer search.
- **Corroborated on other data.** (Ours, `20260701_er_seam_audit.md`.) On OpenSanctions
  Pairs, MIPROv2 lifted only **~1–2 F1**, and in-context examples were
  **neutral-to-negative**.
- **The counter-evidence, and its caveat.** Breunig's ER example — Qwen3-0.6B
  **60.7% → 82%** via MIPROv2 ([dbreunig.com, 2025-06-10](https://www.dbreunig.com/2025/06/10/let-the-model-write-the-prompt.html))
  — is the inspiration for the whole prompt-optimization thread and points the *other*
  way. Two caveats on file, both load-bearing: the figure is **binary-match accuracy with
  unreported class balance** (so it is not comparable to our F1 numbers), and a widely
  circulated *recap* of that talk contained two claims absent from the primary. Cite the
  primary; do not cite the recap. `[partially verified — the 60.7→82 figures are the
  author's own; the class balance is not published]`
- **The obvious reconciliation is a hypothesis, not a finding.** *Hypothesis:* MIPROv2
  helps most where the *prompt* is the bottleneck (a small model that does not yet know
  the task shape — Breunig's Qwen3-0.6B at 60.7%) and little where the signature has
  already supplied the task shape (our M4 case, post-0.757). That would make our result
  and Breunig's consistent rather than contradictory. **Untested.** It predicts the lift
  from compilation should *shrink as the signature improves* — which is directly
  measurable, and is the cheapest part of this item.
- **The measurement trap is known.** (Ours, PR-G.) **F1@0.5 is a dishonest gate for small
  yes/no LMs** — gate on separation/AUC and "beats-base" instead. Also on file: Qwen3's
  thinking mode breaks the yes/no logprob probe. Any ceiling curve reported at a fixed
  0.5 threshold will measure the threshold, not the model.
- **The handoff has partial prior art, pointing at "no".** *Fine-tuning LLMs for EM*
  (Steiner, Peeters & Bizer 2024, [2409.08185](https://arxiv.org/abs/2409.08185)) reports
  **mixed** results for LLM-generated EM training data. Read before designing a
  prompt→finetune bridge that assumes the prompt's demos make good training data.

**What would settle it.** Two measurements, in order — the first is cheap and settles the
reconciliation hypothesis:

1. **The ceiling surface.** Sweep {model size} × {labeled-pair count} × {signature
   quality: naive vs. precision-tuned}, measuring uncompiled vs. MIPROv2-compiled. Report
   **AUC and separation**, never F1@0.5 (see the trap above). The specific prediction to
   falsify: *compilation lift shrinks as signature quality rises*. If it holds, the
   ladder's first rung is "write a good signature", and MIPROv2 is a small-model
   crutch — a genuinely useful, cheap finding.
2. **The handoff, as an ablation with a real control.** Fine-tune a small LM four ways:
   (a) from raw labeled pairs — the control; (b) from pairs + the optimized instruction
   in-context; (c) from the demos MIPROv2 *selected* (its selection = a difficulty
   signal); (d) distilled from the prompt-tuned model's outputs. **(a) is the null
   baseline and the honest default** — the project's own C7 gate exists precisely to stop
   paid work that never beat a null baseline. If none of (b)–(d) beats (a), the answer is
   *"there is no handoff; the labeled pairs are the only asset that transfers"* — which
   is a clean, publishable-to-ourselves negative and would settle the ladder question for
   good.

**Cost & prerequisites.** (1) is **cheap** — CPU/small-GPU, or **$1–5** if run against a
paid model. (2) is **GPU** (a QLoRA-class run on a single consumer card; the project's
3070 is the intended target). Prerequisites: the `fit()`/`finetune()` training surface
(**shipped**, PR #140) and a benchmark with enough labeled pairs to sweep count (**have**
— registry). **Sequencing:** (1) is independent and can run now; (2) depends on (1)
identifying *where* the plateau actually is — fine-tuning from a point that is not the
ceiling measures nothing. Note (2) shares its entire apparatus with **C4** — run them on
the same harness.

---

## 6. Thread E — grounding

Neither item here produces a method. Both exist to stop us being confidently wrong:
E1 checks that *published* numbers hold, E2 checks that we are not re-deriving a solved
problem under a different vocabulary.

### E1 — Reproduction studies: do published ER results hold?

**The question.** Take a published ER result, re-run it, and check it reproduces. Not
"is it a good method" — *is the number real, and does it survive contact with our
harness?*

**Why it matters to langres.** Two distinct payoffs, and the second is the real one:

1. Every method langres claims to compose is a method langres must be able to *reproduce*.
   A seam that cannot reproduce the thing it wraps is a seam in name only.
2. **Reproduction is the cheapest available test of our own instrument.** When a
   replication misses, the prior should not be "the paper is wrong" — it is usually us.
   A miss localizes a bug in our harness that no unit test would catch, because it is a
   *semantic* bug (wrong split, wrong metric, wrong pair set), not a crashing one.

**What's already known.** This is tractable here, and it has already paid off — twice,
in opposite directions:

- **It works.** The Peeters/MatchGPT replication (`docs/research/`, and the merged
  work behind it) hit **99.25% per-pair agreement** on **all 1206 Abt-Buy pairs**, on
  both `gpt-4o-mini` and `gpt-4o`, for **$0.28** — and the authors' archived answers
  reproduce their published F1 exactly. A full replication of a paid-LLM ER result costs
  less than a coffee.
- **It catches things.** `THEORY.md` §12 (erratum 7) records a citation this project
  itself got wrong: a claimed "face clustering collapses to 0.37" was read off a
  **0–100** scale (so: 0.37 **percent** — ≈100× the natural misreading), from a run that
  scored **BCubed F = 66.96**, using **HAC, not transitive closure**. The doc's own
  verdict: *"A citation that fails this badly under checking is worse than none."*
  Erratum 11 records a "33%" NIL rate with **no source at all**.

**What would settle it.** Pick targets where the authors published enough to be checked
(archived predictions ≫ archived weights ≫ a table in a PDF), and report per-pair
agreement, not just the headline metric — agreement is what localizes a harness bug; a
matching F1 can hide two compensating errors. Priority targets, in order of how much
they'd change what we do:

| Target | Why it, specifically | Reproducible from |
|---|---|---|
| **Ditto** (Li et al., VLDB 2020, [2004.00584](https://arxiv.org/abs/2004.00584)) | The number our LLM placement is measured against; project history says we land 0.14–0.21 short of it. Either the gap is real and instructive, or our harness differs. | Code + published splits |
| **AnyMatch** ([2409.04073](https://arxiv.org/abs/2409.04073)) | The data-recipe bet (`20260715_data_prep_architecture.md`) rests on it. A 124M GPT-2 is cheap to re-run. | Code + LODO protocol |
| **Jellyfish** ([2312.01678](https://arxiv.org/abs/2312.01678)) | Caveat already on file: Amazon-Google was **seen** in training, Abt-Buy is zero-shot. A replication that ignores this reproduces a leak. | Weights on HF |

**Cost & prerequisites.** **Cheap.** Peeters-class paid replication ≈ **$0.30–$5**.
Ditto/AnyMatch are **CPU-feasible, GPU-preferred** (single consumer GPU). Prerequisite:
the benchmark registry (`langres/data/registry.py`) already has the loaders — the
binding constraint is *protocol fidelity* (exact split, exact pair set), not compute.
**Blocked by nothing.** This is the item to run first if the goal is confidence rather
than discovery.

---

### E2 — Cross-domain ER: who else has this problem, and under what name?

**The question.** Many fields rediscover entity resolution under their own vocabulary —
record linkage, deduplication, coreference resolution, entity alignment (KG), author
name disambiguation, face clustering, product matching, identity resolution,
bioinformatics sequence matching. Which of their concepts are **genuinely the same
operation**, and which are **superficially similar and transfer badly**? And which of
their techniques transfer *to the others*?

**Why it matters to langres.** langres's central bet — from `docs/ROADMAP.md` §1 and
`THEORY.md` §3 ("one operation, seven names") — is that these are the *same* operation
at different feasible classes. That bet is either langres's reason to exist or its
biggest unforced error, and **it is currently asserted, not surveyed**. Two concrete
consequences:

- If the shared core is real, the feature-bag/architecture seam should accept a
  coreference or KG-alignment workload with **no new abstraction**. That is a falsifiable
  claim about our API, testable without new research.
- If it is not real — if, say, coreference's discourse structure or KG alignment's graph
  structure is load-bearing rather than incidental — then the honest move is to **narrow
  the claim**, not to widen the framework.

**What's already known.** More than the framing above suggests, and `THEORY.md` already
carries the receipts — this item is a survey, and it starts from a non-empty base:

- **The silos are documented, and are closing.** `THEORY.md` §12 erratum 13 retracts its
  own "the literature is siloed" claim as *"True until recently"* — and notes it was
  documented **by** the field itself: **Leone et al., PVLDB 15(8), 2022**. That paper is
  the natural starting point, and its existence means the survey's premise must be
  *"how far did they get"*, not *"nobody has looked"*.
- **The cross-domain concepts are already partly mapped.** §12 erratum 12 records that
  the assignment constraint, dropped by DB/ER toolkits, is **alive in KG entity
  alignment** — where NIL travels under the name **"dangling entities"**. Erratum 10
  distinguishes Fellegi–Sunter abstention (from *uncertainty*) from NIL (asserting
  *non-existence*) — a genuine, non-obvious semantic difference between two things that
  look identical. Erratum 7 is a cross-domain citation (face clustering) that broke on
  contact.
- **The precedent for a generalist exists.** **Unicorn** (Tu et al., SIGMOD 2023,
  OpenReview `388Cge6WPN`) unifies matching tasks in one MoE encoder — but, per §0
  erratum 9, *"Its axis is **data-element type**, not constraint shape"*. So the
  unification question is live and *partly answered by someone else*, on a different
  axis than ours.
- **Prior art per operation is inventoried** in `THEORY.md` §0 (Fellegi–Sunter 1969,
  AJAX, meta-blocking, BFKPT, Swoosh, Dedupalog). That is the *data-integration* column
  of the table this item needs to complete; the other columns (NLP coreference, KG
  alignment, vision clustering, bibliometrics, bioinformatics) are empty.

**What would settle it.** A survey with a **forced comparison table**, one row per field,
whose columns are langres's own operations — carrier, score $\sigma$, selection $\pi$,
feasible class $\mathcal{F}$, merge, and *what plays the role of blocking*. The
discipline that makes it worth doing rather than a reading list:

- Each cell is filled with the field's **own words + citation**, not our paraphrase
  (this is the `THEORY.md` §0 method, and it is what makes §0 survive review).
- Every claimed transfer must name **a technique that moves**, and be marked
  transferred / **not** transferred / untested. "Both do clustering" is not a finding.
- Deliberately hunt **false friends**. The value is concentrated in the *negatives*:
  erratum 10 (FS-abstention ≠ NIL) is worth more than ten confirmed similarities,
  because it is the kind of mistake the framework would otherwise encode in an API.
- Output is a section in this folder plus, where a transfer is real, **one issue per
  transfer** — not a framework change.

**Cost & prerequisites.** **$0** — literature only, no compute. Bounded by reading time,
which is the real cost (this is the largest-effort $0 item on the board). Prerequisite:
none. **Sequencing:** worth doing *before* any framework generalization work, and it is
the natural companion to landing `docs/theory-foundation` — but nothing downstream
blocks on it, so it is a background thread, not a gate.

---

*(items follow)*
