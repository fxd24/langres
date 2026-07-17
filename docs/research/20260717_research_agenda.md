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

## 4. Thread C — the retrieval/matching model ladder

The unifying question: **ER is a two-stage retrieve-then-rank system, and IR has spent a
decade studying exactly that.** Thread C asks which of IR's hard-won results transfer to
ER data, where the "query" is a record and the "corpus" is the other side. C4 and C5 are
the two with the strongest published evidence and the lowest cost; **C4 is the highest-value
item on this board.**

**Portfolio note:** all five are measured against `all-MiniLM-L6-v2` (384d), which is the
single `DEFAULT_EMBEDDING_MODEL` (`src/langres/core/model_ref.py:80`) feeding the method
registry, the `SearchSpace` default, *and* every loader's pinned `*_BLOCKING_K` constant.
Everything we know about our own blocking recall was measured on that one model. That is
rung 0 of the ladder and the incumbent every item here must beat.

### C1 — The embedding size ladder, and the embedder that *is* the matcher

**The question.** Two, nested:

1. How does ER performance scale with embedder size/category? Is there a knee, and where?
2. The sharp one: **can an embedding model BE the matcher** — making the yes/no decision
   directly via a threshold in the vector space, with **no separate matching stage at all**?

**Why it matters to langres.** (2) would collapse the pipeline. `THEORY.md` §6.5 says
*"Blocking and matching are the same formula at different $t$: blocking sets
$t \approx -\infty$, matching sets $t$ high"* — an embedder-as-matcher is that sentence
taken literally: **one score, two thresholds, no second model**. If it works even on the
easy half of the portfolio, the cost story changes completely (no per-pair LLM call), and
the `architectures` seam gets a genuinely new shape. If it fails, we learn *where* the
vector space stops carrying enough signal, which is exactly the boundary C2 exists to
patch.

**What's already known.** The supporting evidence is strong and comes from IR, not ER:

- **ANCE** (Xiong et al., ICLR 2021, [2007.00808](https://arxiv.org/abs/2007.00808)) §6.1
  reports its *retrieval* "nearly matches the accuracy of the cascade IR with
  interaction-based BERT Reranker" at roughly **100× lower cost**. That is the
  embedder-as-matcher thesis, already demonstrated — in IR. `[see §7 reference tier —
  verify the exact wording and cost figure before quoting]`
- **The ER counter-signal is on file.** `20260707_data_prep_hard_case_mining_survey.md`
  §11–12 records **UniBlocker** as a counter-signal, and documents how SOTA ER *divides*
  blocking recall from matching precision — i.e. the field's current answer to (2) is
  "no". A result that contradicts the field's division is exactly what makes this worth
  running, and exactly what should be met with suspicion if it appears.
- **Our own ladder is unmeasured.** The tried set is small and no sweep was ever run:
  `all-MiniLM-L6-v2` (default), `all-MiniLM-L12-v2`, `all-mpnet-base-v2`, and
  `BAAI/bge-base-en-v1.5` (named as an example backbone in
  `architectures/vector_llm_cascade.py:106`). `[verified — read the source]`
- **The instrument exists.** `SeparabilitySection.auc` already measures exactly the
  quantity (2) turns on: how well a single similarity signal separates positives from
  negatives. `DataProfileReport.from_embedder()` and `EmbeddingComparisonSection`
  (`models: list[EmbeddingModelSummary]`, shared histogram edges) exist **to compare
  embedders side by side**. This is close to built. `[verified]`

**What would settle it.** Sweep the size ladder (MiniLM-L6 → L12 → mpnet-base → bge-base →
a large decoder embedder) and report, per benchmark, **separability AUC** — *not* F1 at a
threshold (see D1's measurement trap). Then for (2): derive a threshold on cosine with
`core.calibration.derive_threshold` and score it as a matcher against the LLM matcher on
the same candidate set. The honest framing: **(2) will not win everywhere**, and "it wins
on `fodors_zagat` and loses on `abt_buy`" is the *expected* result and a useful one — it
maps the boundary. Report the AUC-vs-size curve, and where it flattens.

**Cost & prerequisites.** **Cheap → moderate.** CPU-feasible for the small end; **GPU**
for the large decoder embedders. **$0** (no paid API). Prerequisites: **B2** — a ladder
measured on a saturated benchmark measures the benchmark, so B2 must annotate the
portfolio first. Note the macOS gotcha on file: **faiss + torch OpenMP segfault**
(`KMP_DUPLICATE_LIB_OK=TRUE`).

---

### C2 — Embedder + reranker, measured on ER data

**The question.** The classic two-stage IR setup — bi-encoder retrieves, cross-encoder
reranks. What does it actually buy on ER data, over (a) the embedder alone and (b) the
LLM matcher we already run?

**Why it matters to langres.** This is the *control* for the whole thread. langres's
paid architecture (`VectorLLMCascade`) is already an embedder + an expensive reranker —
the reranker just happens to be an LLM. A cross-encoder reranker is the cheap classical
alternative that IR would reach for first, and **we have never measured it**. Without it,
C1's and C3's numbers have no mid-point to sit against: we know the cheap end
(cosine) and the expensive end (LLM), and nothing between.

**What's already known.**

- **It is already wired.** `RerankingVectorIndex` exists at
  `src/langres/core/indexes/reranking_vector_index.py`, alongside `FAISSIndex` and
  `HybridVectorIndex`. This item is largely *measurement*, not construction. `[verified]`
- **The IR result is the ANCE comparison in C1** — the cascade is the *baseline* ANCE
  nearly matches at 100× the cost. So the interesting question is inverted from IR's: not
  "does the cascade win" (it does, in IR) but **"is the cascade's margin worth its cost on
  ER data, given ER's candidate sets are tiny compared to a web corpus?"**
- **Relevant caution:** reranker behavior at large $k$ is exactly what A1 is measuring
  (see *Drowning in Documents*). C2 and A1 share an apparatus and should share a run.

**What would settle it.** Three-way comparison on the same candidate sets: cosine-only vs.
cosine+cross-encoder vs. cosine+LLM. Report quality **and** cost per resolved record —
cost is the axis this item exists to inform, and `SpendMonitor` already captures it.
The decision it changes: whether `VectorLLMCascade` should have a cheaper sibling
architecture.

**Cost & prerequisites.** **Cheap.** CPU-feasible; **$0** for the cross-encoder arm, small
paid spend (**$1–5**) for the LLM arm as the comparison point. Prerequisite: **B2**.
Shares apparatus with **A1** and **C1** — run together.

---

### C3 — One decoder family, both roles: does prompt tuning transfer to *retrieval*?

**The question.** Take a single decoder-transformer family that ships both a generative
model and an embedding model (e.g. Qwen3 + Qwen3-Embedding). Instruct/prompt the decoder
for matching — then the real question: **does the knowledge DSPy prompt-tuning discovers
on the matching side TRANSFER to the embedding side?** I.e. can prompt optimization
improve **recall/retrieval**, not just matching?

**Why it matters to langres.** This is the project's stated blocking idea — *shared Qwen3
weights serving matcher and embedder* — and it is the only item here that would make
**prompt optimization a blocking technique**. Today `langres.optimize` searches blocking
configs (k, model, field) as an opaque grid. If an *instruction* is a blocking parameter,
the search space changes shape entirely. It also directly attacks A2/A1's problem from the
other side: a blocker that can be *told* what the matcher cares about is a blocker
coupled to $\varphi$ by construction, rather than by an outer optimization loop.

**What's already known.**

- **The seam already exists, and this is the non-obvious find.** `VectorBlocker.__init__`
  takes **`query_prompt`**, documented as supporting **instruction-tuned embedders**
  (`src/langres/core/blockers/vector.py`). So "put an instruction on the retrieval side"
  is a **parameter we already have and have never optimized**. `[verified — read the
  signature]`
- **Instruction-following embedders are established**: E5-mistral
  ([2401.00368](https://arxiv.org/abs/2401.00368)) and the instruction/prompt treatment in
  `20260707_data_prep_hard_case_mining_survey.md` §11–12, which also covers **GritLM**
  (joint generative + embedding training — the exact "one family, both roles" architecture)
  and **Qwen3-Embedding**. The survey is the literature backbone here; this item is its
  measurable core.
- **Our prompt-tuning result is signature-shaped** (D1: 0.409 → 0.757 from the signature,
  MIPROv2 flat). *Hypothesis:* if the lift lives in **task specification** rather than in
  demo selection, it is *more* likely to transfer to an instruction-tuned embedder — a
  signature is a task description, and that is precisely what `query_prompt` accepts.
  **Untested, and this is the item's central bet.**
- **Known gotcha:** Qwen3's **thinking mode breaks the yes/no logprob probe** (ours, PR-G).
  A decoder-as-matcher arm must disable thinking or score differently.

**What would settle it.** Two measurements, and the second is the actual question:

1. **Does an instruction help retrieval at all?** Sweep `query_prompt` over a fixed
   embedder and measure **candidate recall @ fixed k**. `langres.optimize`'s `SearchSpace`
   is a frozen Cartesian grid — adding a `query_prompt` axis is a small, honest change.
   *If a hand-written instruction moves recall, the transfer question is live. If not, C3
   stops here* — cheaply.
2. **Does the tuned instruction transfer?** Take the instruction DSPy optimized **for
   matching**, put it in `query_prompt`, and measure recall against (a) no prompt, (b) a
   naive hand-written prompt, (c) an instruction tuned **directly** on a recall objective.
   **(c) is the control that makes this honest** — if tuning directly on recall beats the
   transferred matching instruction, there is no transfer, just prompt tuning working
   normally on a second task. That is the null result this item must be able to return.

**Cost & prerequisites.** **GPU** (same-family decoder + embedder; the 3070 is the target,
which bounds model size hard). **$0–5**. Prerequisites: **D1(1)** — transferring a tuned
instruction requires knowing the tuning worked and where it plateaus; **B2** for
interpretation. This is the most speculative item on the board and it is priced
accordingly: measurement (1) is cheap and gates the rest.

---

### C4 — Does the RocketQA denoising result hold on ER data? ⭐

**The question.** RocketQA showed that hard negatives mined from a retriever's top-k
**actively hurt** unless a cross-encoder first strips the false negatives. **ER blocking
output is dense with unlabeled true matches — the exact condition that triggers this.**
Does the result reproduce on ER data?

**Why it matters to langres.** This is **the highest-value experiment on the board**, for
one reason: **we already ship the failure mode.** `src/langres/bootstrap/miners.py` has
**`HardNegativeMiner`**, which mines hard negatives by stratified sampling over the
blocker's `similarity_score` — **with no denoiser in the loop**. That is RocketQA's
undenoised condition, implemented, shipped, and feeding our training surface. If RocketQA's
result transfers, that miner is not merely suboptimal — it is **actively destructive**, and
every downstream training result that used it is contaminated. This item either
invalidates a shipped component or clears it. Few experiments have that leverage.

**What's already known.** The published numbers are stark, and the direction is the whole
point:

- **RocketQA** (Qu et al., NAACL 2021, [2010.08191](https://arxiv.org/abs/2010.08191)),
  Table 3, MRR@10: in-batch negatives **32.39** → hard negatives **undenoised 26.03**
  (**worse than in-batch** — the finding) → hard negatives **denoised by a cross-encoder
  36.38**. Undenoised hard-negative mining is **−6.36 below doing nothing**; denoising
  turns the same data into **+3.99 above it**. `[see §7 reference tier — the numbers are
  as transcribed from the brief; confirm against Table 3 before publishing]`
- **The mechanism is the ER condition exactly.** Top-k from a blocker is, by design,
  where the true matches are. Sampling "negatives" from it without labels samples
  **unlabeled positives**. `20260707_data_prep_hard_case_mining_survey.md` already carries
  the safety rails: **RocketQA** is filed as *"the EM safety rail"*, **NV-Retriever**
  ([2407.15831](https://arxiv.org/abs/2407.15831)) as positive-aware false-negative removal
  (TopK-PercPos), and **GISTEmbed** ([2402.16829](https://arxiv.org/abs/2402.16829)) as a
  guide model masking false in-batch negatives.
- **A denoiser already exists — but it is not RocketQA's.** `data/mining.py` ships
  **`denoise_pairs(..., method="confident_learning")`** (Northcutt confident learning, with
  per-class thresholds for imbalance). That is a *label-noise* denoiser over a trained
  model's OOF predictions, **not** a cross-encoder screening candidates before they become
  negatives. So we have a denoiser, and it is a **different intervention at a different
  point in the pipeline** — which makes "cross-encoder denoising vs. confident learning vs.
  nothing" a three-arm comparison we can run *today*. `[verified — read the source]`
- **A documented tension, already noticed.** `denoise_pairs`'s own caveat is that it
  *"competes with hard-positive mining, since it can strip the very boundary positives"*.
  Denoising and hard-example mining pull against each other; RocketQA's result is that the
  denoiser must win. Whether that holds when the positives are *also* mined for difficulty
  is genuinely open. `[verified — the caveat is in the source]`

**What would settle it.** Train the same small matcher four ways on the same benchmark:
**(a)** random/in-batch negatives — the null baseline; **(b)** `HardNegativeMiner` output,
undenoised — *the shipped path*; **(c)** the same, denoised by a cross-encoder (RocketQA's
actual intervention); **(d)** the same, denoised by `denoise_pairs` (confident learning —
the intervention we happen to own). Report AUC/separation, not F1@0.5 (D1's trap).

**The prediction that makes this falsifiable:** if RocketQA transfers, **(b) < (a)** —
the shipped miner is worse than random. That is a specific, surprising, checkable claim,
and it is the reason to run this first. **(d) vs. (c)** additionally tells us whether the
denoiser we already own is a substitute for the one the paper used, which is worth knowing
regardless of the headline result.

**Cost & prerequisites.** **GPU**, but small (the AnyMatch-class 124M model is the target;
the 3070 suffices). **$0** — no paid API; a local cross-encoder does the denoising.
Prerequisites: **B2** (which benchmark is non-saturated enough to show a difference), and
the training surface (**shipped**). Shares its entire apparatus with **D1(2)** — one
harness, two questions. **Run this one.**

---

### C5 — Query-side-only training: the ADORE analogue

**The question.** ADORE trains **only the query encoder** against a **frozen** document
index — no index refresh, ever — and beats ANCE. The ER analogue: **retrain only the probe
side against a frozen record index.** Does it work here?

**Why it matters to langres.** Index refresh is the dominant cost of the ANCE-style
training loop: every refresh re-embeds the entire corpus. On `dblp_scholar` (**66,879
records**) that is the difference between a loop we can run on a 3070 and one we cannot.
If the ER analogue holds, **the cheapest strong option on this board becomes available** —
blocker improvement with no re-indexing. It also composes cleanly with the existing seam:
a frozen index is exactly what `optimize()` already caches per
`(embedding_model, metric, text_field)`.

**What's already known.**

- **ADORE** ([2104.08051](https://arxiv.org/abs/2104.08051)): trains only the query encoder
  against a frozen document index and reports **MRR@10 0.347 vs. ANCE's 0.338** and
  **R@100 0.876 vs. 0.862**. `[see §7 reference tier — verify the numbers, the exact
  split, and the "frozen index / query-encoder-only" characterization before publishing;
  the paper's title is about hard negatives, not obviously about ADORE]`
- **ANCE** is the thing it beats, and ANCE's defining cost *is* the index refresh
  ([2007.00808](https://arxiv.org/abs/2007.00808), filed in
  `docs/research/README.md` as *"index-refreshed model-mined negatives"*). The contrast is
  the point: ADORE's win is *also* a cost win.
- **The asymmetry question is ER-specific and unresolved.** IR has a real query/document
  asymmetry — a short query, a long document. **ER's two sides are the same kind of
  object** (a record vs. a record), and for `dedupe()` they are *literally the same set*.
  So "train the query side only" may be **ill-posed for dedup and well-posed only for
  two-source linkage**, where A and B are genuinely different corpora. Note every
  registered benchmark is `task="linkage"` — so the well-posed case is the one we can
  actually test, and the dedup case is untestable on the current portfolio. *This is a
  hypothesis about applicability, not a known limitation.*
- **The measurement is already wired**: `evaluate_blocking_with_ranking` computes
  **MRR, MAP, NDCG@K, recall@K** (via the `[eval]` extra / ranx) — the exact metrics ADORE
  reports. `[verified]`

**What would settle it.** On a two-source linkage benchmark, freeze the B-side index,
train only the A-side encoder against it, and compare **recall@k / MRR** to (a) the frozen
untrained baseline and (b) a full ANCE-style loop with index refresh — measuring **both
quality and wall-clock/compute**. The claim to test is not "it wins" but **"it gets most
of the win for a fraction of the cost"**, which is a different and more useful bar. Then
the ER-specific question the IR literature cannot answer for us: **does the asymmetry
survive when both sides are records?** Run it on `dedupe` (single corpus, self-linkage) and
see whether "query side" still means anything.

**Cost & prerequisites.** **GPU**, moderate — but **cheaper than every alternative in this
thread by construction** (no re-indexing). **$0**. Prerequisites: **B2**; a two-source
benchmark (**have** — all 10 are linkage). Independent of C4; can run in parallel.

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
