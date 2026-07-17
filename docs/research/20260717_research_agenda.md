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
| **What's already known** | Cited, with confidence flagged. Often more than expected — several items shrank on contact. |
| **What would settle it** | The concrete measurement — not "investigate". |
| **Cost & prerequisites** | $0 / CPU / GPU / paid-API, and what must land first. |

**Method discipline** (this project has been burned — see `.claude/rules/expert-knowledge.md`).
Two conventions, and they are load-bearing rather than decorative:

- **`[verified — how]`** marks a claim checked against the thing itself during this doc's
  preparation — the LaTeX source of a paper, an OA PDF, or the langres source at a named
  `file:line`. Anything not so marked is reached via a secondary (usually `THEORY.md` or a
  project note) and is **not** independently confirmed here.
- **§7 splits references into two tiers** — *verified against the primary* vs. *do not
  quote without reading* — following `THEORY.md`'s convention. **An entry does not move up
  a tier without someone reading the primary.**

This is not ceremony. **Four claims taken as given at the start of this doc turned out to
be wrong on contact with the primaries** — ADORE's headline number is a different row
(C5), `min(1, n/k)` is not Eq. 4's prescription (A2), *Drowning in Documents* measures no
false-positive rate (A1), and ANCE's "100×" is latency rather than cost (C1). A fifth —
"is there *any* joint blocking+matching training in ER?" — was posed as an open question
and is a populated field with three papers (A3). The tiering is what caught them.

A hypothesis is labelled a hypothesis. Novelty is **not** a goal — the owner's standard is
*"We don't care how much we are contributing with novelty. We care that what we have is
good and correct."* Several items exist specifically to find out that we are **wrong**,
which is the cheapest possible outcome; **A2 is written to be killed by A1**, and that is
the expected result, not a failure of the item.

---

## 1. The board

Ordered by **run order**, not importance. Cost is the *marginal* cost of the experiment,
assuming the framework is ready.

| # | Item | Cost | Prereq | Verdict |
|---|---|---|---|---|
| **B1** | Closure diagnostic — do output clusters contain pairs we scored *no*? | **$0** | none | **Run first.** One script; every part ships today. Can retire a `THEORY.md` section. |
| **B2** | What *is* the benchmark distribution? Which are saturated/unrepresentative? | **$0** | none | **Run first.** Profiler already computes ~all of it. **Gates the reading of C1–C5.** |
| **E1** | Reproduction studies — do published ER results hold? | **$0.30–5** | none | Cheapest confidence on the board. Tests *our instrument*, not just the paper. |
| **A1** | **Measure $\varphi$** on the portfolio; does it decay or grow with rank? | **$5–20** ($0 arm first) | B2 | The number `THEORY.md` §6.5 says it *"exists to motivate"*. **Blocks A2.** |
| **A3** | What is actually unclaimed in joint blocking+matching? | **$0** | none | **Run before A2** — the cheapest way to kill A2. |
| **D1**(1) | Where does prompt tuning plateau? | **$1–5** | none | Independent. Settles the MIPROv2-vs-signature reconciliation. |
| **C4** | ⭐ **Does RocketQA denoising hold on ER data?** | **GPU, $0** | B2 | **Highest value.** We ship the undenoised failure mode (`HardNegativeMiner`). |
| **C2** | Embedder + reranker, measured | **CPU, $1–5** | B2 | The missing mid-point between cosine and LLM. Shares apparatus with A1. |
| **C1** | Embedding size ladder; the embedder that *is* the matcher | **CPU/GPU, $0** | B2 | Supplies frozen indexes for C5. |
| **C5** | Query-side-only training (ADORE analogue) | **GPU, $0** | B2, C1 | Cheapest *strong* option — but see the ADORE correction; it's a cost argument. |
| **D1**(2) | The prompt→fine-tune handoff | **GPU** | D1(1) | Shares its harness with C4. |
| **C3** | One decoder family, both roles; does prompt tuning transfer to *recall*? | **GPU, $0–5** | D1(1), B2 | Most speculative. Step (1) is cheap and gates the rest. |
| **A2** | The Zamani follow-up — is a defensible contribution there? | **$0 compute; costly in thought** | **A1**, A3 | Most likely outcome: **dropped**, because $\varphi$ is tiny. That is a success. |
| **E2** | Cross-domain ER — who else has this problem? | **$0** | none | Largest $0 item (reading time). Background thread; gates nothing. |

**If only two things get run: B1 and B2.** Both are free, both are one script away, and B2
determines whether any *other* number on this board means anything.

**The three items most likely to change what langres does:** **C4** (may invalidate a
shipped component), **B1** (may change a default), **B2** (may invalidate our measurements).
Note none of these is A2.

---

## 2. Thread A — the staging result

This thread is the experimental half of `docs/THEORY.md`@`7a106ce` §6. That section
derives that the first-stage objective's *form* depends on the second stage's quality,
and then says so plainly about its own status:

> **The honest caveat.** […] If $\varphi$ is tiny, the interior optimum sits at
> $r_B \approx 0.999$ and this section is *true but operationally uninteresting*. **The
> claim has teeth only with a measured $\varphi$.** Measuring $\varphi$ for LLM judges on
> our benchmark portfolio is therefore the experiment this document exists to motivate —
> not a follow-up.

**A1 is that experiment. It is the prerequisite for A2, and A2 is not worth starting
without it.** The order matters: A1 can retire A2 entirely, and that is the most likely
outcome.

### A1 — Measure $\varphi$ on the benchmark portfolio

**The question.** Two, and the second is the one with a surprise in it:

1. **What is $\varphi$** — the matcher's fall-out, $P(\text{yes} \mid \text{false})$
   (`THEORY.md` §1) — for our actual matchers on our actual benchmarks?
2. **Does $\varphi$ decay or GROW with candidate rank?**

**Why it matters to langres.** §6.5's consequence is concrete: *"The blocker's optimum is
a function of $\varphi$, a property of the **matcher**. **The blocker cannot be tuned in
isolation.**"* langres currently violates this in code, and §11 lists where —
`core/blocker.py:38` (*"Blocking should have ≥95% recall"*), `core/blockers/vector.py:202`,
and `RecallCurve.optimal_k(target_recall=…)`, which *"names a recall-hitting $k$ 'optimal'"*
while optimizing a quantity **containing no $\varphi$**. A measured $\varphi$ either
justifies that guidance or condemns it. Either way it is a number we do not have and
should.

Question (2) is the sharper one, because **the two candidate answers point in opposite
directions and one of them contradicts our own theory doc**:

- If $\varphi$ **decays** with rank, that **confirms** §6.4's stated self-critique — the
  doc already says *"the marginal candidates admitted at large $k$ are the low-similarity
  ones the matcher rejects most easily, i.e. $\varphi$ decays with rank"*, and flags it as
  *"the first objection any reviewer will raise."*
- If $\varphi$ **grows** with rank, §6.4's stated weakness is **wrong in our favour**, and
  the staging result gets *stronger*: precision collapses faster than the constant-$\varphi$
  model predicts, and the interior optimum sits at a **smaller $k$** than §6.2 says.

**What's already known.**

- **§6.4 currently claims decay**, as a conceded weakness of the model: *"Constant
  $\varphi$ is wrong and load-bearing. Blocker and matcher errors are **correlated** —
  both keyed on similarity […] $P \to 0$ survives iff $\inf \varphi > 0$, but
  constant-$\varphi$ **mis-locates the optimum**."* `[verified — quoted from 7a106ce]`
- **The counter-evidence is weaker than it first appears — and this correction matters.**
  *Drowning in Documents: Consequences of Scaling Reranker Inference* (Jacob, Lindgren,
  Zaharia, Carbin, Khattab & Drozdov, **ReNeuIR@SIGIR 2025 workshop**,
  [2411.11767](https://arxiv.org/abs/2411.11767)) is often read as showing $\varphi$ grows
  with $k$. **It does not show that.** `[verified — LaTeX source]` What it actually
  establishes:
  - **Quantitative:** *"scaling K leads to a substantial decrease in recall"* — Recall@10
    is *lower than retrieval alone* in **53.3%** (academic) and **44.4%** (enterprise) of
    experiments. But degradation is **not monotone**: *"rerankers provide initial
    improvements when scoring progressively more documents, but their effectiveness
    gradually declines and can even degrade quality beyond a certain limit"* — quality
    rises to a peak, *then* falls.
  - **Qualitative only:** *"phantom hits"* is their coined term for *"completely irrelevant
    documents […] scored very highly"*. But the paper **never measures a false-positive
    rate** and never plots any error rate against $k$; the analysis is explicitly
    **outcome-selected** (*"we filtered for queries where the Recall@10 decreased"*).
    **Examples chosen because they failed cannot establish a rate.**
  - The "each added document is another chance at a high-scoring irrelevant one" mechanism
    appears in their Discussion but is **cited to Zamani et al. 2022** — borrowed framing,
    not their measurement.

  > **So: nobody has measured $\varphi(\text{rank})$.** The "phantom hits" story is
  > *suggestive* of growth and is the best available reason to think §6.4's decay
  > assumption may be wrong — but it is a hypothesis, not a result. That is precisely why
  > A1 is worth running rather than citing.
- **A corroborating signal is already in `THEORY.md` §6.4**: Meng et al., *Ranked List
  Truncation for LLM-based Re-Ranking* (SIGIR 2024, [2404.18185](https://arxiv.org/abs/2404.18185))
  — *"a deeper re-ranking cut-off does not consistently result in improvement and can even
  be detrimental to re-ranking quality."* `[verified against the primary — THEORY.md's
  ellipsis drops "to re-ranking quality"; harmless]`
- **The field's own measurement says $\varphi$ is small**, which is the reason §6.5 hedges:
  Papadakis et al. — *"precision […] **significantly raises after matching**."* If that is
  right, A1 returns "$\varphi \approx 0$", §6 is operationally uninteresting, and **A2
  should be dropped**. This is the single most likely outcome and the thread is ordered to
  find it out first and cheaply.
- **The composition/degradation split is already stated** (§6.4): the *composition* effect
  needs no degradation at all (true pairs are $O(n)$, all pairs $O(n^2)$), while the
  *degradation* effect — $\varphi$ growing with group size — *"does not apply to a pairwise
  matcher; **does** apply to set-wise matchers."* We ship one: **`SelectMatcher`**
  (`core/matchers/select_judge.py`, ComEM-style, one call per anchor *group*). So the two
  effects are **separately measurable with components we already have**. `[verified]`

**What would settle it.** $\varphi$ is measurable today, but **not with an off-the-shelf
call, and the obvious way to compute it is wrong**. The inventory is explicit that langres
has **no false-positive-*rate* metric**: `core/metrics.py` computes FP *counts*
(`PairMetrics.fp`, `classify_pairs`) and never counts true negatives. That omission is
*correct* for ER in general — the negative class is $O(n^2)$, so a global FPR is
meaningless — but $\varphi$ needs no global denominator. **Within a candidate set $E_B$ the
false pairs are finite and countable**, and that count is exactly §6.2's $F_B$.

> **The trap, found by reading the source.** The natural formula —
> `fp / (total_candidates - (tp + fn))` — is **wrong**. `classify_pairs`'s docstring is
> explicit that *"`fn` counts gold pairs that were not predicted — **covering pairs the
> blocker never surfaced**, pairs scored below `threshold`, and abstentions"*. So
> $\texttt{tp} + \texttt{fn} = |{\rm gold}|$, the **whole** gold set — not the gold pairs
> the matcher actually saw. That behaviour is right for `PairMetrics`'s own purpose
> (end-to-end pair quality) and wrong for $\varphi$, which is **conditional on the matcher
> having been shown the pair**. Using it would fold the *blocker's* misses into the
> *matcher's* fall-out — i.e. measure the wrong stage, in a thread whose entire point is
> that the two stages must be measured separately. `[verified — read
> `core/metrics.py:281-327`]`

**The fix is a one-line discipline: restrict gold to the candidate set before classifying.**
Let $E_B$ be the candidate pairs and $T_B = |{\rm gold} \cap E_B|$ (THEORY.md's $T_B = r_B T$).
Call `classify_pairs(judgements, gold_pairs & candidate_pairs, threshold)`. Then, from
functions that ship today, **both** of §6's matcher parameters fall out:

$$\eta \;=\; \texttt{PairMetrics.recall} \qquad\qquad \varphi \;=\; \frac{\texttt{fp}}{|E_B| - T_B}$$

$\eta$ is `recall` directly (with gold restricted, $\texttt{tp}+\texttt{fn} = T_B$, so
$\texttt{recall} = \texttt{tp}/T_B = P(\text{yes} \mid \text{true})$ — the definition).
$|E_B|$ is `total_candidates` from `evaluate_blocking`, and $T_B$ is recoverable as
`candidate_recall × |gold|` from the same call. **Three further caveats that decide whether
the number is honest:**

- `PairMetrics` **excludes abstentions** from the predicted set. An abstain is not a
  "yes", so excluding them is right for $\varphi$ — but it must be *reported*, because a
  matcher that abstains often has a flattering $\varphi$ that means something different.
- $\varphi$ must be measured **pre-clustering**. `PairMetrics` exists precisely for this —
  its docstring: *"transitive closure can chain one false-positive edge into many
  false-positive pairs."* Measuring $\varphi$ post-closure measures the clusterer (that is
  B1's job).
- $\varphi$ is a function of the **threshold**, which is the precision/recall dial (§6.5).
  Report $\varphi(t)$ as a curve, not a scalar; a single number silently smuggles in a
  threshold choice.

For (2): bin candidates **by blocker rank** and report $\varphi$ per bin — that curve *is*
the answer, and its **slope sign** is the finding. Run it for a pairwise matcher
(composition effect only) and for `SelectMatcher` at varying group size (degradation effect
too); §6.4 predicts the slopes differ, which is itself a check on the model.

**Cost & prerequisites.** **Moderate, paid.** $\varphi$ for *LLM* matchers is the number
§6.5 asks for, and that means real API spend across the portfolio — estimate **$5–20**
under `SpendMonitor`/`SpendCappedMatcher` (the cap is enforced in one place, `core/spend_cap.py`).
The **$0 arm runs first**: `rapidfuzz` and `embedding_cosine` cost nothing and will show
whether the *shape* of $\varphi(\text{rank})$ is even interesting before any paid call.
Prerequisites: **B2** (to know which benchmarks can show a difference). Shares apparatus
with **C2**. **Blocks A2.**

---

### A2 — The Zamani follow-up: is there a defensible contribution here?

**The question.** `THEORY.md` §6 transplants a published IR result into ER. There is a
structural reason langres's version might be **stronger than its source**. What would it
take to make that a real, defensible contribution — what would need to be **derived**, and
what **measured**? And is it worth doing at all?

**Why it matters to langres.** Mostly it *doesn't* — and that framing is deliberate. The
owner's standard: *"We don't care how much we are contributing with novelty. We care that
what we have is good and correct."* This item earns its place for a different reason:
**working out whether the claim is defensible is the same work as working out whether it is
correct.** The pressure-test is the deliverable; a paper would be a by-product. If A1
returns $\varphi \approx 0$, **this item is dropped and that is a success.**

**What's already known.** The source has now been read in full (authors' own OA copy), and
it both **supports the gap and corrects our description of it**:

- **Eq. 4 is k=1-only — CONFIRMED, stated explicitly twice.** `[verified — full text]`
  Before the equation: *"For illustration, **let us assume 𝑘 = 1**, meaning that we care
  solely about the relevance of the first retrieved document (e.g., Precision@1)."* And in
  §5: *"**We formally derive this connection in Equation (4) for metrics when the ranking
  cutoff 𝑘 = 1.** To complement our theoretical derivation, we demonstrate the impact of 𝜌
  on metrics for deeper ranked lists **through a number of simulations**."*

  > **This is the gap, and it is real.** Deeper $k$ is **simulated, not derived** (their
  > simulation fixes $\varepsilon^+ = \varepsilon^- = 0.05$, $N=2000$, $n=50$, recall
  > $0.5$). So a derivation at general $k$ is genuinely not in the source.
- **⚠ But our description of Eq. 4 is wrong, and `THEORY.md` inherits the error.**
  `[verified — full text]` The paper contains **two** *"instead of Recall@N (i.e., the
  current popular belief)"* sentences with **different prescriptions**:
  - **Optimal reranker, cutoff $k$** (this *precedes* Eq. 4): *"the retrieval model 𝜙
    should **maximize min(1, 𝑛/𝑘)**."*
  - **Sub-optimal reranker, k=1 — this is Eq. 4's actual takeaway:** *"the retrieval model
    𝜙 should **minimize 𝜌 = 𝑁/𝑛**… Note that 𝜌 is equal to the inverse of precision."*

  `THEORY.md` §6's attribution box quotes the **min(1, n/k)** sentence immediately after
  asserting *"whose Eq. 4 — multiplied through by $n$ — **is** the precision formula
  below"*, which reads as attributing min(1,n/k) to Eq. 4. **It belongs to the
  optimal-reranker analysis instead.** Two things follow, and the second is the more
  interesting: `THEORY.md` needs a precision fix; and since Eq. 4 is k=1-only, its
  unqualified *"**is**"* **over-credits Zamani** — the correction runs in *both*
  directions.
- **The load-bearing assumption is confirmed, and it is asserted rather than argued.**
  `[verified — full text]` Zamani: *"In this derivation, **𝜖+ and 𝜖− solely depend on the
  reranker's quality**… On the other hand, **𝜌 solely depends on the retrieval quality by
  𝜙**."* That clean factorization — error from the reranker, $\rho$ from the retriever — is
  what makes the whole result separate. It enters via an **averaging step** that
  homogenizes per-document noise into two constants ($\sum(1-\varepsilon_i) \to
  n(1-\varepsilon^+)$). The paper never argues that a reranker's average noise is invariant
  to *which* documents the retriever hands it. **So constant-$\varphi$ is a modelling
  assumption in Zamani too** — §6.4's conceded weakness is inherited, not introduced.

**The structural claim, stated precisely.** At $k=1$ the objective is pure precision, and
Eq. 4 is monotone decreasing in $\rho$ for fixed $\varepsilon$ — so
$\arg\min_N \rho(N)$ is **$\varepsilon$-invariant**: the retriever's optimum does not move
as the reranker changes. That is exactly why Zamani's prescription (*"minimize $\rho$"*)
contains no $\varepsilon$. langres's objective has no such property:

$$F_1(k) = \frac{2\,\eta\,r_B(k)\,T}{\eta\,r_B(k)\,T + \varphi\,(nk - r_B(k)\,T) + T}$$

Here $\varphi$ multiplies a $nk$ term that trades **against** the recall gain, so
$\partial k^* / \partial \varphi \neq 0$ in general — **$k^*$ shifts continuously with
$\varphi$**. *That* is the difference: Zamani's k=1 objective **factorizes** (retriever
optimum free of reranker quality); ER's F1 objective at general $k$ **does not**.

> **Hypothesis, not result.** The above is a sketch, not a derivation. It has not been
> worked through, the conditions under which $\partial k^*/\partial\varphi \neq 0$ are not
> stated, and the F1 expression itself is `THEORY.md`'s modelling choice. Treat it as the
> thing to be established, not as an established thing.

**What would settle it.** Four pieces, ordered, each able to kill the item:

1. **Derive $\partial k^*/\partial \varphi$ at general $k$, with conditions.** State when it
   is nonzero and when it vanishes. The sharpest form of the claim is a *reduction*: show
   that at $k=1$ the objective collapses to Zamani's and the shift **disappears** — that
   names precisely what is new and precisely what is theirs.
2. **Do not assume constant $\varphi$ — carry $\varphi(\text{rank})$ through.** This is the
   objection §6.4 pre-registers as *"the first objection any reviewer will raise"*, and
   Zamani is exposed to it too. A derivation that merely inherits the assumption adds
   nothing; one that carries a *measured* $\varphi(\text{rank})$ (from **A1**) through the
   optimization is a real advance over the source — and is the strongest version of this
   item.
3. **Measure the prediction.** This is the falsification, and the reason the item is not
   pure theory: **predict $k^*$ from measured $\varphi$, then sweep $k$ and find the
   empirical $\arg\max F_1$.** If they agree, the model has predictive power. If they
   disagree, the model is wrong and A2 dies with a useful negative.
4. **Situate it against the nearest work honestly** (see A3) — RLT and the joint-training
   line, not a claim of open ground.

**The honest risks — any one of these sinks it:**

- **$\varphi$ may be tiny.** §6.5's own caveat: then $k^* \approx$ saturation and the whole
  section is *"true but operationally uninteresting."* **Most likely outcome.**
- **Constant $\varphi$ is load-bearing and known-wrong** — in Zamani too. Piece (2) is
  therefore not optional polish; it is the item.
- **The empirical support for growth is thin.** *Drowning in Documents* does **not**
  measure a false-positive rate (see A1). The strongest honest statement today is *"nobody
  has measured $\varphi(\text{rank})$"*.
- **F1 is itself arbitrary** — §6.5, via Elkan (IJCAI 2001): *"Choosing $F_1$ and then
  finding an interior max is partly an artifact of the metric."* A result that only holds
  for F1 is a result about F1.
- **The reception risk is known and named** in `THEORY.md` §0: *"Expect exactly two
  objections: 'that's meta-blocking' and 'that's BFKPT.'"* Add a third: *"that's Ranked
  List Truncation"* (A3).
- **ER already has an interior optimum.** §6.3: $F_{PC,RR}$ *"**does** have an interior
  optimum in $k$."* The surviving distinction is narrow — RR is a *comparison-count*
  metric, so ER's trade-off is quality-vs-**cost**, ours quality-vs-**quality**. That
  distinction is real but subtle, and it is the whole contribution.

**Cost & prerequisites.** **$0 in compute; expensive in the only currency that matters
here — careful thought.** Piece (3) needs the $k$-sweep, and note the gap the inventory
found: **`optimize()` has only a *blocking* scorer wired** (`candidate_recall`,
`candidate_precision`, `reduction_ratio`, `total_candidates`) — **no end-to-end
match/cluster-quality scorer**. Sweeping $k$ against downstream F1 therefore needs that
scorer written first. The `Objective` machinery already supports it (metric-agnostic,
Pareto + constraints), so this is a scorer, not an architecture. **Hard prerequisite: A1.**
Do not start A2 before A1 reports.

---

### A3 — Joint blocking + matching: what is actually unclaimed?

**The question.** Is there *any* joint blocking+matching training in ER? And is the
**bilevel** framing of it genuinely unoccupied?

**Why it matters to langres.** A2's entire value rests on the size of the gap it claims. If
the gap is smaller than we think, A2 shrinks or dies — and it is far cheaper to find that
out from a literature search than from a reviewer. This item is A2's due diligence.

**What's already known.** ⚠ **This was posed as an open question. It is not one.** The
search returned a populated field, and this is the single most important correction on the
board:

- **Joint blocker–matcher training in ER is an established line with at least three
  papers:** `[verified — abstracts read]`

  | Work | Cite | What it does |
  |---|---|---|
  | **MutualER** | Dou, Shen, Zhou, Bai, Kou, Nie, Cui & Yu, **CIKM 2024**, pp. 508–518, [10.1145/3627673.3679843](https://doi.org/10.1145/3627673.3679843) | *"**integrates and jointly trains the blocker and matcher**, balancing both the consensus and discrepancy between them"* — via Mutual Sample Selection + Similarity Knowledge Transferring. Siamese PLM blocker + cross-encoder/LLM matcher. |
  | **CLER** | Wu, Wu, Dong, Hua & Zhou, **PVLDB 17(3):292–304, 2023**, [10.14778/3632093.3632096](https://doi.org/10.14778/3632093.3632096) | *"an **end-to-end iterative Co-learning framework for ER, aimed at jointly training the blocker and the matcher**"* via iteratively updated pseudo-labels. Code: `wusw14/CLER`. **Currently uncited in our docs.** |
  | **DIAL** | Jain, Sarawagi & Sen, **PVLDB 2022**, [2104.03986](https://arxiv.org/abs/2104.03986) | *"**jointly learns embeddings to maximize recall for blocking and accuracy for matching** blocked pairs"* — joint training *with* stage-specific objectives. (Note: **accuracy**, not precision.) |

- **The IR analogue is also well-established:** **RocketQAv2** (Ren et al., EMNLP 2021,
  [2110.07367](https://arxiv.org/abs/2110.07367)) jointly trains retriever + reranker via
  *"dynamic listwise distillation"*; **AR2** (Zhang et al., ICLR 2022,
  [2110.03611](https://arxiv.org/abs/2110.03611)) optimizes them *"according to a minimax
  adversarial objective"* with the ranker *"providing progressive direct feedback to the
  dual-encoder retriever"* — i.e. training a first-stage retriever with the reranker's
  objective in the loop. `[verified]`
- **"Choose the cutoff for the downstream reranker" is its own IR subfield —
  Ranked List Truncation (RLT):** Meng et al., SIGIR 2024
  ([2404.18185](https://arxiv.org/abs/2404.18185)) study RLT *"from a novel
  'retrieve-then-re-rank' perspective… investigating the impact of different types of
  re-rankers on RLT methods"*; also *Choppy* ([2004.13012](https://arxiv.org/abs/2004.13012),
  SIGIR 2020) and *Learning to Truncate Ranked Lists*
  ([2102.12793](https://arxiv.org/abs/2102.12793)). **This is the third objection A2 must
  answer**, and `THEORY.md` cites Meng et al. already — without noting it is a whole
  subfield. `[verified]`
- **The bilevel negative: real as literally stated, but do NOT over-read it.**
  `[verified — searches re-run July 2026 with positive controls]` All five exact-phrase
  pairs return **0**: `bilevel` × {`reranker`, `entity resolution`, `entity matching`,
  `candidate generation`, `dense retrieval`}. Positive controls confirm the syntax works
  (`bilevel`=1396; `dense retrieval`=790; `bilevel` AND `neural network`=131). **But three
  caveats block the inference:**
  1. **`bilevel` AND `retrieval` = 19, not zero.** All 19 were read: none train a retriever
     via bilevel optimization (nearest: *PR-Attack*, [2504.07717](https://arxiv.org/abs/2504.07717),
     bilevel to **attack** RAG; *Meta-Wrapper*, [2206.14647](https://arxiv.org/abs/2206.14647),
     CTR feature selection). **The substantive negative survives — but via reading, not via
     the zeros.**
  2. arXiv `all:` covers **title/abstract/comments only, not full text**.
  3. **arXiv fuzzy-tokenizes unknown terms**, so an exact-phrase zero is not proof of
     nonexistence — **MutualER proves it**: a DBLP title search returns *zero* hits for
     "MutualER" because the name appears only *inside* the paper.

  > **Required phrasing.** *"No arXiv title/abstract combines 'bilevel' with dense
  > retrieval / reranking / entity matching (searched July 2026); the 19 bilevel×retrieval
  > hits apply bilevel to adversarial RAG or CTR, not to retriever training."* **Never**
  > *"zero hits, therefore unexplored."*

**What survives.** `THEORY.md` §6.3 is **already honest** — it says *"do not strawman the
field"*, already cites MutualER as nearest work, and claims only that *"No ER source we
found derives the blocking operating point from the matcher's error rate."* **That narrow
claim survives the search. Keep it; do not widen it.** The genuinely unoccupied ground is
correspondingly narrow: **not** joint blocker/matcher training (MutualER, CLER, DIAL);
**not** cut-off selection for a downstream reranker (RLT); but the **ER instantiation of a
matcher-error-*derived* blocking operating point**, and any **bilevel formulation** of it in
either field. *(RLT mostly predicts the cutoff from score distributions, and Meng et al.
study reranker impact **empirically** rather than deriving the cutoff from a measured error
rate — so the distinction is real, but it is a distinction, not a chasm.)*

**What would settle it.** Read the three ER papers properly (MutualER is closed-access,
CIKM 2024; CLER has a PDF and code), and answer one question: **does any of them
*derive* the blocker's operating point from the matcher's error, or do they co-train and
let the operating point fall out implicitly?** If any derives it, **A2 is dead** and we
have found the right citation — a cheap, excellent outcome. Then add CLER to
`docs/research/README.md`, and add RLT to `THEORY.md` §6.3's "nearest work".

**Cost & prerequisites.** **$0** — literature only; ~a day. CIKM 2024 access needed for
MutualER. **No prerequisites. Run this before A2** — it is the cheapest way to kill A2, and
killing A2 cheaply is a win.

> **Method note, worth keeping.** This search produced **two confident false negatives**
> before it produced a result: DBLP's title index said "MutualER doesn't exist"; arXiv's
> bigram search said "bilevel×retrieval is empty". Both broke the same way — **an index
> that does not cover the field being queried**. Positive controls and reading the primary
> caught both. This is the `[a summary is not the source]` lesson in a new costume: *a
> search index is not the literature*.

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

- **ANCE** (Xiong et al., ICLR 2021, [2007.00808](https://arxiv.org/abs/2007.00808)) §6.1,
  verbatim: *"ANCE retrieval nearly matches the accuracy of the cascade IR with
  interaction-based BERT Reranker."* That is the embedder-as-matcher thesis, already
  demonstrated — in IR. `[verified — LaTeX source; §6.1 confirmed]`
- **…but the "100×" needs three corrections before we repeat it.** `[verified — literal
  table cells]` It is **latency, not cost** — the paper says *"100x speed up compared to
  BERT Rerank"* and never mentions money. It is **99×** against the rerank step alone
  (BERT Rerank 1.15s ÷ Dense Retrieval Total 11.6ms) or **122×** against the full sparse
  pipeline (1.42s) — quote one and say which. And it **excludes the 10h offline corpus
  encoding**, a real cost the cascade never pays. For langres the honest version is:
  *the embedder amortizes a large offline cost into a ~100× cheaper online step* — which
  is exactly the trade `optimize()`'s per-`(model, metric, field)` index cache already
  makes, and is why the ER analogue is worth measuring at all.
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
  **Table 3** (`tbl-ablation`), MS MARCO passage, MRR@10 — **all three numbers confirmed
  against the literal table cells, and the table number is right**: `[verified — LaTeX
  source]`

  | Strategy | MRR@10 |
  |---|---|
  | In-batch negatives | **32.39** |
  | Cross-batch negatives | 33.32 |
  | **Hard negatives w/o denoising** | **26.03** ← *worse than doing nothing* |
  | **Hard negatives w/ denoising** | **36.38** |
  | + data augmentation | 37.02 |

  Undenoised hard-negative mining lands **−6.36 below the in-batch baseline**; the *same
  data*, denoised, lands **+3.99 above it**. The paper states the direction outright:
  *"the performance of the retriever **significantly decreases** by introducing hard
  negatives without denoising."*
- **The mechanism is the ER condition exactly — and they quantified it.** RocketQA, on the
  top-retrieved passages they sampled negatives from: *"We find that **about 70% of them
  are actually positives or highly relevant.** Hence, it is likely to bring noise if we
  simply sample hard negatives from the top-retrieved passages by the dense retriever,
  **which is a widely adopted strategy**… As a comparison, we propose denoised hard
  negatives by a powerful cross-encoder."* `[verified — LaTeX source]`

  > **That 70% is the whole argument for this item.** A blocker's top-k is, by
  > construction, where the true matches are — ER's candidate sets are *more* positive-dense
  > than a web corpus's top-k, not less. Sampling "negatives" from it without labels samples
  > **unlabeled positives**, and calls them negatives.
- **The safety rails are already on file** in `20260707_data_prep_hard_case_mining_survey.md`:
  **RocketQA** is filed as *"the EM safety rail"*, **NV-Retriever**
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
index — no index refresh, ever. The ER analogue: **retrain only the probe side against a
frozen record index.** Does it work here?

**Why it matters to langres.** Index refresh is the dominant cost of the ANCE-style
training loop: every refresh re-embeds the entire corpus. On `dblp_scholar` (**66,879
records**) that is the difference between a loop we can run on a 3070 and one we cannot.
If the ER analogue holds, blocker improvement becomes available **with no re-indexing** —
and it composes cleanly with the existing seam, since a frozen index is exactly what
`optimize()` already caches per `(embedding_model, metric, text_field)`.

**What's already known.** ⚠ **The headline framing of this item was wrong, and checking it
changed the item.** The correction is the most useful thing on this card:

- **ADORE** — Zhan, Mao, Liu, Guo, Zhang & Ma, *"Optimizing Dense Retrieval Model Training
  with Hard Negatives"*, **SIGIR 2021**, [2104.08051](https://arxiv.org/abs/2104.08051).
  The query-side-only, frozen-index characterization is **confirmed verbatim**:
  *"Before training, ADORE pre-computes the document embeddings with a pre-trained document
  encoder and builds the document index. **They are fixed throughout the entire training
  process.**"* and *"STAR optimizes both the query encoder and the document encoder while
  **ADORE only optimizes the query encoder**."* `[verified — LaTeX source]`
- **But "ADORE beats ANCE 0.347 vs 0.338" is not a fact about ADORE.** Table 2 (TREC 2019
  DL, MARCO Dev Passage) has **no standalone ADORE row** — ADORE is query-side only, so it
  *structurally must* be paired with someone else's document encoder. The real table:
  `[verified — literal table cells]`

  | Model | MRR@10 | R@100 |
  |---|---|---|
  | ANCE | 0.338 | 0.862 |
  | STAR (theirs, alone) | 0.340 | 0.867 |
  | ADORE + In-Batch Neg | **0.316** | 0.860 |
  | ADORE + Rand Neg | **0.326** | 0.865 |
  | ADORE + BM25 Neg | **0.329** | 0.846 |
  | ADORE + ANCE | 0.341 | 0.866 |
  | **ADORE + STAR** | **0.347** | **0.876** |

  **Three of five ADORE variants lose to ANCE.** The 0.347 row is `ADORE+STAR`, and STAR is
  *the same paper's other contribution* — so that row is "both our things vs. ANCE", not
  "ADORE vs. ANCE". The apples-to-apples line, holding the document encoder fixed at
  ANCE's, is **ADORE+ANCE 0.341 vs. ANCE 0.338 — a gain of +0.003**. And **STAR alone
  (0.340) already beats ANCE with no ADORE at all.**
- **What this does to the item.** It survives, but the claim shrinks from *"beats ANCE"* to
  **"gets within noise of ANCE while never refreshing the index"** — a *cost* argument, not
  a quality one. That is still worth testing (index refresh is our binding constraint), but
  it is a much weaker prior, and **ADORE's benefit is evidently conditional on the frozen
  encoder being good** (0.316 with in-batch negatives vs. 0.347 with STAR — a 0.031 spread
  driven entirely by *whose* index it trains against). For ER that is the real lesson:
  query-side-only training inherits its ceiling from the frozen index.
- **ANCE** is the comparison, and its defining cost *is* the index refresh
  ([2007.00808](https://arxiv.org/abs/2007.00808), filed in `docs/research/README.md` as
  *"index-refreshed model-mined negatives"*).
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

**What would settle it.** On a two-source linkage benchmark, freeze the B-side index, train
only the A-side encoder against it, and compare **recall@k / MRR** to (a) the frozen
untrained baseline and (b) a full ANCE-style loop with index refresh — measuring **both
quality and wall-clock/compute**. Given the correction above, the bar is explicitly
**"most of the quality for a fraction of the cost"**, not "it wins"; a result inside noise
on quality but 5× cheaper is a *success* for this item, and should be pre-registered as
such so it is not retro-fitted into a win. Two further asks the table above makes obvious:

- **Vary the frozen index deliberately.** ADORE's spread (0.316 → 0.347) is driven entirely
  by *whose* index it trains against. So the ER question is not "does query-side training
  work" but **"how good must the frozen index be before query-side training pays"** — which
  is directly measurable by freezing indexes of different quality (C1's ladder supplies
  them).
- **Then the ER-specific question the IR literature cannot answer for us: does the
  asymmetry survive when both sides are records?** IR has a real query/document
  asymmetry — short query, long document. Run it on `dedupe` (single corpus, self-linkage)
  and see whether "query side" still means anything.

**Cost & prerequisites.** **GPU**, moderate — but **cheaper than every alternative in this
thread by construction** (no re-indexing). **$0**. Prerequisites: **B2**; a two-source
benchmark (**have** — all 10 are linkage); **C1** to supply frozen indexes of varying
quality. Independent of C4; can run in parallel.

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

- **It works, and we have done it.** The **Peeters, Steiner & Bizer** replication —
  *Entity Matching using Large Language Models*, [2310.11244](https://arxiv.org/abs/2310.11244)
  v4, **EDBT 2025**; repo `wbsg-uni-mannheim/MatchGPT` — hit **99.25% per-pair agreement**
  on **all 1206 Abt-Buy pairs**, on both `gpt-4o-mini` and `gpt-4o`, for **$0.28**, and
  their archived answers reproduce their published F1 exactly. A full replication of a
  paid-LLM ER result costs **less than a coffee**. Ours lives at
  `src/langres/data/peeters.py` (the **$0, offline** half — it replays their archived
  answers, no API calls), walked through in `docs/BENCHMARKS.md`; the figures are in
  `docs/CHANGELOG.md`. `[verified — the numbers are recorded in-repo; the module docstring
  and BENCHMARKS.md corroborate the 1206-pair slice]`
- **…and it shows what makes a target tractable.** Their eval set is a *deterministic
  subset* of a `test.csv` we already ship (a fixed `sample(random_state=42)`), so the pair
  set is reconstructible exactly rather than approximately. **That property — not the
  paper's fame — is what made the replication cheap and its 99.25% meaningful.** Prefer
  targets with it. (Caveat on file: MatchGPT ships **no LICENSE**, so do not vendor it —
  regenerate from our own CSVs, which is what `peeters.py` does.)
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

## 7. References

Two tiers, following `docs/THEORY.md`'s convention. **Do not move an entry up a tier
without reading the primary.** Several claims on this board were *wrong until checked* —
the tiering is the mechanism that caught them.

**Verified against primary sources** — read directly during this doc's preparation
(LaTeX from arXiv `/e-print`, or the authors' own OA PDF), quoted from literal table cells
and body text, not from summaries:

- Zamani, Bendersky, Metzler, Zhuang, Wang. *Stochastic Retrieval-Conditioned Reranking.*
  ICTIR 2022. DOI [10.1145/3539813.3545141](https://doi.org/10.1145/3539813.3545141).
  **(A2's source. Eq. 4 is k=1-only — stated twice; deeper $k$ is simulated, not derived.
  ε⁺/ε⁻ explicitly assumed retriever-independent. Gold OA; the ACM DL page 403s but the
  authors host a copy via research.google. Note: Semantic Scholar's author list for this
  DOI is wrong — it omits three authors.)**
- Xiong, Xiong, Li, Tang, Liu, Bennett, Ahmed, Overwijk. *Approximate Nearest Neighbor
  Negative Contrastive Learning for Dense Text Retrieval.* ICLR 2021,
  [2007.00808](https://arxiv.org/abs/2007.00808). **(ANCE — C1's supporting evidence. §6.1
  quote confirmed. The "100×" is *latency*, not cost: 99× vs. BERT Rerank alone, 122× vs.
  the full sparse pipeline, and it excludes 10h of offline corpus encoding.)**
- Qu, Ding, Liu, Liu, Lv, Zhao, Zhang, She, Wang, Yu, Wu. *RocketQA: An Optimized Training
  Approach to Dense Passage Retrieval for Open-Domain Question Answering.* NAACL 2021,
  [2010.08191](https://arxiv.org/abs/2010.08191). **(C4's source. Table 3 confirmed:
  32.39 / 26.03 / 36.38. ~70% of top-retrieved "negatives" were actually
  positives/highly-relevant.)**
- Zhan, Mao, Liu, Guo, Zhang, Ma. *Optimizing Dense Retrieval Model Training with Hard
  Negatives.* SIGIR 2021, [2104.08051](https://arxiv.org/abs/2104.08051). **(ADORE +
  STAR — C5's source. ⚠ The 0.347/0.876 figures are the `ADORE+STAR` row; 3 of 5 bare
  ADORE variants lose to ANCE. Frozen-index/query-encoder-only characterization confirmed
  verbatim.)**
- Jacob, Lindgren, Zaharia, Carbin, Khattab, Drozdov. *Drowning in Documents: Consequences
  of Scaling Reranker Inference.* ReNeuIR @ SIGIR 2025 workshop,
  [2411.11767](https://arxiv.org/abs/2411.11767). **(⚠ Does NOT measure a false-positive
  rate. Shows recall declining past a non-monotone peak; "phantom hits" is qualitative and
  outcome-selected. Its FP-rate mechanism is cited to Zamani, not measured.)**
- Meng, Arabzadeh, Askari, Aliannejadi, de Rijke. *Ranked List Truncation for Large
  Language Model-based Re-Ranking.* SIGIR 2024,
  [2404.18185](https://arxiv.org/abs/2404.18185). **(A3 — the third objection A2 must
  answer. Also the source of `THEORY.md` §6.4's truncation quote, checked and accurate.)**
- Dou, Shen, Zhou, Bai, Kou, Nie, Cui, Yu. *Enhancing Deep Entity Resolution with
  Integrated Blocker-Matcher Training: Balancing Consensus and Discrepancy.* CIKM 2024,
  508–518. DOI [10.1145/3627673.3679843](https://doi.org/10.1145/3627673.3679843).
  **(MutualER — joint blocker–matcher training. Abstract verified via Semantic Scholar;
  closed access, no arXiv. DBLP title search returns zero: the name is only inside the
  paper.)**
- Wu, Wu, Dong, Hua, Zhou. *Blocker and Matcher Can Mutually Benefit: A Co-Learning
  Framework for Low-Resource Entity Resolution.* PVLDB 17(3):292–304, 2023. DOI
  [10.14778/3632093.3632096](https://doi.org/10.14778/3632093.3632096). **(CLER — the third
  joint-training paper; distinct from MutualER, disjoint authors. Code: `wusw14/CLER`.
  New to our docs.)**
- Jain, Sarawagi, Sen. *Deep Indexed Active Learning for Matching Heterogeneous Entity
  Representations.* PVLDB 2022, [2104.03986](https://arxiv.org/abs/2104.03986). **(DIAL —
  *is* joint training: "jointly learns embeddings to maximize recall for blocking and
  **accuracy** for matching".)**
- Ren, Qu, Liu, Zhao, She, Wu, Wang, Wen. *RocketQAv2: A Joint Training Method for Dense
  Passage Retrieval and Passage Re-ranking.* EMNLP 2021,
  [2110.07367](https://arxiv.org/abs/2110.07367). **(The IR joint-training analogue.)**
- Zhang, Gong, Shen, Lv, Duan, Chen. *Adversarial Retriever-Ranker for Dense Text
  Retrieval.* ICLR 2022, [2110.03611](https://arxiv.org/abs/2110.03611). **(AR2 — the
  reranker's objective in the retriever's loop.)**

**Cited but NOT verified against the primary — do not quote without reading:**

- Li, Li, Suhara, Doan, Tan. *Deep Entity Matching with Pre-Trained Language Models.*
  VLDB 2020, [2004.00584](https://arxiv.org/abs/2004.00584). **(Ditto — E1 target.)**
- Zhang, Rekatsinas et al. *AnyMatch.* 2024, [2409.04073](https://arxiv.org/abs/2409.04073).
  **(E1 target; the data-recipe bet.)**
- *Jellyfish.* [2312.01678](https://arxiv.org/abs/2312.01678). **(E1 target. Caveat on
  file: Amazon-Google was *seen* in training; Abt-Buy is zero-shot.)**
- Steiner, Peeters, Bizer. *Fine-tuning LLMs for EM.* 2024,
  [2409.08185](https://arxiv.org/abs/2409.08185). **(D1 — mixed results on
  LLM-generated EM training data.)**
- Mudgal, Li, Rekatsinas, Doan, Park, Krishnan, Deep, Arcaute, Raghavendra. *Deep Learning
  for Entity Matching.* SIGMOD 2018. **(DeepMatcher/Magellan — B2's saturation motivation.)**
- Moreira et al. *NV-Retriever.* 2024, [2407.15831](https://arxiv.org/abs/2407.15831);
  Solatorio. *GISTEmbed.* 2024, [2402.16829](https://arxiv.org/abs/2402.16829). **(C4's
  false-negative safety rails.)**
- Wang et al. *E5-mistral.* [2401.00368](https://arxiv.org/abs/2401.00368). **(C3 —
  instruction-following embedders.)**
- Leone et al. PVLDB 15(8), 2022. **(E2's starting point — the cross-field survey
  `THEORY.md` erratum 13 credits with documenting the silos.)**
- Tu et al. *Unicorn.* SIGMOD/PACMMOD 2023 *(no arXiv; OpenReview `388Cge6WPN`)*. **(E2 —
  the generalist precedent, on a different axis.)**
- Elkan. *The Foundations of Cost-Sensitive Learning.* IJCAI 2001. **(A2 — "F1 is itself
  arbitrary". Reached via `THEORY.md` §6.5.)**
- Hassanzadeh, Chiang, Lee, Miller. *Framework for Evaluating Clustering Algorithms in
  Duplicate Detection.* PVLDB 2009. **(B1's threshold-fragility numbers — reached via
  `THEORY.md` §7.1, not read here.)**
- Bansal, Blum, Chawla. *Correlation Clustering.* **(B1 — "closure is optimal iff the
  pairwise labels are consistent". Via `THEORY.md` §7.)**
- Ailon, Charikar, Newman. *Aggregating Inconsistent Information: Ranking and Clustering.*
  **(B1 — the pivot algorithm `CorrelationClusterer` implements.)**
- Breunig. *Let the Model Write the Prompt.* [dbreunig.com, 2025-06-10](https://www.dbreunig.com/2025/06/10/let-the-model-write-the-prompt.html).
  **(D1's counter-evidence. Qwen3-0.6B 60.7%→82% is the author's own figure, but it is
  binary-match accuracy with *unreported class balance*. Cite the primary, never a recap —
  a widely circulated recap of this talk contained two claims absent from it.)**
- Papadakis et al. *Pre-trained Embeddings for Entity Resolution*; Papadakis et al., ACM
  CSUR 53(2) 2020; Christen, TKDE 2012. **(A1 — the field's "precision significantly
  raises after matching". Via `THEORY.md` §6.3.)**

---

*End of agenda. Items are added by appending to a thread and updating §1's board — the
threads are stable, the board is not.*
