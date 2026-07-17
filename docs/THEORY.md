# Theory: the mathematical foundation of langres

> **Status:** foundational design document, **v2 — cited**. Descriptive of where
> `langres.core` is heading, **not** of what it ships today. Where the theory
> contradicts current code, §11 says so explicitly.
>
> **Epistemic status:** every claim below has been checked against primary
> sources. Several claims in v1 did not survive that check and have been
> **removed or corrected in place** — see §12 (*Errata*). This document
> deliberately records what is *ours*, what is *prior art*, and what is
> *unverified*, because a formalization whose provenance is vague is worse than
> no formalization.
>
> **What this is for.** langres's components (`Blocker`, `Matcher`, `Clusterer`)
> are named after *positions in a pipeline*. This document argues that position
> is not a type. The value is not elegance — it is that a formalization is
> **falsifiable**. It has already caught two errors in our own design and, under
> literature review, four errors in its own first draft.

---

## 0. Relation to prior work — read this first

**The algebra below is a re-parameterization of established results, not an
invention.** Every operation has a name in the literature. Stating this up front
is not modesty; it is the difference between a document that survives review and
one that does not.

| Prior work | What it already has | What it lacks |
|---|---|---|
| **Fellegi & Sunter 1969** (via Winkler RR93-8) | The carrier: *"In the product space A × B of files A and B…"*. Blocking **and** matching already described as restricting subsets of $A \times B$. The three-region decision. | No shape-parameterized objective. |
| **AJAX** (Galhardas et al., VLDB 2001) | Five composable logical operators — *mapping, view, matching, clustering, merging*. Matching = "compute a distance for every pair… return those within ε" = our $(E,s)$ + $t$. Merging = our fuse. | Blocking is **physical**, not logical (see below). Clustering is procedural (`BY METHOD`), not an argmax. |
| **Meta-blocking** (Papadakis et al., TKDE 2014) | Our carrier (weighted blocking graph), our $\sigma$ ("Edge Weighting"), our $\pi$ ("Graph Pruning"), and a 2×2 of $\mathcal{F}$: weight-vs-cardinality × global-vs-local, incl. *Top-K Edges* and *≤k per node*. | Confined to **blocking**. Never reaches matching or clustering. |
| **BFKPT** (Burdick, Fagin, Kolaitis, Popa, Tan; ICDT 2015 / TODS 2016) | *"a single formalism… in which the constraint language, the sets of constraints allowed, and the weight function… are **parameters of the definition**"*. Max-weight repairs; FDs give many-to-one and one-to-one. | Blocking explicitly **outside** the formalism. Disclaims the equivalence shape. |
| **Swoosh** (Benjelloun et al., VLDBJ 2009) | Generic match+merge with **ICAR**. The theory of our phase 2 (§9). | ER result is a **least fixpoint**, not an argmax. No objective, no threshold, no shape. |
| **Dedupalog** (Arasu, Ré, Suciu; ICDE 2009) | Declarative hard/soft constraints reducing to correlation clustering. | $\mathcal{F}$ **fixed** at equivalence relations. Refuses real-valued weights on purpose (§8). |

**The honest claim.** Two halves exist — *"σ/π/𝓕 over a scored pair set"*
(meta-blocking) and *"max-weight subject to parameterized constraints"* (BFKPT) —
and they have never been joined. **Neither admits blocking into its formalism.**
The contribution here is the *unification and the uniform carrier*, not the
operations. Expect exactly two objections: "that's meta-blocking" and "that's
BFKPT." The answer to both is scope.

**We are taking a contested side.** AJAX places blocking at the *physical* level
— an implementation of matching constrained to produce a superset with *"no false
dismissals"*. Every stage-based work (AJAX, JedAI, BFKPT, Christophides et al.)
refuses to make blocking a logical operation. Our §3 claim that blocking is $\pi$
at a permissive $\mathcal{F}$ is a **position**, not a neutral observation, and
§7 is the argument for it.

---

## 1. Notation

| Symbol | Name | Meaning |
|---|---|---|
| $A$, $B$ | the two sides | Record sets. For dedupe, $A = B$. |
| $E$ | the pairs | $E \subseteq A \times B$. The rows of the table. |
| $s$ | the score | $s : E \to S$. The score column. |
| $S$ | the score space | $\mathbb{R}$, $\{0,1\}$, or $\mathbb{R}^d$. **Need not be a range.** |
| $R$ | scored relation | $(E, s)$. **The only type that flows.** |
| $t$ | the price | What it costs to keep a row. |
| $\mathcal{F}$ | the feasible class | A rule on the shape of the **whole answer**. |
| $T$ | true pairs | Genuinely matching pairs in the data. |
| $\eta$ | matcher sensitivity | $P(\text{yes} \mid \text{true})$. |
| $\varphi$ | matcher fall-out | $P(\text{yes} \mid \text{false})$. |

---

## 2. The carrier and the operations

$$R = (E, s), \qquad E \subseteq A \times B, \qquad s : E \to S$$

This is Fellegi–Sunter's product space. Every stage maps a scored relation to a
scored relation, so **the composite of two stages is itself a stage** — the
property that makes `nn.Module` nest, here as a consequence rather than a
convention.

**Score.** $\sigma_f (E, s) = (E, f)$ — keep rows, replace numbers.

**Select.** Keep numbers, delete rows:

$$\pi_{\mathcal{F}, t}(E, s) \;=\; \underset{E' \subseteq E,\; E' \in \mathcal{F}}{\arg\max} \;\; \sum_{(a,b) \in E'} \big( s(a,b) - t \big)$$

*Out of every legally-shaped way to keep some rows, take the one whose scores add
up highest after each row pays the price $t$.*

**Combine.** $\cup, \cap, \setminus$ on pair sets. Shipped as
`CompositeBlocker(op=…)`; it is AJAX's multi-pass blocking (2001).
**Not a contribution, and "lattice" is the wrong word for it** — ER's natural
lattice is the *partition refinement lattice* (used with a meet by Whang &
Garcia-Molina), a different object.

**Why $t$ is not a separate concept.** With $\mathcal{F}$ unconstrained,
maximizing $\sum (s-t)$ means exactly *keep every row where $s > t$*. The
threshold is what the formula does when nothing else constrains it.

---

## 3. One operation, seven names

| Name | $\mathcal{F}$ | $t$ | Algorithm forced | Cost |
|---|---|---|---|---|
| all-pairs blocking | no rule | $-\infty$ | enumerate | $O(n^2)$ |
| top-k blocking | $\deg_L(a) \le k$ | $-\infty$ | index lookup | see §4 |
| threshold matching | no rule | $t$ | compare | $O(\lvert E\rvert)$ |
| three-region | 3-way split | $t_{lo}, t_{hi}$ | compare ×2 | $O(\lvert E\rvert)$ |
| mention → KB (+NIL) | $\deg_L(a) \le 1$ | $t$, else NIL | argmax per anchor | $O(\lvert E\rvert)$ |
| 1-to-1 assignment | $\deg_L \le 1 \wedge \deg_R \le 1$ | $t$ | Hungarian / LSAP | $O(n^3)$ |
| clustering | equivalence relations | $t$ | **not approximable** (§8) | — |

A blocker is a scorer with a permissive $\mathcal{F}$ and $t \approx -\infty$; a
matcher is a scorer with a strict $t$. A reranker is a second $\sigma$ over an
existing $E$.

**Prior art for the row structure.** Meta-blocking's 2×2 (TKDE 2014) already
enumerates weight/cardinality × global/local — i.e. rows 1–3 of this table —
within blocking. BFKPT already parameterizes the constraint language. This table
is the merge of the two.

---

## 4. Decomposability buys precomputation — *not* sublinearity

A scorer is **decomposable** iff $s(a,b) = \oplus(g(a), h(b))$.

**This is DPR's word, with DPR's meaning** (Karpukhin et al., EMNLP 2020, §3.1):
*"the similarity function needs to be **decomposable** so that the
representations of the collection of passages can be pre-computed."* Note DPR's
justification stops at **pre-computation** — that scope is correct, and v1 of this
document overreached past it. In the IR taxonomy (Guo et al., arXiv:1903.06902)
the general form is $f(s,t) = g(\psi(s), \phi(t), \eta(s,t))$; ours is that with
$\eta = \emptyset$, which they call **representation-focused** (vs.
*interaction-focused*). Use their names in prose.

> **Correction (v1 was wrong).** v1 claimed decomposable ⟹ sublinear retrieval.
> **False.** Counterexample: **ARC-I** (Hu, Lu, Li, Chen; NIPS 2014) is fully
> decomposable — every $h(b)$ is precomputable — but $\oplus$ is a learned MLP, so
> **no index exists** and you still score all $N$. Decomposability and
> indexability are two independent facts and v1 collapsed them into one.

So the design has **two orthogonal axes**. The bit lives on the scorer; the
retrievability enum lives on $\oplus$:

```
Scorer  = Joint(f)                       # no precompute; rescore candidates only
        | Decomposed(g, h, combine)      # h(b) precomputable over the corpus

Combine = Indexed(kind)                  # equality→hash | cosine/IP→ANN | Jaccard→MinHash
        | TwoStage(proxy, rescore)       # MaxSim→token-MIPS+rescore; overlap@θ→prefix+verify
        | ScanOnly                       # arbitrary learned ⊕: precompute yes, retrieve no
```

| $g$ reduces a record to | $\oplus$ | $S$ | retrievability | in langres |
|---|---|---|---|---|
| a normalized key | equality | $\{0,1\}$ | `Indexed` — hash bucket | `KeyBlocker` ✓ |
| a dense unit vector | cosine | $\mathbb{R}$ | `Indexed` — ANN | `VectorBlocker` ✓ |
| a dense vector | inner product | $\mathbb{R}$ | `Indexed` — MIPS | — |
| a sparse tf-idf vector | weighted IP | $\mathbb{R}$ | `Indexed` — inverted index + WAND | **absent; the IR/ER default** |
| a shingle set | Jaccard | $\mathbb{R}$ | `Indexed` — MinHash/LSH | absent |
| a q-gram set | overlap @ θ | $\mathbb{R}$ | `TwoStage` — prefix filter + verify | absent |
| token embeddings | MaxSim | $\mathbb{R}$ | `TwoStage` — token-ANN + rescore | via Qdrant index |
| anything | learned MLP | $\mathbb{R}$ | **`ScanOnly`** | — |
| — *must see both* | — | any | **none possible** | `LLMMatcher` |

Three consequences the sources force:

1. **ColBERT is decomposable, not a middle case.** $g$ = query token embeddings,
   $h$ = doc token embeddings, both independent; MaxSim is cheap and has *"no
   trainable parameters"*. Every clause holds. Its *codomain* is set-valued; that
   is what makes $\oplus$ `TwoStage`, not what makes it joint. (We suspected
   ColBERT refuted the bit. It does not.)
2. **$\oplus \to$ index must be a registry lookup, not a hardcoded switch.**
   MUVERA (arXiv:2405.19504) reduces MaxSim to plain MIPS via fixed-dimensional
   encodings — moving it from `TwoStage` to `Indexed`. **The mapping is the best
   known reduction, not an eternal property of the scorer.** This is exactly
   langres's existing `method_registry` pattern.
3. **`TwoStage` is lossy** (MUVERA: single-vector MIPS on individual query
   embeddings *"can fail to find the true MV nearest neighbors"*), whereas
   `Indexed` is exact modulo ANN. That distinction should be visible in the type,
   since it is precisely what `score_blocking` measures.

> **Correction (v1 was wrong).** v1 said $\oplus$ must "admit sublinear NN
> search," implying a metric condition. Metric-ness is **neither necessary**
> (MIPS needs asymmetric LSH — Shrivastava & Li, NIPS 2014; and FAISS/HNSW need
> no guarantee at all) **nor sufficient** (edit distance is a metric with no good
> LSH). The correct, weaker condition is *"$\oplus$ admits an index structure."*

**A hidden dependency worth naming:** $g$ and $h$ frequently depend on **corpus
statistics** (IDF; a learned tuple embedding). $s(a,b) = \oplus(g(a), h(b))$
conceals that $g,h$ require a **fit** over the corpus — directly relevant to
langres's fit/transform seam.

---

## 5. The score space must be rich enough for $\mathcal{F}$

$S$ need not be a range. If $s \in \{0,1\}$ and $0 < t < 1$, maximizing
$\sum(s-t)$ means exactly "keep the yeses." `KeyBlocker` is this case: it buckets
by a normalized key and emits candidates carrying **no score at all**.

But $\pi$ needs a total order, and a two-valued score gives massive ties:

| $\mathcal{F}$ | boolean $S$ | why |
|---|---|---|
| no rule (threshold) | ✓ | "keep the yeses" |
| equivalence (clustering) | ✓ | the original $\pm1$ correlation-clustering formulation (Bansal et al. 2004) |
| $\deg \le k$ (top-k) | ✗ **degenerate** | if 40 pairs say yes, *which 25*? arbitrary |
| assignment | ✗ **degenerate** | any perfect matching is equally optimal |

`PairwiseJudgement`'s existing `decision: bool | None` vs `score: float | None`
split already encodes this. The algebra states the implication: **a decider
cannot drive top-k or assignment**, and nothing currently prevents wiring one
that way. This is a checkable wiring rule.

$S = \mathbb{R}^d$ (a `ComparisonVector`) is not orderable, so $\pi$ on a
vector-valued relation is a type error; a scalarizer must sit between. That is
what `Comparator` → `WeightedAverageMatcher` already is. Principled, just untyped.

---

## 6. Staging: one theorem, and a corollary that does not follow

> **Attribution first.** The result in §6.2 is **not ours.** It was published for
> ranking metrics by **Zamani, Bendersky, Metzler, Zhuang & Wang, "Stochastic
> Retrieval-Conditioned Reranking", ICTIR 2022**, whose Eq. 4 — multiplied
> through by $n$ — **is** the precision formula below, with their $\varepsilon^+
> / \varepsilon^-$ equal to our $\eta / \varphi$. Verbatim: the retrieval model
> *"should maximize min(1, n/k), **instead of Recall@N (i.e., the current popular
> belief)**"*. Underneath that, the precision formula is **Bayes** — positive
> predictive value as a function of prevalence (Vecchio, NEJM 1966).
>
> **What is ours is only the transplant**: ER's objective counts every emitted
> pair (F1), not a rank-truncated Precision@k, and ER's literature has not
> absorbed the correction. Do not present the derivation as new.

### 6.1 The theorem

Because $\pi$ only ever deletes rows, for $E_1 \supseteq E_2 \supseteq \dots$:

$$\text{recall}(E_1) \;\ge\; \text{recall}(E_2) \;\ge\; \dots \;\ge\; \text{recall}(E_n)$$

A true pair deleted at stage $k$ cannot exist at stage $k+1$. **Final recall
$\le$ blocking recall.** Omissions are permanent; commissions are recoverable.
Not in dispute.

### 6.2 The corollary that does not follow

> ~~"Therefore early stages must maximize recall."~~

A ceiling constrains the *achievable set*; it says nothing about where the
*optimum* lies. Conflating "the ceiling rises with $r_B$" with "the optimum rises
with $r_B$" is a non-sequitur.

With $T_B = r_B T$ and $F_B$ the surviving false candidates, and modelling the
matcher per-pair:

$$R = \eta \, r_B \qquad\qquad P = \frac{\eta \, r_B \, T}{\eta \, r_B \, T + \varphi \, F_B}$$

Parameterize by $k$ (top-$k$ over $n$ anchors), $F_B \approx nk - r_B T$. As
$k \to \infty$: $r_B \to 1$ and **saturates**, so $R \to \eta$; $F_B \to \infty$,
so $P \to 0$ and $F_1 \to 0$. $F_1$ is positive at moderate $k$, so it attains an
interior maximum.

> **Result (generic, not universal).** For $\varphi > 0$, $F_1$ generically
> attains an interior maximum at finite $k$, so the optimal blocking recall is
> **generically < 1**.

**Why not "strictly < 1":** $dF_1/dk < 0$ at saturation requires
$r_B'(k^*) = 0$. With finite data and discrete $k$, recall reaches 1 with a
**kink** ($r_B' > 0$), the sign becomes ambiguous, and the optimum **can** sit at
$r_B = 1$. v1 overclaimed; this is the corrected statement.

### 6.3 The boundary case is the doctrine

At $\varphi = 0$, $P = 1$ for any $F_B$, so $F_1 = 2R/(1+R)$ is monotone in $r_B$
and you *should* maximize blocking recall.

> **"Maximize blocking recall" is the $\varphi \to 0$ limit.** It is a corollary
> with a hypothesis, and the hypothesis is rarely stated.

ER states it, unconditioned, in the field's own words:

- Papadakis et al., *Pre-trained Embeddings for Entity Resolution*, §5.1:
  *"Recall is the most critical evaluation measure for blocking, as it typically
  sets the upper bound for the subsequent matching step"*; *"precision is
  typically low after blocking… but **significantly raises after matching**."*
- *Towards Universal Dense Blocking for Entity Resolution* (arXiv:2404.14831):
  *"Since the matching phase provides further assess to pair quality and
  discarded pairs cannot be recovered, the blocking phase typically prioritizes
  pair completeness."*

**But do not strawman the field.** ER already has Pair Completeness / Reduction
Ratio / Pairs Quality (Christen, TKDE 2012: *"This trade-off between PC and PQ is
similar to the precision-recall trade-off in information retrieval"*; Papadakis
et al., ACM CSUR 53(2) 2020: *"a blocking scheme should achieve a good balance
between these two competing objectives"*), and uses
$F_{PC,RR} = 2 \cdot PC \cdot RR / (PC+RR)$, which **does** have an interior
optimum in $k$. **The distinction that survives:** RR is a *comparison-count*
metric, so ER's tradeoff is quality-vs-**cost**. Ours is quality-vs-**quality**.
No ER source we found derives the blocking operating point from the matcher's
error rate. (Nearest ER work: **MutualER**, CIKM 2024, joint blocker–matcher
training.)

### 6.4 Two distinct effects — and $\varphi$ is not constant

1. **Composition effect** — always present, needs *no* degradation. With
   $\eta, \varphi$ constant, $P$ still collapses, because $P$ depends on the
   composition of $E_B$: true pairs are $O(n)$, all pairs $O(n^2)$.
2. **Degradation effect** — $\varphi$ grows with group size. Does not apply to a
   pairwise matcher; **does** apply to set-wise matchers (`GroupwiseMatcher`, a
   ComEM-style `SelectMatcher`). Empirically corroborated for LLM rerankers
   (Meng et al., *Ranked List Truncation*: *"a deeper re-ranking cut-off does not
   consistently result in improvement and can even be detrimental"*).

> **Known weakness of the model.** Constant $\varphi$ is wrong and load-bearing.
> Blocker and matcher errors are **correlated** — both keyed on similarity — so
> the marginal candidates admitted at large $k$ are the low-similarity ones the
> matcher rejects most easily, i.e. $\varphi$ decays with rank. $P \to 0$ survives
> iff $\inf \varphi > 0$, but constant-$\varphi$ **mis-locates the optimum**. This
> is the first objection any reviewer will raise.

### 6.5 The consequence, and the experiment it demands

The blocker's optimum is a function of $\varphi$, a property of the **matcher**.
**The blocker cannot be tuned in isolation.** Any procedure selecting a blocking
parameter without reference to the downstream matcher optimizes something that is
not the objective.

> **The honest caveat.** ER's own measurement — *"precision significantly raises
> after matching"* — is evidence that $\varphi$ is **small** in practice. If
> $\varphi$ is tiny, the interior optimum sits at $r_B \approx 0.999$ and this
> section is *true but operationally uninteresting*. **The claim has teeth only
> with a measured $\varphi$.** Measuring $\varphi$ for LLM judges on our benchmark
> portfolio is therefore the experiment this document exists to motivate — not a
> follow-up.

**A framing correction.** One does not "maximize precision" — that has a trivial
winner (return nothing: $P=1$, $R=0$). $t$ **is** the precision/recall dial.
Blocking and matching are the same formula at different $t$: blocking sets
$t \approx -\infty$, matching sets $t$ high. Hence: **use a cheap score to decide
which pairs are worth paying the expensive score for.**

**And $F_1$ is itself arbitrary.** Elkan (*The Foundations of Cost-Sensitive
Learning*, IJCAI 2001, Eq. 2) gives the optimal threshold
$p^* = \frac{C(1,0)-C(0,0)}{[C(1,0)-C(0,0)]+[C(0,1)-C(1,1)]}$, which reduces to
$t = C_{FP}/(C_{FP}+C_{FN})$ **only when correct classifications cost zero**. Under
a cost objective the "optimal recall < 1" statement changes form. Choosing $F_1$
and then finding an interior max is partly an artifact of the metric.

---

## 7. Transitive closure, stated correctly

$\pi$ as defined requires $E' \subseteq E$ — it may only *delete*. **Transitive
closure adds rows**: given $E'$ it returns $\mathrm{cl}(E')$, and every edge in
$\mathrm{cl}(E') \setminus E'$ was never scored. So closure is not a $\pi$ over
$E$, and §2's formula as literally written cannot express it.

> **Correction (v1 was wrong, twice).** v1 claimed closure "assigns $+\infty$ to
> every edge it invents but never scored," and that the treatment of *unobserved*
> pairs distinguishes the algorithms. Both are false.
>
> 1. **$+\infty$ on all unobserved pairs yields one giant cluster**, not closure —
>    nearly every pair is unobserved, and each would be forced intra-cluster.
> 2. **"$+\infty$ on the edges it invents" is circular**: $\mathrm{cl}(E')\setminus E'$
>    is *defined by the output*, so it cannot define the objective producing it.
> 3. **Wrong axis.** Closure and correlation-clustering-over-observed-edges *both*
>    price unobserved pairs at 0. That is not what separates them.

**The correct statement.** Transitive closure is correlation clustering with:

- **$+\infty$ on observed positive edges** — a hard constraint: never cut a $+$ edge;
- **$0$ on everything else** — observed negatives *and* unobserved pairs alike;
- tie-broken to the **finest** zero-cost partition (= connected components).

So what actually separates the algorithms is (a) **hard vs. soft positives** and
(b) **whether observed negatives are priced at all**. (b) is the load-bearing
mechanism, and it has near-verbatim prior art: Draisbach, Christen & Naumann,
*Transforming Pairwise Duplicates to Entity Clusters for High-quality Duplicate
Detection*, ACM JDIQ 12(1), 2019 — *"As most of these algorithms use pairwise
comparisons, the resulting (transitive) clusters can be inconsistent: **Not all
records within a cluster are sufficiently similar to be classified as
duplicate.**"* Their EMCC exists to produce consistent clusters.

**When closure is exactly right — the cleanest available statement.** Bansal,
Blum & Chawla (*Correlation Clustering*, Machine Learning 56, 2004): *"if there
exists a perfect clustering… the optimal clustering is easy to find: just delete
all '−' edges and output the connected components."* **Closure is optimal iff the
pairwise labels are consistent.** Its failure mode is precisely inconsistency.

**"0 = neutral" is not ours either** — it is Bansal's own footnote 1, in an ER
example: *"the natural edge label is $\log(\Pr(\text{same})/\Pr(\text{different}))$.
**This is 0 if the classifier is unsure**"*. And "0 = no information = ignored =
correlation clustering (general version)" is textbook (Garvardt & Komusiewicz,
arXiv:2605.13917; Demaine et al., TCS 2006 — weighted CC on non-complete graphs is
APX-hard with an $O(\log n)$ approximation).

**What survives as ours — as engineering, not discovery.** Pricing unobserved
pairs at the **prior log-odds** ($\approx -\log n$ per pair, making a size-$k$
cluster cost $\sim k^2 \log n / 2$ and killing chaining). Its spirit is
microclustering (Miller, Betancourt, Zaidi, Wallach, Steorts, arXiv:1512.00792) —
priors designed so ER cluster sizes grow sublinearly — via a different mechanism.

### 7.1 Closure is threshold-fragile, not simply bad

The strong form ("closure over-merges, therefore it is poor") is **overstated**.
Hassanzadeh, Chiang, Lee & Miller (*Framework for Evaluating Clustering Algorithms
in Duplicate Detection*, PVLDB 2(1), 2009) compared 12 algorithms including
Partitioning (= transitive closure). Their conclusion: *"confirms the common
wisdom that this scalable approach results in poor quality of duplicate groups…
even when compared to other clustering algorithms that are also efficient."* But
their Table 3 (medium-error, 500 true clusters) shows:

| $\theta$ | Partitioning F1 | #clusters found | CENTER F1 | MCL F1 |
|---|---|---|---|---|
| 0.2 | **0.177** | **51** | 0.666 | 0.599 |
| 0.3 | 0.622 | 354 | 0.825 | 0.841 |
| 0.4 | **0.850** | 704 | 0.887 | 0.906 |

At $\theta = 0.4$ Partitioning reaches F1 **0.850** — beating MinCut (0.771) and
near CENTER. **The defensible claim is threshold-fragility**: 0.850 → 0.177 across
two threshold notches, 500 true clusters collapsing to 51, and no way to know the
right threshold a priori. (All datasets synthetic — UIS generator over 2,139
company names / 10,425 DBLP titles.) Corroborated on real and 10M-scale data by
Saeedi, Peukert & Rahm (*Scalable Matching and Clustering of Entities with FAMER*,
CSIMQ 16, 2018): *"Connected Components reaches the lowest F-Measure for all
datasets and almost all threshold values because it suffers from very poor
precision values."* — with the caveat that FAMER assumes duplicate-free,
source-consistent inputs that its winners exploit and connected components cannot.

Note also (Binette & Steorts): *"any linkage that does not satisfy transitive
closure is impossible"* — transitivity is **required** of ground truth. The
question is only *how* it is enforced: closure enforces it by always merging;
correlation clustering by minimizing disagreement.

### 7.2 What to do about it — and who already did it

**Cluster verification / repair is an active field, not a gap.**

- **FAMER SplitMerge** (Saeedi et al. 2018): connected components → **split** → merge.
- **Christen, Obraczka, Hofer, Franke, Rahm**, *Graph-based Active Learning for
  Entity Cluster Repair*, arXiv:2401.14992 (2024): *"Cluster repair methods aim to
  determine errors in clusters and modify them."* Follow-up with LLM active
  learning: ACM JDIQ 2025, DOI 10.1145/3735511.
- **Fu, Tang, Khan, Mehrotra, Ke, Gao**, *In-context Clustering-based Entity
  Resolution with LLMs*, PACMMOD / SIGMOD 2026 — LLMs cluster records directly,
  with a **Misclustering Detection Guardrail** and record-set regeneration.
- **ComEM** (arXiv:2405.16884) — multi-record selection beyond pairwise.

langres would be **joining this line, not opening it.**

**The cheap, targeted move that is still worth doing:** score
$\mathrm{cl}(E') \setminus E'$ — the edges closure invented. It is pairwise (no
set-level score needed), small, and aimed at the unjustified assumption. **And the
zero-cost version first:** the edges that *were* candidates and scored **below**
$t$ are already in the `JudgementLog`. The diagnostic *"does this cluster contain a
pair we scored below threshold?"* costs **$0** and directly measures (b) above.

---

## 8. A hard limit: our $\pi$ at $\mathcal{F} = $ equivalence is not approximable

Dedupalog (§V.A) explains why it refused real-valued weights: *"can we still
provide theoretical guarantees of quality? **We give evidence of a negative
answer**… Demaine et al. show that the existence of a constant factor
approximation for correlated clustering would provide a constant factor
approximation for the **MULTICUT** problem, which is a very hard problem believed
to have an unbounded approximation factor."*

**Our $\pi$ at $\mathcal{F} = $ equivalence with real scores *is* weighted
correlation clustering.** We must not promise optimization there. The unweighted
$\pm1$ case is a 3-approximation (Ailon, Charikar & Newman, JACM 2008 — CC-Pivot);
the weighted general-graph case is APX-hard with $O(\log n)$ (Demaine et al., TCS
2006). **The type system should not let a user ask for something we cannot deliver.**

Dedupalog also attacks our decomposition directly (§VI): *"The approach of record
matching followed by clustering is not amenable to collective clustering and
cannot exploit constraints."* See §10.

---

## 9. Phase 2: the algebra over clusters — and Swoosh already has its theory

Two operations do not fit §2, and they fit each other:

- **Verify** — "are these four really one entity?" scores a *set*, not a pair.
- **Fuse** — every §2 operation is *table → table*; fusion is *table → records*.
  It creates a record never in $A$ or $B$. **It is the exit.**

```
phase 1 — carrier: a table of scored PAIRS
  σ  rescore    ·  π  select  ·  ∪ ∩ ∖  combine

phase 2 — carrier: CLUSTERS
  verify  cluster → clusters     (split what closure over-merged)
  fuse    cluster → record       (`Canonicalizer`)
```

**This is Swoosh, and Swoosh has the theorem we need.** Its ICAR properties
(Garcia-Molina keynote, verified):

- Idempotence: $M(r_1,r_1) = \text{true}$; $\langle r_1,r_1\rangle = r_1$
- Commutativity: $M(r_1,r_2) = M(r_2,r_1)$; $\langle r_1,r_2\rangle = \langle r_2,r_1\rangle$
- Associativity: $\langle r_1,\langle r_2,r_3\rangle\rangle = \langle\langle r_1,r_2\rangle,r_3\rangle$
- **Representativity:** *"If $\langle r_1,r_2\rangle = r_3$, then for any $r_4$
  such that $M(r_1,r_4)$ is true we also have $M(r_3,r_4)$."*

They guarantee *"ER result independent of processing order"* — and **R-Swoosh
*"merges records as soon as they match"*.**

> **This answers the batch-vs-incremental merge fork directly.**
> **Representativity is exactly the licence to merge during matching**: the merged
> record inherits every match of its parents, so comparing $C$ against $AB$ (and
> discarding $A$, $B$) loses nothing. Merging on match is safe **iff** ICAR holds.

Two consequences we must absorb:

1. **Idempotence + commutativity + associativity make merge a join-semilattice
   operation**, which is exactly the condition under which a binary merge lifts to
   a well-defined, order-independent function on *sets*. So `fuse: cluster → record`
   is Swoosh's binary merge folded over a cluster, and **ICAR is the theory of when
   that fold is well-defined.** *(This lifting is our inference from their verified
   definitions, not their stated claim.)*
2. **Scores break it.** Menestrina, Benjelloun & Garcia-Molina (*Generic Entity
   Resolution with Data Confidences*, CleanDB 2006): *"without confidences, the
   order in which records are merged may be unimportant… **However, confidences may
   make order critical.**"* They also flag that the domination property *"may or may
   not hold in a given application"* and that P3 *"rules out negative evidence."*
   **A scored fuse forfeits ICAR's order-independence guarantee.** Either drop
   scores from fuse or drop the guarantee — knowingly.

**Un-merging has prior art too.** Whang & Garcia-Molina, *Entity Resolution with
Evolving Rules*, PVLDB 3: *"We also allow input clusters to be **un-merged**…
Un-merging could occur when an ER algorithm decides that some records were
incorrectly clustered."* That paper also formalizes what our incremental fork
needs — *rule monotonic*, *context free*, *general incremental*, *order
independent* — over the **partition refinement lattice** with a meet.

**Naming.** `fuse` already has four names: AJAX's **merging** operator (2001),
Swoosh's **merge** (2009), **data fusion** (Bleiholder & Naumann, ACM CSUR 41(1),
2008), and **canonicalization** (Binette & Steorts, arXiv:2008.04443). langres
already ships `Canonicalizer`. **Use an existing name; do not coin `fuse`.**

---

## 10. What the algebra does not cover

**Collective resolution — and it is worse than "a deferral."** Dedupalog's
constraints span entity *types*: clustering two papers forces clustering their
publishers ($\gamma_6$); $\gamma_7/\gamma_8$ are recursive and group-wise. **A
single $A \times B$ scored relation cannot express them.** Dedupalog names our
exact decomposition as the culprit: *"record matching followed by clustering is
not amenable to collective clustering and cannot exploit constraints."*

**Incremental resolution is a fixpoint, not a fold.** If $s$ depends on the answer
and the answer on $s$, you iterate to convergence. Swoosh's ER *is* a least
fixpoint (*"the smallest set $S$ such that…"*), which is a different semantics from
our argmax — the two do not compose trivially.

Both are declared V1.1 deferrals (`docs/USE_CASES.md`). The requirement on the
design is: **do not adopt a shape that forbids them.** An `iterate(stage, until=…)`
combinator and a multi-relation carrier are the extension points.

---

## 11. What this contradicts in the current code

Listed, not silently fixed; each is a change with its own blast radius.

| Location | States | Status |
|---|---|---|
| `core/blocker.py:38` | "High recall: Blocking should have ≥95% recall" | The $\varphi=0$ rule as an unconditional design principle (§6.3). |
| `core/blockers/vector.py:202` | "k_neighbors should be tuned to achieve >= 95% recall" | Same. |
| `RecallCurve.optimal_k(target_recall=…)` | names a recall-hitting $k$ "optimal" | Optimizes a quantity containing no $\varphi$ (§6.5). |
| `core/clusterers/correlation.py` | thresholds via `predicted_match` before weighting | Discards observed negatives — mechanism (b) of §7. |
| `core/clusterer.py` | `Clusterer(threshold=0.5)` | $t$ is a parameter of $\pi$, not of aggregation (§2). |

**In fairness to the optimizers:** `BlockerOptimizer` accepts an arbitrary
objective and `primary_metric`, and `core/autoresearch/objective.py` already
supports Pareto goals and constraints. The doctrine lives in the **guidance and
`optimal_k`**, not the machinery.

---

## 12. Errata — what v1 of this document got wrong

Recorded because the corrections are more instructive than the claims.

1. **§4 — "decomposable ⟹ sublinear."** False. ARC-I is decomposable with a
   learned $\oplus$ and no index. Decomposability buys precomputation only.
2. **§4 — "$\oplus$ must admit sublinear NN search."** Metric-ness is neither
   necessary (MIPS, HNSW) nor sufficient (edit distance).
3. **§6 — "optimal blocking recall is *strictly* < 1."** Generic, not universal:
   discrete $k$ gives a kink at $r_B = 1$.
4. **§6 — presented as our derivation.** It is Zamani et al. (ICTIR 2022) for
   ranking metrics, over Bayes/PPV.
5. **§7 — "closure assigns $+\infty$ to unobserved pairs."** False (yields one
   giant cluster) and circular. The correct statement is $+\infty$ on observed
   **positives**, 0 elsewhere.
6. **§7 — "the treatment of unobserved pairs is the design choice."** Wrong axis;
   the axis is hard-vs-soft positives and whether observed negatives are priced.
7. **§7 — the "face clustering collapses to 0.37" claim. DELETED.** Verified at
   source (Yang et al., CVPR 2020, Table 1: HAC, MS-Celeb-1M, 5.21M) and wrong in
   three ways: the scale is **0–100**, so it is 0.37 **percent** (≈100× the
   natural misreading); the **same run scores BCubed F = 66.96**, so the collapse
   is entirely metric-dependent; and it is **HAC, not transitive closure**. A
   citation that fails this badly under checking is worse than none.
8. **§7 — "no clustering room published cluster verification."** False. It is an
   active field (FAMER SplitMerge 2018; Christen et al. 2024/2025; Fu et al.
   SIGMOD 2026).
9. **§0 — "no unified cross-shape formalism exists."** Overstated. **Unicorn**
   (Tu et al., SIGMOD 2023) unifies seven matching tasks incl. EM, entity linking
   and entity alignment, and claims to be first. Its axis is **data-element type**,
   not constraint shape — so the constraint-shape claim survives, reframed.
10. **"NIL abstention has no analogue in pairwise EM."** False. Fellegi–Sunter is
    a decision *with* abstention (link / possible link / nonlink) since 1969. The
    real distinction: FS abstains from **uncertainty**; NIL asserts
    **non-existence**.
11. **NIL rates "20–33%".** Only **20.4%** is verified (AIDA/CoNLL: 7,136 of
    34,956 mentions — Hoffart et al., EMNLP 2011, Table 1). The 33% has no source.
    **TAC-KBP must not be cited** for a natural NIL rate: it *"tr[ies] to achieve a
    balance between the queries with and without KB entry linkages"* and selects
    *"confusable queries"* — the rate measures the sampling design.
12. **"The assignment constraint was dropped."** True of DB/ER toolkits
    (`recordlinkage`'s `OneToOneLinking`: *"Only 'greedy' is supported at the
    moment"*, `[EXPERIMENTAL]`) — but **false of KG entity alignment**, which uses
    Hungarian/Sinkhorn/optimal transport routinely, and which has NIL under the
    name **"dangling entities"**. That reinvention is *stronger* evidence for the
    siloing thesis than the false claim was.
13. **"The literature is siloed."** True *until recently*, and documented **by**
    the paper that broke it: Leone et al. (PVLDB 15(8), 2022) *"draw a parallel
    between EA and record linkage"* and benchmark Ditto on EA.

---

## References

**Verified against primary sources.**

- Galhardas, Florescu, Shasha, Simon, Saita. *Declarative Data Cleaning: Language, Model, and Algorithms.* VLDB 2001, 371–380. **(AJAX)**
- Benjelloun, Garcia-Molina, Menestrina, Su, Whang, Widom. *Swoosh: a generic approach to entity resolution.* VLDB Journal 18(1):255–276, 2009. *(ICAR verified via Garcia-Molina's keynote slides + the CleanDB 2006 paper; the VLDBJ text itself was not accessible.)*
- Menestrina, Benjelloun, Garcia-Molina. *Generic Entity Resolution with Data Confidences.* CleanDB 2006.
- Whang, Garcia-Molina. *Entity Resolution with Evolving Rules.* PVLDB 3.
- Arasu, Ré, Suciu. *Large-Scale Deduplication with Constraints using Dedupalog.* ICDE 2009, 952–963.
- Papadakis, Koutrika, Palpanas, Nejdl. *Meta-Blocking: Taking Entity Resolution to the Next Level.* IEEE TKDE 26(8), 2014.
- Burdick, Fagin, Kolaitis, Popa, Tan. *A Declarative Framework for Linking Entities.* ICDT 2015 / ACM TODS 2016; *Expressive Power of Entity-Linking Frameworks*, ICDT 2017.
- Fellegi, Sunter. *A Theory for Record Linkage.* JASA 64(328):1183–1210, 1969. *(Via Winkler, Matching and Record Linkage, Census RR93-8, and Winkler & Thibaudeau, RR91-9.)*
- Bansal, Blum, Chawla. *Correlation Clustering.* Machine Learning 56(1–3):89–113, 2004.
- Ailon, Charikar, Newman. *Aggregating Inconsistent Information: Ranking and Clustering.* JACM 55(5), 2008.
- Demaine, Emanuel, Fiat, Immorlica. *Correlation clustering in general weighted graphs.* TCS 361(2–3):172–187, 2006.
- Hassanzadeh, Chiang, Lee, Miller. *Framework for Evaluating Clustering Algorithms in Duplicate Detection.* PVLDB 2(1):1282–1293, 2009.
- Saeedi, Peukert, Rahm. *Scalable Matching and Clustering of Entities with FAMER.* CSIMQ 16:61–83, 2018.
- Draisbach, Christen, Naumann. *Transforming Pairwise Duplicates to Entity Clusters for High-quality Duplicate Detection.* ACM JDIQ 12(1), 2019.
- Christen, Obraczka, Hofer, Franke, Rahm. *Graph-based Active Learning for Entity Cluster Repair.* arXiv:2401.14992, 2024.
- Fu, Tang, Khan, Mehrotra, Ke, Gao. *In-context Clustering-based Entity Resolution with LLMs.* PACMMOD / SIGMOD 2026.
- Zamani, Bendersky, Metzler, Zhuang, Wang. *Stochastic Retrieval-Conditioned Reranking.* ICTIR 2022. **(§6's actual source.)**
- Elkan. *The Foundations of Cost-Sensitive Learning.* IJCAI 2001.
- Vecchio. *Predictive Value of a Single Diagnostic Test in Unselected Populations.* NEJM 274:1171–1173, 1966.
- Karpukhin et al. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020. **(the word "decomposable".)**
- Guo, Fan, Pang, Yang, Ai, Zamani, Wu, Croft, Cheng. *A Deep Look into Neural Ranking Models for Information Retrieval.* arXiv:1903.06902.
- Hu, Lu, Li, Chen. *Convolutional Neural Network Architectures for Matching Natural Language Sentences.* NIPS 2014. **(ARC-I — the §4 counterexample.)**
- Khattab, Zaharia. *ColBERT.* SIGIR 2020; Jayaram et al. *MUVERA.* arXiv:2405.19504.
- Shrivastava, Li. *Asymmetric LSH for Sublinear Time MIPS.* NIPS 2014.
- Papadakis et al. *Blocking and Filtering Techniques for Entity Resolution: A Survey.* ACM CSUR 53(2), 2020; Christen. *A Survey of Indexing Techniques…* IEEE TKDE 24(9), 2012.
- Barlaug, Gulla. *Neural Networks for Entity Matching: A Survey.* ACM TKDD 15(3), 2021. *(Defines EM as "the largest possible binary relation M ⊆ A × B" — unconstrained. Its 16-alias table contains no EL, EA, or coreference: the silo boundary, visible in the field's own synonym list.)*
- Tu, Fan, Li, Wang, Du, Jia, Gao, Tang. *Unicorn: A Unified Multi-tasking Model…* PACMMOD 1(1) Art. 84, SIGMOD 2023.
- Leone, Huber, Arora, García-Durán, West. *A Critical Re-evaluation of Neural Methods for Entity Alignment.* PVLDB 15(8):1712–1725, 2022.
- Hoffart et al. *Robust Disambiguation of Named Entities in Text.* EMNLP 2011.
- Miller, Betancourt, Zaidi, Wallach, Steorts. *Microclustering.* arXiv:1512.00792.
- Yang, Chen, Zhan, Zhao, Loy, Lin. *Learning to Cluster Faces via Confidence and Connectivity Estimation.* CVPR 2020. *(Cited only to retract erratum 7.)*

**Cited but NOT verified against the primary — do not quote without reading:**

- Chow. *On Optimum Recognition Error and Reject Tradeoff.* IEEE TIT 16(1):41–46, 1970. *(Paywalled. Corroborated by secondaries only. Prefer Fellegi–Sunter for ER's three-region rule — it is native and a year earlier.)*
- Jaro. *Advances in Record-Linkage Methodology…* JASA 84(406):414–420, 1989. *(Paywalled. Winkler & Thibaudeau RR91-9 confirms Jaro used Burkard–Derigs LSAP — but that is Winkler describing Jaro.)*
- Wang, Lin, Metzler. *A cascade ranking model for efficient ranked retrieval.* SIGIR 2011. *(Inaccessible; no claim rests on it.)*
- Bleiholder, Naumann. *Data Fusion.* ACM CSUR 41(1), 2008. *(Metadata only.)*
- Rynkiewicz et al. *Universal entity linking.* *(Inaccessible; the "universal = across KBs" reading rests on secondary summaries.)*
