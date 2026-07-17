# Theory: the mathematical foundation of langres

> **Status:** foundational design document. Descriptive of where langres is
> heading, **not** of what it ships today. Where the theory contradicts current
> code, this document says so explicitly rather than pretending agreement — see
> [§8](#8-what-this-contradicts-in-the-current-code).
>
> **Purpose.** langres's components (`Blocker`, `Matcher`, `Clusterer`) are named
> after *positions in a pipeline*. This document shows that position is not a
> type: blocking, matching, reranking, top-k, three-region routing, assignment
> and clustering are **one operation** under different parameters. The point is
> not elegance. It is that a formalization is *falsifiable* — it lets us check
> whether one set of contracts spans every use case we care about, and it has
> already caught errors that prose review did not.

---

## 1. Notation

Everything below uses these symbols and nothing else.

| Symbol | Name | Meaning |
|---|---|---|
| $A$, $B$ | the two sides | Record sets being matched. For dedupe, $A = B$. |
| $E$ | the pairs | $E \subseteq A \times B$. The rows of the table. |
| $s$ | the score | $s : E \to S$. The score column. |
| $S$ | the score space | $\mathbb{R}$ (cosine), $\{0,1\}$ (a yes/no verdict), $\mathbb{R}^d$ (a feature vector). **Need not be a range.** |
| $R$ | scored relation | The pair $(E, s)$ — rows plus scores. **The only type that flows.** |
| $t$ | the price | What it costs to keep a row. A row earns its place only if its score beats $t$. |
| $\mathcal{F}$ | the feasible class | A rule on the shape of the **whole answer**, not on any single row. |
| $T$ | true pairs | The number of genuinely matching pairs in the data. |
| $\eta$ | matcher sensitivity | $P(\text{matcher says yes} \mid \text{pair is true})$. |
| $\varphi$ | matcher fall-out | $P(\text{matcher says yes} \mid \text{pair is false})$. The false-positive rate. |

---

## 2. The carrier and the two operations

**The carrier.** At every point in an ER system you hold exactly one thing: a
table of pairs with a number on each.

$$R = (E, s), \qquad E \subseteq A \times B, \qquad s : E \to S$$

Every stage maps a scored relation to a scored relation. Because the carrier
type never changes, **the composite of two stages is itself a stage** — the
property that makes `nn.Module` nest indefinitely, obtained here as a
consequence rather than a convention.

**Operation 1 — score.** Keep the rows, replace the numbers.

$$\sigma_f (E, s) = (E, f)$$

Every scorer is this: an embedding cosine, a string ratio, an LLM judge.

**Operation 2 — select.** Keep the numbers, delete rows.

$$\pi_{\mathcal{F}, t}(E, s) \;=\; \underset{E' \subseteq E,\; E' \in \mathcal{F}}{\arg\max} \;\; \sum_{(a,b) \in E'} \big( s(a,b) - t \big)$$

In words: *out of every legally-shaped way to keep some rows, take the one whose
scores add up highest after each row pays the price $t$.*

**Operation 3 — combine.** The lattice on relations: $R_1 \cup R_2$,
$R_1 \cap R_2$, $R_1 \setminus R_2$. Already shipped as
`CompositeBlocker(op="union"|"intersection"|"difference")` — but scoped to
blockers only.

**Why $t$ is not a separate concept.** With $\mathcal{F}$ unconstrained,
maximizing $\sum (s - t)$ means exactly *keep every row where $s > t$*. The
threshold is not a rule you add. It is what the formula does when nothing else
constrains it.

---

## 3. One operation, seven names

Every component langres has, and every one it lacks, is $\pi$ at a different
$\mathcal{F}$. Identical mathematics; the algorithms differ wildly because
$\mathcal{F}$ is what makes a problem linear or NP-hard.

| Name | $\mathcal{F}$ | $t$ | Algorithm forced | Cost |
|---|---|---|---|---|
| all-pairs blocking | no rule | $-\infty$ | enumerate | $O(n^2)$ |
| top-k blocking | $\deg_L(a) \le k$ | $-\infty$ | index lookup | $o(n^2)$ † |
| threshold matching | no rule | $t$ | compare | $O(\lvert E\rvert)$ |
| three-region / cascade | 3-way split of $E$ | $t_{lo}, t_{hi}$ | compare ×2 | $O(\lvert E\rvert)$ |
| mention → KB (+NIL) | $\deg_L(a) \le 1$ | $t$, else NIL | argmax per anchor | $O(\lvert E\rvert)$ |
| 1-to-1 assignment | $\deg_L \le 1 \wedge \deg_R \le 1$ | $t$ | Hungarian / LSAP | $O(n^3)$ |
| clustering | consistency: $a\equiv b \wedge b \equiv c \Rightarrow a \equiv c$ | $t$ | NP-hard → approximate | — |

† sublinear **only if $\sigma$ is decomposable** (§4). That condition is the
entire reason blocking exists.

**Consequence.** A blocker is a scorer with a *permissive* $\mathcal{F}$ and
$t \approx -\infty$. A matcher is a scorer with a *strict* $t$. They are not
different kinds of thing; they are the same operation at different operating
points. A reranker is simply a second $\sigma$ over an existing $E$.

---

## 4. Decomposability decides what can be retrieved

A scorer is **decomposable** iff a per-record function exists:

$$s(a,b) = \oplus\big(g(a),\, h(b)\big)$$

That is the whole condition. It says nothing about embeddings, and nothing about
the score space. **$\oplus$ decides which index is possible:**

| $g$ reduces a record to | $\oplus$ | $S$ | index | in langres |
|---|---|---|---|---|
| a normalized key | equality | $\{0,1\}$ | hash bucket | `KeyBlocker` ✓ |
| a dense vector | cosine | $\mathbb{R}$ | ANN | `VectorBlocker` ✓ |
| a shingle set | Jaccard | $\mathbb{R}$ | MinHash / LSH | absent |
| — *nothing; must see both* | — | any | **none possible** | `LLMMatcher` |

If $\sigma$ is decomposable, $g$ can be precomputed over the corpus and indexed,
so $\pi$ at $\mathcal{F} = \{\deg \le k\}$ is computable in $o(\lvert A\rvert \cdot \lvert B\rvert)$.
**That is retrieval.** If $\sigma$ is joint, nothing can be precomputed and $\pi$
must be handed $E$ by someone else — which is why an LLM can never block.

**This single bit is the entire blocker/matcher distinction.** It is a property
of the scoring function, not a position in a pipeline.

**Corollary — blocking can vanish.** In the architecture
$A \times A \xrightarrow{\sigma_{bi}} \pi(\mathcal{F} = \text{consistent},\, t = 0.8)$
there is no blocking stage. Because $\sigma_{bi}$ is decomposable, the runtime
builds an index and evaluates $\pi$ sublinearly. **Blocking is an execution
strategy the framework derives, not a component the user wires.** Step count is
not architecture; it is a consequence of decomposability and $\mathcal{F}$.

---

## 5. The score space must be rich enough for $\mathcal{F}$

$S$ need not be a range. If $s \in \{0,1\}$ and $0 < t < 1$, then maximizing
$\sum (s-t)$ means exactly "keep the yeses" — the formula degenerates correctly.
`KeyBlocker` is this case: it buckets by a normalized key and emits candidates
carrying **no score at all**.

But $\pi$ needs a total order, and a two-valued score gives massive ties:

| $\mathcal{F}$ | boolean $S$ | why |
|---|---|---|
| no rule (threshold) | ✓ works | "keep the yeses" |
| consistency (clustering) | ✓ works | this is the original $\pm 1$ correlation-clustering formulation |
| $\deg \le k$ (top-k) | ✗ **degenerate** | if 40 pairs all say yes, *which 25*? Arbitrary. |
| assignment | ✗ **degenerate** | any perfect matching is equally optimal |

langres's existing `PairwiseJudgement` split — `decision: bool \| None` (a
decider) vs `score: float \| None` (a ranker) — already encodes this distinction.
The algebra states its implication: **a decider cannot drive top-k or
assignment**, and nothing currently prevents wiring one that way. This is a
checkable wiring rule.

Note also that $S = \mathbb{R}^d$ (a `ComparisonVector`) is **not** orderable, so
$\pi$ on a vector-valued relation is a type error; a scalarizer must sit between.
That is exactly what `Comparator` → `WeightedAverageMatcher` is: $\sigma$ into
$\mathbb{R}^d$, then $\sigma$ from $\mathbb{R}^d$ to $\mathbb{R}$. The split is
principled; it is simply not typed. Today it is a runtime raise.

---

## 6. Staging: one theorem, and a corollary that is false

This section corrects an error that survived two rounds of review.

### 6.1 What is a theorem

Because $\pi$ **only ever deletes rows**, for any pipeline
$E_1 \supseteq E_2 \supseteq \dots \supseteq E_n$:

$$\text{recall}(E_1) \;\ge\; \text{recall}(E_2) \;\ge\; \dots \;\ge\; \text{recall}(E_n)$$

A true pair deleted at stage $k$ cannot exist at stage $k+1$. Recall is
monotonically non-increasing, so **final recall $\le$ blocking recall**.
Omissions are permanent; commissions are recoverable (a later stage can always
delete a false pair). This is a hard ceiling and it is not in dispute.

### 6.2 What is not a theorem

> ~~"Therefore early stages must maximize recall."~~

**This does not follow, and it is false in general.** A ceiling constrains the
*achievable set*; it says nothing about where the *optimum* lies. Conflating
"the ceiling rises with $r_B$" with "the optimum rises with $r_B$" is a
non-sequitur.

**Derivation.** Let $r_B$ = blocking recall, $T_B = r_B T$ the true pairs
surviving blocking, and $F_B = \lvert E_B\rvert - T_B$ the false candidates
surviving. Model the matcher per-pair with constant $\eta, \varphi$:

$$\text{TP} = \eta \, r_B \, T, \qquad \text{FP} = \varphi \, F_B$$

$$R = \eta \, r_B \qquad\qquad P = \frac{\eta \, r_B \, T}{\eta \, r_B \, T + \varphi \, F_B} = \frac{1}{1 + \dfrac{\varphi F_B}{\eta \, r_B \, T}}$$

Parameterize the blocker by $k$ (top-$k$ over $n$ anchors), so
$\lvert E_B \rvert \approx nk$ and $F_B \approx nk - r_B T$. As $k \to \infty$:

- $r_B \to 1$ and **saturates** (it is bounded by 1), hence $R \to \eta$, a constant;
- $F_B \to \infty$ **without bound**;
- therefore $P \to 0$, and $F_1 = \frac{2PR}{P+R} \to 0$.

$F_1(k)$ is strictly positive at moderate $k$ and tends to $0$ as $k \to \infty$.
Therefore:

> **Result.** For any $\varphi > 0$, $F_1$ attains an interior maximum at finite
> $k$. **The optimal blocking recall is strictly less than 1.**

### 6.3 The boundary case is the doctrine

Set $\varphi = 0$. Then $P = 1$ for any $F_B$, so $F_1 = \frac{2R}{1+R}$ is
monotone increasing in $R = \eta \, r_B$ — and you should indeed maximize
blocking recall.

> **"Maximize blocking recall" is the $\varphi \to 0$ limit of the real rule.**
> It is a corollary with a hypothesis. The hypothesis — *the matcher makes
> essentially no false positives* — is almost never true and almost never stated.

The result is robust to the objective: for **any** objective penalizing false
positives ($C_{FP} > 0$), the interior optimum exists. Only $C_{FP} = 0$ ("recall
is all I care about") or $\varphi = 0$ ("my matcher is perfect") recovers the
doctrine.

### 6.4 Two distinct effects — the composition effect needs no degradation

It is tempting to explain §6.2 by saying the matcher *degrades* on larger
candidate sets. It is worth separating two mechanisms:

1. **Composition effect** — *always present, requires no degradation whatsoever.*
   With $\eta, \varphi$ held **constant**, $P$ still collapses, because $P$
   depends on the composition of $E_B$, not on the matcher's quality. Loosening a
   blocker admits candidates that are overwhelmingly false: true pairs are $O(n)$,
   all pairs are $O(n^2)$. This alone proves the result.
2. **Degradation effect** — $\varphi$ itself grows with $\lvert E_B \rvert$ or
   group size. This does **not** apply to a pairwise matcher, where $\varphi$ is a
   property of the model, not of the batch. It **does** apply to set-wise matchers
   (`GroupwiseMatcher`, a ComEM-style `SelectMatcher`): an LLM choosing among 100
   candidates does worse than among 10. There the two effects compound.

### 6.5 The consequence: the stages are coupled

The blocker's optimal operating point is a function of $\varphi$ — a property of
the **matcher**. Therefore:

> **The blocker cannot be tuned in isolation.** Any procedure that selects a
> blocking parameter without reference to the downstream matcher is optimizing a
> quantity that is not the objective.

This is the formal argument for joint optimization over the topology, and it is a
derivation rather than a preference. It also gives the crossover to a
single-model architecture as a *measurement*: a two-stage pipeline is justified
only while the second scorer's precision gain exceeds the recall its blocker
forfeits at the ceiling. Every term is measurable with the existing harness.

**Correction of a common framing.** One does not "maximize precision" — that has
a trivial winner (return nothing: $P = 1$, $R = 0$). One maximizes a full
objective, and $t$ **is** the precision/recall dial inside it. Blocking and
matching are the same formula at different $t$: blocking sets $t \approx -\infty$
(never delete a true pair), matching sets $t$ high (delete anything uncertain).
Hence the one-sentence description of every ER pipeline ever built: **use a cheap
score to decide which pairs are worth paying the expensive score for.**

---

## 7. Unobserved pairs: what transitive closure assumes

$\pi$ as defined in §2 requires $E' \subseteq E$ — it may only *delete* rows.
**Transitive closure adds rows**: given selected edges $E'$, it returns
$\mathrm{cl}(E')$, and every edge in $\mathrm{cl}(E') \setminus E'$ was **never
scored**. So closure is not a $\pi$, and §2's formula as literally written cannot
express it.

The repair is illuminating. Clustering's $\pi$ ranges over equivalence relations
on $A \times A$, where $s$ is **partial** — undefined on pairs the blocker never
generated. What you assume about $s$ on unobserved pairs *is* the algorithm:

| assume $s(a,c) =$ for unscored implied pairs | you get |
|---|---|
| $+\infty$ | **transitive closure** |
| $0$ (neutral) | correlation clustering over observed edges |
| the prior (negative; base rate is $O(1/n)$) | the Bayesian-correct choice |
| *go and measure it* | see below |

> **Transitive closure assigns $+\infty$ to every edge it invents and never
> scored.** Stated plainly, that is an indefensible prior — and it is exactly why
> one bad edge welds two entities together permanently.

This explains an empirical finding rather than restating it. Every clustering
room surveyed engineers *around* closure (face clustering collapses to 0.37 at
5.21M records). That is not five independent discoveries; it is one algorithm
being wrong in five places, and §7 predicts it.

**Two classes of invented edge**, and they differ in how bad they are:

1. **Never a candidate** — the blocker never generated $(a,c)$. Closure asserts
   the merge with *zero* evidence.
2. **Was a candidate, scored below $t$** — closure **overrides a measurement it
   already has.**

Class 2 is detectable for **$0** — those scores are already in the
`JudgementLog`. The diagnostic is: *does this cluster contain a pair we scored
below threshold?* If so, closure overrode our own evidence.

**Note the interaction with `CorrelationClusterer`.** It is designed to catch
exactly this, but `_build_adjacency` calls `predicted_match(judgement, threshold)`
and `continue`s past every sub-threshold pair before weighting. Real correlation
clustering (Bansal–Blum–Chawla; the Ailon–Charikar–Newman pivot) runs on the
**signed** graph where every pair carries $s - t$, positive *and negative*.
langres discards the negative evidence before the aggregator can see it — a live
accuracy cost, and a direct consequence of decide-then-aggregate being two steps
instead of one.

**The cheap fix, in the algebra:** score $\mathrm{cl}(E') \setminus E'$ — exactly
the edges closure invented. The set is small, precisely targeted at the
unjustified assumption, and **pairwise**, so it needs no set-level score and stays
in phase 1.

---

## 8. Phase 2: the algebra over clusters

Two operations do not fit §2, and they fit each other:

- **Cluster verification** — "are these four really one entity?" scores a *set*,
  not a pair. There is no term for it in §2.
- **Merge / fusion** — every operation above has signature *table → table*.
  Fusion is *table → records*: it creates a record that was never in $A$ or $B$
  (name from one, address from another). It is not exotic; **it is the exit.**

Both take a cluster, both sit at the tail, neither is pairwise. That is not two
exceptions — it is a **second phase with its own two operations**, mirroring the
first:

```
phase 1 — carrier: a table of scored PAIRS
  σ  rescore              (embedding, string, LLM)
  π  select               (block, threshold, top-k, assign, cluster)
  ∪ ∩ ∖  combine

phase 2 — carrier: CLUSTERS
  verify  cluster → clusters     (split what closure over-merged)
  fuse    cluster → record       (the exit; `Canonicalizer`)
```

This also explains the survey's oddest finding — five fields each named fusion
separately, and ER's standard textbook has no chapter for it. **It was never a
pairwise operation, so the pairwise literature had nowhere to put it.**

**One fork worth naming.** Merging *at the end* (cluster, then fuse each cluster
once) is terminal and harmless. Merging *during* — so that record $C$ is compared
against a merged $AB$ rather than against $A$ and $B$ — means merging has changed
the data later scores see. That is the loop case (§9), and it is precisely the
difference between `dedupe()` (batch) and `stream_against()` (incremental,
currently a `NotImplementedError` stub). **Same word, two architectures.**

---

## 9. What the algebra does not cover

**Collective / incremental resolution is a fixpoint, not a fold.** If two papers
matching raises the score that their authors match — which then raises the score
for other papers by those authors — then $s$ depends on the answer and the answer
depends on $s$. You iterate to convergence. A DAG cannot express this; it needs an
`iterate(stage, until=…)` combinator.

This is a declared V1.1 deferral (`docs/USE_CASES.md`), so there is nothing to
build. The requirement on the design is weaker and important: **do not adopt a
shape that forbids it.**

**Coverage as of this writing: 11 of 12 tested use cases.** Pairwise decision,
clustering, 1-to-1 assignment, mention→KB, cascade, set-wise, disjunctive
blocking, exact-key blocking, cross-modal, cluster verification, and merge/fusion
all express. Only the loop breaks.

---

## 10. What this contradicts in the current code

Listed rather than silently fixed; each is a code change with its own blast
radius and should be made deliberately.

| Location | States | Status under §6 |
|---|---|---|
| `core/blocker.py:38` | "High recall: Blocking should have ≥95% recall (don't miss true matches)" | The $\varphi = 0$ rule stated as an unconditional design principle. |
| `core/blockers/vector.py:202` | "k_neighbors should be tuned to achieve >= 95% recall" | Same. |
| `RecallCurve.optimal_k(target_recall=…)` | names a recall-hitting $k$ "optimal" | Optimizes a quantity containing no $\varphi$; the true optimum depends on the matcher. |
| `core/clusterers/correlation.py` | thresholds via `predicted_match` before weighting | Discards the negative evidence the pivot needs (§7). |
| `core/clusterer.py` | `Clusterer(threshold=0.5)` | $t$ is a parameter of $\pi$, not of aggregation (§2). |

In fairness to the optimizers: `BlockerOptimizer` accepts an arbitrary
user-supplied objective and `primary_metric`, and `core/autoresearch/objective.py`
already supports Pareto goals and constraints. **The doctrine lives in the
guidance and in `optimal_k`, not in the optimizer machinery.**

---

## 11. Consequences for the contracts

The algebra implies five contracts across two phases:

| Phase | Contract | Signature | Declares | Replaces |
|---|---|---|---|---|
| 1 | `Encoder` | record → $R$ | — | the encoder trapped inside `VectorBlocker` |
| 1 | `Scorer` ($\sigma$) | $(a,b) \to S$ | `decomposable`, $\oplus$ | `Blocker` + `Matcher` + reranker + `Comparator` |
| 1 | `Select` ($\pi$) | $(E,s) \to E' \subseteq E$ | $\mathcal{F}$, $t$, `scope` | three separate thresholds + top-k + `Clusterer` + abstention |
| 2 | `Verify` | cluster → clusters | — | nothing — new |
| 2 | `Fuse` | cluster → record | — | `Canonicalizer` |

Plus the lattice over phase 1's carrier (shipped, scoped to blockers) and a
reserved `iterate` for §9.

**Two declarations, one shape.** `Scorer.decomposable` says whether the *inputs*
are independent (→ can we retrieve?). `Select.scope` — `pair` / `group` /
`global` — says whether the *decisions* are independent (→ can we stream?).
Both are statements about independence structure; together they let the framework
derive execution instead of encoding it as special cases in a factory.

**The one non-negotiable at implementation time:** `Select` must subsume the
`Clusterer` **on day one**. A `Decision` object that sits *next to* a `Clusterer`
rebuilds the same weld in a new box — and §7 shows that weld already costs
accuracy today.

---

## References

- Fellegi, I. P. & Sunter, A. B. (1969). *A Theory for Record Linkage.* JASA.
  Error-bounded three-region decisions — the $\mathcal{F}$ = 3-way-split row of §3.
- Bansal, N., Blum, A. & Chawla, S. (2004). *Correlation Clustering.* Machine Learning.
  The $\pm 1$ signed-graph formulation referenced in §5 and §7.
- Ailon, N., Charikar, M. & Newman, A. (2008). *Aggregating inconsistent information:
  ranking and clustering.* JACM. The pivot algorithm of §7.

**Companion analyses** (not in-repo): the contract audit of the seven misfiled
concepts, and the nine-rooms field survey that supplies the six task shapes and
the licensing gate.
