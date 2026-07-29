# Is there a fixed per-score-family match threshold that beats 0.5?

> **Generated document — do not edit in place.** The prose lives in
> `docs/research/_20260728_threshold_constant.body.md`; every table below is
> printed by `examples/research/threshold_constant_sweep.py --render` and spliced
> in by `tools/render_threshold_constant_writeup.py`. Edit the body, re-run the
> tool. Three PRs on 2026-07-28 shipped factual errors and **every one was in
> hand-typed prose sitting beside a correct generated table**, which is what this
> split is defending against.
>
> **What that does and does not guarantee** — stated narrowly, because two
> earlier drafts of this note overclaimed it and review caught both. What is
> guaranteed: **every table and every count in them is generated** from the
> committed artifacts — no table is transcribed, and the constant-vs-labels
> scoreboard under the ladder is computed by the harness rather than typed.
>
> What is **not** guaranteed: the interpretive prose *quotes* individual figures
> from those tables — the shipped `0.90`, `dblp_acm`'s ≈`0.22` at the old cut,
> the four benchmarks that tie on e5, the claim that every eligible `sim_cos`
> interval sits above zero. Those are hand-typed and **can drift**. They are
> quotes from the generated tables in this same document, so a reader can check
> any of them in place — but "check it against the table" is the safeguard here,
> not "it was generated".
>
> The prose also quotes figures from *other* documents (`45 of 54` and the
> `0.174–0.695` derived-cut span, both from PR #250); each was re-read in its
> source before being repeated here.

- **Date**: 2026-07-28 (checkpoint-transfer run, §7, completed 2026-07-29)
- **Harness**: `examples/research/threshold_constant_sweep.py` (committed before the run)
- **Artifacts**: `examples/research/results/threshold_constant_sweep.json` (the
  portfolio) and `…_sweep.e5.json` (the checkpoint-transfer variant, §7)
- **Spend**: $0. Every scorer swept here is free and offline.

## 1. The question, and why it was still open

`langres` ships six front doors that hard-code a `0.5` match cut (`FuzzyString`,
`VectorLLMCascade`, and the four `architectures/retrieval.py` recipes) — though
strictly only four of them *cut* at it: `RetrieveLLM` and `RetrieveRerankLLM`
declare the number and never read it, which is its own bug and is dealt with
separately (§9). `MethodSpec.default_threshold` has existed alongside all six the
whole time — declared per method, documented, and read by **nothing**.

PR #250 (`docs/research/20260728_threshold_default.md`) measured that *deriving*
the cut from labels beats `0.5` in 45 of 54 cells, and recommended fixing the
constant next. But its §5.2 is explicit about what it did **not** do: it scored
F1 at `0.5` and at each dataset's own derived cut, and nowhere else. It never
evaluated a *shared replacement constant*. So the numbers a reader would be
tempted to reach for — the medians of those derived cuts — are a **documented
prior, not a measurement of the constant**. Shipping them would be the exact
error this repo keeps re-learning: treating your own earlier summary as a
finding.

This study measures the thing you would actually ship: **one fixed number per
score family, chosen without access to the dataset it is graded on.**

## 2. What can be measured for free, and what cannot

@@FAMILIES@@

Only two families are in scope, and the boundary is not arbitrary. `prob_llm` /
`prob_group_llm` bill a paid completion for every score, so a portfolio-wide grid
sweep over them is a real invoice, not a free one. `prob_fs` / `prob_rf` are
*fitted* matchers whose score scale is re-estimated **per dataset**, so "the best
shared out-of-the-box constant" is not even the same question for them.

> An earlier draft justified excluding both with "a user with no labels cannot
> run them at all". Review checked that against the library and it is **false for
> `prob_fs`**: `FellegiSunterMatcher.fit_unlabeled` performs an unsupervised
> random-pair u-estimate plus EM. Only `prob_rf` actually requires labels. The
> exclusion stands — a per-dataset fit is the reason — but the stated reason was
> wrong, and wrong about a public capability of this library rather than about
> this study's data.

**Those four families are left alone, and this document is not evidence about
them.**

## 3. Protocol

Each cell is one `(benchmark, method, seed)`.

1. **Split by corpus, not by pairs.** `Benchmark.split(seed=...)` assigns whole
   gold clusters to one side, so no entity and no match pair straddles the
   boundary. The label-derived cut (Youden's J) is fitted on the **train**
   corpus; every number reported here is from the **disjoint test** corpus.
2. **Score once, sweep for free.** Thresholding does not change scores, so a
   single blocking + scoring pass yields the exact F1 at all 99 grid constants,
   at `0.5`, at the derived cut, and at the oracle.
3. **The oracle is exact.** It sweeps every distinct score, not the grid — so
   "how much of the achievable gain does a constant capture?" is measured against
   the true ceiling rather than the best point of an arbitrary grid.
4. **Leave-one-benchmark-out selection.** For each held-out benchmark the
   constant is re-selected from the *other* benchmarks only, then graded on the
   held-out one. This is the number that answers "what does shipping this do on a
   dataset I have never seen?" The in-sample argmax is also printed, labeled as
   the optimistic one.
5. **Paired cluster bootstrap for the intervals.** Candidate pair rows are not
   independent — every pair touching one entity rises and falls together — so
   resampling pair rows yields intervals that are far too narrow. The resampling
   unit is the **gold cluster**; each judged pair is assigned to `min` of its two
   endpoints' units, so every pair is counted exactly once. The same resample
   grades both the candidate constant and `0.5`, and the interval is on their
   difference.

   **What that `min` does and does not buy** — an earlier draft of this section
   claimed more than it should, and review caught it. A *gold* pair has both
   endpoints in the same cluster by definition, so `min` is a no-op for it: the
   true-positive numerator and the recall denominator are attributed exactly, and
   an entity's gold pairs do move as a block. A *non-gold candidate* pair spans
   two clusters, and attributing it to one endpoint means resampling the other
   endpoint's cluster does not move it — so the model is exact on the recall side
   and approximate on the precision side, biasing intervals slightly **too
   narrow**. It is disclosed rather than fixed because the alternative (requiring
   both clusters to be drawn) makes inclusion quadratic and measures a different
   quantity, and because **neither verdict is marginal**: `sim_cos` ships on
   point estimates positive in every eligible cell, which no interval width can
   reverse, and `heuristic`'s veto — the one direction that could be sensitive —
   rests on deltas several half-widths below zero on three independent seeds and
   fails safe, since its effect is to keep the incumbent.
6. **No cross-benchmark average is ever reported as *performance*.** Every
   performance figure in this document is per-benchmark, per-seed. Cross-benchmark
   aggregation does occur, and an earlier draft wrongly said it happened in
   "exactly one place" — review checked that against the generated tables and it
   is two, both of which *decide* rather than *report*:

   - the LOBO **selection** criterion, which must reduce the other benchmarks to
     one ordering to pick a constant; and
   - the pre-registered **ship** and **transfer** rules, whose clause (3) / (2)
     is a median per-benchmark mean Δ-F1 — printed in the verdict and transfer
     tables as the quantity the rule tests.

   Both are medians over benchmarks of a mean over seeds, and both are labeled as
   decision statistics where they appear. The distinction that matters is
   decision vs. reporting, not one place vs. two.
7. **The ship rule was pre-registered** in the harness (`SHIP_RULE`) before the
   numbers were seen, and §6's verdict table is computed from the artifact by
   applying it.

Two conventions are computed for every cell, because the choice of denominator
can flip a conclusion and it should be visible if it does:

- **blocked-gold** — F1 over the gold pairs the blocker actually surfaced. This
  isolates the *threshold's* effect from the blocker's recall ceiling, and is the
  convention the headline tables use.
- **all-gold** — F1 against the full gold set, i.e. end-to-end, with the
  blocker's misses counted as false negatives. §5 repeats the comparison under
  it.

### Verification the harness is measuring what it claims

The sweep computes F1 from its own reverse-cumulative sufficient statistics
rather than by re-running the metric 99 times, which is fast but easy to get
subtly wrong. So it checks itself: `_assert_matches_classify_pairs` recomputes
three grid points per cell through the library's own `classify_pairs` and aborts
the run on any disagreement above `1e-9`. Every cell in the artifact passed it.

## 4. Results

Every cell measured, no aggregation:

@@CELLS@@

The headline: a constant chosen **without** the benchmark it is graded on.

@@LOBO@@

## 5. The same comparison, end-to-end

If the blocked-gold and all-gold conventions disagreed about whether a constant
helps, that disagreement would be the finding. They are reported side by side so
nobody has to take the denominator on trust:

@@ALLGOLD@@

## 6. Is it actually *one* constant?

This is the question that decides whether a "default" is a default or a
per-dataset tuning wearing a constant's clothes:

@@STABILITY@@

And the decision framed as a ladder — what `0.5` gets you, what a free shipped
constant gets you, what labels get you, and what is left on the table after all
three:

@@LADDER@@

## 7. Does a `sim_cos` constant even mean anything? (the checkpoint test)

A `heuristic` cut is a cut on rapidfuzz's ratio, which is fixed. A `sim_cos` cut
is a cut on a **cosine scale**, and the scale belongs to the *encoder*, not to
the family tag — two models can both emit "cosine similarity" and disagree about
what `0.9` means.

This matters concretely here, and it is easy to miss: every benchmark loader in
this repo pins `all-MiniLM-L6-v2` for its blocker, but `DEFAULT_EMBEDDING_MODEL`
moved to `intfloat/e5-base-v2` (the embedder-ladder PR). So the constant §4
selects is a constant **for MiniLM cosine**, while the number it would be written
into is read by users running e5. Assuming it transfers is exactly the
"documented prior treated as a measured finding" error this study was written to
avoid, so it is measured instead: the identical protocol, re-run with the
blocker re-pointed at the shipped default (`--embedder`), and the baseline's
constant graded on those cells.

@@TRANSFER@@

### The same test in the other direction

Run one way only, a transfer test is a cherry-pick. "Does *my* constant survive
elsewhere?" is the flattering question; the one that actually constrains the
choice is "would the *other* checkpoint's constant have been safe **here**?" The
e5 run selects its own constant by the same leave-one-out procedure, so that
question is free to answer — and it is the one that decides what ships:

@@TRANSFER_REVERSE@@

### Reading both directions together

The two runs agree on the finding that matters and disagree on the number, and
both halves are load-bearing:

- **They agree that `0.5` is never the better choice for a cosine.** On *both*
  checkpoints `0.90` is **never worse** than `0.5`, and on the pinned checkpoint
  it is dramatically better. Two drafts of this sentence overclaimed and review
  caught both, so it is worth stating exactly what the tables show:
  - It is not "beaten on every eligible benchmark". On the e5 run, `F1@0.90`
    **exactly equals** `F1@0.50` for every seed of `dblp_acm`, `fodors_zagat`,
    `walmart_amazon` and `wdc_computers` — those are **ties, not wins**, because
    on that encoder almost every candidate pair already sits above `0.90`, so the
    cut selects the same set.
  - Nor does `0.5` always leave F1 "in the low hundredths": `dblp_acm` sits near
    0.22 there and the 12-record `tiny_fixture` at 0.50.

  What survives both corrections is the claim the decision actually rests on:
  never worse anywhere, and enormously better where the cut bites.
- **They disagree about how high.** Each checkpoint's own leave-one-out
  selection is internally rock-solid — and they land in different places. That is
  not noise between two unstable estimates; it is two stable estimates of two
  different scales, which is precisely what "the cosine scale belongs to the
  encoder" predicts.
- **The asymmetry is the verdict.** The lower constant is safe on both
  checkpoints. The higher one is not: carried back, it lands **significantly
  below `0.5` on `abt_buy`, on every seed, with intervals entirely below zero** —
  the same clause-(2) failure that disqualifies `heuristic` in §8. So the higher
  constant is not "the better number measured on the more relevant checkpoint";
  it is a number with a demonstrated victim.
- **What the safe constant is worth is honestly asymmetric.** On the pinned
  checkpoint it is transformative. On the shipped default it is *safe but close
  to inert* — several benchmarks score identically to `0.5`, because on that
  encoder almost every candidate pair already sits above the cut. Its Δ there is
  positive and never negative, which is what the pre-registered rule asks, but
  the honest word for the magnitude is "small", and the table says so rather
  than the prose.

That combination — a real floor, safe everywhere, worth a great deal on one
scale and little on another — is exactly what a *default* should be, and exactly
why it is not a substitute for tuning. A shipped constant's job is to stop the
out-of-the-box path being catastrophically wrong. It is not to be optimal on
your encoder, and this table is the evidence that it cannot be.

**What "checkpoint" identifies here, precisely.** Both artifacts record their
encoder by *name* — `all-MiniLM-L6-v2` and `intfloat/e5-base-v2` — and neither
pins a Hub revision, so each names a **branch as it resolved on the measurement
date**, not a commit. Raised in review, and worth stating rather than leaving to
the reader: a name is not an identity. (An aside outside this study's
measurements, but the reason the point is not academic: the Hub cache on the
machine that ran this holds three distinct `all-MiniLM-L6-v2` snapshots.
Whatever fetched them, the name alone does not say which one a given run
loaded.) The harness now accepts `--embedder repo@revision` and forwards it to
`SentenceTransformer` as a real pin, so a later run can be exact; these two were
measured before it could, and that is the one dimension of this study that is
reproducible by convention rather than by construction: an exact re-run of these
numbers is not guaranteed by the command in §10 alone. What that does and does
not put at risk is worth separating. The shipped decision rests on `0.90` being
**never worse than `0.5`** on both encoders, and that was verified on the
weights these runs actually loaded; whether it survives a *different* revision
of either model is untested here, which is the same open question as any other
untested encoder — and §7's whole point is that the answer is a property of the
scale, not of the family tag.

## 8. Verdict

@@VERDICT@@

### Reading the `sim_cos` row: the incumbent was the bug

For cosine similarity the result is not marginal. In the leave-one-out table
(§4), **every selection-eligible cell improves, and every 95% interval sits
entirely above zero** — there is no benchmark, and no seed, where the constant
loses. The `capture` column shows it recovers most of the headroom an *oracle
with test labels* could reach.

The reason is visible in the `F1@0.50` column: `0.5` is not a mildly
mistuned cut on a cosine scale, it is a catastrophic one. Normalized embeddings
put almost every candidate pair above `0.5`, so the threshold accepts nearly
everything the blocker proposed and precision collapses. Several benchmarks score
in the low hundredths. This is less "we found a better constant" than "the shipped
constant was never on the right scale", which is exactly the failure mode a
per-family default exists to prevent.

That is also why this is the family where wiring the default mattered most: the
same `0.5` is defensible for a `heuristic` score and indefensible for a cosine,
and a single global constant cannot be both.

**What ships is the *safe* constant, not the best one.** §7 measured a second
checkpoint and the two directions are not symmetric: the constant selected here
is harmless on the other encoder, while that encoder's own (higher) constant
predictably damages `abt_buy` when carried back. Where the two disagree this
study ships the one with no demonstrated victim, which is the same standard
clause (2) applies to `heuristic` — applied to our own preferred answer rather
than only to the one we were ready to reject.

### Reading the `heuristic` row: it is not instability

The obvious guess — that rapidfuzz has no single constant because every dataset
wants a different one — is **wrong here, and the table says so**. Its
leave-one-out constants agree closely (§6): drop any one dataset and the choice
barely moves. PR #250's derived cuts spanned 0.174–0.695, which made
non-generalization the expected outcome; the *fixed*-constant question turns out
to have a different answer than the *derived*-cut question did.

What kills it is clause (2), not clause (1). One constant, chosen without seeing
the dataset it is graded on, **helps on most of the portfolio and reliably
damages `abt_buy`** — on every seed, with the 95% interval entirely below zero.
That is not noise and it is not a tie; it is a predictable loss for a whole class
of data (short, noisy product titles where a high string-similarity bar
throws away true matches faster than it removes false ones).

**The conventions agree, so this does not rest on the denominator.** The
end-to-end (all-gold) table in §5 shows `abt_buy` losing by the same margins with
intervals just as clearly below zero, and `febrl_dedup`'s mild losses harden
there rather than softening. If anything the end-to-end view is *less* favourable
to shipping a constant than the blocked-gold view the verdict is computed on.

A default that improves the median while predictably harming a known dataset
class is not a default — it is a recommendation with an undisclosed victim.
Clause (2) was pre-registered precisely so this case could not be argued away
after the fact by pointing at a healthy median. So `heuristic` keeps `0.5`, and
the honest guidance for it is unchanged: **derive the cut from labels** (PR
#250's seam, `derive_threshold`).

**But read the ladder in §6 carefully before concluding that labels simply
dominate — they do not, and an earlier draft of this sentence said they did.**
Compare its `F1@LOBO` and `F1@derived` columns — and read the count generated
directly beneath that table rather than one typed here, because the *rejected*
constant scores higher than the label-derived cut on a **majority** of
benchmarks, for both families. Neither approach dominates this portfolio. What makes deriving the
right advice here is not that it wins on average — it is *where* it wins: on
`abt_buy`, the single benchmark whose harm vetoes the constant, the derived cut
beats both the constant and `0.5`. Labels buy you the case a constant cannot
cover, which is a different and more useful claim than "labels are better".

Note also what the ladder shows about the incumbent: on the harder product
benchmarks `0.5` is not a mild compromise, it is close to worthless. The reason
to leave it in place is that nothing free and *safe* beats it, not that it is
good.

## 9. What was measured vs. what was inferred

**Measured** (this artifact, held-out, multi-seed, with intervals):

- Per-cell F1 at all 99 grid constants, at `0.5`, at the label-derived cut, and
  at the exact oracle, for the `heuristic` and `sim_cos` families across the
  loadable portfolio.
- The LOBO-selected constant's Δ-F1 against `0.5` per benchmark and seed, with
  95% paired cluster-bootstrap intervals.
- The stability of that selection under dropping one dataset.
- Whether a `sim_cos` constant survives a change of embedding checkpoint (§7),
  measured on `intfloat/e5-base-v2` — the library's own default — under the same
  protocol, **in both directions**: this study's constant graded on that
  checkpoint's cells, and that checkpoint's own leave-one-out constant graded
  back on these.

**Inferred, not measured** — flagged because it would be easy to read this
document as covering it:

- **`prob_llm`'s `0.7`.** `0.7` has been the declared LLM-family value since the
  field was introduced, and `VectorLLMCascade`'s out-of-the-box cut moves from
  `0.5` to `0.7` to match it. **That number is not measured here** — it is the
  value the codebase already declared, now honoured instead of ignored. Sweeping
  it costs paid completions on every benchmark and is a separate study.

  Be precise about the mechanism, because an earlier draft was not: the
  architecture does **not** read the registry. It calls
  `resolve_threshold(threshold, "prob_llm")` directly against
  `DEFAULT_THRESHOLDS`, and `MethodSpec.default_threshold` still has no runtime
  reader at all — a front door knows its score *family*, not a method *name*.
  What the change buys is that the registry field and the shipped behaviour are
  now derived from the same map, so they can no longer silently disagree.
- **`prob_fs` / `prob_rf`.** Untouched, for the reason in §2.
- **The emitted `score_type` tags do not match the families used to resolve the
  defaults**, and review caught this study leaning on the tag as if it did.
  `RetrieveRerank` cuts a *cross-encoder's* score but is tagged `heuristic`
  (`Rerank`'s `out_space` default). Worse for this document: the `Retrieve` op
  stamps its rows `heuristic` too (`resources/retrieve.py:174`, hard-coded), even
  though they are cosines and `Retrieve` resolves its default against `sim_cos`.
  The constant is justified by **what the score is** — these really are embedding
  cosines, which is what was swept — not by the tag, and an earlier draft of the
  `Retrieve` docstring wrongly claimed the op emitted `sim_cos`.

  So one coarse tag currently covers three different scales (rapidfuzz ratios,
  cross-encoder scores, cosines). Nothing here justifies a *cross-encoder*
  constant, and none was set. Fixing the tags means changing a resource's emitted
  contract, which is a separate change from a threshold default; it is recorded
  here and in both class docstrings rather than quietly worked around.

## 10. Reproducing

```bash
OMP_NUM_THREADS=1 uv run --env-file .env python \
    examples/research/threshold_constant_sweep.py
```

`OMP_NUM_THREADS=1` is not optional on macOS: faiss plus a torch model in one
process deadlock silently without it — no error, no output, no CPU, forever.
(`KMP_DUPLICATE_LIB_OK` suppresses the OpenMP *abort*, not the *deadlock*.) The
module sets it via `os.environ.setdefault` before any import that could pull
either in, so a run is protected however it was launched.

Each benchmark runs in its own subprocess to bound torch's non-releasing MPS
allocator, and every benchmark is checkpointed — an interrupted sweep continues
with `--resume`. To re-print every table from the committed artifact without
measuring anything:

```bash
uv run --env-file .env python examples/research/threshold_constant_sweep.py \
    --render examples/research/results/threshold_constant_sweep.json
```

The §7 checkpoint variant, and the two transfer directions it feeds:

```bash
OMP_NUM_THREADS=1 uv run --env-file .env python \
    examples/research/threshold_constant_sweep.py \
    --methods embedding_cosine --embedder intfloat/e5-base-v2 \
    --out examples/research/results/threshold_constant_sweep.e5.json

# --compare BASELINE VARIANT: select on BASELINE, grade on VARIANT's cells.
uv run --env-file .env python examples/research/threshold_constant_sweep.py \
    --compare examples/research/results/threshold_constant_sweep.json \
              examples/research/results/threshold_constant_sweep.e5.json
```

To regenerate this whole document — every table above — from the committed
artifacts:

```bash
uv run --env-file .env python tools/render_threshold_constant_writeup.py
```
