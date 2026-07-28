# Is there a fixed per-score-family match threshold that beats 0.5?

> **Generated document — do not edit in place.** The prose lives in
> `docs/research/_20260728_threshold_constant.body.md`; every table below is
> printed by `examples/research/threshold_constant_sweep.py --render` and spliced
> in by `tools/render_threshold_constant_writeup.py`. Edit the body, re-run the
> tool. Three PRs on 2026-07-28 shipped factual errors and **every one was in
> hand-typed prose sitting beside a correct generated table**, which is why
> **no number produced by this study is typed by hand anywhere in this
> document** — every one arrives in a generated table. The prose does quote
> figures from *other* documents (`45 of 54` and the `0.174–0.695` derived-cut
> span, both from PR #250); each was re-read in its source before being repeated
> here.

- **Date**: 2026-07-28 (checkpoint-transfer run, §7, completed 2026-07-29)
- **Harness**: `examples/research/threshold_constant_sweep.py` (committed before the run)
- **Artifacts**: `examples/research/results/threshold_constant_sweep.json` (the
  portfolio) and `…_sweep.e5.json` (the checkpoint-transfer variant, §7)
- **Spend**: $0. Every scorer swept here is free and offline.

## 1. The question, and why it was still open

`langres` ships six front doors that cut matches at a hard-coded `0.5`
(`FuzzyString`, `VectorLLMCascade`, and the four `architectures/retrieval.py`
recipes). `MethodSpec.default_threshold` has existed alongside them the whole
time — declared per method, documented, and read by **nothing**.

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
*fitted* matchers: a user with no labels cannot run them at all, so "the best
out-of-the-box constant" is not even the same question for them. **Those four
families are left alone, and this document is not evidence about them.**

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
6. **No averaging across benchmarks in any reported number.** Aggregation appears
   in exactly one place — the LOBO *selection* criterion, which has to reduce the
   other benchmarks to a single ordering — and it is a median over benchmarks of
   a mean over seeds, labeled a selection statistic and never reported as
   performance.
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

- **They agree that `0.5` is the wrong order of magnitude for a cosine.** On
  *both* checkpoints, on every benchmark, cutting at `0.5` leaves F1 in the low
  hundredths. Whatever the right constant is, it is nowhere near `0.5`.
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
#250's seam, `derive_threshold`), which the ladder in §6 shows is worth
substantially more than any constant anyway.

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

- **`prob_llm`'s `0.7`.** The registry has declared `0.7` for the LLM families
  since the field was introduced; wiring `threshold=None` to the registry makes
  `VectorLLMCascade` actually *use* it, changing its out-of-the-box cut from
  `0.5` to `0.7`. **That number is not measured here** — it is the value the
  codebase already declared, now honoured instead of ignored. Sweeping it costs
  paid completions on every benchmark and is a separate study.
- **`prob_fs` / `prob_rf`.** Untouched, for the reason in §2.
- **`RetrieveRerank`'s family tag.** It cuts a *cross-encoder's* score but is
  tagged `heuristic` because `Rerank`'s `out_space` defaults to it. That is a
  coarse tag covering two different scales; a constant justified on rapidfuzz
  scores is not thereby justified on cross-encoder scores. Called out in the
  class docstring.

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
