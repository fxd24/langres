# Is there a fixed per-score-family match threshold that beats 0.5?

> **Generated document — do not edit in place.** The prose lives in
> `docs/research/_20260728_threshold_constant.body.md`; every table below is
> printed by `examples/research/threshold_constant_sweep.py --render` and spliced
> in by `tools/render_threshold_constant_writeup.py`. Edit the body, re-run the
> tool. Three PRs on 2026-07-28 shipped factual errors and **every one was in
> hand-typed prose sitting beside a correct generated table**, which is why this
> document has no hand-typed numbers in it.

- **Date**: 2026-07-28
- **Harness**: `examples/research/threshold_constant_sweep.py` (committed before the run)
- **Artifact**: `examples/research/results/threshold_constant_sweep.json`
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
   endpoints' units, so every pair is counted exactly once and an entity's pairs
   move as a block. The same resample grades both the candidate constant and
   `0.5`, and the interval is on their difference.
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

## 7. Verdict

@@VERDICT@@

## 8. What was measured vs. what was inferred

**Measured** (this artifact, held-out, multi-seed, with intervals):

- Per-cell F1 at all 99 grid constants, at `0.5`, at the label-derived cut, and
  at the exact oracle, for the `heuristic` and `sim_cos` families across the
  loadable portfolio.
- The LOBO-selected constant's Δ-F1 against `0.5` per benchmark and seed, with
  95% paired cluster-bootstrap intervals.
- The stability of that selection under dropping one dataset.

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

## 9. Reproducing

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
