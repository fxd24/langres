# M3 Direction Memo — what the paid race tells us, and where M4 goes

**Status:** M3 shipped. A real-money, budget-capped race (total **$2.1778** / $15 cap)
over an *easy* (Fodors-Zagat) and a *hard* (Amazon-Google) ER dataset, comparing free
scorers against an open-source (GLM-5.2) and a frontier (gpt-4o) LLM judge. Full table:
[`data/benchmarks/m3/M3_RESULTS.md`](../data/benchmarks/m3/M3_RESULTS.md). This memo is
the decision layer: cost/quality frontier, error map, the M4 teacher-selection rule,
what we deliberately did **not** learn, and a plain "are we good?".

> **Read this first.** An initial reading of the run (graded against full gold instead
> of the in-scope subsample) suggested the LLM judges had ~0.25 recall and "lost to free
> embedding everywhere." That was a **measurement artifact** (recall capped at 61/234
> because only 61 of 234 gold pairs were in the 600-pair subsample). After the fix
> (gold restricted to candidate-realizable pairs; cells re-graded from stored confusion
> counts, no new spend), the picture below is the verified one — and it is materially
> different: the LLM judges are **high recall**, and the **frontier judge wins on hard
> data**.

---

## 1. The headline finding

**On hard data, a frontier LLM judge is the best scorer we have; the cheap OSS judge is
not. On easy data, don't pay at all.**

Amazon-Google (hard), pair-level F1, same 600-pair subsample:

| Method | F1 | P | R | $/1k pairs |
|---|---|---|---|---|
| **gpt-4o `llm_judge`** | **0.667** | 0.54 | 0.87 | ~$1.52 |
| `embedding_cosine` (free) | 0.471 | 0.37 | 0.66 | $0 |
| GLM-5.2 `llm_judge` | 0.409 | 0.26 | 0.90 | ~$0.79 |
| `weighted_average` (free) | 0.288 | 0.32 | 0.26 | $0 |
| `rapidfuzz` (free) | 0.271 | 0.25 | 0.30 | $0 |

- **gpt-4o (0.667)** lands inside the literature SOTA band (DeepMatcher/Ditto pairwise-F1
  ≈ 0.5–0.75) and beats free embedding by **+0.20 F1** — the blocking + LLM-judge
  architecture *can* reach competitive accuracy on a hard benchmark.
- **GLM-5.2 (0.409)** is *below* free embedding **and** costs money — the worst trade on
  hard data as configured.
- On **easy** Fodors-Zagat, free `embedding_cosine` wins outright (pair-F1 0.816;
  pipeline BCubed 0.980 via `weighted_average`), and the GLM judge **degenerates**
  (F1 0.233, precision 0.13, recall 1.0 — it accepts almost everything).

## 2. The mechanism — it's precision, not recall

Both LLM judges are **high recall** (gpt-4o 0.87, GLM-5.2 0.90 on AG; GLM 1.0 on FZ).
The differentiator is **precision**: gpt-4o 0.54 vs GLM-5.2 0.26 on AG, and GLM collapses
to 0.13 on FZ. **GLM-5.2-as-judge over-accepts** — it says "match" to too many
near-misses. With **zero** score-extraction failures across every paid call, this is real
model behaviour, not a parsing bug.

Why does this matter? The over-acceptance is exactly what a generic, un-tuned prompt and
a small uncalibrated decision boundary produce. Because GLM-5.2 is a strong OSS model and
gpt-4o is frontier, the most likely cause of GLM's low precision is **prompt/setup**, not
a hard capability ceiling — which makes it the prime M4 lever (see §5). We have **not**
proven that, though; it remains the leading hypothesis.

## 3. Cost / quality frontier

- **Free embedding is the bar:** F1 0.47 (AG) / 0.82 (FZ) at $0. Any paid method must
  clear it to justify spend.
- **GLM-5.2 judge** clears the bar on neither dataset *and* costs ~$0.79/1k pairs (AG) —
  reject as-is.
- **gpt-4o judge** clears it decisively on AG (+0.20 F1) at ~$1.52/1k pairs. Worth it when
  accuracy matters and volume is bounded — or, better, when an LLM only adjudicates an
  *uncertain band* rather than every pair (the cascade we deferred).
- **Latency** tracks the same story: GLM-5.2 (reasoning model) 4.8–6.0 s/pair; gpt-4o
  1.8 s/pair. The OSS model is both less precise *and* slower here.

## 4. Error map

| Method | Dominant error | Consequence |
|---|---|---|
| `rapidfuzz` / `weighted_average` | low recall on AG (0.26–0.30) | misses semantically-equal but lexically-different products |
| `embedding_cosine` | moderate precision (0.37 AG) → over-merges in the pipeline (BCubed 0.30 FZ, 0.59 AG) | transitive closure amplifies false-positive pairs into giant wrong clusters |
| GLM-5.2 `llm_judge` | **low precision / over-acceptance** (0.26 AG, 0.13 FZ) | false-positive matches; pipeline BCubed −0.59 below floor |
| gpt-4o `llm_judge` | residual recall gap (0.87) and cost | a few hard matches missed; pays per pair |

The recurring failure across *both* the embedding pipeline and the GLM judge is the same:
**false positives that transitive-closure clustering amplifies.** Precision is the metric
that pays off downstream, and it's where the cheap options fail.

## 5. M4 direction & the (contingent) teacher-selection rule

The race reshapes M4 from "add an LLM judge" to **"make a precise judge cheap."** Ordered
by expected leverage:

1. **Optimize the judge prompt (DSPy) — the biggest untested lever.** GLM-5.2's high
   recall + low precision is the classic signature of a generic prompt. Optimize for
   precision (few-shot hard negatives, explicit "different model/size/edition ⇒ non-match"
   guidance) and re-measure. *If* an optimized prompt lifts GLM-5.2 precision toward
   gpt-4o's, we get frontier-class judging at OSS cost.
2. **Calibrate, then tune, a cascade.** The deferred cascade's 0.3/0.9 thresholds are
   naive magic constants. Before re-running it, **inspect the embedding-score
   distribution** on each dataset and set the uncertain band from the actual distribution
   (e.g. percentile-based), then tune. The goal: embedding handles the confident bulk for
   free, the LLM adjudicates only the uncertain band — turning gpt-4o's $1.52/1k into a
   fraction of that.
3. **Try a stronger / domain embedding model.** We used `all-MiniLM-L6-v2` (small, general).
   A larger or product-domain embedder likely lifts the free bar on AG and shrinks the
   uncertain band the LLM must touch.

**Contingent teacher-selection rule for M4 distillation:**

- **Default teacher = gpt-4o** (or the strongest available frontier model): its precision
  (0.54 vs GLM 0.26) yields cleaner labels, and label precision is what a distilled student
  inherits.
- **GLM-5.2 as teacher only with a precision gate.** Its labels are high-recall but noisy;
  use it for *candidate proposal* paired with a precision filter (embedding-score gate or
  frontier spot-check), never as the sole labeler.
- **Re-evaluate after step 1:** if prompt optimization closes GLM-5.2's precision gap, it
  becomes a viable low-cost teacher and the default flips on cost grounds. The rule is
  contingent on that measurement, which we have **not** yet made.

## 6. What we did NOT learn (explicit gaps)

- **Cascade cost/quality** — deferred. Cascade currently records no token counts and can't
  price OpenRouter responses (its $cost reads $0), and its thresholds are uncalibrated. Its
  numbers would have been dishonest, so we did not run it. It is the top M4 measurement.
- **Frontier on Fodors-Zagat** — deferred (FZ is saturated by free methods; no headroom).
- **Whether GLM-5.2's precision gap is prompt or capability** — untested; §5.1 is the
  experiment.
- **Multi-seed LLM variance** — each paid cell is a single pass; the LLM F1s carry no
  seed-variance band (the free cells do: 5 seeds).
- **Subsample size** — AG LLM cells are a 600-pair stratified subsample (61 positives);
  embedding is stable across subsample↔full (0.471↔0.469), but the LLM F1s have a wider CI
  than the full split would give.
- **Person / multilingual / temporal signal** — both datasets are structured, English,
  product/restaurant records. We learned nothing about name-heavy, multilingual, or
  streaming/temporal resolution (tracked in `docs/USE_CASES.md`).

## 7. Absolute viability gate — are we good?

**Gate:** is the target architecture competitive with the ER literature on a hard dataset?
**Best AG pair-F1 = 0.667 (gpt-4o judge), inside the SOTA band (~0.5–0.75). → Gate PASSED,
but only with a frontier judge and at real cost.**

**Plain answer: qualified yes on direction, no on "just bolt on an LLM."**

- ✅ The blocking + LLM-judge architecture **can** reach competitive accuracy on hard data
  (gpt-4o 0.667, beating the free 0.471 bar and reaching SOTA range).
- ❌ The **cheap** path isn't there yet: the OSS GLM-5.2 judge is *worse than free* and
  costs money (low precision). Naively adding a cheap LLM judge everywhere is the **worst**
  option on both datasets.
- ❌ On **easy** data, paying for an LLM is pure waste — free embedding wins.

So M4 is not "add the judge" — it's **make a precise judge cheap**: optimize the prompt
(DSPy), calibrate and tune the cascade against the real embedding-score distribution,
distill frontier-quality labels into a cheap student, and upgrade the embedding model.
The frontier result proves the ceiling is worth chasing; the OSS result proves we have to
*engineer* our way to it rather than buy it off the shelf.
