# M3 Benchmark Race — Results

**A budget-capped, real-money multi-method race over two ER datasets.**
Total spend: **$2.1778** (hard cap $15.00). All numbers below are computed from the
committed per-cell results under `data/benchmarks/m3/results/`. No cell is faked or
extrapolated.

## Setup

| | |
|---|---|
| **Datasets** | Fodors-Zagat (FZ, *easy* restaurant dedup) · Amazon-Google (AG, *hard* product linkage, literature test split) |
| **Methods (free)** | `rapidfuzz`, `weighted_average`, `embedding_cosine` (all-MiniLM-L6-v2) |
| **Methods (paid)** | `llm_judge` with **GLM-5.2** (`z-ai/glm-5.2`, strong OSS) and **gpt-4o** (frontier), both via OpenRouter |
| **Primary metric** | pair-level **F1** at the best threshold on a widened grid (0.05–0.99) — isolates the *scorer* from clustering |
| **Secondary** | end-to-end pipeline **BCubed F1** vs the all-singletons sanity floor |
| **Blocking** | held constant across methods (they differ only in the scorer) |

**Scope of the paid run.** GLM-5.2 is a reasoning model (~3–6 s/call), so the full
6 800-call race was infeasible in one session. We ran the **validated `llm_judge`
path** only: GLM-5.2 on a stratified **600-pair AG subsample** (61 in-scope
positives) + the full FZ band; gpt-4o on the **identical** AG subsample (so
GLM-vs-frontier is directly comparable). **Deferred** (see the direction memo):
the cascade/hybrid and the frontier FZ pass — cascade needs a source fix before its
$cost is honest, and isn't on the M3 go/no-go path. Score-extraction failures across
all paid calls: **0** — every reported score is real model behaviour.

> **Measurement correction (disclosed).** The first paid run graded the 600-pair AG
> subsample against the *full* 234-pair gold, capping recall at 61/234≈0.26 for every
> method — an artifact. `evaluate_judge_on_candidates` now restricts gold to
> candidate-realizable pairs, and the committed AG LLM cells were re-graded from their
> stored confusion counts (no new LLM calls; see `examples/m3_regrade_subsample.py`
> and each cell's `regrade` block). The numbers below are the corrected ones.

---

## Headline — Amazon-Google (hard), pair-level F1

All methods on the **same 600-pair subsample** (61 in-scope positives) — apples-to-apples:

| Method | F1 | Precision | Recall | Cost | Latency |
|---|---|---|---|---|---|
| **gpt-4o `llm_judge`** | **0.6667** | 0.5408 | 0.8689 | $0.9114 | 1.77 s/pair |
| `embedding_cosine` (free) | 0.4706 | 0.3670 | 0.6557 | $0 | — |
| GLM-5.2 `llm_judge` | 0.4089 | 0.2644 | 0.9016 | $0.4729 | 6.05 s/pair |
| `weighted_average` (free) | 0.2883 | 0.3200 | 0.2623 | $0 | — |
| `rapidfuzz` (free) | 0.2707 | 0.2500 | 0.2951 | $0 | — |

*Reference — free methods on the full 2293-pair AG test split:* `embedding_cosine`
**0.4689**, `weighted_average` 0.3361, `rapidfuzz` 0.3062. Embedding on the subsample
(0.4706) ≈ on the full set (0.4689), confirming the subsample is representative.

**Reading it:** on the hard dataset the **frontier LLM judge wins clearly (0.667)** and
lands in the literature SOTA band (~0.5–0.75), beating free embedding (0.471). The
**open-source GLM-5.2 judge is high-recall but low-precision** (R 0.90, P 0.26 → F1
0.41) — it *finds* the matches but over-accepts, ending below free embedding. Both LLM
judges are **high recall (0.87–0.90)**; the earlier "LLMs miss 75%" reading was the
grading artifact, not real behaviour.

---

## Fodors-Zagat (easy), pair-level F1 — full 1023-pair band (33 positives)

| Method | F1 | Precision | Recall | Cost |
|---|---|---|---|---|
| `embedding_cosine` (free) | **0.8158** | 0.7209 | 0.9394 | $0 |
| `weighted_average` (free) | 0.7246 | 0.6944 | 0.7576 | $0 |
| `rapidfuzz` (free) | 0.7059 | 0.6857 | 0.7273 | $0 |
| GLM-5.2 `llm_judge` | 0.2332 | 0.1320 | 1.0000 | $0.7935 |

On the *easy* dataset **free embedding dominates** and the GLM judge is **degenerate**:
it scores ~250 band candidates ≥0.90 with no discrimination (precision flat at 0.13
across every threshold, recall 1.0). Same high-recall/low-precision failure as on AG,
but fatal here because FZ's positive rate is low (33/1023).

## Pipeline BCubed F1 (end-to-end, 5 seeds)

| | Fodors-Zagat | Amazon-Google |
|---|---|---|
| all-singletons floor | 0.9317 | 0.8545 |
| `rapidfuzz` | 0.9764 ± 0.0037 | 0.7397 ± 0.0044 |
| **`weighted_average`** | **0.9798 ± 0.0040** | 0.8168 ± 0.0057 |
| `embedding_cosine` | 0.2999 ± 0.0457 | 0.5904 ± 0.0210 |
| GLM-5.2 `llm_judge` (1 seed) | 0.3405 | — |

FZ pipeline is saturated — `weighted_average` clears the floor (+0.048); `embedding_cosine`
over-merges via transitive closure (−0.63) and the GLM judge's low precision does the
same (−0.59). On AG **no free method clears the floor** (best 0.817 < 0.855): the hard
dataset has no easy end-to-end win.

---

## Cost & honesty

| Cell | Model | Cost |
|---|---|---|
| `agfixed_llm_judge` | GLM-5.2 | $0.4729 |
| `fzband_llm_judge` | GLM-5.2 | $0.7935 |
| `agfixed_llm_judge_frontier` | gpt-4o | $0.9114 |
| (9 free cells) | — | $0 |
| **Total** | | **$2.1778** |

Cost is priced deterministically from the prompt/completion **token counts** the judge
records in provenance against pinned per-1M rates (litellm's `completion_cost` returns
$0 for OpenRouter responses — their dated model id carries no provider prefix). Non-zero,
auditable, and well under the $15 cap.

**The threshold-grid fix mattered.** The shared race grid capped at 0.80, which crushed
score-based judges whose useful scores sit above it. Widening to 0.99 lifted
`embedding_cosine` from F1 0.10 → **0.82** (FZ) and 0.24 → **0.47** (AG). All numbers
here use the widened grid.
