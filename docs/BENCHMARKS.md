# The benchmark portfolio (and scoring your own data)

langres ships a **portfolio** of entity-resolution datasets behind one
discoverable, serializable registry, plus a one-liner to score **your own**
labeled pairs with the same honest pair-level metrics. This doc covers both:

1. [Discover + race the portfolio](#1-discover-and-race-the-portfolio) — the
   `data/registry` manifest and the `portfolio_race` example.
2. [The datasets, and why each is a target](#2-the-datasets-and-why-each-is-a-target).
3. [Score your own data](#3-score-your-own-data) — `evaluate(judge, candidates,
   gold_pairs)`.
4. [Replicate a published LLM benchmark, offline](#4-replicate-a-published-llm-benchmark-offline)
   — the Peeters et al. (EDBT 2025) LLM-EM replay at $0.

> **Entry point:** `langres.eval` is the curated evaluation surface. It
> re-exports `evaluate`, `list_benchmarks` / `get_benchmark`, and the ER metrics
> (`reduction_ratio`, BCubed/pairwise, …) so you import them from one place
> instead of reaching into `core.benchmark` / `core.metrics` / `data.registry`.
> The ranking metrics (MRR/NDCG/MAP, via
> `core.metrics.evaluate_blocking_with_ranking`) additionally need the opt-in
> `[eval]` extra: `pip install 'langres[eval]'`.

---

## 1. Discover and race the portfolio

The datasets live behind a static, import-light manifest in
`langres.data.registry` (re-exported through the curated `langres.eval` facade)
— a **name → benchmark** map you can list without importing any loader (so it is
safe in a core-only install and never pulls the `[semantic]` stack into
`sys.modules`):

```python
from langres.eval import list_benchmarks, get_benchmark

for entry in list_benchmarks():           # metadata only, no dataset load
    print(entry.name, entry.task, entry.domain, "loadable" if entry.loadable else "external")

bench = get_benchmark("dblp_acm")         # imports ONLY the selected loader, lazily
```

`get_benchmark(name)` returns a ready benchmark conforming to
`langres.core.benchmark.Benchmark` (and the `langres.methods.BlockingBenchmark`
contract), so it drops straight into the race harness. A non-loadable entry
(`opensanctions`, below) raises a clear external-only error instead of a broken
load.

**Race the whole portfolio** with the offline, zero-spend methods via
[`examples/research/portfolio_race.py`](../examples/research/portfolio_race.py) —
it iterates the registry so a newly-registered dataset appears automatically,
with no edit to the script:

```bash
uv run python examples/research/portfolio_race.py           # full portfolio
uv run python examples/research/portfolio_race.py --fast     # FZ + DBLP-ACM only
uv run python examples/research/portfolio_race.py --fast --paid --budget 5   # + a capped LLM row
```

> **What the registry actually buys you — honestly.** `run_methods` already
> collapsed the per-method racing boilerplate into one call; the registry adds the
> missing half — **name → benchmark discoverability** and a serializable manifest
> — replacing the scattered, hand-maintained `_BENCHMARKS = (FodorsZagatBenchmark(),
> …)` tuples each race script used to carry. The `--paid` LLM row is graded
> **judged-once** (via `evaluate_judge_on_candidates` under a `BudgetedModuleRunner`
> + one `SpendMonitor`), never through `run_methods` — see the KISS warning in
> [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) about re-judging a paid judge per grid
> threshold.

---

## 2. The datasets, and why each is a target

Every entry is a **cross-source linkage** benchmark (find the pairs across two
sources that refer to the same entity). All ship in-repo (`loadable=True`) except
OpenSanctions.

| Benchmark | Domain | Why it's in the portfolio |
| --- | --- | --- |
| `fodors_zagat` | restaurant | **Saturated regression guard.** Tiny, easy, blocking Pair-Completeness ≥ 0.99 — the "nothing should ever regress here" floor. |
| `abt_buy` | product | **Product regression guard.** Short, noisy product titles; a standard DeepMatcher band to hold the line against. |
| `amazon_google` | product | **Hard product guard.** Unsaturated — blocking PC ~0.84 caps recall, so it separates weak from strong scorers (used as the discrimination check). |
| `dblp_acm` | bibliographic | **Clean, 1:1, gate-passing.** High-quality bibliographic records; blocking PC saturates early. The "does the honest pipeline pass a clean gate" check. |
| `dblp_scholar` | bibliographic | **Many-to-many bibliographic.** Match graph has clusters > 2 records. **Caveat:** ~60% of gold pairs are *intra-source*, so the cross-source blocking PC reads ~0.39 — a many-to-many closure artifact, **not** a blocking failure (true cross-source recall is ~0.99). Read the loader header before concluding "blocking is bad". |
| `walmart_amazon` | product | **Harder product.** Long, structured product records; blocking honestly *misses* — highest measured PC ~0.877 (gate not met), recorded in `WALMART_AMAZON_ACHIEVED_PC`. A dataset where the ceiling is a real, documented shortfall. |
| `wdc_computers` | product | **Title-only, textually hard** — each record is one noisy free-text `title` blob (specs + brand + multilingual retailer fragments). Also exposes a derived **seen/unseen slice** (`wdc_slice_map`) to demonstrate the honest seen → unseen F1 drop at a fixed threshold. |
| `febrl_person` | person | **Synthetic person set (FEBRL4).** A *second entity type* (not a product/bibliographic record) — the generality check that the pipeline isn't product-shaped. |
| `opensanctions` | person/org | **External baseline only — not bundled.** CC-BY-NC 4.0 is incompatible with langres's Apache-2.0 license, so it is never vendored; `get_benchmark` raises `ExternalBenchmarkError` pointing at where to fetch it and the published matcher F1 baselines. |

---

## 3. Score your own data

Have your own records plus some labeled `(id_a, id_b, match?)` pairs? Score any
judge over them at honest pair-level Precision/Recall/F1 with the one-liner
`evaluate(judge, candidates, gold_pairs)` (`langres.eval`, re-exported from
`langres.core.benchmark`) — no blocking-recall ceiling, no clustering
amplification, so the number is directly comparable to pairwise-F1 SOTA. By
default (`threshold=None`) it sweeps a grid and returns `best_threshold`, the
best-F1 cut — but that argmax is fitted to the very `gold_pairs` it then reports
F1 against, so the number is optimistically biased (an upper bound, not a
held-out estimate), and the call emits a one-shot `UserWarning` saying exactly
that. Pass `threshold=<float>` to grade once at a fixed, honest cut instead:
`graded_threshold` records the cut used and `best_threshold` is `None`. The sweep
stays the default deliberately — there is no single fixed cut that serves every
judge, because an embedding judge's cosine non-matches sit well above 0.5 (around
0.70–0.80), so a global 0.5 would call them all matches and wreck its F1, while a
binary LLM judge wants 0.5. Honesty comes from the warning and the `threshold=`
opt-out, not from imposing one universal default.

The reusable bridge from raw `(id_a, id_b, label)` rows to scored candidates is
`FixedSplitPairBenchmark.from_loaders` (`langres.data.fixed_split_pair_benchmark`):
give it your schema and two loaders, and `build(split)` returns the
`ERCandidate`s (each carrying a comparison vector) and the gold pair set.

```python
from pydantic import BaseModel

from langres.eval import evaluate
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.data.fixed_split_pair_benchmark import FixedSplitPairBenchmark


class Product(BaseModel):
    id: str
    title: str | None = None
    brand: str | None = None


# Your data: a corpus of records + labeled pairs (1 = match, 0 = non-match).
records = [Product(id="a1", title="Canon EOS 80D"), Product(id="b1", title="canon eos 80d dslr"), ...]
labeled_pairs = [("a1", "b1", 1), ("a1", "b2", 0), ...]

bench = FixedSplitPairBenchmark.from_loaders(
    name="my_products",
    schema=Product,
    corpus_loader=lambda: (records, None, None),   # only element 0 (the corpus) is used
    pair_split_loader=lambda: {"test": labeled_pairs},
)
data = bench.build("test")                          # candidates (comparison attached) + gold

# Any Module works as the judge; WeightedAverageJudge scores the attached
# comparison vector using the auto-derived StringComparator's features.
judge = WeightedAverageJudge(feature_specs=bench.feature_specs)

# Default: sweep the grid. Warns that best_threshold is argmax-fitted to gold.
result = evaluate(judge, data.candidates, data.gold)
print(result.pair.precision, result.pair.recall, result.pair.f1,
      result.best_threshold, result.graded_threshold)

# Honest fixed cut: grade once at a chosen threshold — no warning, no argmax.
fixed = evaluate(judge, data.candidates, data.gold, threshold=0.5)
print(fixed.pair.f1, fixed.best_threshold, fixed.graded_threshold)  # best_threshold is None
```

Swap the judge for anything else — an `EmbeddingScoreJudge`, an `LLMJudge`, a
fitted `RandomForestJudge` — and the call is identical. `evaluate()` is already
spend-capped by default (`budget_usd=`, resolving to `DEFAULT_BUDGET_USD` =
$1.00), so a **paid or compiled** judge can't run away; reach for the fuller
`evaluate_judge_on_candidates` only when you need the **raw judgements** back, a
**caller-owned runner**, or **custom cost accounting** — its `runner=` /
`price_per_token_or_pair=` / `cost_track_fn=` knobs live there, and it returns
`(JudgePairEval, judgements)`. To reflect a seen → unseen split, pass `slice_fn=`
— every slice is graded at the one global cut (`graded_threshold`), never a
per-slice argmax.

**Scoring against a *bundled* benchmark instead of your own pairs?** Skip the
`FixedSplitPairBenchmark` construction: `candidates_for` (also on `langres.eval`)
blocks any registered benchmark's split into the same `(candidates, gold_pairs)`
pair `evaluate` wants — `candidates, gold = candidates_for(get_benchmark("dblp_acm"),
split="test")`, then `evaluate(judge, candidates, gold)`. That is the seam that
keeps this walkthrough out of `Resolver`'s private candidate generator: you never
block a split by hand.

---

## 4. Replicate a published LLM benchmark, offline

To trust our metric code, reproduce a *published* number without spending a cent.
`langres.data.peeters` replicates *Entity Matching using Large Language Models*
(Peeters, Steiner & Bizer, arXiv 2310.11244 v4, EDBT 2025;
[`wbsg-uni-mannheim/MatchGPT`](https://github.com/wbsg-uni-mannheim/MatchGPT))
by **replaying the authors' archived model answers** — no API key, no LLM call,
`$0`:

```python
from langres.data.peeters import (
    get_peeters_replication, render_sample_prompts,
    judgements_from_answers, gold_match_pairs,
)
from langres.core.metrics import classify_pairs

spec = get_peeters_replication("abt-buy")     # or "amazon-google"
prompts = render_sample_prompts(spec)          # our records + their serializer + prompt
answers = [...]                                # their archived raw "Yes"/"No" answers
judgements = judgements_from_answers(prompts, answers)
m = classify_pairs(judgements, gold_match_pairs(prompts), threshold=0.5)
print(m.precision, m.recall, m.f1)             # binary pairwise P/R/F1
```

- **It's a *slice*, not a new benchmark.** Their eval set is a deterministic
  `sample(random_state=42)` subset of the DeepMatcher `test.csv` we already
  vendor. `regenerate_sample_rows(spec)` reproduces it from our own CSVs (a
  numpy-only reproduction of `pandas.sample`), and the tracked
  `datasets/<ds>/peeters_sampled_test.csv` is verified **exactly equal** to the
  authors' published `sampled_gs` (abt-buy 1206, amazon-google 1234; 0 label
  mismatches). Because the protocol is a fixed **binary pair-classification** task
  (no blocking, no clustering, no threshold sweep), it stays out of the
  `data/registry` clustering manifest and gets its own small
  `list_peeters_replications()` / `get_peeters_replication()` seam.
- **No MatchGPT data is vendored.** MatchGPT ships no LICENSE (`license: null`);
  langres is Apache-2.0. The pair lists are regenerated from *our* data; the
  ~186 MB answer archive is fetched transiently by the harness to a gitignored
  cache dir.

Run the full replay — regenerate the sample, render every prompt, diff it against
the archived prompt, parse the answers, and score — with
[`examples/research/peeters_llm_em_replication.py`](../examples/research/peeters_llm_em_replication.py):

```bash
uv run python examples/research/peeters_llm_em_replication.py   # abt-buy / gpt-4-0613
```

It reproduces arXiv v4 **Table 2** `abt-buy` / `gpt-4-0613` /
`domain-complex-force` → **F1 95.15** at a **100.00% byte-exact** prompt
round-trip. (amazon-google round-trips 99.51%; the 6 diffs are float-repr
artifacts in *their* gold standard's `price`, e.g. `6.5600000000000005` vs our
`6.56` — a data-provenance difference, not a serializer bug.)

### 4a. Run it *live* (paid), budget-capped

The same harness can run a **real** `LLMJudge` over the identical 1206-pair slice
— the paper's `domain-complex-force` template, the Peeters per-dataset
`record_serializer`, `response_parser=parse_binary_yes_no`, `temperature=0.0` —
so the number is ours, not a replay. It is off by default and triple-guarded (an
explicit `--yes-spend-money` flag, a priced-model assertion, and a hard
`SpendMonitor` cap). **Preview the cost at $0 first** (renders every prompt,
counts tokens, makes **zero** API calls):

```bash
uv run python examples/research/peeters_llm_em_replication.py --mode dry-run
# -> 100,256 input tokens/model; est $0.017 (gpt-4o-mini) + $0.27 (gpt-4o) = ~$0.29

# The paid run (needs OPENROUTER_API_KEY; run with the sandbox disabled):
uv run python examples/research/peeters_llm_em_replication.py --mode live --yes-spend-money
```

It races two dated snapshots and prints a comparison table (F1, P/R, the
aggregated `LLMUsage` vector, the **real** OpenRouter-billed cost + `cost_is_real`,
and `$/1k pairs`), with the paper's published F1 as a column:

| model (OpenRouter id) | paper "name" | published Abt-Buy F1 | ~est cost |
|---|---|---|---|
| `openai/gpt-4o-mini-2024-07-18` | GPT-mini | **90.95** (P 89.25 / R 92.72) | ~$0.017 |
| `openai/gpt-4o-2024-08-06` | GPT-4o | **90.47** (P 83.27 / R 99.03) | ~$0.27 |

`gpt-4-0613` (the F1 **95.15** cell §4 replays) would cost ~$3.15 live and is
**deliberately declined** — not worth the spend, and it retires 2026-10-23;
`gpt-3.5-turbo-0613`/`-0301` were shut down 2024-09-13. The default budget cap is
**$1.00** for both models combined (measured total ≈ $0.29, a ~3.4× margin). Every
raced model MUST be priced in `langres.clients.openrouter.PRICES_PER_1M` — an
unpriced model silently contributes $0 to the cap, so the script refuses to start
without a price entry.

The one deviation from the paper's setup is that we route the same dated snapshot
through **OpenRouter** rather than calling OpenAI directly, so the live judge pins
`provider={"order": ["OpenAI"], "allow_fallbacks": False}` (`LLMJudge(provider=…)`
→ `extra_body["provider"]`) — OpenRouter must serve the request from OpenAI's own
backend and can't silently swap in a different provider/quantization.

### 4b. Cheaper trials + per-pair agreement against the authors' answers

`--limit N` runs a **stratified** subset of `N` pairs (preserving the ~17.1%
positive ratio, deterministic under `--seed`, default 0) — the file is a positive
block followed by a negative block, so a naive first-`N` would be all matches. A
150-pair gpt-4o-mini trial costs **~$0.002**.

`--compare-archived` (`--mode live`) judges each pair live **and** compares our
parsed verdict to the authors' archived per-pair answer for the *same* model
(reusing the replay harness's cached download). It reports the **per-pair
agreement rate**, a **2×2 confusion** of ours-vs-theirs (both-yes / both-no /
we-yes-they-no / we-no-they-yes), up to **10 concrete disagreeing pairs** (record
text, gold label, their raw answer, our raw answer), and **our** F1/P/R on the
judged subset next to **their** F1/P/R recomputed on that *same* subset (plus the
published full-set number). Both verdicts are parsed through the one canonical
`parse_binary_yes_no`, and it **fails loudly** if our rendered prompt does not
match the archived one — a mismatch means the alignment is off and every
downstream comparison would be meaningless.

```bash
# $0 preview of the 150-pair subset cost:
uv run python examples/research/peeters_llm_em_replication.py --mode dry-run \
    --model openrouter/openai/gpt-4o-mini-2024-07-18 --limit 150
# PAID (~$0.002), with per-pair archive agreement (run with the sandbox disabled):
uv run python examples/research/peeters_llm_em_replication.py --mode live \
    --model openrouter/openai/gpt-4o-mini-2024-07-18 --limit 150 \
    --compare-archived --yes-spend-money
```

### 4c. First-token credence probe (`--logprobs`)

`--logprobs` (`--mode live`) runs the *same* live judge with
`LLMJudge(confidence="logprob")`: it requests first-token logprobs and records a
P(Yes) credence per pair — `p_yes` (renormalised over the yes/no two-way subspace),
`leaked_mass` (never normalised away), `p_yes_is_bound`, and `correct = verdict ==
gold` — in **v2** result rows, so "does the model's own first-token credence predict
its errors?" is answerable from the rows alone. It is an evidence-gathering probe:
**nothing is added to `PairwiseJudgement`**. Because the binary protocol answers with
a single output token, `top_logprobs` adds **zero** output tokens, so the probe
re-runs at ~the replication cost (~$0.016 gpt-4o-mini, ~$0.27 gpt-4o). Probe rows
land in a distinct `…__logprobs.jsonl` (a contamination firewall — it cannot
overwrite the committed replication rows) and `--results-dir` defaults to the
**committed** `examples/research/results/peeters` so the paid probe's rows are durable.

```bash
# PAID credence probe over both models (writes …__logprobs.jsonl to the committed dir):
uv run python examples/research/peeters_llm_em_replication.py --mode live --logprobs \
    --yes-spend-money
```

---

## See also

- [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) — the two measurement surfaces
  (`run_methods` vs. `evaluate_judge_on_candidates`), the DSPy loop, calibration,
  and the `SpendMonitor` budget seam.
- [`docs/TUTORIAL_YOUR_OWN_CSV.md`](TUTORIAL_YOUR_OWN_CSV.md) — the *clustering*
  counterpart: a messy CSV → entity clusters via `dedupe`, at $0.
- [`examples/research/portfolio_race.py`](../examples/research/portfolio_race.py)
  — the registry-driven race this doc describes.
- [`examples/research/peeters_llm_em_replication.py`](../examples/research/peeters_llm_em_replication.py)
  — the offline Peeters et al. LLM-EM replay (§4).
