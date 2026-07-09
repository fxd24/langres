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
`langres.core.benchmark`). It grades
the judge at the best-F1 threshold over a grid and returns a `JudgePairEval` — no
blocking-recall ceiling, no clustering amplification, so the number is directly
comparable to pairwise-F1 SOTA.

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

result = evaluate(judge, data.candidates, data.gold)
print(result.pair.precision, result.pair.recall, result.pair.f1, result.best_threshold)
```

Swap the judge for anything else — an `EmbeddingScoreJudge`, an `LLMJudge`, a
fitted `RandomForestJudge` — and the call is identical. For a **paid or compiled**
judge that needs a spend cap or the raw judgements back, call the fuller
`evaluate_judge_on_candidates` directly (pass a `BudgetedModuleRunner` via
`runner=`); `evaluate()` is the thin, common-case one-liner over it. To reflect a
seen → unseen split, pass `slice_fn=` — every slice is graded at the one global
best-F1 threshold (never a per-slice argmax).

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
