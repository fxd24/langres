# langres: Technical Documentation & API Reference

Welcome to the langres documentation. This document provides a deep dive into the layered API, the core architectural pillars, and the data contracts that power the library.

## 1. The langres Layered API

langres exposes a **three-layer API** — each layer a thin shell over the one below — so common tasks are one-liners while bespoke pipelines stay fully composable:

- **Verbs — `langres.link` / `langres.dedupe`** (High-Level): the schema-optional, zero-label front door and the recommended entry point. `judge="auto"` picks an LLM judge from your API key under a default $1 spend cap; results are self-describing (`LinkVerdict`; a `dedupe` result carrying `judge_used` / `score_type`).
- **`langres.Resolver`** (Mid-Level): the declarative pipeline. `Resolver.from_schema(schema, judge=...)` wires a default blocker + comparator + judge + clusterer from a Pydantic schema; `.resolve(records)` runs it and `.save`/`.load` serialize it (config registry, no pickle).
- **`langres.core`** (Low-Level): the "power-user" API — the base classes (`Module`, `Blocker`, `Comparator`, `Clusterer`, judges) you compose into entirely new logic from scratch.

> There is **no** `langres.tasks` / `langres.flows` module — those names were never built. The real layering is verbs → `Resolver` → `langres.core`.

## 2. The Abstraction Layer: langres as a "Glue" Framework

A primary goal of langres is to act as a powerful "glue" framework, simplifying and abstracting best-in-class libraries into a single, cohesive workflow. For contributors and advanced users, it's important to understand what langres is managing under the hood.

### What each dependency actually does

langres keeps a **small always-installed core** and pushes every heavy library
behind an opt-in extra. Nothing below is auto-orchestrated by a magic
"Optimizer"; the map is simply which library backs which component.

**Core (always installed, `uv sync`):**

- **pydantic** — every schema, `ERCandidate`, and `PairwiseJudgement` is a
  Pydantic model; all validation and (de)serialization runs through it.
- **rapidfuzz** — string similarity behind `StringComparator` and the default
  `WeightedAverageJudge` scorer (the `judge="string"` path).
- **networkx** — `Clusterer` builds an undirected graph from the judgements that
  clear the threshold and takes connected components (transitive closure). This
  is the *only* clustering backend — there is no scipy / hierarchical path.
- **numpy** — vector math shared by the embedding helpers.

**Opt-in extras** (`pip install langres[<extra>]`):

- **`[semantic]`** — sentence-transformers / torch / faiss-cpu / qdrant-client
  (plus onnxruntime/optimum, fastembed): the embedding + ANN stack behind
  `VectorBlocker`, `EmbeddingScoreJudge`, and `judge="embedding"`.
- **`[llm]`** — litellm / dspy-ai / openai: `LLMJudge`, the DSPy-compiled
  `DSPyJudge`, and `judge="zero_shot_llm"` / `judge="auto"`. DSPy prompt
  optimization is real, but it lives *inside* `DSPyJudge` — it is not an
  automatic compile pass over a whole pipeline.
- **`[trained]`** — scikit-learn: `RandomForestJudge` and
  `core.calibration.derive_threshold`. Note the BCubed / pairwise metrics in
  `core.metrics` are a **vetted internal implementation** (Amigó et al. 2009),
  *not* sklearn — sklearn is pulled in only by these trained-judge / calibration
  paths.
- **`[eval]`** — ranx: the ranking metrics (MRR / NDCG / MAP) in
  `core.metrics.evaluate_blocking_with_ranking`, imported lazily so the rest of
  `core.metrics` / `core.benchmark` stays importable without it. (There is no
  `pytrec_eval` anywhere in the tree.)

Hyperparameter search is opt-in too: `core.optimizers.BlockerOptimizer` (Optuna)
tunes *blocker* parameters — see §5. A general `Optimizer` over full pipelines is
roadmap (`docs/ROADMAP.md`), not implemented.

**Lazy loading.** These heavy symbols resolve through a PEP 562 `__getattr__`
seam in `langres/core/__init__.py` and `langres/clients/__init__.py`, so a bare
`import langres` never drags torch / litellm / faiss / scikit-learn / ranx into
`sys.modules`. `tests/test_import_budget.py` guards this.

### core.review.ReviewQueue (Human-in-the-Loop)

The HITL system splits a storage backend from a labeling surface:

- **`core.review.ReviewQueue`:** the storage backend — it writes the selected pairs to a plain `review_queue.jsonl` snapshot (one JSON line per pair; ids-only unless you pass `records=` to join content back on). `core.review.select_for_review` picks *which* pairs are worth a human's attention (uncertainty / disagreement / audit strategies).
- **The `langres` CLI:** the labeling surface. `langres export-csv` turns a queue into a spreadsheet, a reviewer fills the `label` column, and `langres import-csv` reads it back into a `corrections.jsonl` log; `langres review` is the equivalent quick terminal loop. The harvested corrections feed `core.harvest` and `core.calibration.derive_threshold` back into the pipeline.

### Observability & Tracing (TBD)

The design for full tracing (e.g., via OpenTelemetry) is still to be determined. The foundation for this is the PairwiseJudgement's provenance field, which is designed to capture all necessary metadata for a future tracing system.

## 3. High-Level API: the verbs (`dedupe` / `link`)

The two verbs are the one-liner front door. They infer an ephemeral schema from your records' keys (or take an explicit `schema=<YourModel>`), resolve a judge, and run the full blocking → scoring → clustering pipeline under a spend cap. The runnable, offline version lives in [`examples/quickstart_verbs.py`](../examples/quickstart_verbs.py).

### dedupe

**Definition:** Group a batch of records into entity clusters (single-source deduplication).

`dedupe(records, *, judge="auto", schema=None, threshold=None, budget_usd=None, log=None, ...)` returns a `DedupeResult` — a `list[set[str]]` of the multi-record clusters (singletons are dropped) that additionally carries `judge_used` and `score_type`.

**Example** (offline — `judge="string"` pins the zero-spend judge, no API key or network needed):

```python
from langres import dedupe

records = [
    {"id": "1", "name": "Acme Corporation", "city": "New York"},
    {"id": "2", "name": "Acme Corp", "city": "New York"},
    {"id": "3", "name": "Totally Different Co", "city": "Chicago"},
]

result = dedupe(records, judge="string", threshold=0.6)
# result -> [{'1', '2'}]   (singleton "3" is dropped)
print(result.judge_used, result.score_type)   # "string" "heuristic"
```

`judge="auto"` (the default) instead picks a real LLM judge from `OPENROUTER_API_KEY` / `OPENAI_API_KEY` (needs the `[llm]` extra) and raises `NoJudgeAvailableError` if no key is set — langres never silently falls back to fuzzy matching. Every judge runs under a default $1 spend cap (`budget_usd=`); a breach raises `BudgetExceeded` carrying the partial judgements.

### link

**Definition:** Decide whether **two records** are the same entity — a single pairwise verdict.

`link(left, right, *, judge="auto", schema=None, threshold=None, ...)` returns a `LinkVerdict` — truthy iff it's a match — carrying `.score`, `.judge_used`, `.score_type`, and `.reasoning`.

```python
from langres import link

verdict = link(
    {"id": "a", "name": "Acme Corp", "city": "New York"},
    {"id": "b", "name": "Acme Corporation", "city": "New York"},
    judge="string",
)
if verdict:                       # LinkVerdict is truthy iff it's a match
    print(verdict.score, verdict.judge_used)
```

> **Cross-source / incremental linking is not built yet.** `link()` above compares a single pair, not two datasets. The dataset-to-dataset methods `Resolver.link` / `Resolver.stream_against` are reserved `NotImplementedError` stubs (roadmap M5) — do not treat them as working.

## 4. Mid-Level API: the `Resolver`

The verbs are thin sugar over `Resolver`. Drop to it directly when you want an explicit, serializable pipeline built from a Pydantic schema.

```python
from pydantic import BaseModel
from langres import Resolver

class Company(BaseModel):
    id: str
    name: str
    city: str

resolver = Resolver.from_schema(Company, judge="string", threshold=0.6)
clusters = resolver.resolve(records)      # -> list[set[str]]
resolver.save("company_resolver.json")    # config-registry serialization (no pickle)
# later, in a fresh process:
resolver = Resolver.load("company_resolver.json")
```

`from_schema` auto-derives a missing-aware `StringComparator` from the schema's string fields, a `WeightedAverageJudge` scorer, an `AllPairsBlocker` (or a `VectorBlocker` when `judge="embedding"`), and a `Clusterer`. `judge=` accepts `"string"` (default), `"embedding"`, `"zero_shot_llm"`, or a `Module` instance. This is the low-level, explicit switch: **no** `"auto"` key-resolution and **no** spend cap (that magic lives in the verbs), so a paid judge built here runs uncapped.

See [DX_RESOLVER.md](DX_RESOLVER.md) for the before/after of the manual lambda pipeline vs. the declarative `from_schema` + `save`/`load` path.

## 5. Core API: The Five Pillars (langres.core)

This is the low-level "PyTorch" layer. You use these base classes to build your own custom components from scratch.

### core.Blocker (Base Class)

**Definition:** The Blocker is the Data Loader & Transformer of the pipeline. It has two jobs:

- **Generate Pairs:** Efficiently find candidate pairs (e.g., via ANN search) to avoid N² comparisons.
- **Normalize Schema:** Act as the ETL layer. It transforms raw data from one or more sources into the single, clean, internal Pydantic schema that the Flow expects.

**What it's not:** It does not compare records. It only loads and normalizes them.

**Key Methods:**

- `stream(data: List[Any]) -> Iterator[ERCandidate]`
- `stream_against(source: List[Any], target: List[Any]) -> Iterator[ERCandidate]`
- `stream(datasets: List[List[Any]]) -> Iterator[ERCandidate]`

**Example (Custom Blocker):**

```python
from pydantic import BaseModel
from langres.core import Blocker, ERCandidate

class MyInternalSchema(BaseModel):
    id: str
    name_field: str
    text_field: str

class MyCustomBlocker(Blocker):
    def __init__(self, name_map: str, text_map: str):
        # This blocker is configured with the user's column names
        self.name_map = name_map
        self.text_map = text_map
        # ... initialize ANN index ...

    def stream(self, data: List[dict]) -> Iterator[ERCandidate[MyInternalSchema]]:
        # 1. Find pairs using ANN logic (not shown)
        for raw_a, raw_b in self.find_pairs(data):

            # 2. Normalize schema
            norm_a = MyInternalSchema(
                id=raw_a["uuid"],
                name_field=raw_a[self.name_map],
                text_field=raw_a[self.text_map]
            )
            norm_b = MyInternalSchema(
                id=raw_b["uuid"],
                name_field=raw_b[self.name_map],
                text_field=raw_b[self.text_map]
            )

            # 3. Yield the clean, standardized pair
            yield ERCandidate(left=norm_a, right=norm_b)
```

### core.Module (Base Class - The "Flow")

**Definition:** The Module (or "Flow") is the "Brain" of the pipeline. It is the central Estimator that performs the pairwise comparison.

**What it's not:** It is not a data loader. It must operate on the clean, normalized schema provided by the Blocker. This separation of concerns is what makes it reusable.

**Key Methods (both are `abstractmethod`s — a subclass must implement both):**

- `forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]`: your custom comparison logic — one judgement per pair.
- `inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> ScoreInspectionReport`: label-free exploration of a run's score distribution (used before you have ground truth).

**Example (Custom Judge):**

`MyProductJudge` is a *user-defined* `Module` subclass — `Module` is the base
class; there is no `Flow` type in langres. This one combines two rapidfuzz
similarities with a tunable weight (no torch, no learnable model — see
`WeightedAverageJudge` / `EmbeddingScoreJudge` in `langres.core.judges` for the
shipped judges):

```python
from collections.abc import Iterator

import rapidfuzz.fuzz

from langres.core import ERCandidate, Module, PairwiseJudgement
from langres.core.reports import ScoreInspectionReport

class MyProductJudge(Module[MyInternalSchema]):
    def __init__(self, name_weight: float = 0.5) -> None:
        self.name_weight = name_weight  # a tunable hyperparameter

    def forward(
        self, candidates: Iterator[ERCandidate[MyInternalSchema]]
    ) -> Iterator[PairwiseJudgement]:
        for pair in candidates:
            name_sim = rapidfuzz.fuzz.WRatio(pair.left.name_field, pair.right.name_field) / 100.0
            text_sim = rapidfuzz.fuzz.token_set_ratio(pair.left.text_field, pair.right.text_field) / 100.0
            score = self.name_weight * name_sim + (1.0 - self.name_weight) * text_sim
            yield PairwiseJudgement(
                left_id=pair.left.id,
                right_id=pair.right.id,
                score=score,
                score_type="heuristic",
                decision_step="weighted_rapidfuzz",
                provenance={"name_sim": name_sim, "text_sim": text_sim},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        # Required alongside forward(): summarize the score distribution and
        # suggest a threshold before you have labels. Body elided — the shipped
        # judges delegate to a shared implementation.
        ...
```

### core.Clusterer (Base Class)

**Definition:** Consumes the PairwiseJudgement stream and builds the final entity clusters.

**Key Methods:**

- `__init__(self, threshold: float = 0.5)`
- `cluster(self, judgements: Iterator[PairwiseJudgement] | list[PairwiseJudgement]) -> list[set[str]]`

**Behavior:** builds an undirected graph from every judgement whose `score >= threshold` and returns the connected components (full transitive closure, via networkx) — so a chain A–B, B–C merges A, B, and C even with no direct A–C edge. This is the single built-in strategy; there is no `method`/`hierarchical` option and no cannot-link `constraints` argument. For a merge-resistant alternative that resists that transitive over-merge, use `CorrelationClusterer` (§9).

**Example:**

```python
from langres.core import Clusterer

clusterer = Clusterer(threshold=0.75)
clusters = clusterer.cluster(judgements_stream)   # -> list[set[str]]
```

### core.optimizers.BlockerOptimizer (Optuna)

**Definition:** The one optimizer that ships today. It runs an Optuna study over
a **blocker's** hyperparameters (e.g. embedding model, `k_neighbors`) to maximize
a metric you compute in an objective function.

> There is **no** general `Optimizer` that "compiles"/"finetunes" a whole
> pipeline — no `compile()`, no `finetune()`, no PyTorch training loop. A general
> `Optimizer` over full pipelines is roadmap (`docs/ROADMAP.md`), not
> implemented. DSPy prompt optimization exists separately, inside `DSPyJudge`
> (`[llm]` extra).

**Constructor:**

- `BlockerOptimizer(objective_fn, search_space, primary_metric="value", direction="maximize", n_trials=50, wandb_kwargs=None)`
  - `objective_fn(trial, params) -> dict[str, float]` builds a blocker from
    `params`, runs the pipeline, and returns a metrics dict; `primary_metric`
    names which key to optimize.
  - `search_space`: `{"param": [choices...]}` for categorical, `{"param": (lo, hi)}` for integer ranges.

**Key Method:** `optimize(self) -> dict` — runs the study and returns the best hyperparameters.

**Example:**

```python
from langres.core.optimizers import BlockerOptimizer

search_space = {
    "embedding_model": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    "k_neighbors": (5, 50),
}

def objective(trial, params):
    # build a VectorBlocker from params, run the pipeline, score it
    # (pipeline/metric computation elided)
    return {"bcubed_f1": 0.85}

optimizer = BlockerOptimizer(
    objective_fn=objective,
    search_space=search_space,
    primary_metric="bcubed_f1",
    direction="maximize",
    n_trials=20,
)
best_params = optimizer.optimize()   # -> {"embedding_model": ..., "k_neighbors": ...}
```

(Optuna lives in the dev dependency group, not a runtime extra — `BlockerOptimizer`
is an eval-time tool, not part of the `link()`/`dedupe()` path.)

### core.Canonicalizer (`langres.core.canonicalizer`, M5/W2.3) — ✅ ships today

**Definition:** The "last mile" of Master Data Creation (Use Case 4): merge one
entity's records into a single **golden record** via field-by-field
*survivorship* rules. Dict-in / dict-out — it consumes the same raw record dicts
`resolve`/`assign`/`AnchorStore` pass around and emits one golden dict of the
same schema shape. It owns only the survivorship policy and knows nothing about
how the group was formed, so it composes with any grouping.

**Key methods:**

- `canonicalize(records: list[dict], *, entity_id: str | None = None) -> dict` —
  merge a whole group. Each attribute field is resolved by its configured
  strategy over the group; a field only *some* records carry is still filled
  (the rest count as missing for it). The golden record's `id` is `entity_id`
  when given, else the first record's id. A single-record group returns a copy of
  that record.
- `enrich(golden: dict, mention: dict, *, entity_id: str | None = None) -> dict`
  — the **enrichment loop**: fold a newly-linked mention into an existing golden
  record. It is exactly `canonicalize([golden, mention])` with `golden`'s id
  preserved — the *same* survivorship path, not a parallel one — so a sparse
  mention from `Resolver.assign` fills fields the golden record lacked.
- `save(path)` / `load(path)` — round-trip the policy through a pickle-free
  `canonicalizer.json` (config-registry seam; `type_name = "canonicalizer"`).

**Survivorship strategies** (named, per-field overridable; default `most_complete`):

| Name | Winner |
|------|--------|
| `most_complete` (default) | Value from the record with the most non-missing fields overall (trust the richest source); present beats absent. |
| `longest` | The longest non-missing string value. |
| `most_frequent` | The most common non-missing value (mode). |
| `most_recent` | Value from the record with the greatest `timestamp_field` (must be configured). |
| `first` / `source_priority` | First non-missing value in group order. |

All ties break deterministically to the **first-seen** value; `None` / empty
strings are "missing" while `0` / `False` are present values. `id` is stamped as
the master id, never survivorship'd.

**Example:**

```python
from langres.core import Canonicalizer

canon = Canonicalizer(
    default_strategy="most_complete",
    field_strategies={"name": "longest", "phone": "most_frequent"},
)
golden = canon.canonicalize(entity_records)      # merge a whole cluster/entity
golden = canon.enrich(golden, new_mention)       # fold in a linked sparse mention
```

## 6. Core API: langres.data

`langres.data` is the **benchmark dataset layer** — an import-light registry over the
bundled entity-resolution benchmark loaders that the eval harness runs against.
(There is no synthetic-data generator; `SyntheticGenerator` was never built.)

- `list_benchmarks() -> list[BenchmarkEntry]` returns each registered benchmark's metadata (name, task, domain, `loadable`) **without importing any loader**.
- `get_benchmark(name)` imports only the selected loader lazily and returns a ready `Benchmark` (records + gold clusters).

```python
from langres.data import list_benchmarks, get_benchmark

for entry in list_benchmarks():
    print(entry.name, entry.task, entry.loadable)
    # fodors_zagat linkage True / amazon_google linkage True / abt_buy linkage True / ...

bench = get_benchmark("fodors_zagat")   # loads just this one dataset
```

See [BENCHMARKS.md](BENCHMARKS.md) for the full portfolio (each dataset, why it's a
target, and its caveats), the `list_benchmarks` / `get_benchmark` discoverability
seam, and the bring-your-own-data `evaluate()` walkthrough.

### core.review.ReviewQueue

**Definition:** The flywheel's Human-in-the-Loop half. `core.review.ReviewQueue` is a JSONL-file-backed **snapshot** of a review selection — `write(items)` truncates and rewrites `review_queue.jsonl` so the queue always reflects exactly one selection (regenerate it, never hand-edit it). `core.review.select_for_review` reads `JudgementLog` rows and picks the pairs worth a human's attention.

**What it does:**

- `select_for_review(rows, strategy=...)` selects pairs by `"uncertainty"` (near the decision margin), `"disagreement"` (student vs. teacher verdicts differ), or `"audit"` (a seeded governance sample), returning `list[ReviewItem]`.
- `ReviewQueue(path).write(items)` snapshots that selection to `review_queue.jsonl`; items are ids-only unless you pass `records=` to `select_for_review`.
- The `langres` CLI labels the queue (`export-csv` → spreadsheet → `import-csv` → `corrections.jsonl`, or the `langres review` terminal loop). `core.harvest` folds those corrections back into `core.calibration.derive_threshold` / `fit()` — the active-learning loop.

**Example:**

```python
from langres import dedupe
from langres.core.judgement_log import JudgementLog
from langres.core.review import ReviewQueue, select_for_review

# 1. Log every judge call while resolving (the flywheel inlet).
dedupe(records, judge="string", threshold=0.6, log="judgements.jsonl")

# 2. Select the pairs worth a human's attention, near the decision margin.
rows = JudgementLog("judgements.jsonl").read()
items = select_for_review(rows, strategy="uncertainty", threshold=0.6)

# 3. Snapshot them to a queue the CLI can label.
ReviewQueue("review_queue.jsonl").write(items)
# $ langres export-csv review_queue.jsonl to_label.csv   # label in a spreadsheet
# $ langres import-csv  to_label.csv review_queue.jsonl  # -> corrections.jsonl
```

## 7. Core Data Contracts (Pydantic Models)

### ERCandidate[SchemaT]

The internal data wrapper passed into a Flow.

- `left: SchemaT`
- `right: SchemaT`
- `blocker_name: str`

### PairwiseJudgement

The rich data object passed out of a Flow. This is the auditable log of a decision.

- `left_id: str`
- `right_id: str`
- `score: float`: The combined score (0.0 to 1.0).
- `score_type: Literal["sim_cos", "prob_llm", "heuristic", "calibrated_prob", "prob_fs", "prob_rf", "prob_group_llm"]`: What kind of score is this? Critical for calibration and clustering. `prob_fs`/`prob_rf`/`prob_group_llm` (added W1.0) are reserved for a Fellegi-Sunter judge, an sklearn RandomForest judge, and a set-wise (group) judge respectively — none are implemented in core yet, but the literal is open for the branches that add them.
- `decision_step: str`: Which logic branch made this decision (e.g., "string_sim" or "llm_judge").
- `reasoning: Optional[str]`: The LLM's natural language explanation.
- `provenance: Dict[str, Any]`: A full audit trail (e.g., `{"model": "e5-small", "rapidfuzz_score": 0.85}`).

<!-- TODO(parse-error): another agent is revising the parse-error / evaluate() behaviour described in this paragraph (the score-0.0 abstain path). Do not edit this paragraph here. -->
**LLM-judge provenance keys.** `LLMJudge` / `DSPyJudge` / `SelectJudge` write
`model`, `cost_usd`, `provider`, the legacy `prompt_tokens` / `completion_tokens`
(kept for `JudgementLog`, `bootstrap.labelers`, `openrouter.make_token_cost_track`),
and — added here — a typed **`usage`** vector: `LLMUsage.model_dump()`
(`langres.core.usage`). It follows the OpenTelemetry GenAI vocabulary (snake_case,
SUBSET semantics): `input_tokens` / `output_tokens` are the *inclusive* totals
(`input_tokens` == `prompt_tokens`), and `cache_read_input_tokens`,
`cache_creation_input_tokens`, `reasoning_tokens` are subsets of them, plus the
serving `provider` and `model` id. LiteLLM already normalizes Anthropic's raw
`input_tokens` up to the inclusive total, so the subsets are never re-added. An
`LLMJudge` under `on_parse_error="abstain"` (the default) also sets
`provenance["parse_error"] = True` on a response its `response_parser` could not
parse (score `0.0`, distinguishable downstream); `evaluate()` /
`evaluate_judge_on_candidates()` surface the count as `JudgePairEval.n_parse_errors`
and warn when non-zero.

**`LLMJudge` paper-replication seams (constructor).** To run a published paper's
prompt without subclassing: `response_parser` (default `parse_score_response`; the
shipped `parse_binary_yes_no` covers the Yes/No prompt family), `record_serializer`
(default `default_record_serializer` = `model_dump_json(indent=2)`), `system_prompt`
(sends `system`+`user` when set), and `on_parse_error` (`"abstain"` | `"raise"`).
`prompt_template` requires literal `{left}`/`{right}` placeholders and preserves all
other braces verbatim (a paper's JSON schema is safe). `temperature` defaults to
`0.0`. `system_prompt` / `on_parse_error` serialize via `config`; the parser and
serializer callables do not (they revert to defaults on `Resolver.load`, like the
client).

### ClusterDelta (`langres.core`, M5/W2.2)

The result of one incremental `Resolver.assign(record)` / `AnchorStore.assign(record)` — the outcome of assigning a single new record against a prior batch's anchor set.

- `type: Literal["new", "link", "merge", "split", "reject"]`: `new` (a fresh entity was minted) or `link` (attached to an existing entity). `merge`/`split`/`reject` are **reserved** for the wider entity-maintenance surface (W2.4/M6) so the contract stays stable; W2.2 only ever emits `new`/`link`.
- `record_id: str`: The assigned record's id.
- `entity_id: str`: The **stable** entity id the record now belongs to (freshly minted for `new`, existing for `link`). Never changes on later assigns (append-only allocator).
- `matched_anchor_ids: list[str]`: Anchor record ids that cleared the threshold (evidence for a `link`; empty for `new` and for the idempotent already-assigned-id `link`).
- `score: Optional[float]`: Best matching score across judged candidates (observability); `None` when there were no candidates or on the idempotent path.
- `reasoning: Optional[str]`: Human-readable note about the decision.

## 8. Group + Fit Contracts (W1.0) + SelectJudge (W1.1)

W1.0 froze two interfaces that later branches build against: this section
documents the contracts. W1.1 shipped the first concrete `GroupwiseModule` —
`SelectJudge` (`langres.core.modules.select_judge`) — proving the contract
against a real set-wise judge; `FellegiSunterJudge` / `RandomForestJudge` (trained
judges over `ComparisonVector`, a later branch) still build against the
contracts below without shipping yet.

### ERCandidateGroup[SchemaT] (`langres.core.groups`)

The set-wise counterpart to `ERCandidate`: "one anchor + K candidate
members" instead of one pair.

- `anchor: SchemaT`
- `members: list[SchemaT]`
- `group_id: str`

`derive_groups_from_pairs(candidates)` derives groups from an existing
pairwise `ERCandidate` stream by grouping on each pair's `left` entity. It is
**buffered and anchor-skewed**: an entity that never appears as `left` (e.g.
because an upstream blocker canonicalizes pair order by id) never becomes an
anchor. It is lossless over *pairs* despite the skew (flattening the derived
groups back to canonical pairs recovers exactly the input pair set — no
dupes, no losses), but it is **not** representative of a blocker's true
candidate structure, so it must not be used to benchmark a set-wise judge.

### Blocker.stream_groups() (`langres.core.blocker`)

- **Default** (inherited by every `Blocker` subclass): derives groups from
  `self.stream(data)` via `derive_groups_from_pairs` — buffered/skew-prone,
  as above. Exists so every blocker gets a working `stream_groups()` for
  free; not for benchmark use.
- **`VectorBlocker.stream_groups()`** overrides this natively: its kNN search
  is already per-anchor, so each entity's own search result IS its group —
  one group per entity, its (deduplicated) k nearest neighbors as members, no
  derivation, no skew. This is the implementation a set-wise judge should be
  benchmarked against.

Both forms satisfy the same pairs-equivalence property EXACTLY, not just at
the SET level: the pairs recoverable from `stream_groups()` equal the pairs
from `stream()` — no losses AND no duplicates. `VectorBlocker.stream_groups()`
threads a single `seen_pairs` set across all entities (same iteration order,
same first-seen-wins rule `stream()` uses), so each undirected pair is
assigned to exactly one group — whichever anchor is processed first. Without
this cross-anchor dedup, two mutual nearest neighbors (A's nearest neighbor is
B AND B's nearest neighbor is A — common with real ANN indexes on
near-duplicate records) would appear as a member edge in *both* groups, and a
consumer that treats each group as an independent unit of work (e.g. a
set-wise judge issuing one LLM call per group, for cost accounting) would
emit and charge for the same undirected pair twice. See
`VectorBlocker.stream_groups()`'s docstring and the count-based
regression/property tests in `tests/core/blockers/test_vector.py`
(`test_vector_blocker_stream_groups_dedupes_mutual_neighbor_pairs`,
`test_vector_blocker_stream_groups_pairs_equivalence_property`).

### GroupwiseModule (`langres.core.module`)

`GroupwiseModule` **is a `Module`** — it does not introduce a parallel
execution path. Its concrete `forward()` derives groups internally from
whatever pairwise `ERCandidate` stream it receives (via
`derive_groups_from_pairs`, the same buffered default as above — `forward()`
only ever sees a flat pairwise stream, never the blocker object, so it
cannot reach a blocker's native grouping) and dispatches to the abstract
`forward_groups()`, decomposing the result back to `Iterator[PairwiseJudgement]`.
Concrete set-wise judges implement only `forward_groups()` and
`inspect_scores()`. Because the group structure never leaves `forward()`,
the Resolver execution spine (`Resolver._judgements` → `module.forward`),
`inspect_scores`, the JudgementLog boundary, and benchmark dispatch
(`BudgetedModuleRunner`, `run_method`) all work unchanged.

```python
# Illustrative pseudocode predating the shipped implementation below --
# `self._call_llm` / `self._last_call_cost` are placeholders, not real
# SelectJudge attributes (see the real cost/call plumbing in select_judge.py).
class SelectJudge(GroupwiseModule[MySchema]):
    def forward_groups(self, groups: Iterator[ERCandidateGroup[MySchema]]) -> Iterator[PairwiseJudgement]:
        for group in groups:
            # One LLM call per group: "which of these K candidates match the anchor?"
            selected_ids = self._call_llm(group)
            judgements = [
                PairwiseJudgement(
                    left_id=group.anchor.id,
                    right_id=member.id,
                    score=1.0 if member.id in selected_ids else 0.0,
                    score_type="prob_group_llm",
                    decision_step="select_judge",
                    provenance={},
                )
                for member in group.members
            ]
            yield from stamp_group_cost(judgements, call_cost_usd=self._last_call_cost, group_id=group.group_id)

    def inspect_scores(self, judgements, sample_size=10):
        ...
```

**Shipped (W1.1):** `langres.core.modules.select_judge.SelectJudge` is the
real implementation of the skeleton above — a DSPy `ChainOfThought` over a
`SelectSignature` asking the LLM to select **at most one** matching candidate
id per group (mirroring ComEM's own "selecting" strategy: Wang et al., COLING
2025, choosing "the" single most-likely match, not an arbitrary subset). A
malformed response, a selection naming a candidate outside the group, or a
selection of more than one candidate (`select_error`, CEO #12) all map to
whole-group "no match" judgements carrying `provenance["select_error"]` —
never a raised exception. Selectable by name as `"select_judge"` in
`langres.methods` (the benchmark/method-registry dispatch site only — not
wired into `Resolver.from_schema(judge=...)` or the verbs' `judge="auto"`
dispatch, since it is not yet part of the zero-label default path). See
`data/benchmarks/w1/W1_RESULTS.md` for the measured call-count/cost reduction
(35.56x on Amazon-Google) and group-size distribution.

### Group-call cost convention (E5)

One LLM call scores a whole group (K pairs). Pricing each of the K resulting
judgements at the call's full cost would silently overcount total spend by a
factor of K. `stamp_group_cost(judgements, call_cost_usd, group_id)`
(`langres.core.module`) applies the fix: the full `call_cost_usd` goes on the
**first** judgement's `provenance["cost_usd"]`, every sibling gets `$0`, and
`provenance["group_id"]` is set on all of them. Existing cost aggregation
(`_judgement_cost`/`_cost_track` in `langres.core.benchmark`, which already
read `provenance["cost_usd"]`) then sums a group to exactly one call's cost
with no changes on their end.

`stamp_group_cost` also sets `provenance["group_end"] = True` on (only) the
**last** judgement of the group — a boundary marker that lets a consumer
draining a whole group from a lazy stream (`_SpendCappedModule.forward` in
`langres.core.presets`, the hard spend cap the verb layer wraps every judge
in) know exactly when to stop pulling, without peeking at the next
judgement's `group_id` — which for a real `GroupwiseModule` would resume the
generator into (and pay for) the next group before there's anything to
compare against.

**Atomicity caveat:** `BudgetedModuleRunner` scores exactly one `ERCandidate`
per `module.forward()` call. A `GroupwiseModule` run through it today derives
a single, trivial, size-1 group per call — so a group is never *split*
mid-call (there is never more than one pair per call to split), but a real
multi-pair group is also not yet *batched* into one priced call: no cost
amortization happens through the runner yet. Extending the runner (or adding
a group-aware variant) to pre-flight and price whole groups atomically is
deferred to the branch that lands the first concrete `GroupwiseModule`.

### Fit hooks (`langres.core.fit`)

Two runtime-checkable, structural `Protocol`s — **not** abstract methods on
`Module` (that would break every existing, non-learnable module):

- `SupervisedFitMixin.fit(candidates: Iterator[ERCandidate[SchemaT]], labels: Sequence[bool]) -> None`
- `UnsupervisedFitMixin.fit_unlabeled(candidates: Iterator[ERCandidate[SchemaT]]) -> None`

A module opts in by implementing the method with the matching name — no
subclassing required. `Resolver.fit(data, labels=None)` detects this with
`isinstance(module, SupervisedFitMixin)` / `isinstance(module, UnsupervisedFitMixin)`:

- Module implements `SupervisedFitMixin`: `labels` is required; omitting it
  **raises** (a genuinely trainable module silently not being trained is the
  exact footgun this hook exists to prevent).
- Module implements `UnsupervisedFitMixin`: `fit_unlabeled` is called
  unconditionally; passing `labels` to it raises (they would otherwise be
  silently ignored).
- Module implements **neither** hook (e.g. `WeightedAverageJudge`): `fit()`
  is a no-op returning `self` — the original sklearn-style symmetry is
  preserved for non-learnable pipelines — unless `labels` was passed, which
  raises rather than silently discarding them.

No concrete judge implements either hook yet; `FellegiSunterJudge` and
`RandomForestJudge` (a later branch) are the first.

## 9. Blocking Algebra + Merge-Resistant Clustering (W1.3)

### KeyBlocker (`langres.core.blockers.key`)

Buckets records by a configurable key and emits all pairs within each bucket.
Mirrors `AllPairsBlocker`'s declarative/callable split: `key_field=` (a schema
attribute name, serializable) or `key_fn=` (full callable control, not
serializable), same mutual-exclusion rule as `schema=`/`schema_factory=`.
`normalize=True` (default) lowercases + strips non-`None` keys before
bucketing. A record whose key extracts to `None` gets no candidates from this
blocker — trading recall for precision/speed, by design; compose with a
recall-oriented blocker (e.g. `VectorBlocker`) via `CompositeBlocker` to
recover it.

### CompositeBlocker (`langres.core.blockers.composite`)

Set algebra over 2+ child `Blocker`s: `op="union"` (default,
recall-maximizing — a pairs-superset of every child), `"intersection"`
(pairs in every child), or `"difference"` (`children[0]` minus the rest).
Dedups by the canonical undirected pair key with first-seen semantics (the
same guarantee every base blocker already gives). `blocker_name` on each
surviving candidate is rewritten to
`"composite_{op}(label1+label2+...)"`, naming exactly which child(ren)
produced *that* pair — real per-pair provenance, not just "this came from a
composite." Necessarily buffers every child in full (set membership must be
known across all children before a pair can be kept or dropped) — like
`derive_groups_from_pairs`, this trades the streaming-first contract for
correctness. Relies on the inherited `Blocker.stream_groups()` default
(no native per-anchor override): composite pair sets, especially under
intersection/difference across heterogeneous children, have no natural
single-anchor structure the way `VectorBlocker`'s kNN search does.
Registered (`"composite_blocker"`) and config-serializable as long as every
child is too (a child with out-of-band state, e.g. a built `VectorBlocker`
index, is not preserved through a composite's `config`/`from_config`
round-trip — persist such a pipeline via the `Resolver` artifact instead).

Measured on Fodors-Zagat + Amazon-Google (Pair-Completeness / Reduction-Ratio,
composite vs. each dataset's pinned `VectorBlocker` alone):
`examples/research/w1_blocking_algebra_output.md`.

### CorrelationClusterer (`langres.core.clusterers.correlation`, "C6")

A `Clusterer` subclass (drop-in for the `clusterer=` slot; inherits
`config`/`from_config`/`evaluate`/`inspect_clusters` unchanged, overrides only
`cluster()`). The base `Clusterer` builds a graph from edges `>= threshold`
and takes connected components — full transitive closure, so a chain of
edges (A-B, B-C) with no direct A-C edge still merges all three (the
documented M3 −0.63 BCubed over-merge failure mode). `CorrelationClusterer`
implements the *pivot algorithm* for correlation clustering (Ailon, Charikar
& Newman, JACM 2008): process nodes highest-confidence-edge-first
(deterministic, ties broken by id); each pivot's cluster is itself plus only
its *direct* neighbours `>= threshold`. A node with only an indirect path to
a cluster is never pulled in by transitivity alone — merge-resistant relative
to the base `Clusterer` — while a genuinely well-connected clique (every pair
directly compared and matched) still merges fully under both.

**Not the default.** Benchmarked head-to-head against the base `Clusterer` on
Fodors-Zagat + Amazon-Google (same blocking + judge pipeline, only the
clusterer differs): a wash on Fodors-Zagat (+0.0006 BCubed F1), a clear win
on Amazon-Google (+0.0324 BCubed F1, +0.0715 precision at −0.016 recall) —
see `examples/research/w1_blocking_algebra_output.md` for the full tables and the
default-flip decision (kept opt-in; recommended for harder/messier
entity-resolution problems, not flipped globally on a single hard-dataset
win).
