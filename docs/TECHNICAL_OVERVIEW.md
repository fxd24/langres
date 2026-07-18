# langres: Technical Documentation & API Reference

Welcome to the langres documentation. This document provides a deep dive into the layered API, the core architectural pillars, and the data contracts that power the library.

## 1. The langres Layered API

langres exposes a **three-layer API** — each layer a thin shell over the one below — so common tasks are one-liners while bespoke pipelines stay fully composable:

- **Named architectures — `langres.architectures.FuzzyString` / `VectorLLMCascade`** (High-Level): the schema-optional, zero-label front door and the recommended entry point. A whole ER pipeline is a class you construct — `FuzzyString()` ($0, offline, no key: it has no paid model slot) or `VectorLLMCascade(llm=...)` (paid, spend-capped at $1 by default, because you constructed it — never because a heuristic sniffed an API key) — and call `.dedupe(records)` / `.compare(a, b)` on. Results are self-describing (`LinkVerdict`; a `DedupeResult` carrying `architecture` / `backbone` / `score_type` / `threshold` — every result names the model class *and* the underlying backbone that ran).
- **`langres.Resolver`** (Mid-Level, a plain alias of the `ERModel` base class every named architecture subclasses): the declarative pipeline. `Resolver.from_schema(schema, matcher=...)` wires a default blocker + comparator + judge + clusterer from a Pydantic schema; `.resolve(records)` runs it and `.save`/`.load` serialize it (config registry, no pickle).
- **`langres.core`** (Low-Level): the "power-user" API — the base classes (`Matcher`, `Blocker`, `Comparator`, `Clusterer`, judges) you compose into entirely new logic from scratch.

> There is **no** `langres.tasks` / `langres.flows` module — those names were never built. There is also no `langres.link` / `langres.dedupe` module-level verb and no `matcher="auto"` key-sniffing — naming a model is the caller's job, not a heuristic's. The real layering is named architectures → `Resolver` (`ERModel`) → `langres.core`.

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
  `WeightedAverageMatcher` scorer (the `matcher="string"` path).
- **networkx** — `Clusterer` builds an undirected graph from the judgements that
  clear the threshold and takes connected components (transitive closure). This
  is the *only* clustering backend — there is no scipy / hierarchical path.
- **numpy** — vector math shared by the embedding helpers.

**Opt-in extras** (`pip install langres[<extra>]`):

- **`[semantic]`** — sentence-transformers / torch / faiss-cpu / qdrant-client
  (plus onnxruntime/optimum, fastembed): the embedding + ANN stack behind
  `VectorBlocker`, `EmbeddingScoreMatcher`, and `matcher="embedding"`.
- **`[llm]`** — litellm / dspy-ai / openai: `LLMMatcher`, the DSPy-compiled
  `DSPyMatcher`, `Resolver.from_schema(matcher="zero_shot_llm")`, and the
  `VectorLLMCascade` architecture's escalation backbone. DSPy prompt
  optimization is real, but it lives *inside* `DSPyMatcher` — it is not an
  automatic compile pass over a whole pipeline.
- **`[trained]`** — scikit-learn: `RandomForestMatcher` and
  `training.calibration.derive_threshold`. Note the BCubed / pairwise metrics in
  `core.metrics` are a **vetted internal implementation** (Amigó et al. 2009),
  *not* sklearn — sklearn is pulled in only by these trained-judge / calibration
  paths.
- **`[eval]`** — ranx: the ranking metrics (MRR / NDCG / MAP) in
  `core.metrics.evaluate_blocking_with_ranking`, imported lazily so the rest of
  `core.metrics` / `core.benchmark` stays importable without it. (There is no
  `pytrec_eval` anywhere in the tree.)
- **`[hub]`** — huggingface-hub: the optional remote transport used by
  `ERModel.from_pretrained(..., repo_id=...)` and `ERModel.push_to_hub(...)`.
  Local `save_pretrained` / `from_pretrained` stays available without this
  extra.

Hyperparameter search is opt-in too: `langres.autoresearch.blocker_optimizer.BlockerOptimizer` (Optuna)
tunes *blocker* parameters — see §5. A general `Optimizer` over full pipelines is
roadmap (`docs/ROADMAP.md`), not implemented.

**Lazy loading.** These heavy symbols resolve through a PEP 562 `__getattr__`
seam in `langres/core/__init__.py` and `langres/clients/__init__.py`, so a bare
`import langres` never drags torch / litellm / faiss / scikit-learn / ranx into
`sys.modules`. `tests/test_import_budget.py` guards this.

**Where the exports live.** `langres/__init__.py` and `langres/core/__init__.py`
are thin aggregators: they keep the `__getattr__` seam but hold no per-*name*
content. Each export — its eager import or its lazy `name -> module` + `[extra]`
entry — lives in a per-domain fragment under `langres/_exports/` and
`langres/core/_exports/` (see those packages' docstrings for the contract). To
add a public symbol, edit the one fragment that owns its domain; the aggregator
composes `__all__` and the lazy maps from whatever the fragments declare. This
split exists because a single sorted ~100-name `__all__` is unmergeable: N
parallel work-streams each inserting a name at its sorted position produce N
guaranteed conflicts.

### core.review.ReviewQueue (Human-in-the-Loop)

The HITL system splits a storage backend from a labeling surface:

- **`core.review.ReviewQueue`:** the storage backend — it writes the selected pairs to a plain `review_queue.jsonl` snapshot (one JSON line per pair; ids-only unless you pass `records=` to join content back on). `core.review.select_for_review` picks *which* pairs are worth a human's attention (uncertainty / disagreement / audit strategies).
- **The `langres` CLI:** the labeling surface. `langres export-csv` turns a queue into a spreadsheet, a reviewer fills the `label` column, and `langres import-csv` reads it back into a `corrections.jsonl` log; `langres review` is the equivalent quick terminal loop. The harvested corrections feed `core.harvest` and `training.calibration.derive_threshold` back into the pipeline.

### Observability & Tracing

Three shipped layers, all built on `PairwiseJudgement.provenance`:

- **Per-call provenance:** every LLM judge writes `model`, `cost_usd`,
  `provider`, and a typed `LLMUsage` token vector following the OpenTelemetry
  GenAI vocabulary into provenance (see §7, "LLM-judge provenance keys").
- **The judgement log:** `tracking.judgement_log.JudgementLog` / `LoggingMatcher`
  persist every judge call (ids, score, verdict, model, cost) to JSONL — the
  flywheel's signal inlet.
- **Experiment tracking:** `tracking.runs` (`RunContext`/`RunRecord`, content-
  addressed `recipe_id`, JSONL `RunStore`) plus the pluggable
  `tracking.trackers.ExperimentTracker` seam with lazy `MlflowTracker` /
  `WandbTracker` adapters. (Observability lives in the sibling `langres.tracking`
  package, not in `core`; the `langres.core` facade still re-exports these names.)

## 3. High-Level API: named architectures (`ERModel.dedupe` / `.compare`)

`langres.architectures.FuzzyString` / `VectorLLMCascade` are the one-liner front door: a whole ER pipeline is a class you construct, and `.dedupe()` / `.compare()` are methods on it. There is no `matcher="auto"` that sniffs your environment for an API key — the free path (`FuzzyString`, no paid model slot, cannot spend) and the paid path (`VectorLLMCascade(llm=...)`, spend-capped) both need you to name which one you want. Both infer an ephemeral schema from your records' keys (or take an explicit `schema=<YourModel>` at construction) and run the full blocking → scoring → clustering pipeline. The runnable, offline version lives in [`examples/quickstart_models.py`](https://github.com/fxd24/langres/blob/main/examples/quickstart_models.py).

### ERModel.dedupe

**Definition:** Group a batch of records into entity clusters (single-source deduplication).

`model.dedupe(records, *, log=None)` returns a `DedupeResult` — a `list[set[str]]` of the multi-record clusters (singletons are dropped) that additionally carries `architecture`, `backbone`, `score_type`, and the effective `threshold`. `architecture` is the model class that ran (e.g. `"FuzzyString"`) — the topology, named by the caller at construction. `backbone` is the underlying model that actually scored the batch: the resolved LLM id (e.g. `"openrouter/openai/gpt-4o-mini"`) for `VectorLLMCascade`, the sentence-transformers embedder name for `Resolver.from_schema(matcher="embedding")`, an injected `Matcher`'s own `model` attribute, and `None` for pure-string similarity (`FuzzyString`). The same value backfills the `model` column of `JudgementLog` rows (`log=`) whenever the matcher doesn't stamp its own, so result and log always agree.

**Example** (offline — `FuzzyString` has no paid model slot, so no API key or network is ever needed):

```python
from langres.architectures import FuzzyString

records = [
    {"id": "1", "name": "Acme Corporation", "city": "New York"},
    {"id": "2", "name": "Acme Corp", "city": "New York"},
    {"id": "3", "name": "Totally Different Co", "city": "Chicago"},
]

result = FuzzyString(threshold=0.6).dedupe(records)
# result -> [{'1', '2'}]   (singleton "3" is dropped)
print(result.architecture, result.backbone, result.score_type)   # "FuzzyString" None "heuristic"
```

`VectorLLMCascade(llm=...)` instead runs vector blocking, a free embedding student on every pair, and escalates only the uncertain band to a real LLM judge from the `llm=` backbone you named (needs the `[llm]` + `[semantic]` extras) — no key-sniffing, no silent fallback: without a valid key the first paid call fails with the provider's own error. Every model — `FuzzyString` included, for symmetry — runs under a default $1 spend cap (`budget_usd=`); a breach raises `BudgetExceeded` (root-exported) carrying the partial judgements.

### ERModel.compare

**Definition:** Decide whether **two records** are the same entity — a single pairwise verdict.

`model.compare(left, right, *, log=None)` returns a `LinkVerdict` — truthy iff it's a match — carrying `.score` (`float | None`: a *decider* judge, e.g. a binary Yes/No `LLMMatcher`, has no score), `.architecture`, `.backbone` (the resolved underlying model — same contract as `DedupeResult.backbone` above), `.score_type`, the effective `.threshold`, and `.reasoning`. If the matcher **abstains** — neither decides nor scores, e.g. an `LLMMatcher` whose response fails to parse under the default `on_parse_error="abstain"` — `.compare()` raises `MatcherAbstainedError` (root-exported) instead of fabricating a match/no-match verdict; a caller writing `if verdict.match:` would otherwise read "I don't know" as a confident no. (`.dedupe()` instead leaves an abstained pair unmerged, so one bad judgement can't sink a whole batch — see `resolver.py`'s `dedupe`/`compare` docstrings for why the two differ.)

```python
from langres.architectures import FuzzyString

verdict = FuzzyString(threshold=0.6).compare(
    {"id": "a", "name": "Acme Corp", "city": "New York"},
    {"id": "b", "name": "Acme Corporation", "city": "New York"},
)
if verdict:                       # LinkVerdict is truthy iff it's a match
    print(verdict.score, verdict.architecture)   # 0.86 "FuzzyString"
```

> **Cross-source / incremental linking is not built yet.** `.compare()` above compares a single pair, not two datasets. The dataset-to-dataset methods `Resolver.link` / `Resolver.stream_against` are reserved `NotImplementedError` stubs (roadmap M5) — do not treat them as working. `.compare()` is deliberately not named `link`: that name is reserved for `Resolver.link`.

## 4. Mid-Level API: the `Resolver`

The named architectures are thin sugar over `Resolver` (a plain alias of the `ERModel` base class — `FuzzyString` and `VectorLLMCascade` both subclass it). Drop to it directly when you want an explicit, serializable pipeline built from a Pydantic schema with a hand-picked matcher.

```python
from pydantic import BaseModel
from langres import Resolver

class Company(BaseModel):
    id: str
    name: str
    city: str

resolver = Resolver.from_schema(Company, matcher="string", threshold=0.6)
clusters = resolver.resolve(records)      # -> list[set[str]]
resolver.save("company_resolver.json")    # config-registry serialization (no pickle)
# later, in a fresh process:
resolver = Resolver.load("company_resolver.json")
```

`from_schema` auto-derives a missing-aware `StringComparator` from the schema's string fields, a `WeightedAverageMatcher` scorer, an `AllPairsBlocker` (or a `VectorBlocker` when `matcher="embedding"`), and a `Clusterer`. `matcher=` accepts `"string"` (default), `"embedding"`, `"zero_shot_llm"`, `"prompt_llm"` (bring-your-own-prompt, with `prompt_template` / `system_prompt` / `response_parser` kwargs — a named parser makes the whole prompt-judge artifact `save`/`load` round-trippable), `"random_forest"` (a supervised sklearn `RandomForestMatcher` over the comparator's per-feature similarities — needs the `[trained]` extra and is trainable, so `fit(records, pairs=...)` / `labels=...` it with labeled data before it can score), or a `Matcher` instance. This is the low-level, explicit switch: **no** `"auto"` key-resolution exists anywhere in langres — a named architecture doesn't sniff a key either, it just fixes its own topology instead of taking a `matcher=` string. A paid matcher built here **is** spend-capped: `Resolver(..., budget_usd=)` / `from_schema(..., budget_usd=)` defaults to `DEFAULT_BUDGET_USD` ($1.00) and the cap binds for the Resolver's whole lifetime, so two `resolve()` calls share one budget rather than getting one each. `budget_usd=None` means "the default", **not** "uncapped" — pass `langres.core.spend_cap.UNCAPPED_BUDGET_USD` (`float("inf")`) to opt out deliberately. The bound is honest but not magic: spend is capped at `budget_usd` **plus at most one further call**, since an LLM call's cost is only knowable once it has been made. Scope, precisely: the cap meters **every seam that scores through the matcher** — `resolve`, `predict`, `fit`, and `AnchorStore.assign` — because they all route through one internal capped-scorer accessor. The single exception is `distil()` / `fit(method=MIPRO())`: DSPy's compile calls never reach the matcher, so that ledger cannot see them; it caps them via its own `method.budget_usd` monitor, which records $0 until DSPy-compile spend capture lands (issue #100) — so a paid `MIPROv2` compile is **effectively unbounded** today.

**`"prompt_llm"` example** — run a published paper's prompt, or your own, through the low-level switch:

```python
from langres import Resolver

resolver = Resolver.from_schema(
    Company,
    matcher="prompt_llm",
    prompt_template="Do these match? Answer Yes or No.\nA: {left}\nB: {right}",
    response_parser="binary_yes_no",   # a registered, serializable parser name
)
verdict = resolver.compare(a, b)
verdict.architecture   # "ERModel" (from_schema returns the base class, not a named architecture)
verdict.backbone       # "openrouter/openai/gpt-4o-mini" (override with model=)
```

### What `save`/`load` records: components *and* class

`resolver.json` is an `ArtifactManifest`: `artifact_version`, `langres_version`,
an optional `model_class`, and either the ordered four-slot `components` or the
ordered explicit-chain `ops`. Stateful components/operations use sidecar
directories; no pickle or downloaded Python is executed.

`model_class` is the *architecture's* identity: the name a Resolver subclass
registered with `langres.core.register_model("fuzzy_string")`. `save` stamps it;
`load` looks it up and reconstructs **that** class, so `Resolver.load(<a
FuzzyString artifact>)` hands back a `FuzzyString`, not a plain `Resolver`.
Without it, a save/load would quietly erase which architecture a pipeline was.

It is **optional on purpose**, and the compatibility rules follow from that:

- **Absent ⇒ plain `Resolver`** — which is exactly what every pre-0.4 artifact
  is, and what an *unregistered* Resolver subclass still saves as (not an error;
  it degrades to the old behaviour).
- **Classic artifacts stay version `"1"`; explicit chains use version `"2"`.**
  The reader accepts both layouts. A classic model therefore keeps identical
  bytes and recipe identity as the explicit-chain format evolves additively.
- Models are their **own registry namespace** (`register_model` / `get_model` /
  `model_type_name`), separate from components and schemas. A component fills a
  slot; a model owns the slots — sharing one namespace would let
  `"type_name": "fuzzy_string"` resolve into a blocker slot.
- `config_dict()` still emits only `components`, so `model_class` stays **outside**
  `recipe_id`'s hash domain and no existing recipe hash forks.

> **A registered model's `__init__` stays ergonomic.** `load` reconstructs
> through `from_components` (a classmethod that builds directly from the four
> slots), **not** by calling the class's own `__init__` with component
> keywords. So a named architecture is free to narrow its constructor to its
> own signature (`FuzzyString(threshold=0.8)`, no `blocker=`/`comparator=`/…)
> without breaking `load` — `Resolver.load(<a FuzzyString artifact>)` hands
> back a `FuzzyString` either way. This was a real constraint before the
> `from_components` split landed (a registered architecture's `__init__` had
> to keep accepting the raw component keywords or `load` raised `TypeError`);
> it no longer applies.

Both name-dispatch paths — `from_schema` and the benchmark harness (`langres.methods`) — resolve judge names through the single **method registry** (`langres.core.method_registry`): one `MethodSpec` per name carrying its builder, `score_type`, `default_threshold`, and `default_model`. A name means the same thing everywhere; `/` in a method id is reserved for future `author/method` namespacing of third-party methods (model ids like `openrouter/openai/gpt-4o-mini` keep their slashes in the orthogonal `model=` kwarg). (A third path, the deleted verbs' `core.presets.build_judge`, existed before named architectures replaced the verbs and resolved names the same way; naming a model explicitly replaced it, not a better heuristic.)

See [DX_RESOLVER.md](DX_RESOLVER.md) for the before/after of the manual lambda pipeline vs. the declarative `from_schema` + `save`/`load` path.

### Explicit topology authoring and execution

Advanced users can compose `Source -> Score/Select -> ClusterStage` directly
with `ERModel.from_topology(ops=[...])`. The public authoring contracts live at
`langres.core`: `Op`, `Score`, `Select`, `Source`, `ClusterStage`,
`ThresholdSelect`, `TopKSelect`, `Sequential`, `Feasible`, and the optional
`SpendMonitorBindable` capability for a `Spending` Op.

An explicit model derives `.schema` and `.is_bound` from its Source. Both classic
and explicit vector retrieval call `Source.prepare(records)` before streaming,
so a `BlockerSource(VectorBlocker(...))` builds an unbound index, reuses it for
the same corpus, and rebuilds it for changed input. `dedupe(log=...)` and
`compare(log=...)` wrap every `MatcherScore` for that call without mutating the
saved topology. Multi-stage rows carry a stable `stage_id`, the model identity
of that scorer, and a binary `verdict` only when its score feeds a
`ThresholdSelect` directly; retrieval/top-k scores log `verdict=None` instead of
borrowing the final match threshold. The logged `stage_id` is exactly the
corresponding `execution_plan().steps[*].stage_id`, so logs and plans join
without a second naming convention. `compare` walks Scores and applicable
Selects in order, and its `LinkVerdict.backbone` names the scorer that actually
ran before the deciding Select.

`model.execution_plan()` returns ordered, stable stage ids derived from each
stage's safe runtime metadata plus its ordinal. Runtime inspection does not
require artifact registration: runnable custom stages and opaque schema
factories can be planned and executed, while `save()` remains fail-closed and
requires registered serializers. `model.execute(records,
observer=...)` runs that same Op spine and returns selected pairs, clusters, and
immutable start/finish/failure events. Observers receive counts and durations,
never records or mutable carriers; their return value is ignored. Callback
exceptions cannot abort or alter inference and are surfaced as sanitized
type-only diagnostics; exception messages are suppressed so observer metadata
cannot leak record or credential content. Stage-failure events follow the same
rule: observers see only a bounded exception type plus a generic failure
message, while the original exception is still re-raised to the direct caller.

A component-free custom `Score` or `Select` can opt into safe persistence:

```python
from pydantic import BaseModel, ConfigDict

from langres.core import Score, register_op

class AcmeScoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    weight: float

@register_op("acme_score")
class AcmeScore(Score):
    config_model = AcmeScoreConfig

    def forward(self, pairs): ...

    @property
    def config(self) -> dict[str, object]: ...

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "AcmeScore": ...
```

The strict, extra-forbidding `config_model` is required. Artifact loading
validates the complete role-specific parameter envelope before any nested
component lookup or construction. Built-in roles use the same validator seam,
so missing, unknown, or wrongly typed parameters fail closed.

Registration is fail-closed and exact-class. Loading an unknown role raises; it
does not import a module named by the artifact. Subclasses register separately
so parent serializers cannot silently drop their state.

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

### core.Matcher (Base Class - The "Flow")

**Definition:** The Matcher (or "Flow") is the "Brain" of the pipeline. It is the central Estimator that performs the pairwise comparison.

**What it's not:** It is not a data loader. It must operate on the clean, normalized schema provided by the Blocker. This separation of concerns is what makes it reusable.

**Key Methods:**

- `forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]`: your custom comparison logic — one judgement per pair. This is the **only `abstractmethod`**: implementing it is the whole of what it means to be a Matcher.
- `inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> ScoreInspectionReport`: label-free exploration of a run's score distribution (used before you have ground truth). **Optional and opt-in** — implement it to satisfy the `Inspectable` Protocol (`langres.core.inspection`, re-exported from `langres.core`); callers detect it with `isinstance(matcher, Inspectable)`. Skip it and your Matcher still constructs and runs; only the label-free inspection path is unavailable (the two wrappers that forward it — the spend cap and the JudgementLog — raise `TypeError` naming `Inspectable`).

**Example (Custom Judge):**

`MyProductJudge` is a *user-defined* `Matcher` subclass — `Matcher` is the base
class; there is no `Flow` type in langres. This one combines two rapidfuzz
similarities with a tunable weight (no torch, no learnable model — see
`WeightedAverageMatcher` / `EmbeddingScoreMatcher` in `langres.core.matchers` for the
shipped judges):

```python
from collections.abc import Iterator

import rapidfuzz.fuzz

from langres.core import ERCandidate, Matcher, PairwiseJudgement
from langres.core.reports import ScoreInspectionReport

class MyProductJudge(Matcher[MyInternalSchema]):
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
        # OPTIONAL (opt-in via the Inspectable Protocol): summarize the score
        # distribution and suggest a threshold before you have labels. Body
        # elided — the shipped judges delegate to a shared implementation.
        ...
```

### core.Clusterer (Base Class)

**Definition:** Consumes the PairwiseJudgement stream and builds the final entity clusters.

**Key Methods:**

- `__init__(self, threshold: float = 0.5)`
- `cluster(self, judgements: Iterator[PairwiseJudgement] | list[PairwiseJudgement]) -> list[set[str]]`

**Behavior:** builds an undirected graph from every judgement that `predicted_match` marks a match — a *decider*'s `decision`, else `score >= threshold`; an abstention (neither set) is excluded — and returns the connected components (full transitive closure, via networkx) — so a chain A–B, B–C merges A, B, and C even with no direct A–C edge. This is the single built-in strategy; there is no `method`/`hierarchical` option and no cannot-link `constraints` argument. For a merge-resistant alternative that resists that transitive over-merge, use `CorrelationClusterer` (§9).

**Example:**

```python
from langres.core import Clusterer

clusterer = Clusterer(threshold=0.75)
clusters = clusterer.cluster(judgements_stream)   # -> list[set[str]]
```

### langres.autoresearch.blocker_optimizer.BlockerOptimizer (Optuna)

**Definition:** The one optimizer that ships today. It runs an Optuna study over
a **blocker's** hyperparameters (e.g. embedding model, `k_neighbors`) to maximize
a metric you compute in an objective function.

> There is **no** general `Optimizer` that "compiles"/"finetunes" a whole
> *pipeline* — no `compile()`, no pipeline-level training loop; a general
> `Optimizer` over full pipelines is roadmap (`docs/ROADMAP.md`), not implemented.
> *Component*-level training does exist, though: `langres.finetune()` QLoRA-fine-tunes
> a single small-LM matcher (a real peft/trl training loop, `[finetune]` extra) into
> a servable `model_ref`, and DSPy prompt optimization lives inside `DSPyMatcher`
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
from langres.autoresearch.blocker_optimizer import BlockerOptimizer

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
is an eval-time tool, not part of the `.dedupe()`/`.compare()` path.)

> For a higher-level, self-tuning blocking search — a declarative `SearchSpace`
> driven by an `Objective` through the `propose → run → evaluate → keep` loop, with
> every trial persisted — see `langres.optimize` in §10.

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
from langres.core.canonicalizer import Canonicalizer

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

- `select_for_review(rows, strategy=...)` selects pairs by `"uncertainty"` (near the decision margin), `"disagreement"` (student vs. teacher verdicts differ), or `"audit"` (a seeded governance sample), returning `list[ReviewItem]`. `"uncertainty"` ranks by the logged **`confidence`** when present (`|confidence − 0.5|`, most-uncertain first), else falls back to `|score − threshold|`; a decision-only/binary log with neither a usable `confidence` nor a non-degenerate `score` now **raises** `ValueError` (naming `strategy="disagreement"` or `LLMMatcher(confidence="logprob")` as the fix) rather than silently returning `[]`. `ReviewItem` also carries `reasoning` / `confidence` / `confidence_source`.
- `ReviewQueue(path).write(items)` snapshots that selection to `review_queue.jsonl`; items are ids-only unless you pass `records=` to `select_for_review`.
- The `langres` CLI labels the queue (`export-csv` → spreadsheet → `import-csv` → `corrections.jsonl`, or the `langres review` terminal loop). `core.harvest` folds those corrections back into `training.calibration.derive_threshold` / `fit()` — the active-learning loop.

**Example:**

```python
from langres.architectures import FuzzyString
from langres.tracking.judgement_log import JudgementLog
from langres.core.review import ReviewQueue, select_for_review

# 1. Log every judge call while resolving (the flywheel inlet).
FuzzyString(threshold=0.6).dedupe(records, log="judgements.jsonl")

# 2. Select the pairs worth a human's attention, near the decision margin.
rows = JudgementLog("judgements.jsonl").read()
items = select_for_review(rows, strategy="uncertainty", threshold=0.6)

# 3. Snapshot them to a queue the CLI can label.
ReviewQueue("review_queue.jsonl").write(items)
# $ langres export-csv review_queue.jsonl to_label.csv   # label in a spreadsheet
# $ langres import-csv  to_label.csv review_queue.jsonl  # -> corrections.jsonl
```

## 7. Core Data Contracts (Pydantic Models)

### ModelRef (`langres.core.model_ref`) — the ONE backbone contract

An *architecture* is a topology (which components, in what order). A **backbone**
is what fills one of its model slots. `ModelRef` is how langres names one, and
**swapping a backbone never mints a new architecture**.

It is a frozen dataclass (not Pydantic — the module is a stdlib-only leaf that
imports nothing from `langres`, so it stays out of every heavy-dep path),
validated in `__post_init__`, and **weightless**: reference strings only, never
weight bytes, so it round-trips as plain JSON config.

```python
ModelRef(
    base: str,                 # the id or path
    kind: BackboneKind,        # "api" | "endpoint" | "hf" | "local"  (REQUIRED)
    adapter: str | None = None,    # unmerged PEFT adapter; in-process kinds only
    api_base: str | None = None,   # required by — and exclusive to — kind="endpoint"
    revision: str | None = None,   # HF Hub git revision; kind="hf" only
)
```

**`kind` is the discriminator and the sole input to routing** —
`backend_for(kind)` is the entire rule, and it never touches the filesystem:

| `kind` | `base` names | routes to |
|---|---|---|
| `api` | a litellm id (`openai/gpt-4o`, `gpt-5-mini`) | litellm |
| `endpoint` | a model served at `api_base` (vLLM/Ollama/OpenAI-compatible) | litellm |
| `hf` | a Hugging Face Hub id (`org/name`) | transformers (in-process) |
| `local` | a local directory path | transformers (in-process) |

**Surface forms** — `normalize_model_ref(model, *, api_base=None)` accepts all
three and infers `kind` by **syntax alone** when it is not named:

```python
normalize_model_ref("gpt-5-mini")                    # -> kind="api"   (bare id)
normalize_model_ref("openai/gpt-4o")                 # -> kind="api"   (known provider prefix)
normalize_model_ref("BAAI/bge-small-en-v1.5")        # -> kind="hf"    (org/name)
normalize_model_ref("./my-ft")                       # -> kind="local" (path syntax)
normalize_model_ref("m", api_base="http://x:8000/v1")  # -> kind="endpoint"
normalize_model_ref({"base": "org/m", "kind": "api"})   # explicit kind wins
normalize_model_ref({"base": "b", "adapter": "a"})      # -> in-process (QLoRA, unmerged)
```

Two rules that surprise people, both deliberate:

- **A bare relative directory name is NOT a path.** `"my-model"` is an `api` id
  even if `./my-model` exists — write `"./my-model"` or pass `kind="local"`.
  Probing the filesystem is what made routing depend on the working directory:
  the same saved config resolved to a *different backend* elsewhere.
- **`org/name` is never second-guessed.** It cannot be distinguished from a
  typo'd provider by syntax, so a caller who means an API model says
  `{"base": "...", "kind": "api"}`. Conversely a **multi-slash** id
  (`nvidia_nim/meta/llama3-8b`) is *never* a Hub id — those carry exactly one
  slash — so it infers as `api` and litellm, which has the full provider list,
  produces the error if the provider is wrong.

Inference is **total over non-empty strings**: every one names a kind, so the
only `InvalidModelRefError` from a bare string is the empty one.

**`revision` is in the v1 schema on purpose.** Without it an `org/name` ref
drifts as the Hub moves, so two runs of an "identical versioned config" are not
identical across time.

**Serialization** (`to_config` / `normalize_model_ref` are inverses) emits the
compact bare-`base` string exactly when that string re-normalizes to an equal
ref — so the common case stays **byte-identical to pre-`kind` artifacts** — and
widens to `{"base", "kind", ...}` otherwise. The pinned invariant:
`normalize_model_ref(to_config(ref)) == ref` for every ref. A stored dict without
`kind` still loads (it is inferred), so older artifacts keep working.

**Typed errors** (both `ValueError` subclasses):

- `InvalidModelRefError` — the ref is malformed (an empty base, an unknown
  `kind`, an adapter on a served kind, `api_base` on a non-endpoint, a `revision`
  on a non-`hf` ref, an `adapter_revision` without an adapter, a conflicting
  `api_base`, or a non-string `adapter`/`adapter_revision`/`revision`).
- `UnsupportedBackboneError` — the ref is *fine*, but the slot cannot run it:
  a DSPy-backed matcher handed a local dir/adapter (`require_litellm_routable`),
  or a method with no model slot handed any `model=` at all
  (`MethodSpec.check_backbone`).

> **DSPy-backed slots are litellm-only.** `DSPyMatcher` / `SelectMatcher` route
> every completion through litellm — DSPy has no in-process route — so a `local`
> directory or an unmerged adapter raises at construction rather than dying deep
> inside litellm. Use `LLMMatcher` (litellm **and** a transformers backend) to run
> local weights, or serve the model and pass an `endpoint` ref.

### ERCandidate[SchemaT]

The internal data wrapper passed into a Flow.

- `left: SchemaT`
- `right: SchemaT`
- `blocker_name: str`

### PairwiseJudgement

The rich data object passed out of a Flow. This is the auditable log of a decision.

A judge is one of two shapes (and a logprob judge may be both): a **decider** emits a boolean `decision` directly (a binary LLM answers Yes/No; the threshold is irrelevant to it), a **ranker** emits a confidence-ordered `score` (the caller's threshold turns it into a match). Neither set = an **abstention**.

- `left_id: str`
- `right_id: str`
- `decision: bool | None` (default `None`): The judge's explicit match verdict, set by a *decider*. `None` when the judge only ranks (emits a `score`) or abstains. **Takes precedence over `score`** in `predicted_match`.
- `score: float | None` (default `None`, 0.0 to 1.0): The confidence-ordered match score for a *ranker*, else `None`. **Widened from a required `float`:** a decider has no score, so `None` means "this judge does not rank", *not* "score of zero" — a fabricated `0.0`/`1.0` would lie.
- `score_type: Literal["sim_cos", "prob_llm", "heuristic", "calibrated_prob", "prob_fs", "prob_rf", "prob_group_llm"]`: What kind of score is this? Stays **required** even when `score` is `None`: it doubles as the judge-*family* tag, so it names the family (e.g. `"prob_llm"` for a binary LLM judge) rather than a score. Critical for calibration and clustering. `prob_fs`, `prob_rf`, and `prob_group_llm` are emitted by `FellegiSunterMatcher`, `RandomForestMatcher`, and the set-wise `SelectMatcher` respectively.
- `confidence: float | None` (default `None`, 0.0 to 1.0): Optional "how sure am I", orthogonal to the decision. Set today only by `LLMMatcher(confidence="logprob")` (the OpenAI-family first-token credence probe); `None` for every other judge.
- `confidence_source: Literal["none", "unrequested", "logprob", "calibrated", "heuristic"]` (default `"none"`): Provenance of `confidence`. `"none"` = this judge structurally has no confidence to give; `"unrequested"` = it *could* (a decision judge that can expose logprobs) but was not asked; `"logprob"` = an earned first-token credence. The literal set is provisional, not a frozen API.
- `decision_step: str`: Which logic branch made this decision (e.g., "string_sim" or "llm_judge").
- `reasoning: Optional[str]`: The LLM's natural language explanation.
- `provenance: Dict[str, Any]`: A full audit trail (e.g., `{"model": "e5-small", "rapidfuzz_score": 0.85}`).

**Is this pair a match?** Ask `predicted_match(judgement, threshold) -> bool | None` (a module function in `langres.core.models`, exported from `langres.core`) — never a raw `score >= threshold`. It gives `decision` precedence over `score` (a decider already decided; the threshold never overrides it), applies `score >= threshold` to a ranker, and returns `None` for an abstention (neither set). The `is_abstain` property is `True` in exactly that neither-set case. An abstention is **not** a "no": `classify_pairs` and the clusterers *exclude* it from the predicted set rather than grading it a confident non-match.

**LLM-judge provenance keys.** `LLMMatcher` / `DSPyMatcher` / `SelectMatcher` write
`model`, `cost_usd`, `provider`, the legacy `prompt_tokens` / `completion_tokens`
(kept for `JudgementLog`, `bootstrap.labelers`, `openrouter.make_token_cost_track`),
and — added here — a typed **`usage`** vector: `LLMUsage.model_dump()`
(`langres.core.usage`). It follows the OpenTelemetry GenAI vocabulary (snake_case,
SUBSET semantics): `input_tokens` / `output_tokens` are the *inclusive* totals
(`input_tokens` == `prompt_tokens`), and `cache_read_input_tokens`,
`cache_creation_input_tokens`, `reasoning_tokens` are subsets of them, plus the
serving `provider` and `model` id. LiteLLM already normalizes Anthropic's raw
`input_tokens` up to the inclusive total, so the subsets are never re-added.

**Abstention (parse error).** An `LLMMatcher` under `on_parse_error="abstain"` (the
default) whose `response_parser` could not parse a verdict — and a `DSPyMatcher` on
a parse/validation error — now emits `decision=None, score=None` (so `is_abstain
== True`, **not** the old `score=0.0`) with `provenance["parse_error"] = True`.
Both judges abstain identically. `predicted_match` returns `None` for it, so
`classify_pairs` (and the clusterers) *exclude* the pair from the predicted set —
it is no longer graded a confident "no". `evaluate()` /
`evaluate_judge_on_candidates()` surface the count as
`JudgePairEval.n_parse_errors` / `.n_abstained` and warn when non-zero;
`on_parse_error="raise"` turns the same case into an immediate `LLMParseError`.

**`LLMMatcher` paper-replication seams (constructor).** To run a published paper's
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

## 8. Group + Fit Contracts (W1.0) + SelectMatcher (W1.1)

W1.0 froze two interfaces that later branches build against: this section
documents the contracts. W1.1 shipped the first concrete `GroupwiseMatcher` —
`SelectMatcher` (`langres.core.matchers.select_judge`) — proving the contract
against a real set-wise judge. The trained judges over `ComparisonVector` have
since shipped too: `FellegiSunterMatcher` (`langres.core.matchers.fellegi_sunter`)
and `RandomForestMatcher` (`langres.core.matchers.random_forest_judge`, `[trained]`).

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

### GroupwiseMatcher (`langres.core.matcher`)

`GroupwiseMatcher` **is a `Matcher`** — it does not introduce a parallel
execution path. Its concrete `forward()` derives groups internally from
whatever pairwise `ERCandidate` stream it receives (via
`derive_groups_from_pairs`, the same buffered default as above — `forward()`
only ever sees a flat pairwise stream, never the blocker object, so it
cannot reach a blocker's native grouping) and dispatches to the abstract
`forward_groups()`, decomposing the result back to `Iterator[PairwiseJudgement]`.
Concrete set-wise judges implement only `forward_groups()` (plus, to be
inspectable, `inspect_scores()`). Because the group structure never leaves `forward()`,
the Resolver execution spine (`Resolver._judgements` → `module.forward`),
`inspect_scores`, the JudgementLog boundary, and benchmark dispatch
(`BudgetedModuleRunner`, `run_method`) all work unchanged.

```python
# Illustrative pseudocode predating the shipped implementation below --
# `self._call_llm` / `self._last_call_cost` are placeholders, not real
# SelectMatcher attributes (see the real cost/call plumbing in select_judge.py).
class SelectMatcher(GroupwiseMatcher[MySchema]):
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

**Shipped (W1.1):** `langres.core.matchers.select_judge.SelectMatcher` is the
real implementation of the skeleton above — a DSPy `ChainOfThought` over a
`SelectSignature` asking the LLM to select **at most one** matching candidate
id per group (mirroring ComEM's own "selecting" strategy: Wang et al., COLING
2025, choosing "the" single most-likely match, not an arbitrary subset). A
malformed response, a selection naming a candidate outside the group, or a
selection of more than one candidate (`select_error`, CEO #12) all map to
whole-group "no match" judgements carrying `provenance["select_error"]` —
never a raised exception. Selectable by name as `"select_judge"` in
`langres.methods` (the benchmark/method-registry dispatch site only — not
wired into `Resolver.from_schema(matcher=...)` or any named architecture's
topology, since it is not yet part of the zero-label default path). See
`data/benchmarks/w1/W1_RESULTS.md` for the measured call-count/cost reduction
(35.56x on Amazon-Google) and group-size distribution.

### Group-call cost convention (E5)

One LLM call scores a whole group (K pairs). Pricing each of the K resulting
judgements at the call's full cost would silently overcount total spend by a
factor of K. `stamp_group_cost(judgements, call_cost_usd, group_id)`
(`langres.core.matcher`) applies the fix: the full `call_cost_usd` goes on the
**first** judgement's `provenance["cost_usd"]`, every sibling gets `$0`, and
`provenance["group_id"]` is set on all of them. Existing cost aggregation
(`_judgement_cost`/`_cost_track` in `langres.core.benchmark`, which already
read `provenance["cost_usd"]`) then sums a group to exactly one call's cost
with no changes on their end.

`stamp_group_cost` also sets `provenance["group_end"] = True` on (only) the
**last** judgement of the group — a boundary marker that lets a consumer
draining a whole group from a lazy stream (`SpendCappedMatcher.forward` in
`langres.core.spend_cap`, the hard spend cap every `ERModel` wraps its matcher
in) know exactly when to stop pulling, without peeking at the next
judgement's `group_id` — which for a real `GroupwiseMatcher` would resume the
generator into (and pay for) the next group before there's anything to
compare against.

**Atomicity caveat:** `BudgetedModuleRunner` scores exactly one `ERCandidate`
per `module.forward()` call. A `GroupwiseMatcher` run through it today derives
a single, trivial, size-1 group per call — so a group is never *split*
mid-call (there is never more than one pair per call to split), but a real
multi-pair group is also not yet *batched* into one priced call: no cost
amortization happens through the runner yet. Extending the runner (or adding
a group-aware variant) to pre-flight and price whole groups atomically is
deferred to the branch that lands the first concrete `GroupwiseMatcher`.

### Fit-hook contract (`langres.core.fit`)

Runtime-checkable, structural `Protocol`s — **not** abstract methods on any base
class (that would break every existing, non-learnable component). A component
opts in by implementing the method with the matching name — no subclassing. The
taxonomy spans the **three trainable pipeline roles**, each with its own
signature:

- **Matcher** —
  `SupervisedFitMixin.fit(candidates: Iterator[ERCandidate[SchemaT]], labels: Sequence[bool]) -> None`
  and `UnsupervisedFitMixin.fit_unlabeled(candidates: Iterator[ERCandidate[SchemaT]]) -> None`.
  `FellegiSunterMatcher` (unsupervised EM over `ComparisonVector`) and
  `RandomForestMatcher` (supervised, `[trained]`) are the concrete implementers.
- **Blocker** — `BlockerFitMixin.fit_blocker(records: Sequence[Any], pairs: Sequence[tuple[str, str]]) -> None`
  (learn a high-recall blocking key/index from known match pairs). Contract-only
  for now — the concrete `TrainableVectorBlocker` is a later PR.
- **Calibrator** — `CalibratorFitMixin.fit_calibrator(scores: Sequence[float], labels: Sequence[bool]) -> None`
  (learn a score→probability map) + `transform(scores) -> list[float]` (apply
  it). The concrete `Calibrator` (Platt logistic / isotonic, `[trained]`) in
  `langres.training.calibration` implements it, consumed by
  `Resolver.fit(method=Platt()/Isotonic())` — see the `method=` seam below.

`Resolver.fit(data, labels=None, *, pairs=None, split=None, seed=0)` consumes the
**matcher** mixins, detected with `isinstance(module, SupervisedFitMixin)` /
`isinstance(module, UnsupervisedFitMixin)`:

- Matcher implements `SupervisedFitMixin`: supervision comes from either
  pre-aligned `labels` **or** id-keyed `pairs` (see below); passing neither
  **raises** (a genuinely trainable module silently not being trained is the
  exact footgun this hook exists to prevent), and passing both raises.
- Matcher implements `UnsupervisedFitMixin`: `fit_unlabeled` is called
  unconditionally; passing `labels`/`pairs` to it raises.
- Matcher implements **neither** hook (e.g. `WeightedAverageMatcher`): `fit()`
  is a no-op returning `self` — the original sklearn-style symmetry is
  preserved for non-learnable pipelines — unless `labels`/`pairs` was passed,
  which raises rather than silently discarding them.

Every non-raising path sets `resolver.fit_report_` (an sklearn
trailing-underscore digest, `langres.training.fit_report.FitReport`) and returns
`self`, so `resolver.fit(...).resolve(...)` still chains.

### `align_pairs` + coverage + `FitReport`

`langres.core.harvest.align_pairs(candidates, labels, *, split=None, seed=0) ->
AlignedPairs` is the id-join bridge for supervised fit: it joins id-keyed labels
(a `corrections.jsonl` path, or a `Sequence` of `LabeledPair`/`Correction` —
`PairLabel` is a thin alias of `LabeledPair`, not a forked schema) to the blocked
candidates order-independently, and returns a named result with:

- `.train` / `.valid` — positionally-aligned `(candidates, labels)` splits for
  `SupervisedFitMixin.fit`. The split is **entity-disjoint** (union-find over the
  labeled pairs, whole components assigned to one side) — a row-random split
  would leak an entity across train/valid and inflate held-out metrics. A single
  all-connected component keeps `valid` empty rather than emptying `train`.
- `.coverage` — a `GoldCoverage` guardrail (reusing `metrics.evaluate_blocking`)
  surfacing `gold_coverage` (fraction of labeled positives that survived
  blocking) and the `dropped_positives` id-pairs blocking never proposed.

`Resolver.fit(..., pairs=...)` runs `align_pairs` internally, trains on `.train`,
and — when a `split` was given — scores held-out pair P/R/F1 on `.valid` via
`metrics.classify_pairs`, all captured in the `FitReport` (`.to_markdown()` for a
digest). `FitReport` is import-light (Pydantic + `harvest`/`metrics` only, no
sklearn/torch) and references the enclosing `RunRecord`'s `attempt_id` via
`run_ref` for lineage rather than duplicating it.

### Training strategies: the `method=` seam + `describe()`

Beyond the module-hook default above, `Resolver.fit(..., method=<Method>)` takes
a `langres.core.methods_api.Method` — a declarative object naming *how* to train
— and dispatches on `method.kind` to a per-kind handler *before* the
isinstance-on-the-module chain (`method=None` leaves that default byte-for-byte
unchanged):

- **prompt-optimize** (`kind="prompt"`, implemented) — `Bootstrap()` /
  `MIPRO(auto=..., budget_usd=...)` / `GEPA(auto=..., reflection_model=...,
  max_metric_calls=..., budget_usd=...)` from `langres.training.methods_prompt`
  compile a `DSPyMatcher`'s prompt from labeled pairs (the optimizer's
  `BootstrapFewShot` / `MIPROv2` / `dspy.GEPA`). `GEPA` is the *reflective*
  strategy: it reflects on execution traces — via a separate reflection LM
  (`reflection_model`, defaulting to the matcher's own LM) — to rewrite the
  *instruction*, and selects candidates on a Pareto frontier; carry either the
  `auto` preset or a precise `max_metric_calls` budget (its native cost lever, and
  mutually exclusive with `auto`). Supervision comes from `pairs=` (reusing
  `align_pairs` + the entity-disjoint split, whose `valid` fold feeds the
  `MIPROv2`/GEPA valset) or pre-aligned `labels=`. The `FitReport` names the demos
  learned, teacher model, and declared budget. Requires a `DSPyMatcher` (else a
  clear error); DSPy stays lazy-imported, so `Bootstrap()`/`MIPRO()`/`GEPA()`
  construct without pulling `dspy`. Both `Bootstrap` and `GEPA` run zero-spend
  under a `DummyLM` (GEPA reflects with the same dummy LM), so the unit suite
  exercises them; `MIPRO` is paid-only. `budget_usd` threads through the existing
  `SpendMonitor` seam (DSPy-compile spend capture is deferred to #100, so it
  observes `$0` today).
- **fine-tune** (`kind="finetune"`, implemented) — `QLoRA(base=..., ...)` from
  `langres.training.finetune` fine-tunes a small LM on the labeled pairs, repoints
  the Resolver's matcher at an in-process logprob-scoring `LLMMatcher` over the
  produced `model_ref`, and records GPU-seconds / derived-$ + held-out P/R/F1 in
  the `FitReport`. The training stack (peft/trl) stays lazy-imported.
- **calibrate** (`kind="calibrate"`, implemented) — `Platt()` / `Isotonic()`
  from `langres.training.methods_calibrate` fit a score→probability `Calibrator`
  (`langres.training.calibration`, `[trained]`) from labeled pairs, attach it to the
  Resolver, and map every raw judgement score to a calibrated probability in
  `predict()`/`resolve()` (the clusterer then thresholds a real probability; the
  matcher and clusterer are untouched). Supervision comes from `pairs=` (reusing
  `align_pairs` + the entity-disjoint split) or pre-aligned `labels=`. When a
  `valid` split exists the `FitReport` carries the **Brier/ECE before-vs-after**
  on it — the honest proof the map calibrates. The learned params are a handful
  of plain floats carried inline in the artifact, so `save`/`load` round-trips a
  *fitted* calibrator with no pickle and no weight files (`transform` applies the
  map with pure NumPy). `Platt()`/`Isotonic()` are import-light (no scikit-learn
  until the fit runs).

#### Which kinds a class accepts: `accepted_method_kinds`

Note the fine-tune bullet above: `fit(method=QLoRA(...))` **repoints the matcher
slot**. That is deliberate and correct for a plain `Resolver`, which claims no
identity — "a resolver" is not a topology, so a topology change falsifies
nothing. It stops being free the moment a *class* names an architecture: a
`FuzzyString` that quietly became LLM-backed while still being class
`FuzzyString` is a name that describes a pipeline it no longer is.

So a Resolver subclass may declare the kinds it can absorb without ceasing to be
itself:

```python
class FuzzyString(Resolver):
    accepted_method_kinds = frozenset({"calibrate"})   # no prompt/finetune
```

`fit()` checks it at the boundary, before dispatch, and refuses anything else
with `langres.core.UnsupportedMethodKind` (a `TypeError` — `Method.kind` is a
`ClassVar`, i.e. strategy-*type* identity, so rejecting a kind is rejecting a
type) naming the class, the kind, `method.describe()`, and the accepted kinds.
The three real kinds are `"prompt"`, `"finetune"`, `"calibrate"`.

The base `Resolver` leaves `accepted_method_kinds = None` — **permissive**, every
kind accepted, every fit path unchanged. An empty `frozenset()` is the opposite
extreme: a fully frozen topology that takes no `method=` at all.

`Resolver.describe()` is the pre-fit honesty device: a per-component string
tagging each pipeline role **TRAINABLE** (implements a `langres.core.fit`
Protocol, or is a prompt-compilable `DSPyMatcher`) or **frozen**. A pure string
builder — it never trains, imports a backend, or mutates state — so it is safe to
call on a fresh Resolver to see what a `fit()` would (and would not) tune.

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

Both clusterers gate edges through `predicted_match` (a *decider*'s `decision`,
else `score >= threshold`). For the confidence-ordered edge **weight** that drives
its pivot order, `CorrelationClusterer` uses the judgement's `score`, falling back
to `confidence`, then a unit `1.0` — so a bare "Yes" decision (no score) is still a
full-strength edge, never a silent zero that would drop the merge.

**Not the default.** Benchmarked head-to-head against the base `Clusterer` on
Fodors-Zagat + Amazon-Google (same blocking + judge pipeline, only the
clusterer differs): a wash on Fodors-Zagat (+0.0006 BCubed F1), a clear win
on Amazon-Google (+0.0324 BCubed F1, +0.0715 precision at −0.016 recall) —
see `examples/research/w1_blocking_algebra_output.md` for the full tables and the
default-flip decision (kept opt-in; recommended for harder/messier
entity-resolution problems, not flipped globally on a single hard-dataset
win).

## 10. Self-tuning: the autoresearch loop (`langres.optimize`)

The autoresearch loop (epic #145, M1) is a small **propose → run → evaluate →
keep-if-better** hill-climber: it enumerates blocker configs, scores each into
blocking metrics, and keeps the one an `Objective` prefers. Because ER F1
saturates near 99%, it steers on a **loss-like** signal (`candidate_recall@budget`,
`log_loss`, quality×cost Pareto) rather than a thresholded F1.

`optimize` / `score_blocking` are **root exports** (`from langres import optimize`)
and **import-light**: every heavy import (faiss, the benchmark loader,
`langres.autoresearch.factory`) is lazy inside the call, so a bare `import langres`
never pulls the `[semantic]` stack (`tests/test_import_budget.py` guards this). The
proposal/objective types (`SearchSpace`, `Objective`) are pure-stdlib and safe to
import anywhere; only the concrete blocking scorer touches faiss.

> This is a **blocking-search** facade, distinct from its package sibling
> `langres.autoresearch.blocker_optimizer.BlockerOptimizer` (§5): `BlockerOptimizer`
> runs an Optuna study returning best params, while `optimize` runs the
> deterministic keep-if-better loop over a declarative `SearchSpace`, gated by an
> `Objective`, persisting every trial. Two search strategies, one package — both
> live under `langres.optimize` because search is not ER modelling and so sits
> outside `langres.core`. A general `Optimizer` over full *pipelines* remains
> roadmap.

### langres.optimize (the loop facade)

`optimize(space, objective, benchmark, *, seed=None, store=None, dedup=True, split="full", embedder=None, tracker=None) -> LoopResult`

Loads `benchmark` **once**, fingerprints it once, wraps an index-caching blocking
scorer (one vector index per `(embedding_model, metric, text_field)` group, reused
across every `k`), and drives the loop over `space.configs()`.

- `space: SearchSpace` — the declarative config grid (below).
- `objective: Objective` — the immutable keep-if-better decision (below).
- `benchmark: str | Benchmark` — a registered benchmark **name** (loaded via
  `langres.data`) or an already-built benchmark object (offline / test path).
- `seed: int | None` — recorded on every run for provenance under
  `seeds["optimize"]` (blocking is deterministic, so this only labels the run).
- `store: str | Path | RunStore | None` — where to persist run records;
  **`store=None` writes nothing.**
- `dedup: bool = True` — skip a config whose `recipe_id` was already scored this
  run (degenerate `all_pairs` repeats are collapsed first).
- `split: str = "full"` — split label recorded on every run (M1 measures over the
  whole loaded corpus).
- `embedder: EmbeddingProvider | None` — optional pre-built embedder (a fake keeps
  tests offline); production leaves it `None` to load the real SentenceTransformer.
- `tracker: TrackerSpec` (`str | ExperimentTracker | Sequence[str | ExperimentTracker] | None`)
  — a backend name (`"trackio"`/`"mlflow"`/`"wandb"`), an already-built instance, a
  sequence of either (fan-out), or `None` (default, no-op). `run_loop` resolves the
  spec internally via `resolve_tracker`, mirroring the `matcher="..."` string
  dispatch — write `tracker="trackio"`, not `tracker=resolve_tracker("trackio")`.
  `"trackio"` is local-first (no credentials/network); an HF Space/Dataset sync
  needs a `TrackioTracker(space_id=...)` instance or `TRACKIO_SPACE_ID`/`HF_TOKEN`.

### score_blocking (the one-config scorer)

`score_blocking(config, benchmark, *, embedder=None, index=None) -> dict[str, float]`

Blocking metrics for **one** config — builds the index + blocker the config
describes, streams the full corpus to candidates, and evaluates blocking. This is
the concrete scorer `optimize` wraps; call it directly to score a single config.
`index=` reuses a prebuilt vector index instead of building one. Returns a plain
metrics dict (all values `float`):

| Key | Meaning |
|---|---|
| `candidate_recall` | Fraction of true match pairs the blocker surfaced (the recall signal the loop maximizes). |
| `reduction_ratio` | Fraction of the `O(|A|·|B|)` (or `num_records`-choose-2) comparison space eliminated — the budget/cost axis. |
| `candidate_precision` | Fraction of surfaced candidates that are true matches. |
| `total_candidates` | Count of candidate pairs emitted (as a float). |

For a two-source (linkage) corpus the candidates are filtered to cross-source
pairs and `reduction_ratio` uses `|A|·|B|`; otherwise it uses `num_records`.

### SearchSpace (`langres.autoresearch.search_space`)

A frozen, declarative Cartesian grid of blocker configs. Each field is a
non-empty tuple of candidate values for one axis (an empty axis raises).

- `blocker: tuple[str, ...] = ("vector",)` — `"vector"` and/or `"all_pairs"`
  (the vector axes below are ignored by `"all_pairs"`).
- `embedding_model: tuple[str, ...] = ("all-MiniLM-L6-v2",)`
- `metric: tuple[str, ...] = ("cosine",)` — FAISS metric `"L2"` / `"cosine"`.
- `text_field: tuple[str, ...] = ("name",)` — record attribute holding the
  blocking text (dataset-specific; override to your schema's field).
- `k_neighbors: tuple[int, ...] = (5, 10, 20)`

`configs() -> Iterator[dict[str, Any]]` yields the Cartesian product as config
dicts (keys `blocker`, `embedding_model`, `metric`, `text_field`, `k_neighbors`).
**Ordering contract (the loop relies on it):** `k_neighbors` is the **innermost**
varying axis, so consecutive configs hold `(blocker, embedding_model, metric,
text_field)` fixed while `k` sweeps its full range — letting `optimize` build one
index per group and reuse it across every `k`. `len(space)` is the product of the
axis sizes.

### Objective (`langres.autoresearch.objective`)

The immutable keep-if-better scorer, metric-source-agnostic (it operates on a
plain `Mapping[str, float]` and never computes a metric itself). It bundles one or
more `Goal`s (optimization targets) with zero or more `Constraint`s (feasibility
gates). Prefer the three ergonomic constructors:

- `Objective.maximize(metric, *, subject_to=())` — one maximize goal.
- `Objective.minimize(metric, *, subject_to=())` — one minimize goal (e.g.
  `log_loss`, cost).
- `Objective.pareto(goals, *, subject_to=())` — a multi-objective Pareto front,
  `goals` = `(metric, direction)` pairs; **never scalarized**.

`subject_to` is an iterable of `(metric, op, threshold)` triples with `op` in
`>= <= > <`; a missing metric raises (it never defaults to `0.0`).

**`is_better(candidate, incumbent) -> bool`** — the loop's decision, in order:
(1) **feasibility first** — an infeasible candidate is never better; a feasible
candidate beats a `None` or infeasible incumbent; (2) **Pareto improvement** —
with both feasible, the candidate wins iff it *dominates* the incumbent (`>=` on
every goal, `>` on at least one; for a single goal, a strict scalar improvement).
A tie or an incomparable trade-off keeps the incumbent, so the decision is
deterministic and monotone.

### LoopResult / Trial (`langres.autoresearch.loop`)

`run_loop` (the driver `optimize` calls) returns a frozen `LoopResult`:

- `best_config: dict[str, Any] | None` — the winning config, or `None` if no
  config was ever accepted (empty input, or every trial infeasible/failed).
- `best_metrics: dict[str, float] | None` — the winning config's metrics (or `None`).
- `trials: tuple[Trial, ...]` — every trial in evaluation order (accepted,
  rejected, and failed), for reconstructing why the incumbent won.

Each `Trial` is frozen: `config` (the scored dict), `metrics` (`dict | None` —
`None` if the scorer raised), `accepted` (whether it displaced the incumbent),
`recipe_id` (the content-addressed dedup key), `attempt_id` (the run record PK),
and `status` (`"completed"` or `"failed"` — one bad config is logged and skipped,
never aborting the sweep).

### Persistence — local JSONL only, today

Every trial (including over-budget rejects and scorer failures) is appended to the
`store` path's `RunStore` JSONL (the `tracking.runs` spine, §2). `store=None` writes
nothing; read a trail back with `RunStore(path).read()` (each `RunRecord`'s
`metrics["accepted"]` is `1.0`/`0.0`, so the incumbent timeline is reconstructable
from the store alone). `RunStore` is local-only by design; a durable off-laptop
dashboard is available via `tracker="trackio"` (local-first; an HF Space/Dataset
sync is a `space_id`/`HF_TOKEN` opt-in). A model's state-free configuration and
bounded result summary can separately be shared through the pretrained Hub
lifecycle below. Still deferred: an Optuna/LLAMBO proposer and the matching vertical
(`log_loss` / AUC-PR steering) + fine-tuning. See
[EXPERIMENTS.md](EXPERIMENTS.md#self-tuning-the-autoresearch-loop-langresoptimize)
for the worked amazon_google proof and `examples/research/blocking_recall_autoresearch.py`.
## Pretrained artifact lifecycle

`ERModel.save_pretrained`, `ERModel.from_pretrained`, and
`ERModel.push_to_hub` delegate lazily to `langres.hub`. They wrap the unchanged
local `resolver.json` in a strict `langres-artifact.json` envelope containing an
exact regular-file allowlist, sizes, SHA-256 checksums, resource `ModelRef`
facts, compatibility metadata, and an optional bounded measurement summary.
The first artifact version is deliberately state-free: it rejects Resolver
sidecars because current sidecars can contain corpus rows, compiled prompts, or
native binary state. In-place overwrite is also refused; publish a new validated
directory or Hub commit and then switch the reference.
Prompt-bearing JSON configuration is also excluded by default. Publishing it
requires an explicit `allow_sensitive_config=True`, after which the manifest and
model card disclose that prompts are included.

Remote loading resolves the requested Hub revision to an immutable commit SHA
before downloading the exact manifest-derived allowlist. The snapshot is fully
validated and its registered component/operation types are preflighted before
`ERModel.load` reconstructs anything. The loader accepts the same langres minor
release (patches may differ), ignores repository files outside the manifest
without downloading them, and never serializes Hub tokens or raw experiment
data. See `docs/HUGGING_FACE.md`.
