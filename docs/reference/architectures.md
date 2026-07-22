# Architectures, resources, operations, and recipes

Named ER models. You construct one, then call `.dedupe()` / `.compare()` on it:

```python
from langres.architectures import FuzzyString, VectorLLMCascade

FuzzyString().dedupe(records)                                   # $0, offline, no key

VectorLLMCascade(
    embedder="BAAI/bge-base-en-v1.5",
    llm="openrouter/deepseek-v4",
).dedupe(records)                                               # paid, because you named it
```

An **architecture** is a topology. In the research API, **resources** are the
model-bearing capabilities, **operations** are the ordered transformations, and
a **recipe** is a named architecture equipped with resources. Swapping a
resource or its `ModelRef` creates a configuration variant, not a new
architecture.

There is no `matcher="auto"`. Nothing here reads your environment to decide what
to run: `FuzzyString` has no paid slot to fill, and `VectorLLMCascade` only bills
you because you named an `llm=`. Choosing the model is your job, not a
heuristic's.

## Research recipes

The research front door names topology in the same vocabulary as the model
ecosystem:

| Recipe | Topology |
|---|---|
| `Retrieve` | Retrieve → threshold Select → Cluster |
| `RetrieveRerank` | Retrieve → Rerank → threshold Select → Cluster |
| `RetrieveLLM` | Retrieve → top-k Select → Generate → Parse → `ThresholdSelect(0.5)` → Cluster |
| `RetrieveRerankLLM` | Retrieve → Rerank → top-k Select → Generate → Parse → `ThresholdSelect(0.5)` → Cluster |

Each model slot accepts either a resource object or a model reference. The
resource name describes what it does, while the following `Select` determines
how its score is used. The same `Reranker` works before a top-k candidate cut or
before a final threshold; there are no separate blocker and matcher reranker
classes.

```python
from langres.architectures import RetrieveRerankLLM

architecture = RetrieveRerankLLM(
    embedder="sentence-transformers/all-MiniLM-L6-v2",
    reranker="cross-encoder/ms-marco-MiniLM-L6-v2",
    llm={"base": "openai/gpt-4o-mini", "kind": "api"},
    retrieve_k=50,
    source_field="source",  # linkage: exclude same-source hits before top-k
    llm_k=10,
)
```

`Retrieve` indexes the embedder's vectors in Qdrant. The default uses Qdrant's
zero-configuration local mode; the adapter is also verified against a real
Qdrant server with Testcontainers. For two-source linkage, pass
`source_field="source"`: Qdrant applies that exclusion inside each nearest-
neighbour query, before `retrieve_k` is consumed. Omit it for ordinary
single-source deduplication. An explicitly named collection that already exists
is never deleted: an injected client must use a fresh collection name for each
index instance. Install `langres[semantic]` to enable the Qdrant adapter.

Recipes infer a schema from records on their first `.dedupe()` or `.compare()`
call. Pass `schema=YourPydanticModel` explicitly when saving an artifact; an
inferred runtime schema is intentionally ephemeral.

`Retrieve` and `RetrieveRerank` validate `threshold` in `[0, 1]`. For the paid
recipes, `dedupe(..., log=...)` and `compare(..., log=...)` record each parsed
LLM decision, including its model, usage/cost provenance, verdict, and stage id.
Pass either `budget_usd=` for a recipe-local cap or `monitor=` to adopt an
existing `SpendMonitor`. Experiment factories use `monitor=` so every recipe in
a matrix charges the same cumulative ledger; the two arguments are mutually
exclusive.

When `Experiment` sweeps this decision threshold, it selects the value with the
highest pair F1 on the declared training split. The declared grid is the
baseline/fallback; replayable scored rows also contribute their exact observed
score breakpoints in one sorted pass, without rerunning model inference. The
threshold strategy and observed score range/counts are logged with the run.
B-Cubed remains a downstream clustering report metric; it is not the threshold
objective, because a singleton-heavy dataset can otherwise reward predicting no
matching pairs.

`architecture.resources` returns every slot as `dict[str, ModelRef]`.
`architecture.backbone` remains compatibility sugar only for `Retrieve`, the
one recipe with exactly one model. Multi-model recipes return `None` rather than
hiding two of their resources.

API, endpoint, Hugging Face, and local LLM references all use the same `llm=`
slot; `ModelRef.kind` chooses served versus in-process execution. Every recipe
ends in the existing transitive-closure `Clusterer` by default and accepts a
custom existing `clusterer=`. The recipe's `Select` owns the one match cut, so
the supplied clusterer's algorithm is reused with a threshold-free copy; the
caller's clusterer object is not mutated. This milestone adds no clustering
algorithm.
Advanced users can compose the same `langres.resources.Retrieve`, `Rerank`,
`Generate`, and `Parse` operations through `ERModel.from_topology()`; the
experiment path does not dispatch on recipe names.

See [Research vocabulary and legacy migration](research-vocabulary.md) for the
explicit mapping from Blocker/Matcher/Judge-era terms.

::: langres.architectures.retrieval

::: langres.architectures.fuzzy_string

::: langres.architectures.vector_llm_cascade
