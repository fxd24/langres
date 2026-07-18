# Architectures

Named ER models. You construct one, then call `.dedupe()` / `.compare()` on it:

```python
from langres.architectures import FuzzyString, VectorLLMCascade

FuzzyString().dedupe(records)                                   # $0, offline, no key

VectorLLMCascade(
    embedder="BAAI/bge-base-en-v1.5",
    llm="openrouter/deepseek-v4",
).dedupe(records)                                               # paid, because you named it
```

An **architecture** is a *topology* — which components run, in what order. A
**backbone** is what fills a model slot. Swapping a backbone never mints a new
architecture: `VectorLLMCascade(llm="a")` and `VectorLLMCascade(llm="b")` are the
same architecture with different weights behind it.

There is no `matcher="auto"`. Nothing here reads your environment to decide what
to run: `FuzzyString` has no paid slot to fill, and `VectorLLMCascade` only bills
you because you named an `llm=`. Choosing the model is your job, not a
heuristic's.

## Research recipes

The research front door names topology in the same vocabulary as the model
ecosystem:

| Recipe | Topology |
|---|---|
| `Retrieve` | Retrieve → threshold → Cluster |
| `RetrieveRerank` | Retrieve → Rerank → threshold → Cluster |
| `RetrieveLLM` | Retrieve → top-k → LLM Generate/Parse → Cluster |
| `RetrieveRerankLLM` | Retrieve → Rerank → top-k → LLM Generate/Parse → Cluster |

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
    llm_k=10,
)
```

Recipes infer a schema from records on their first `.dedupe()` or `.compare()`
call. Pass `schema=YourPydanticModel` explicitly when saving an artifact; an
inferred runtime schema is intentionally ephemeral.

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

::: langres.architectures.retrieval

::: langres.architectures.fuzzy_string

::: langres.architectures.vector_llm_cascade
