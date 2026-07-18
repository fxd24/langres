# Model resources

`langres.resources` separates reusable model capabilities from their position in
an entity-resolution topology:

- `Embedder.embed(texts) -> EmbeddingBatch`
- `Reranker.rerank(pairs) -> RerankBatch`
- `LLM.generate(requests) -> GenerationBatch`

Every resource exposes a stable `ModelRef` and a frozen runtime configuration.
Constructing a resource records configuration only; production weights and
optional libraries load on first use.

```python
from langres.core.model_ref import ModelRef
from langres.resources import (
    CrossEncoderReranker,
    RerankerRuntimeConfig,
)

reranker = CrossEncoderReranker(
    ModelRef(
        base="cross-encoder/ms-marco-MiniLM-L6-v2",
        kind="hf",
        revision="<commit-sha>",
    ),
    runtime_config=RerankerRuntimeConfig(
        batch_size=64,
        device="cpu",
    ),
)
```

The Hugging Face revision, device, dtype, batch size, backend, and offline-cache
policy remain configuration. They do not load weights and can be captured before
an experiment runs.

## Deterministic offline resources

`FakeEmbedder`, `FakeReranker`, and `FakeLLM` are deterministic, zero-network
implementations of the same protocols. They are intended for topology tests and
experiment preflight—not as alternate execution paths.

```python
from langres.resources import FakeReranker, RerankRequest

resource = FakeReranker(scores={"pair-1": 0.9})
batch = resource.rerank(
    [RerankRequest(pair_id="pair-1", left="Acme", right="ACME")]
)
assert batch.scores == (0.9,)
```

## Operations own role

`Rerank` adapts any `Reranker` resource to the existing `Score` operation. The
resource only emits scores. Whether those scores perform candidate pruning or a
final match decision is determined by the following `Select`.

`Generate` invokes an `LLM`; `Parse` turns the generated response into a typed
score, decision, or abstention. They exchange a versioned `GenerationEnvelope`
through the existing `Pairs` provenance rather than introducing another pair
carrier.

Raw generated content is process-local by default:

- ordinary `GenerationEnvelope.model_dump()` output excludes it;
- `Parse` removes the private envelope and retains only its safe summary, usage,
  and parsed outcome;
- a declared local replay cache must explicitly call `local_payload()` to retain
  content.

This prevents experiment reports and trackers from publishing prompts, records,
or model responses simply because they serialize pair provenance.

`GenerationUsage` records input/output totals plus cache-read, cache-creation,
and reasoning subsets. Providers do not expose every field, so an absent count
is `None`; a real zero remains `0`. This distinction prevents benchmarks from
silently treating unmeasured usage as free usage. LiteLLM's normalized input and
output totals already include their subsets, so callers must not add cache or
reasoning counts to those totals.

## API and local LLMs

`LiteLLM` and `TransformersLLM` implement the same `LLM` protocol.
`llm_from_model_ref()` selects between them using only `ModelRef.kind`:

```python
from langres.resources import llm_from_model_ref

api_llm = llm_from_model_ref("openai/gpt-4o-mini")
local_llm = llm_from_model_ref(
    {"base": "./checkpoints/my-model", "kind": "local"}
)
```

`LLMMatcherAdapter` exposes `LLM + Generate + Parse` through the existing
`Matcher.forward()` contract so legacy resolver construction remains usable
during the additive migration.
