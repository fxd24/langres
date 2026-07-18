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

The Hugging Face base revision, optional adapter revision, device, dtype, batch
size, backend, seed, and offline-cache policy remain configuration. They do not
load weights and can be captured before an experiment runs. A base and PEFT
adapter are pinned independently:

```python
ModelRef(
    base="org/base",
    kind="hf",
    revision="<base-commit-sha>",
    adapter="org/adapter",
    adapter_revision="<adapter-commit-sha>",
)
```

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

The built-in `Generate` prompt asks for `MATCH` / `NO_MATCH`, and the default
`Parse` and `LLMMatcherAdapter` parser consumes that same binary shape. Numeric
`Score: 0..1` output remains available explicitly through
`parse_score_response`.

Paid `Generate` operations implement the structural `SpendMonitorBindable`
capability. An
explicit topology binds them to the model's cumulative `SpendMonitor`; duplicate
request ids are rejected before any provider call, and measured envelope costs
are added to that ledger. Once bound, `Generate` invokes the resource one request
at a time and checks the ledger between calls, preserving the framework-wide
budget guarantee: the configured budget plus the cost of at most one further
paid call. A finite-budget API/endpoint run also fails closed after the first
successful response whose cost is unknown. The typed
`UnknownGenerationCostError.outputs` retains that response for logging or
explicit recovery, but the operation never treats it as `$0` or starts a second
call. The shared ledger is permanently marked unknown, so catching the exception
cannot resume paid work through the same budget. Direct unbound resource use and
explicitly uncapped runs remain nonfatal.

`GenerationUsage` records input/output totals plus cache-read, cache-creation,
and reasoning subsets. Providers do not expose every field, so an absent count
is `None`; a real zero remains `0`. This distinction prevents benchmarks from
silently treating unmeasured usage as free usage. LiteLLM's normalized input and
output totals already include their subsets, so callers must not add cache or
reasoning counts to those totals.

`GenerationEnvelope` also records the actual served model, serving provider,
provider request id, and whether cost came from real provider billing or an
estimate. Provider billing is preferred over a price-table estimate. Pricing is
post-call observability: if estimation fails after generation succeeds, langres
keeps the generated result and records cost as unknown.

OpenRouter `LiteLLM` calls always request provider usage accounting with
`extra_body={"usage": {"include": true}}`. Existing `extra_body` keys are
preserved, and an explicit `provider=` routing block wins over an inherited
provider block so benchmark routing is reproducible. When a tracking run is
active, the outbound LiteLLM metadata includes its attempt id and the generation
request id. Provider routing and extra-body construction specs accept strict
JSON objects only, so malformed or executable artifact values fail during
construction rather than reaching a provider.

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

Constructing `TransformersLLM("distilgpt2")` treats the bare name as an
in-process Hugging Face id, consistently with embedding and reranker resources.
At `temperature=0`, local generation is greedy. A positive temperature enables
sampling; set `LLMRuntimeConfig(seed=...)` to make that runtime choice explicit
and reproducible.

The four production resources (`SentenceTransformer`, `CrossEncoderReranker`,
`LiteLLM`, and `TransformersLLM`) expose registered, weightless construction
specs. Their model reference and runtime configuration round-trip without model
weights, credentials, or injected clients. Custom request builders and parsers
remain runtime callables and are not silently serialized.

`LLMMatcherAdapter` exposes `LLM + Generate + Parse` through the existing
`Matcher.forward()` contract so legacy resolver construction remains usable
during the additive migration. It yields one judgement after each provider call
so an outer `SpendCappedMatcher` can enforce the ledger before pulling the next
candidate. The adapter also propagates the resource's
`requires_cost_accounting` capability: a paid resource whose cost is unknown
poisons the outer finite ledger after that first successful call. This capability
comes from the resource, not model-id syntax, because LiteLLM provider ids and
Hugging Face ids can have the same slash-delimited shape.

### Topology merge checkpoint

`Rerank`, `Generate`, and `Parse` already expose safe named-callable configs and
round-trip constructors. Their `OpSerializer` registrations intentionally land
only after this resource branch is merged with the topology branch that defines
that registry; adding a parallel registry here would create two persistence
contracts. The merged follow-up must register these three adapters against the
existing topology seam and add a full topology save/load test.
