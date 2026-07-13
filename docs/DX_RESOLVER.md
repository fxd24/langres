# Developer Experience: the Resolver

The `Resolver` (M0, Wave 3) is the spine that turns four loose primitives into
one runnable, serializable pipeline:

```
blocker  ->  comparator (optional)  ->  module (scorer)  ->  clusterer
```

This page is the short before/after: what deduplicating companies looked like
with the raw primitives, and what it looks like now.

## Before — manual wiring, lambdas, no persistence

Building a dedup pipeline meant hand-writing a schema factory, a text
extractor, per-field extractor lambdas with weights, and a manual index
lifecycle. Nothing was serializable — there was no `save`/`load`.

```python
# Schema normalization: a hand-written factory + text extractor.
def company_factory(record: dict) -> CompanySchema:
    return CompanySchema(
        id=record["id"],
        name=record["name"],
        address=record.get("address"),
        phone=record.get("phone"),
        website=record.get("website"),
    )

def company_text_extractor(c: CompanySchema) -> str:
    return c.name

# Blocker: inject the callables; for VectorBlocker, build the index by hand.
blocker = VectorBlocker(
    schema_factory=company_factory,
    text_field_extractor=company_text_extractor,
    vector_index=index,
    k_neighbors=10,
)
texts = [company_text_extractor(company_factory(r)) for r in records]
blocker.vector_index.create_index(texts)          # easy to forget -> RuntimeError
candidates = list(blocker.stream(records))

# Scorer: per-field extractor lambdas, missing handled ad hoc (`x.address or ""`).
module = RapidfuzzModule(
    field_extractors={
        "name": (lambda x: x.name, 0.7),
        "address": (lambda x: x.address or "", 0.2),
        "website": (lambda x: x.website or "", 0.1),
    },
)
judgements = module.forward(candidates)

# Cluster.
clusters = Clusterer(threshold=0.7).cluster(judgements)

# Persist? Not possible — the lambdas can't round-trip through JSON.
```

Friction:

- **~25 lines of glue** before the first cluster.
- **Lambdas everywhere** (`schema_factory`, `text_field_extractor`,
  `field_extractors={"name": (lambda x: x.name, .7)}`) — none serializable.
- **Manual missing-handling** (`lambda x: x.address or ""`) — an empty string
  silently compares as a real value (two blanks "match").
- **Manual index lifecycle** — forget `create_index` and `stream` raises.
- **No `save`/`load`** — a tuned pipeline lives only in the process that built
  it.

## After — declarative, missing-aware, persistent

```python
from langres.core import Resolver
from langres.core.models import CompanySchema

resolver = Resolver.from_schema(
    CompanySchema,
    threshold=0.7,
    weights={"name": 0.6, "address": 0.2, "phone": 0.1, "website": 0.1},
)

clusters = resolver.resolve(records)        # block -> compare -> score -> cluster
resolver.save("artifacts/company_v0")       # human-readable resolver.json
reloaded = Resolver.load("artifacts/company_v0")
assert reloaded.resolve(records) == clusters  # identical
```

What changed:

- **One declarative line** replaces the schema factory, text extractor, scorer
  lambdas, and manual index build. `from_schema` derives a missing-aware
  `StringComparator` from the schema's string fields, excludes `id`, and wires
  a `WeightedAverageJudge` + `Clusterer`.
- **Missing-aware by construction.** A `None`/empty field is dropped, never
  compared as a blank string; the evidence floor stops a single weak feature
  from over-merging. No `x.address or ""`.
- **Transparent index lifecycle.** For an index-backed blocker (`VectorBlocker`),
  `resolve` builds the index in place (logs `Embedding N records…`); for
  `AllPairsBlocker` it is a no-op. The caller never calls `create_index`.
- **Real persistence.** `save` writes a version-stamped `resolver.json` listing
  each slot's `type_name` + config (the embedder persists by `model_name`, not
  model bytes); `load` rebuilds every slot from the component registry — **no
  pickle, no code execution**. FAISS index state round-trips via sidecar files.

### Ceremony removed (company dedup)

| | Before | After |
|---|---|---|
| Lines to first cluster | ~25 | **4** |
| Serializable lambdas | 0 of ~5 | n/a (none needed) |
| Manual `create_index` calls | 1 (forgettable) | **0** |
| `save` / `load` | not possible | **2 calls** |

### Power path still open

The four-slot constructor is unchanged for advanced use — swap in a
`VectorBlocker`, a custom `Module`, or `comparator=None` for a self-contained
scorer:

```python
comparator = Comparator.from_schema(CompanySchema, weights={...})
Resolver(
    blocker=AllPairsBlocker(schema=CompanySchema),
    comparator=comparator,
    module=WeightedAverageJudge(feature_specs=comparator.feature_specs),
    clusterer=Clusterer(threshold=0.7),
)
```

Runnable end-to-end: [`examples/resolver_company_dedup.py`](../examples/resolver_company_dedup.py).

## Consuming a saved artifact

The M2 walking skeleton (`examples/research/m2_walking_skeleton_fodors_zagat.py`) builds,
tunes, and **saves** a Resolver artifact. The integrator who *consumes* that
artifact writes none of the build code — only load + resolve:

```python
import langres.data.er_benchmarks  # registers RestaurantSchema (must precede load)
from langres.core.resolver import Resolver

# records: list[dict] — raw rows in the saved corpus's schema (e.g. RestaurantSchema).
clusters = Resolver.load("artifacts/fodors_zagat").resolve(records)
# -> list[set[str]] of multi-record clusters (singletons are dropped).
```

That is the entire consumer path: `save(<dir>)` writes the artifact **directory**
(a human-readable `resolver.json` manifest plus any FAISS sidecar — no pickle, no
code execution), and `Resolver.load(<dir>).resolve(records)` reconstructs the
pipeline from the registry and runs it. The import line is load-bearing: a fresh
process must register the record schema **before** `Resolver.load`, or the
manifest's `schema_type_name` won't resolve. A fresh-process identity proof
(`tests/data/test_m2_artifact_slow.py`) asserts the reloaded artifact produces
clusters identical to the in-process run.

**Three consumption modes:**

- **Batch dedup of a corpus**: `resolver.resolve(records)` — cluster a record
  list into entities.
- **Incremental single-record assignment** (shipped):
  `resolver.build_anchor_store(records)` once, then `resolver.assign(new_record)`
  → `ClusterDelta` (`link` to a stable entity id, or `new`); the serializable
  `AnchorStore` persists alongside the artifact. Runnable:
  [`examples/incremental_assign.py`](../examples/incremental_assign.py).
- **Cross-source linking** (not yet available): `.link()` / `.stream_against()`
  raise `NotImplementedError` today — reserved stubs on the roadmap.
