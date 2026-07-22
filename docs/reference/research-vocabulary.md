# Research vocabulary and legacy migration

The research API uses **resources, operations, and recipes**:

- a **resource** is a model-bearing capability (`Embedder`, `Reranker`, `LLM`);
- an **operation** transforms a carrier (`Retrieve`, `Rerank`, `Select`,
  `Generate`, `Parse`, `Cluster`);
- a **recipe** is a named, readable ordered operation topology equipped with
  resources.

The four-slot API remains supported for compatibility. Use this map when
reading older docs, code, and artifacts:

| Legacy term | Research-foundation term | Migration note |
|---|---|---|
| `Blocker` | usually a `Source`; vector candidate generation becomes `Retrieve` | `Blocker` remains a supported classic component. Do not mechanically rename key/all-pairs blocking to an embedding resource. |
| `Comparator` | a resource-free `Score` operation | It still exists in the classic path and may attach a `ComparisonVector`. |
| `Matcher`, `Judge`, or historical `Module` | a `Score`, or `Generate` + `Parse` operations backed by an `LLM` resource | “LLM” names the capability; `Generate` and `Parse` state how it is used. |
| `Clusterer` slot | conceptual `Cluster` operation, implemented by `ClustererStage` through the `ClusterStage` class | The algorithm remains reusable; the implementation places it in the topology. |
| four slots: blocker → comparator → matcher → clusterer | explicit ordered operation topology | Compatibility models still derive an operation chain internally. |
| backbone/model string | resource `ModelRef` in a named slot | Use `architecture.resources`; singular `.backbone` is compatibility sugar only when exactly one slot exists. |
| named architecture | recipe when referring to a shipped, equipped research topology | “Architecture” remains the general topology concept. A recipe is its named user-facing construction. |
| matcher method name / `run_methods` race | `ArchitectureFactory` variant in an `Experiment` | Legacy benchmark helpers remain for older scorer-specific workflows. |
| judgement cost attached to a matcher | resource/stage usage plus `PriceSnapshot` | Preserve stored tokens so they can be repriced without inference. |

There is still no `matcher="auto"` and no module-level `langres.dedupe` or
`langres.link`. Construct a recipe/architecture explicitly, or build an
`ERModel.from_topology(ops=[...])`.

The research `Retrieve` operation keeps the `Embedder` as the model-bearing
resource and delegates vector storage/search to Qdrant. In a linkage recipe,
`source_field="source"` excludes same-source records inside the Qdrant query,
before top-k; in a deduplication recipe the option is omitted.
