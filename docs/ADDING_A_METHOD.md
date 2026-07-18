# Adding a resource or architecture

Start by deciding what is actually new:

- new weights/provider/runtime behind the same capability → add or configure a
  **resource**;
- a new transformation of the record/pair stream → add an **operation**;
- a different ordered topology → add a named **recipe/architecture**;
- a classic four-slot scorer needed by `Resolver.from_schema(matcher="...")` →
  use the legacy method registry path described at the end.

Do not mint a new architecture merely because a model id changed. Do not hide a
topology change behind a boolean flag.

## Add a resource

Implement one import-light protocol from `langres.resources`: `Embedder`,
`Reranker`, or `LLM`. The object owns:

- a complete `model_ref`;
- an immutable runtime config;
- one bounded method (`embed`, `rerank`, or `generate`);
- measured facts/usage in its return contract.

Heavy libraries and model loading stay inside the first method call. Importing
the resource module must not load torch, transformers, sentence-transformers,
LiteLLM, or a Hub client.

A custom in-memory resource can be passed directly to a recipe with no
registration. If it must survive `save`/`load`, give it strict JSON
`config`/`from_config`, register the exact supported type, and add a
fresh-process round-trip test. Never serialize credentials, injected clients,
Python module paths, or arbitrary code.

## Add an operation

Reuse the public topology contracts: `Source`, `Op`, `Score`, `Select`, the
conceptual `Cluster` operation, and `Sequential`. In Python, terminal clustering
implements the `ClusterStage` class (normally through `ClustererStage`). An
operation has one responsibility and must preserve its carrier contract.

Persistence requires:

- `@register_op("<stable-role>")`;
- strict Pydantic `config_model` with `extra="forbid"` and `strict=True`;
- complete `config` and deterministic `from_config`;
- an exact-class, fresh-process artifact round-trip.

If an operation can bill, it must declare `Spending`, implement
`SpendMonitorBindable`, and bind to the experiment/model's exact shared
`SpendMonitor`. A paid operation that cannot bind is rejected. Never call a raw
matcher/resource around the capped execution seam.

`Source.prepare(records)` owns input-dependent binding and index construction.
Do not create a second executor, cache, tracker, or index lifecycle.

## Add a named recipe

Put the readable ordered operations in `langres.architectures`. Each concrete
recipe should spell out its topology; small duplication is preferable to a
flag-driven builder that hides which stages run. Expose every resource slot via
`resources: dict[str, ModelRef]`.

Use the four shipped recipes as the conformance ladder:

```text
Retrieve
Retrieve → Rerank
Retrieve → Select(top_k) → Generate → Parse
Retrieve → Rerank → Select(top_k) → Generate → Parse
```

### Conformance checklist

- runs with deterministic fake resources and no network;
- accepts API, endpoint, Hub, or local refs only where the resource can honor
  their semantics;
- `execution_plan()` shows the expected ordered roles;
- the same `Reranker` object works before either top-k or threshold selection;
- spend uses one shared monitor and the declared cap;
- cache semantics are declared deterministic, seeded, or stochastic;
- resource/config variants have distinct `variant_id` values in experiments;
- `save`/`load` and, when eligible, `save_pretrained`/`from_pretrained`
  round-trip in a fresh process;
- bare `import langres` stays within the import budget;
- docs and the migration map change in the same PR.

## Experiment acceptance

Run the new recipe through `ArchitectureFactory` and `Experiment`, first on
`EvaluationProtocol.smoke()` with fakes. Add multiple benchmarks/splits/seeds
only after the one-cell path works. Compare quality inside one evaluation
identity and performance inside one compatible hardware cohort. Failed and
missing cells remain in the report.

## Legacy: add a name-selectable four-slot matcher

`Matcher`, `MethodSpec`, and the method registry remain supported for the
classic `Resolver.from_schema(matcher="...")` and older benchmark helpers. Use
that path only when the public feature is genuinely a classic matcher name:

1. implement `Matcher.forward` and emit `PairwiseJudgement` with the correct
   `score_type` and honest cost/usage;
2. register one import-light `MethodSpec` in
   `langres.core.method_registry` with accepted `ModelRef.kind` values,
   default threshold/model, comparator requirement, and optional extra;
3. register strict component persistence if saved models need it;
4. test both dispatch sites and a fresh-process round-trip.

For the explicit mapping from `Blocker`/`Matcher`/`Judge` terminology, see
[Research vocabulary and legacy migration](reference/research-vocabulary.md).
