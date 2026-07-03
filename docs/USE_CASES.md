# langres: Use Case Taxonomy & Development Roadmap

> **⚠️ Reality check (2026-07-02).** This document is a **taxonomy + roadmap**,
> not an API reference. Several component names used below never existed as
> code and were doc fiction: the whole `langres.tasks.*` layer
> (`DeduplicationTask`, `EntityLinkingTask`, `RecordLinkageTask`),
> `langres.flows.*` / `blockers.*` (`CompanyFlow`, `DedupeBlocker`,
> `LinkingBlocker`), `core.Optimizer`,
> `data.ReviewQueue`, `data.SyntheticGenerator`, `Clusterer(constraints=...)`,
> and `Blocker.stream_against` / `Resolver.link` **as working code**.
>
> **What actually ships today** for the use cases below is the verb DX layer
> (`link` / `dedupe` — two verbs; a third, incremental one is roadmap for M5) +
> `Resolver` + `langres.core` primitives:
>
> - **Deduplication (UC1):** ✅ `langres.dedupe(records)` or
>   `Resolver.from_schema(schema).resolve(records)`.
> - **Pairwise match:** ✅ `langres.link(left, right)` → `LinkVerdict`.
> - **Incremental single-record assignment (UC10):** ✅ (M5/W2.2)
>   `resolver.build_anchor_store(records)` then `resolver.assign(new_record)`
>   → `ClusterDelta` (`link` to a stable entity id, or `new`); the
>   serializable `AnchorStore` persists it. See `examples/incremental_assign.py`.
> - **Cross-source entity linking (UC2, UC3):** 🚧 `Resolver.link()` and
>   `Resolver.stream_against()` remain `NotImplementedError` stubs reserved for
>   later **M5** waves (distinct from the single-record `assign` above).
> - **Master Data / golden records (UC4):** ✅ (M5/W2.3)
>   `core.Canonicalizer` merges an entity's records into one golden record via
>   named survivorship strategies (`most_complete` default + per-field
>   overrides); `enrich(golden, mention)` is the sparse-mention → golden-record
>   enrichment loop over `assign`. See `examples/canonicalizer_enrichment.py`.
> - **Negative constraints (UC9):** 🚧 `Clusterer` takes only a `threshold`;
>   no cannot-link support today.
>
> Below, "langres Implementation" blocks describe the **intended** design and
> are kept for the taxonomy; treat the API names as roadmap unless listed as
> shipping above. See **[ROADMAP.md](ROADMAP.md)** (§2 use-case compass) for the
> milestone-accurate source of truth and **[../README.md](../README.md)** for the
> real API. The Status column in §3 has been corrected to match the code.

## 1. Introduction

Entity Resolution (ER) is not a single problem but a wide spectrum of use cases. A core design principle of langres is to provide a robust, flexible, and lightweight framework for the most common and critical ER tasks.

This document serves two purposes:

- **Formal Taxonomy:** To provide clear, formal definitions for the primary use cases in the ER landscape.
- **Development Roadmap:** To define the scope of langres, showing which use cases are supported in the initial release (V1 Core) and which are planned for future development.

langres is intentionally designed to be a component-based "glue" framework, not a monolithic "do-everything" system. We will begin by mastering the most critical batch-oriented tasks (Deduplication and Linking) and build upon that foundation.

## 2. Formal Taxonomy of ER Use Cases

To formally distinguish use cases, we use a consistent framework for classification.

### Framework for Classification

To formally distinguish use cases, we specify five key properties:

- **Input Structure:** The number of datasets and their relationships (e.g., 1 dataset, 2 datasets).
- **Output Structure:** The artifact being produced (e.g., clusters, 1:1 mappings).
- **Cardinality:** The relationship mapping between records (e.g., N:1, N:M).
- **Authority Model:** The "source of truth" (e.g., no authority, target is authority).
- **Temporal Aspect:** The time dimension of the data (e.g., static snapshot, streaming).

### Use Case 1: Deduplication (Single-Source Resolution)

**Formal Definition:**

- **Input Structure:** A single dataset, D.
- **Output Structure:** A set of clusters (equivalence classes), C = {c_1, c_2, ... c_k}.
- **Cardinality:** N:1 (many records map to one discovered entity).
- **Authority Model:** No pre-existing authority; clusters discover latent entities.
- **Temporal Aspect:** Static snapshot.

**langres Implementation — ✅ ships today:**

This is the primary "hello world" use case for langres.

- **DX layer:** `langres.dedupe(records)` (schema-optional, zero-label,
  spend-capped).
- **Core path:** `Resolver.from_schema(schema).resolve(records)`, composing an
  `AllPairsBlocker`/`VectorBlocker`, a `StringComparator` + judge (`Module`),
  and a `core.Clusterer`.

### Use Case 2: Entity Linking (Asymmetric Resolution)

**Formal Definition:**

- **Input Structure:** A source dataset S and an authoritative target dataset T.
- **Output Structure:** A mapping M from each record s in S to one record t in T, or to NIL.
- **Cardinality:** N:1 (many source records can map to one target entity).
- **Authority Model:** The target dataset T is the fixed "source of truth."
- **Temporal Aspect:** Static snapshot.

**langres Implementation — 🚧 roadmap (M5):**

Cross-source, asymmetric linking is **not yet built**. `Resolver.link(left,
right)` and `Resolver.stream_against(records)` exist only as
`NotImplementedError` stubs reserved for later M5 waves. What *does* ship (M5/
W2.2) is **single-record incremental assignment against an anchor store**:
`resolver.build_anchor_store(target_records)` then `resolver.assign(new_record)`
→ `ClusterDelta` (`link` to a stable entity id in T, or `new`). For a *pairwise*
match decision today, use `langres.link(left, right)` → `LinkVerdict`; for
single-source clustering, use `dedupe` / `Resolver.resolve` (UC1).

### Use Case 10: Fuzzy Foreign Key Resolution

**Formal Definition:**

A specialized sub-type of Entity Linking (Use Case 2).

- **Input Structure:** A source table S (e.g., Orders) and a target table T (e.g., Customers).
- **Output Structure:** A mapping from S to the primary keys of T.
- **Cardinality:** N:1 (many "dirty" foreign keys map to one primary key).
- **Authority Model:** Target table T is the authority.
- **Temporal Aspect:** Static snapshot.

**langres Implementation — 🚧 roadmap (M5):**

A special case of Entity Linking (UC2); it lands with the same M5
`Resolver.link` / `stream_against` work. Not available today.

### Use Case 3: Record Linkage (Multi-Source Symmetric Resolution)

**Formal Definition:**

- **Input Structure:** Two or more datasets, D_1, D_2, ... D_k.
- **Output Structure:** A set of clusters C, where each cluster can contain record IDs from any input dataset.
- **Cardinality:** N:M (records can link across sources, which are then resolved to N:1 clusters).
- **Authority Model:** All datasets are equal peers; no single source of truth.
- **Temporal Aspect:** Static snapshot.

**langres Implementation — 🚧 roadmap (post-M5):**

A natural extension of the core architecture, but **not built**. It would need a
multi-source blocker; `core.Clusterer` already produces clusters from whatever
pairs it is given. Tracked as post-M5/config work in
[ROADMAP.md](ROADMAP.md).

### Use Case 4: Master Data Creation (Consolidation)

**Formal Definition:**

- **Input Structure:** A set of clusters from a Deduplication or Record Linkage task.
- **Output Structure:** A new, canonical dataset M ("golden records").
- **Cardinality:** N:1 (many clustered records are merged into one master record).
- **Authority Model:** The new master dataset M becomes the authoritative source.
- **Temporal Aspect:** Static (creates a new snapshot).

**langres Implementation — ✅ ships today (M5/W2.3):**

The "last mile" of ER. `core.Canonicalizer` merges a group of records (a
`resolve` cluster, an `AnchorStore` entity, or any `list[dict]`) into one golden
record by resolving each field independently with a named survivorship strategy:
`most_complete` (the default — prefer the value from the richest source record),
`longest`, `most_frequent`, `most_recent` (needs a designated `timestamp_field`),
and `first`/`source_priority`, all per-field overridable. `enrich(golden,
mention)` folds a newly-linked mention into an existing golden record via the
*same* survivorship path — the progressive-enrichment loop over `Resolver.assign`
(a sparse mention fills fields the golden record lacked). The policy round-trips
through the config-registry artifact seam (no pickle). See
`examples/canonicalizer_enrichment.py`.

### Use Case 9: Negative Constraints (Constrained Clustering)

**Formal Definition:**

- **Input Structure:** A standard ER task plus a set of "cannot-link" constraints N.
- **Output Structure:** A set of clusters C that respects all constraints in N.
- **Cardinality:** N:1 (same as deduplication, but constrained).
- **Authority Model:** The constraints are an additional, definitive authority.
- **Temporal Aspect:** Static snapshot.

**langres Implementation — 🚧 roadmap:**

**Not built.** `core.Clusterer` today takes only a `threshold`; there is no
`constraints` / cannot-link parameter. Constrained clustering is a planned
extension.

### Use Case 8: Privacy-Preserving Record Linkage (PPRL)

**Formal Definition:**

- **Input Structure:** Two or more datasets from parties who cannot share raw data.
- **Output Structure:** A set of matches (links or clusters) computed on encrypted or hashed data.
- **Cardinality:** N:M (same as Record Linkage).
- **Authority Model:** Distributed; no party has a full view.
- **Temporal Aspect:** Static snapshot.

**langres Implementation (Future Scope):**

The langres architecture is flexible enough to support this. It requires implementing a PPRLBlocker and a PPRLFlow that operate on these encoded representations. This is a specialized extension planned for the future.

### Use Case 7: Collective (Graph) Resolution

**Formal Definition:**

- **Input Structure:** A set of entities and the relationships between them (a graph).
- **Output Structure:** A set of clusters C where the decision is jointly inferred using both attributes and relational evidence.
- **Cardinality:** N:M (constrained by the graph structure).
- **Authority Model:** Collaborative; evidence is combined from attributes and relations.
- **Temporal Aspect:** Static snapshot.

**langres Implementation (Out of Scope):**

This is architecturally different. Our `core.Module` is stateless and pairwise. Collective resolution requires a stateful, graph-native inference engine. langres is not designed for this.

### Use Case 5: Streaming Resolution

**Formal Definition:**

- **Input Structure:** A stream of single, incoming records r_1, r_2, ...
- **Output Structure:** For each record, a real-time decision: MERGE or CREATE.
- **Cardinality:** 1:1 (one incoming record maps to one decision).
- **Authority Model:** A persistent, growing set of master entities.
- **Temporal Aspect:** Streaming (real-time).

**langres Implementation (Out of Scope):**

This is a fundamentally different (online, low-latency, stateful) architecture. langres is a batch-oriented framework.

**How langres helps:** the intended role is to build and calibrate the matching
**judge** (the "brain") that a streaming application (built on Flink, Spark
Streaming, etc.) would then import and reuse for its real-time scoring logic.
langres provides the brain, not the real-time body.

### Use Case 6: Temporal Evolution

**Formal Definition:**

- **Input Structure:** Time-series snapshots of datasets, D(t_1), D(t_2), ...
- **Output Structure:** A model of entity identity through time, including events like splits and mergers.
- **Cardinality:** N:M (entities can split (1:N) or merge (N:1) over time).
- **Authority Model:** Each snapshot is authoritative for its time period.
- **Temporal Aspect:** Historical (time-series).

**langres Implementation (Out of Scope):**

This is a temporal graph problem, not a standard clustering problem. Our clustering model (based on transitive closure) does not support non-transitive "split" events.

## 3. langres Development Roadmap

This table outlines our development priorities, clearly separating the initial, core release from planned extensions.

The "langres Component(s)" column lists the **real, shipping** API where one
exists, and the intended (not-yet-built) design otherwise.

| Use Case | langres Component(s) | Status |
|----------|---------------------|--------|
| 1. Deduplication | `dedupe(records)` / `Resolver.from_schema(...).resolve(...)`, `core.Clusterer` | ✅ **Shipping** |
| Pairwise match verdict | `link(left, right)` → `LinkVerdict` | ✅ **Shipping** |
| Optimization / calibration | `core.calibration.derive_threshold`, `core.optimizers.BlockerOptimizer` (Optuna) | ✅ **Partial** (threshold calibration + blocker tuning; no full `Optimizer`) |
| 2. Entity Linking (cross-source) | `Resolver.link` / `stream_against` (stubs today) | 🚧 Roadmap (M5) |
| 10. Incremental single-record assign | `Resolver.build_anchor_store(...)` → `Resolver.assign(record)` → `ClusterDelta`; `core.AnchorStore` | ✅ **Shipping** (M5/W2.2) |
| 4. Master Data Creation | `core.Canonicalizer` (survivorship + `enrich` loop) | ✅ **Shipping** (M5/W2.3) |
| Set-wise / trained judge families | SelectJudge, Fellegi–Sunter, RandomForest — not built | 🚧 Roadmap (M4.5) |
| 9. Negative Constraints | constrained `Clusterer` — not built | 🚧 Roadmap |
| Human-in-the-Loop | correction-harvest contract (`Correction`/`CorrectionLog`) + `harvest_labeled_pairs` → `derive_threshold`; review-queue UX stays downstream | 🟡 **Harvest shipping** (M5/W2.4); `fit()` wiring next |
| Data Generation | synthetic generator — not built | 🚧 Roadmap |
| 3. Record Linkage | multi-source blocker — not built | 🚧 Roadmap (post-M5) |
| 8. Privacy-Preserving (PPRL) | custom PPRL blocker + judge | ⚪ Future / out of scope |
| 7. Collective (Graph) | graph-native clusterer | ⚪ Out of scope |
| 5. Streaming Resolution | (architectural mismatch — see note) | ⚪ Out of scope |
| 6. Temporal Evolution | (architectural mismatch) | ⚪ Out of scope |

**Note on "Out of Scope":** langres is a **batch** framework, not a streaming or
temporal engine. Its intended role there is to build and calibrate the matching
**judge** (the "brain") that an external streaming/temporal system can then reuse
for its own real-time scoring.
