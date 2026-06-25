# langres: Use Case Taxonomy & Development Roadmap

> **Note (2026-06-25):** Several components named below — `tasks.*`
> (`DeduplicationTask`, `EntityLinkingTask`, `RecordLinkageTask`),
> `core.Canonicalizer`, `data.ReviewQueue`, `data.SyntheticGenerator`, and
> `Blocker.stream_against` — are **not yet implemented**; they describe the
> intended API. See **[ROADMAP.md](ROADMAP.md)** for the delivery timeline
> (the golden-record / canonicalization loop lands in M5; the cold-start
> data generator in M1). The "Status" column below predates the roadmap; the
> roadmap's use-case compass (§2.5) is the current source of truth.

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

**langres Implementation (V1 Core):**

This is the primary "hello world" use case for langres.

- **High-Level API:** `langres.tasks.DeduplicationTask`
- **Core Components:** `blockers.DedupeBlocker` (using `stream(data)`), any Flow (e.g., `flows.CompanyFlow`), `core.Clusterer`.

### Use Case 2: Entity Linking (Asymmetric Resolution)

**Formal Definition:**

- **Input Structure:** A source dataset S and an authoritative target dataset T.
- **Output Structure:** A mapping M from each record s in S to one record t in T, or to NIL.
- **Cardinality:** N:1 (many source records can map to one target entity).
- **Authority Model:** The target dataset T is the fixed "source of truth."
- **Temporal Aspect:** Static snapshot.

**langres Implementation (V1 Core):**

This is the second core use case, supported out-of-the-box.

- **High-Level API:** `langres.tasks.EntityLinkingTask`
- **Core Components:** `blockers.LinkingBlocker` (using `stream_against(source, target)` and handling schema mapping), any Flow. The Task orchestrator handles the 1:1 mapping logic (instead of clustering) and NIL thresholding.

### Use Case 10: Fuzzy Foreign Key Resolution

**Formal Definition:**

A specialized sub-type of Entity Linking (Use Case 2).

- **Input Structure:** A source table S (e.g., Orders) and a target table T (e.g., Customers).
- **Output Structure:** A mapping from S to the primary keys of T.
- **Cardinality:** N:1 (many "dirty" foreign keys map to one primary key).
- **Authority Model:** Target table T is the authority.
- **Temporal Aspect:** Static snapshot.

**langres Implementation (V1 Core):**

Handled identically to Use Case 2 by the `langres.tasks.EntityLinkingTask`.

### Use Case 3: Record Linkage (Multi-Source Symmetric Resolution)

**Formal Definition:**

- **Input Structure:** Two or more datasets, D_1, D_2, ... D_k.
- **Output Structure:** A set of clusters C, where each cluster can contain record IDs from any input dataset.
- **Cardinality:** N:M (records can link across sources, which are then resolved to N:1 clusters).
- **Authority Model:** All datasets are equal peers; no single source of truth.
- **Temporal Aspect:** Static snapshot.

**langres Implementation (V1.1 Extension):**

This is a natural extension of our core architecture.

- **High-Level API:** `langres.tasks.RecordLinkageTask`
- **Core Components:** Requires a Blocker that can use `stream(datasets: List[...])` to find pairs across all datasets. The `core.Clusterer` already supports this.

### Use Case 4: Master Data Creation (Consolidation)

**Formal Definition:**

- **Input Structure:** A set of clusters from a Deduplication or Record Linkage task.
- **Output Structure:** A new, canonical dataset M ("golden records").
- **Cardinality:** N:1 (many clustered records are merged into one master record).
- **Authority Model:** The new master dataset M becomes the authoritative source.
- **Temporal Aspect:** Static (creates a new snapshot).

**langres Implementation (V1.1 Extension):**

This is the "last mile" of ER, which we support as a distinct architectural pillar.

- **Core Components:** `langres.core.Canonicalizer`. This V1.1 module will provide survivorship rules (e.g., "most_recent", "most_frequent", "merge_unique") to build the master records from the clusters.

### Use Case 9: Negative Constraints (Constrained Clustering)

**Formal Definition:**

- **Input Structure:** A standard ER task plus a set of "cannot-link" constraints N.
- **Output Structure:** A set of clusters C that respects all constraints in N.
- **Cardinality:** N:1 (same as deduplication, but constrained).
- **Authority Model:** The constraints are an additional, definitive authority.
- **Temporal Aspect:** Static snapshot.

**langres Implementation (V1 Core):**

This is a critical, production-ready feature.

- **Core Components:** `langres.core.Clusterer` has a `constraints` parameter (`cluster(..., constraints=N)`) that will enforce these rules during clustering.

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

**How langres helps:** You can use langres to train and compile the Flow (the "brain") that a streaming application (built on Flink, Spark Streaming, etc.) would then import and use for its real-time scoring logic. langres provides the brain, not the real-time body.

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

| Use Case | langres Component(s) | Status |
|----------|---------------------|--------|
| 1. Deduplication | `tasks.DeduplicationTask`, `core.Clusterer` | V1 Core |
| 2. Entity Linking | `tasks.EntityLinkingTask`, `blockers.LinkingBlocker` | V1 Core |
| 10. Fuzzy FK Resolution | (Covered by `tasks.EntityLinkingTask`) | V1 Core |
| 9. Negative Constraints | `core.Clusterer(constraints=...)` | V1 Core |
| Human-in-the-Loop | `data.ReviewQueue`, `langres.ui` | V1 Core |
| Optimization | `core.Optimizer` (Optuna, DSPy) | V1 Core |
| Data Generation | `data.SyntheticGenerator` | V1 Core |
| 3. Record Linkage | `tasks.RecordLinkageTask`, `Blocker.stream(datasets=...)` | V1.1 Extension |
| 4. Master Data Creation | `core.Canonicalizer` (Survivorship) | V1.1 Extension |
| 8. Privacy-Preserving (PPRL) | Custom PPRLFlow and PPRLBlocker | Future Scope |
| 7. Collective (Graph) | (Requires new GraphClusterer) | Out of Scope |
| 5. Streaming Resolution | (Architectural Mismatch - See Note) | Out of Scope |
| 6. Temporal Evolution | (Architectural Mismatch) | Out of Scope |

**Note on "Out of Scope":** langres is not designed to be a streaming or temporal engine itself. However, its core purpose is to train and compile the `core.Module` ("Flow") which can then be exported and used by an external streaming/temporal system.
