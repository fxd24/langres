# DataProfileReport

A self-contained HTML **data**-profile tearsheet -- and its text-first surfaces --
that profiles *the data itself* (not the pipeline output): class balance, gold
cluster structure, per-field data quality, how separable matches are from
non-matches, and how your precomputed embeddings are distributed.

## What it profiles

The report is a **bag of sections**, one self-contained `ProfileSection` per
metric family:

| Section | Reports |
|---|---|
| `HeroSection` | the at-a-glance KPIs: records, clusters, positive-pair prevalence, `1:N` class imbalance, separability AUC |
| `LabelStructureSection` | gold cluster-size distribution, singleton rate, positive-pair prevalence + class imbalance |
| `CorpusFieldSection` | per-field non-null rate, cardinality, value-length distribution (most-missing first) |
| `SeparabilitySection` | matches-vs-non-matches similarity histogram + AUC for a signal (rapidfuzz string by default, or an embedding cosine) |
| `EmbeddingSection` | one precomputed model's L2-norm distribution (provenance, health) |
| `EmbeddingComparisonSection` | several models' norm distributions as shared-axis small multiples |

## The composability contract (graceful degradation)

The report **holds whatever sections you give it and renders exactly those** -- it
computes nothing itself. Two consequences:

- **Compose the subset you want.** Build only the sections you care about and hand
  them to `DataProfileReport([...])`, or use `from_benchmark` / `from_records` and
  narrow with `include=` (a selector of section *kinds*; an unknown kind raises).
- **A missing input is never an error.** No gold clustering -> no label-structure
  or separability section. No `schema` -> no string separability. No embeddings ->
  no embedding sections. The section is simply absent; the report still renders.

## Text-first; HTML optional

Every section and the report expose the same render ladder, so **HTML is never
required**:

- `print(report)` / `report.to_markdown()` -- the primary way to read it.
- `report.summary` -- a flat dict of headline numbers (log it, assert on it).
- `report.to_dict()` -- the machine/JSON surface (persist, diff across runs).
- `section.rows()` -- `pd.DataFrame(section.rows())`-ready, with no pandas dependency.
- `report.to_html()` -- the optional `$0`, self-contained tearsheet (inline SVG, no
  CDN, no matplotlib, light/dark aware).

## Memory-efficient embeddings (precomputed, consumed-only)

The report **never generates embeddings** -- it consumes vectors a pipeline
already paid to compute, through the read-only `EmbeddingSource` protocol. It
carries **no `[semantic]` dependency**: profiling a given matrix needs only numpy.

- `ArraySource(name, ids, matrix)` -- wraps an in-memory matrix (tests, small corpora).
- `NpySource(name, path, ids)` -- memmaps a `.npy` (`mmap_mode="r"`) and gathers only
  the requested rows, so a 30 GB matrix profiles in `O(batch * dim)` memory.
- `NpySource.from_anchor_store(state_dir, name)` -- reuses a persisted vector index.

If you have no vectors yet, the `[semantic]`-gated `from_embedder(records, model,
out_path=...)` embeds once with sentence-transformers, writes a `.npy` (+ an ids
sidecar), and returns an `NpySource` -- a separate on-ramp that keeps the report
itself dependency-free.

## Usage

```python
from langres.core.data_profile import ArraySource, DataProfileReport

# Precomputed vectors (from a VectorBlocker, an AnchorStore, or a .npy on disk).
emb = ArraySource("all-MiniLM-L6-v2", ids, matrix)

report = DataProfileReport.from_records(
    records,                 # list of field dicts
    gold=gold_clusters,      # optional: enables label structure + separability
    schema=MySchema,         # optional: enables the string-similarity signal
    embeddings=[emb],        # optional: one embedding section per source
    include={"hero", "label_structure", "separability"},  # optional kind selector
)

print(report)                # text-first: markdown to the terminal
report.summary               # {"Overview.prevalence": 0.0012, ...}
report.to_html("out.html")   # optional $0 tearsheet
```

`from_benchmark("abt_buy", embeddings=[...])` is the registered-benchmark twin.

::: langres.core.data_profile
