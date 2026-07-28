# FEBRL3 person **deduplication** benchmark (full dataset)

These two CSVs are the **complete FEBRL3** dataset, the standard **synthetic**
person *deduplication* benchmark: **one** table of 5000 records in which 2000
originals are joined by 3000 corrupted duplicates. Unlike FEBRL4 (vendored
separately under `febrl_dedup`'s sibling `febrl_person/`, which splits originals
and duplicates into two files for a cross-source 1:1 *linkage* task), FEBRL3 is
intra-source: the duplicates sit in the same table as their originals, so the
task is "partition one record set into entities" — deduplication.

## Files

| File | Rows | Description |
|---|---|---|
| `records.csv` | 5000 | The single-source person corpus (`id` + 10 attribute columns) |
| `gold_clusters.csv` | 5000 | Ground-truth entity membership (`record_id,cluster_id`) |

Record columns: `id,given_name,surname,street_number,address_1,address_2,
suburb,postcode,state,date_of_birth,soc_sec_id`. Empty cells are missing values
(FEBRL blanks fields as one of its corruptions; every attribute column except
`postcode` and `soc_sec_id` has some).

### Ground truth is entity membership, not a pair list

`gold_clusters.csv` gives each record's **entity** directly, so the gold
partition is read off, never reconstructed. This matters: gold clusters built by
taking the *transitive closure* of a pairwise link file inherit that file's
errors and can fuse unrelated records into giant components (DBLP-Scholar's
37-record component is this repo's worked example), which inflates every
cluster-based metric. FEBRL3 has no such artifact — the entity is the generator's
own, known by construction.

The generation script asserts this rather than assuming it: `recordlinkage`'s
`load_febrl3(return_links=True)` also ships a pairwise `links` index, and the
script verifies that it is **exactly** the set of within-entity pairs implied by
the record ids (6538 = 6538, set-equal). If a future `recordlinkage` release
changed either, regenerating would fail loudly instead of silently vendoring a
different task.

### Ids are opaque on purpose

Upstream, FEBRL record ids are `rec-<N>-org` and `rec-<N>-dup-<K>`, where `N`
**is** the entity number — original `rec-1496-org` and its duplicate
`rec-1496-dup-1` share it. Keeping those ids verbatim would leak the label into
the record: a schema-less `dedupe()` infers its schema from the record dict, so a
string comparator would score `rec-1496-org` against `rec-1496-dup-1` as a near
match and win for free. So this fixture assigns **opaque** ids `r0000`..`r4999`
in the order `recordlinkage` ships the rows (already shuffled upstream — row 0 is
`rec-1496-org`, row 1 `rec-552-dup-3` — so neighbouring ids are not co-referent
either), and cluster ids `e0000`.. by each entity's first appearance in that
order.

## Cluster-size distribution

2000 entities over 5000 records:

| cluster size | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| entities | 835 | 368 | 256 | 212 | 161 | 168 |

6538 gold match pairs. The 835 singletons are real non-duplicated people, so the
closed-world partition is complete without any synthetic padding — and the
multi-record clusters (up to size 6) are what make this a *dedup* benchmark
rather than a linkage one wearing a different hat: only a gold entity that spans
6 records can charge a matcher for assembling 3 of its 15 within-entity pairs and
stopping. (Over-merging needs no such thing — a false edge between two size-2
entities already produces an oversized predicted component that BCubed
penalises — so multi-record gold buys the **under**-merge direction, not the
over-merge one.)

## Data is fully synthetic (no PII)

FEBRL (**F**reely **E**xtensible **B**iomedical **R**ecord **L**inkage) generates
**fictitious** people from ANU name/address frequency tables, then injects
realistic corruptions (typos, OCR errors, field swaps, missing values) to create
the duplicates. No real person is represented — there is no personally
identifiable information here.

## Attribution & provenance

- **Tool:** vendored via the `recordlinkage` Python toolkit
  (`recordlinkage.datasets.load_febrl3`, version 0.16), which redistributes the
  FEBRL datasets. `recordlinkage` is **BSD-3-Clause** licensed.
  Source: <https://github.com/J535D165/recordlinkage>
- **Data origin:** FEBRL, developed by Peter Christen et al. at the Australian
  National University (ANU). The FEBRL data-generation code and shipped datasets
  are distributed under the **ANUOS License 1.1** (an MPL-style,
  redistribute-with-attribution license).
- **Reference:** P. Christen, "Febrl -- an open source data cleaning,
  deduplication and record linkage system with a graphical user interface,"
  KDD 2008.

**No NonCommercial restriction.** Neither the `recordlinkage` BSD-3-Clause
license nor the ANUOS 1.1 data license carries a NonCommercial term, so bundling
this dataset alongside an Apache-2.0 library is compatible — the same clearance
that admitted FEBRL4 and excluded OpenSanctions (whose Pairs data is CC-BY-NC).

## How this fixture was generated

`recordlinkage` is **not** a langres dependency — it is needed once, transiently,
to materialize this fixture:

```bash
uv run --with recordlinkage --no-project python tmp/gen_febrl_dedup.py
```

```python
import csv, itertools, re
from pathlib import Path
from recordlinkage.datasets import load_febrl3

OUT = Path("src/langres/data/datasets/febrl_dedup")
COLUMNS = ("given_name", "surname", "street_number", "address_1", "address_2",
           "suburb", "postcode", "state", "date_of_birth", "soc_sec_id")
_REC_ID = re.compile(r"rec-(\d+)-(?:org|dup-\d+)")

df, links = load_febrl3(return_links=True)
assert list(df.columns) == list(COLUMNS)

entity_of = {rec_id: _REC_ID.fullmatch(rec_id).group(1) for rec_id in df.index}
rec_to_id = {rec_id: f"r{i:04d}" for i, rec_id in enumerate(df.index)}
cluster_ids: dict[str, str] = {}
for rec_id in df.index:
    cluster_ids.setdefault(entity_of[rec_id], f"e{len(cluster_ids):04d}")

# The shipped pair index must be EXACTLY the within-entity pairs (no closure).
groups: dict[str, list[str]] = {}
for rec_id in df.index:
    groups.setdefault(entity_of[rec_id], []).append(rec_id)
derived = {frozenset(p) for m in groups.values() for p in itertools.combinations(sorted(m), 2)}
assert derived == {frozenset(p) for p in links}

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "records.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(("id", *COLUMNS))
    for rec_id, row in df.iterrows():
        writer.writerow((rec_to_id[rec_id], *["" if v != v or v is None else str(v) for v in row]))
with (OUT / "gold_clusters.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(("record_id", "cluster_id"))
    for rec_id in df.index:
        writer.writerow((rec_to_id[rec_id], cluster_ids[entity_of[rec_id]]))
```

## Usage note

Vendored **for research and benchmarking use only**, to give langres's primary
shipped verb — `dedupe()` — a real single-source deduplication target. No
ownership is claimed over the data; all rights remain with the original authors.
If you redistribute, preserve this attribution.
