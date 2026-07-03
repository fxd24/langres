# FEBRL4 person entity-resolution benchmark (subset)

These three CSVs are a **~500-per-side subset of FEBRL4**, a standard
**synthetic** person record-linkage benchmark. Two sources each list people;
`dataset A` holds originals (`rec-N-org`) and `dataset B` holds one corrupted
duplicate each (`rec-N-dup-0`). The task is the cross-source 1:1 linkage: find
the pairs that refer to the same synthetic person.

## Files

| File | Rows | Description |
|---|---|---|
| `person_a.csv` | 500 | Original person records (subset of FEBRL4 `dataset4a`) |
| `person_b.csv` | 500 | Their corrupted duplicates (subset of FEBRL4 `dataset4b`) |
| `person_perfectMapping.csv` | 500 | Ground-truth 1:1 links (`id_a,id_b`) |

Record columns: `rec_id,given_name,surname,street_number,address_1,address_2,
suburb,postcode,state,date_of_birth,soc_sec_id`. Fields are plain (no wrapping
quotes); empty cells are missing values. The loader prefixes each `rec_id` with
its source (`a`/`b`) to make ids globally unique, mirroring the Fodors-Zagat
(`f`/`z`) and Amazon-Google (`a`/`g`) loaders.

## Data is fully synthetic (no PII)

FEBRL (**F**reely **E**xtensible **B**iomedical **R**ecord **L**inkage) generates
**fictitious** people from ANU name/address frequency tables, then injects
realistic corruptions (typos, OCR errors, field swaps, missing values) to create
the duplicates. No real person is represented — there is no personally
identifiable information here.

## Attribution & provenance

- **Tool:** vendored via the `recordlinkage` Python toolkit
  (`recordlinkage.datasets.load_febrl4`), which redistributes the FEBRL datasets.
  `recordlinkage` is **BSD-3-Clause** licensed.
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
this subset alongside an Apache-2.0 library is compatible. (This is why FEBRL4
was chosen over OpenSanctions, whose Pairs data is CC-BY-NC.)

## How this subset was generated

```python
from recordlinkage.datasets import load_febrl4

dfA, dfB, links = load_febrl4(return_links=True)
# Sort the true (org, dup) links by numeric rec index, take the first 500,
# emit dfA rows -> person_a.csv, dfB rows -> person_b.csv, and the pairs ->
# person_perfectMapping.csv (columns id_a,id_b). NaN cells become empty strings.
```

Run transiently with `uv run --with recordlinkage python ...` (recordlinkage is
**not** a langres dependency — it is only needed once to materialize this
fixture).

## Usage note

Vendored **for research and benchmarking use only**, to validate that langres
resolves a second entity type (Person) config-only at $0. No ownership is claimed
over the data; all rights remain with the original authors. If you redistribute,
preserve this attribution.
