# Fodors-Zagat restaurant entity-resolution benchmark

These three CSVs are the **Fodors-Zagat** restaurant matching benchmark, a
standard entity-resolution (record-linkage) dataset. Two restaurant guides
(Fodor's and Zagat) list overlapping sets of restaurants; the task is to find
the cross-source pairs that refer to the same real-world restaurant.

## Files

| File | Rows | Description |
|---|---|---|
| `fodors.csv` | 533 | Fodor's guide restaurant records |
| `zagats.csv` | 331 | Zagat guide restaurant records |
| `fodors-zagats_perfectMapping.csv` | 112 | Ground-truth matching pairs (`fodors_id,zagats_id`) |

Record columns: `id,name,addr,city,phone,type,class`. Fields are wrapped in
single quotes by the source (with backslash-escaped inner quotes); the loader
strips the wrapping quotes. The `class` column is a cluster id shared by
matching records, but the canonical ground truth used here is the explicit
`perfectMapping` file.

## Attribution & provenance

Vendored from the **QCRI DeepER** repository, which redistributes the dataset
for entity-resolution research:

- Source: <https://github.com/qcri/DeepER> —
  `data/DataSets/fodors-zagats/{fodors.csv, zagats.csv, fodors-zagats_perfectMapping.csv}`
- DeepER reference: Ebraheem et al., "Distributed Representations of Tuples for
  Entity Resolution," PVLDB 2018.

The Fodors-Zagat benchmark itself originates from the restaurant-matching
dataset long used by the database / record-linkage community (e.g. the RIDDLE
repository and subsequent ER literature).

## Usage note

Vendored **for research and benchmarking use only**, to validate langres
blocking and gold-set bootstrapping. No ownership is claimed over the data;
all rights remain with the original authors. If you redistribute, preserve
this attribution.
