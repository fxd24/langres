# DBLP-ACM bibliographic entity-resolution benchmark

These five CSVs are the **DBLP-ACM** citation matching benchmark, a standard
entity-resolution (record-linkage) dataset. Two bibliographic sources (the
**DBLP** and **ACM** digital libraries) each list publications; the task is to
find the cross-source pairs that refer to the same paper. Unlike the
textual-hard product benchmarks (Abt-Buy, Amazon-Google), DBLP-ACM is a
**clean, near-saturated** benchmark (reported SOTA pairwise F1 ~0.98): each
record carries four well-populated, comparable fields (`title`, `authors`,
`venue`, `year`), so blocking and matching are comparatively easy — it is the
clean end of the difficulty spectrum in the replication matrix.

## Files

| File | Rows | Description |
|---|---|---|
| `tableA.csv` | 2616 | DBLP citation records (`id,title,authors,venue,year`) |
| `tableB.csv` | 2294 | ACM citation records (same columns) |
| `train.csv` | 7417 | Labeled pairs (`ltable_id,rtable_id,label`) — fixed literature split |
| `valid.csv` | 2473 | Labeled pairs — fixed literature split |
| `test.csv` | 2473 | Labeled pairs — fixed literature split |

`ltable_id` references a row in `tableA` (DBLP, prefixed `a` in the corpus);
`rtable_id` references a row in `tableB` (ACM, prefixed `b`). `label == 1` is a
match, `label == 0` a non-match. The three splits are the **fixed literature
pair splits** (DeepMatcher/Magellan), so results are directly comparable to
published numbers. Matches are all cross-source (DBLP <-> ACM) and strictly
1:1: positive pairs are 1332 (train) + 444 (valid) + 444 (test) = **2220**
(no duplicate pair spans two splits), yielding 2220 gold match clusters (each a
single DBLP-ACM pair) plus 470 unmatched singletons over the 4910-record corpus.

## Attribution & provenance

Vendored from the **matchbench/DBLP-ACM** dataset on the Hugging Face Hub,
which redistributes the benchmark for entity-resolution research:

- Source: <https://huggingface.co/datasets/matchbench/DBLP-ACM> —
  `{tableA.csv, tableB.csv, train.csv, valid.csv, test.csv}`

The DBLP-ACM benchmark itself originates from the **Magellan / DeepMatcher**
entity-matching benchmark suite (the "Structured/DBLP-ACM" task), which in turn
draws on the DB-group benchmark data released by the AnHai Doan group at
UW-Madison and the Leipzig DB group:

- Mudgal et al., "Deep Learning for Entity Matching: A Design Space Exploration,"
  SIGMOD 2018.
- <https://github.com/anhaidgroup/deepmatcher>

License: **CC-BY 4.0**, via the matchbench redistribution.

## Usage note

Vendored **for research and benchmarking use only**, to validate langres
blocking and entity-resolution on a clean, literature-comparable bibliographic
benchmark (Wave C). No ownership is claimed over the data; all rights remain
with the original authors. If you redistribute, preserve this attribution.
