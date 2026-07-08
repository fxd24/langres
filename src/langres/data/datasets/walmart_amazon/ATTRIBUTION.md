# Walmart-Amazon (structured) product entity-resolution benchmark

These five CSVs are the **structured** variant of the **Walmart-Amazon** product
matching benchmark, a standard entity-resolution (record-linkage) dataset. Two
e-commerce sources (Walmart and Amazon) each list electronics products across a
clean, comparable set of columns (`id,title,category,brand,modelno,price`); the
task is to find the cross-source pairs that refer to the same real-world
product. This is the **non-`-SM` (full attribute) structured variant** — the one
the entity-matching (DeepMatcher/Ditto/Jellyfish) literature reports on — not the
summarised `-SM` variant.

Like Amazon-Google and Abt-Buy it is a genuinely **hard, unsaturated** benchmark
(reported SOTA pairwise F1 well below 1.0), and it is many-to-many: an Amazon
listing can match several Walmart listings (and vice versa), so gold clusters are
the connected components of the match graph — clusters exceed two records (here
up to size 4).

## Files

| File | Rows | Description |
|---|---|---|
| `tableA.csv` | 2554 | Walmart product records (`id,title,category,brand,modelno,price`) |
| `tableB.csv` | 22074 | Amazon product records (same columns) |
| `train.csv` | 6144 | Labeled pairs (`ltable_id,rtable_id,label`) — fixed literature split |
| `valid.csv` | 2049 | Labeled pairs — fixed literature split |
| `test.csv` | 2049 | Labeled pairs — fixed literature split |

`ltable_id` references a row in `tableA` (Walmart); `rtable_id` references a row
in `tableB` (Amazon). `label == 1` is a match, `label == 0` a non-match. The
three splits are the **fixed literature pair splits** (DeepMatcher/Magellan), so
results are directly comparable to published numbers. Matches are all
cross-source (Walmart <-> Amazon).

**Positive pairs:** 576 (train) + 193 (valid) + 193 (test) = **962** positive
labels (10242 labeled pairs total). Pooled and deduped across splits these give
962 unique positive edges; the closed-world connected-components partition of
those edges yields **1092** within-cluster gold pairs (transitive closure over
the 846 match components — 744 pairs, 88 triples, 14 quads — plus 22820
singletons over the 24628-record corpus).

## Attribution & provenance

Vendored from the **matchbench/Walmart-Amazon** dataset on the Hugging Face Hub,
which redistributes the benchmark for entity-resolution research:

- Source: <https://huggingface.co/datasets/matchbench/Walmart-Amazon> —
  `{tableA.csv, tableB.csv, train.csv, valid.csv, test.csv}` (structured, non-`-SM`).

The Walmart-Amazon benchmark itself originates from the **Magellan / DeepMatcher**
entity-matching benchmark suite (the "Structured/Walmart-Amazon" task) released
by the AnHai Doan group at UW-Madison:

- Mudgal et al., "Deep Learning for Entity Matching: A Design Space Exploration,"
  SIGMOD 2018.
- <https://github.com/anhaidgroup/deepmatcher>

License: redistributed under **CC-BY 4.0** via matchbench.

## Usage note

Vendored **for research and benchmarking use only**, to validate langres blocking
and entity-resolution on a structured, literature-comparable benchmark
(eval-readiness Wave C). No ownership is claimed over the data; all rights remain
with the original authors. If you redistribute, preserve this attribution.
