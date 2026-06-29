# Amazon-Google product entity-resolution benchmark

These five CSVs are the **Amazon-Google** product matching benchmark, a standard
entity-resolution (record-linkage) dataset. Two e-commerce sources (Amazon and
Google Products) each list software products; the task is to find the
cross-source pairs that refer to the same real-world product. Unlike the
(near-saturated) Fodors-Zagat benchmark, Amazon-Google is genuinely hard —
reported state-of-the-art pairwise F1 is roughly 0.50-0.75.

## Files

| File | Rows | Description |
|---|---|---|
| `tableA.csv` | 1363 | Amazon product records (`id,title,manufacturer,price`) |
| `tableB.csv` | 3226 | Google product records (same columns) |
| `train.csv` | 6874 | Labeled pairs (`ltable_id,rtable_id,label`) — fixed literature split |
| `valid.csv` | 2293 | Labeled pairs — fixed literature split |
| `test.csv` | 2293 | Labeled pairs — fixed literature split |

`ltable_id` references a row in `tableA` (Amazon); `rtable_id` references a row
in `tableB` (Google). `label == 1` is a match, `label == 0` a non-match. The
three splits are the **fixed literature pair splits** (DeepMatcher/Magellan), so
results are directly comparable to published numbers. Matches are all
cross-source (Amazon <-> Google). Positive pairs: 699 (train) + 234 (valid) +
234 (test) = 1167.

## Attribution & provenance

Vendored from the **matchbench/amazon-google** dataset on the Hugging Face Hub,
which redistributes the benchmark for entity-resolution research:

- Source: <https://huggingface.co/datasets/matchbench/amazon-google> —
  `{tableA.csv, tableB.csv, train.csv, valid.csv, test.csv}`

The Amazon-Google benchmark itself originates from the **Magellan / DeepMatcher**
entity-matching benchmark suite (the "Structured/Amazon-Google" task) released by
the AnHai Doan group at UW-Madison:

- Mudgal et al., "Deep Learning for Entity Matching: A Design Space Exploration,"
  SIGMOD 2018.
- <https://github.com/anhaidgroup/deepmatcher>

## Usage note

Vendored **for research and benchmarking use only**, to validate langres
blocking and entity-resolution on a harder, literature-comparable benchmark. No
ownership is claimed over the data; all rights remain with the original authors.
If you redistribute, preserve this attribution.
