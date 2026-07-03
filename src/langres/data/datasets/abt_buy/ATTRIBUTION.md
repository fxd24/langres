# Abt-Buy product entity-resolution benchmark

These five CSVs are the **Abt-Buy** product matching benchmark, a standard
entity-resolution (record-linkage) dataset. Two e-commerce sources (Abt
Electronics and Buy.com) each list products; the task is to find the
cross-source pairs that refer to the same real-world product. Unlike
Fodors-Zagat (near-saturated, clean multi-field records) and closer to
Amazon-Google, Abt-Buy is a **textual-hard** benchmark — records carry a
free-text `description` field (frequently missing on the Buy side) rather
than a rich set of clean, comparable columns, so it stresses judges that rely
on short, noisy text.

## Files

| File | Rows | Description |
|---|---|---|
| `tableA.csv` | 1081 | Abt product records (`id,name,description,price`) |
| `tableB.csv` | 1092 | Buy product records (same columns) |
| `train.csv` | 5743 | Labeled pairs (`ltable_id,rtable_id,label`) — fixed literature split |
| `valid.csv` | 1916 | Labeled pairs — fixed literature split |
| `test.csv` | 1916 | Labeled pairs — fixed literature split |

`ltable_id` references a row in `tableA` (Abt); `rtable_id` references a row
in `tableB` (Buy). `label == 1` is a match, `label == 0` a non-match. The
three splits are the **fixed literature pair splits** (DeepMatcher/Magellan),
so results are directly comparable to published numbers. Matches are all
cross-source (Abt <-> Buy). Positive pairs: 616 (train) + 206 (valid) + 206
(test) = 1028.

## Attribution & provenance

Vendored from the **matchbench/Abt-Buy** dataset on the Hugging Face Hub,
which redistributes the benchmark for entity-resolution research:

- Source: <https://huggingface.co/datasets/matchbench/Abt-Buy> —
  `{tableA.csv, tableB.csv, train.csv, valid.csv, test.csv}`

The Abt-Buy benchmark itself originates from the **Magellan / DeepMatcher**
entity-matching benchmark suite (the "Structured/Textual/Abt-Buy" task)
released by the AnHai Doan group at UW-Madison:

- Mudgal et al., "Deep Learning for Entity Matching: A Design Space Exploration,"
  SIGMOD 2018.
- <https://github.com/anhaidgroup/deepmatcher>

## Usage note

Vendored **for research and benchmarking use only**, to validate langres
blocking and entity-resolution on a textual-hard, literature-comparable
benchmark (M4.5/W1.2). No ownership is claimed over the data; all rights
remain with the original authors. If you redistribute, preserve this
attribution.
