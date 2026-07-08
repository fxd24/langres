# WDC-computers product entity-resolution benchmark

These five CSVs are the **computers** category of the **WDC (Web Data Commons)
product-matching** corpus, a standard entity-resolution (record-linkage)
benchmark built from schema.org product offers crawled from the public web. Two
sources (`tableA`, `tableB`) each list web product offers; the task is to find
the cross-source pairs that refer to the same real-world product. Like
Amazon-Google and Abt-Buy, this is a **textual-hard** benchmark: each record is a
single free-text `title` blob (title + specs + brand + retailer noise, often
containing multiple language-tagged fragments), so it stresses judges that rely
on short, noisy, heterogeneous text.

## Files

| File | Rows | Description |
|---|---|---|
| `tableA.csv` | 2204 | Left web-offer records (`id,title`) |
| `tableB.csv` | 2443 | Right web-offer records (same columns) |
| `train.csv` | 2231 | Labeled pairs (`ltable_id,rtable_id,label`) — fixed literature split |
| `valid.csv` | 536 | Labeled pairs — fixed literature split |
| `test.csv` | 1098 | Labeled pairs — fixed literature split |

`ltable_id` references a row in `tableA`; `rtable_id` references a row in
`tableB`. `label == 1` is a match, `label == 0` a non-match. The three splits are
the **fixed literature pair splits** (Magellan/DeepMatcher-style), so results are
directly comparable to published numbers. Matches are all cross-source
(A <-> B). Positive pairs: 557 (train) + 149 (valid) + 299 (test) = 1005 labeled
positives, which pool to **986 unique** positive pairs (a few positives recur
across splits).

**Many-to-many, closed-world gold.** Unlike a strict 1:1 benchmark, a single
offer can match several others: the connected components of the 986 positive
pairs form **877 match clusters** (784 of size 2, 77 of size 3, 16 of size 4)
over 1863 matched records. The loader completes this to a closed-world partition
(match components + singletons over the 4647-record corpus), so its
`gold_pairs` are the **transitive closure**: **1111** within-cluster pairs (the
count the loader-contract test pins, mirroring Amazon-Google's many-to-many
gold).

## Derived seen/unseen slice (Wave D)

`wdc_computers.wdc_slice_map(split)` tags each `split` pair `seen`/`half_seen`/
`unseen` by how many of its two record ids appear in **any** `train.csv` pair
(positive or negative). This is a **derived** slice computed from train-pair
membership — it is **not** a shipped tag in this matchbench mirror. It lets Wave D
demonstrate the honest seen -> unseen F1 drop at a single fixed threshold. Note:
this is the *computers* category of the older WDC product corpus, **not** the
newer full "WDC Products" benchmark that ships its own seen/unseen entity split.

## Attribution & provenance

Vendored from the **matchbench/WDC-computers** dataset on the Hugging Face Hub,
which redistributes the benchmark for entity-resolution research:

- Source: <https://huggingface.co/datasets/matchbench/WDC-computers> —
  `{tableA.csv, tableB.csv, train.csv, valid.csv, test.csv}`

The WDC product-matching corpus originates from the **Web Data Commons** project
(University of Mannheim), which extracts schema.org product data from the Common
Crawl and packages product-matching training/gold sets by category (computers,
cameras, watches, shoes):

- Primbs, Petrovski, Bizer et al., "The WDC Training Dataset and Gold Standard
  for Large-Scale Product Matching."
- <http://webdatacommons.org/largescaleproductcorpus/>

The two-table + fixed `train`/`valid`/`test` pair-split packaging here follows the
**Magellan / DeepMatcher** entity-matching benchmark convention (AnHai Doan
group, UW-Madison):

- Mudgal et al., "Deep Learning for Entity Matching: A Design Space Exploration,"
  SIGMOD 2018.
- <https://github.com/anhaidgroup/deepmatcher>

License: **CC-BY 4.0** (as redistributed via the matchbench Hugging Face mirror).

## Usage note

Vendored **for research and benchmarking use only**, to validate langres
blocking and entity-resolution on a textual-hard, literature-comparable
benchmark, and to supply Wave D's derived seen/unseen F1-drop slice. No ownership
is claimed over the data; all rights remain with the original authors. If you
redistribute, preserve this attribution.
