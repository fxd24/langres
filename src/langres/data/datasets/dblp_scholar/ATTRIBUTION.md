# DBLP-Scholar bibliographic entity-resolution benchmark

These five CSVs are the **DBLP-Scholar** citation-matching benchmark, a standard
entity-resolution (record-linkage) dataset. Two bibliographic sources — **DBLP**
(a clean computer-science bibliography) and **Google Scholar** (a large, noisy
web-scraped index) — each list publications; the task is to find the
cross-source pairs that refer to the same paper. Unlike the small, near-saturated
product benchmarks, DBLP-Scholar is **large and noisy**: Scholar carries tens of
thousands of records with truncated/garbled titles and frequently missing venues
or years, and a single DBLP paper can match several Scholar entries (and vice
versa), so it is genuinely **many-to-many**.

## Files

| File | Rows | Description |
|---|---|---|
| `tableA.csv` | 2616 | DBLP publication records (`id,title,authors,venue,year`) |
| `tableB.csv` | 64263 | Google Scholar publication records (same columns) |
| `train.csv` | 17223 | Labeled pairs (`ltable_id,rtable_id,label`) — fixed literature split |
| `valid.csv` | 5742 | Labeled pairs — fixed literature split |
| `test.csv` | 5742 | Labeled pairs — fixed literature split |

`ltable_id` references a row in `tableA` (DBLP, prefixed `a` in the corpus);
`rtable_id` references a row in `tableB` (Scholar, prefixed `b`). `label == 1` is
a match, `label == 0` a non-match. The three splits are the **fixed literature
pair splits** (DeepMatcher/Magellan), so results are directly comparable to
published numbers. Matches are all cross-source (DBLP <-> Scholar).

Positive (`label == 1`) pairs: **3207 (train) + 1070 (valid) + 1070 (test) =
5347** cross-source match pairs (no pair repeats across splits).

Because the benchmark is many-to-many, the loader's closed-world gold partition
(connected components of the match graph; largest component = 37 records) has
**2351 match clusters** over the **66879-record** corpus, whose within-cluster
transitive closure yields **13763 gold pairs** — this is the count the loader
contract test pins (`Benchmark.load()` re-derives gold pairs from the clusters,
including the intra-source pairs the closure introduces).

## Attribution & provenance

Vendored from the **matchbench/DBLP-Scholar** dataset on the Hugging Face Hub,
which redistributes the benchmark for entity-resolution research:

- Source: <https://huggingface.co/datasets/matchbench/DBLP-Scholar> —
  `{tableA.csv, tableB.csv, train.csv, valid.csv, test.csv}`
- License: **CC-BY 4.0** (as redistributed via matchbench).

The DBLP-Scholar benchmark itself originates from the **Magellan / DeepMatcher**
entity-matching benchmark suite (the "Structured/DBLP-GoogleScholar" task),
descended from the original DBLP-Scholar dataset released by the AnHai Doan group
at UW-Madison:

- Mudgal et al., "Deep Learning for Entity Matching: A Design Space Exploration,"
  SIGMOD 2018.
- <https://github.com/anhaidgroup/deepmatcher>

## Usage note

Vendored **for research and benchmarking use only**, to validate langres blocking
and entity-resolution on a large, noisy, many-to-many, literature-comparable
bibliographic benchmark (Wave C, eval-readiness). No ownership is claimed over the
data; all rights remain with the original authors. If you redistribute, preserve
this attribution.
