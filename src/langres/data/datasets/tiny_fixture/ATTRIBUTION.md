# Tiny fixture benchmark (fully synthetic)

These five CSVs are a **fully synthetic**, deliberately tiny cross-source
entity-resolution fixture. They exist only to validate the generic DeepMatcher
loader factory (`langres.data._deepmatcher_loader`) and the benchmark registry
end-to-end, and to give CI a fast, offline smoke test — they are **not** a real
benchmark and carry no evaluation meaning.

## Files

| File | Rows | Description |
|---|---|---|
| `tableA.csv` | 6 | Source-A product records (`id,name,description`) |
| `tableB.csv` | 6 | Source-B product records (same columns) |
| `train.csv` | 4 | Labeled pairs (`ltable_id,rtable_id,label`) |
| `valid.csv` | 2 | Labeled pairs |
| `test.csv` | 3 | Labeled pairs |

`ltable_id` references a row in `tableA` (prefixed `a` in the corpus);
`rtable_id` references a row in `tableB` (prefixed `b`). `label == 1` is a match.
The three cross-source matches are `a1↔b1` (iPhone 12), `a3↔b2` (Sony WH-1000XM4),
and `a4↔b3` (Dell XPS 13). `b5` (iPhone 12 Mini) is a deliberate hard negative.

## Attribution & licensing

Authored from scratch for the langres test suite — invented product names, no
real records, no PII, no third-party data. **No license concern**: this fixture
is part of langres and shares its license.
