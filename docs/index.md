# langres

**Composable, optimizable entity resolution for Python.**

langres resolves records that refer to the same real-world entity — deduplicating
one dataset or linking two — through a layered API: two user-facing **verbs**
(`langres.link` / `langres.dedupe`) over a declarative **`Resolver`** over
low-level **`langres.core`** primitives — blockers (pick candidate pairs),
judges (score whether a pair matches), clusterers (group the matches) — that
you can swap, tune, and evaluate independently.

```python
import langres

clusters = langres.dedupe(records)  # judge="auto" picks the best available judge
```

Install from source (PyPI release pending):

```bash
pip install "langres @ git+https://github.com/fxd24/langres"
```

## Where to go next

- **[Getting Started](GETTING_STARTED.md)** — install, first dedupe, and the
  review → harvest → calibrate improvement loop.
- **[Tutorial: Your Own CSV](TUTORIAL_YOUR_OWN_CSV.md)** — end-to-end walkthrough
  on your own data.
- **[Technical Overview](TECHNICAL_OVERVIEW.md)** — architecture, data contracts,
  and the component model.
- **[Benchmarks](BENCHMARKS.md)** — the benchmark portfolio and how to score your
  own data.
- **[Experiments](EXPERIMENTS.md)** — the experimentation DX: races, judged-once
  evaluation, threshold calibration.
- **[API Reference](reference/index.md)** — the public surface, generated from
  docstrings.

## Project status

langres is in early development (pre-alpha). The verbs / `Resolver` / `core`
contracts are stable enough to build on, but expect breaking changes between
releases — see the [Roadmap](ROADMAP.md) and [Changelog](CHANGELOG.md).
