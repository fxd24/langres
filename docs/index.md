# langres

**Composable, optimizable entity resolution for Python.**

langres resolves records that refer to the same real-world entity — deduplicating
one dataset or linking two — through a layered API: named **architectures**
(`FuzzyString`, `VectorLLMCascade`) — whole ER pipelines you construct — over
the declarative **`ERModel`** (aliased as `Resolver`) over low-level
**`langres.core`** primitives — blockers (pick candidate pairs), matchers
(score whether a pair matches), clusterers (group the matches) — that you can
swap, tune, and evaluate independently.

```python
from langres.architectures import FuzzyString

clusters = FuzzyString().dedupe(records)  # $0, offline, no key — you named the model
```

Install from PyPI:

```bash
pip install langres
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

langres is a 0.x beta, [released on PyPI](https://pypi.org/project/langres/)
under Apache-2.0. The named architectures / `ERModel` (`Resolver`) / `core`
contracts are stable enough to build on, but expect breaking changes between
0.x releases — see the [Roadmap](ROADMAP.md) and [Changelog](CHANGELOG.md).
