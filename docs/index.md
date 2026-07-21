# langres

**Composable, optimizable entity resolution for Python.**

langres resolves records that refer to the same real-world entity and makes ER
research repeatable. **Resources** (`Embedder`, `Reranker`, `LLM`) equip ordered
**operations** (`Retrieve`, `Rerank`, `Select`, `Generate`, `Parse`, cluster);
named **recipes** provide readable complete topologies. `EvaluationProtocol`
and `ExperimentReport` keep benchmark, split, seed, measurement, and
infrastructure cohorts explicit.

```bash
uv run python examples/research/first_experiment.py  # real runner, local fixture, $0
```

Install from PyPI:

```bash
pip install langres
```

## Where to go next

- **[Getting Started](GETTING_STARTED.md)** — offline experiment, four recipes,
  matrix expansion, then the review → harvest → calibrate loop.
- **[Tutorial: Your Own CSV](TUTORIAL_YOUR_OWN_CSV.md)** — end-to-end walkthrough
  on your own data.
- **[Technical Overview](TECHNICAL_OVERVIEW.md)** — architecture, data contracts,
  and the component model.
- **[Benchmarks](BENCHMARKS.md)** — the benchmark portfolio and how to score your
  own data.
- **[Experiments](EXPERIMENTS.md)** — protocols, matrices, cohorts, reports,
  repricing, Trackio, and guarded paid proof.
- **[Reproducibility](REPRODUCIBILITY.md)** — identities, clean/dirty claims,
  local/Trackio/Hub handoff, privacy.
- **[API Reference](reference/index.md)** — the public surface, generated from
  docstrings.

## Project status

langres is a 0.x beta, [released on PyPI](https://pypi.org/project/langres/)
under Apache-2.0. The named architectures / `ERModel` (`Resolver`) / `core`
contracts are stable enough to build on, but expect breaking changes between
0.x releases — see the [Roadmap](ROADMAP.md) and [Changelog](CHANGELOG.md).
