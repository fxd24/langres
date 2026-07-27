# langres

**Composable, optimizable entity resolution for Python.**

langres resolves records that refer to the same real-world entity and makes ER
research repeatable. **Resources** (`Embedder`, `Reranker`, `LLM`) equip ordered
**operations** (`Retrieve`, `Rerank`, `Select`, `Generate`, `Parse`, cluster);
named **recipes** provide readable complete topologies. `EvaluationProtocol`
and `ExperimentReport` keep benchmark, split, seed, measurement, and
infrastructure cohorts explicit.

Install from PyPI:

```bash
pip install langres
```

The core install needs no extras, no API key and no network — that is the
whole offline quickstart:

```python
from langres.architectures import FuzzyString

records = [
    {"id": "1", "name": "Acme Corporation", "city": "New York"},
    {"id": "2", "name": "Acme Corp", "city": "New York"},
    {"id": "3", "name": "Unrelated Bakery", "city": "Miami"},
]

print(FuzzyString(threshold=0.6).dedupe(records))
# DedupeResult([{'1', '2'}], architecture='FuzzyString', backbone=None,
#              score_type='heuristic', threshold=0.6)
```

From a git checkout the same walkthrough is a runnable script:

<!-- docs-gate: requires-repo -->
```bash
git clone https://github.com/fxd24/langres.git && cd langres && uv sync
uv run python examples/quickstart_models.py  # FuzzyString dedupe, real clusters, $0
```

!!! note "The example scripts need a git checkout"

    `examples/` is not part of the published wheel, so every `examples/...`
    path in these docs assumes a clone. The research runner
    (`examples/research/first_experiment.py`) additionally needs the
    `[semantic]` extra, because research `Retrieve` builds a Qdrant index. It
    executes the real runner over a bundled fixture with a deterministic
    **fake** embedder — proof that the contracts compose, not a quality
    result.

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
