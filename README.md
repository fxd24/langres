# langres

[![Status](https://img.shields.io/badge/status-early%20development-yellow.svg)](https://github.com/raisesquad/langres)
[![Tests](https://github.com/raisesquad/langres/actions/workflows/test.yml/badge.svg)](https://github.com/raisesquad/langres/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/raisesquad/langres/branch/main/graph/badge.svg)](https://codecov.io/gh/raisesquad/langres)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A composable, optimizable entity resolution framework with a two-layer API, intelligent blocking, and human-in-the-loop capabilities.

---

## ⚠️ Project Status

**This library is currently in early design/development phase.** The documentation below represents the planned architecture and vision for what langres will become. Core components are still being implemented.

**What exists now:**
- ✅ Project structure and architecture design
- ✅ Comprehensive documentation of planned features
- 🚧 Core implementation (in progress)

**Not yet available:**
- ❌ Installable package on PyPI
- ❌ Working code examples
- ❌ Pre-built Flows and Blockers

See the [project overview](docs/PROJECT_OVERVIEW.md) for the complete vision and roadmap.

---

## Vision

**langres** aims to be a Python-native, "batteries-included" library for building, optimizing, and deploying entity resolution (ER) pipelines. It will replace rigid, configuration-driven systems with a flexible, testable, and transparent framework that leverages modern data science tools.

langres will act as a "glue" library providing:

- A clear, type-safe API built on **Pydantic**
- A powerful optimization engine using **Optuna** and **DSPy**
- Seamless integration with tools like **PyTorch**, **sentence-transformers**, **rapidfuzz**, and **networkx**

### Philosophy: The Two-Layer API

langres is designed to be accessible to all skill levels without sacrificing power:

- **`langres.tasks`** (High-Level): For 80% of use cases. Pre-built task runners (`DeduplicationTask`, `EntityLinkingTask`) will abstract away the underlying components. Think **scikit-learn's Pipeline**.

- **`langres.core`** (Low-Level): For 20% of use cases. Composable "Lego bricks" (`Module`, `Blocker`, `Optimizer`) for building bespoke pipelines. Think **PyTorch's primitives**.

---

## Planned Features

### 🧩 Composable Architecture

- **Pre-built Flows**: Out-of-the-box matching logic for companies, products, and custom entities
- **Reusable Components**: Same "brain" (Flow) works across deduplication, linking, and record linkage tasks
- **Schema Mapping**: Blockers handle data normalization, keeping Flows clean and portable

### 🚀 Intelligent Blocking

- **High-Recall Candidate Generation**: Avoid N² comparisons with ANN search (via FAISS, HNSW)
- **Cascade Strategies**: Start with cheap string matching, fallback to semantic embeddings
- **Task-Specific Blockers**: `DedupeBlocker`, `LinkingBlocker`, `CascadeBlocker`

### 🎯 Auto-Optimization

- **Hyperparameter Tuning**: Optuna-powered search for optimal thresholds and weights
- **Prompt Optimization**: DSPy integration for LLM-based matchers
- **Cluster-Level Metrics**: Optimize for BCubed F1, V-Measure, not just pairwise accuracy
- **Two-Stage Process**: Separate model training (`finetune()`) from hyperparameter search (`compile()`)

### 👤 Human-in-the-Loop

- **Review Queue**: Built-in storage for uncertain matches (SQLite/file-based)
- **Active Learning**: Export verified labels to improve your model
- **Pre-built UI**: (Coming Soon) Standalone Streamlit app for labeling

### 🧪 Synthetic Data Generation

- **LLM-Powered**: Generate realistic training data with typos, synonyms, abbreviations
- **Pydantic-Driven**: Automatically create variations based on your schema

### 🏆 Master Data Creation

- **Canonicalization**: Merge clusters into golden records (V1.1)
- **Survivorship Rules**: Configurable per-field logic (most_recent, most_frequent, merge_unique)

---

## Installation

**Note: Package not yet published to PyPI**

When available, installation will be:

```bash
pip install langres
```

**Requirements:**

- Python >=3.12

---

## Planned Usage

The examples below show the intended API design. **These are not yet functional.**

### Deduplication (Use Case 1)

```python
from langres.tasks import DeduplicationTask
from langres.flows import CompanyFlow
from langres.blockers import DedupeBlocker
from langres.data import SyntheticGenerator

# 1. Set up components
flow = CompanyFlow()  # Pre-built company matching logic
blocker = DedupeBlocker()  # Simple single-schema blocker
task = DeduplicationTask(flow=flow, blocker=blocker)

# 2. Generate training data (or provide your own)
gold_data = SyntheticGenerator(Company).generate(5000)

# 3. Optimize the task (finds best thresholds)
task.compile(gold_data, metric="bcubed_f1")

# 4. Run on your data
clusters = task.run(all_companies)
```

### Entity Linking (Use Case 2)

```python
from langres.tasks import EntityLinkingTask
from langres.flows import CompanyFlow
from langres.blockers import LinkingBlocker

# 1. Reuse the same Flow, different Blocker
flow = CompanyFlow()  # Same brain!

# 2. Configure schema mapping
sfdc_map = {"sfdc_name": "name", "sfdc_addr": "address"}
internal_map = {"name": "name", "address": "address"}
blocker = LinkingBlocker(source_map=sfdc_map, target_map=internal_map)

# 3. Set up and optimize
task = EntityLinkingTask(flow=flow, blocker=blocker)
task.compile(linking_gold_data, metric="pairwise_f1")

# 4. Link source to target
matches = task.run(source_data=sfdc_records, target_data=internal_companies)
```

### Custom Flow (Low-Level API)

```python
from langres.core import Module, Optimizer, Clusterer
import rapidfuzz.fuzz
import torch.nn as nn

# 1. Define custom matching logic
class MyProductFlow(Module):
    def __init__(self):
        self.embed_sim = EmbedSim(model="e5-small")
        self.combiner = MyCombinerModel()  # Custom PyTorch model
        self.name_weight = 0.5  # Tunable hyperparameter

    def forward(self, candidates):
        for pair in candidates:
            # Custom feature extraction
            name_sim = rapidfuzz.fuzz.WRatio(pair.left.name, pair.right.name)
            desc_sim = self.embed_sim(pair.left.description, pair.right.description)

            # Learnable combination
            score = self.combiner([name_sim, desc_sim])

            yield PairwiseJudgement(
                left_id=pair.left.id,
                right_id=pair.right.id,
                score=score,
                score_type="calibrated_prob"
            )

# 2. Manually orchestrate optimization
flow = MyProductFlow()
optimizer = Optimizer(metric="bcubed_f1")

# Train PyTorch weights
trained_flow = optimizer.finetune(flow, gold_data, epochs=10)

# Tune hyperparameters (e.g., name_weight)
compiled_flow = optimizer.compile(trained_flow, gold_data)

# 3. Run the pipeline
judgements = compiled_flow.forward(blocker.stream(all_products))
clusters = Clusterer().cluster(judgements)
```

---

## Core Components

### The Five Pillars (`langres.core`)

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Blocker** | Candidate generation & schema normalization | ANN search, cascade strategies, schema mapping |
| **Module** (Flow) | Pairwise comparison logic (the "brain") | Classical (rapidfuzz), semantic (embeddings), learnable (PyTorch) |
| **Clusterer** | Entity formation from pairs | Transitive closure, hierarchical clustering, cannot-link constraints |
| **Optimizer** | Hyperparameter & prompt tuning | Optuna for HPs, DSPy for prompts, BCubed F1 optimization |
| **Canonicalizer** | Master record creation (V1.1) | Survivorship rules, field-level merge strategies |

---

## Use Cases

langres is designed for **batch, attribute-based resolution**. Here's what it supports:

### ✅ Supported (V1 Core)

| Use Case | Task | Description |
|----------|------|-------------|
| **1. Deduplication** | `DeduplicationTask` | Find duplicates within a single dataset |
| **2. Entity Linking** | `EntityLinkingTask` | Link source records to authoritative target |
| **10. Fuzzy FK Resolution** | `EntityLinkingTask` | Resolve dirty foreign keys to clean primary keys |
| **9. Negative Constraints** | `Clusterer(constraints=...)` | Enforce cannot-link rules during clustering |

### 🚧 Planned (V1.1 Extension)

- **Use Case 3: Record Linkage** - Multi-source symmetric resolution
- **Use Case 4: Master Data Creation** - `Canonicalizer` with survivorship rules

### ⚠️ Out of Scope

- **Use Case 5: Streaming Resolution** - Use langres to train the Flow, deploy it in your streaming app (Flink, Kafka Streams)
- **Use Case 6: Temporal Evolution** - Requires temporal graph database (entity splits/mergers)
- **Use Case 7: Collective (Graph) Resolution** - Requires stateful, graph-native inference

See [docs/USE_CASES.md](docs/USE_CASES.md) for the complete taxonomy.

---

## Documentation

- [Project Overview](docs/PROJECT_OVERVIEW.md) - Philosophy and architecture
- [Technical Overview](docs/TECHNICAL_OVERVIEW.md) - API reference and data contracts
- [Use Cases](docs/USE_CASES.md) - Formal taxonomy and roadmap
- [Dependencies](docs/DEPENDENCIES.md) - Supply-chain security policy and dependency management
- [Examples](examples/) - Sample scripts and usage patterns

---

## Why langres?

### vs. Configuration-Driven Tools (Dedupe.io, AWS Entity Resolution)

- **Code-First**: Define logic in Python, not YAML
- **Testable**: Unit test your Flows like any other Python class
- **Transparent**: Full control over the matching logic

### vs. Black-Box SaaS (Tamr, Senzing)

- **Open Source**: No vendor lock-in
- **Cost-Aware**: Run optimization with budget constraints
- **Portable**: Export your compiled Flow to any environment

### vs. Research Libraries (py_stringmatching, recordlinkage)

- **Production-Ready**: Pydantic validation, full observability, HITL workflows
- **Optimizes for the Right Metric**: BCubed F1, not just pairwise accuracy
- **Modern Stack**: PyTorch, sentence-transformers, Optuna, DSPy

---

## Design Principles

- **Pydantic-First**: Fail-fast validation, IDE autocomplete, powers SyntheticGenerator
- **Full Observability**: Every `PairwiseJudgement` will carry provenance, reasoning, and score type
- **Cost & Safety**: PII redaction hooks, budget-aware optimization, cost-aware cascades

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

TBD

---

## Acknowledgments

Planned integrations with:

- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [DSPy](https://github.com/stanfordnlp/dspy) - Prompt optimization
- [sentence-transformers](https://www.sbert.net/) - Semantic embeddings
- [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) - String similarity
- [networkx](https://networkx.org/) - Graph clustering
- [PyTorch](https://pytorch.org/) - Deep learning
