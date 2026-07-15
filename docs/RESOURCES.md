# Resources

This document catalogs external resources that inform langres's design, architecture, and entity resolution approaches. Each resource is annotated with what langres learns or adopts from it.

---

## Framework Architecture & Design Patterns

### PyTorch

- **URL**: <https://pytorch.org/>
- **What we learn**: Modular, composable primitives that can be combined flexibly. The `nn.Matcher` abstraction for stateful components with `forward()` methods. Clear separation between high-level (`torch.nn`) and low-level (`torch.Tensor`) APIs.
- **Applied in langres**: Our `Matcher` abstraction mirrors PyTorch's design. The two-layer API (tasks vs. core) follows PyTorch's philosophy of simple primitives + convenient wrappers.

### PyTorch Lightning

- **URL**: <https://lightning.ai/>
- **What we learn**: Separation of research/model code from engineering concerns (logging, checkpointing, distributed training). Structured training loops with clear hooks. The "organized PyTorch" philosophy.
- **Applied in langres**: Inspires our separation of matching logic (Flows) from optimization/execution concerns (Tasks, Optimizer). The `.compile()` method concept for optimization.

### FastAPI

- **URL**: <https://fastapi.tiangolo.com/>
- **What we learn**: Pydantic-first design for validation and developer experience. Type hints as runtime behavior. Automatic validation, serialization, and documentation from type annotations.
- **Applied in langres**: Heavy use of Pydantic for all data models (`PairwiseJudgement`, `Candidate`, etc.). Type-driven API design. Validation at boundaries.

### SQLModel

- **URL**: <https://sqlmodel.tiangolo.com/>
- **What we learn**: Single source of truth for data models (Pydantic + SQLAlchemy). Reduces boilerplate by unifying validation, ORM, and serialization. Type-driven database schemas.
- **Applied in langres**: Philosophy of eliminating redundant data definitions. Potential future use for persisting resolution results, audit logs, and provenance tracking.

### DSPy

- **URL**: <https://github.com/stanfordnlp/dspy>
- **What we learn**: Declarative prompt programming with automatic optimization. Separation of program logic from prompt engineering. Compiler-based approach to LLM systems.
- **Applied in langres**: Core dependency for LLM-based matchers (Approach 3). The `Optimizer` component will use DSPy for prompt optimization. Inspires our "compile-then-run" workflow.

---

## Entity Resolution Systems & Libraries

### Ditto

- **URL**: <https://github.com/megagonlabs/ditto>
- **Paper**: "Deep Entity Matching with Pre-Trained Language Models" (VLDB 2021)
- **What we learn**: State-of-the-art deep learning for entity matching. Data augmentation techniques (MixDA). Transfer learning with pre-trained LMs. Benchmarking methodology on standard datasets (Magellan, WDC).
- **Applied in langres**: Reference for semantic matching approaches (Approach 2). Benchmark comparison target. Data augmentation ideas for training learnable matchers.

### Dedupe.io

- **URL**: <https://github.com/dedupeio/dedupe>
- **What we learn**: Active learning for training custom matchers. User-in-the-loop labeling workflows. Blocking with learned similarity functions. Mature Python library design.
- **Applied in langres**: Reference for active learning integration (future). Blocking strategies. User experience patterns for configuration and tuning.

### Splink

- **URL**: <https://github.com/moj-analytical-services/splink>
- **What we learn**: Probabilistic linkage at scale (Fellegi-Sunter model). SQL-based execution on big data backends (Spark, DuckDB). Clear API for blocking rules and match weights.
- **Applied in langres**: Inspiration for scalable blocking strategies. Reference for match score calibration. Comparison target for benchmarks.

### Zingg

- **URL**: <https://github.com/zinggAI/zingg>
- **What we learn**: ML-based entity resolution on big data platforms. Support for Spark, Snowflake, Databricks. Active learning workflows. Training pipeline design.
- **Applied in langres**: Reference for distributed execution patterns (future). Active learning UX. Integration with data platforms.

### pyJedAI

- **URL**: <https://github.com/AI-team-UoA/pyJedAI>
- **What we learn**: Only open-source ER library exploiting latest deep learning and NLP breakthroughs for both blocking and matching. High time efficiency, scalability, and effectiveness. Modern Python implementation of JedAI framework.
- **Applied in langres**: Reference for DL-based blocking approaches (Approach 2). Comparison target for benchmarks. Integration patterns for transformer-based matching.

### Awesome Entity Resolution

- **URL**: <https://github.com/OlivierBinette/Awesome-Entity-Resolution>
- **What we learn**: Comprehensive survey of ER tools, papers, datasets, and benchmarks. State of the field overview.
- **Applied in langres**: Source for discovering related work, benchmark datasets, and research trends. Gap analysis vs. existing tools.

---

## Evaluation & Benchmarking

### pytrec_eval

- **URL**: <https://github.com/cvangysel/pytrec_eval>
- **What we learn**: Standard information retrieval metrics (precision, recall, MAP, NDCG). Efficient C++ implementations with Python bindings. Established evaluation conventions from TREC community.
- **Applied in langres**: Potential integration for candidate generation (blocking) evaluation. Standardized metric reporting. Benchmarking blocking recall vs. efficiency trade-offs.

### Entity Resolution Benchmarks

- **Magellan Datasets**: Standard ER benchmark collection (structured data)
- **WDC Product Matching**: Large-scale e-commerce entity matching
- **OAEI**: Ontology alignment evaluation (schema matching)
- **Applied in langres**: POC validation datasets. Performance comparison baselines. Regression testing.

---

## Research Papers

### Foundational Entity Resolution

1. **"A Theory of Record Linkage"** - Fellegi & Sunter (1969)
   - Seminal probabilistic model for record linkage
   - Match/non-match weight calculation
   - Foundation for modern ER systems

2. **"Swoosh: A Generic Approach to Entity Resolution"** - Benjelloun et al. (VLDB 2009)
   - Generic ER framework with merge functions
   - Theoretical properties (idempotence, commutativity)
   - Influences our canonicalization and clustering design

### Blocking & Candidate Generation

1. **"Blocking for Large-Scale Entity Resolution"** - Papadakis et al. (ACM CSUR 2020)
   - Comprehensive survey of blocking techniques
   - Schema-agnostic blocking methods
   - Performance/efficiency trade-offs

2. **"Meta-Blocking: Taking Entity Resolution to the Next Level"** - Papadakis et al. (IEEE TKDE 2014)
   - Pruning redundant candidate pairs
   - Graph-based blocking refinement
   - Inspires our `Blocker` optimization strategies

### Deep Learning for Entity Matching

1. **"Deep Entity Matching with Pre-Trained Language Models"** - Li et al. (VLDB 2021)
   - Ditto system architecture
   - Data augmentation for ER
   - Benchmark on standard datasets

2. **"Deep Learning for Entity Matching: A Design Space Exploration"** - Mudgal et al. (SIGMOD 2018)
   - Systematic comparison of DL architectures for ER
   - Embedding strategies, interaction patterns
   - Informs our semantic matching (Approach 2)

### Clustering & Correlation Clustering

1. **"Correlation Clustering"** - Bansal et al. (FOCS 2002)
   - Theoretical foundation for clustering with similarity/dissimilarity
   - Approximation algorithms
   - Basis for our `Clusterer` component

2. **"Adaptive Blocking: Learning to Scale Up Record Linkage"** - Bilenko et al. (ICDM 2006)
   - Learning-based blocking strategies
   - Adaptive candidate generation
   - Future direction for learnable blockers

### LLM-Based Entity Resolution

1. **"Can Foundation Models Wrangle Your Data?"** - Narayan et al. (VLDB 2023)
   - LLMs for data wrangling tasks including ER
   - Zero-shot vs. few-shot performance
   - Prompt engineering strategies

2. **"Large Language Models for Entity Matching"** - Peeters & Bizer (DI2KG 2023)
   - Systematic evaluation of GPT-3/4 for ER
   - Cost vs. accuracy trade-offs
   - Informs our LLM-based matcher design (Approach 3)

### Active Learning & Human-in-the-Loop

1. **"Active Learning for Entity Resolution"** - Sarawagi & Bhamidipaty (VLDB 2002)
   - Uncertainty sampling for ER
   - Query selection strategies
   - Future integration with langres Optimizer

### Evaluation & Metrics

1. **"Evaluating Entity Resolution Results"** - Menestrina et al. (VLDB 2010)
   - BCubed metrics for clustering evaluation
   - Comparison with pairwise F1
   - Why we use BCubed F1 for POC success criteria

---

## Cost-Aware ML Systems

1. **"FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"** - Chen et al. (2023)
    - LLM cascade strategies
    - Cost-performance trade-offs
    - Informs our multi-stage matcher design (cheap blockers → expensive LLM judge)

---

## Related Frameworks & Inspiration

### Sentence Transformers

- **URL**: <https://www.sbert.net/>
- **What we learn**: Efficient semantic similarity with embeddings. Pre-trained models for various domains. FAISS/Annoy integration for approximate nearest neighbors.
- **Applied in langres**: Core dependency for semantic blocking (Approach 2). Vector index integration for candidate generation.

### Optuna

- **URL**: <https://optuna.org/>
- **What we learn**: Automated hyperparameter optimization. Pruning unpromising trials. Distributed optimization support.
- **Applied in langres**: Core dependency for `Optimizer`. Hyperparameter tuning for matchers, blocking thresholds, clustering parameters.

### NetworkX

- **URL**: <https://networkx.org/>
- **What we learn**: Graph algorithms for clustering, connected components, community detection.
- **Applied in langres**: Core dependency for `Clusterer`. Correlation clustering implementations.

---

## Talks & Conference Presentations

### "Entity Resolution for the Best Outcomes on Your Data"

- **URL**: <https://www.youtube.com/watch?v=3TKGBYveTIA>
- **What we learn**: Practical insights on ER challenges in real-world data pipelines. Best practices for data quality, blocking strategies, and production deployment. Common pitfalls and how to avoid them.
- **Applied in langres**: Informs our design priorities for the high-level `tasks` API. Real-world use case validation. Guidance on what "production-ready" means for ER systems.

### "Let the LLM Write the Prompts: An Intro to DSPy in Compound AI Pipelines"

- **URL**: <https://www.youtube.com/watch?v=I9ZtkgYZnOw>
- **What we learn**: DSPy's approach to optimizing LLM programs in compound AI systems. How to build multi-stage pipelines with automatic prompt optimization. Integration patterns for DSPy in production systems.
- **Applied in langres**: Critical for Approach 3 (LLM-based matcher). Guidance on integrating DSPy's optimizer with our blocking + judging pipeline. Understanding DSPy's compilation model for our `.compile()` method.

### "Navigating RAG Optimization with an Evaluation Driven Compass"

- **URL**: <https://www.youtube.com/watch?v=DId2KP8Ykz4>
- **Speakers**: Atita Arora and Deanna Emery
- **What we learn**: Evaluation-driven optimization for complex AI pipelines (RAG). Systematic approaches to measuring and improving multi-stage systems. Balancing cost, latency, and quality in production deployments.
- **Applied in langres**: Philosophical alignment with our `.compile()` method - optimize based on gold data and metrics. Relevance to multi-stage pipelines (blocking → matching → clustering). Cost-aware optimization strategies.

---

## Community & Discourse

### Entity Resolution Community

- **SIGMOD/VLDB/ICDE Conferences**: Premier venues for ER research
- **Entity Resolution Workshop (ER)**: Annual workshop at KDD
- **Data Integration & Matching (VLDB Workshop)**: Practical ER systems

### Relevant Python Data Science Ecosystem

- **scikit-learn**: API design patterns for estimators, transformers, pipelines
- **Hugging Face**: Model zoo, evaluation metrics, dataset management
- **Weights & Biases**: Experiment tracking, hyperparameter optimization visualization

---

## Papers & Research to Review

**Status**: These recent papers (2024-2025) have been identified as potentially valuable but require detailed evaluation before integration into the main resource list.

### Recent LLM-Based Entity Resolution (2024-2025)

1. **"Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching"** (COLING 2025)
   - **URL**: <https://arxiv.org/abs/2405.16884>
   - **Why relevant**: Most recent comprehensive study of LLM-based ER strategies. Introduces ComEM framework for multi-strategy composition. Evaluated on 8 datasets with 10 different LLMs.
   - **Potential value**: Could inform Approach 3 (LLM hybrid) design decisions. May offer insights on when to use matching vs. comparing vs. selecting strategies.

2. **"Fine-tuning Large Language Models for Entity Matching"** (September 2024, updated May 2025)
   - **URL**: <https://arxiv.org/abs/2409.08185>
   - **Why relevant**: Addresses fine-tuning LLMs specifically for ER. Analyzes training example selection and generation using LLMs.
   - **Potential value**: Critical for understanding LLM optimization trade-offs. Mixed results for larger models suggest careful consideration needed.

3. **"In-context Clustering-based Entity Resolution with Large Language Models"** (June 2025)
   - **URL**: <https://arxiv.org/abs/2506.02509>
   - **Why relevant**: Novel approach that packs multiple records for direct clustering rather than pairwise comparison. Claims to be more efficient and scalable.
   - **Potential value**: Could fundamentally change how we think about clustering in Approach 3. Generalizes pairwise approaches.

4. **"A Deep Dive Into Cross-Dataset Entity Matching"** (March 2025)
   - **URL**: <https://openproceedings.org/2025/conf/edbt/paper-224.pdf>
   - **Why relevant**: Extensive experimental study (425+ GPU hours, $290+ OpenAI API costs). Focuses on cross-dataset context with computational cost analysis.
   - **Potential value**: Cost-aware optimization insights. Real-world production considerations for LLM-based ER.

### State-of-the-Art Blocking & Candidate Generation (2024)

1. **"SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines"** (ESWC 2024)
   - **URL**: <https://2024.eswc-conferences.org/wp-content/uploads/2024/04/146640116.pdf>
   - **Why relevant**: State-of-the-art blocking method. 50% smaller candidate sets, 1.5-4x faster pipelines with no F1 loss.
   - **Potential value**: Could significantly improve Blocker component performance. Supervised contrastive learning approach is cutting-edge.

2. **"Towards Universal Dense Blocking for Entity Resolution (UniBlocker)"** (April 2024)
   - **URL**: <https://arxiv.org/abs/2404.14831>
   - **Why relevant**: Pre-trained blocker using self-supervised contrastive learning. Domain-independent, outperforms previous unsupervised methods.
   - **Potential value**: Perfect fit for semantic blocking (Approach 2). Could be integrated as pre-built blocker component.

3. **"Pre-trained Language Models for Entity Blocking"** (NAACL 2024)
   - **URL**: <https://aclanthology.org/2024.naacl-long.483.pdf>
   - **Why relevant**: Generalization study on transformer-based models (BERT, RoBERTa) for blocking tasks.
   - **Potential value**: Guidance on which pre-trained models work best for blocking. Transfer learning insights.

4. **"GSM: Generalized Supervised Meta-blocking for Scalable Entity Resolution"** (2023)
   - **URL**: <https://www.sciencedirect.com/science/article/pii/S0306437923001436>
   - **Why relevant**: Combines multiple scores per comparison into feature vectors for binary classification. Flexible candidate generation.
   - **Potential value**: Meta-blocking could be a second-stage refinement for Blocker component.

### Related Topics to Explore

1. **Geospatial Data Conflation**
   - **Concepts**: Entity resolution for geospatial data (matching roads, buildings, POIs across map datasets like OpenStreetMap)
   - **Key tools**: Hootenanny, JOSM Conflation Plugin, RoadMatcher
   - **What we might learn**: Spatial similarity metrics (beyond string/semantic), geometric matching algorithms, graph-based matching for road networks, handling structured spatial data.
   - **Relevance to langres**: Techniques may generalize to other structured entity types. Graph-based matching could inform clustering approaches.

---

## Maintenance Notes

**When adding new resources:**

1. Categorize appropriately (architecture, ER system, paper, etc.)
2. Include URL and brief "What we learn" annotation
3. Explain how it influences langres design or roadmap
4. Keep focused on *actionable* learnings, not exhaustive bibliography

**Scope**: This is a *curated* list of resources that directly inform langres development. For comprehensive ER bibliographies, see [Awesome Entity Resolution](https://github.com/OlivierBinette/Awesome-Entity-Resolution).
