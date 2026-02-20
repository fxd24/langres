"""Progressive Pipeline Building: Exploratory Entity Resolution.

This example demonstrates how to use component inspection methods to build
an entity resolution pipeline iteratively WITHOUT requiring ground truth labels.

**Workflow Demonstrated**:
1. Inspect blocking candidates → tune k_neighbors
2. Inspect scoring distribution → tune scoring approach
3. Inspect clustering results → tune threshold
4. Iterate based on recommendations
5. Run full pipeline with calibrated parameters

**Key Insight**: Inspection enables parameter discovery before expensive labeling.

This is the recommended workflow for:
- Initial parameter selection
- Understanding pipeline behavior
- Debugging quality issues
- Exploratory data analysis

For optimization with ground truth labels, see:
- examples/deduplication_with_blocker_optimization.py
"""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.modules.rapidfuzz import RapidfuzzModule
from langres.core.vector_index import FAISSIndex

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMA DEFINITION
# ============================================================================


class CompanySchema(BaseModel):
    """Company entity schema."""

    id: str = Field(description="Unique company identifier")
    name: str = Field(description="Company name")
    address: str | None = Field(default=None, description="Company address")
    website: str | None = Field(default=None, description="Company website")


# ============================================================================
# DATA LOADING
# ============================================================================


def load_companies(data_path: Path) -> list[CompanySchema]:
    """Load company records from JSON file."""
    with open(data_path) as f:
        data = json.load(f)
    return [CompanySchema(**record) for record in data]


# ============================================================================
# SCHEMA FACTORY AND TEXT EXTRACTION
# ============================================================================


def company_factory(record: dict[str, str]) -> CompanySchema:
    """Convert raw dict to CompanySchema."""
    if isinstance(record, CompanySchema):
        return record
    return CompanySchema(**record)


def company_text_extractor(company: CompanySchema) -> str:
    """Extract text representation for embedding."""
    parts = [company.name]
    if company.address:
        parts.append(company.address)
    if company.website:
        parts.append(company.website)
    return " | ".join(parts)


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


def main() -> None:
    """Run progressive pipeline building workflow."""
    # Load data
    data_dir = Path(__file__).parent / "data"
    entities = load_companies(data_dir / "companies.json")
    logger.info("Loaded %d companies for deduplication\n", len(entities))

    # Convert to dicts for blocker
    entity_dicts = [entity.model_dump() for entity in entities]

    # ========================================================================
    # STEP 1: EXPLORE BLOCKING - Initial Parameters
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: EXPLORE BLOCKING - Initial Parameters")
    logger.info("=" * 80)
    logger.info("")

    # Initialize with conservative k=3 (low recall risk)
    embedding_provider = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    vector_index = FAISSIndex(metric="L2")

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=company_text_extractor,
        embedding_provider=embedding_provider,
        vector_index=vector_index,
        k_neighbors=3,  # Start conservative
    )

    candidates = list(blocker.stream(entity_dicts))

    # INSPECT CANDIDATES
    logger.info("Running candidate inspection (k=3)...")
    report = blocker.inspect_candidates(candidates, entities, sample_size=5)
    logger.info("\n%s", report.to_markdown())
    logger.info("")

    # ========================================================================
    # STEP 2: TUNE BLOCKING - Increase k based on recommendation
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 2: TUNE BLOCKING - Increase k based on recommendation")
    logger.info("=" * 80)
    logger.info("")

    # Re-run with increased k
    blocker_tuned = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=company_text_extractor,
        embedding_provider=embedding_provider,
        vector_index=FAISSIndex(metric="L2"),
        k_neighbors=5,  # Increased from 3
    )

    candidates_tuned = list(blocker_tuned.stream(entity_dicts))
    logger.info("Running candidate inspection (k=5)...")
    report_tuned = blocker_tuned.inspect_candidates(candidates_tuned, entities, sample_size=5)
    logger.info("\n%s", report_tuned.to_markdown())
    logger.info("")

    # Compare before/after
    logger.info("COMPARISON:")
    logger.info(
        "  k=3: %d candidates (avg %.1f per entity)",
        len(candidates),
        report.avg_candidates_per_entity,
    )
    logger.info(
        "  k=5: %d candidates (avg %.1f per entity)",
        len(candidates_tuned),
        report_tuned.avg_candidates_per_entity,
    )
    logger.info(
        "  → Improvement: +%d candidates (+%.0f%%)\n",
        len(candidates_tuned) - len(candidates),
        ((len(candidates_tuned) - len(candidates)) / len(candidates)) * 100,
    )

    # ========================================================================
    # STEP 3: EXPLORE SCORING - Understand score distribution
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 3: EXPLORE SCORING - Understand score distribution")
    logger.info("=" * 80)
    logger.info("")

    # For POC, use RapidfuzzModule (no LLM costs)
    module: RapidfuzzModule[CompanySchema] = RapidfuzzModule(
        field_extractors={
            "name": (lambda x: x.name, 0.7),
            "address": (lambda x: x.address or "", 0.2),
            "website": (lambda x: x.website or "", 0.1),
        },
        threshold=0.0,  # Get all scores
    )
    judgements = list(module.forward(iter(candidates_tuned)))

    # INSPECT SCORES
    logger.info("Running score inspection...")
    score_report = module.inspect_scores(judgements, sample_size=5)
    logger.info("\n%s", score_report.to_markdown())
    logger.info("")

    # ========================================================================
    # STEP 4: EXPLORE CLUSTERING - Initial threshold
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 4: EXPLORE CLUSTERING - Initial threshold")
    logger.info("=" * 80)
    logger.info("")

    # Use reasonable default threshold
    initial_threshold = 0.5
    clusterer = Clusterer(threshold=initial_threshold)
    clusters = clusterer.cluster(judgements)

    # INSPECT CLUSTERS
    logger.info("Running cluster inspection (threshold=%.1f)...", initial_threshold)
    cluster_report = clusterer.inspect_clusters(clusters, entities, sample_size=3)
    logger.info("\n%s", cluster_report.to_markdown())
    logger.info("")

    # ========================================================================
    # STEP 5: TUNE CLUSTERING - Adjust threshold based on recommendations
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 5: TUNE CLUSTERING - Adjust threshold based on recommendations")
    logger.info("=" * 80)
    logger.info("")

    # Adjust threshold based on singleton rate
    # If high singleton rate, lower threshold
    # If low singleton rate, raise threshold
    tuned_threshold = 0.4  # Based on recommendations

    clusterer_tuned = Clusterer(threshold=tuned_threshold)
    clusters_tuned = clusterer_tuned.cluster(judgements)

    logger.info("Running cluster inspection (threshold=%.1f)...", tuned_threshold)
    cluster_report_tuned = clusterer_tuned.inspect_clusters(clusters_tuned, entities, sample_size=3)
    logger.info("\n%s", cluster_report_tuned.to_markdown())
    logger.info("")

    # Compare before/after
    logger.info("COMPARISON:")
    logger.info(
        "  threshold=%.1f: %d clusters (%.1f%% singletons)",
        initial_threshold,
        cluster_report.total_clusters,
        cluster_report.singleton_rate,
    )
    logger.info(
        "  threshold=%.1f: %d clusters (%.1f%% singletons)",
        tuned_threshold,
        cluster_report_tuned.total_clusters,
        cluster_report_tuned.singleton_rate,
    )
    logger.info("")

    # ========================================================================
    # SUMMARY: Calibrated Parameters
    # ========================================================================
    logger.info("=" * 80)
    logger.info("SUMMARY: Calibrated Parameters")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Through progressive inspection, we discovered:")
    logger.info("  • Blocking: k_neighbors=5 (from initial k=3)")
    logger.info("  • Scoring: RapidfuzzModule with default similarity")
    logger.info(
        "  • Clustering: threshold=%.1f (from initial %.1f)",
        tuned_threshold,
        initial_threshold,
    )
    logger.info("")
    logger.info("These parameters were calibrated WITHOUT ground truth labels!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Label a small sample (~50-100 entities) for validation")
    logger.info("  2. Use PipelineDebugger.analyze_*() to measure quality (precision/recall)")
    logger.info("  3. If quality is good, proceed to full optimization with BlockerOptimizer")
    logger.info("  4. See: examples/deduplication_with_blocker_optimization.py")
    logger.info("")


if __name__ == "__main__":
    main()
