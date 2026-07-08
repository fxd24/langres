"""DBLP-Scholar bibliographic entity-resolution benchmark, via the loader factory (Wave C).

A ~30-line dataset module that adds the **DBLP-Scholar** citation-matching
benchmark through :func:`~langres.data._deepmatcher_loader.make_deepmatcher_benchmark`:
a dataset-namespaced schema (:class:`DblpScholarSchema`) plus one factory call
yields the ``load`` / ``load_pair_splits`` / :class:`DblpScholarBenchmark` triple.

Two bibliographic sources — DBLP (``tableA``) and Google Scholar (``tableB``) —
each list publications; the task is *linkage*: find the cross-source pairs that
refer to the same paper. Unlike the product benchmarks, Scholar is large and
noisy (~64k records, truncated/garbled titles, missing venues/years), and a DBLP
paper can match several Scholar entries, so the gold clusters are the connected
components of the match graph (many-to-many, clusters can exceed two records).

Blocking constants below are **measured** (a vector-blocking Pair-Completeness
sweep over the full corpus), not guessed — see the pinned sweep in the constant
docstrings. The heavy ``[semantic]`` stack stays lazy inside the factory's
``build_blocker`` (import budget: loading + splitting pull no faiss).
"""

from typing import Literal

from pydantic import BaseModel, computed_field

from langres.data._deepmatcher_loader import SourceTable, make_deepmatcher_benchmark

__all__ = [
    "DBLP_SCHOLAR_ACHIEVED_PC",
    "DBLP_SCHOLAR_BLOCKING_K",
    "DBLP_SCHOLAR_GATE_MET",
    "DBLP_SCHOLAR_RECALL_GATE",
    "DBLP_SCHOLAR_THRESHOLD_GRID",
    "DblpScholarBenchmark",
    "DblpScholarSchema",
    "load_dblp_scholar",
    "load_dblp_scholar_pair_splits",
]

_DATASET_PACKAGE = "langres.data.datasets.dblp_scholar"

DblpScholarSource = Literal["a", "b"]

# Pinned blocking k for the cross-source DBLP<->Scholar matches, measured with
# VectorBlocker over SentenceTransformer("all-MiniLM-L6-v2") cosine similarity on
# ``embed_text`` (title + authors + venue + year), over the full 66879-record
# corpus. Measured cross-source Pair-Completeness sweep (candidate_recall of the
# cross-source candidates against the closed-world gold clusters; reproduce via
# ``_benchmark_utils.sweep_blocking_k`` — the tmp/ script is noted in the
# ATTRIBUTION):
#   k= 5 -> 0.3727
#   k=10 -> 0.3863
#   k=20 -> 0.3911
#   k=30 -> 0.3926
#   k=50 -> 0.3945
#
# READ THIS BEFORE CONCLUDING "BLOCKING IS BAD": the low PC is a MANY-TO-MANY
# CLOSURE ARTIFACT, not a blocking failure. DBLP-Scholar is heavily many-to-many
# (2351 match clusters, largest = 37 records), so the closed-world partition's
# within-cluster gold pairs (13763 total, what ``evaluate_blocking`` scores recall
# against) are only 5473 cross-source (39.77%) + 8290 intra-source (60.23%,
# DBLP-DBLP / Scholar-Scholar pairs implied by the transitive closure). Blocking
# candidates are filtered to CROSS-source (this is a linkage task), so the 8290
# intra-source gold pairs are structurally un-recallable — capping PC at 0.3977
# (= 5473/13763) no matter how good blocking is. Achieved PC 0.3945 is therefore
# 0.3945/0.3977 = 0.9921 of the *achievable* ceiling, i.e. vector blocking
# actually surfaces ~99% of the true cross-source gold matches (true cross-source
# recall by k: 0.937/0.971/0.983/0.987/0.992). k=20 already gives ~0.983 true
# recall, so a future tuner can drop k to shrink the candidate set.
#
# Following the spec + ``amazon_google`` methodology, we pin the LITERAL
# sweep_blocking_k output: no k reaches the 0.90 PC gate (all plateau ~0.39, an
# artifact ceiling of 0.3977), so ``pick_blocking_k`` returns the best-PC k=50 and
# ``GATE_MET`` is honestly False. The "miss" is a metric-vs-many-to-many artifact,
# NOT a real recall shortfall (contrast amazon_google, whose 0.84 IS a genuine
# shortfall). ``DBLP_SCHOLAR_ACHIEVED_PC`` / ``DBLP_SCHOLAR_GATE_MET`` record the
# faithful, reproducible numbers.
DBLP_SCHOLAR_BLOCKING_K = 50

#: Pair-Completeness gate the blocking k-sweep aims to clear (mirrors Amazon-Google
#: / Abt-Buy at 0.90). NOT met here — but the miss is a many-to-many closure
#: artifact (see the sweep note above), not a blocking failure: true cross-source
#: recall is ~0.99; the PC metric is capped at 0.3977 by intra-source closure pairs.
DBLP_SCHOLAR_RECALL_GATE = 0.90

#: Cross-source Pair-Completeness achieved at :data:`DBLP_SCHOLAR_BLOCKING_K`,
#: recorded from the measured sweep so callers report realised blocking recall
#: without re-running embeddings. Deflated by the closure artifact (ceiling
#: 0.3977); the true cross-source blocking recall is ~0.9921 (see the note above).
DBLP_SCHOLAR_ACHIEVED_PC = 0.3945

#: Whether :data:`DBLP_SCHOLAR_ACHIEVED_PC` clears :data:`DBLP_SCHOLAR_RECALL_GATE`.
#: False — honestly recorded (mirrors ``amazon_google``): the literal PC metric
#: (0.3945) falls short of 0.90, here for a benign many-to-many reason, not hidden.
DBLP_SCHOLAR_GATE_MET = DBLP_SCHOLAR_ACHIEVED_PC >= DBLP_SCHOLAR_RECALL_GATE

#: Clusterer thresholds swept when racing methods (mirrors the other adapters).
DBLP_SCHOLAR_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class DblpScholarSchema(BaseModel):
    """A single publication record from the DBLP-Scholar benchmark.

    ``id`` is globally unique across both sources (the loader prefixes the raw
    source id with ``a``/``b``). ``embed_text`` is the serializable blocking text
    used by the :class:`VectorBlocker` (referenced as ``text_field``).

    Attributes:
        id: Globally-unique record id (e.g. ``"a12"`` / ``"b345"``).
        title: Publication title (always present).
        authors: Author list as a raw string, if present.
        venue: Publication venue, if present.
        year: Publication year as a raw string, if present.
        source: Originating table (``"a"`` for DBLP ``tableA``, ``"b"`` for
            Scholar ``tableB``).
    """

    id: str
    title: str
    authors: str | None = None
    venue: str | None = None
    year: str | None = None
    source: DblpScholarSource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: title, authors, venue and year joined by spaces.

        Omits missing fields so an absent author/venue/year doesn't inject empty
        tokens (mirrors ``ProductSchema.embed_text``).
        """
        return " ".join(p for p in [self.title, self.authors, self.venue, self.year] if p)


load_dblp_scholar, load_dblp_scholar_pair_splits, DblpScholarBenchmark = make_deepmatcher_benchmark(
    name="dblp_scholar",
    schema=DblpScholarSchema,
    dataset_package=_DATASET_PACKAGE,
    table_a=SourceTable(file="tableA.csv", source="a", id_prefix="a"),  # DBLP
    table_b=SourceTable(file="tableB.csv", source="b", id_prefix="b"),  # Scholar
    split_files={"train": "train.csv", "valid": "valid.csv", "test": "test.csv"},
    blocking_k=DBLP_SCHOLAR_BLOCKING_K,
    threshold_grid=DBLP_SCHOLAR_THRESHOLD_GRID,
    achieved_pc=DBLP_SCHOLAR_ACHIEVED_PC,
    gate_met=DBLP_SCHOLAR_GATE_MET,
)
