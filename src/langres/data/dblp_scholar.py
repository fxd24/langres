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
# corpus. Measured cross-source Pair-Completeness sweep (5443 gold pairs;
# reproduce via ``_benchmark_utils.sweep_blocking_k`` — see the tmp/ script noted
# in the ATTRIBUTION):
#   k= 5 -> 0.9151
#   k=10 -> 0.9491
#   k=20 -> 0.9686
#   k=30 -> 0.9761
#   k=50 -> 0.9827
# DBLP-Scholar's titles are discriminative, so vector blocking clears the 0.90
# gate comfortably. ``pick_blocking_k`` picks the SMALLEST k clearing the gate
# (k=5 already reaches 0.9151), keeping the candidate set small; the realised
# recall is recorded in ``DBLP_SCHOLAR_ACHIEVED_PC`` / ``DBLP_SCHOLAR_GATE_MET``.
DBLP_SCHOLAR_BLOCKING_K = 5

#: Pair-Completeness gate the blocking k-sweep aims to clear (mirrors Amazon-Google
#: / Abt-Buy at 0.90). DBLP-Scholar clears it (see ``DBLP_SCHOLAR_GATE_MET``).
DBLP_SCHOLAR_RECALL_GATE = 0.90

#: Cross-source Pair-Completeness achieved at :data:`DBLP_SCHOLAR_BLOCKING_K`,
#: recorded from the measured sweep so callers report realised blocking recall
#: without re-running embeddings.
DBLP_SCHOLAR_ACHIEVED_PC = 0.9151

#: Whether :data:`DBLP_SCHOLAR_ACHIEVED_PC` clears :data:`DBLP_SCHOLAR_RECALL_GATE`.
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
