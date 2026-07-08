"""DBLP-ACM bibliographic entity-resolution benchmark, built via the loader factory (Wave C).

Loads the vendored DBLP-ACM benchmark into a single corpus of
:class:`DblpAcmSchema` records plus cross-source ground-truth pairs, using the
generic :func:`~langres.data._deepmatcher_loader.make_deepmatcher_benchmark`
factory. Like the other DeepMatcher datasets, this is a *linkage* task: two
sources (DBLP ``tableA``, ACM ``tableB``) each list bibliographic citations, and
the matches we care about are the cross-source pairs that refer to the same
publication.

DBLP-ACM is a **clean, near-saturated** bibliographic benchmark (reported SOTA
pairwise F1 ~0.98): records carry four well-populated, comparable fields
(``title`` / ``authors`` / ``venue`` / ``year``), so vector blocking over the
joined citation text reaches the 0.90 Pair-Completeness gate at a small ``k``
(contrast the textual-hard Abt-Buy / Amazon-Google, which plateau lower). The
pinned :data:`DBLP_ACM_BLOCKING_K` / :data:`DBLP_ACM_ACHIEVED_PC` /
:data:`DBLP_ACM_GATE_MET` come from a real ``sweep_blocking_k`` measurement (see
the sweep evidence in the comment on the factory call below), not a guess.
"""

from typing import Literal

from pydantic import BaseModel, computed_field

from langres.data._deepmatcher_loader import SourceTable, make_deepmatcher_benchmark

__all__ = [
    "DBLP_ACM_ACHIEVED_PC",
    "DBLP_ACM_BLOCKING_K",
    "DBLP_ACM_GATE_MET",
    "DBLP_ACM_THRESHOLD_GRID",
    "DblpAcmBenchmark",
    "DblpAcmSchema",
    "load_dblp_acm",
    "load_dblp_acm_pair_splits",
]

_DATASET_PACKAGE = "langres.data.datasets.dblp_acm"

DblpAcmSource = Literal["a", "b"]

# Pinned blocking k for the cross-source DBLP-ACM matches, measured with
# VectorBlocker over SentenceTransformer("all-MiniLM-L6-v2") cosine similarity on
# ``embed_text`` (title + authors + venue + year). Measured sweep (cross-source
# Pair-Completeness, 2220 gold pairs over the full 4910-record corpus; reproduce
# via ``load_dblp_acm()`` + ``_bu.sweep_blocking_k(..., ks=(5,10,20,30,50))``):
#   k= 5 -> 0.9910
#   k=10 -> 0.9968
#   k=20 -> 0.9982
#   k=30 -> 0.9991
#   k=50 -> 0.9991
# DBLP-ACM is clean and near-saturated: the four well-populated bibliographic
# fields make almost every true match a near neighbour, so PC clears the 0.90 gate
# already at k=5 (0.9910). ``pick_blocking_k(sweep, 0.9)`` therefore returns the
# smallest gate-clearing k = 5; larger k buys only marginal recall
# (0.9910 -> 0.9991 by k=30), so k=5 is pinned for cheap blocking.
DBLP_ACM_BLOCKING_K = 5

#: Pair-Completeness gate the blocking k-sweep aims to clear (mirrors Abt-Buy /
#: Amazon-Google's 0.90 target). DBLP-ACM clears it comfortably (see the sweep).
DBLP_ACM_RECALL_GATE = 0.90

#: Cross-source Pair-Completeness achieved at :data:`DBLP_ACM_BLOCKING_K`,
#: recorded from the measured sweep above so callers can report the realised
#: blocking recall without re-running embeddings.
DBLP_ACM_ACHIEVED_PC = 0.991

#: Whether :data:`DBLP_ACM_ACHIEVED_PC` clears :data:`DBLP_ACM_RECALL_GATE`. True
#: here: unlike the textual-hard product datasets, DBLP-ACM's clean multi-field
#: citations make the 0.90 gate easy to hit at a small k.
DBLP_ACM_GATE_MET = DBLP_ACM_ACHIEVED_PC >= DBLP_ACM_RECALL_GATE

#: Clusterer thresholds swept when racing methods on DBLP-ACM (mirrors the other
#: adapters' grids).
DBLP_ACM_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class DblpAcmSchema(BaseModel):
    """A single bibliographic citation record from the DBLP-ACM benchmark.

    ``id`` is globally unique across both sources (the loader prefixes the raw
    source id with ``a``/``b``). ``embed_text`` is the serializable blocking text
    used by the :class:`~langres.core.blockers.vector.VectorBlocker`.

    Attributes:
        id: Globally-unique record id (e.g. ``"a534"`` / ``"b219"``).
        title: Publication title (always present).
        authors: Author list as a raw string, if present (a handful of ACM rows
            have no authors).
        venue: Publication venue (always present).
        year: Publication year as a raw string (always present); kept as a string
            so it joins cleanly into ``embed_text``.
        source: Originating table (``"a"`` for DBLP ``tableA``, ``"b"`` for ACM
            ``tableB``).
    """

    id: str
    title: str
    authors: str | None = None
    venue: str
    year: str
    source: DblpAcmSource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: title, authors, venue, and year joined by a space.

        Omits a missing ``authors`` (and any empty field) so an absent value
        doesn't inject empty tokens (mirrors ``AbtBuySchema.embed_text``).
        """
        return " ".join(p for p in [self.title, self.authors, self.venue, self.year] if p)


load_dblp_acm, load_dblp_acm_pair_splits, DblpAcmBenchmark = make_deepmatcher_benchmark(
    name="dblp_acm",
    schema=DblpAcmSchema,
    dataset_package=_DATASET_PACKAGE,
    table_a=SourceTable(file="tableA.csv", source="a", id_prefix="a"),
    table_b=SourceTable(file="tableB.csv", source="b", id_prefix="b"),
    split_files={"train": "train.csv", "valid": "valid.csv", "test": "test.csv"},
    blocking_k=DBLP_ACM_BLOCKING_K,
    threshold_grid=DBLP_ACM_THRESHOLD_GRID,
    achieved_pc=DBLP_ACM_ACHIEVED_PC,
    gate_met=DBLP_ACM_GATE_MET,
    benchmark_class_name="DblpAcmBenchmark",
)
