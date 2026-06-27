"""Entity-resolution benchmark adapters (Fodors-Zagat restaurant matching).

Loads the vendored Fodors-Zagat benchmark into a single corpus of
:class:`RestaurantSchema` records plus cross-source ground-truth pairs, and
provides a blocking k-sweep that pins the Pair-Completeness >= 0.95 gate.

The benchmark is a *linkage* task: two sources (Fodor's, Zagat) each list
restaurants, and the matches we care about are cross-source. The k-sweep
therefore filters candidate pairs to cross-source ones before measuring recall
(intra-source pairs are noise for this task; see DESIGN-REVIEW B2).
"""

import csv
import logging
from importlib import resources
from typing import Literal

from pydantic import BaseModel, computed_field

from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.metrics import evaluate_blocking
from langres.core.models import ERCandidate

logger = logging.getLogger(__name__)

_DATASET_PACKAGE = "langres.data.datasets.fodors_zagat"
_FODORS_FILE = "fodors.csv"
_ZAGATS_FILE = "zagats.csv"
_MAPPING_FILE = "fodors-zagats_perfectMapping.csv"

RestaurantSource = Literal["fodors", "zagat"]

# Pinned blocking k that meets Pair-Completeness >= 0.95 on the cross-source
# Fodors-Zagat matches, measured with VectorBlocker over
# SentenceTransformer("all-MiniLM-L6-v2") cosine similarity on ``embed_text``.
#
# Measured sweep (cross-source Pair-Completeness, 112 gold pairs):
#   k= 5 -> 0.9911
#   k=10 -> 0.9911
#   k=20 -> 0.9911
#   k=30 -> 0.9911
#   k=50 -> 0.9911
# Recall is flat: 111/112 pairs are captured within each record's top-5
# cross-source neighbours, so larger k adds no new true matches. The single
# missed pair (f640/z325, "masa's") has *identical* embed_text in both sources;
# VectorBlocker's position-based self-skip drops one exact-duplicate neighbour,
# so this pair is structurally missed at every k (a pre-existing blocker edge
# case, out of scope here). The 0.95 gate is still cleared at the minimum k.
# DEFAULT_BLOCKING_K is the MIN k clearing 0.95 (see ``pick_blocking_k``).
DEFAULT_BLOCKING_K = 5

#: Pair-Completeness gate the blocking k-sweep must clear (DESIGN-REVIEW W7).
RECALL_GATE = 0.95


class RestaurantSchema(BaseModel):
    """A single restaurant record from the Fodors-Zagat benchmark.

    ``id`` is globally unique across both sources (the loader prefixes the raw
    source id with ``f``/``z``). ``embed_text`` is the serializable blocking
    text used by the :class:`VectorBlocker` (referenced as ``text_field``).

    Attributes:
        id: Globally-unique record id (e.g. ``"f534"`` / ``"z219"``).
        name: Restaurant name (always present).
        addr: Street address, if present.
        city: City, if present.
        phone: Phone number, if present.
        type: Cuisine / category, if present.
        source: Originating guide (``"fodors"`` or ``"zagat"``).
    """

    id: str
    name: str
    addr: str | None = None
    city: str | None = None
    phone: str | None = None
    type: str | None = None
    source: RestaurantSource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: name, city and address joined by spaces.

        Used as the :class:`VectorBlocker` ``text_field`` and as the text fed to
        the vector index. Omits missing components so absent fields don't inject
        empty tokens.
        """
        return " ".join(p for p in [self.name, self.city, self.addr] if p)


def _unquote(value: str) -> str:
    """Strip the dataset's wrapping single quotes and unescape inner quotes.

    The source wraps text fields in single quotes and escapes inner quotes as
    ``\\'`` (e.g. ``'art\\'s delicatessen'``). Numeric / unquoted fields pass
    through unchanged.

    Args:
        value: A raw CSV cell.

    Returns:
        The cleaned cell value.
    """
    value = value.strip()
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value.replace("\\'", "'")


def _read_csv_rows(filename: str) -> list[dict[str, str]]:
    """Read a packaged benchmark CSV into a list of header-keyed row dicts."""
    text = resources.files(_DATASET_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    return [dict(row) for row in reader]


def _record_from_row(
    row: dict[str, str], source: RestaurantSource, prefix: str
) -> RestaurantSchema:
    """Build a :class:`RestaurantSchema` from a raw CSV row dict."""

    def field(name: str) -> str | None:
        cleaned = _unquote(row.get(name, ""))
        return cleaned or None

    return RestaurantSchema(
        id=f"{prefix}{_unquote(row['id'])}",
        name=field("name") or "",
        addr=field("addr"),
        city=field("city"),
        phone=field("phone"),
        type=field("type"),
        source=source,
    )


def load_fodors_zagat() -> tuple[list[RestaurantSchema], list[set[str]]]:
    """Load the Fodors-Zagat benchmark as one corpus plus cross-source matches.

    Both sources are combined into a single corpus of 864 records with globally
    unique, source-prefixed ids (``f<id>`` for Fodor's, ``z<id>`` for Zagat).
    Ground truth comes from the explicit ``perfectMapping`` file: each row
    yields a 2-element set ``{f<fodors_id>, z<zagats_id>}``.

    Returns:
        ``(corpus, gold_clusters)`` where ``corpus`` is the combined record list
        and ``gold_clusters`` is the list of 2-element cross-source match sets.
    """
    corpus: list[RestaurantSchema] = [
        _record_from_row(row, "fodors", "f") for row in _read_csv_rows(_FODORS_FILE)
    ]
    corpus += [_record_from_row(row, "zagat", "z") for row in _read_csv_rows(_ZAGATS_FILE)]

    gold_clusters: list[set[str]] = [
        {f"f{_unquote(row['fodors_id'])}", f"z{_unquote(row['zagats_id'])}"}
        for row in _read_csv_rows(_MAPPING_FILE)
    ]

    logger.info(
        "Loaded Fodors-Zagat: %d records (%d fodors + %d zagat), %d gold pairs",
        len(corpus),
        sum(1 for r in corpus if r.source == "fodors"),
        sum(1 for r in corpus if r.source == "zagat"),
        len(gold_clusters),
    )
    return corpus, gold_clusters


def _cross_source(
    candidates: list[ERCandidate[RestaurantSchema]],
) -> list[ERCandidate[RestaurantSchema]]:
    """Keep only candidate pairs whose two records come from different sources."""
    return [c for c in candidates if c.left.source != c.right.source]


def sweep_blocking_k(
    corpus: list[RestaurantSchema],
    gold_clusters: list[set[str]],
    ks: tuple[int, ...] = (5, 10, 20, 30, 50),
) -> dict[int, float]:
    """Measure cross-source Pair-Completeness of vector blocking across ``ks``.

    Builds the FAISS index once over ``embed_text`` and reuses it across all
    ``k`` (only ``k_neighbors`` changes). For each ``k`` the candidates are
    filtered to cross-source pairs (DESIGN-REVIEW B2) before recall is measured
    via :func:`evaluate_blocking`.

    Args:
        corpus: Combined record list from :func:`load_fodors_zagat`.
        gold_clusters: Cross-source match sets from :func:`load_fodors_zagat`.
        ks: Neighbor counts to sweep.

    Returns:
        Mapping of ``k`` to cross-source Pair-Completeness (``candidate_recall``).
    """
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    index = FAISSIndex(embedder=embedder, metric="cosine")
    index.create_index([r.embed_text for r in corpus])

    blocker: VectorBlocker[RestaurantSchema] = VectorBlocker(
        vector_index=index,
        schema=RestaurantSchema,
        text_field="embed_text",
        k_neighbors=max(ks),
    )
    records = [r.model_dump() for r in corpus]

    recalls: dict[int, float] = {}
    for k in ks:
        blocker.k_neighbors = k
        candidates = _cross_source(list(blocker.stream(records)))
        recall = evaluate_blocking(candidates, gold_clusters).candidate_recall
        recalls[k] = recall
        logger.info("blocking k=%d -> cross-source recall=%.4f", k, recall)
    return recalls


def pick_blocking_k(recalls: dict[int, float], threshold: float = RECALL_GATE) -> int:
    """Pick the smallest ``k`` whose recall clears ``threshold``.

    If no ``k`` reaches ``threshold``, returns the ``k`` with the highest recall
    (the honest best-effort fallback; callers should document the shortfall
    rather than fake the gate).

    Args:
        recalls: Mapping of ``k`` to recall, e.g. from :func:`sweep_blocking_k`.
        threshold: Minimum acceptable recall.

    Returns:
        The chosen ``k``.

    Raises:
        ValueError: If ``recalls`` is empty.
    """
    if not recalls:
        raise ValueError("recalls is empty; nothing to pick from")
    passing = [k for k in sorted(recalls) if recalls[k] >= threshold]
    if passing:
        return passing[0]
    return max(recalls, key=lambda k: recalls[k])
