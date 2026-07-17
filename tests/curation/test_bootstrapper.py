"""Fast, network-free tests for the Wave-4 :class:`Bootstrapper` orchestrator.

Every collaborator is faked: a blocker that records its index build and yields
canned :class:`ERCandidate`s, a miner and a labeler that record the lists they
receive. This proves the orchestrator wires the steps in order (build index ->
materialize -> filter -> mine -> label -> assemble), applies the optional
``candidate_filter``, and assembles a :class:`GoldSet` + :class:`BootstrapReport`
with honest counts and metadata -- with no embeddings and no LLM.
"""

from collections.abc import Iterator
from typing import Any

from langres.curation.base import Labeler
from langres.curation.bootstrapper import Bootstrapper
from langres.curation.models import GoldPair, GoldSet
from langres.curation.report import BootstrapReport
from langres.core.models import CompanySchema, ERCandidate


def _cand(left_id: str, right_id: str, score: float) -> ERCandidate[CompanySchema]:
    return ERCandidate[CompanySchema](
        left=CompanySchema(id=left_id, name=left_id),
        right=CompanySchema(id=right_id, name=right_id),
        blocker_name="fake",
        similarity_score=score,
    )


class _RecordingIndex:
    """Captures the texts passed to ``create_index`` so the test can assert it."""

    def __init__(self) -> None:
        self.created_with: list[str] | None = None
        self.create_calls = 0

    def create_index(self, texts: list[str]) -> None:
        self.create_calls += 1
        self.created_with = list(texts)


class _FakeBlocker:
    """Duck-typed VectorBlocker: builds an index and yields canned candidates.

    ``stream`` returns a one-shot iterator so the test proves the orchestrator
    materializes it (a second consumer would otherwise see nothing).
    """

    def __init__(self, candidates: list[ERCandidate[Any]]) -> None:
        self._candidates = candidates
        self.vector_index = _RecordingIndex()
        self.k_neighbors = 7
        self.schema_factory = lambda record: CompanySchema(**record)
        self.text_field_extractor = lambda entity: entity.name
        self.stream_arg: list[Any] | None = None

    def stream(self, data: list[Any]) -> Iterator[ERCandidate[Any]]:
        self.stream_arg = data
        return iter(self._candidates)


class _RecordingMiner:
    """Records the candidates it was asked to mine and returns a fixed subset."""

    def __init__(self, returns: list[ERCandidate[Any]]) -> None:
        self._returns = returns
        self.seen: list[ERCandidate[Any]] | None = None
        self.seen_max_pairs: int | None = None

    def mine(
        self, candidates: list[ERCandidate[Any]], *, max_pairs: int | None = None
    ) -> list[ERCandidate[Any]]:
        self.seen = candidates
        self.seen_max_pairs = max_pairs
        return self._returns


class _RecordingLabeler:
    """Records the mined pairs and emits one canned GoldPair each."""

    def __init__(self, *, total_spent_usd: float = 0.0, cap: int | None = None) -> None:
        self.seen: list[ERCandidate[Any]] | None = None
        self.total_spent_usd = total_spent_usd
        self._cap = cap

    def max_labelable(self, n_candidates: int) -> int | None:
        return self._cap

    def label(self, candidates: list[ERCandidate[Any]]) -> list[GoldPair]:
        self.seen = candidates
        return [
            GoldPair(
                left_id=c.left.id,
                right_id=c.right.id,
                label=c.similarity_score is not None and c.similarity_score >= 0.5,
                source="fake",
                confidence=c.similarity_score,
            )
            for c in candidates
        ]


_CORPUS: list[dict[str, str]] = [
    {"id": "a", "name": "Acme"},
    {"id": "b", "name": "Acme Inc"},
    {"id": "c", "name": "Globex"},
]


def test_build_wires_steps_in_order_and_builds_index() -> None:
    """build() builds the index, materializes, mines, labels, and assembles."""
    candidates = [_cand("a", "b", 0.9), _cand("a", "c", 0.2)]
    blocker = _FakeBlocker(candidates)
    miner = _RecordingMiner(returns=candidates)
    labeler = _RecordingLabeler()

    gold, report = Bootstrapper(blocker, miner, labeler).build(  # type: ignore[arg-type]
        _CORPUS, gold_clusters=[{"a", "b"}]
    )

    # Index built EXACTLY once from the blocker's own text extractor, in corpus
    # order (stream() reads the pre-built index, it never re-builds it).
    assert blocker.vector_index.created_with == ["Acme", "Acme Inc", "Globex"]
    assert blocker.vector_index.create_calls == 1
    # stream received the raw corpus.
    assert blocker.stream_arg == _CORPUS
    # Miner saw the materialized candidate list (one-shot iterator consumed once).
    assert miner.seen == candidates
    # Labeler saw the mined pairs.
    assert labeler.seen == candidates
    # GoldSet + report returned with the right shapes.
    assert isinstance(gold, GoldSet)
    assert isinstance(report, BootstrapReport)
    assert len(gold.pairs) == 2


def test_build_applies_candidate_filter() -> None:
    """A candidate_filter drops non-matching candidates before mining."""
    keep = _cand("a", "b", 0.9)
    drop = _cand("a", "c", 0.2)
    blocker = _FakeBlocker([keep, drop])
    miner = _RecordingMiner(returns=[keep])
    labeler = _RecordingLabeler()

    gold, _ = Bootstrapper(blocker, miner, labeler).build(  # type: ignore[arg-type]
        _CORPUS, candidate_filter=lambda c: c.similarity_score == 0.9
    )

    # The filter kept only the high-score candidate; the miner never saw `drop`.
    assert miner.seen == [keep]
    assert gold.metadata["total_candidates"] == 2
    assert gold.metadata["filtered_candidates"] == 1


def test_build_without_filter_mines_all_candidates() -> None:
    """Without a filter, every blocker candidate reaches the miner."""
    candidates = [_cand("a", "b", 0.9), _cand("a", "c", 0.2)]
    blocker = _FakeBlocker(candidates)
    miner = _RecordingMiner(returns=candidates)
    labeler = _RecordingLabeler()

    Bootstrapper(blocker, miner, labeler).build(_CORPUS)  # type: ignore[arg-type]

    assert miner.seen == candidates
    # Uncapped labeler -> miner mines with no max_pairs (stratify the whole pool).
    assert miner.seen_max_pairs is None


def test_build_passes_labeler_cap_to_miner() -> None:
    """A budget-capped labeler's cap reaches miner.mine so stratification happens
    BEFORE truncation (not bypassed by input-order labeler truncation)."""
    candidates = [_cand("a", "b", 0.9), _cand("a", "c", 0.2), _cand("b", "c", 0.5)]
    blocker = _FakeBlocker(candidates)
    miner = _RecordingMiner(returns=candidates[:2])
    labeler = _RecordingLabeler(cap=2)  # e.g. a tight teacher budget

    Bootstrapper(blocker, miner, labeler).build(_CORPUS)  # type: ignore[arg-type]

    # The miner received the labeler's cap, so it stratified down to 2 up front.
    assert miner.seen_max_pairs == 2


def test_build_metadata_records_counts_and_cost() -> None:
    """Metadata carries honest counts, the cost from the labeler, and config refs."""
    candidates = [_cand("a", "b", 0.9), _cand("a", "c", 0.2)]
    blocker = _FakeBlocker(candidates)
    miner = _RecordingMiner(returns=candidates)
    labeler = _RecordingLabeler(total_spent_usd=1.25)

    gold, _ = Bootstrapper(blocker, miner, labeler).build(_CORPUS)  # type: ignore[arg-type]

    md = gold.metadata
    assert md["blocker"] == "_FakeBlocker"
    assert md["k_neighbors"] == 7
    assert md["labeler"] == "_RecordingLabeler"
    assert md["total_cost_usd"] == 1.25
    assert md["corpus_size"] == 3
    assert md["total_candidates"] == 2
    assert md["mined"] == 2
    assert md["labeled"] == 2
    assert md["matches"] == 1  # only (a,b) cleared 0.5
    assert md["non_matches"] == 1


def test_build_without_gold_clusters_still_builds_report() -> None:
    """gold_clusters=None yields a report with empty ground truth (no agreement)."""
    candidates = [_cand("a", "b", 0.9)]
    blocker = _FakeBlocker(candidates)
    miner = _RecordingMiner(returns=candidates)
    labeler = _RecordingLabeler()

    _, report = Bootstrapper(blocker, miner, labeler).build(_CORPUS)  # type: ignore[arg-type]

    # No ground truth -> no agreement / calibration, but blocking still reports.
    assert report.agreement is None
    assert report.calibration is None
    assert report.blocking.total_candidates == 1


def test_build_cost_defaults_to_zero_when_labeler_overrides_nothing() -> None:
    """A Labeler that implements only label() inherits the ABC defaults:
    zero spend (total_cost_usd 0.0) and no cap (max_labelable -> None)."""

    class _NoSpendLabeler(Labeler):
        def label(self, candidates: list[ERCandidate[Any]]) -> list[GoldPair]:
            return []

    blocker = _FakeBlocker([])
    miner = _RecordingMiner(returns=[])
    gold, _ = Bootstrapper(blocker, miner, _NoSpendLabeler()).build(  # type: ignore[arg-type]
        _CORPUS
    )
    assert gold.metadata["total_cost_usd"] == 0.0
    assert gold.pairs == []
    assert miner.seen_max_pairs is None  # inherited uncapped default


def test_build_with_empty_corpus_produces_empty_gold_set() -> None:
    """An empty corpus builds an (empty) index and yields no pairs."""
    blocker = _FakeBlocker([])
    miner = _RecordingMiner(returns=[])
    labeler = _RecordingLabeler()

    gold, report = Bootstrapper(blocker, miner, labeler).build([])  # type: ignore[arg-type]

    assert blocker.vector_index.created_with == []
    assert blocker.vector_index.create_calls == 1
    assert gold.pairs == []
    assert gold.metadata["corpus_size"] == 0
    assert gold.metadata["total_candidates"] == 0
    assert report.blocking.total_candidates == 0


def test_build_filter_dropping_all_candidates_mines_nothing() -> None:
    """A filter that rejects everything leaves the miner and labeler with []."""
    candidates = [_cand("a", "b", 0.9), _cand("a", "c", 0.2)]
    blocker = _FakeBlocker(candidates)
    miner = _RecordingMiner(returns=[])
    labeler = _RecordingLabeler()

    gold, _ = Bootstrapper(blocker, miner, labeler).build(  # type: ignore[arg-type]
        _CORPUS, candidate_filter=lambda c: False
    )

    assert miner.seen == []
    assert labeler.seen == []
    assert gold.pairs == []
    assert gold.metadata["filtered_candidates"] == 0
