"""Cold-start gold-set orchestrator (M1 Wave 4).

:class:`Bootstrapper` wires the Wave 1-3 pieces into one pass: it builds the
blocker's vector index, materializes blocker candidates, optionally filters them
with a caller-supplied predicate, mines a stratified subset, labels it, and
assembles a :class:`~langres.curation.models.GoldSet` plus a
:class:`~langres.curation.report.BootstrapReport`.

It is a plain class with three injected dependencies (design-review W1) and is
deliberately *entity-type-agnostic*: any domain-specific selection rule (e.g. the
Fodors-Zagat cross-source rule ``lambda c: c.left.source != c.right.source``) is
passed IN as ``candidate_filter`` rather than baked in (design-review B2).
"""

import logging
from collections.abc import Callable
from typing import Any

from langres.curation.base import Labeler, Miner
from langres.curation.models import GoldSet
from langres.curation.report import BootstrapReport
from langres.core.blockers.vector import VectorBlocker
from langres.core.models import ERCandidate

logger = logging.getLogger(__name__)


class Bootstrapper:
    """Orchestrate one cold-start bootstrap run: block -> filter -> mine -> label.

    The three collaborators are injected so each can be swapped or faked:

    Args:
        blocker: A :class:`VectorBlocker` whose index this orchestrator builds
            from the corpus (the private ``Resolver._ensure_index_built`` is not
            reused here -- design-review W3).
        miner: Selects which candidate pairs are worth labeling.
        labeler: Assigns match / non-match labels to the mined pairs.
    """

    def __init__(self, blocker: VectorBlocker[Any], miner: Miner, labeler: Labeler) -> None:
        self.blocker = blocker
        self.miner = miner
        self.labeler = labeler

    def build(
        self,
        corpus: list[Any],
        *,
        candidate_filter: Callable[[ERCandidate[Any]], bool] | None = None,
        gold_clusters: list[set[str]] | None = None,
    ) -> tuple[GoldSet, BootstrapReport]:
        """Run the bootstrap pipeline and return the gold set plus its report.

        Args:
            corpus: Raw records (typically dicts) the blocker's ``schema_factory``
                consumes -- the same shape passed to ``blocker.stream``.
            candidate_filter: Optional predicate keeping only the candidates worth
                mining (e.g. cross-source pairs for a linkage task). When ``None``
                every blocker candidate is mined. Kept as a parameter so the
                orchestrator stays entity-type-agnostic (design-review B2).
            gold_clusters: Optional ground-truth match sets, used only by the
                report for pair-completeness and teacher-vs-truth agreement. When
                ``None`` the report still builds (with empty ground truth).

        Returns:
            ``(gold_set, report)``.
        """
        # 1. Build the blocker's index from the corpus (W3: orchestrator owns this).
        #    Reuse the blocker's own factory + text extractor so we stay agnostic
        #    to the entity type -- this is exactly what stream() does internally.
        entities = [self.blocker.schema_factory(record) for record in corpus]
        texts = [self.blocker.text_field_extractor(entity) for entity in entities]
        self.blocker.vector_index.create_index(texts)

        # 2. Materialize candidates: stream() is a single-pass iterator and the
        #    miner needs the full list for its percentile strata (W3).
        candidates = list(self.blocker.stream(corpus))

        # 3. Apply the caller's domain filter (e.g. cross-source -- B2).
        filtered = (
            [c for c in candidates if candidate_filter(c)]
            if candidate_filter is not None
            else candidates
        )

        # 4. Mine a stratified subset, then 5. label it. If the labeler caps how
        #    many pairs it can afford (e.g. a budget-capped teacher), pass that cap
        #    to the miner so the high/mid/low strata are honored UP FRONT -- mining
        #    the full pool and letting the labeler truncate in input order would
        #    bypass the stratified allocation on tight-budget runs.
        cap = self.labeler.max_labelable(len(filtered))
        mined = self.miner.mine(filtered, max_pairs=cap)
        pairs = self.labeler.label(mined)

        logger.info(
            "Bootstrap: %d candidates -> %d filtered -> %d mined -> %d labeled",
            len(candidates),
            len(filtered),
            len(mined),
            len(pairs),
        )

        # 6. Assemble the gold set (honest counts + cost) and its report.
        matches = sum(1 for p in pairs if p.label)
        gold = GoldSet(
            pairs=pairs,
            metadata={
                "blocker": type(self.blocker).__name__,
                "k_neighbors": self.blocker.k_neighbors,
                "labeler": type(self.labeler).__name__,
                "total_cost_usd": float(getattr(self.labeler, "total_spent_usd", 0.0)),
                "corpus_size": len(corpus),
                "total_candidates": len(candidates),
                "filtered_candidates": len(filtered),
                "mined": len(mined),
                "labeled": len(pairs),
                "matches": matches,
                "non_matches": len(pairs) - matches,
            },
        )
        # Pair-completeness is measured on the FILTERED candidates -- for a
        # linkage task the meaningful blocking output is the cross-source pairs
        # that actually feed the pipeline (mirrors the data adapter's k-sweep).
        report = BootstrapReport.build(gold, filtered, gold_clusters or [])
        return gold, report
