"""``Reranker``: retrieve-then-rerank -- a Score AFTER a Select. $0, offline, no key.

One architecture, one file, deliberately self-contained -- see
:mod:`langres.architectures` for why this file repeats topology that other
architectures also build.

**Why this class exists.** The four fixed slots (blocker -> comparator -> matcher
-> clusterer) pin the matcher to ONE position, before the clusterer. A *reranker*
scores twice -- a cheap first pass, a TOP-K prune, then a richer rescore of the
survivors -- so its second ``Score`` sits AFTER a ``Select``. That topology is
inexpressible in four fixed slots. ``Reranker`` builds it through the PUBLIC
:meth:`~langres.core._model_state.ModelState.from_topology` door as an explicit Op
chain, adding **no** core change: it is the expressiveness payoff of epic #193.
"""

from __future__ import annotations

from typing import ClassVar, Self

from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparators import StringComparator
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.op import Stage, ThresholdSelect, TopKSelect
from langres.core.op_adapters import (
    BlockerSource,
    ClustererStage,
    ComparatorScore,
    MatcherScore,
)
from langres.core.registry import register_model
from langres.core.resolver import ERModel

__all__ = ["Reranker"]


@register_model("reranker")
class Reranker(ERModel):
    """Retrieve-then-rerank: a cheap first pass, a top-k prune, then a richer rescore.

    **The Score-after-Select architecture.** It runs offline, needs no API key,
    touches no network, and is deterministic -- the same records give the same
    clusters every time. It exists to demonstrate a topology the four fixed slots
    cannot place: a second ``Score`` AFTER a ``Select``::

        from langres.architectures import Reranker
        from langres.core.models import CompanySchema

        model = Reranker.for_schema(CompanySchema, k=5, threshold=0.85)
        clusters = model.dedupe(records)

    Topology (an explicit 7-op chain, built via the public
    :meth:`~langres.core._model_state.ModelState.from_topology` door -- **not** the
    four-slot ``_topology`` hook the other architectures use):

    ==================  ========================================================
    1 source            ``BlockerSource(AllPairsBlocker)`` -- every pair, O(N^2)
    2 score (vector)    ``ComparatorScore(StringComparator)`` -- per-feature
                        ``ComparisonVector`` (``out_space="vector"``), score left
                        unset
    3 score (scalar)    ``MatcherScore(WeightedAverageMatcher, first feature)`` --
                        the CHEAP first pass; overwrites the vector with a scalar
                        BEFORE the select (so the select is legal)
    4 select            ``TopKSelect(k)`` -- keep each anchor's k best by the cheap
                        score
    5 score (scalar)    ``MatcherScore(WeightedAverageMatcher, all features)`` --
                        the RERANK: rescores the survivors on the full evidence.
                        **This is the Score after the Select.**
    6 select            ``ThresholdSelect(threshold)`` -- the match cut
    7 cluster           ``ClustererStage(Clusterer(0.0))`` -- pure transitive
                        closure over the survivors (the cut already ran at step 6)
    ==================  ========================================================

    **The two passes are a different feature subset, on purpose.** Both
    ``WeightedAverageMatcher``\\ s read the SAME per-feature ``ComparisonVector``
    the ``ComparatorScore`` attached once at step 2 -- they differ only in which
    features they weight. The cheap pass scores on the schema's **first** comparable
    field alone (a coarse single-signal retrieve, like matching on a name); the
    rerank scores on **all** comparable fields (the full evidence). So a pair that
    is a cheap-pass winner (its first field matches) can be demoted by the rerank
    (its other fields disagree) -- which is exactly what separates a same-name /
    different-address near-duplicate. The rerank meaningfully changes the ranking
    whenever the schema has **>= 2** comparable fields; with one field the two
    passes coincide (a degenerate but still-legal reranker).

    **On quality, honestly:** like ``FuzzyString`` this is unsupervised fuzzy
    matching, so it over-merges on unlabeled data -- the rerank narrows that, it
    does not cure it. Calibrate the threshold against labels before trusting it on
    anything that matters.

    **It persists with zero core change.** ``save()``/``load()`` round-trip through
    the explicit-chain persist path (epic #193 persist v2): the manifest stores an
    ``ops`` list, ``model_class="reranker"`` is stamped, and a loaded artifact comes
    back a ``Reranker`` whose paid Scores are re-secured against a fresh spend
    ledger. Both ``WeightedAverageMatcher``\\ s are pure-data (their
    ``feature_specs`` serialize via ``config``/``from_config``), and the
    ``AllPairsBlocker`` references its schema by name -- so nothing here is a
    closure and the whole chain reloads intact. Pass a real ``schema`` (not an
    inferred one) for anything you intend to ``save``.

    Args:
        schema: The entity schema. **Required** at construction -- a reranker is an
            explicit Op chain, so there is no lazy schema-inference (contrast
            ``FuzzyString``, whose ``_topology`` defers binding until first use).
        k: The number of survivors ``TopKSelect`` keeps per anchor after the cheap
            first pass.
        threshold: The match cut on the reranked (full-evidence) score.
        budget_usd: Spend cap for this model's lifetime. Present for symmetry and
            metered like any other model's -- but this architecture reports $0 per
            pair (free string matchers only), so the cap can never trip.
    """

    #: Refuses every fit kind. A reranker's identity is its two-pass Op chain, and
    #: every fit kind repoints a single matcher slot -- which an explicit-chain
    #: model does not have (it has two matcher Scores, not one ``module`` slot). So
    #: there is nothing coherent for ``fit`` to repoint without changing what the
    #: class means; refuse all, exactly as ``VectorLLMCascade`` does for its own
    #: (different) reason. See ``ERModel.accepted_method_kinds``.
    accepted_method_kinds: ClassVar[frozenset[str] | None] = frozenset()

    @classmethod
    def for_schema(
        cls,
        schema: type[BaseModel],
        *,
        k: int,
        threshold: float,
        budget_usd: float | None = None,
    ) -> Self:
        """Build a ``Reranker`` for ``schema`` as an explicit 7-op chain.

        Named ``for_schema`` rather than ``from_schema`` deliberately: the inherited
        :meth:`~langres.core.resolver.ERModel.from_schema` builds the classic
        FOUR-SLOT pipeline (one matcher position), which is the opposite of what a
        reranker is. This constructor derives the comparable features from
        ``schema`` the way the other architectures do, assembles the two-pass chain,
        and wires it through the public
        :meth:`~langres.core._model_state.ModelState.from_topology` door -- so the
        four slots stay empty and the model runs the explicit chain.

        Args:
            schema: The entity schema (required -- see the class docstring).
            k: Survivors ``TopKSelect`` keeps per anchor after the cheap pass.
            threshold: The match cut on the reranked score.
            budget_usd: Spend cap for the instance's lifetime (``None`` -> the
                default; this architecture cannot spend regardless).

        Returns:
            A ``Reranker`` wired to run the explicit chain, ready to
            ``dedupe``/``compare``.

        Raises:
            NoComparableFeatures: If ``schema`` yields no comparable string field
                (raised by :meth:`StringComparator.from_schema`).
        """
        # ONE comparator, attached once (step 2); both passes read the vectors it
        # produces. The cheap pass weights the FIRST comparable field alone (a
        # coarse retrieve); the rerank weights ALL of them (the full evidence).
        comparator: StringComparator[BaseModel] = StringComparator.from_schema(schema)
        first_pass = [comparator.feature_specs[0]]
        full_pass = comparator.feature_specs

        ops: list[Stage] = [
            BlockerSource(AllPairsBlocker(schema=schema)),
            ComparatorScore(comparator),
            # The cheap scalar OVERWRITES the vector score before the Select, so the
            # TopKSelect is over an orderable scalar (Sequential rejects a Select on
            # a vector out_space).
            MatcherScore(WeightedAverageMatcher(feature_specs=first_pass), out_space="heuristic"),
            TopKSelect(k),
            # The Score AFTER the Select -- the reranker. A different feature subset
            # from pass 1, so it genuinely re-ranks the survivors.
            MatcherScore(WeightedAverageMatcher(feature_specs=full_pass), out_space="heuristic"),
            ThresholdSelect(threshold),
            ClustererStage(Clusterer(threshold=0.0)),
        ]
        return cls.from_topology(ops=ops, budget_usd=budget_usd)
