"""``FuzzyString``: offline fuzzy string matching. $0, deterministic, no key.

One architecture, one file, deliberately self-contained -- see
:mod:`langres.architectures` for why this file repeats topology that other
architectures also build.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparators import StringComparator
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.registry import register_model
from langres.core.resolver import ERModel

__all__ = ["FuzzyString"]


@register_model("fuzzy_string")
class FuzzyString(ERModel):
    """All-pairs blocking + per-field string similarity + weighted average.

    **The $0 architecture.** It runs offline, needs no API key, touches no
    network, and is deterministic -- the same records give the same clusters
    every time. It cannot spend money because it has no model slot to put a paid
    backbone in, not because a spend cap happens to hold::

        from langres.architectures import FuzzyString

        clusters = FuzzyString().dedupe(records)
        verdict = FuzzyString().compare(records[0], records[1])

    Topology (fixed -- this is what "a new topology is a new class" means):

    ===============  ===========================================================
    blocker          ``AllPairsBlocker`` -- every pair, O(N^2)
    comparator       ``StringComparator`` -- per-field, missing-aware similarity
    matcher          ``WeightedAverageMatcher`` -- weighted mean of the features
    clusterer        ``Clusterer`` -- transitive closure above ``threshold``
    ===============  ===========================================================

    **On quality, honestly:** unsupervised fuzzy matching over-merges on
    unlabeled data. That is *why* it was never the silent fallback for a missing
    API key -- a wrong answer for free is still a wrong answer. Calibrate the
    threshold against labels with
    :func:`~langres.training.calibration.derive_threshold`, or
    ``fit(method=Platt())``, before trusting it on anything that matters.

    Scaling: ``AllPairsBlocker`` is O(N^2) pairs by construction. That is fine
    for the thousands and wrong for the millions -- an architecture that blocks
    sub-quadratically is a *different* architecture (see
    :class:`~langres.architectures.vector_llm_cascade.VectorLLMCascade`), not a
    flag on this one.

    Args:
        threshold: The match cut on the weighted-average similarity (a
            ``"heuristic"`` score, not a probability -- 0.5 by default).
        weights: Per-feature weight overrides for the comparator. Defaults to
            equal weights; pass name-dominant weights (e.g.
            ``{"name": 0.6, "address": 0.2}``) to recover name-only duplicates
            that equal weights gate out via the evidence floor.
        exclude: Field names to skip when deriving features (``{"id"}`` by
            default, handled by the comparator).
        clusterer: The grouping algorithm. ``None`` (default) builds the base
            transitive-closure :class:`~langres.core.clusterer.Clusterer` -- the
            unchanged shipped behaviour. Pass
            :class:`~langres.core.clusterers.correlation.CorrelationClusterer`
            to opt into pivot clustering, which measures better on the benchmark
            portfolio (``docs/research/20260727_closure_diagnostic.md``) and is
            the recommended choice::

                from langres.core.clusterers import CorrelationClusterer

                FuzzyString(clusterer=CorrelationClusterer()).dedupe(records)

            This selects the *algorithm* only: the clusterer is rebuilt at this
            model's ``threshold``, so any threshold set on the object you pass is
            ignored and ``threshold=`` stays the one match cut (the value
            ``DedupeResult.threshold`` reports). Topology is otherwise unchanged
            -- swapping the clusterer is still a ``FuzzyString``.
        schema: The entity schema. Omit it and the schema is **inferred** from
            the records' own keys on first use -- which is what makes
            ``FuzzyString().dedupe(records)`` work. Pass it explicitly for
            anything you intend to ``save``: an inferred schema is an ephemeral
            class that a fresh process cannot import back.
        budget_usd: Spend cap for this model's lifetime. Present for symmetry
            and metered like any other model's -- but this architecture reports
            $0 per pair, so the cap can never trip.
    """

    #: This architecture can absorb a ``calibrate`` fit (a Platt/Isotonic map is
    #: a *post-hoc* score transform -- the topology is untouched, so it is still
    #: FuzzyString). It refuses ``prompt`` and ``finetune``: both mint or repoint
    #: a model-backed matcher, and ``_fit_finetune`` in particular *replaces the
    #: matcher slot*, which would leave an LLM pipeline still calling itself
    #: "FuzzyString" -- a name that lies. See ``ERModel.accepted_method_kinds``.
    accepted_method_kinds: ClassVar[frozenset[str] | None] = frozenset({"calibrate"})

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        weights: dict[str, float] | None = None,
        exclude: set[str] | None = None,
        clusterer: Clusterer | None = None,
        schema: type[BaseModel] | None = None,
        budget_usd: float | None = None,
    ) -> None:
        # Hyperparameters only -- sklearn's rule, and here it is load-bearing
        # rather than stylistic: the components cannot exist yet, because the
        # schema may not be known until dedupe() sees the records.
        self.threshold = threshold
        self.weights = weights
        self.exclude = exclude
        # NOT ``self.clusterer``: that name is an ERModel SLOT (a property whose
        # getter raises until the model is bound). This is the un-bound *choice*,
        # read by _topology at bind time -- same split, same name, as the four
        # retrieval recipes in architectures/retrieval.py.
        self.clusterer_override = clusterer
        self._init_state(budget_usd=budget_usd)
        if schema is not None:
            self._bind(schema)

    def _topology(self, schema: type[BaseModel]) -> dict[str, Any]:
        """Build the four slots for ``schema``. Called once, on binding."""
        comparator: StringComparator[Any] = StringComparator.from_schema(
            schema, exclude=self.exclude, weights=self.weights
        )
        return {
            "blocker": AllPairsBlocker(schema=schema),
            "comparator": comparator,
            "matcher": WeightedAverageMatcher(feature_specs=comparator.feature_specs),
            "clusterer": self._build_clusterer(),
        }

    def _build_clusterer(self) -> Clusterer:
        """The clusterer slot: the caller's algorithm, at THIS model's threshold.

        Rebuilt through ``config``/``from_config`` rather than used as passed, so
        ``threshold=`` stays the single match cut no matter what threshold the
        caller happened to set on the object. That is the same clone seam
        ``ERModel`` already puts every clusterer through on every resolve (its
        ``_closure_clusterer``), so a clusterer that survives a run survives this,
        and ``from_config`` re-runs ``__init__`` -- keeping the range validation a
        plain attribute assignment would have skipped.
        """
        if self.clusterer_override is None:
            return Clusterer(threshold=self.threshold)
        return type(self.clusterer_override).from_config(
            {**self.clusterer_override.config, "threshold": self.threshold}
        )
