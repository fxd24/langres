"""What an :class:`~langres.core.resolver.ERModel` hands back: the two result types.

Split from the judgement contract in :mod:`langres.core.models` on purpose.
``models.py`` describes what a *matcher emits* (a
:class:`~langres.core.models.PairwiseJudgement` per pair); this module describes
what a *model returns* to the caller (:meth:`~langres.core.resolver.ERModel.compare`
-> :class:`LinkVerdict`, :meth:`~langres.core.resolver.ERModel.dedupe` ->
:class:`DedupeResult`). Different producers, different consumers, one job each.

Both are **self-describing**, and the vocabulary is the refactor's (W4):

- ``architecture`` -- the model class that ran (``"FuzzyString"``). The topology.
  Before W4 this field was ``judge_used`` and named a *string preset* resolved
  from an env var; now it names the class the caller constructed themselves.
- ``backbone`` -- what filled the model slot (an LLM id, an embedder name), or
  ``None`` when nothing with weights ran (pure string similarity). Swapping a
  backbone must never change ``architecture`` -- that is the invariant the two
  fields exist to make *visible*, and
  ``TestProof3BackboneSwapKeepsArchitectureIdentity`` in
  ``tests/architectures/test_w4_proofs.py`` pins it.

This module is import-light by construction (pydantic + ``core.models`` only, no
heavy extras) and imports nothing that imports it back.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from langres.core.models import PairwiseJudgement

__all__ = ["DedupeResult", "LinkVerdict"]


class LinkVerdict(BaseModel):
    """The result of :meth:`~langres.core.resolver.ERModel.compare`: one pair, decided.

    Truthy iff :attr:`match` (``if model.compare(a, b): ...``); ``repr`` shows the
    verdict, the score and the architecture for a friendly REPL/notebook read.
    """

    match: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = None

    #: The model class that decided this pair (e.g. ``"FuzzyString"``) -- the
    #: topology, named by the caller at construction rather than sniffed from the
    #: environment.
    architecture: str

    #: The underlying model that actually scored: the LLM id for an LLM-backed
    #: matcher, the sentence-transformers embedder for an embedding-backed one,
    #: and ``None`` for pure string similarity (nothing with weights ran). Never
    #: invisible, never fabricated -- a matcher that reports no model id reports
    #: ``None`` here.
    backbone: str | None = None

    score_type: str

    #: The effective match cut this verdict was decided at -- the model's own
    #: clusterer threshold. Feed it to ``select_for_review(threshold=...)``
    #: instead of remembering the float.
    threshold: float

    judgement: PairwiseJudgement

    def __bool__(self) -> bool:
        return self.match

    def __repr__(self) -> str:
        verdict = "MATCH" if self.match else "NO MATCH"
        score = "n/a" if self.score is None else f"{self.score:.3f}"
        return f"LinkVerdict({verdict}, score={score}, architecture={self.architecture!r})"


class DedupeResult(list[set[str]]):
    """The clusters :meth:`~langres.core.resolver.ERModel.dedupe` returns.

    A plain ``list[set[str]]`` -- identical to what
    :meth:`~langres.core.resolver.ERModel.resolve` hands back -- that additionally
    carries what ran (:attr:`architecture`, :attr:`backbone`), what its scores mean
    (:attr:`score_type`), and the cut they were thresholded at (:attr:`threshold`),
    so a caller can feed ``threshold`` straight to
    :func:`~langres.curation.review.select_for_review` without a remembered constant.

    :attr:`threshold` is ``None`` only for the ``len(records) < 2`` short-circuit,
    where no pair exists and nothing was ever scored.
    """

    def __init__(
        self,
        clusters: Iterable[set[str]],
        *,
        architecture: str,
        backbone: str | None,
        score_type: str,
        threshold: float | None,
    ) -> None:
        super().__init__(clusters)
        self.architecture = architecture
        self.backbone = backbone
        self.score_type = score_type
        self.threshold = threshold

    def __repr__(self) -> str:
        return (
            f"DedupeResult({list.__repr__(self)}, architecture={self.architecture!r}, "
            f"backbone={self.backbone!r}, score_type={self.score_type!r}, "
            f"threshold={self.threshold!r})"
        )
