"""Scorer Modules (judges) for the M0 Resolver.

A judge is a :class:`~langres.core.module.Module` that turns candidate pairs
into :class:`~langres.core.models.PairwiseJudgement` scores. Wave 2a ships the
:class:`~langres.core.judges.weighted_average.WeightedAverageJudge`; M3 adds the
zero-spend :class:`~langres.core.judges.embedding_score.EmbeddingScoreJudge`.

Importing both here fires their ``@register`` decorators on package import, so a
fresh process can ``Resolver.load`` an artifact using either judge.
"""

from langres.core.judges.embedding_score import EmbeddingScoreJudge
from langres.core.judges.weighted_average import WeightedAverageJudge

__all__ = ["EmbeddingScoreJudge", "WeightedAverageJudge"]
