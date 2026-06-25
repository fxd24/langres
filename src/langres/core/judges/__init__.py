"""Scorer Modules (judges) for the M0 Resolver.

A judge is a :class:`~langres.core.module.Module` that turns candidate pairs
into :class:`~langres.core.models.PairwiseJudgement` scores. Wave 2a ships the
:class:`~langres.core.judges.weighted_average.WeightedAverageJudge`.
"""

from langres.core.judges.weighted_average import WeightedAverageJudge

__all__ = ["WeightedAverageJudge"]
