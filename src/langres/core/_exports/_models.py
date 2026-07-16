"""Data contracts: the records, the judgement, the comparison feature bag.

See ``langres.core._exports`` for the fragment contract.
"""

from langres.core.feature import ComparisonLevel, ComparisonVector, FeatureSpec
from langres.core.groups import ERCandidateGroup, derive_groups_from_pairs
from langres.core.models import (
    CompanySchema,
    EntityProtocol,
    ERCandidate,
    MatcherAbstainedError,
    PairwiseJudgement,
    predicted_match,
)

__all__ = [
    "CompanySchema",
    "ComparisonLevel",
    "ComparisonVector",
    "derive_groups_from_pairs",
    "EntityProtocol",
    "ERCandidate",
    "ERCandidateGroup",
    "FeatureSpec",
    "MatcherAbstainedError",
    "PairwiseJudgement",
    "predicted_match",
]

LAZY_SUBMODULES: tuple[str, ...] = ()
LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SUBMODULES, *LAZY_SYMBOLS)
