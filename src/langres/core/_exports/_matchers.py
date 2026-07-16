"""The ``Matcher`` (judge) ABC and the concrete matcher family.

See ``langres.core._exports`` for the fragment contract.
"""

from typing import TYPE_CHECKING

from langres.core.matcher import GroupwiseMatcher, Matcher, stamp_group_cost
from langres.core.matchers.cascade_judge import CascadeMatcher
from langres.core.matchers.embedding_score import EmbeddingScoreMatcher
from langres.core.matchers.fellegi_sunter import FellegiSunterMatcher
from langres.core.matchers.weighted_average import WeightedAverageMatcher

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy names visible to `mypy --strict`
    # without pulling litellm/dspy/scikit-learn into a bare `import langres`.
    from langres.core.matchers.llm_judge import LLMMatcher
    from langres.core.matchers.random_forest_judge import RandomForestMatcher
    from langres.core.matchers.select_judge import SelectMatcher

__all__ = [
    "CascadeMatcher",
    "EmbeddingScoreMatcher",
    "FellegiSunterMatcher",
    "GroupwiseMatcher",
    "Matcher",
    "stamp_group_cost",
    "WeightedAverageMatcher",
]

LAZY_SUBMODULES: tuple[str, ...] = ()

LAZY_SYMBOLS: dict[str, str] = {
    "LLMMatcher": "langres.core.matchers.llm_judge",
    "RandomForestMatcher": "langres.core.matchers.random_forest_judge",
    "SelectMatcher": "langres.core.matchers.select_judge",
}

EXTRA_BY_SYMBOL: dict[str, str] = {
    "LLMMatcher": "llm",
    "RandomForestMatcher": "trained",
    "SelectMatcher": "llm",
}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SUBMODULES, *LAZY_SYMBOLS)
