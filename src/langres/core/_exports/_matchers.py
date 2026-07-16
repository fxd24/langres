"""The ``Matcher`` (judge) **contract**: the ABCs and the group-cost helper.

Only the contract lives here. The concrete matcher family
(``CascadeMatcher``, ``EmbeddingScoreMatcher``, ``FellegiSunterMatcher``,
``WeightedAverageMatcher``, ``LLMMatcher``, ``RandomForestMatcher``,
``SelectMatcher``) are **implementations**, and ``langres.core`` no longer
re-exports them -- import them from the package that owns them::

    from langres.core.matchers import CascadeMatcher, LLMMatcher
    from langres.core.matchers.fellegi_sunter import FellegiSunterMatcher

``langres.core.matchers`` keeps its own lazy ``__getattr__`` for the names
needing an extra, so those imports stay as import-light as they were here.

See ``langres.core._exports`` for the fragment contract, and
``langres.core.__init__`` for why the facade carries contracts only.
"""

from langres.core.matcher import GroupwiseMatcher, Matcher, stamp_group_cost

__all__ = [
    "GroupwiseMatcher",
    "Matcher",
    "stamp_group_cost",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
