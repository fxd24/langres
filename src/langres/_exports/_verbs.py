"""The front door: the two verbs, their presets, and the exceptions to catch.

See ``langres._exports`` for the fragment contract.
"""

from langres.clients.openrouter import BudgetExceeded
from langres.core.models import MatcherAbstainedError
from langres.core.presets import DEFAULT_AUTO_MODEL, NoMatcherAvailableError
from langres.verbs import LinkVerdict, dedupe, link

__all__ = [
    "BudgetExceeded",
    "DEFAULT_AUTO_MODEL",
    "dedupe",
    "link",
    "LinkVerdict",
    "MatcherAbstainedError",
    "NoMatcherAvailableError",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
