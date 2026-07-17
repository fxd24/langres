"""The front door: the models you name, their results, and the exceptions to catch.

Was ``_verbs.py``, which exported ``link``/``dedupe``/``DEFAULT_AUTO_MODEL``/
``NoMatcherAvailableError``. W4 deleted all four: the verbs became methods on the
model (:meth:`~langres.core.resolver.ERModel.dedupe` /
:meth:`~langres.core.resolver.ERModel.compare`), and the ``matcher="auto"``
key-sniffing that produced the other two is gone rather than relocated.

See ``langres._exports`` for the fragment contract.
"""

from langres.architectures import FuzzyString, VectorLLMCascade
from langres.clients.openrouter import BudgetExceeded
from langres.core.models import MatcherAbstainedError
from langres.core.resolver import ERModel
from langres.core.results import DedupeResult, LinkVerdict

__all__ = [
    "BudgetExceeded",
    "DedupeResult",
    "ERModel",
    "FuzzyString",
    "LinkVerdict",
    "MatcherAbstainedError",
    "VectorLLMCascade",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
