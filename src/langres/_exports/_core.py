"""The core primitives surfaced at the root: the ``Resolver`` + its data contracts.

See ``langres._exports`` for the fragment contract.
"""

from typing import TYPE_CHECKING

from langres.core import CompanySchema, ERCandidate, PairwiseJudgement, Resolver

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy name visible to `mypy --strict`
    # without pulling litellm into a bare `import langres`.
    # The F401 below is deliberate: "unused" is the point. The name is
    # re-exported lazily via LAZY_SYMBOLS, not by this import; deleting it
    # would blind `mypy --strict` to the lazy symbol.
    from langres.core.matchers.llm_judge import LLMMatcher  # noqa: F401

__all__ = [
    "CompanySchema",
    "ERCandidate",
    "PairwiseJudgement",
    "Resolver",
]

#: The LLM matcher (serve a fine-tuned model_ref, a vLLM api_base, or a paid
#: judge). Importing the class pulls litellm -> the [llm] extra, so it stays
#: lazy: a bare `import langres` never touches litellm.
LAZY_SYMBOLS: dict[str, str] = {
    "LLMMatcher": "langres.core.matchers.llm_judge",
}

EXTRA_BY_SYMBOL: dict[str, str] = {
    "LLMMatcher": "llm",
}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
