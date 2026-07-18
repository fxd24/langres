"""Named ER architectures: whole pipelines you construct by name.

This package is what W4 is *for*. Before it, nothing in langres named a whole ER
pipeline: ``link()``/``dedupe()`` took a ``matcher=`` string x a ``model=``
string, and ``matcher="auto"`` sniffed the environment for an API key and spent
money on whatever it found -- which cost real money, more than once. The fix is
not a better heuristic. It is making the model **explicit, constructed, and
inspectable**::

    from langres.architectures import FuzzyString, VectorLLMCascade

    FuzzyString().dedupe(records)                          # $0, offline, no key
    VectorLLMCascade(llm="openrouter/...").dedupe(records) # paid, because you said so

The vocabulary, used precisely
------------------------------
- **Architecture** = the *topology*: which components, in what order. A new
  topology is a **new class** in this package.
- **Backbone** = what fills one model slot, named by a
  :class:`~langres.core.model_ref.ModelRef`. Swapping a backbone must **never**
  mint a new architecture -- ``VectorLLMCascade(llm=A)`` and
  ``VectorLLMCascade(llm=B)`` are the same architecture, differently equipped.

If you find yourself adding a boolean that changes which components get built,
that is a new architecture wearing a flag. Write the new class.

Why these files repeat each other (and why that is not a bug)
-------------------------------------------------------------
**Topology stays readable in one place, and DRY is deliberately suspended for
the wiring itself** -- the policy transformers applies to ``modeling_*.py``, for
the same reason. Standalone architectures have self-contained files; the four
closely related retrieval recipes are co-located as one readable family, but
each concrete class still spells out its own ordered operations. A shared
``_build_pipeline(flags=...)`` helper would save a few lines and hide the
property that makes these models worth naming: which stages actually run.
Likewise, ``fuzzy_string.py`` and ``vector_llm_cascade.py`` both build their own
comparator and text extractor on purpose.

**The anti-DRY licence stops at the package boundary.** It covers *topology* --
which components this model wires and how. It does **not** cover contracts:
input normalization (:mod:`langres.core.inputs`), the result types
(:mod:`langres.core.results`), the spend cap, the model registry and the
``ERModel`` base are shared, DRY, and must stay that way. Every architecture has
to normalize a record identically; that is exactly what a contract is for.
"""

import importlib
from typing import TYPE_CHECKING, Any

from langres.architectures.fuzzy_string import FuzzyString
from langres.architectures.reranker import Reranker
from langres.architectures.vector_llm_cascade import VectorLLMCascade

if TYPE_CHECKING:
    from langres.architectures.retrieval import (
        Retrieve,
        RetrieveLLM,
        RetrieveRerank,
        RetrieveRerankLLM,
    )

_LAZY_SYMBOLS = {
    name: "langres.architectures.retrieval"
    for name in ("Retrieve", "RetrieveLLM", "RetrieveRerank", "RetrieveRerankLLM")
}

__all__ = [
    "FuzzyString",
    "Reranker",
    "Retrieve",
    "RetrieveLLM",
    "RetrieveRerank",
    "RetrieveRerankLLM",
    "VectorLLMCascade",
]


def __getattr__(name: str) -> Any:
    """Resolve research recipes without loading their resource adapters eagerly."""
    module_name = _LAZY_SYMBOLS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
