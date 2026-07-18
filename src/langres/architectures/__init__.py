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
**Each architecture is one self-contained, readable file, and DRY is
deliberately suspended inside this package** -- the policy transformers applies
to ``modeling_*.py``, for the same reason. An architecture file is meant to be
*read end to end* by someone deciding whether this is the model they want, and
then copied as the starting point for their own. A shared ``_build_blocker()``
helper would save a few lines and cost exactly the property that makes these
files worth having: you would have to read three files to learn what one model
does, and a change made for one architecture would silently reshape the others.
So ``fuzzy_string.py`` and ``vector_llm_cascade.py`` both build their own
comparator and their own text extractor, on purpose.

**The anti-DRY licence stops at the package boundary.** It covers *topology* --
which components this model wires and how. It does **not** cover contracts:
input normalization (:mod:`langres.core.inputs`), the result types
(:mod:`langres.core.results`), the spend cap, the model registry and the
``ERModel`` base are shared, DRY, and must stay that way. Every architecture has
to normalize a record identically; that is exactly what a contract is for.
"""

from langres.architectures.fuzzy_string import FuzzyString
from langres.architectures.reranker import Reranker
from langres.architectures.vector_llm_cascade import VectorLLMCascade

__all__ = ["FuzzyString", "Reranker", "VectorLLMCascade"]
