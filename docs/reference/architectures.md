# Architectures

Named ER models. You construct one, then call `.dedupe()` / `.compare()` on it:

```python
from langres.architectures import FuzzyString, VectorLLMCascade

FuzzyString().dedupe(records)                                   # $0, offline, no key

VectorLLMCascade(
    embedder="BAAI/bge-base-en-v1.5",
    llm="openrouter/deepseek-v4",
).dedupe(records)                                               # paid, because you named it
```

An **architecture** is a *topology* — which components run, in what order. A
**backbone** is what fills a model slot. Swapping a backbone never mints a new
architecture: `VectorLLMCascade(llm="a")` and `VectorLLMCascade(llm="b")` are the
same architecture with different weights behind it.

There is no `matcher="auto"`. Nothing here reads your environment to decide what
to run: `FuzzyString` has no paid slot to fill, and `VectorLLMCascade` only bills
you because you named an `llm=`. Choosing the model is your job, not a
heuristic's.

::: langres.architectures.fuzzy_string

::: langres.architectures.vector_llm_cascade
