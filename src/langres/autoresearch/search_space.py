"""The declarative search space the autoresearch loop enumerates.

The autoresearch loop is ``propose → run → evaluate → keep-if-better`` (epic
#145). A :class:`SearchSpace` is the *proposal substrate*: it names the candidate
values per pipeline parameter and enumerates their Cartesian product as plain
config dicts (``dict[str, Any]``) that the loop turns into runnable blockers via
``langres.autoresearch.factory``.

**Import-light by design.** This module is pure stdlib + typing (``itertools`` /
``dataclasses``) — it constructs no blocker and imports no
faiss/torch/sentence-transformers, so it can sit on the public API surface (users
build a ``SearchSpace`` to call ``langres.optimize``) without pulling the heavy
[semantic] stack into a bare ``import langres`` (see ``tests/test_import_budget.py``).
The heavy construction lives in the sibling ``factory`` module, which this one
never imports.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """A declarative Cartesian grid of blocker configs for the autoresearch loop.

    Each field holds the candidate values for one pipeline parameter as a tuple;
    :meth:`configs` yields their Cartesian product as config dicts. The defaults
    describe a small vector-blocker ``k`` sweep (the one axis usually swept) with
    a MiniLM + cosine baseline.

    Attributes:
        blocker: Blocker kinds to try (``"vector"`` and/or ``"all_pairs"``). The
            vector-specific axes below are ignored by ``"all_pairs"``.
        embedding_model: sentence-transformers model names for the vector index.
        metric: FAISS distance metrics (``"L2"`` / ``"cosine"``).
        text_field: Record attribute names holding each record's blocking text.
            Dataset-specific — override the default to match your schema's field.
        k_neighbors: Nearest-neighbour counts to sweep. The **innermost** axis of
            :meth:`configs` (see its ordering contract).
        query_prompt: Instruction prefixes to try on the **query** side, with
            ``None`` meaning "no instruction" (the symmetric default). Instructional
            embedders (EmbeddingGemma / E5 / BGE / Qwen3-Embedding) are trained to
            read a task prefix, so this is a real quality axis for them and a
            no-op-but-costly one for models that ignore prompts. Documents stay
            generic either way; a prompt costs one extra encode pass over the
            corpus per search, which is why it is not free to leave on.
            In :meth:`configs` this axis varies **outside** ``k_neighbors`` so the
            index is still built once per index-defining group (the prompt affects
            queries, not the index) — the *field* order below is unrelated to that.
    """

    blocker: tuple[str, ...] = ("vector",)
    embedding_model: tuple[str, ...] = ("all-MiniLM-L6-v2",)
    metric: tuple[str, ...] = ("cosine",)
    text_field: tuple[str, ...] = ("name",)
    k_neighbors: tuple[int, ...] = (5, 10, 20)
    # LAST on purpose. A new field in a non-terminal position silently re-binds
    # every positional caller: `SearchSpace(("vector",), ("m",), ("cosine",),
    # ("name",), (5, 10, 20))` would hand that k tuple to `query_prompt`, which
    # __post_init__ accepts (it only checks emptiness) and which surfaces much
    # later as `encode(prompt=5)`. Field order is independent of the
    # itertools.product order in configs(), so the innermost-k contract is
    # unaffected by keeping this last.
    query_prompt: tuple[str | None, ...] = (None,)

    def __post_init__(self) -> None:
        # Fail loud: an empty axis would silently collapse the whole product to
        # zero configs, so the loop would run nothing without ever erroring.
        for f in fields(self):
            if not getattr(self, f.name):
                raise ValueError(f"SearchSpace.{f.name} must be a non-empty tuple")

    def configs(self) -> Iterator[dict[str, Any]]:
        """Yield the Cartesian product of the axes as config dicts.

        **Ordering contract (relied on by the loop):** ``k_neighbors`` is the
        **innermost** varying dimension, so consecutive configs hold
        ``(blocker, embedding_model, metric, text_field, query_prompt)`` fixed
        while ``k`` varies across its full range before any outer axis advances.
        This lets the downstream loop build **one** vector index per
        ``(embedding_model, metric, text_field)`` and reuse it across every
        ``k`` value (``k`` lives on the blocker, not the index), instead of
        re-embedding the corpus for each ``k``.

        ``query_prompt`` sits **outside** ``k`` and **inside** the three
        index-defining axes on purpose: it changes how *queries* are encoded, not
        what is indexed, so it never invalidates the cached index — but it does
        change results at every ``k``, so it must not be innermost either.

        Yields:
            One ``dict[str, Any]`` per grid point, with keys ``blocker``,
            ``embedding_model``, ``metric``, ``text_field``, ``query_prompt``,
            ``k_neighbors`` (in that order).
        """
        # itertools.product varies its LAST argument fastest, so listing
        # k_neighbors last makes it the innermost dimension (the contract above).
        for blocker, embedding_model, metric, text_field, query_prompt, k in itertools.product(
            self.blocker,
            self.embedding_model,
            self.metric,
            self.text_field,
            self.query_prompt,
            self.k_neighbors,
        ):
            yield {
                "blocker": blocker,
                "embedding_model": embedding_model,
                "metric": metric,
                "text_field": text_field,
                "query_prompt": query_prompt,
                "k_neighbors": k,
            }

    def __len__(self) -> int:
        """The number of configs :meth:`configs` yields (the product of axis sizes)."""
        return (
            len(self.blocker)
            * len(self.embedding_model)
            * len(self.metric)
            * len(self.text_field)
            * len(self.query_prompt)
            * len(self.k_neighbors)
        )
