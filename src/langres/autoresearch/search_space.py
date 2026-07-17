"""The declarative search space the autoresearch loop enumerates.

The autoresearch loop is ``propose → run → evaluate → keep-if-better`` (epic
#145). A :class:`SearchSpace` is the *proposal substrate*: it names the candidate
values per pipeline parameter and enumerates their Cartesian product as plain
config dicts (``dict[str, Any]``) that the loop turns into runnable blockers via
``langres.optimize.factory``.

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
    """

    blocker: tuple[str, ...] = ("vector",)
    embedding_model: tuple[str, ...] = ("all-MiniLM-L6-v2",)
    metric: tuple[str, ...] = ("cosine",)
    text_field: tuple[str, ...] = ("name",)
    k_neighbors: tuple[int, ...] = (5, 10, 20)

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
        ``(blocker, embedding_model, metric, text_field)`` fixed while ``k``
        varies across its full range before any outer axis advances. This lets
        the downstream loop build **one** vector index per
        ``(embedding_model, metric, text_field)`` and reuse it across every
        ``k`` value (``k`` lives on the blocker, not the index), instead of
        re-embedding the corpus for each ``k``.

        Yields:
            One ``dict[str, Any]`` per grid point, with keys ``blocker``,
            ``embedding_model``, ``metric``, ``text_field``, ``k_neighbors`` (in
            that order).
        """
        # itertools.product varies its LAST argument fastest, so listing
        # k_neighbors last makes it the innermost dimension (the contract above).
        for blocker, embedding_model, metric, text_field, k in itertools.product(
            self.blocker,
            self.embedding_model,
            self.metric,
            self.text_field,
            self.k_neighbors,
        ):
            yield {
                "blocker": blocker,
                "embedding_model": embedding_model,
                "metric": metric,
                "text_field": text_field,
                "k_neighbors": k,
            }

    def __len__(self) -> int:
        """The number of configs :meth:`configs` yields (the product of axis sizes)."""
        return (
            len(self.blocker)
            * len(self.embedding_model)
            * len(self.metric)
            * len(self.text_field)
            * len(self.k_neighbors)
        )
