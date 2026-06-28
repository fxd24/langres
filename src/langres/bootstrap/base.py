"""Abstract interfaces for cold-start gold-set bootstrapping (M1).

Two tools cooperate to turn raw blocker candidates into a labeled
:class:`~langres.bootstrap.models.GoldSet`:

- A :class:`Miner` selects *which* candidate pairs are worth labeling (e.g.
  hard-negative / stratified sampling) — it neither labels nor spends.
- A :class:`Labeler` assigns a match / non-match label to candidate pairs,
  emitting one :class:`~langres.bootstrap.models.GoldPair` per candidate. The
  label may come from a teacher LLM, benchmark ground truth, or a human.

These are plain ABCs, NOT Resolver slot components: they are not registered via
``@register`` and carry no serializable ``config`` — bootstrapping is an offline
data-preparation step, not a slot in the resolve pipeline.

Each ABC declares exactly one honest signature. Subclasses must implement it
without widening (Liskov): a caller holding a ``Miner``/``Labeler`` reference
must be able to rely on the contract here regardless of the concrete class.
"""

from abc import ABC, abstractmethod
from typing import Any

from langres.bootstrap.models import GoldPair
from langres.core.models import ERCandidate


class Miner(ABC):
    """Selects candidate pairs worth labeling from a larger candidate pool."""

    @abstractmethod
    def mine(
        self,
        candidates: list[ERCandidate[Any]],
        *,
        max_pairs: int | None = None,
    ) -> list[ERCandidate[Any]]:
        """Return the subset of ``candidates`` worth labeling.

        Args:
            candidates: The candidate pairs to mine from (e.g. blocker output).
            max_pairs: Optional cap on the number of returned pairs. ``None``
                means "no cap" — return every selected pair.

        Returns:
            A deduplicated list of selected candidate pairs.
        """


class Labeler(ABC):
    """Assigns match / non-match labels to candidate pairs."""

    @abstractmethod
    def label(self, candidates: list[ERCandidate[Any]]) -> list[GoldPair]:
        """Label each candidate pair, returning one :class:`GoldPair` per input.

        Implementations may return *fewer* pairs than given (e.g. when a budget
        cap or a per-pair failure drops some), but never more, and never widen
        this signature.

        Args:
            candidates: The candidate pairs to label.

        Returns:
            The labeled pairs.
        """
