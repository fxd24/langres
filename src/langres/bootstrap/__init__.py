"""Cold-start gold-set bootstrapping (M1).

Public surface for turning raw blocker candidates into a labeled
:class:`GoldSet`:

- Data contract: :class:`GoldPair`, :class:`GoldSet`.
- Interfaces: :class:`Miner`, :class:`Labeler`.
- Miners: :class:`HardNegativeMiner` (stratified sampling).
- Labelers: :class:`GroundTruthLabeler` (zero-spend, deterministic),
  :class:`TeacherLabeler` (budget-capped LLM teacher).
"""

from langres.bootstrap.base import Labeler, Miner
from langres.bootstrap.labelers import BlindCostError, GroundTruthLabeler, TeacherLabeler
from langres.bootstrap.miners import HardNegativeMiner
from langres.bootstrap.models import GoldPair, GoldSet

__all__ = [
    "BlindCostError",
    "GoldPair",
    "GoldSet",
    "GroundTruthLabeler",
    "HardNegativeMiner",
    "Labeler",
    "Miner",
    "TeacherLabeler",
]
