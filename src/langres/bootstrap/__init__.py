"""Cold-start gold-set bootstrapping (M1).

Public surface for turning raw blocker candidates into a labeled
:class:`GoldSet`:

- Data contract: :class:`GoldPair`, :class:`GoldSet`.
- Interfaces: :class:`Miner`, :class:`Labeler`.
- Miners: :class:`HardNegativeMiner` (stratified sampling).
- Labelers: :class:`GroundTruthLabeler` (zero-spend, deterministic),
  :class:`FakeLabeler` (zero-spend similarity-threshold teacher stand-in),
  :class:`TeacherLabeler` (budget-capped LLM teacher).
- Orchestrator: :class:`Bootstrapper` (block -> filter -> mine -> label).
- Report: :class:`BootstrapReport` (coverage + calibration health check).
"""

from langres.bootstrap.base import Labeler, Miner
from langres.bootstrap.bootstrapper import Bootstrapper
from langres.bootstrap.labelers import (
    BlindCostError,
    FakeLabeler,
    GroundTruthLabeler,
    TeacherLabeler,
)
from langres.bootstrap.miners import HardNegativeMiner
from langres.bootstrap.models import GoldPair, GoldPairSource, GoldSet
from langres.bootstrap.report import BootstrapReport

__all__ = [
    "BlindCostError",
    "BootstrapReport",
    "Bootstrapper",
    "FakeLabeler",
    "GoldPair",
    "GoldPairSource",
    "GoldSet",
    "GroundTruthLabeler",
    "HardNegativeMiner",
    "Labeler",
    "Miner",
    "TeacherLabeler",
]
