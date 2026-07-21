"""Explicit topology authoring and slot-neutral execution contracts."""

from langres.core.op import (
    ClusterStage,
    ExecutionCheckpoint,
    ExecutionEvent,
    ExecutionObserver,
    ExecutionObserverError,
    ExecutionPlan,
    ExecutionResult,
    ExecutionStep,
    Feasible,
    Finalize,
    Op,
    Score,
    Select,
    Sequential,
    Source,
    SpendMonitorBindable,
    ThresholdSelect,
    TopKSelect,
)
from langres.core.registry import register_op

__all__ = [
    "ClusterStage",
    "ExecutionEvent",
    "ExecutionCheckpoint",
    "ExecutionObserver",
    "ExecutionObserverError",
    "ExecutionPlan",
    "ExecutionResult",
    "ExecutionStep",
    "Feasible",
    "Finalize",
    "Op",
    "register_op",
    "Score",
    "Select",
    "Sequential",
    "Source",
    "SpendMonitorBindable",
    "ThresholdSelect",
    "TopKSelect",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
