"""Versioned data models for cold-start gold-set bootstrapping (M1).

A *gold set* is a collection of labeled record pairs (match / non-match) used to
seed entity resolution before any human labeling exists: labels can come from a
teacher LLM, from a benchmark's ground truth, or from a human reviewer.

These are PLAIN, versioned Pydantic data models -- not Resolver slot components.
They are NOT part of the ``SerializableState`` protocol and are NOT registered
via ``@register``: they carry no behavior and no nested components, just data.
Persist a :class:`GoldSet` with :meth:`GoldSet.save` (``model_dump_json``) and
reload it with :meth:`GoldSet.load` (``model_validate_json``).
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

GoldPairSource = Literal["teacher", "ground_truth", "human"]
"""Where a gold label came from: a teacher LLM, benchmark ground truth, or a human."""


class GoldPair(BaseModel):
    """A single labeled record pair in a gold set.

    Attributes:
        left_id: Globally-unique id of the left record.
        right_id: Globally-unique id of the right record.
        label: ``True`` if the pair is a match, ``False`` otherwise.
        source: Provenance of the label (``"teacher"``, ``"ground_truth"``,
            or ``"human"``).
        confidence: Optional label confidence in ``[0, 1]`` (e.g. a teacher
            model's probability). ``None`` when not applicable.
        reasoning: Optional free-text rationale for the label (e.g. a teacher
            model's explanation).
        provenance: Free-form metadata about how the label was produced
            (model name, prompt id, timestamps, cost, ...).
    """

    left_id: str
    right_id: str
    label: bool
    source: GoldPairSource
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = None
    provenance: dict[str, object] = Field(default_factory=dict)


class GoldSet(BaseModel):
    """A versioned collection of labeled pairs plus run metadata.

    Serialize with :meth:`save` and reload with :meth:`load`. The
    ``schema_version`` field guards forward compatibility: bump it when the
    on-disk shape changes so loaders can branch on it.

    Attributes:
        schema_version: On-disk schema version. Defaults to ``"1"``.
        pairs: The labeled record pairs.
        metadata: Run-level metadata -- dataset name, blocker config, teacher
            model, ``total_cost_usd``, label counts, etc.
    """

    schema_version: str = "1"
    pairs: list[GoldPair]
    metadata: dict[str, object] = Field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Write the gold set to ``path`` as indented UTF-8 JSON.

        Args:
            path: Destination file path. Parent directories must already exist,
                otherwise a :class:`FileNotFoundError` is raised.
        """
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GoldSet":
        """Load a gold set previously written by :meth:`save`.

        Args:
            path: Path to a JSON file produced by :meth:`save`.

        Returns:
            The validated :class:`GoldSet`.
        """
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
