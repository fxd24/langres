"""CompositeBlocker implementation: set algebra over child blockers.

Composes 2+ child :class:`~langres.core.blocker.Blocker` instances' candidate
pair-sets via a set operation (``"union"`` / ``"intersection"`` /
``"difference"``), deduping by the canonical undirected pair key with
first-seen semantics (the same guarantee every base blocker already gives:
each unordered pair appears at most once). ``"union"`` is the recall-maximizing
default -- composing a cheap, high-precision :class:`~langres.core.blockers.
key.KeyBlocker` with a recall-oriented :class:`~langres.core.blockers.vector.
VectorBlocker` recovers the recall a single blocking key alone would miss.
"""

from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, Literal

from langres.core.blocker import Blocker, SchemaT
from langres.core.models import ERCandidate
from langres.core.registry import get_component, register
from langres.core.reports import CandidateInspectionReport

CompositeOp = Literal["union", "intersection", "difference"]

_VALID_OPS: tuple[CompositeOp, ...] = ("union", "intersection", "difference")


def _pair_key(candidate: ERCandidate[Any]) -> frozenset[str]:
    """Canonical undirected pair key for dedup (matches the base blockers' guarantee)."""
    return frozenset((candidate.left.id, candidate.right.id))


@register("composite_blocker")
class CompositeBlocker(Blocker[SchemaT]):
    """Set algebra (union/intersection/difference) over child blockers.

    Each child is fully materialized into a ``{pair_key: candidate}`` map (set
    operations need to know a pair's full membership across children, so this
    -- like :func:`~langres.core.groups.derive_groups_from_pairs` -- is
    necessarily buffered, unlike the streaming-first single-blocker contract).
    The retained ``ERCandidate`` for a kept pair is the first child's own
    candidate object (preserving its ``left``/``right`` orientation), with
    ``blocker_name`` replaced to carry provenance: which child(ren) actually
    produced that specific pair.

    - ``"union"`` (default, recall-maximizing): every pair produced by ANY
      child.
    - ``"intersection"``: only pairs produced by EVERY child.
    - ``"difference"``: pairs produced by the FIRST child and by no other
      child (``children[0] - (children[1] | children[2] | ...)``).

    ``stream_groups()`` is NOT overridden: it relies on the inherited
    ``Blocker`` default (buffered derivation from ``stream()``), which is
    already proven pairs-equivalent for any blocker (see
    ``test_blocker_stream_groups_default_pairs_equivalence_property`` and this
    module's own equivalence tests). CompositeBlocker's own pair generation
    already fully buffers every child, and its pair sets (especially for
    intersection/difference across heterogeneous children) have no natural
    single "anchor" structure, so a native per-anchor override -- the
    treatment ``VectorBlocker`` gets, per W1.0 -- would add complexity without
    a corresponding benefit here.

    Example:
        # Recall-first: cheap key blocking backstopped by semantic search.
        composite = CompositeBlocker(
            children=[
                KeyBlocker(schema=CompanySchema, key_field="postal_code"),
                VectorBlocker(schema=CompanySchema, text_field="name", ...),
            ],
            op="union",
        )
        candidates = list(composite.stream(company_records))

    Note:
        Serialization (``config``/``from_config``) only supports children
        whose own ``config`` is plain data (e.g. ``KeyBlocker``,
        ``AllPairsBlocker``, another ``CompositeBlocker`` of such children). A
        child with out-of-band state (e.g. a built ``VectorBlocker`` index)
        is not preserved through a composite's save/load round-trip -- persist
        such a pipeline via the ``Resolver`` artifact instead, which already
        handles that state explicitly.
    """

    # Registry key, mirrored as a class attribute so the Resolver's uniform
    # serialization helper can discover the type name (see resolver.py).
    type_name: ClassVar[str] = "composite_blocker"

    def __init__(self, children: Sequence[Blocker[SchemaT]], op: CompositeOp = "union"):
        """Initialize CompositeBlocker.

        Args:
            children: 2+ child blockers to compose. All must normalize to the
                same entity schema (they compose over the same input ``data``).
            op: The set operation: ``"union"`` (default), ``"intersection"``,
                or ``"difference"`` (first child minus the rest).

        Raises:
            ValueError: If fewer than 2 children are given, or ``op`` is not
                one of ``"union"``, ``"intersection"``, ``"difference"``.
        """
        if len(children) < 2:
            raise ValueError(
                f"CompositeBlocker requires at least 2 child blockers (got {len(children)})."
            )
        if op not in _VALID_OPS:
            raise ValueError(f"Unknown composite op {op!r}; expected one of {_VALID_OPS}.")

        self.children = list(children)
        self.op: CompositeOp = op

    def _label(self, child: Blocker[SchemaT]) -> str:
        """Provenance label for a child: its registry type_name, or class name."""
        return str(getattr(child, "type_name", type(child).__name__))

    def _child_pair_maps(self, data: list[Any]) -> list[dict[frozenset[str], ERCandidate[SchemaT]]]:
        """Fully materialize each child's candidates into a pair-key -> candidate map."""
        maps: list[dict[frozenset[str], ERCandidate[SchemaT]]] = []
        for child in self.children:
            pair_map: dict[frozenset[str], ERCandidate[SchemaT]] = {}
            for candidate in child.stream(data):
                key = _pair_key(candidate)
                if key not in pair_map:
                    pair_map[key] = candidate
            maps.append(pair_map)
        return maps

    def _keep(self, key: frozenset[str], maps: list[dict[frozenset[str], Any]]) -> bool:
        """Whether ``key`` survives this composite's set operation."""
        if self.op == "union":
            return True  # reached only if present in at least one map
        if self.op == "intersection":
            return all(key in pm for pm in maps)
        # difference: in the first child, and in no other child.
        return key in maps[0] and not any(key in pm for pm in maps[1:])

    def stream(self, data: list[Any]) -> Iterator[ERCandidate[SchemaT]]:
        """Generate candidate pairs by combining children via ``self.op``.

        Args:
            data: List of raw data items, passed through unchanged to every
                child's ``stream()``.

        Yields:
            ERCandidate[SchemaT] objects surviving the set operation, each
            pair exactly once, in first-seen order across children (in
            ``self.children`` order, then each child's own stream order).
            ``blocker_name`` is rewritten to
            ``"composite_{op}(label1+label2+...)"`` listing the child(ren)
            that actually produced that pair.
        """
        maps = self._child_pair_maps(data)
        labels = [self._label(child) for child in self.children]

        seen: set[frozenset[str]] = set()
        for pair_map in maps:
            for key, candidate in pair_map.items():
                if key in seen:
                    continue
                seen.add(key)
                if not self._keep(key, maps):
                    continue
                contributing = [label for pm, label in zip(maps, labels, strict=True) if key in pm]
                name = f"composite_{self.op}({'+'.join(contributing)})"
                yield candidate.model_copy(update={"blocker_name": name})

    @property
    def config(self) -> dict[str, object]:
        """Serializable construction config for the registry.

        Returns:
            ``{"op": ..., "children": [{"type_name": ..., "config": ...}, ...]}``.

        Raises:
            ValueError: If any child has no registry ``type_name``, or any
                child's own ``config`` raises (e.g. it was built from a
                non-serializable callable).
        """
        child_specs: list[dict[str, object]] = []
        for child in self.children:
            child_type_name = getattr(child, "type_name", None)
            if child_type_name is None:
                raise ValueError(
                    f"CompositeBlocker child {child!r} has no registry 'type_name'; "
                    "construct with a registered Blocker subclass to persist."
                )
            child_config: Any = child.config  # type: ignore[attr-defined]
            child_specs.append({"type_name": child_type_name, "config": child_config})
        return {"op": self.op, "children": child_specs}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "CompositeBlocker[SchemaT]":
        """Rebuild a CompositeBlocker from its serialized config.

        Args:
            config: A mapping with ``"op"`` and ``"children"`` (see
                :attr:`config`).

        Returns:
            A blocker equivalent to the one that produced ``config``.
        """
        children: list[Blocker[SchemaT]] = []
        child_specs: Any = config["children"]
        for spec in child_specs:
            child_cls: Any = get_component(str(spec["type_name"]))
            children.append(child_cls.from_config(spec["config"]))
        return cls(children=children, op=config["op"])  # type: ignore[arg-type]

    def inspect_candidates(
        self,
        candidates: list[ERCandidate[SchemaT]],
        entities: list[SchemaT],
        sample_size: int = 10,
    ) -> CandidateInspectionReport:
        """Explore CompositeBlocker candidates without ground truth.

        Args:
            candidates: List of generated candidate pairs.
            entities: Original list of entities.
            sample_size: Number of example pairs to include in report.

        Returns:
            CandidateInspectionReport with totals, per-entity distribution,
            samples, and a recommendation naming the composite's op.
        """
        n = len(entities)
        total_candidates = len(candidates)
        avg_candidates_per_entity = (2 * total_candidates / n) if n > 0 else 0.0

        counts: dict[str, int] = {str(getattr(e, "id", id(e))): 0 for e in entities}
        for cand in candidates:
            left_id = str(getattr(cand.left, "id", id(cand.left)))
            right_id = str(getattr(cand.right, "id", id(cand.right)))
            counts[left_id] = counts.get(left_id, 0) + 1
            counts[right_id] = counts.get(right_id, 0) + 1

        distribution: dict[str, int] = {}
        for count in counts.values():
            key = str(count)
            distribution[key] = distribution.get(key, 0) + 1

        examples: list[dict[str, str]] = []
        for cand in candidates[:sample_size]:
            examples.append(
                {
                    "left_id": str(getattr(cand.left, "id", id(cand.left))),
                    "right_id": str(getattr(cand.right, "id", id(cand.right))),
                    "left_text": self._extract_text(cand.left),
                    "right_text": self._extract_text(cand.right),
                }
            )

        recommendations: list[str] = [
            f"CompositeBlocker(op={self.op!r}) over {len(self.children)} children "
            f"generated {total_candidates:,} candidates "
            f"(avg {avg_candidates_per_entity:.1f} per entity, n={n}). "
            + (
                "'union' maximizes recall at the cost of more candidates; "
                "'intersection'/'difference' trade recall for precision -- "
                "measure Pair-Completeness against a labeled sample before relying on them."
                if self.op == "union"
                else "Verify Pair-Completeness against a labeled sample: "
                "'intersection'/'difference' can silently drop true matches "
                "that only one child's blocking strategy would have found."
            )
        ]

        return CandidateInspectionReport(
            total_candidates=total_candidates,
            avg_candidates_per_entity=avg_candidates_per_entity,
            candidate_distribution=distribution,
            examples=examples,
            recommendations=recommendations,
        )

    def _extract_text(self, entity: SchemaT) -> str:
        """Extract human-readable text from entity."""
        if hasattr(entity, "name"):
            return str(entity.name)
        return str(entity)
