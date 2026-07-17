"""Memory-efficient, read-only access to *precomputed* embedding vectors.

The data-profile report never *produces* embeddings -- it **consumes** vectors a
pipeline already paid to compute (a ``VectorBlocker``'s corpus, an
``AnchorStore``'s built index, a bare ``.npy`` a user hands us). This module is
the seam that exposes those vectors to the embedding sections **without ever
loading the whole corpus into RAM**: an :class:`EmbeddingSource` resolves a batch
of record ids to their rows and gathers *only* those rows, so a 30 GB matrix is
profiled in ``O(batch * dim)`` memory, never ``O(corpus * dim)``.

Three concrete sources, one contract:

- :class:`ArraySource` -- wraps a matrix already in RAM (tests, small corpora).
- :class:`NpySource` -- memmaps a ``.npy`` (``mmap_mode="r"``) and fancy-indexes
  the requested rows, so only those pages fault in. The 30 GB-safe default.
- :meth:`NpySource.from_anchor_store` -- reuses the pipeline's own
  ``corpus_embeddings.npy`` + the :class:`~langres.curation.anchor_store.AnchorStoreManifest`
  id order, reading the metric from ``corpus.json`` (a cosine index is
  pre-normalized, so its norms are all ~1.0 -- a caveat the sections surface).

**No heavy imports.** Module scope is pure standard library + numpy -- never
sentence-transformers / torch / faiss. We read vectors someone else wrote; we do
not build an embedder. The one langres import (the anchor-store manifest schema)
is deferred into :meth:`NpySource.from_anchor_store` so importing this module
stays core-only.

**Guards (a mis-aligned id map corrupts every downstream metric):**

- Unknown ids are **dropped and logged** (a partial miss -- the profile proceeds
  over the rows that resolve).
- A **total** id miss (nothing resolves) **raises** -- that is a namespace
  mismatch (wrong id set entirely), not a profile worth computing.
- A bare ``ndarray`` passed where a source is expected raises an actionable
  :class:`TypeError` (wrap it in :class:`ArraySource`).
- An id-order **fingerprint** guards the vectors<->ids alignment: a length check
  cannot catch a *same-length permutation*, which silently re-maps every vector.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingSource(Protocol):
    """The read-only contract the embedding sections consume vectors through.

    Any object exposing a model ``name``, an embedding ``dim``, and a
    memory-bounded :meth:`vectors_for` is a source -- the sections never see the
    whole matrix, only the rows they ask for. :class:`ArraySource`,
    :class:`NpySource`, and :meth:`NpySource.from_anchor_store` are the shipped
    implementations; the protocol lets a caller supply any other backing (a
    remote store, a lazily-decompressed shard) without this package knowing.
    """

    name: str
    dim: int

    def vectors_for(self, ids: Sequence[Hashable]) -> NDArray[np.floating]:
        """Return the ``(n_resolved, dim)`` rows for ``ids`` in ``O(len(ids) * dim)``.

        Unknown ids are dropped (``n_resolved <= len(ids)``); a total miss (none
        resolve) raises. Memory MUST be bounded by the request, never the corpus.
        """
        ...  # pragma: no cover


# Sidecar written next to an anchor store's vectors on first
# :meth:`NpySource.from_anchor_store`, then compared on every later load: a
# trust-on-first-use guard that catches a same-length id-order permutation.
_FINGERPRINT_SIDECAR = "embedding_ids.fingerprint"

# Filenames the pipeline persists (see ``FAISSIndex.save_state`` /
# ``AnchorStore.save``). Located by name so this module never imports the
# resolver / faiss to read them.
_CORPUS_VECTORS = "corpus_embeddings.npy"
_CORPUS_JSON = "corpus.json"
_ANCHOR_MANIFEST = "anchor_store.json"


def _fingerprint(id_order: Sequence[Hashable]) -> str:
    """Order-sensitive digest of an id sequence.

    Two id orders collide only if they are element-wise identical, so a
    same-length permutation (which a length check misses) yields a *different*
    fingerprint. ``str(id)`` handles any :class:`~collections.abc.Hashable` id
    (``str``/``int``/tuple); the count and a control-char separator prevent
    boundary collisions (``["a", "bc"]`` vs ``["ab", "c"]``).
    """
    digest = hashlib.sha256()
    digest.update(str(len(id_order)).encode("utf-8"))
    for identifier in id_order:
        digest.update(b"\x1f")
        digest.update(str(identifier).encode("utf-8"))
    return digest.hexdigest()


class _RowGatherSource:
    """Shared id->row resolution + memory-safe row gather for both sources.

    Holds an id order (row ``i`` of the matrix belongs to ``id_order[i]``), a
    ``{id: row}`` lookup, and the backing matrix -- an in-memory ``ndarray``
    (:class:`ArraySource`) or a read-only ``np.memmap`` (:class:`NpySource`).
    :meth:`vectors_for` gathers requested rows with a single numpy fancy-index,
    which materialises **only** those rows (``O(len(ids) * dim)``); a memmap
    backing never pages the whole corpus in.

    Not constructed directly -- use :class:`ArraySource` / :class:`NpySource`.

    Attributes:
        name: A short label for the embedding model (shown in the report).
        pre_normalized: ``True`` when the vectors are already unit-norm (a cosine
            index normalizes in place), so a norm distribution is degenerate
            (~1.0) -- the sections render a caveat instead of a flat chart.
        metric: The source metric string when known (e.g. ``"cosine"`` / ``"L2"``),
            else ``None``.
    """

    def __init__(
        self,
        name: str,
        id_order: Sequence[Hashable],
        matrix: NDArray[np.floating] | np.memmap,
        *,
        pre_normalized: bool = False,
        metric: str | None = None,
    ) -> None:
        """Validate the id<->row alignment and build the lookup.

        Raises:
            ValueError: If ``matrix`` is not 2-D, or its row count does not equal
                ``len(id_order)`` (a row/id desync would corrupt every metric).
        """
        if matrix.ndim != 2:
            raise ValueError(
                f"embedding matrix for {name!r} must be 2-D (n_vectors, dim), "
                f"got shape {matrix.shape!r}"
            )
        id_list = list(id_order)
        if len(id_list) != matrix.shape[0]:
            raise ValueError(
                f"id/row desync for source {name!r}: {len(id_list)} ids but "
                f"{matrix.shape[0]} matrix rows -- every vector would be "
                "mis-attributed. Pass exactly one id per row, in row order."
            )
        self.name = str(name)
        self.pre_normalized = bool(pre_normalized)
        self.metric = metric
        self._matrix = matrix
        self._id_order = id_list
        self._row_of: dict[Hashable, int] = {}
        for row, identifier in enumerate(id_list):
            # First-wins on a duplicate id (a later dup cannot be addressed, but
            # dropping it silently beats a corrupt double-mapping).
            self._row_of.setdefault(identifier, row)

    @property
    def dim(self) -> int:
        """Embedding dimensionality (matrix column count)."""
        return int(self._matrix.shape[1])

    @property
    def id_order(self) -> list[Hashable]:
        """A copy of the record ids in row order (row ``i`` -> ``id_order[i]``)."""
        return list(self._id_order)

    @property
    def id_fingerprint(self) -> str:
        """Order-sensitive digest of :attr:`id_order` (see :func:`_fingerprint`)."""
        return _fingerprint(self._id_order)

    def vectors_for(self, ids: Sequence[Hashable]) -> NDArray[np.floating]:
        """Gather the vectors for ``ids``, in request order, dropping unknowns.

        Resolves each id to its row and gathers exactly those rows with one
        fancy-index -- ``O(len(ids) * dim)`` memory, so a memmapped backing pages
        in only the requested rows and the returned array is an **independent**
        copy (never a view onto the corpus).

        Args:
            ids: Record ids to fetch. May be a subset of :attr:`id_order`, in any
                order, with repeats.

        Returns:
            An ``(n_resolved, dim)`` float array, where ``n_resolved <= len(ids)``
            (unknown ids are dropped) and equals ``len(ids)`` when all resolve.

        Raises:
            KeyError: If ``ids`` is non-empty and **none** resolve -- a total
                miss, i.e. the wrong id namespace entirely.
        """
        rows: list[int] = []
        n_missing = 0
        for identifier in ids:
            row = self._row_of.get(identifier)
            if row is None:
                n_missing += 1
            else:
                rows.append(row)
        if n_missing and not rows:
            raise KeyError(
                f"none of the {n_missing} requested id(s) exist in source "
                f"{self.name!r} -- wrong id namespace? (a total miss, not a "
                "partial one, is raised so a mis-pointed profile fails loudly)"
            )
        if n_missing:
            logger.warning(
                "EmbeddingSource %r: dropped %d unknown id(s) of %d requested",
                self.name,
                n_missing,
                len(rows) + n_missing,
            )
        # Fancy-indexing a memmap materialises only these rows into a fresh,
        # independent ndarray (owns its data; not an np.memmap, not a view).
        return np.array(self._matrix[rows])

    def verify_alignment(self, expected_fingerprint: str) -> None:
        """Raise if this source's id order no longer matches ``expected_fingerprint``.

        The alignment guard a length check cannot provide: a same-length
        permutation of the ids (with the vectors unchanged) silently re-maps
        every row, and only a fingerprint comparison catches it.

        Raises:
            ValueError: On a fingerprint mismatch (the id order changed under the
                same vectors).
        """
        actual = self.id_fingerprint
        if actual != expected_fingerprint:
            raise ValueError(
                f"id-order fingerprint mismatch for source {self.name!r}: the "
                "recorded vector<->id alignment no longer holds (a permuted id "
                "order under the same vectors). Rebuild the source from the "
                "current id order."
            )


class ArraySource(_RowGatherSource):
    """An :class:`EmbeddingSource` over a matrix already in memory.

    For tests and small corpora where the vectors comfortably fit in RAM. The
    row/id length check is enforced at construction because a desync corrupts
    every downstream metric.

    Example::

        src = ArraySource("minilm", ["r1", "r2"], np.array([[0.1, 0.2], [0.3, 0.4]]))
        src.vectors_for(["r2"])          # -> shape (1, 2)
    """

    def __init__(
        self,
        name: str,
        ids: Sequence[Hashable],
        matrix: NDArray[np.floating],
        *,
        pre_normalized: bool = False,
        metric: str | None = None,
    ) -> None:
        """Wrap ``matrix`` (rows aligned to ``ids``).

        Args:
            name: Short model label.
            ids: One id per matrix row, in row order.
            matrix: An ``(n_vectors, dim)`` float array.
            pre_normalized: Mark the vectors as already unit-norm.
            metric: Optional metric string (e.g. ``"cosine"``).

        Raises:
            ValueError: If ``matrix`` is not 2-D or ``len(ids) != matrix.shape[0]``.
        """
        super().__init__(
            name, ids, np.asarray(matrix), pre_normalized=pre_normalized, metric=metric
        )


class NpySource(_RowGatherSource):
    """An :class:`EmbeddingSource` over a memmapped ``.npy`` (the 30 GB-safe default).

    Loads the array with ``mmap_mode="r"`` -- the file is **not** read into RAM;
    :meth:`vectors_for` fancy-indexes the requested rows so only those pages
    fault in. The memmap is preserved as the backing store (dropping
    ``mmap_mode`` would silently re-introduce the ``O(corpus)`` load this class
    exists to avoid).

    The id order can be passed explicitly or discovered from a companion JSON
    sidecar (``<stem>.ids.json`` or ``ids.json``) next to the ``.npy``, so a
    saved ``(vectors, ids)`` pair self-aligns.
    """

    def __init__(
        self,
        name: str,
        path: str | Path,
        id_order: Sequence[Hashable] | None = None,
        *,
        pre_normalized: bool = False,
        metric: str | None = None,
    ) -> None:
        """Memmap ``path`` and align it to ``id_order`` (or a sidecar).

        Args:
            name: Short model label.
            path: Path to a ``.npy`` holding an ``(n_vectors, dim)`` float matrix.
            id_order: One id per row, in row order. When ``None``, an adjacent
                ``<stem>.ids.json`` or ``ids.json`` (a JSON list) is loaded.
            pre_normalized: Mark the vectors as already unit-norm.
            metric: Optional metric string.

        Raises:
            ValueError: If ``id_order`` is ``None`` and no sidecar is found, or if
                the matrix is not 2-D / the id count does not match the rows.
        """
        matrix = np.load(Path(path), mmap_mode="r")
        if id_order is None:
            id_order = self._load_sidecar_ids(Path(path))
        super().__init__(name, id_order, matrix, pre_normalized=pre_normalized, metric=metric)

    @staticmethod
    def _load_sidecar_ids(path: Path) -> list[Hashable]:
        """Load a companion ids JSON next to ``path`` (``<stem>.ids.json`` / ``ids.json``).

        Raises:
            ValueError: If neither sidecar exists.
        """
        candidates = [path.with_name(f"{path.stem}.ids.json"), path.with_name("ids.json")]
        for candidate in candidates:
            if candidate.exists():
                loaded = json.loads(candidate.read_text())
                logger.info("NpySource: loaded %d ids from sidecar %s", len(loaded), candidate)
                return list(loaded)
        raise ValueError(
            f"NpySource for {path} needs an id_order: pass one, or place a "
            f"companion ids file ({candidates[0].name} or ids.json) next to the .npy."
        )

    @classmethod
    def from_anchor_store(cls, state_dir: str | Path, name: str) -> NpySource:
        """Build a source from a persisted ``AnchorStore`` / vector-index artifact.

        Reuses the pipeline's own ``corpus_embeddings.npy`` (row order == corpus
        order) and the :class:`~langres.curation.anchor_store.AnchorStoreManifest`
        ``anchor_ids`` (the record-id <-> row order), reading ``metric`` from
        ``corpus.json``. A cosine index L2-normalizes its vectors in place, so the
        source is marked :attr:`~_RowGatherSource.pre_normalized` and the sections
        render the "norms are all ~1.0" caveat rather than a flat chart.

        An id-order fingerprint is persisted next to the artifact on first call
        and compared on every later call, so a same-length permutation of the
        anchor ids (which a length check misses) fails loudly.

        Args:
            state_dir: The persisted artifact directory (an ``AnchorStore.save``
                output, or any dir containing the three sidecar files, possibly
                nested).
            name: Short model label for the source.

        Returns:
            An :class:`NpySource` memmapping the artifact's vectors, aligned to the
            manifest's anchor id order.

        Raises:
            FileNotFoundError: If the vectors or the manifest cannot be located.
            ValueError: On an id-order fingerprint mismatch (a permuted id order).
        """
        # Deferred import keeps this module core-only at import time; the manifest
        # schema itself is import-light (pydantic + core.models, no faiss).
        from langres.curation.anchor_store import AnchorStoreManifest

        root = Path(state_dir)
        vectors_path = _locate(root, _CORPUS_VECTORS)
        manifest_path = _locate(root, _ANCHOR_MANIFEST)
        manifest = AnchorStoreManifest.model_validate_json(manifest_path.read_text())
        anchor_ids: list[Hashable] = list(manifest.anchor_ids)

        metric: str | None = None
        corpus_json = vectors_path.with_name(_CORPUS_JSON)
        if corpus_json.exists():
            metric = json.loads(corpus_json.read_text()).get("metric")

        # Trust-on-first-use alignment guard: persist the fingerprint beside the
        # artifact, then compare it on every later load.
        fingerprint = _fingerprint(anchor_ids)
        fp_path = root / _FINGERPRINT_SIDECAR
        if fp_path.exists():
            previous = fp_path.read_text().strip()
            if previous != fingerprint:
                raise ValueError(
                    f"anchor-store id order changed under the same vectors in "
                    f"{root} -- a same-length permutation would silently re-map "
                    "every embedding. Delete "
                    f"{_FINGERPRINT_SIDECAR} to re-baseline if this is intentional."
                )
        else:
            fp_path.write_text(fingerprint)

        return cls(
            name,
            vectors_path,
            anchor_ids,
            pre_normalized=(metric == "cosine"),
            metric=metric,
        )


def _locate(root: Path, filename: str) -> Path:
    """Find ``filename`` at ``root`` or anywhere beneath it (deterministically).

    ``AnchorStore.save`` nests the vector-index sidecars under a ``resolver/``
    subtree, so a plain ``root / filename`` is not enough; a name search that
    tolerates the nesting (without importing the resolver to learn its layout) is.

    Raises:
        FileNotFoundError: If no file with that name exists under ``root``.
    """
    direct = root / filename
    if direct.exists():
        return direct
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"{filename!r} not found in {root} (searched recursively) -- is this a "
            "persisted anchor-store / vector-index artifact with a built index?"
        )
    if len(matches) > 1:
        # More than one artifact under the same root: the first (sorted) still
        # resolves so this stays non-breaking, but we name the choice instead of
        # picking silently -- matching this module's "never silent" convention.
        logger.warning(
            "%r matched %d files under %s; using the first (%s). Point state_dir at a "
            "single artifact to disambiguate.",
            filename,
            len(matches),
            root,
            matches[0],
        )
    return matches[0]


def _ensure_source(obj: object) -> None:
    """Reject a value that is not an :class:`EmbeddingSource` with a clear message.

    The most likely mistake is passing the raw vector matrix where a *source* is
    expected; that would type-check as an ``ndarray`` and then fail obscurely, so
    it earns a targeted, actionable error.

    Raises:
        TypeError: If ``obj`` is an ``ndarray`` (or otherwise lacks
            ``vectors_for``).
    """
    if isinstance(obj, np.ndarray):
        raise TypeError(
            "expected an EmbeddingSource (ArraySource / NpySource / ...), got a "
            "bare numpy ndarray -- wrap it: ArraySource(name, ids, matrix)."
        )
    if not callable(getattr(obj, "vectors_for", None)):
        raise TypeError(
            f"expected an EmbeddingSource with a vectors_for(ids) method, got {type(obj).__name__}."
        )


def cosine_signal(source: EmbeddingSource) -> Callable[[Hashable, Hashable], float | None]:
    """A pairwise cosine-similarity callable backed by ``source`` (the A<->B bridge).

    Returns ``signal(id_a, id_b) -> float | None`` with the same shape as the
    separability section's similarity signal: it looks up both ids' vectors via
    ``source``, L2-normalizes them, and returns their cosine similarity -- or
    ``None`` when either id is missing or its vector is degenerate (zero-norm or
    non-finite), so a caller can score a stream of pairs without guarding each
    lookup.

    Args:
        source: Any :class:`EmbeddingSource`. A bare ndarray raises a
            :class:`TypeError` (see :func:`_ensure_source`).

    Returns:
        A ``(id_a, id_b) -> float | None`` similarity function.
    """
    _ensure_source(source)

    def signal(id_a: Hashable, id_b: Hashable) -> float | None:
        try:
            pair = source.vectors_for([id_a, id_b])
        except KeyError:
            return None  # neither id resolved (total miss)
        if pair.shape[0] != 2:
            return None  # exactly one resolved -- cannot form a pair
        vec_a = np.asarray(pair[0], dtype=np.float64)
        vec_b = np.asarray(pair[1], dtype=np.float64)
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if not math.isfinite(norm_a) or not math.isfinite(norm_b) or norm_a == 0.0 or norm_b == 0.0:
            return None  # degenerate vector -- cosine undefined
        # Both norms are finite and non-zero, so by Cauchy-Schwarz the dot product
        # is bounded by their product and the ratio is a finite value in [-1, 1].
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    return signal
