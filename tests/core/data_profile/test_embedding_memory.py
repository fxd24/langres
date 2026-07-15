"""Memory-behaviour test for ``NpySource``: profiling a big matrix stays O(batch).

The whole point of the memmap-backed source is that a 30 GB corpus is never
loaded to gather a handful of rows. This pins that: a ~512 MB ``.npy`` is written
to ``$TMPDIR``, wrapped in an ``NpySource``, and ``vectors_for`` on a 100-row
subset is measured -- in a **subprocess**, so ``tracemalloc``'s heap peak isolates
the gather -- to be a tiny fraction of the file. It also guards the two
regressions that would silently reintroduce the O(corpus) load: dropping
``mmap_mode`` (the backing must stay an ``np.memmap``) and returning a view onto
the corpus instead of an independent O(batch) copy.

Marked ``slow`` (writes ~512 MB); run explicitly with ``-m slow`` or by path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from langres.core.data_profile.embedding_source import NpySource

_N_ROWS = 500_000
_DIM = 256
_SUBSET = 100

# Runs in a fresh interpreter so the tracemalloc peak reflects ONLY the
# vectors_for gather -- not the fixture write, the memmap open, or the id map
# (all allocated before tracemalloc.start()).
_SUBPROCESS = r"""
import json, sys, tracemalloc
import numpy as np
from langres.core.data_profile.embedding_source import NpySource

path, n_rows, dim, subset_n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
src = NpySource("big", path, list(range(n_rows)))
step = max(n_rows // subset_n, 1)
subset = list(range(0, n_rows, step))[:subset_n]

tracemalloc.start()
out = src.vectors_for(subset)
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(json.dumps({
    "peak": peak,
    "backing_is_memmap": isinstance(src._matrix, np.memmap),
    "out_is_memmap": isinstance(out, np.memmap),
    "out_base_is_none": out.base is None,
    "out_shape": list(out.shape),
    "out_nbytes": int(out.nbytes),
}))
"""


def _write_big_npy(path: str) -> int:
    """Write an ``(_N_ROWS, _DIM)`` float32 ``.npy`` in chunks (writer stays O(chunk))."""
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(_N_ROWS, _DIM))
    rng = np.random.default_rng(0)
    chunk = 50_000
    for start in range(0, _N_ROWS, chunk):
        end = min(start + chunk, _N_ROWS)
        matrix[start:end] = rng.standard_normal((end - start, _DIM)).astype(np.float32)
    matrix.flush()
    del matrix
    return os.path.getsize(path)


@pytest.mark.slow
def test_npysource_gather_is_o_batch_not_o_corpus() -> None:
    # $TMPDIR-backed temp dir; auto-removed on exit (cleans up the ~512 MB file).
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "big.npy")
        file_size = _write_big_npy(path)
        assert file_size > 400 * 1024 * 1024  # ~512 MB on disk

        # In-process contract: backing stays a memmap; the gather is an
        # independent, batch-sized array (not a view onto the corpus).
        src = NpySource("big", path, list(range(_N_ROWS)))
        assert isinstance(src._matrix, np.memmap), "mmap_mode dropped -> O(corpus) load"
        out = src.vectors_for(list(range(0, _N_ROWS, _N_ROWS // _SUBSET))[:_SUBSET])
        assert out.shape == (_SUBSET, _DIM)
        assert not isinstance(out, np.memmap)
        assert out.base is None  # owns its data; independent of the memmap
        out[0, 0] = 12345.0  # mutating the copy must not touch the file
        assert np.load(path, mmap_mode="r")[0, 0] != 12345.0
        del src, out

        # Subprocess: isolate the heap peak of vectors_for on a 100-row subset.
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS, path, str(_N_ROWS), str(_DIM), str(_SUBSET)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout.strip().splitlines()[-1])

        assert result["backing_is_memmap"] is True
        assert result["out_is_memmap"] is False
        assert result["out_base_is_none"] is True
        assert result["out_shape"] == [_SUBSET, _DIM]

        # The gather touches ~100*256*4 ~= 100 KB; the file is ~512 MB. The heap
        # peak must be a tiny fraction of the file (never an O(corpus) load).
        batch_bytes = _SUBSET * _DIM * 4
        assert result["peak"] < 0.05 * file_size
        assert result["peak"] < batch_bytes + 20 * 1024 * 1024  # generous headroom
