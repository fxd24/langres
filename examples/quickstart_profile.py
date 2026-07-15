"""Quickstart: profile a dataset -- the data, not the pipeline -- for free.

Before you judge a single pair, the data itself has a story: how rare a true match
is (class imbalance), how clean the fields are, whether *any* signal separates
matches from non-matches, and how your embeddings are distributed. This example
builds a :class:`~langres.core.data_profile.DataProfileReport` over a toy corpus +
gold clustering + two (synthetic) embedding models, then shows the **text-first**
surfaces (``print``/markdown/dict/rows) and writes a single self-contained
``data_profile.html`` tearsheet -- a KPI hero, label structure, string- and
cosine-separability charts, per-field stats, and side-by-side embedding norms.

Fully offline on purpose: no API key, no network, no torch, no
sentence-transformers. The two embedding sources are plain in-memory
:class:`~langres.core.data_profile.ArraySource` matrices (what a
``VectorBlocker``'s corpus, an ``AnchorStore``, or a ``.npy`` on disk would give
you). The report *consumes* precomputed vectors -- it never generates them -- so
this whole quickstart runs on a bare core-only install.

Run it:
    uv run python examples/quickstart_profile.py
"""

from pathlib import Path

import numpy as np

from langres.core.data_profile import ArraySource, DataProfileReport
from langres.core.models import CompanySchema

# A toy cross-source corpus and its true clustering (what a small labeled sample
# gives you). Four entities: three matched pairs and one singleton.
records = [
    {"id": "1", "name": "Acme Corporation"},
    {"id": "2", "name": "Acme Corp"},
    {"id": "3", "name": "Globex Inc"},
    {"id": "4", "name": "Globex Incorporated"},
    {"id": "5", "name": "Initech"},
    {"id": "6", "name": "Initech LLC"},
    {"id": "7", "name": "Unrelated Bakery"},
]
gold_clusters = [{"1", "2"}, {"3", "4"}, {"5", "6"}, {"7"}]

# Two synthetic embedding models over the SAME record ids. Each record's vector is
# its cluster's base direction plus noise, so within-cluster cosine runs high --
# giving the cosine-separability chart something real to show. Different dims (8 vs
# 16) and scales make the side-by-side norm comparison (and its dims-differ caveat)
# render. No sentence-transformers involved: these are plain matrices.
rng = np.random.default_rng(0)
ids = [record["id"] for record in records]
cluster_of = {rid: ci for ci, cluster in enumerate(gold_clusters) for rid in cluster}


def _synthetic_matrix(dim: int, scale: float) -> np.ndarray:
    bases = {ci: rng.normal(size=dim) for ci in range(len(gold_clusters))}
    rows = [(bases[cluster_of[rid]] + rng.normal(scale=0.35, size=dim)) * scale for rid in ids]
    return np.asarray(rows)


embeddings = [
    ArraySource("mini-8d", ids, _synthetic_matrix(dim=8, scale=1.0)),
    ArraySource("large-16d", ids, _synthetic_matrix(dim=16, scale=2.5)),
]

# BUILD: compose the default section set (hero -> labels -> separability ->
# fields -> embeddings -> comparison). Omitting embeddings= would simply drop the
# embedding sections -- no error. schema= enables the rapidfuzz string signal.
report = DataProfileReport.from_records(
    records,
    gold=gold_clusters,
    schema=CompanySchema,
    embeddings=embeddings,
)

# 1. TEXT-FIRST: print the whole report as markdown (the default way to read it).
print(report)

# 2. HEADLINE NUMBERS: a flat dict you can log or assert on.
print("\nsummary:", report.summary)

# 3. TABULAR: any section's rows() feed pd.DataFrame(...) with no pandas dep.
label = report["Label structure"]
print("\nlabel-structure rows:", label.rows())

# 4. TEARSHEET (nice-to-have): one self-contained HTML file, $0, no server.
out_path = Path("data_profile.html")
out_path.write_text(report.to_html(title="langres data profile"), encoding="utf-8")
print(f"\nWrote {out_path.resolve()} -- open it in a browser (no server needed).")
