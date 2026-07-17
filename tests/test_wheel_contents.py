"""Licence gate: the published artifacts must ship only data we may redistribute.

`[tool.hatch.build].exclude` in `pyproject.toml` keeps the large third-party
DeepMatcher/Magellan benchmark corpora (vendored for research reproducibility,
with ATTRIBUTION.md but **no explicit redistribution licence**) out of the
PyPI artifacts. Those excludes are 14 hand-written path literals, and their
failure mode is **silent**: rename `src/langres/data/datasets/`, or a dataset
directory under it, and every literal quietly matches nothing. The build then
succeeds with zero warnings and ships the corpora anyway. Measured on this
tree, that is 30 extra CSVs / 13.9 MB -- 87.7% of the resulting wheel.

This has already happened once (a 91%-third-party-data wheel shipped), so this
file is a regression gate, not a hypothetical.

## Why an explicit manifest, and not a derived rule

The redistribution decision is **not derivable from any property of a file**.
Measured against this tree, every mechanical rule is wrong:

* "no `.csv` ships" -- false: 13 CSVs ship deliberately.
* "small CSVs ship" -- false: `wdc_computers/valid.csv` (6,106 B) is excluded
  while `fodors_zagat/fodors.csv` (44,963 B) ships. No threshold separates them.
* "a dir with ATTRIBUTION.md ships nothing" -- false: `tiny_fixture` has one and
  ships all 5 CSVs; `abt_buy`/`amazon_google` have one and ship
  `peeters_sampled_test.csv`.
* "only id/label pair files ship" -- false: `fodors.csv` and `person_a.csv` are
  full record text and ship.

The decision is a per-file human licence judgement, so the expectation has to be
a per-file human declaration.

Deriving the expectation from the excludes themselves (expected = files on disk
minus files matching an exclude pattern) is the tempting alternative and it is
**vacuous**: after a rename the patterns match nothing, so the derived
expectation becomes "everything ships" and the wheel matching it *passes*. An
expectation regenerated from the thing that broke cannot detect it breaking.

So: two *independent* descriptions of one decision -- hatchling's deny patterns
in `pyproject.toml`, and the allow manifest below -- cross-checked against the
real build output. The direction of rot is what matters. A deny list rots
**open** (a stale literal silently ships data); this allow list rots **closed**
(anything the excludes stop catching becomes an unexpected member and fails).
It is not the 14 literals restated: it is their complement, and it fails on
drift the literals cannot see -- a rename, a *new* unlicensed dataset, a deleted
exclude, a build-backend swap.

Keeping the manifest in sync is the intended cost. A new file under
`src/langres/data/` failing this test is the gate working: adding a dataset is
exactly when someone must decide, on the record, whether it may be redistributed.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every non-`.py` file the published artifacts may contain under the `langres`
# package, as a package-relative path. Scoped to non-`.py` because this gate is
# about *data*: `.py` is langres's own source, while a redistributed corpus can
# arrive as .csv today and .parquet/.json/.jsonl tomorrow -- naming the file
# types we ship rather than the ones we fear keeps those out too.
#
# Each entry is a redistribution decision. Do not add one without a reason:
#
#   tiny_fixture/*   -- fully synthetic, authored for langres's test suite.
#                       No third-party data, shares langres's licence.
#   febrl_person/*   -- FEBRL4 subset: synthetic (no PII), BSD-3 (recordlinkage).
#   fodors_zagat/*   -- small classic benchmark vendored from QCRI DeepER.
#   abt_buy/, amazon_google/
#                    -- third-party corpora, NOT redistributable. Only the
#                       ATTRIBUTION.md and `peeters_sampled_test.csv` ship: the
#                       latter is id/label triples with no third-party record
#                       text, so the replication seam's sampling stays
#                       inspectable from a pip install.
#   dblp_acm/, dblp_scholar/, walmart_amazon/, wdc_computers/
#                    -- third-party corpora, NOT redistributable. ATTRIBUTION.md
#                       only; every CSV is excluded. Loading one from a pip
#                       install raises BenchmarkDataNotFoundError pointing at a
#                       git checkout (langres/data/_benchmark_utils.py).
SHIPPED_NON_PY_FILES = frozenset(
    {
        "py.typed",
        # -- third-party corpora: attribution only, no records --------------
        "data/datasets/dblp_acm/ATTRIBUTION.md",
        "data/datasets/dblp_scholar/ATTRIBUTION.md",
        "data/datasets/walmart_amazon/ATTRIBUTION.md",
        "data/datasets/wdc_computers/ATTRIBUTION.md",
        # -- third-party corpora: attribution + id/label pairs only ---------
        "data/datasets/abt_buy/ATTRIBUTION.md",
        "data/datasets/abt_buy/peeters_sampled_test.csv",
        "data/datasets/amazon_google/ATTRIBUTION.md",
        "data/datasets/amazon_google/peeters_sampled_test.csv",
        # -- synthetic, langres-authored ------------------------------------
        "data/datasets/tiny_fixture/ATTRIBUTION.md",
        "data/datasets/tiny_fixture/tableA.csv",
        "data/datasets/tiny_fixture/tableB.csv",
        "data/datasets/tiny_fixture/train.csv",
        "data/datasets/tiny_fixture/valid.csv",
        "data/datasets/tiny_fixture/test.csv",
        # -- synthetic FEBRL4 subset (BSD-3, no PII) ------------------------
        "data/datasets/febrl_person/SOURCE.md",
        "data/datasets/febrl_person/person_a.csv",
        "data/datasets/febrl_person/person_b.csv",
        "data/datasets/febrl_person/person_perfectMapping.csv",
        # -- small classic benchmark (QCRI DeepER) --------------------------
        "data/datasets/fodors_zagat/SOURCE.md",
        "data/datasets/fodors_zagat/fodors.csv",
        "data/datasets/fodors_zagat/zagats.csv",
        "data/datasets/fodors_zagat/fodors-zagats_perfectMapping.csv",
    }
)

# Backstop for bloat the manifest cannot see -- a corpus inlined into a `.py`
# literal, or vendored outside the `langres` package. A proxy, deliberately
# secondary to the exact-contents assertion above, and generous so it never
# bikesheds a legitimate few hundred KB.
#
# Uncompressed, matching the vocabulary of the `[tool.hatch.build]` comment in
# pyproject.toml, and more stable than the compressed size (which moves with
# zlib's behaviour). Measured at the time of writing: 2.16 MB shipping, 16.07 MB
# with the excludes dead.
WHEEL_UNCOMPRESSED_CEILING_BYTES = 4_000_000


@pytest.fixture(scope="session")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build the real sdist + wheel with the same command `publish.yml` runs.

    `uv build` (no target flag) builds the sdist and then the wheel *from* that
    sdist, exactly as the release pipeline does -- so this gate sees what PyPI
    would get, not a wheel built by a different path. ~1.6s with the build env
    warm (`uv sync` in CI warms it); the timeout only guards a cold fetch of the
    build backend.
    """
    if shutil.which("uv") is None:  # pragma: no cover - uv is the repo's paved road
        pytest.fail(
            "`uv` is not on PATH, so the packaging licence gate cannot build the "
            "artifacts it exists to check. Run the suite with `uv run pytest`."
        )

    out_dir = tmp_path_factory.mktemp("dist")
    subprocess.run(
        ["uv", "build", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        timeout=300,
    )
    (wheel,) = out_dir.glob("*.whl")
    (sdist,) = out_dir.glob("*.tar.gz")
    return wheel, sdist


def _hatch_exclude_patterns() -> list[str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    excludes = config["tool"]["hatch"]["build"]["exclude"]
    assert isinstance(excludes, list)
    return excludes


def test_every_exclude_pattern_still_matches_real_files() -> None:
    """No exclude literal may go dead.

    This is the *upstream* half of the gate and it needs no build, so it runs on
    every PR (the wheel assertions below are the ground truth, but `test-full`
    only runs them post-merge). A dead pattern is the exact silent failure this
    file exists for, and catching it here names the offending literal instead of
    reporting a mysteriously fat wheel.

    Simple `pathlib` globbing is faithful for the patterns actually in use
    (a literal path, optionally with `*`). It is not a reimplementation of
    hatchling's matcher -- the artifact assertions below are what hold the line
    if a future pattern needs richer semantics.
    """
    dead = [p for p in _hatch_exclude_patterns() if not any(REPO_ROOT.glob(p))]
    assert not dead, (
        f"{len(dead)} exclude pattern(s) in [tool.hatch.build] match no file on disk:\n"
        + "\n".join(f"  - {p}" for p in dead)
        + "\n\nThese patterns keep unlicensed third-party benchmark data out of the "
        "published wheel/sdist. Matching nothing means they are no longer doing it, "
        "and the build will NOT warn you. If a dataset moved or was renamed, update "
        "the pattern to the new path; if it was deleted, drop the pattern."
    )


def test_wheel_ships_exactly_the_declared_data_files(
    built_artifacts: tuple[Path, Path],
) -> None:
    """The wheel's non-`.py` payload must equal the declared manifest."""
    wheel, _ = built_artifacts
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

    shipped = {
        name[len("langres/") :]
        for name in names
        if name.startswith("langres/") and not name.endswith(".py")
    }
    _assert_manifest(shipped, artifact=wheel.name)


def test_sdist_ships_exactly_the_declared_data_files(
    built_artifacts: tuple[Path, Path],
) -> None:
    """Same invariant for the sdist.

    `publish.yml` runs `uv build` + `uv publish`, which uploads BOTH artifacts,
    and `[tool.hatch.build].exclude` is target-agnostic -- so the sdist carries
    the identical exposure and deserves the identical gate.
    """
    _, sdist = built_artifacts
    marker = "/src/langres/"
    with tarfile.open(sdist) as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]

    # Locating the package by a path marker already fails closed -- if the sdist
    # layout moved, nothing matches and every manifest entry reports as missing.
    # But "23 files missing" would misdiagnose a layout change as a deletion, and
    # a gate whose message points at the wrong cause wastes the incident. Say so.
    assert any(marker in name for name in names), (
        f"{sdist.name} contains no member under '{marker}', so this test cannot see "
        "the package at all. The sdist layout changed (a src-layout move, or a "
        "different build backend) -- update the marker; do NOT assume the data is "
        "simply gone."
    )

    shipped = {
        name.split(marker, 1)[1] for name in names if marker in name and not name.endswith(".py")
    }
    _assert_manifest(shipped, artifact=sdist.name)


def _assert_manifest(shipped: set[str], *, artifact: str) -> None:
    # Both callers filter `.py` out of `shipped`, so a `.py` entry in the manifest
    # could never match and would surface as a bogus "missing file" -- a real
    # licence decision is not being made about langres's own source. Name the
    # actual mistake instead.
    stray_py = {p for p in SHIPPED_NON_PY_FILES if p.endswith(".py")}
    assert not stray_py, (
        f"SHIPPED_NON_PY_FILES lists {len(stray_py)} `.py` path(s): {sorted(stray_py)}. "
        "The manifest declares the non-`.py` data payload only; `.py` files are "
        "langres's own source and are not part of this gate."
    )

    unexpected = shipped - SHIPPED_NON_PY_FILES
    missing = SHIPPED_NON_PY_FILES - shipped
    assert not unexpected, (
        f"{artifact} ships {len(unexpected)} data file(s) that are NOT declared "
        f"redistributable in SHIPPED_NON_PY_FILES:\n"
        + "\n".join(f"  + {p}" for p in sorted(unexpected))
        + "\n\nMost likely an exclude pattern in [tool.hatch.build] stopped matching "
        "(a renamed/moved dataset directory) and third-party benchmark data is now "
        "in the published artifact -- the build does not warn about this. Either fix "
        "the exclude, or, if this file really is ours to redistribute, add it to the "
        "manifest with the licence reason."
    )
    assert not missing, (
        f"{artifact} is MISSING {len(missing)} file(s) the manifest says it ships:\n"
        + "\n".join(f"  - {p}" for p in sorted(missing))
        + "\n\nAn over-broad exclude, or a deleted/renamed fixture. If the removal was "
        "intended, drop the entry from SHIPPED_NON_PY_FILES."
    )


def test_wheel_stays_under_the_size_ceiling(built_artifacts: tuple[Path, Path]) -> None:
    """Catch-all for payload the manifest cannot see (see the ceiling's comment)."""
    wheel, _ = built_artifacts
    with zipfile.ZipFile(wheel) as zf:
        total = sum(info.file_size for info in zf.infolist())

    assert total <= WHEEL_UNCOMPRESSED_CEILING_BYTES, (
        f"{wheel.name} is {total:,} B uncompressed, over the "
        f"{WHEEL_UNCOMPRESSED_CEILING_BYTES:,} B ceiling. Something large is being "
        "published. If it is legitimate, raise the ceiling deliberately."
    )
