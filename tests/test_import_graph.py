"""Tests for `tools/import_graph.py`.

This tool exists to justify architectural decisions, so it has to be trustworthy
itself. The load-bearing test here is `test_edges_match_grimp_exactly`: grimp is the
engine behind import-linter, so matching it edge-for-edge is what makes the tool's
numbers admissible as evidence. The originals in `tmp/w0-graph/` failed exactly this
(they credited `from pkg import submodule` to the package) -- hence this suite.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

from import_graph import (  # noqa: E402
    DEFAULT_PACKAGE_ROOT,
    ImportKind,
    MappingError,
    PackageMapping,
    _checked_section,
    build_graph,
    counterfactual,
    main,
)


@pytest.fixture(scope="module")
def graph():
    """The real langres graph -- built once, it is pure parsing and side-effect free."""
    return build_graph()


def _write_pkg(root: Path, name: str, files: dict[str, str]) -> Path:
    """Materialise a tiny fixture package and return its root."""
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for filename, source in files.items():
        target = pkg / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    return pkg


# --------------------------------------------------------------------------------------
# The AST classifier: toplevel vs function-local vs TYPE_CHECKING
# --------------------------------------------------------------------------------------


def test_classifies_toplevel_function_local_and_type_checking(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "leaf.py": "VALUE = 1\n",
            "user.py": (
                "from typing import TYPE_CHECKING\n"
                "import fixt.leaf\n"  # toplevel
                "if TYPE_CHECKING:\n"
                "    from fixt import other\n"  # type-checking
                "def go():\n"
                "    from fixt import deep\n"  # function-local
                "    return deep\n"
            ),
            "other.py": "",
            "deep.py": "",
        },
    )
    graph = build_graph(pkg)
    kinds = {e.imported: e.kind for e in graph.edges if e.importer == "fixt.user"}

    assert kinds["fixt.leaf"] is ImportKind.TOPLEVEL
    assert kinds["fixt.other"] is ImportKind.TYPE_CHECKING
    assert kinds["fixt.deep"] is ImportKind.FUNCTION_LOCAL


def test_nearest_enclosing_construct_wins(tmp_path):
    """TYPE_CHECKING inside a function is type-checking; a nested def stays local."""
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "a.py": "",
            "b.py": "",
            "user.py": (
                "from typing import TYPE_CHECKING\n"
                "def outer():\n"
                "    if TYPE_CHECKING:\n"
                "        from fixt import a\n"
                "    def inner():\n"
                "        from fixt import b\n"
                "    return inner\n"
            ),
        },
    )
    kinds = {e.imported: e.kind for e in build_graph(pkg).edges}
    assert kinds["fixt.a"] is ImportKind.TYPE_CHECKING
    assert kinds["fixt.b"] is ImportKind.FUNCTION_LOCAL


def test_type_checking_else_branch_runs_at_runtime(tmp_path):
    """The `else` of an `if TYPE_CHECKING` block executes -- it is a toplevel import."""
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "a.py": "",
            "b.py": "",
            "user.py": (
                "from typing import TYPE_CHECKING\n"
                "if TYPE_CHECKING:\n"
                "    from fixt import a\n"
                "else:\n"
                "    from fixt import b\n"
            ),
        },
    )
    kinds = {e.imported: e.kind for e in build_graph(pkg).edges}
    assert kinds["fixt.a"] is ImportKind.TYPE_CHECKING
    assert kinds["fixt.b"] is ImportKind.TOPLEVEL


@pytest.mark.parametrize(
    ("header", "condition", "expected"),
    [
        # The two real spellings of a type-checking guard.
        ("from typing import TYPE_CHECKING", "TYPE_CHECKING", ImportKind.TYPE_CHECKING),
        ("import typing", "typing.TYPE_CHECKING", ImportKind.TYPE_CHECKING),
        # `not TYPE_CHECKING` inverts the guard: this body RUNS at runtime.
        ("from typing import TYPE_CHECKING", "not TYPE_CHECKING", ImportKind.TOPLEVEL),
        # Lookalikes a substring/subtree search would wrongly swallow.
        ("MY_TYPE_CHECKING_FLAG = True", "MY_TYPE_CHECKING_FLAG", ImportKind.TOPLEVEL),
        ("x = 'other'", "x == 'TYPE_CHECKING'", ImportKind.TOPLEVEL),
    ],
)
def test_only_a_real_type_checking_guard_marks_its_body_lazy(header, condition, expected, tmp_path):
    """A guard is matched structurally, not by searching the test for a string.

    Every TOPLEVEL row is one the old `"TYPE_CHECKING" in ast.dump(test)` check got
    wrong, and each error pointed the same dangerous way: a real runtime import
    filed as lazy, i.e. hidden from the runtime view the tool exists to report.
    """
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {"a.py": "", "user.py": f"{header}\nif {condition}:\n    from fixt import a\n"},
    )
    kinds = {e.imported: e.kind for e in build_graph(pkg).edges}
    assert kinds["fixt.a"] is expected


# --------------------------------------------------------------------------------------
# The edge resolver
# --------------------------------------------------------------------------------------


def test_resolves_symbol_import_to_its_module(graph):
    """`from langres.core.matchers.llm_judge import X` -> the llm_judge module itself."""
    assert "langres.core.matchers.llm_judge" in graph.modules
    imports = {e.imported for e in graph.edges if e.importer == "langres.core.matchers.llm_judge"}
    # It imports symbols out of core.models; the edge lands on the module, never a symbol.
    assert "langres.core.models" in imports
    assert not any(i.startswith("langres.core.models.") for i in imports)


def test_deep_symbol_target_walks_up_to_nearest_real_module(tmp_path):
    """A target naming no module resolves to its nearest real ancestor."""
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "leaf.py": "class Thing:\n    Nested = 1\n",
            "user.py": "from fixt.leaf.Thing import Nested\n",
        },
    )
    edges = {(e.importer, e.imported) for e in build_graph(pkg).edges}
    assert ("fixt.user", "fixt.leaf") in edges


def test_from_package_import_submodule_credits_the_submodule(tmp_path):
    """The bug that made the ad-hoc scripts inadmissible, pinned as a regression test."""
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "sub/__init__.py": "",
            "sub/thing.py": "NAME = 1\n",
            "user.py": "from fixt.sub import thing\n",
        },
    )
    edges = {(e.importer, e.imported) for e in build_graph(pkg).edges}
    assert ("fixt.user", "fixt.sub.thing") in edges
    assert ("fixt.user", "fixt.sub") not in edges


def test_mixed_from_import_splits_module_and_symbol_targets(tmp_path):
    """`from pkg import submodule, Symbol` -> one edge each, to submodule and to pkg."""
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "sub/__init__.py": "Symbol = 1\n",
            "sub/thing.py": "",
            "user.py": "from fixt.sub import thing, Symbol\n",
        },
    )
    edges = {(e.importer, e.imported) for e in build_graph(pkg).edges}
    assert ("fixt.user", "fixt.sub.thing") in edges
    assert ("fixt.user", "fixt.sub") in edges


def test_relative_imports_resolve(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "leaf.py": "",
            "sub/__init__.py": "",
            "sub/user.py": "from ..leaf import thing\nfrom . import sibling\n",
            "sub/sibling.py": "",
        },
    )
    edges = {(e.importer, e.imported) for e in build_graph(pkg).edges}
    assert ("fixt.sub.user", "fixt.leaf") in edges
    assert ("fixt.sub.user", "fixt.sub.sibling") in edges


def test_third_party_and_self_imports_are_excluded(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {"user.py": "import os\nimport numpy\nimport fixt.user\nfrom fixt import user\n"},
    )
    assert build_graph(pkg).edges == ()


def test_missing_package_root_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        build_graph(tmp_path / "nope")


# --------------------------------------------------------------------------------------
# Ground truth: the tool must agree with grimp, the engine behind import-linter
# --------------------------------------------------------------------------------------


def test_edges_match_grimp_exactly(graph):
    """Edge-for-edge agreement with grimp on the real package.

    grimp cannot report import *kind*, which is why this tool exists -- but for the
    edge set itself grimp is ground truth, and any divergence is a bug here.
    """
    grimp = pytest.importorskip("grimp")
    built = grimp.build_graph("langres", include_external_packages=False)
    theirs = {
        (importer, imported)
        for importer in built.modules
        for imported in built.find_modules_directly_imported_by(importer)
    }
    # grimp includes namespace/parent modules langres itself does not ship as files.
    theirs = {(a, b) for a, b in theirs if a in graph.modules and b in graph.modules}
    assert graph.edge_pairs() == theirs


# --------------------------------------------------------------------------------------
# Graph queries
# --------------------------------------------------------------------------------------


def test_fan_in_and_fan_out_count_distinct_modules(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "leaf.py": "",
            "a.py": "from fixt.leaf import x\nfrom fixt.leaf import y\n",  # 2 stmts, 1 edge
            "b.py": "import fixt.leaf\n",
        },
    )
    graph = build_graph(pkg)
    assert graph.fan_in()["fixt.leaf"] == 2
    assert graph.fan_out()["fixt.a"] == 1
    assert len(graph.importers_of("fixt.leaf")) == 3  # three statements


def test_kind_filter_separates_runtime_from_lazy_edges(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "leaf.py": "",
            "a.py": "def go():\n    from fixt import leaf\n    return leaf\n",
        },
    )
    graph = build_graph(pkg)
    assert graph.edge_pairs([ImportKind.TOPLEVEL]) == set()
    assert graph.edge_pairs([ImportKind.FUNCTION_LOCAL]) == {("fixt.a", "fixt.leaf")}


def test_sccs_find_a_mutual_import_cycle(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "a.py": "def go():\n    from fixt import b\n",
            "b.py": "from fixt import a\n",
        },
    )
    graph = build_graph(pkg)
    assert graph.sccs() == [["fixt.a", "fixt.b"]]
    # Only the lazy edge closes the loop, so at runtime there is no cycle at all.
    assert graph.sccs([ImportKind.TOPLEVEL]) == []


# --------------------------------------------------------------------------------------
# The counterfactual
# --------------------------------------------------------------------------------------


def test_package_mapping_precedence_exact_over_longest_prefix():
    mapping = PackageMapping(
        exact={"pkg.a.special": "carved_out"},
        prefix={"pkg": "broad", "pkg.a": "narrow"},
    )
    assert mapping.target("pkg.a.special") == "carved_out"  # exact wins
    assert mapping.target("pkg.a.other") == "narrow"  # longest prefix wins
    assert mapping.target("pkg.zzz") == "broad"
    assert mapping.target("pkg.andnot") == "broad"  # prefix must match on a dot boundary
    assert mapping.target("elsewhere") is None


def test_counterfactual_reports_cross_package_cycle_and_unmapped(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        "fixt",
        {
            "a.py": "from fixt import b\n",
            "b.py": "from fixt import a\n",
            "lonely.py": "",
        },
    )
    graph = build_graph(pkg)
    mapping = PackageMapping(exact={"fixt.a": "left", "fixt.b": "right"}, prefix={})
    result = counterfactual(graph, mapping)

    assert result.cycles == [["left", "right"]]
    assert result.cycle_count == 1
    assert result.mutual_pairs == [("left", "right")]
    assert {k: len(v) for k, v in result.cross_edges.items()} == {
        ("left", "right"): 1,
        ("right", "left"): 1,
    }
    assert "fixt.lonely" in result.unmapped
    assert "fixt" in result.unmapped


def test_counterfactual_ignores_intra_package_edges(tmp_path):
    pkg = _write_pkg(tmp_path, "fixt", {"a.py": "from fixt import b\n", "b.py": ""})
    graph = build_graph(pkg)
    mapping = PackageMapping(exact={}, prefix={"fixt": "one"})
    result = counterfactual(graph, mapping)
    assert result.cycles == []
    assert result.mutual_pairs == []
    assert result.cross_edges == {}
    assert result.unmapped == []


def test_mapping_round_trips_through_json(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"exact": {"a.b": "x"}, "prefix": {"a": "y"}}))
    mapping = PackageMapping.from_json(path)
    assert mapping.target("a.b") == "x"
    assert mapping.target("a.c") == "y"


def test_shipped_refactor_mapping_covers_every_module(graph):
    """'No wave may discover a homeless file' -- the mapping must be total."""
    mapping = PackageMapping.from_json(TOOLS / "refactor_target_packages.json")
    assert mapping.unmapped(graph.modules) == []


# --------------------------------------------------------------------------------------
# Mapping validation: a bad mapping must fail loudly, never mis-report confidently
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("{not json", "not valid JSON"),
        ('["exact", "prefix"]', "top level must be a JSON object"),
        ('{"prefix": {}}', "missing required key 'exact'"),
        ('{"exact": {}}', "missing required key 'prefix'"),
        ('{"exact": [], "prefix": {}}', "'exact' must be a JSON object"),
        ('{"exact": {}, "prefix": "pkg"}', "'prefix' must be a JSON object"),
        ('{"exact": {"a": 1}, "prefix": {}}', "must be string -> string"),
        ('{"exact": {"": "pkg"}, "prefix": {}}', "empty module or package name"),
        ('{"exact": {"a": "  "}, "prefix": {}}', "empty module or package name"),
    ],
)
def test_malformed_mapping_is_rejected(payload, expected, tmp_path):
    """Each payload previously yielded an empty/partial mapping -- and a confident
    wrong cycle count from the counterfactual, with nothing to signal it."""
    path = tmp_path / "m.json"
    path.write_text(payload)
    with pytest.raises(MappingError, match=re.escape(expected)):
        PackageMapping.from_json(path)


def test_non_string_mapping_key_is_rejected():
    """`json.loads` cannot produce a non-string key, so this guards the constructor
    invariant rather than a reachable file -- checked directly, at the seam."""
    with pytest.raises(MappingError, match=re.escape("must be string -> string")):
        _checked_section({1: "pkg"}, Path("m.json"), "exact")


def test_a_valid_mapping_with_extra_top_level_keys_loads():
    """The shipped mapping carries a `_comment`; unknown keys stay allowed."""
    mapping = PackageMapping.from_json(TOOLS / "refactor_target_packages.json")
    assert mapping.exact and mapping.prefix


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["fan", "--top", "3"],
        ["fan", "--top", "3", "--toplevel-only"],
        ["kinds"],
        ["importers", "langres.core.benchmark"],
        ["importers", "langres.not_a_module"],
        ["cycles"],
        ["cycles", "--toplevel-only"],
    ],
)
def test_cli_subcommands_run(argv, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out.strip()


def test_cli_counterfactual_runs(capsys):
    argv = [
        "counterfactual",
        "--mapping",
        str(TOOLS / "refactor_target_packages.json"),
        "--show-edges",
        "2",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "cross-package cycles" in out
    assert "mutual package pairs" in out


def test_cli_reports_a_bad_mapping_cleanly(tmp_path, capsys):
    """A bad mapping is user input: a clean stderr message + nonzero exit, no traceback.

    stdout stays empty -- an error must not contaminate the stream a caller reads
    as the tool's data.
    """
    path = tmp_path / "m.json"
    path.write_text('{"prefix": {}}')
    assert main(["counterfactual", "--mapping", str(path)]) == 2
    captured = capsys.readouterr()
    assert "missing required key 'exact'" in captured.err
    assert captured.out == ""


def test_cli_accepts_a_custom_package_root(tmp_path, capsys):
    pkg = _write_pkg(tmp_path, "fixt", {"leaf.py": "", "a.py": "from fixt import leaf\n"})
    assert main(["--package-root", str(pkg), "fan"]) == 0
    assert "fixt.leaf" in capsys.readouterr().out


def test_default_package_root_points_at_langres():
    assert DEFAULT_PACKAGE_ROOT.is_dir()
    assert (DEFAULT_PACKAGE_ROOT / "__init__.py").is_file()
