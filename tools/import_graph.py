"""AST import-graph tool for a Python package.

Builds a module-level import graph by parsing source with `ast` -- no imports are
executed, so heavy optional dependencies (torch, litellm, faiss) are never loaded.

**Why not just use grimp?** `grimp` (the engine behind `import-linter`) resolves
imports authoritatively, and this tool is cross-validated against it
(`tests/test_import_graph.py`). But grimp reports *edges*, not the *kind* of each
import: it cannot distinguish a module-level import from a function-local one. That
distinction is the whole point here -- langres deliberately uses function-local and
`TYPE_CHECKING` imports to keep `import langres` free of heavy dependencies (see
`tests/test_import_budget.py`), while grimp counts those lazy imports as real edges.
Telling the two views apart is what this tool adds.

Edge resolution matches grimp:

* ``from pkg import submodule`` -> edge to ``pkg.submodule`` (NOT to ``pkg``).
* ``from pkg.mod import Symbol`` -> edge to ``pkg.mod`` (``Symbol`` is not a module).
* A target that names no real module walks up to its nearest real ancestor.

Usage::

    uv run python tools/import_graph.py fan --top 25
    uv run python tools/import_graph.py kinds
    uv run python tools/import_graph.py importers langres.core.benchmark
    uv run python tools/import_graph.py cycles
    uv run python tools/import_graph.py counterfactual --mapping tools/refactor_target_packages.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import networkx as nx

# The package root this repo cares about, relative to the repo root. Derived, never
# hardcoded to an absolute path -- the tool must run from any checkout/worktree.
DEFAULT_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "langres"


class ImportKind(StrEnum):
    """How an import is reached at runtime.

    Only `TOPLEVEL` imports execute on ``import <module>``. The other two are
    invisible at runtime import but *are* counted as edges by grimp/import-linter.
    """

    TOPLEVEL = "toplevel"
    FUNCTION_LOCAL = "function-local"
    TYPE_CHECKING = "type-checking"


@dataclass(frozen=True, slots=True)
class Edge:
    """One resolved import of a first-party module by another."""

    importer: str
    imported: str
    kind: ImportKind
    lineno: int


@dataclass(frozen=True, slots=True)
class ImportGraph:
    """A parsed, resolved module-level import graph."""

    package: str
    modules: dict[str, Path]
    edges: tuple[Edge, ...]

    def edge_pairs(self, kinds: Iterable[ImportKind] | None = None) -> set[tuple[str, str]]:
        """Unique ``(importer, imported)`` pairs, optionally restricted to `kinds`."""
        keep = frozenset(kinds) if kinds is not None else frozenset(ImportKind)
        return {(e.importer, e.imported) for e in self.edges if e.kind in keep}

    def fan_in(self, kinds: Iterable[ImportKind] | None = None) -> dict[str, int]:
        """Number of distinct modules importing each module."""
        counts: dict[str, int] = defaultdict(int)
        for _, imported in self.edge_pairs(kinds):
            counts[imported] += 1
        return counts

    def fan_out(self, kinds: Iterable[ImportKind] | None = None) -> dict[str, int]:
        """Number of distinct modules each module imports."""
        counts: dict[str, int] = defaultdict(int)
        for importer, _ in self.edge_pairs(kinds):
            counts[importer] += 1
        return counts

    def importers_of(self, module: str) -> list[Edge]:
        """Every edge pointing at `module`, sorted by importer then line."""
        return sorted(
            (e for e in self.edges if e.imported == module),
            key=lambda e: (e.importer, e.lineno),
        )

    def sccs(self, kinds: Iterable[ImportKind] | None = None) -> list[list[str]]:
        """Strongly connected components with >1 member, largest first."""
        graph = nx.DiGraph(sorted(self.edge_pairs(kinds)))
        found = (sorted(c) for c in nx.strongly_connected_components(graph) if len(c) > 1)
        return sorted(found, key=lambda c: (-len(c), c))


def _module_name(path: Path, package_root: Path) -> str:
    """Map a source path to its dotted module name (``__init__.py`` -> its package)."""
    rel = path.relative_to(package_root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _discover_modules(package_root: Path) -> dict[str, Path]:
    """Every ``.py`` module under `package_root`, keyed by dotted module name."""
    return {
        _module_name(p, package_root): p
        for p in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def _resolve(target: str, modules: dict[str, Path]) -> str | None:
    """Walk `target` up to the nearest real module, or None if it names nothing."""
    candidate = target
    while candidate and candidate not in modules:
        candidate = candidate.rsplit(".", 1)[0] if "." in candidate else ""
    return candidate or None


def _absolute_module(node: ast.ImportFrom, importer: str, is_package: bool) -> str | None:
    """Resolve an ``ImportFrom``'s module to an absolute name, handling ``from .`` levels."""
    if not node.level:
        return node.module
    base = importer.split(".")
    pkg = base if is_package else base[:-1]
    up = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
    return ".".join(up + ([node.module] if node.module else []))


def _is_type_checking_test(test: ast.expr) -> bool:
    """True if an ``if`` test gates a TYPE_CHECKING block."""
    return "TYPE_CHECKING" in ast.dump(test)


def _iter_imports(node: ast.AST, kind: ImportKind) -> Iterator[tuple[ast.stmt, ImportKind]]:
    """Yield every import statement at or under `node`, tagged with how it is reached.

    `kind` propagates down the tree, so the *nearest* enclosing construct wins: an
    import inside ``if TYPE_CHECKING:`` inside a function is TYPE_CHECKING, and an
    import inside a function nested in a function is still function-local.
    """
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node, kind
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        kind = ImportKind.FUNCTION_LOCAL
    elif isinstance(node, ast.If) and _is_type_checking_test(node.test):
        # Only the body is type-checking-only; the `else` branch runs at runtime.
        for stmt in node.body:
            yield from _iter_imports(stmt, ImportKind.TYPE_CHECKING)
        for stmt in node.orelse:
            yield from _iter_imports(stmt, kind)
        return
    for child in ast.iter_child_nodes(node):
        yield from _iter_imports(child, kind)


def _edge_targets(
    node: ast.stmt, importer: str, is_package: bool, modules: dict[str, Path]
) -> Iterator[str]:
    """Yield the raw (unresolved) first-party import targets of one statement."""
    if isinstance(node, ast.Import):
        yield from (a.name for a in node.names)
        return
    if not isinstance(node, ast.ImportFrom):
        return
    module = _absolute_module(node, importer, is_package)
    if not module:
        return
    for alias in node.names:
        # grimp semantics: `from pkg import submodule` binds the SUBMODULE, so the
        # edge is to `pkg.submodule`, not to `pkg`. A non-module name (a class, a
        # function) leaves the edge on `pkg` itself.
        submodule = f"{module}.{alias.name}"
        yield submodule if submodule in modules else module


def build_graph(package_root: Path = DEFAULT_PACKAGE_ROOT) -> ImportGraph:
    """Parse every module under `package_root` into a resolved `ImportGraph`."""
    package_root = package_root.resolve()
    if not package_root.is_dir():
        raise NotADirectoryError(f"package root does not exist: {package_root}")
    modules = _discover_modules(package_root)
    package = package_root.name
    edges: set[Edge] = set()

    for importer, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        is_package = path.name == "__init__.py"
        for node, kind in _iter_imports(tree, ImportKind.TOPLEVEL):
            for target in _edge_targets(node, importer, is_package, modules):
                if not target.startswith(package):
                    continue
                imported = _resolve(target, modules)
                if imported and imported != importer:
                    edges.add(Edge(importer, imported, kind, node.lineno))

    return ImportGraph(package=package, modules=modules, edges=tuple(sorted(edges, key=_edge_key)))


def _edge_key(edge: Edge) -> tuple[str, str, str, int]:
    return (edge.importer, edge.imported, str(edge.kind), edge.lineno)


# --------------------------------------------------------------------------------------
# Counterfactual: what does a proposed module -> package mapping do to the cycle count?
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PackageMapping:
    """A proposed module -> target-package assignment.

    `exact` wins over `prefix`; among prefixes the longest (most specific) wins.
    """

    exact: dict[str, str]
    prefix: dict[str, str]

    @classmethod
    def from_json(cls, path: Path) -> PackageMapping:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(exact=raw.get("exact", {}), prefix=raw.get("prefix", {}))

    def target(self, module: str) -> str | None:
        """The package `module` lands in, or None if the mapping does not cover it."""
        if module in self.exact:
            return self.exact[module]
        best: tuple[int, str] | None = None
        for pref, pkg in self.prefix.items():
            if (module == pref or module.startswith(f"{pref}.")) and (
                best is None or len(pref) > best[0]
            ):
                best = (len(pref), pkg)
        return best[1] if best else None

    def unmapped(self, modules: Iterable[str]) -> list[str]:
        return sorted(m for m in modules if self.target(m) is None)


@dataclass(frozen=True, slots=True)
class Counterfactual:
    """The cross-package cycle result of applying a `PackageMapping`."""

    cycles: list[list[str]]
    cross_edges: dict[tuple[str, str], list[Edge]]
    unmapped: list[str]

    @property
    def cycle_count(self) -> int:
        return len(self.cycles)


def counterfactual(
    graph: ImportGraph, mapping: PackageMapping, kinds: Iterable[ImportKind] | None = None
) -> Counterfactual:
    """Condense `graph` onto `mapping`'s packages and find the cross-package cycles.

    A "cycle" here is a strongly connected component of the *package* graph -- the unit
    import-linter's `layers`/`independence` contracts fail on.
    """
    keep = frozenset(kinds) if kinds is not None else frozenset(ImportKind)
    cross: dict[tuple[str, str], list[Edge]] = defaultdict(list)
    for edge in graph.edges:
        if edge.kind not in keep:
            continue
        src, dst = mapping.target(edge.importer), mapping.target(edge.imported)
        if src and dst and src != dst:
            cross[(src, dst)].append(edge)

    pkg_graph = nx.DiGraph(sorted(cross))
    cycles = sorted(
        (sorted(c) for c in nx.strongly_connected_components(pkg_graph) if len(c) > 1),
        key=lambda c: (-len(c), c),
    )
    return Counterfactual(
        cycles=cycles, cross_edges=dict(cross), unmapped=mapping.unmapped(graph.modules)
    )


# --------------------------------------------------------------------------------------
# CLI. This is a tool, not library code: subcommands print to stdout by design.
# --------------------------------------------------------------------------------------


def _out(line: str = "") -> None:
    print(line)  # noqa: T201  -- CLI output is this module's product


def _kind_filter(args: argparse.Namespace) -> tuple[ImportKind, ...] | None:
    return (ImportKind.TOPLEVEL,) if getattr(args, "toplevel_only", False) else None


def cmd_fan(graph: ImportGraph, args: argparse.Namespace) -> None:
    kinds = _kind_filter(args)
    fan_in, fan_out = graph.fan_in(kinds), graph.fan_out(kinds)
    _out(f"{'fan-in':>6} {'fan-out':>7}  module")
    ranked = sorted(graph.modules, key=lambda m: (-fan_in[m], fan_out[m], m))
    for module in ranked[: args.top]:
        _out(f"{fan_in[module]:6d} {fan_out[module]:7d}  {module}")


def cmd_kinds(graph: ImportGraph, args: argparse.Namespace) -> None:
    by_kind = {k: graph.edge_pairs([k]) for k in ImportKind}
    stmt_counts = {k: sum(1 for e in graph.edges if e.kind == k) for k in ImportKind}
    _out("import STATEMENTS by kind (one statement can carry several edges):")
    for kind, count in stmt_counts.items():
        _out(f"  {kind:>14} = {count}")
    _out()
    _out("unique edges by kind:")
    for kind, pairs in by_kind.items():
        _out(f"  {kind:>14} = {len(pairs)}")
    lazy = (by_kind[ImportKind.FUNCTION_LOCAL] | by_kind[ImportKind.TYPE_CHECKING]) - by_kind[
        ImportKind.TOPLEVEL
    ]
    _out()
    _out(f"edges that exist ONLY via function-local/TYPE_CHECKING: {len(lazy)}")
    _out("  (invisible to `import langres` at runtime -- but VISIBLE to grimp/import-linter)")
    _out()
    _out(f"SCC sizes, ALL edges (what import-linter sees) : {[len(c) for c in graph.sccs()]}")
    top_sccs = graph.sccs([ImportKind.TOPLEVEL])
    _out(f"SCC sizes, TOPLEVEL only (what runtime sees)  : {[len(c) for c in top_sccs]}")
    _out()
    _out("=== runtime (toplevel-only) SCCs ===")
    for scc in top_sccs:
        _out(f"  {scc}")
    _out()
    _out(f"=== SCCs import-linter sees (size > {args.min_scc}) ===")
    for scc in graph.sccs():
        if len(scc) > args.min_scc:
            _out(f"  {scc}")


def cmd_importers(graph: ImportGraph, args: argparse.Namespace) -> None:
    for module in args.modules:
        if module not in graph.modules:
            _out(f"--- {module}: NOT A MODULE in this package")
            continue
        edges = graph.importers_of(module)
        by_importer = sorted({e.importer for e in edges})
        fan_out = graph.fan_out().get(module, 0)
        _out(f"--- {module}: fan-in={len(by_importer)} fan-out={fan_out}")
        for importer in by_importer:
            hits = [e for e in edges if e.importer == importer]
            detail = ", ".join(f"{e.lineno} [{e.kind}]" for e in hits)
            _out(f"      {importer}:{detail}")


def cmd_cycles(graph: ImportGraph, args: argparse.Namespace) -> None:
    kinds = _kind_filter(args)
    label = "TOPLEVEL-only" if kinds else "ALL"
    _out(f"=== module SCCs ({label} edges, size > 1) ===")
    for scc in graph.sccs(kinds):
        _out(f"  [{len(scc)}] {scc}")
    _out()
    _out(f"=== 2-cycles (mutual imports, {label} edges) ===")
    pairs = graph.edge_pairs(kinds)
    seen: set[tuple[str, str]] = set()
    for a, b in sorted(pairs):
        if (b, a) in pairs and (b, a) not in seen:
            seen.add((a, b))
            here = [e for e in graph.edges if (e.importer, e.imported) == (a, b)]
            back = [e for e in graph.edges if (e.importer, e.imported) == (b, a)]
            _out(f"  {a} <-> {b}")
            _out(f"      {a}:{[f'{e.lineno} [{e.kind}]' for e in here]}")
            _out(f"      {b}:{[f'{e.lineno} [{e.kind}]' for e in back]}")


def cmd_counterfactual(graph: ImportGraph, args: argparse.Namespace) -> None:
    mapping = PackageMapping.from_json(args.mapping)
    for label, kinds in (
        ("ALL edges (import-linter)", None),
        ("TOPLEVEL only (runtime)", (ImportKind.TOPLEVEL,)),
    ):
        result = counterfactual(graph, mapping, kinds)
        _out(f"=== {label} ===")
        if result.unmapped:
            _out(f"  UNMAPPED MODULES ({len(result.unmapped)}) -- mapping is incomplete:")
            for module in result.unmapped:
                _out(f"      {module}")
        _out(f"  cross-package cycles: {result.cycle_count}")
        for cycle in result.cycles:
            _out(f"      [{len(cycle)}] {' <-> '.join(cycle)}")
        if args.show_edges:
            _out("  cross-package edges:")
            for (src, dst), edges in sorted(result.cross_edges.items()):
                _out(f"      {src} -> {dst}: {len(edges)} edges")
                for edge in edges[: args.show_edges]:
                    _out(
                        f"          {edge.importer}:{edge.lineno} [{edge.kind}] -> {edge.imported}"
                    )
        _out()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="import_graph",
        description="AST import-graph analysis (no imports executed).",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT,
        help="package source root (default: this repo's src/langres)",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    fan = subs.add_parser("fan", help="fan-in/fan-out table")
    fan.add_argument("--top", type=int, default=25)
    fan.add_argument("--toplevel-only", action="store_true")
    fan.set_defaults(func=cmd_fan)

    kinds = subs.add_parser("kinds", help="toplevel vs function-local vs TYPE_CHECKING + SCCs")
    kinds.add_argument("--min-scc", type=int, default=2)
    kinds.set_defaults(func=cmd_kinds)

    imp = subs.add_parser("importers", help="who imports these modules, with line numbers")
    imp.add_argument("modules", nargs="+")
    imp.set_defaults(func=cmd_importers)

    cyc = subs.add_parser("cycles", help="SCC / mutual-import enumeration")
    cyc.add_argument("--toplevel-only", action="store_true")
    cyc.set_defaults(func=cmd_cycles)

    cf = subs.add_parser("counterfactual", help="cross-package cycles under a proposed mapping")
    cf.add_argument("--mapping", type=Path, required=True, help="JSON {exact:{}, prefix:{}}")
    cf.add_argument(
        "--show-edges", type=int, default=0, metavar="N", help="list up to N edges each"
    )
    cf.set_defaults(func=cmd_counterfactual)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    graph = build_graph(args.package_root)
    args.func(graph, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
