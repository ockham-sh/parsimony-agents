import ast
from collections.abc import Iterable

import pandas as pd

INDEX_PRODUCING_METHODS: set[str] = {
    # Common pandas ops that often introduce / rely on index semantics.
    # Note: this is intentionally conservative; we can tune over time.
    "groupby",
    "pivot",
    "pivot_table",
    "merge",
    "join",
    "set_index",
    "sort_index",
    "reindex",
    "stack",
    "unstack",
    "resample",
}


def _is_pandas_type(t: type | None) -> bool:
    if t is None:
        return False
    if not isinstance(t, type):
        return False
    return issubclass(t, (pd.DataFrame, pd.Series))


def _iter_assigned_names(targets: Iterable[ast.AST]) -> Iterable[str]:
    for t in targets:
        if isinstance(t, ast.Name):
            yield t.id
        elif isinstance(t, (ast.Tuple, ast.List)):
            yield from _iter_assigned_names(t.elts)


def _get_base_name(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    return None


def _has_kw_true(call: ast.Call, *, kw_name: str) -> bool:
    for kw in call.keywords:
        if kw.arg != kw_name:
            continue
        if isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


class IndexPolicyLinter(ast.NodeVisitor):
    """
    Lints for our "no pandas index semantics" policy:
      - Avoid reading `.index`
      - Ensure index-producing ops are followed by `.reset_index()` in the same chain,
        or (heuristically) on the next line for simple assignments.
    """

    def __init__(self, type_map: dict[str, type] | None = None):
        self.type_map = type_map
        self.issues: list[str] = []
        self._parents: dict[ast.AST, ast.AST] = {}

    def run(self, tree: ast.AST) -> list[str]:
        self._build_parent_map(tree)

        # Heuristic, statement-order enforcement for simple patterns like:
        #   x = df.groupby(...).sum()
        #   x = x.reset_index()
        self._check_scopes_for_next_line_reset(tree)

        # Also run expression-based checks (e.g. `.index` reads)
        self.visit(tree)
        return self.issues

    def _build_parent_map(self, tree: ast.AST) -> None:
        self._parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self._parents[child] = parent

    def _is_pandas_like_base(self, base_name: str | None) -> bool:
        if not self.type_map or not base_name:
            # If we don't have types, we choose to lint anyway (notebook code is pandas-heavy).
            return True
        return _is_pandas_type(self.type_map.get(base_name))

    def _has_reset_index_ancestor(self, node: ast.AST) -> bool:
        cur: ast.AST | None = node
        while cur is not None:
            parent = self._parents.get(cur)
            if parent is None:
                break
            if (
                isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Attribute)
                and parent.func.attr == "reset_index"
            ):
                return True
            cur = parent
        return False

    def _find_index_ops_without_reset(self, expr: ast.AST) -> list[tuple[str, int]]:
        """
        Returns [(op_name, lineno), ...] for index-producing operations in `expr`
        that are NOT followed by `.reset_index()` in the same call chain.
        """
        issues: list[tuple[str, int]] = []
        for node in ast.walk(expr):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            op = node.func.attr
            if op not in INDEX_PRODUCING_METHODS:
                continue

            base_name = _get_base_name(node.func.value)
            if not self._is_pandas_like_base(base_name):
                continue

            if not self._has_reset_index_ancestor(node):
                issues.append((op, getattr(node, "lineno", 0)))
        return issues

    def _is_simple_reset_index_stmt(self, stmt: ast.stmt) -> str | None:
        """
        Detect:
          - x = x.reset_index(...)
          - x.reset_index(..., inplace=True)
        Returns the variable name if matched, else None.
        """
        # x = x.reset_index(...)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            target_name = stmt.targets[0].id
            if isinstance(stmt.value, ast.Call) and isinstance(
                stmt.value.func, ast.Attribute
            ) and (
                stmt.value.func.attr == "reset_index"
                and _get_base_name(stmt.value.func.value) == target_name
            ):
                return target_name

        # x.reset_index(..., inplace=True)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "reset_index" and _has_kw_true(
                call, kw_name="inplace"
            ):
                return _get_base_name(call.func.value)

        return None

    def _check_stmt_list_for_next_line_reset(self, body: list[ast.stmt]) -> None:
        pending: dict[str, tuple[int, str]] = {}

        for stmt in body:
            reset_name = self._is_simple_reset_index_stmt(stmt)
            if reset_name and reset_name in pending:
                pending.pop(reset_name, None)

            # If a statement assigns from an index-producing op (without inline reset),
            # require a `.reset_index()` immediately afterwards (heuristic).
            assigned_names: list[str] = []
            value_expr: ast.AST | None = None

            if isinstance(stmt, ast.Assign):
                assigned_names = list(_iter_assigned_names(stmt.targets))
                value_expr = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                assigned_names = [stmt.target.id]
                value_expr = stmt.value

            if assigned_names and value_expr is not None:
                ops = self._find_index_ops_without_reset(value_expr)
                if ops:
                    # Keep the first op name for message clarity
                    op_name, op_line = ops[0]
                    for name in assigned_names:
                        pending[name] = (op_line or getattr(stmt, "lineno", 0), op_name)

        for name, (lineno, op_name) in pending.items():
            self.issues.append(
                f"Line {lineno}: result assigned to '{name}' comes from '.{op_name}()' and should be "
                "followed by '.reset_index()' to avoid index-based semantics."
            )

    def _check_scopes_for_next_line_reset(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
                self._check_stmt_list_for_next_line_reset(getattr(node, "body", []))

    def visit_Attribute(self, node: ast.Attribute):
        # Flag `.index` reads (we're moving away from index semantics).
        if node.attr == "index" and isinstance(getattr(node, "ctx", None), ast.Load):
            base_name = _get_base_name(node.value)
            if self._is_pandas_like_base(base_name):
                self.issues.append(
                    f"Line {node.lineno}: avoid reading '.index' — prefer explicit columns and call "
                    "'.reset_index()' after index-producing operations."
                )
        self.generic_visit(node)


class RollingLinter(ast.NodeVisitor):
    def __init__(self, type_map: dict[str, type] | None = None):
        self.type_map = type_map
        self.issues: list[str] = []

    def visit_Call(self, node):
        # Identify .rolling() calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "rolling":
            base_name = None
            if isinstance(node.func.value, ast.Name):
                base_name = node.func.value.id

            # Skip if we have a type map and base is not pandas-like
            if self.type_map and base_name:
                obj_type = self.type_map.get(base_name)
                if not _is_pandas_type(obj_type):
                    self.generic_visit(node)
                    return

            # Check for explicit min_periods
            has_min_periods = any(kw.arg == "min_periods" for kw in node.keywords)
            if not has_min_periods:
                self.issues.append(
                    f"Line {node.lineno}: .rolling() should include min_periods parameter to avoid "
                    "introducing NA values."
                )
        self.generic_visit(node)


class RawParquetIOLinter(ast.NodeVisitor):
    """Soft lint discouraging raw parquet/JSON I/O for agent-curated artifacts.

    The canonical idioms are ``return_dataset`` / ``return_chart`` (which
    write open-format containers with embedded framework metadata via the
    framework's I/O codecs). Direct ``df.to_parquet(...)`` /
    ``pd.read_parquet(...)`` writes the same bytes but loses provenance,
    curation, and the file ends up indistinguishable from arbitrary user
    input. This lint surfaces such uses so the agent self-corrects toward
    the typed path before a malformed artifact ships.
    """

    issues: list[str]

    _BARE_WRITE_METHODS = ("to_parquet",)
    _BARE_READ_FUNCTIONS = ("read_parquet",)

    def __init__(self) -> None:
        self.issues = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 — ast hook name
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in self._BARE_WRITE_METHODS:
                self.issues.append(
                    f"Line {node.lineno}: avoid `.{attr}(...)` for agent artifacts. "
                    "Return curated outputs via `return_dataset` / `return_chart` so "
                    "curation metadata is embedded in the file."
                )
            elif attr in self._BARE_READ_FUNCTIONS and _get_base_name(node.func.value) == "pd":
                self.issues.append(
                    f"Line {node.lineno}: prefer `TabularResult.from_parquet(path)` over "
                    "`pd.{attr}(path)` so embedded provenance round-trips into the "
                    "Result wrapper.".format(attr=attr)
                )
        self.generic_visit(node)


class FrameworkImportLinter(ast.NodeVisitor):
    """Reject ``import parsimony_agents...`` from cell code.

    Framework helpers (``load_dataset``, ``connectors``, ``display``,
    ``pd``, ``np``, ``alt``) are pre-injected into every kernel
    namespace. Importing from ``parsimony_agents`` is both unnecessary
    and incorrect — those modules are not user-facing API. The lint
    fires at draft time with a message that names the right path
    (use the pre-injected globals).
    """

    _FORBIDDEN_PREFIXES = ("parsimony_agents",)

    def __init__(self) -> None:
        self.issues: list[str] = []

    def _is_forbidden(self, name: str | None) -> bool:
        if not name:
            return False
        return any(
            name == p or name.startswith(p + ".") for p in self._FORBIDDEN_PREFIXES
        )

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if self._is_forbidden(alias.name):
                self.issues.append(
                    f"Line {node.lineno}: do not import {alias.name!r} — framework "
                    "helpers (load_dataset, connectors, display, pd/np/alt) are "
                    "pre-injected as kernel globals. Use them directly."
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if self._is_forbidden(node.module):
            self.issues.append(
                f"Line {node.lineno}: do not import from {node.module!r} — framework "
                "helpers (load_dataset, connectors, display, pd/np/alt) are "
                "pre-injected as kernel globals. Use them directly."
            )


def check_code(code: str, type_map: dict[str, type] | None = None) -> list[str]:
    tree = ast.parse(code)
    issues: list[str] = []

    rolling = RollingLinter(type_map)
    rolling.visit(tree)
    issues.extend(rolling.issues)

    index_policy = IndexPolicyLinter(type_map)
    issues.extend(index_policy.run(tree))

    raw_io = RawParquetIOLinter()
    raw_io.visit(tree)
    issues.extend(raw_io.issues)

    framework_imports = FrameworkImportLinter()
    framework_imports.visit(tree)
    issues.extend(framework_imports.issues)

    return issues
