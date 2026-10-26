"""Every column a tab shows has its display settings in COL.

Streamlit hides a column whose settings are missing, so a table listing a
column COL does not know would lose it silently (column_config raises in the
app, but only when that table is drawn).
"""

import ast
from pathlib import Path

import trident.ui.ui as ui

UI_DIR = Path(ui.__file__).parent


def _string_list(node, constants):
    """The column names of a list literal, with its *starred constants."""
    names = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            names.append(element.value)
        elif isinstance(element, ast.Starred) and isinstance(element.value, ast.Name):
            names += constants.get(element.value.id, [])
    return names


def _shown_columns(tree, constants):
    """(columns, overridden names) of each column_config() call whose columns
    are a list literal or a name bound to one."""
    calls = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.Module)):
            continue
        local = dict(constants)
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        local[target.id] = _string_list(node.value, local)
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "column_config"
                and node.args
            ):
                arg = node.args[0]
                if isinstance(arg, ast.List):
                    columns = _string_list(arg, local)
                elif isinstance(arg, ast.Name) and arg.id in local:
                    columns = local[arg.id]
                else:
                    continue
                calls.append((columns, {k.arg for k in node.keywords}))
    return calls


def test_every_shown_column_has_display_settings():
    module_constants = {"RANK_COLUMNS": ui.RANK_COLUMNS}
    missing = {}
    for path in sorted(UI_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        constants = dict(module_constants)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = _string_list(node.value, constants)
        for columns, overridden in _shown_columns(tree, constants):
            unknown = [c for c in columns if c not in ui.COL and c not in overridden]
            if unknown:
                missing.setdefault(path.name, set()).update(unknown)
    assert not missing
