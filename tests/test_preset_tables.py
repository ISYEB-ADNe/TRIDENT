"""Preset selectors must read tables that a pipeline actually writes.

``get_recent_param_sets`` takes a SQLite table name, but the session-state
dicts alongside it are named ``*_params``. Passing one of those names produces
a table that never exists, and the empty result is indistinguishable from
"this step has not run yet", so the selector silently never appears.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "trident"


def _declared_input_tables() -> set[str]:
    """Every ``{table}_inputs`` table the @save_to_db decorators create.

    ``table_name`` is passed either positionally or by keyword, so both forms
    have to be read.
    """
    tables = set()
    for path in (SRC / "pipelines").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name != "save_to_db":
                continue
            if node.args and isinstance(node.args[0], ast.Constant):
                tables.add(f"{node.args[0].value}_inputs")
            for kw in node.keywords:
                if kw.arg == "table_name" and isinstance(kw.value, ast.Constant):
                    tables.add(f"{kw.value.value}_inputs")
    return tables


def _preset_table_arguments() -> list[tuple[str, int, str]]:
    """(module, lineno, table_name) for each get_recent_param_sets call."""
    calls = []
    for path in (SRC / "ui").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_recent_param_sets"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
            ):
                calls.append((path.name, node.lineno, node.args[1].value))
    return calls


def test_every_preset_selector_reads_a_real_inputs_table():
    declared = _declared_input_tables()
    calls = _preset_table_arguments()

    assert calls, "no get_recent_param_sets call sites found"
    assert declared, "no @save_to_db table_name declarations found"

    unknown = [
        f"{module}:{lineno} reads '{table}'"
        for module, lineno, table in calls
        if table not in declared
    ]
    assert not unknown, f"preset selectors read non-existent tables: {unknown}"
