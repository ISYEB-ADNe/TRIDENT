"""The tutorial notebook writes each setting as a plain value, with the app
default it copies named in a comment: every such value is still that default,
and every setting (an upper-case name) other than the study's own inputs names
one."""

import ast
import json
import re
from pathlib import Path

from trident.core import constants
from trident.ui import defaults

NOTEBOOK = Path(__file__).parents[1] / "example" / "example_pipeline.ipynb"
SETTING = re.compile(
    r"^\s*(?P<name>\w+)\s*=\s*(?P<value>[^#\n]+?),?\s*(?:#(?P<note>.*))?$"
)
APP_DEFAULT = re.compile(
    r"app default \((?P<module>constants|defaults)\.(?P<name>\w+)\)"
)
# The study's own inputs, not app defaults.
STUDY_INPUTS = {
    "FASTA_FILE",
    "FORCE_RERUN",
    "LATITUDE",
    "LONGITUDE",
    "EXTENTS",
    "FILTER_EXTENT",
    "SEQ_ID",
    "EXCLUDED",
}


def _literal(text):
    try:
        return True, ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return False, None


def test_the_notebook_values_are_the_app_defaults():
    modules = {"constants": constants, "defaults": defaults}
    checked = 0
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for line in cell["source"]:
            m = SETTING.match(line.rstrip("\n"))
            if not m or m["name"] in STUDY_INPUTS:
                continue
            is_literal, value = _literal(m["value"].strip())
            if not is_literal:
                continue  # computed (a call, a name): not a setting
            default = APP_DEFAULT.search(m["note"] or "")
            if not m["name"].isupper() and not default:
                continue  # a call's own argument (e.g. show_circles=True)
            assert default, f"no app default named: {line.strip()}"
            assert value == getattr(modules[default["module"]], default["name"]), line
            checked += 1
    assert checked > 20
