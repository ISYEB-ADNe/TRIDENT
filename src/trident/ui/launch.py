"""Console-script entry point: launch the TRIDENT Streamlit app.

Registered as `trident` via [project.scripts]. Resolves app.py next to this
module so `uv run trident` (or `trident` once installed) works from any
working directory.
"""

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> None:
    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
