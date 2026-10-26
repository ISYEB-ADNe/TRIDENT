"""The app opens an analysis and shows each of its tabs without an error
(AppTest, offline). The click flows of the tabs are checked by the AppTest
scenarios run by hand."""

import socket
from pathlib import Path

from streamlit.testing.v1 import AppTest

from trident.core import config
from trident.pipelines.fasta_pipeline import run_fasta_workflow
from trident.ui.app import TAB_CONFIG

APP = Path(__file__).parents[1] / "src" / "trident" / "ui" / "app.py"


def test_every_tab_of_a_new_analysis_opens_without_error(tmp_path, monkeypatch):
    """No tab sends a request when it opens: every connection is refused and
    recorded, whatever client makes it."""
    attempts = []

    def refuse(sock, address):
        attempts.append(address)
        raise ConnectionError("network blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    # No contact email from the environment or the repository's secrets file.
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(config, "_repo_secrets", lambda: {})
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("RESULTS_DIR", str(results))
    fasta = tmp_path / "demo.fasta"
    fasta.write_text(">m1\nACGTACGTAC\n>m2\nGGGTTTAAAC\n")
    run_fasta_workflow(str(fasta), db_path=str(results / "demo.db"))

    at = AppTest.from_file(str(APP), default_timeout=60)
    # Replaces the secrets Streamlit would read from .streamlit/secrets.toml.
    at.secrets["RESULTS_DIR"] = str(results)
    at.session_state["analysis_name"] = "demo"
    at.run()
    assert at.session_state["sequences_df"]["seq_id"].tolist() == ["m1", "m2"]
    for tab in TAB_CONFIG:
        at.session_state["requested_tab"] = tab
        at.run()
        assert at.session_state["current_tab"] == tab
        assert not at.exception, (tab, [e.value for e in at.exception])
    assert attempts == []
