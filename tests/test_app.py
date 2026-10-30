"""The app opens an analysis and shows each of its tabs without an error, and
a step's run button forces a fresh run only on "Erase saved results" (AppTest,
offline). The click flows of the tabs are checked by the AppTest scenarios run
by hand."""

import socket
from pathlib import Path

from streamlit.testing.v1 import AppTest

from trident.core import config
from trident.pipelines.fasta_pipeline import run_fasta_workflow
from trident.ui.app import TAB_CONFIG
from trident.ui.ui import ERASE_SAVED

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


def test_an_analysis_starts_only_once_a_contact_email_is_set(tmp_path, monkeypatch):
    """The Start tab offers no file, saved analysis or example until Settings
    holds an address: a typo is refused."""
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(config, "_repo_secrets", lambda: {})
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.secrets["RESULTS_DIR"] = str(tmp_path)
    at.run()

    def offers_example():
        return any(b.label.startswith("Try the example") for b in at.button)

    assert not offers_example()
    at.text_input(key="_settings_email").input("name@example").run()
    assert not offers_example()
    at.text_input(key="_settings_email").input("name@example.org").run()
    assert offers_example()
    assert not at.exception


def test_a_step_with_failed_queries_holds_the_next_ones_until_accepted():
    """The steps after a search with failed queries do not use its results
    until it is run again or the user goes on with incomplete results, which
    Results then lists; a new run that fails holds them again."""

    def page():
        import pandas as pd
        import streamlit as st

        from trident.ui.results import show_failures_notice
        from trident.ui.ui import (
            init_session_state,
            require_prerequisite,
            run_step_workflow,
        )

        def search(*, force_rerun, retry_empty):
            st.session_state.runs = st.session_state.get("runs", 0) + 1
            st.session_state.ncbi_search_df = pd.DataFrame({"seq_id": ["m1"]})
            st.session_state.ncbi_search_params = {}
            st.session_state.ncbi_search_failed = (
                ["m2"] if st.session_state.runs == 1 else ["m3"]
            )
            st.session_state.mol_df = pd.DataFrame({"seq_id": ["m1"]})

        init_session_state()
        run_step_workflow(
            df_key="ncbi_search_df",
            params_key="ncbi_search_params",
            flag_key="ncbi_search_flag",
            tab_id="mol",
            compare_keys=[],
            current_params={},
            job_fn=search,
            new_string="",
            btn_key="search",
            failed_key="ncbi_search_failed",
        )
        st.session_state.tax_opens = require_prerequisite("mol_df", "no MOL list")
        show_failures_notice()

    at = AppTest.from_function(page)
    at.run()
    at.button(key="search_0").click().run()
    assert at.session_state["tax_opens"] is False
    at.button(key="continue_ncbi_search_failed").click().run()
    assert at.session_state["tax_opens"] is True
    listed = [w.value for w in at.warning if "These results are incomplete" in w.value]
    assert listed and "m2" in listed[0]
    at.button(key="search_1").click().run()
    assert at.session_state["tax_opens"] is False
    assert not at.exception


def test_changed_settings_rerun_a_step_from_its_cache_unless_erased():
    """A step run after its settings changed keeps the results saved for its
    settings: only "Erase saved results" forces it to compute everything
    again."""

    def step_page():
        import pandas as pd
        import streamlit as st

        from trident.ui.ui import run_step_workflow

        def job(*, force_rerun, retry_empty):
            st.session_state.forced.append(force_rerun)

        st.session_state.setdefault("forced", [])
        st.session_state.setdefault("step_df", pd.DataFrame({"a": [1]}))
        st.session_state.setdefault("step_params", {"extents": [500]})
        run_step_workflow(
            df_key="step_df",
            params_key="step_params",
            flag_key="step_flag",
            tab_id="geo",
            compare_keys=["extents"],
            current_params={"extents": [500, 1000]},
            job_fn=job,
            new_string="",
            btn_key="step",
        )

    at = AppTest.from_function(step_page)
    at.run()
    assert "Settings have changed" in at.warning[0].value
    at.button[0].click().run()
    at.selectbox[0].select(ERASE_SAVED).run()
    at.button[0].click().run()
    assert at.session_state["forced"] == [False, True]
    assert not at.exception


def test_step_that_fails_shows_its_error_once_and_does_not_start_again():
    def step_page():
        import streamlit as st

        from trident.ui.ui import run_step_workflow

        def job(*, force_rerun, retry_empty):
            st.session_state.calls = st.session_state.get("calls", 0) + 1
            raise RuntimeError("step failed")

        st.session_state.setdefault("step_df", None)
        st.session_state.setdefault("step_params", {})
        run_step_workflow(
            df_key="step_df",
            params_key="step_params",
            flag_key="step_flag",
            tab_id="geo",
            compare_keys=[],
            current_params={},
            job_fn=job,
            new_string="",
            btn_key="step",
        )

    at = AppTest.from_function(step_page)
    at.run()
    at.button[0].click().run()
    assert at.exception
    at.run()
    assert at.session_state["calls"] == 1


def test_preset_fills_empty_fields_and_any_set_replaces_other_values():
    """Empty fields get the latest saved set. Fields holding values of no saved
    set keep them, the selector empty: any set chosen, the first one too,
    fills them."""

    def page(overrides):
        import streamlit as st

        from trident.ui.ui import param_preset_selector

        st.session_state.setdefault("_widget_overrides", dict(overrides))
        param_preset_selector([{"t": 1}, {"t": 2}], lambda p: f"t={p['t']}", key="sel")

    at = AppTest.from_function(page, args=({},))
    at.run()
    assert at.session_state["_widget_overrides"]["t"] == 1

    at = AppTest.from_function(page, args=({"t": "5"},))
    at.run()
    assert at.selectbox[0].value is None
    at.selectbox[0].select("t=1").run()
    assert at.session_state["_widget_overrides"]["t"] == 1


def test_bold_names_go_with_the_extra_search_they_come_from():
    """A reset of the EXTRA search (here by a new GEO list) drops the names
    its BOLD search added, and keeps the others."""

    def page():
        import pandas as pd
        import streamlit as st

        from trident.pipelines.names import NAME_TABLE_COLUMNS
        from trident.ui.ui import reset_state_after

        table = pd.DataFrame(columns=NAME_TABLE_COLUMNS)
        table["source"] = ["gbif", "bold"]
        st.session_state.name_table = table
        reset_state_after("geo_df")

    at = AppTest.from_function(page)
    at.run()
    assert not at.exception
    assert at.session_state["name_table"]["source"].tolist() == ["gbif"]
