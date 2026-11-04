"""The settings widgets of the step tabs (AppTest, offline): a setting starts at
the value the step's results used, and says so once the step has run; the GEO
search waits for a sampling site; the GEO filter shows the values its results
used."""

from streamlit.testing.v1 import AppTest


def _setting_app():
    import streamlit as st

    from trident.ui.ui import setting_widget

    st.session_state.setdefault("_widget_overrides", {})
    setting_widget(
        st.number_input, "ncbi_gap_size", 2, 0, 20, applied=st.session_state.applied
    )


def _run(app, **state):
    at = AppTest.from_function(app, default_timeout=30)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _active(at):
    return [c.value for c in at.caption if c.value.startswith("Active in results")]


def test_a_setting_starts_at_the_value_its_results_used():
    """Before the step has run: the default, and no caption. After: the value
    the results used, 0 included, and a caption naming it."""
    at = _run(_setting_app, applied={})
    assert at.number_input(key="ncbi_gap_size").value == 2
    assert _active(at) == []

    at = _run(_setting_app, applied={"ncbi_gap_size": 0})
    assert at.number_input(key="ncbi_gap_size").value == 0
    assert _active(at) == ["Active in results: **0**"]


def _choice_app():
    import streamlit as st

    from trident.ui.ui import setting_widget

    setting_widget(
        st.selectbox,
        "gbif_filter_extent",
        500,
        [250, 500, "global"],
        applied=st.session_state.applied,
    )


def test_the_value_last_set_wins_unless_the_widget_cannot_take_it():
    """A value set in the widget (kept across tabs) comes before the results'
    one; a value an option widget no longer offers gives way to the results'
    one, and that one to the default."""
    applied = {"gbif_filter_extent": "global"}
    at = _run(
        _choice_app, applied=applied, _widget_overrides={"gbif_filter_extent": "250"}
    )
    assert at.selectbox(key="gbif_filter_extent").value == 250

    at = _run(
        _choice_app, applied=applied, _widget_overrides={"gbif_filter_extent": 1000}
    )
    assert at.selectbox(key="gbif_filter_extent").value == "global"

    at = _run(_choice_app, applied={}, _widget_overrides={"gbif_filter_extent": 1000})
    assert at.selectbox(key="gbif_filter_extent").value == 500


def _site_app():
    import streamlit as st

    from trident.ui.geo import display_gbif_parameters

    st.session_state.setdefault("_widget_overrides", {})
    st.session_state.params = display_gbif_parameters()


def test_the_geo_search_waits_for_a_sampling_site(tmp_path):
    """An uploaded FASTA has no site: the fields start empty and the search
    gets no parameters until both coordinates are entered."""
    at = _run(_site_app, db_path=str(tmp_path / "none.db"), gbif_search_params={})
    assert at.text_input(key="gbif_latitude").value == ""
    assert at.session_state.params is None

    at.text_input(key="gbif_latitude").set_value("78.5").run()
    assert at.session_state.params is None
    at.text_input(key="gbif_longitude").set_value("14.5").run()
    latitude, longitude, extents, area = at.session_state.params
    assert (latitude, longitude) == (78.5, 14.5)


def _filter_app():
    import streamlit as st

    from trident.ui.geo import gbif_filter_controls

    st.session_state.setdefault("_widget_overrides", {})
    gbif_filter_controls()


def test_the_geo_filter_shows_the_values_its_results_used(tmp_path):
    """The filter settings are read from the GEO step's params (the search's
    hold none of them); "global" is kept among the km extents."""
    at = _run(
        _filter_app,
        db_path=str(tmp_path / "none.db"),
        gbif_search_params={"gbif_extents": [250, 500, "global"]},
        geo_params={"gbif_min_occurrences": 4, "gbif_filter_extent": "global"},
    )
    assert at.number_input(key="gbif_min_occurrences").value == 4
    assert at.selectbox(key="gbif_filter_extent").value == "global"
    assert _active(at) == ["Active in results: **4**", "Active in results: **global**"]
