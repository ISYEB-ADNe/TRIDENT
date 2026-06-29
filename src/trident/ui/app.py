"""
TRIDENT Streamlit application entry point.

Run with: uv run trident  (or: streamlit run src/trident/ui/app.py)
"""

from pathlib import Path

import streamlit as st
from trident.ui import ui
from trident.core import config
from trident.clients.ncbi import reload_ncbi_credentials
from trident.logging import LOG_LEVELS, get_log_level, set_log_level, setup_logging

# Resolve the logo next to this module so it works from any working directory.
LOGO = str(Path(__file__).with_name("logo.png"))

# Tab configuration: (Display Name, Session State Key)
TAB_CONFIG = {
    "start": ("Start Analysis", "sequences_df"),
    "mol": ("1 · MOL", "mol_df"),
    "tax": ("2 · TAX", "tax_df"),
    "geo": ("3 · GEO", "geo_df"),
    "extra": ("4 · EXTRA", "extra_df"),
    "hypo": ("5 · HYPO", "hypo_df"),
    "results": ("Results", None),
    "parameters": ("Parameters", None),
}


def _app_config():
    """Settings panel — expander (open if email missing)."""
    email_from_config = config.get(config.CONTACT_EMAIL)
    email_is_preset = bool(email_from_config)

    # Initialize the widget key from preset or stored value
    if email_is_preset:
        st.session_state.setdefault("_settings_email", email_from_config)
    else:
        st.session_state.setdefault("_settings_email", "")

    email_missing = not email_is_preset and not st.session_state.get("_settings_email")

    label = "Settings" if not email_missing else "⚠️ Settings — contact email required"
    with st.expander(label, expanded=email_missing):
        email = st.text_input(
            "Contact email",
            key="_settings_email",
            disabled=email_is_preset,
            help=(
                "Sent with every request to NCBI, GBIF, WoRMS and BOLD. "
                "Required by these services to identify your application. "
                "Pre-fill via CONTACT_EMAIL in secrets.toml or .env."
            ),
        )
        if not email_is_preset and email:
            config.set(config.CONTACT_EMAIL, email)
            reload_ncbi_credentials()

        current_level = get_log_level()
        level = st.selectbox(
            "Execution detail",
            LOG_LEVELS,
            index=LOG_LEVELS.index(current_level) if current_level in LOG_LEVELS else 0,
            format_func=lambda level: "Standard" if level == "INFO" else "Verbose",
            help=(
                "Controls how much detail appears in the status boxes "
                "while a step is running. "
                "**Standard** shows progress and key results. "
                "**Verbose** adds internal details like cache hits, API calls, "
                "and parameter values — useful for troubleshooting."
            ),
        )
        if level != current_level:
            set_log_level(level)


def get_tab_label(tab_id: str) -> str:
    """Builds tab labels with dynamic checkmarks."""
    name, state_key = TAB_CONFIG.get(tab_id, (tab_id, None))
    is_complete = state_key and st.session_state.get(state_key) is not None
    return f"{'✅ ' if is_complete else ''}{name}"


def main():
    # Turn on trident logging for the app process. Idempotent: a no-op after the
    # first run, so re-running on each Streamlit interaction costs nothing.
    setup_logging()

    # Raise the pandas Styler cell cap so the colored full-list tables render
    # for large runs (the styled_or_plain fallback still guards extreme cases).
    ui.apply_styler_limit()

    st.set_page_config(layout="wide", page_title="TRIDENT", page_icon=LOGO)
    ui.init_session_state()

    st.logo(LOGO, size="large")

    st.title("TRIDENT")
    st.caption("Taxonomic Resolution and IDentification using Environmental dNa Traces")
    if st.session_state.get("analysis_name"):
        st.caption(f"📁 {st.session_state.analysis_name}")

    _app_config()
    st.divider()

    # Tab State Initialization
    if "requested_tab" in st.session_state:
        st.session_state.current_tab = st.session_state.pop("requested_tab")
    st.session_state.setdefault("current_tab", "start")

    # Navigation Bar
    current_tab = st.segmented_control(
        label="Navigation",
        options=list(TAB_CONFIG.keys()),
        format_func=get_tab_label,
        key="current_tab",
        width="stretch",
        label_visibility="collapsed",
    )

    # Tab Content Rendering
    if current_tab == "start":
        from trident.ui import inputs

        inputs.upload_and_start_analysis()

    elif current_tab == "mol":
        from trident.ui import mol

        mol.mol_analysis()

    elif current_tab == "tax":
        from trident.ui import tax

        tax.tax_analysis()

    elif current_tab == "geo":
        from trident.ui import geo

        geo.geo_analysis()

    elif current_tab == "extra":
        from trident.ui import extra

        extra.extra_analysis()

    elif current_tab == "hypo":
        from trident.ui import hypo

        hypo.hypo_analysis()

    elif current_tab == "results":
        from trident.ui import results

        results.results_main()

    elif current_tab == "parameters":
        from trident.ui import dashboard

        dashboard.dashboard_main()


if __name__ == "__main__":
    main()
