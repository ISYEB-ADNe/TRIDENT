"""
Dashboard: the parameters used across the pipeline steps, with a download of
the parameter report. The results and the analysis file are downloaded from
the Results tab.
"""

import pandas as pd
import streamlit as st

from trident.core import config
from trident.logging import get_log_level
from trident.core.utils import SETTING_SECTIONS, shown_settings
from trident.ui.ui import LOG_LEVEL_LABELS


# --------------------------------------------------------------------
# PARAMETER DEFINITIONS
# -------------------------------------------------------------------


# The title of each step's section (a step missing here is shown under its
# name); the steps, their settings and their order are those of
# SETTING_SECTIONS, the values those of the step's ``{step}_params``.
SECTION_TITLES = {
    "mol": "MOL: NCBI BLAST",
    "tax": "TAX: WoRMS",
    "geo": "GEO: GBIF",
    "extra": "EXTRA: BOLD Sequences",
    "hypo": "HYPO: Hypothetical Species",
}


# --------------------------------------------------------------------
# PARAMETER DISPLAY
# -------------------------------------------------------------------


def _build_params_text() -> str:
    """Build a plain-text parameter report."""
    analysis_name = st.session_state.get("analysis_name", "analysis")
    lines = [f"TRIDENT Parameters: {analysis_name}", "=" * 50, ""]

    for step, labels in SETTING_SECTIONS.items():
        title = SECTION_TITLES.get(step, step.upper())
        params = st.session_state.get(f"{step}_params", {})
        if not params:
            continue

        lines.append(title)
        lines.append("-" * len(title))
        for label, value in shown_settings(params, labels):
            lines.append(f"  {label}: {value}")
        lines.append("")

    return "\n".join(lines)


def _show_params_section(step: str, labels: dict[str, str]):
    """Display one parameter section."""
    title = SECTION_TITLES.get(step, step.upper())
    params = st.session_state.get(f"{step}_params", {})
    if params is None:
        st.caption(f"*{title}: skipped (not needed)*")
        return
    if not params:
        st.caption(f"*{title}: not yet completed*")
        return

    st.markdown(f"**{title}**")
    rows = [
        {"Parameter": label, "Value": value}
        for label, value in shown_settings(params, labels)
    ]

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
        )


def _show_setup_section():
    """Display app-level setup parameters."""
    email = config.contact_email() or "—"
    ua = config.user_agent() or "—"
    level = get_log_level()

    st.markdown("**Setup**")
    st.dataframe(
        pd.DataFrame(
            [
                {"Parameter": "Contact Email", "Value": email},
                {"Parameter": "User-Agent", "Value": ua},
                {
                    "Parameter": "Execution detail",
                    "Value": LOG_LEVEL_LABELS.get(level, level),
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def show_parameters():
    """Display all pipeline parameters."""
    with st.container(border=True):
        st.markdown("### Parameters")

        _show_setup_section()

        for step, labels in SETTING_SECTIONS.items():
            _show_params_section(step, labels)

        # Download params as text
        params_text = _build_params_text()
        if params_text.count("\n") > 3:  # has at least one section
            analysis_name = st.session_state.get("analysis_name", "analysis")
            st.download_button(
                label="📥 Download Parameters (.txt)",
                data=params_text,
                file_name=f"{analysis_name}_parameters.txt",
                mime="text/plain",
                width="stretch",
            )


# --------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


def dashboard_main():
    st.title("📊 Parameters")
    st.divider()

    show_parameters()
