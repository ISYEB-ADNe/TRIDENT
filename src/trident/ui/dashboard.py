"""
Dashboard: the parameters used across the pipeline steps, with a download of
the parameter report. The results and the analysis file are downloaded from
the Results tab.
"""

import pandas as pd
import streamlit as st

from trident.core import config
from trident.logging import get_log_level
from trident.ui.mol import FILTER_METHOD_LABELS
from trident.ui.ui import LOG_LEVEL_LABELS, SETTING_LABELS


# --------------------------------------------------------------------
# PARAMETER DEFINITIONS
# -------------------------------------------------------------------


# Sections: (title, params_key, param_keys)
SECTIONS = [
    (
        "MOL: NCBI BLAST",
        "mol_params",
        [
            "ncbi_max_hits",
            "ncbi_ev_exponent",
            "ncbi_query_cover",
            "ncbi_method",
            "ncbi_gap_size",
            "ncbi_gap_min_top",
            "ncbi_low_identity_threshold",
            "ncbi_enforce_low_identity",
        ],
    ),
    (
        "TAX: WoRMS",
        "tax_params",
        [
            "worms_marine_only",
            "worms_resolve_names",
            "tax_drop_mol_rejected",
        ],
    ),
    (
        "GEO: GBIF",
        "geo_params",
        [
            "gbif_latitude",
            "gbif_longitude",
            "gbif_extents",
            "gbif_area",
            "gbif_filter_extent",
            "gbif_min_occurrences",
        ],
    ),
    (
        "EXTRA: BOLD Sequences",
        "extra_params",
        [
            "bold_keep_only_COI5P",
            "bold_keep_ncbi",
            "bold_longest_n",
            "bold_similarity",
        ],
    ),
    (
        "HYPO: Hypothetical Species",
        "hypo_params",
        [
            "hypo_max_hits",
            "hypo_ev_exponent",
            "hypo_identity_cutoff",
            "hypo_ntop",
            "hypo_query_cover",
            "hypo_identity",
            "hypo_check_ev_exponent",
        ],
    ),
]


# --------------------------------------------------------------------
# PARAMETER DISPLAY
# -------------------------------------------------------------------


# Settings whose values the tabs show under another name.
VALUE_LABELS = {"ncbi_method": FILTER_METHOD_LABELS}


def _format_value(value, key: str = "") -> str:
    """Format a parameter value for display, as its tab names it."""
    if key in VALUE_LABELS:
        return VALUE_LABELS[key].get(value, str(value))
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _build_params_text() -> str:
    """Build a plain-text parameter report."""
    analysis_name = st.session_state.get("analysis_name", "analysis")
    lines = [f"TRIDENT Parameters: {analysis_name}", "=" * 50, ""]

    for title, params_key, param_keys in SECTIONS:
        params = st.session_state.get(params_key, {})
        if not params:
            continue

        lines.append(title)
        lines.append("-" * len(title))
        for key in param_keys:
            if key in params:
                label = SETTING_LABELS.get(key, key)
                lines.append(f"  {label}: {_format_value(params[key], key)}")
        lines.append("")

    return "\n".join(lines)


def _show_params_section(title: str, params_key: str, param_keys: list[str]):
    """Display one parameter section."""
    params = st.session_state.get(params_key, {})
    if params is None:
        st.caption(f"*{title}: skipped (not needed)*")
        return
    if not params:
        st.caption(f"*{title}: not yet completed*")
        return

    st.markdown(f"**{title}**")
    rows = []
    for key in param_keys:
        if key in params:
            rows.append(
                {
                    "Parameter": SETTING_LABELS.get(key, key),
                    "Value": _format_value(params[key], key),
                }
            )

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

        for title, params_key, param_keys in SECTIONS:
            _show_params_section(title, params_key, param_keys)

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
