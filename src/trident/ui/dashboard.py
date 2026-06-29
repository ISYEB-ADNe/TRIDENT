"""
Dashboard — parameter summary and export.

Displays all parameters used across pipeline steps and provides
download buttons for the parameter report and final results CSV.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from trident.core import config
from trident.logging import get_log_level
from trident.pipelines.results_pipeline import build_results_df, EXPORT_COLS


# --------------------------------------------------------------------
# PARAMETER DEFINITIONS
# -------------------------------------------------------------------

# (session_state_key, label) — ordered within each section
PARAM_LABELS: dict[str, str] = {
    # MOL (NCBI BLAST)
    "ncbi_max_hits": "Max Hits per Sequence",
    "ncbi_ev_exponent": "E-value Exponent",
    "ncbi_query_cover": "Query Cover (%)",
    "ncbi_method": "Filter Method",
    "ncbi_gap_size": "Gap / Similarity Size (%)",
    "ncbi_gap_min_top": "Gap Minimum Top Identity (%)",
    "ncbi_low_identity_threshold": "Low Identity Threshold (%)",
    "ncbi_enforce_low_identity": "Enforce Low Identity Threshold",
    # GEO (GBIF)
    "gbif_latitude": "Latitude",
    "gbif_longitude": "Longitude",
    "gbif_extents": "Search Extents (km)",
    "gbif_filter_extent": "Selected Extent (km)",
    "gbif_min_occurrences": "Min. Occurrences",
    # EXTRA (BOLD)
    "bold_keep_only_COI5P": "Keep Only COI-5P",
    "bold_keep_ncbi": "Keep NCBI Records",
    "bold_longest_n": "Longest N",
    "bold_similarity": "Similarity Threshold (%)",
    # HYPO (BOLD + NCBI)
    "hypo_max_hits": "Max Hits per Sequence",
    "hypo_ev_exponent": "E-value Exponent",
    "hypo_identity_cutoff": "Identity Cutoff (%)",
    "hypo_ntop": "N Top Sequences",
    "hypo_query_cover": "Query Cover (%)",
    "hypo_identity": "Identity Threshold (%)",
    "hypo_check_ev_exponent": "Check E-value Exponent",
}

# Sections: (title, params_key, param_keys)
SECTIONS = [
    (
        "MOL — NCBI BLAST",
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
        "GEO — GBIF",
        "geo_params",
        [
            "gbif_latitude",
            "gbif_longitude",
            "gbif_extents",
            "gbif_filter_extent",
            "gbif_min_occurrences",
        ],
    ),
    (
        "EXTRA — BOLD Sequences",
        "extra_params",
        [
            "bold_keep_only_COI5P",
            "bold_keep_ncbi",
            "bold_longest_n",
            "bold_similarity",
        ],
    ),
    (
        "HYPO — BOLD/NCBI Validation",
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


def _format_value(value) -> str:
    """Format a parameter value for display."""
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
    lines = [f"TRIDENT Parameters — {analysis_name}", "=" * 50, ""]

    for title, params_key, param_keys in SECTIONS:
        params = st.session_state.get(params_key, {})
        if not params:
            continue

        lines.append(title)
        lines.append("-" * len(title))
        for key in param_keys:
            if key in params:
                label = PARAM_LABELS.get(key, key)
                lines.append(f"  {label}: {_format_value(params[key])}")
        lines.append("")

    return "\n".join(lines)


def _show_params_section(title: str, params_key: str, param_keys: list[str]):
    """Display one parameter section."""
    params = st.session_state.get(params_key, {})
    if params is None:
        st.caption(f"*{title} — skipped (not needed)*")
        return
    if not params:
        st.caption(f"*{title} — not yet completed*")
        return

    st.markdown(f"**{title}**")
    rows = []
    for key in param_keys:
        if key in params:
            rows.append(
                {
                    "Parameter": PARAM_LABELS.get(key, key),
                    "Value": _format_value(params[key]),
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
                {"Parameter": "Log Level", "Value": level},
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
# RESULTS CSV DOWNLOAD
# -------------------------------------------------------------------


def show_results_download():
    """Download button for the final results as CSV."""
    with st.container(border=True):
        st.markdown("### Results Export")

        if st.session_state.get("geo_df") is None:
            st.caption("*Complete at least MOL → TAX → GEO to export results.*")
            return

        results_df = build_results_df(
            sequences_df=st.session_state.sequences_df,
            geo_df=st.session_state.geo_df,
            mol_df=st.session_state.mol_df,
            hypo_df=st.session_state.get("hypo_df"),
        )

        if results_df.empty:
            st.caption("*No results to export.*")
            return

        # Keep only export columns that exist
        export_cols = [c for c in EXPORT_COLS if c in results_df.columns]
        export_df = results_df[export_cols]

        csv_data = export_df.to_csv(index=False)
        analysis_name = st.session_state.get("analysis_name", "analysis")

        st.caption(
            f"{len(export_df)} rows · "
            f"{export_df['scientificName'].dropna().nunique()} species · "
            f"{export_df['seq_id'].nunique()} sequences"
        )

        st.download_button(
            label="📥 Download Results (.csv)",
            data=csv_data,
            file_name=f"{analysis_name}_results.csv",
            mime="text/csv",
            width="stretch",
        )


# --------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


def show_db_download():
    """Download button for the SQLite database — always available."""
    with st.container(border=True):
        st.markdown("### Database Export")

        db_path_str = st.session_state.get("db_path")
        if not db_path_str:
            st.caption("*No analysis started yet.*")
            return

        db_path = Path(db_path_str)
        if not db_path.exists():
            st.caption("*Database file not found.*")
            return

        analysis_name = st.session_state.get("analysis_name", "analysis")
        st.download_button(
            label=f"📥 Download Database ({analysis_name}.db)",
            data=db_path.read_bytes(),
            file_name=f"{analysis_name}.db",
            mime="application/x-sqlite3",
            width="stretch",
        )


def dashboard_main():
    st.title("📊 Parameters")
    st.divider()

    show_parameters()
    st.divider()
    show_results_download()
    st.divider()
    show_db_download()
