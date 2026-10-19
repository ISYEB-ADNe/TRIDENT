"""
Results UI Module (old layout), kept for comparison until it is removed.

Displays the combined MOL+GEO and HYPO species results table
with per-sequence and per-species views, and export options.
"""

import pandas as pd
import streamlit as st
from pathlib import Path

from trident.pipelines.results_pipeline import (
    build_results_df,
    find_sequence_exclusion_step,
    add_low_identity_warning,
    filter_excluded_results,
    RESULT_COLS,
)

from trident.ui.results import (
    RESULTS_COLUMN_CONFIG,
    _show_all_hits_expander,
    _species_breakdown_columns,
    show_provenance_section,
)
from trident.ui.ui import (
    styled_or_plain,
    highlight_low_identity,
    highlight_check_hit,
    CHECK_HIT_LEGEND,
    highlight_empty_sequence,
    sequence_selector,
)


# --------------------------------------------------------------------
# DISPLAY HELPERS
# -------------------------------------------------------------------


# --------------------------------------------------------------------
# OVERVIEW & PER SEQUENCE
# -------------------------------------------------------------------


def show_results_overview(results_df):
    """Display full results table with low-identity highlighting."""

    with st.container(border=True):
        col_header, col_slider = st.columns([2, 1], vertical_alignment="top")

        with col_header:
            st.markdown("### 📊 Results Overview")

        with col_slider:
            low_identity = st.slider(
                "Low Identity Threshold (%)",
                90,
                100,
                97,
                key="low_id",
                help="Highlight species below this identity threshold",
            )

        display_df = add_low_identity_warning(results_df.copy(), low_identity)

        # Apply highlights: check_hit first, then low_identity and empty_sequence
        # override (red takes priority over orange)
        styled_df = (
            styled_or_plain(display_df, highlight_check_hit)
            .apply(highlight_low_identity, axis=1)
            .apply(highlight_empty_sequence, axis=1)
        )

        legends = []
        if display_df["low_identity_warning"].any():
            legends.append(f":red[■] Low identity (<{low_identity}%)")
        if display_df["check_hit"].any():
            legends.append(CHECK_HIT_LEGEND)
        if display_df["scientificName"].isna().any():
            legends.append(":gray[■] No species assigned")
        if legends:
            st.caption(" · ".join(legends))

        st.dataframe(
            styled_df,
            width="stretch",
            hide_index=True,
            column_order=RESULT_COLS,
            column_config=RESULTS_COLUMN_CONFIG,
        )


def show_results_per_sequence(results_df):
    """Display per-sequence detail view with sequence selector."""

    with st.container(border=True):
        col_header, col_slider = st.columns([2, 1], vertical_alignment="top")
        with col_header:
            st.markdown("### 🔬 Results per (M)OTU/ASV")
        with col_slider:
            low_identity_seq = st.slider(
                "Low Identity Threshold (%)",
                90,
                100,
                97,
                key="low_id_seq",
                help="Highlight species below this identity threshold",
            )

        # Build per-seq_id warning flags for the selector
        def _seq_label(seq_id):
            sdf = results_df[results_df["seq_id"] == seq_id]
            flags = ""
            if sdf["scientificName"].isna().all():
                return f"{seq_id} ∅"
            identity = pd.to_numeric(
                sdf.get("ncbi_top_identity_percentage"), errors="coerce"
            )
            if identity.notna().any() and identity.min() < low_identity_seq:
                flags += " ⚠"
            if sdf.get("check_hit") is not None and sdf["check_hit"].any():
                flags += " ⚠"
            return f"{seq_id}{flags}" if flags else seq_id

        selected_seq = sequence_selector(format_fn=_seq_label)
        seq_df = results_df[results_df["seq_id"] == selected_seq]

        if seq_df.empty or seq_df["scientificName"].isna().all():
            exclusion = find_sequence_exclusion_step(
                all_seq_ids=[selected_seq],
                ncbi_search_df=st.session_state.get("ncbi_search_df"),
                mol_df=st.session_state.get("mol_df"),
                tax_df=st.session_state.get("tax_df"),
                geo_df=st.session_state.get("geo_df"),
            )
            if not exclusion.empty:
                step = exclusion.iloc[0]["pipeline_step"]
                st.info(f"No species assigned — sequence lost at **{step}**.")
            else:
                st.info("No species assigned to this sequence.")
            return

        n_mol_geo = (seq_df["validation_step"] == "MOL+GEO").sum()
        n_hypo = (seq_df["validation_step"] == "HYPO").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Species", len(seq_df))
        c2.metric("MOL+GEO", n_mol_geo)
        c3.metric("HYPO", n_hypo)

        display_df = add_low_identity_warning(seq_df.copy(), low_identity_seq)

        legends = []
        if display_df["low_identity_warning"].any():
            legends.append(f":red[■] Low Identity (<{low_identity_seq}%)")
        if display_df["check_hit"].any():
            legends.append(CHECK_HIT_LEGEND)
        if legends:
            st.caption(" · ".join(legends))

        styled_df = styled_or_plain(
            display_df, highlight_check_hit, highlight_low_identity
        )

        st.dataframe(
            styled_df,
            width="stretch",
            hide_index=True,
            column_order=RESULT_COLS,
            column_config=RESULTS_COLUMN_CONFIG,
        )

        # Curation: checkboxes to exclude species
        species_flags = (
            display_df.dropna(subset=["scientificName"])
            .groupby(["scientificName", "validation_step"], sort=False)
            .agg(
                low_id=("low_identity_warning", "any"),
                below=("check_hit", "any"),
            )
            .reset_index()
            .sort_values("scientificName")
        )
        if not species_flags.empty:
            excluded = set(st.session_state.excluded_results)
            st.caption("Uncheck to exclude a species from the curated results:")
            changed = False
            for _, row in species_flags.iterrows():
                flags = ""
                if row["low_id"]:
                    flags += " :red[⚠]"
                if row["below"]:
                    flags += " :orange[⚠]"
                key = f"{selected_seq}||{row['scientificName']}"
                kept = st.checkbox(
                    f"{row['scientificName']} ({row['validation_step']}){flags}",
                    value=key not in excluded,
                    key=f"keep_{key}",
                )
                if kept and key in excluded:
                    excluded.discard(key)
                    changed = True
                elif not kept and key not in excluded:
                    excluded.add(key)
                    changed = True
            if changed:
                st.session_state.excluded_results = sorted(excluded)


# --------------------------------------------------------------------
# CURATION
# -------------------------------------------------------------------


def show_curated_results(results_df):
    """Show the curated results list and CSV export."""
    from trident.pipelines.results_pipeline import EXPORT_COLS, build_gbif_export_df

    curated_df = filter_excluded_results(
        results_df, set(st.session_state.excluded_results)
    )
    n_excluded = len(results_df.dropna(subset=["scientificName"])) - len(
        curated_df.dropna(subset=["scientificName"])
    )

    with st.container(border=True):
        col_header, col_slider = st.columns([2, 1], vertical_alignment="top")
        with col_header:
            st.markdown("### ✅ Curated Results")
        with col_slider:
            low_identity_curated = st.slider(
                "Low Identity Threshold (%)",
                90,
                100,
                97,
                key="low_id_curated",
                help="Highlight species below this identity threshold",
            )

        valid_curated = curated_df.dropna(subset=["scientificName"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total Species",
            valid_curated["scientificName"].nunique(),
            help="Unique species remaining after manual curation.",
        )
        c2.metric(
            "MOL+GEO",
            (valid_curated["validation_step"] == "MOL+GEO").sum(),
            help="Species confirmed via direct NCBI hit + geographic validation.",
        )
        c3.metric(
            "HYPO",
            (valid_curated["validation_step"] == "HYPO").sum(),
            help="Species confirmed via CO1 proxy validation (BOLD/NCBI).",
        )
        c4.metric(
            "Excluded",
            n_excluded,
            delta=f"-{n_excluded}" if n_excluded else None,
            delta_color="inverse",
            help="Species manually removed via the 'Keep' checkboxes above.",
        )

        # Styled table with highlights
        display_df = add_low_identity_warning(curated_df.copy(), low_identity_curated)

        legends = []
        if display_df["low_identity_warning"].any():
            legends.append(f":red[■] Low Identity (<{low_identity_curated}%)")
        if display_df["check_hit"].any():
            legends.append(CHECK_HIT_LEGEND)
        if display_df["scientificName"].isna().any():
            legends.append(":gray[■] No species assigned")
        if legends:
            st.caption(" · ".join(legends))

        styled_df = (
            styled_or_plain(display_df, highlight_check_hit)
            .apply(highlight_low_identity, axis=1)
            .apply(highlight_empty_sequence, axis=1)
        )

        st.dataframe(
            styled_df,
            width="stretch",
            hide_index=True,
            column_order=RESULT_COLS,
            column_config=RESULTS_COLUMN_CONFIG,
        )

        # CSV exports
        analysis_name = st.session_state.get("analysis_name", "results")

        export_cols = [c for c in EXPORT_COLS if c in curated_df.columns]
        csv_full = curated_df[export_cols].to_csv(index=False)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="📥 Full results (CSV)",
                data=csv_full,
                file_name=f"{analysis_name}_results.csv",
                mime="text/csv",
                width="stretch",
            )
        with dl2:
            if st.button("📥 GBIF export (CSV)", width="stretch"):
                with st.spinner("Looking up higher-level taxa in WoRMS..."):
                    gbif_df = build_gbif_export_df(curated_df)
                st.session_state._gbif_export_csv = gbif_df.to_csv(index=False)

            if "_gbif_export_csv" in st.session_state:
                st.download_button(
                    label="⬇️ Download GBIF export",
                    data=st.session_state._gbif_export_csv,
                    file_name=f"{analysis_name}_gbif_export.csv",
                    mime="text/csv",
                    width="stretch",
                )


# --------------------------------------------------------------------
# PER SPECIES
# -------------------------------------------------------------------


def _show_species_breakdown_table(df, is_hypo, threshold):
    """Render a styled breakdown table for a subset of species rows."""
    col_order, col_config = _species_breakdown_columns(df, is_hypo)
    col_config["check_hit"] = None
    display_df = add_low_identity_warning(df.copy(), threshold)
    styled_df = styled_or_plain(display_df, highlight_check_hit, highlight_low_identity)
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_order=col_order,
        column_config=col_config,
    )


def show_results_per_species(results_df):
    """Display per-species detail view with species selector."""

    with st.container(border=True):
        st.markdown("### 🐟 Species Summary & Links")

        valid_df = results_df.dropna(subset=["scientificName"])
        species_list = sorted(valid_df["scientificName"].unique().tolist())

        if not species_list:
            st.info("No validated species yet.")
            return

        selected_species = st.selectbox(
            "Select a species:",
            species_list,
            index=0,
            key="results_species_selector",
        )

        sp_df = valid_df[valid_df["scientificName"] == selected_species]
        row = sp_df.iloc[0]

        # Taxonomy card
        tax_parts = []
        for rank in ["kingdom", "phylum", "class", "order", "family", "genus"]:
            val = row.get(rank)
            if pd.notna(val):
                tax_parts.append(f"**{rank.capitalize()}:** {val}")
        if pd.notna(row.get("scientificNameAuthorship")):
            tax_parts.append(f"**Authorship:** {row['scientificNameAuthorship']}")
        st.markdown(" · ".join(tax_parts))

        # Links
        links = []
        if pd.notna(row.get("taxonURL")):
            links.append(f"[🔗 WoRMS]({row['taxonURL']})")
        if pd.notna(row.get("gbif_taxonURL")):
            links.append(f"[🔗 GBIF]({row['gbif_taxonURL']})")
        if pd.notna(row.get("ncbi_top_hit_url")):
            links.append(f"[🔗 NCBI]({row['ncbi_top_hit_url']})")
        if pd.notna(row.get("bold_seq_url")):
            links.append(f"[🔗 BOLD]({row['bold_seq_url']})")
        if links:
            st.markdown(" · ".join(links))

        st.divider()

        # Per-sequence breakdown — split by validation path when both exist
        low_identity = st.session_state.get("low_id", 97)
        has_mol_geo = (sp_df["validation_step"] == "MOL+GEO").any()
        has_hypo = (sp_df["validation_step"] == "HYPO").any()

        if has_mol_geo and has_hypo:
            # Both paths — show separate tables with appropriate columns
            st.markdown("#### MOL+GEO Sequences")
            mol_geo_df = sp_df[sp_df["validation_step"] == "MOL+GEO"]
            _show_species_breakdown_table(
                mol_geo_df, is_hypo=False, threshold=low_identity
            )
            _show_all_hits_expander(selected_species, is_hypo=False)

            st.markdown("#### HYPO Sequences")
            hypo_df = sp_df[sp_df["validation_step"] == "HYPO"]
            _show_species_breakdown_table(hypo_df, is_hypo=True, threshold=low_identity)
            _show_all_hits_expander(selected_species, is_hypo=True)
        else:
            st.markdown("#### Sequences")
            _show_species_breakdown_table(
                sp_df, is_hypo=has_hypo, threshold=low_identity
            )
            _show_all_hits_expander(selected_species, is_hypo=has_hypo)


# --------------------------------------------------------------------
# DOWNLOAD
# -------------------------------------------------------------------


def show_db_download_button():
    """Download button for the SQLite database."""
    db_path_str = st.session_state.get("db_path")
    if not db_path_str:
        st.caption("*No analysis started yet.*")
        return

    db_path = Path(db_path_str)
    if not db_path.is_file():
        st.caption("*Database file not found.*")
        return

    with open(db_path, "rb") as f:
        db_bytes = f.read()

    analysis_name = st.session_state.get("analysis_name", "analysis")
    st.download_button(
        label=f"📥 Download Database ({analysis_name}.db)",
        data=db_bytes,
        file_name=f"{analysis_name}.db",
        mime="application/x-sqlite3",
        width="stretch",
    )


# --------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


def _show_export_section():
    """Database export — always visible when an analysis is active."""
    with st.container(border=True):
        st.markdown("### 💾 Export")
        st.caption(
            "Download the full SQLite database containing all intermediate results, "
            "cached searches, and parameters from every pipeline step. "
            "This file can be re-imported at the Start tab to restore or share an analysis."
        )
        show_db_download_button()


def results_main():
    st.title("📦 Analysis Results")
    st.divider()

    _show_export_section()
    show_provenance_section()

    if st.session_state.get("geo_df") is None:
        st.info(
            "📊 No results yet. Complete at least steps 1–3 "
            "(MOL → TAX → GEO) to see results."
        )
        return

    results_df = build_results_df(
        sequences_df=st.session_state.sequences_df,
        geo_df=st.session_state.geo_df,
        mol_df=st.session_state.mol_df,
        hypo_df=st.session_state.get("hypo_df"),
    )

    if results_df.empty:
        st.info("📊 No results available.")
        return

    st.divider()
    show_results_overview(results_df)
    st.divider()
    show_results_per_sequence(results_df)
    st.divider()
    show_curated_results(results_df)
    st.divider()
    show_results_per_species(results_df)
