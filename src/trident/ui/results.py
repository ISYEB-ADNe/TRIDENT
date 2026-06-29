"""
Results UI Module

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
from trident.core.database import load_provenance

from trident.ui.ui import (
    styled_or_plain,
    COL,
    highlight_low_identity,
    highlight_below_mol,
    highlight_empty_sequence,
    sequence_selector,
)


# --------------------------------------------------------------------
# DISPLAY HELPERS
# -------------------------------------------------------------------


RESULTS_COLUMN_CONFIG = {
    "seq_id": COL["seq_id"],
    "validation_step": st.column_config.TextColumn(
        "Validation",
        help="MOL+GEO = confirmed via NCBI + geography. HYPO = confirmed via BOLD/NCBI proxy.",
    ),
    "scientificName": COL["scientificName"],
    "taxonURL": COL["taxonURL"],
    "kingdom": COL["kingdom"],
    "phylum": COL["phylum"],
    "class": COL["class"],
    "order": COL["order"],
    "family": COL["family"],
    "genus": COL["genus"],
    "specificEpithet": COL["specificEpithet"],
    "scientificNameAuthorship": COL["scientificNameAuthorship"],
    "ncbi_top_identity_percentage": st.column_config.NumberColumn(
        "Identity (%)",
        format="%.2f",
        help="Best NCBI BLAST identity for this species.",
    ),
    "ncbi_top_query_cover": st.column_config.NumberColumn(
        "Query Cover (%)",
        format="%.2f",
        help="Query coverage of the best NCBI hit.",
    ),
    "ncbi_top_hit_url": COL["ncbi_top_hit_url"],
    "bold_seq_url": COL["bold_seq_url"],
    "gbif_occurrences": COL["gbif_occurrences"],
    "gbif_taxonURL": COL["gbif_taxonURL"],
    "low_identity_warning": None,
    "below_mol": None,
    "dna_sequence": None,
}


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

        # Apply highlights: below_mol first, then low_identity and empty_sequence
        # override (red takes priority over orange)
        styled_df = (
            styled_or_plain(display_df, highlight_below_mol)
            .apply(highlight_low_identity, axis=1)
            .apply(highlight_empty_sequence, axis=1)
        )

        legends = []
        if display_df["low_identity_warning"].any():
            legends.append(f":red[■] Low identity (<{low_identity}%)")
        if display_df["below_mol"].any():
            legends.append(
                ":orange[■] Found in NCBI but would not have passed the MOL filter"
            )
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
            if sdf.get("below_mol") is not None and sdf["below_mol"].any():
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
        if display_df["below_mol"].any():
            legends.append(
                ":orange[■] Found in NCBI but would not have passed the MOL filter"
            )
        if legends:
            st.caption(" · ".join(legends))

        styled_df = styled_or_plain(
            display_df, highlight_below_mol, highlight_low_identity
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
                below=("below_mol", "any"),
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
        if display_df["below_mol"].any():
            legends.append(
                ":orange[■] Found in NCBI but would not have passed the MOL filter"
            )
        if display_df["scientificName"].isna().any():
            legends.append(":gray[■] No species assigned")
        if legends:
            st.caption(" · ".join(legends))

        styled_df = (
            styled_or_plain(display_df, highlight_below_mol)
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


def _species_breakdown_columns(sp_df, is_hypo):
    """Build column_order and column_config for the per-species breakdown table."""
    if is_hypo:
        col_order = [
            "seq_id",
            "validation_step",
            "proxy_identity_percentage",
            "proxy_query_cover",
            "gbif_occurrences",
        ]
        col_config = {
            "seq_id": COL["seq_id"],
            "validation_step": RESULTS_COLUMN_CONFIG["validation_step"],
            "proxy_identity_percentage": st.column_config.NumberColumn(
                "Identity (%)",
                format="%.2f",
                help="BLAST identity from CO1 proxy validation.",
            ),
            "proxy_query_cover": st.column_config.NumberColumn(
                "Query Cover (%)",
                format="%.2f",
                help="Query coverage from CO1 proxy validation.",
            ),
            "gbif_occurrences": COL["gbif_occurrences"],
        }
        if sp_df["ncbi_top_identity_percentage"].notna().any():
            col_order.extend(["ncbi_top_identity_percentage", "ncbi_top_query_cover"])
            col_config["ncbi_top_identity_percentage"] = st.column_config.NumberColumn(
                "Check Identity (%)",
                format="%.2f",
                help="NCBI marker check identity (species found for target marker).",
            )
            col_config["ncbi_top_query_cover"] = st.column_config.NumberColumn(
                "Check Query Cover (%)",
                format="%.2f",
                help="NCBI marker check query coverage.",
            )
    else:
        col_order = [
            "seq_id",
            "validation_step",
            "ncbi_top_identity_percentage",
            "ncbi_top_query_cover",
            "gbif_occurrences",
        ]
        col_config = {
            "seq_id": COL["seq_id"],
            "validation_step": RESULTS_COLUMN_CONFIG["validation_step"],
            "ncbi_top_identity_percentage": RESULTS_COLUMN_CONFIG[
                "ncbi_top_identity_percentage"
            ],
            "ncbi_top_query_cover": RESULTS_COLUMN_CONFIG["ncbi_top_query_cover"],
            "gbif_occurrences": COL["gbif_occurrences"],
        }

    col_config["low_identity_warning"] = None
    return col_order, col_config


def _show_species_breakdown_table(df, is_hypo, threshold):
    """Render a styled breakdown table for a subset of species rows."""
    col_order, col_config = _species_breakdown_columns(df, is_hypo)
    col_config["below_mol"] = None
    display_df = add_low_identity_warning(df.copy(), threshold)
    styled_df = styled_or_plain(display_df, highlight_below_mol, highlight_low_identity)
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_order=col_order,
        column_config=col_config,
    )


def _show_all_hits_expander(species, is_hypo):
    """Expander showing all individual validation hits for a species."""
    if is_hypo:
        hypo_filter_df = st.session_state.get("hypo_filter_df")
        if hypo_filter_df is None:
            return
        hits = hypo_filter_df[hypo_filter_df["scientificName"] == species]
        if hits.empty:
            return

        col_order = [
            "seq_id",
            "scientificName_hit",
            "identity_percentage",
            "query_cover",
            "hit_url",
        ]
        col_config = {
            "seq_id": COL["seq_id"],
            "scientificName_hit": st.column_config.TextColumn("Proxy Species"),
            "identity_percentage": COL["identity_percentage"],
            "query_cover": COL["query_cover"],
            "hit_url": COL["hit_url"],
        }
        if "seq_url" in hits.columns:
            col_order.append("seq_url")
            col_config["seq_url"] = st.column_config.LinkColumn(
                "BOLD Link", display_text="🔗 View"
            )

        with st.expander(f"All proxy validation hits ({len(hits)})"):
            st.dataframe(
                hits,
                width="stretch",
                hide_index=True,
                column_order=col_order,
                column_config=col_config,
            )
    else:
        mol_df = st.session_state.get("mol_df")
        if mol_df is None:
            return
        hits = mol_df[mol_df["scientificName"] == species]
        if hits.empty:
            return

        with st.expander(f"All NCBI hits ({len(hits)})"):
            st.dataframe(
                hits,
                width="stretch",
                hide_index=True,
                column_order=[
                    "seq_id",
                    "hit_def",
                    "identity_percentage",
                    "query_cover",
                    "hit_url",
                ],
                column_config={
                    "seq_id": COL["seq_id"],
                    "hit_def": COL["hit_def"],
                    "identity_percentage": COL["identity_percentage"],
                    "query_cover": COL["query_cover"],
                    "hit_url": COL["hit_url"],
                },
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


# Search steps whose query date reflects the state of a live external database.
_SOURCE_STEPS = [
    ("ncbi_search", "NCBI", "MOL"),
    ("worms_search", "WoRMS", "TAX"),
    ("gbif_search", "GBIF", "GEO"),
    ("bold_search", "BOLD", "EXTRA"),
    ("hypo_search", "NCBI", "HYPO"),
]


def show_provenance_section():
    """List the date each external database was queried, plus trident version."""
    db_path_str = st.session_state.get("db_path")
    if not db_path_str or not Path(db_path_str).is_file():
        return

    prov = load_provenance(db_path_str)
    if prov.empty:
        return

    versions = prov["trident_version"].dropna()
    version = versions.iloc[0] if not versions.empty else "unknown"

    rows = []
    for table, source, step in _SOURCE_STEPS:
        dates = sorted(prov.loc[prov["step"] == table, "queried_on"].dropna().unique())
        if not dates:
            continue
        when = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
        rows.append({"Source": source, "Step": step, "Queried (UTC)": when})

    if not rows:
        return

    with st.expander("🔎 Data sources & provenance", expanded=False):
        st.caption(
            f"Produced with trident {version}. Dates are when each live database "
            "was queried (UTC); they reflect the database state used for this analysis."
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


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
        ncbi_search_df=st.session_state.get("ncbi_search_df"),
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
