"""
BOLD Analysis UI Module

This module handles only UI rendering and user interaction.
"""

import streamlit as st
import pandas as pd

from loguru import logger

from trident.core import config
import trident.pipelines as pipe
from trident.logging import get_log_level

from trident.ui.ui import (
    COL,
    get_recent_param_sets,
    param_preset_selector,
    persist_value,
    save_widget,
    StreamlitLogSink,
    run_with_progress,
    reset_state_after,
    sequence_selector,
    run_step_workflow,
    next_step_button,
    require_prerequisite,
    show_missing_results_banner,
)

from trident.ui.defaults import BOLD_SIMILARITY_DEFAULT


# --------------------------------------------------------------------
# SEARCH FUNCTIONS
# -------------------------------------------------------------------


def bold_search_job(*, force_rerun, retry_empty=False, status, progress_bar, params):
    """Run BOLD search for species missing from NCBI."""
    # Setup Logging
    logsink = StreamlitLogSink(status, prefix="bold_")
    handler_id = logger.add(
        logsink.write, filter=logsink.thread_filter, level=get_log_level()
    )

    # Prepare Inputs and Progress Tracker
    geo_df = st.session_state.geo_df
    species_list, _ = pipe.prepare_bold_input(geo_df, st.session_state.ncbi_search_df)
    st.session_state.bold_input_species = species_list
    total_names = len(species_list)
    progress_info = {"current": 0, "total": total_names}

    # Background Execution
    def _poll():
        curr = progress_info["current"]
        if total_names > 0:
            percent = min(curr / total_names, 1.0)
            progress_bar.progress(
                percent,
                text=f"Queried {curr}/{total_names} species from BOLD...",
            )

    bold_search_df, bold_search_params = run_with_progress(
        logsink.wrap(pipe.run_bold_search),
        species_list,
        keep_only_COI5P=params["bold_keep_only_COI5P"],
        keep_ncbi=params["bold_keep_ncbi"],
        db_path=st.session_state.db_path,
        force_rerun=force_rerun,
        retry_empty=retry_empty,
        progress_handler=progress_info,
        user_agent=config.user_agent(),
        logsink=logsink,
        on_poll=_poll,
    )

    # Save Results to Session State
    st.session_state.bold_search_df = bold_search_df
    st.session_state.bold_search_params = bold_search_params

    # Cleanup
    logger.remove(handler_id)
    progress_bar.empty()
    reset_state_after("bold_search_df")


def bold_search_workflow():
    """Handle BOLD search execution."""

    with st.container(border=True):
        st.markdown("### ⚙️ Search Configuration and Execution")

        # Early exit: no species to query
        species_to_query = len(st.session_state.bold_input_species)
        if species_to_query == 0:
            st.success("No species to query BOLD for, all previously validated")
            st.session_state.extra_df = pd.DataFrame()
            st.session_state.extra_params = {}
            reset_state_after("extra_df")
            return

        # Advanced settings
        with st.expander("🛠️ Advanced Search Settings", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                persist_value("bold_keep_only_COI5P", True)
                keep_only_COI5P = st.checkbox(
                    "COI-5P records only",
                    key="bold_keep_only_COI5P",
                    help="Only retain records with the COI-5P barcode marker.",
                )
                save_widget("bold_keep_only_COI5P")
            with c2:
                persist_value("bold_keep_ncbi", False)
                keep_ncbi = st.checkbox(
                    "Include NCBI-mined records",
                    key="bold_keep_ncbi",
                    help="Retain records that BOLD mined from GenBank/NCBI.",
                )
                save_widget("bold_keep_ncbi")

        current_params = {
            "bold_keep_only_COI5P": keep_only_COI5P,
            "bold_keep_ncbi": keep_ncbi,
        }

        mode = run_step_workflow(
            df_key="bold_search_df",
            params_key="bold_search_params",
            flag_key="bold_search_flag",
            tab_id="extra",
            compare_keys=["bold_keep_only_COI5P", "bold_keep_ncbi"],
            current_params=current_params,
            job_fn=bold_search_job,
            job_kwargs={"params": current_params},
            new_string=f"Ready to query BOLD for **{species_to_query}** species.",
            btn_key="bold_search_apply",
            logs_prefix="bold_",
            threaded=True,
            status_label=f"🗺️ Running BOLD search for **{species_to_query}** species...",
        )

    # Raw Results Preview
    if mode != "NEW" and st.session_state.bold_search_df is not None:
        show_missing_results_banner(
            st.session_state.bold_input_species,
            st.session_state.bold_search_df,
            "scientificName",
            item_label="species",
        )
        with st.expander(
            f"Inspect full BOLD search results ({len(st.session_state.bold_search_df)} records)",
            expanded=False,
        ):
            df = st.session_state.bold_search_df
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_order=[
                    "scientificName",
                    "seq_url",
                    "genus",
                    "specificEpithet",
                    "family",
                    "order",
                    "class",
                    "phylum",
                    "kingdom",
                    "dna_sequence",
                ],
                column_config={
                    "seq_id": None,
                    "scientificName": COL["scientificName"],
                    "seq_url": COL["bold_seq_url"],
                    "genus": COL["genus"],
                    "specificEpithet": COL["specificEpithet"],
                    "family": COL["family"],
                    "order": COL["order"],
                    "class": COL["class"],
                    "phylum": COL["phylum"],
                    "kingdom": COL["kingdom"],
                    "dna_sequence": COL["bold_dna_sequence"],
                    "taxonRank": None,
                    "taxonID": None,
                    "taxonID_db": None,
                },
            )


# --------------------------------------------------------------------
# FILTER FUNCTIONS
# -------------------------------------------------------------------


def bold_filter_controls() -> tuple[int, int | None]:
    """Render BOLD filtering parameter controls and return current values."""

    presets = get_recent_param_sets(
        st.session_state.db_path, "bold_filter_inputs", ["bold_similarity"]
    )
    param_preset_selector(
        presets,
        format_fn=lambda p: f"similarity: {p['bold_similarity']}%",
        key="bold_filter_preset",
    )
    stored = st.session_state.extra_params

    def_similarity = int(stored.get("bold_similarity") or BOLD_SIMILARITY_DEFAULT)

    col1, col2 = st.columns(2)
    with col1:
        persist_value("bold_similarity", def_similarity)
        similarity = st.slider(
            "Identity threshold (%)",
            min_value=90,
            max_value=100,
            step=1,
            key="bold_similarity",
            help="Maximum sequence similarity (%) between the longest sequences kept per species.",
        )
        save_widget("bold_similarity")
        active_similarity = stored.get("bold_similarity", "None")
        st.caption(f"Active in results: **{active_similarity}**")
    with col2:
        persist_value("bold_longest_n", None)
        longest_n = st.number_input(
            "Max sequences per species",
            min_value=1,
            value=None,
            step=1,
            key="bold_longest_n",
            placeholder="No limit",
            help="Leave empty to keep all non-redundant sequences.",
        )
        save_widget("bold_longest_n")

    return similarity, longest_n


def bold_filter_job(*, force_rerun, retry_empty=False, status, progress_bar, params):
    """Execution logic for BOLD filtering."""

    # Setup Logging
    logsink = StreamlitLogSink(status, prefix="bold_filter_")
    handler_id = logger.add(
        logsink.write, filter=logsink.thread_filter, level=get_log_level()
    )

    # Prepare Inputs and Progress Tracker
    bold_search_df = st.session_state.bold_search_df
    geo_df = st.session_state.geo_df
    total_species = bold_search_df["scientificName"].nunique()
    progress_info = {"current": 0, "total": total_species}

    # Background Execution
    def _poll():
        curr = progress_info["current"]
        if total_species > 0:
            percent = min(curr / total_species, 1.0)
            progress_bar.progress(
                percent,
                text=f"Filtering: {curr}/{total_species} species...",
            )

    bold_filter_df, bold_filter_params = run_with_progress(
        logsink.wrap(pipe.run_bold_filter),
        bold_search_df,
        similarity=params["bold_similarity"],
        longest_n=params["bold_longest_n"],
        bold_search_params=st.session_state.bold_search_params,
        db_path=st.session_state.db_path,
        force_rerun=force_rerun,
        progress_handler=progress_info,
        logsink=logsink,
        on_poll=_poll,
    )
    bold_merge_df, bold_merge_params = pipe.run_bold_merge(
        bold_filter_df,
        geo_df,
        bold_filter_params=bold_filter_params,
        geo_params=st.session_state.geo_params,
        db_path=st.session_state.db_path,
        force_rerun=force_rerun,
    )
    extra_df, extra_params = pipe.finalize_extra_results(
        bold_merge_df,
        db_path=st.session_state.db_path,
        bold_merge_params=bold_merge_params,
        force_rerun=force_rerun,
    )
    logsink.flush_to_ui()

    # Save Results and UI Cleanup
    progress_bar.empty()

    extra_summary_df = pipe.build_extra_summary(
        extra_df, geo_df, bold_input_species=st.session_state.bold_input_species
    )

    st.session_state.bold_filter_df = bold_filter_df
    bold_species = (
        bold_filter_df["scientificName"].unique().tolist()
        if not bold_filter_df.empty
        else []
    )
    st.session_state.bold_species = bold_species
    st.session_state.bold_missing_species = sorted(
        set(st.session_state.bold_input_species) - set(bold_species)
    )
    st.session_state.extra_summary_df = extra_summary_df
    st.session_state.extra_df = extra_df
    st.session_state.extra_params = extra_params
    reset_state_after("extra_df")
    logger.remove(handler_id)


def _filter_success_message():
    """Show success message when filter results exist."""
    extra_df = st.session_state.extra_df
    if extra_df is not None and st.session_state.bold_filter_df is not None:
        st.success(
            f"✅ Filtering complete: {len(st.session_state.bold_filter_df):,}/{len(st.session_state.bold_search_df):,} sequences retained"
        )


def bold_filter_workflow():
    """Main BOLD filter controller."""
    with st.container(border=True):
        st.markdown("### 🔍 Filtering Search Results")

        if st.session_state.bold_search_df.empty:
            st.info("No BOLD records to filter.")
            st.session_state.extra_df = pd.DataFrame()
            st.session_state.extra_params = {}
            reset_state_after("extra_df")
            return

        # Parameter Inputs
        similarity, longest_n = bold_filter_controls()
        current_params = {
            "bold_similarity": similarity,
            "bold_longest_n": longest_n,
        }
        st.divider()

        run_step_workflow(
            df_key="extra_df",
            params_key="extra_params",
            flag_key="bold_filter_flag",
            tab_id="extra",
            compare_keys=["bold_similarity", "bold_longest_n"],
            current_params=current_params,
            job_fn=bold_filter_job,
            job_kwargs={"params": current_params},
            new_string=f"Ready to filter **{len(st.session_state.bold_search_df)}** sequences.",
            btn_key="bold_filter_apply",
            logs_prefix="bold_filter_",
            threaded=True,
            status_label="🔍 Filtering BOLD results...",
            before_button=_filter_success_message,
        )


# --------------------------------------------------------------------
# DISPLAY FUNCTIONS
# -------------------------------------------------------------------


def display_bold_input_summary():
    """Show non-MOL species from the GEO list that will be queried in BOLD."""

    geo_df = st.session_state.geo_df
    bold_input = st.session_state.bold_input_species

    if not bold_input:
        return

    non_mol_df = geo_df[geo_df["scientificName"].isin(bold_input)]
    n_to_query = len(bold_input)

    with st.container(border=True):
        st.markdown(f"### 📋 Input: {n_to_query} species to query in BOLD")
        st.caption(
            "Species from the GEO list without a direct NCBI hit — to be searched in BOLD."
        )

        priority = [
            "seq_id",
            "scientificName",
            "taxonURL",
            "genus",
            "specificEpithet",
            "family",
            "order",
            "class",
            "phylum",
            "kingdom",
        ]
        cols = non_mol_df.columns.tolist()
        ordered = priority + [c for c in cols if c not in priority]
        st.dataframe(
            non_mol_df,
            hide_index=True,
            width="stretch",
            column_order=ordered,
            column_config={
                "seq_id": COL["seq_id"],
                "scientificName": COL["scientificName"],
                "taxonURL": COL["taxonURL"],
                "genus": COL["genus"],
                "specificEpithet": COL["specificEpithet"],
                "family": COL["family"],
                "order": COL["order"],
                "class": COL["class"],
                "phylum": COL["phylum"],
                "kingdom": COL["kingdom"],
                "in_mol": None,
                "mol_top_identity_percentage": None,
                "mol_top_query_cover": None,
                "gbif_occurrences": None,
                "gbif_taxonURL": None,
                "dna_sequence": None,
                "scientificNameAuthorship": None,
                "taxonRank": None,
                "taxonID": None,
                "taxonID_db": None,
            },
        )


def render_bold_metrics():
    """Render BOLD results metrics summary."""

    bold_filter_df = st.session_state.bold_filter_df
    missing = st.session_state.bold_missing_species

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Records Retrieved",
        f"{len(bold_filter_df):,}",
        help="Number of sequences retrieved from BOLD.",
    )
    c2.metric(
        "Species Found",
        len(st.session_state.bold_species),
        help="Unique species identified in BOLD results",
    )
    c3.metric(
        "Missing Species",
        len(missing),
        delta=f"{len(missing)} not in BOLD" if missing else "All found",
        delta_color="inverse",
        help="Species queried but not found in BOLD database",
    )
    if missing:
        with st.expander(f"⚠️ {len(missing)} species not found in BOLD"):
            st.dataframe(
                [{"Species Name": n} for n in missing],
                hide_index=True,
                width="stretch",
            )


def display_extra_summary():
    """Display EXTRA results summary tables."""
    tab_seq, tab_species = st.tabs(["per (M)OTU/ASV", "per species"])
    with tab_seq:
        seq_summary = pipe.build_extra_seq_summary(st.session_state.extra_summary_df)
        st.dataframe(
            seq_summary,
            width="stretch",
            hide_index=True,
            column_config={
                "seq_id": COL["seq_id"],
                "total_records": COL["total_records"],
                "species_queried": COL["species_queried"],
                "species_found": COL["species_found"],
                "species_missing": COL["species_missing"],
            },
        )
    with tab_species:
        st.dataframe(
            st.session_state.extra_summary_df,
            width="stretch",
            hide_index=True,
            column_config={
                "seq_id": COL["seq_id"],
                "scientificName": COL["scientificName"],
                "total_records": COL["total_records"],
            },
        )


def display_bold_sequence_metrics(seq_df, selected_seq):
    # Species queried for this seq_id but not found in BOLD
    bold_input = set(st.session_state.bold_input_species)
    geo_df = st.session_state.geo_df
    queried_for_seq = set(
        geo_df.loc[
            (geo_df["seq_id"] == selected_seq)
            & (geo_df["scientificName"].isin(bold_input)),
            "scientificName",
        ]
    )
    found_for_seq = set(seq_df["scientificName"].unique())
    missing_for_seq = sorted(queried_for_seq - found_for_seq)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Records Retrieved",
        len(seq_df),
        help="Number of sequences retrieved from BOLD.",
    )
    c2.metric(
        "Species Found",
        seq_df["scientificName"].nunique(),
        help="Unique species identified in BOLD results",
    )
    c3.metric(
        "Missing Species",
        len(missing_for_seq),
        help="Species queried but not found in BOLD database",
    )
    if missing_for_seq:
        with st.expander(f"⚠️ {len(missing_for_seq)} species not found in BOLD"):
            st.dataframe(
                [{"Species Name": n} for n in missing_for_seq],
                hide_index=True,
                width="stretch",
            )


def display_bold_sequence_table(seq_df):
    display_df = seq_df[["scientificName", "dna_sequence_extra", "seq_url"]].copy()
    display_df["seq_length"] = display_df["dna_sequence_extra"].apply(len)
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_order=["scientificName", "seq_url", "seq_length", "dna_sequence_extra"],
        column_config={
            "scientificName": COL["scientificName"],
            "seq_url": COL["bold_seq_url"],
            "seq_length": st.column_config.NumberColumn("Sequence Length", format="%d"),
            "dna_sequence_extra": COL["bold_dna_sequence"],
        },
    )


def display_bold_sequence():
    """Display detailed BOLD results for a selected sequence."""

    selected_seq = sequence_selector()
    df = st.session_state.extra_df
    seq_df = df[df["seq_id"] == selected_seq]
    if seq_df.empty:
        st.info(f"No BOLD data available for sequence: **{selected_seq}**")
        return

    display_bold_sequence_metrics(seq_df, selected_seq)

    # Per-species breakdown for this sequence
    pair_summary = st.session_state.extra_summary_df
    seq_species = pair_summary[pair_summary["seq_id"] == selected_seq].drop(
        columns="seq_id"
    )
    with st.expander(
        f"Species breakdown ({len(seq_species)} species queried)", expanded=False
    ):
        st.dataframe(
            seq_species,
            width="stretch",
            hide_index=True,
            column_config={
                "scientificName": COL["scientificName"],
                "total_records": COL["total_records"],
            },
        )

    display_bold_sequence_table(seq_df)


def display_bold_results():
    """Display BOLD results."""

    extra_df = st.session_state.extra_df

    # Case where search was performed but 0 records found or skipped
    if extra_df.empty:
        st.success("✅ **All species previously validated — no BOLD search needed.**")
        return

    with st.container(border=True):
        st.markdown("### 📊 EXTRA List Overview")

        render_bold_metrics()
        display_extra_summary()

        with st.expander(
            f"View full EXTRA list ({len(extra_df):,} records)", expanded=False
        ):
            st.dataframe(
                extra_df,
                width="stretch",
                hide_index=True,
                column_order=[
                    "seq_id",
                    "scientificName",
                    "taxonURL",
                    "seq_url",
                    "genus",
                    "specificEpithet",
                    "family",
                    "order",
                    "class",
                    "phylum",
                    "kingdom",
                    "seq_id_extra",
                    "dna_sequence_extra",
                ],
                column_config={
                    "seq_id": COL["seq_id"],
                    "scientificName": COL["scientificName"],
                    "taxonURL": COL["taxonURL"],
                    "seq_url": COL["bold_seq_url"],
                    "genus": COL["genus"],
                    "specificEpithet": COL["specificEpithet"],
                    "family": COL["family"],
                    "order": COL["order"],
                    "class": COL["class"],
                    "phylum": COL["phylum"],
                    "kingdom": COL["kingdom"],
                    "seq_id_extra": st.column_config.TextColumn("BOLD Process ID"),
                    "dna_sequence_extra": COL["bold_dna_sequence"],
                    "dna_sequence": None,
                    "scientificNameAuthorship": None,
                    "taxonRank": None,
                    "taxonID": None,
                    "taxonID_db": None,
                },
            )

    with st.container(border=True):
        st.markdown("### 🔬 EXTRA List per (M)OTU/ASV")
        display_bold_sequence()


# --------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------


def extra_analysis():
    """Main BOLD analysis workflow."""

    st.header("🧬 EXTRA: BOLD Sequences Search")
    st.caption(
        "Step 4: Search the Barcode of Life Data Systems (BOLD) for species identified in GBIF."
        " This step specifically targets species missing from NCBI/BLAST."
    )
    st.divider()

    if not require_prerequisite(
        "geo_df",
        "👈 **GEO list not found.** Please complete **Step 3 · GEO** "
        "to generate the species list required for BOLD querying.",
    ):
        return

    # Compute and cache input species list (avoid re-logging on every Streamlit rerun)
    if not st.session_state.bold_input_species:
        species_list, _ = pipe.prepare_bold_input(
            st.session_state.geo_df, st.session_state.ncbi_search_df
        )
        st.session_state.bold_input_species = species_list

    # Input Summary
    display_bold_input_summary()

    # BOLD Data Acquisition
    bold_search_workflow()

    # BOLD Data Filtering
    if st.session_state.bold_search_df is not None:
        st.divider()
        bold_filter_workflow()

    # Results Display
    if st.session_state.extra_df is not None and not st.session_state.extra_df.empty:
        st.divider()
        display_bold_results()
        next_step_button("Next: HYPO ➔", "hypo")

    st.divider()
