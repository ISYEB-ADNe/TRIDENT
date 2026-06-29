"""
HYPO Analysis UI Module

Handles UI rendering and user interaction for the hypothetical species
validation pipeline (BOLD sequences validated via NCBI BLAST).
"""

import pandas as pd
import streamlit as st

from loguru import logger

from trident.clients.ncbi import set_ncbi_email
from trident.core import config
import trident.pipelines as pipe
from trident.logging import get_log_level
from trident.pipelines.results_pipeline import add_below_mol

from trident.ui.ui import (
    styled_or_plain,
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
    highlight_below_mol,
)

from trident.ui.defaults import (
    HYPO_MAX_HITS_DEFAULT,
    HYPO_EV_EXPONENT_DEFAULT,
    HYPO_IDENTITY_CUTOFF_DEFAULT,
    HYPO_NTOP_DEFAULT,
    HYPO_NUM_THREADS_DEFAULT,
    HYPO_BATCH_SIZE_DEFAULT,
    HYPO_QUERY_COVER_DEFAULT,
    HYPO_IDENTITY_DEFAULT,
    HYPO_CHECK_EVALUE_EXPONENT_DEFAULT,
)


# --------------------------------------------------------------------
# SEARCH FUNCTIONS
# -------------------------------------------------------------------


def display_hypo_parameters():
    """Renders parameters for HYPO BLAST search."""

    db_cols = [
        "hypo_max_hits",
        "hypo_ev_exponent",
        "hypo_identity_cutoff",
        "hypo_ntop",
    ]

    presets = get_recent_param_sets(
        st.session_state.db_path, "hypo_search_inputs", db_cols
    )
    param_preset_selector(
        presets,
        format_fn=lambda p: (
            f"E-value: 1e-{p['hypo_ev_exponent']}, Max hits: {p['hypo_max_hits']}, Top: {p['hypo_ntop']}"
        ),
        key="hypo_search_preset",
    )
    stored = st.session_state.hypo_merge_params

    def_hits = int(stored.get("hypo_max_hits") or HYPO_MAX_HITS_DEFAULT)
    default_ev_exp = int(stored.get("hypo_ev_exponent") or HYPO_EV_EXPONENT_DEFAULT)
    def_id_cutoff = int(
        stored.get("hypo_identity_cutoff") or HYPO_IDENTITY_CUTOFF_DEFAULT
    )
    def_ntop = int(stored.get("hypo_ntop") or HYPO_NTOP_DEFAULT)
    # --- Row 1: Primary Scientific Parameters ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        persist_value("hypo_max_hits", def_hits)
        max_hits = st.number_input(
            "Max hits per query",
            1,
            1000,
            help="Number of NCBI hits returned per sequence",
            key="hypo_max_hits",
        )
        save_widget("hypo_max_hits")
        active_hits = stored.get("hypo_max_hits", "None")
        st.caption(f"Active in results: **{active_hits}**")

    with col2:
        persist_value("hypo_ev_exponent", default_ev_exp)
        ev_exponent = st.number_input(
            "E-value exponent ($10^{-x}$)",
            0,
            50,
            help="Exponent of the significance threshold for matches",
            key="hypo_ev_exponent",
        )
        save_widget("hypo_ev_exponent")
        active_ev = stored.get("hypo_ev_exponent", "None")
        threshold_str = (
            f"(e-value: $10^{{-{active_ev}}}$)" if active_ev != "None" else ""
        )
        st.caption(f"Active in results: **{active_ev}** {threshold_str}")

    with col3:
        persist_value("hypo_identity_cutoff", def_id_cutoff)
        identity_cutoff = st.number_input(
            "Identity Cutoff (%)",
            0,
            100,
            help="Minimum identity percentage to consider a valid NCBI hit",
            key="hypo_identity_cutoff",
        )
        save_widget("hypo_identity_cutoff")
        active_id = stored.get("hypo_identity_cutoff", "None")
        st.caption(f"Active in results: **{active_id}**%")

    with col4:
        persist_value("hypo_ntop", def_ntop)
        ntop = st.number_input(
            "Top N hits to retain",
            1,
            100,
            help="Number of top NCBI hits to retain per species after applying identity cutoff",
            key="hypo_ntop",
        )
        save_widget("hypo_ntop")
        active_ntop = stored.get("hypo_ntop", "None")
        st.caption(f"Active in results: **{active_ntop}**")

    # --- Row 2: Advanced Settings  ---
    with st.expander(
        "🛠️ Advanced Search Settings",
        expanded=False,
    ):
        st.markdown(
            "These settings affect processing speed but not biological results, and should be left at default values most of the time."
        )
        tcol1, tcol2 = st.columns(2)

        with tcol1:
            persist_value("hypo_batch_size", HYPO_BATCH_SIZE_DEFAULT)
            batch_size = st.number_input(
                "Batch size",
                1,
                100,
                help="Number of sequences processed in each batch for NCBI BLAST requests",
                key="hypo_batch_size",
            )
            save_widget("hypo_batch_size")
        with tcol2:
            persist_value("hypo_num_threads", HYPO_NUM_THREADS_DEFAULT)
            threads = st.number_input(
                "Parallel Threads",
                1,
                10,
                help="Number of batches processed in parallel for NCBI BLAST requests",
                key="hypo_num_threads",
            )
            save_widget("hypo_num_threads")

    return max_hits, ev_exponent, identity_cutoff, ntop, batch_size, threads


def hypo_search_job(*, force_rerun, retry_empty=False, status, progress_bar, params):
    """Run HYPO BLAST search + merge job."""

    # 1. Setup Logging
    logsink = StreamlitLogSink(status, prefix="hypo_")
    handler_id = logger.add(
        logsink.write, filter=logsink.thread_filter, level=get_log_level()
    )

    # 2. Get Data
    extra_df = st.session_state.extra_df
    geo_df = st.session_state.geo_df
    sequences_dict = pipe.prepare_hypo_input(extra_df, geo_df)
    email = config.contact_email()
    total_sequences = len(extra_df)
    progress_info = {"current": 0, "total": total_sequences}
    name_str = "sequence" if total_sequences == 1 else "sequences"

    # 3. Search (background thread for progress tracking)
    def _poll():
        curr = progress_info["current"]
        if total_sequences > 0:
            percent = min(curr / total_sequences, 1.0)
            progress_bar.progress(
                percent,
                text=f"Search: {curr} / {total_sequences} {name_str} blasted...",
            )

    hypo_search_df, hypo_search_params = run_with_progress(
        logsink.wrap(pipe.run_hypo_search, before=lambda: set_ncbi_email(email)),
        sequences_dict,
        batch_size=params["batch_size"],
        num_threads=params["num_threads"],
        ev_exponent=params["hypo_ev_exponent"],
        max_hits=params["hypo_max_hits"],
        identity_cutoff=params["hypo_identity_cutoff"],
        ntop=params["hypo_ntop"],
        force_rerun=force_rerun,
        retry_empty=retry_empty,
        db_path=st.session_state.db_path,
        progress_handler=progress_info,
        logsink=logsink,
        on_poll=_poll,
    )

    # 4. Merge (fast — no progress bar needed)
    hypo_merge_df, hypo_merge_params = pipe.run_hypo_merge(
        hypo_search_df,
        extra_df,
        hypo_search_params=hypo_search_params,
        extra_params=st.session_state.extra_params,
        force_rerun=force_rerun,
        db_path=st.session_state.db_path,
    )
    logsink.flush_to_ui()

    # 5. Save Results and Cleanup
    st.session_state.hypo_merge_df = hypo_merge_df
    st.session_state.hypo_merge_params = hypo_merge_params
    logger.remove(handler_id)
    progress_bar.empty()
    reset_state_after("hypo_merge_df")


def hypo_search_workflow():
    """Handle HYPO BLAST search execution."""

    with st.container(border=True):
        st.markdown("### ⚙️ Search Configuration and Execution")

        # Parameter Inputs (NCBI BLAST parameters)
        max_hits, ev_exponent, identity_cutoff, ntop, batch_size, n_threads = (
            display_hypo_parameters()
        )
        current_params = {
            "hypo_max_hits": max_hits,
            "hypo_ev_exponent": ev_exponent,
            "hypo_identity_cutoff": identity_cutoff,
            "hypo_ntop": ntop,
            "batch_size": batch_size,
            "num_threads": n_threads,
        }
        st.divider()

        unique_records = len(st.session_state.extra_df)

        mode = run_step_workflow(
            df_key="hypo_merge_df",
            params_key="hypo_merge_params",
            flag_key="hypo_search_flag",
            tab_id="hypo",
            compare_keys=[
                "hypo_max_hits",
                "hypo_ev_exponent",
                "hypo_identity_cutoff",
                "hypo_ntop",
            ],
            current_params=current_params,
            job_fn=hypo_search_job,
            job_kwargs={"params": current_params},
            new_string=f"Ready to analyze **{unique_records:,}** BOLD sequences against NCBI.",
            btn_key="hypo_search_apply",
            logs_prefix="hypo_",
            threaded=True,
            status_label=f"🔍 Running NCBI BLAST on **{unique_records}** sequences...",
        )

    # Raw Results Preview
    if mode != "NEW" and st.session_state.hypo_merge_df is not None:
        with st.expander(
            f"Inspect full HYPO search results ({len(st.session_state.hypo_merge_df)} records)",
            expanded=False,
        ):
            st.dataframe(
                st.session_state.hypo_merge_df,
                width="stretch",
                hide_index=True,
                column_order=[
                    "seq_id",
                    "scientificName",
                    "scientificName_hit",
                    "identity_percentage",
                    "query_cover",
                    "hit_url",
                    "hit_def",
                ],
                column_config={
                    "seq_id": COL["seq_id"],
                    "scientificName": COL["scientificName"],
                    "scientificName_hit": st.column_config.TextColumn(
                        "BLAST Hit Species"
                    ),
                    "identity_percentage": COL["identity_percentage"],
                    "query_cover": COL["query_cover"],
                    "hit_url": COL["hit_url"],
                    "hit_def": COL["hit_def"],
                },
            )


# --------------------------------------------------------------------
# FILTER FUNCTIONS
# -------------------------------------------------------------------


def hypo_filter_controls():
    """Render HYPO filtering parameter controls and return current values."""

    db_cols = ["hypo_query_cover", "hypo_identity"]

    presets = get_recent_param_sets(
        st.session_state.db_path, "hypo_filter_inputs", db_cols
    )
    param_preset_selector(
        presets,
        format_fn=lambda p: (
            f"QC: {p['hypo_query_cover']}%, identity: {p['hypo_identity']}%"
        ),
        key="hypo_filter_preset",
    )
    stored = st.session_state.hypo_filter_params

    def_qc = int(stored.get("hypo_query_cover") or HYPO_QUERY_COVER_DEFAULT)
    def_identity = int(stored.get("hypo_identity") or HYPO_IDENTITY_DEFAULT)

    col1, col2 = st.columns(2)
    with col1:
        persist_value("hypo_query_cover", def_qc)
        query_cover = st.slider(
            "Query Coverage (%)",
            min_value=0,
            max_value=100,
            help="Minimum query cover percentage to filter NCBI hits",
            key="hypo_query_cover",
        )
        save_widget("hypo_query_cover")
        active_qc = stored.get("hypo_query_cover", "None")
        st.caption(f"Active in results: **{active_qc}**")
    with col2:
        persist_value("hypo_identity", def_identity)
        identity = st.slider(
            "Identity (%)",
            min_value=90,
            max_value=100,
            step=1,
            help="Minimum identity percentage to filter NCBI hits",
            key="hypo_identity",
        )
        save_widget("hypo_identity")
        active_id = stored.get("hypo_identity", "None")
        st.caption(f"Active in results: **{active_id}**")

    return query_cover, identity


def hypo_filter_job(*, force_rerun, retry_empty=False, query_cover, identity):
    """Filter HYPO results based on current filter settings."""

    hypo_filter_df, hypo_filter_params = pipe.run_hypo_filter(
        st.session_state.hypo_merge_df,
        query_cover=query_cover,
        identity=identity,
        hypo_merge_params=st.session_state.hypo_merge_params,
        force_rerun=force_rerun,
        db_path=st.session_state.db_path,
    )

    st.session_state.hypo_filter_df = hypo_filter_df
    st.session_state.hypo_filter_params = hypo_filter_params
    reset_state_after("hypo_filter_df")


def _filter_success_message():
    """Show success message when filter results exist."""
    hypo_filter_df = st.session_state.hypo_filter_df
    if hypo_filter_df is None:
        return

    merged = st.session_state.hypo_merge_df
    total = len(merged) if merged is not None else 0
    base = (
        f"✅ Filtering complete: {len(hypo_filter_df):,}/{total:,} sequences retained"
    )

    # Empty filter results come back from the cache with no columns at all.
    if "scientificName" in hypo_filter_df.columns and not hypo_filter_df.empty:
        n_species = hypo_filter_df["scientificName"].nunique()
        st.success(f"{base} ({n_species:,} unique species).")
    else:
        st.success(f"{base}.")


def hypo_filter_workflow():
    """Main HYPO filter controller."""

    with st.container(border=True):
        st.markdown("### 🔍 Filtering Search Results")

        # Parameters Inputs
        qc, identity = hypo_filter_controls()
        current_params = {
            "hypo_query_cover": qc,
            "hypo_identity": identity,
        }
        st.divider()

        run_step_workflow(
            df_key="hypo_filter_df",
            params_key="hypo_filter_params",
            flag_key="hypo_filter_flag",
            tab_id="hypo",
            compare_keys=["hypo_query_cover", "hypo_identity"],
            current_params=current_params,
            job_fn=hypo_filter_job,
            job_kwargs={"query_cover": qc, "identity": identity},
            new_string=f"Ready to filter **{len(st.session_state.hypo_merge_df):,}** sequences.",
            btn_key="hypo_filter_apply",
            status_label="🔬 Filtering HYPO results...",
            before_button=_filter_success_message,
        )

    # Results Preview
    if st.session_state.hypo_filter_df is not None:
        with st.expander(
            f"Inspect filtered HYPO results ({len(st.session_state.hypo_filter_df)} records)",
            expanded=False,
        ):
            st.dataframe(
                st.session_state.hypo_filter_df,
                width="stretch",
                hide_index=True,
                column_order=[
                    "seq_id",
                    "scientificName",
                    "scientificName_hit",
                    "identity_percentage",
                    "query_cover",
                    "hit_url",
                    "hit_def",
                ],
                column_config={
                    "seq_id": COL["seq_id"],
                    "scientificName": COL["scientificName"],
                    "scientificName_hit": st.column_config.TextColumn(
                        "BLAST Hit Species"
                    ),
                    "identity_percentage": COL["identity_percentage"],
                    "query_cover": COL["query_cover"],
                    "hit_url": COL["hit_url"],
                    "hit_def": COL["hit_def"],
                },
            )


# --------------------------------------------------------------------
# CHECK FUNCTIONS
# -------------------------------------------------------------------


def display_hypo_check_controls():
    """Display HYPO check controls and return current filter settings."""

    db_cols = ["hypo_check_ev_exponent"]

    presets = get_recent_param_sets(
        st.session_state.db_path, "hypo_check_params", db_cols
    )
    param_preset_selector(
        presets,
        format_fn=lambda p: f"E-value: 1e-{p['hypo_check_ev_exponent']}",
        key="hypo_check_preset",
    )
    stored = st.session_state.hypo_check_params

    default_ev_exp = int(
        stored.get("hypo_check_ev_exponent") or HYPO_CHECK_EVALUE_EXPONENT_DEFAULT
    )

    with st.expander(
        "🛠️ Advanced Search Settings",
        expanded=False,
    ):
        persist_value("hypo_check_ev_exponent", default_ev_exp)
        ev_exponent = st.number_input(
            "E-value exponent ($10^{-x}$)",
            0,
            10,
            help="Exponent of the significance threshold for matches",
            key="hypo_check_ev_exponent",
        )
        save_widget("hypo_check_ev_exponent")
        active_ev = stored.get("hypo_check_ev_exponent", "None")
        threshold_str = (
            f"(e-value: $10^{{-{active_ev}}}$)" if active_ev != "None" else ""
        )
        st.caption(f"Active in results: **{active_ev}** {threshold_str}")

    return ev_exponent


def hypo_check_job(*, force_rerun, retry_empty=False, status, progress_bar, params):
    """Run HYPO BLAST check + finalize job."""

    # Setup Logging
    logsink = StreamlitLogSink(status, prefix="hypo_check_")
    handler_id = logger.add(
        logsink.write, filter=logsink.thread_filter, level=get_log_level()
    )

    # Get Data
    hypo_filter_df = st.session_state.hypo_filter_df
    email = config.contact_email()
    sequences_df = (
        hypo_filter_df[["seq_id", "scientificName", "dna_sequence"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    total_sequences = len(sequences_df)
    progress_info = {"current": 0, "total": total_sequences}
    name_str = "sequence" if total_sequences == 1 else "sequences"

    # Check (background thread for progress tracking)
    def _poll():
        curr = progress_info["current"]
        if total_sequences > 0:
            percent = min(curr / total_sequences, 1.0)
            progress_bar.progress(
                percent,
                text=f"Check: {curr} / {total_sequences} {name_str} processed...",
            )

    # Save check results
    hypo_check_df, hypo_check_params = run_with_progress(
        logsink.wrap(pipe.run_hypo_check, before=lambda: set_ncbi_email(email)),
        hypo_filter_df=hypo_filter_df,
        ev_exponent=params["hypo_check_ev_exponent"],
        db_path=st.session_state.db_path,
        force_rerun=force_rerun,
        retry_empty=retry_empty,
        progress_handler=progress_info,
        logsink=logsink,
        on_poll=_poll,
    )
    st.session_state.hypo_check_df = hypo_check_df
    st.session_state.hypo_check_params = hypo_check_params
    st.session_state.hypo_check_summary_df = pipe.build_hypo_check_summary(
        hypo_check_df
    )

    # Finalize — build final HYPO species list
    hypo_df, hypo_params = pipe.finalize_hypo_results(
        hypo_filter_df=hypo_filter_df,
        hypo_check_df=hypo_check_df,
        hypo_filter_params=st.session_state.hypo_filter_params,
        hypo_check_params=hypo_check_params,
        db_path=st.session_state.db_path,
        force_rerun=force_rerun,
    )
    st.session_state.hypo_df = hypo_df
    st.session_state.hypo_params = hypo_params

    progress_bar.empty()
    logger.remove(handler_id)


def hypo_check_display():
    res_df = st.session_state.hypo_check_df
    summary_df = st.session_state.hypo_check_summary_df
    n_found = res_df[res_df["hit_found"].astype(bool)]["scientificName"].nunique()
    n_not_found = res_df[~res_df["hit_found"].astype(bool)]["scientificName"].nunique()

    tab_summary, tab_no_hits, tab_full = st.tabs(
        [
            f"Species Found ({n_found})",
            f"Species Not Found ({n_not_found})",
            "Raw BLAST Data",
        ]
    )
    with tab_summary:
        found_df = add_below_mol(
            summary_df[summary_df["hit_count"] > 0].copy(),
            st.session_state.mol_df,
            st.session_state.ncbi_search_df,
            identity_col="identity_percentage",
        )

        st.caption("Best hit matching in the NCBI database.")
        if found_df["below_mol"].any():
            st.caption(
                ":orange[■] Found in NCBI but would not have passed the MOL filter"
            )
        st.dataframe(
            styled_or_plain(found_df, highlight_below_mol),
            width="stretch",
            hide_index=True,
            column_config={
                "seq_id": COL["seq_id"],
                "scientificName": COL["scientificName"],
                "hit_count": COL["hit_count"],
                "identity_percentage": COL["identity_percentage"],
                "query_cover": COL["query_cover"],
                "hit_url": COL["hit_url"],
                "below_mol": None,
            },
        )

    with tab_no_hits:
        st.caption("No NCBI matches were found for these species.")
        st.dataframe(
            summary_df[summary_df["hit_count"] == 0],
            width="stretch",
            hide_index=True,
            column_config={
                "seq_id": COL["seq_id"],
                "scientificName": COL["scientificName"],
                "hit_count": None,
                "identity_percentage": None,
                "query_cover": None,
                "hit_url": None,
            },
        )
    with tab_full:
        st.dataframe(
            res_df,
            width="stretch",
            hide_index=True,
            column_order=[
                "seq_id",
                "scientificName",
                "hit_found",
                "identity_percentage",
                "query_cover",
                "hit_url",
                "hit_def",
                "scientificName_NCBI",
            ],
            column_config={
                "seq_id": COL["seq_id"],
                "scientificName": COL["scientificName"],
                "hit_found": st.column_config.CheckboxColumn("Hit Found"),
                "identity_percentage": COL["identity_percentage"],
                "query_cover": COL["query_cover"],
                "hit_url": COL["hit_url"],
                "hit_def": COL["hit_def"],
                "scientificName_NCBI": st.column_config.TextColumn("NCBI Species"),
                "dna_sequence": None,
            },
        )


def hypo_final_check_workflow():
    """Handle NCBI final check execution."""

    with st.container(border=True):
        st.markdown("### 🔬 NCBI Marker Verification")
        st.caption(
            "Verify whether each candidate species actually has the target marker "
            "in GenBank by BLASTing the original sequences against NCBI."
        )

        # Parameter Inputs
        ev_exponent = display_hypo_check_controls()
        current_params = {
            "hypo_check_ev_exponent": ev_exponent,
        }
        st.divider()

        hypo_filter_df = st.session_state.hypo_filter_df
        if hypo_filter_df.empty:
            st.info("No HYPO species to check.")
            return
        unique_species = hypo_filter_df["scientificName"].nunique()

        mode = run_step_workflow(
            df_key="hypo_check_df",
            params_key="hypo_check_params",
            flag_key="hypo_check_flag",
            tab_id="hypo",
            compare_keys=["hypo_check_ev_exponent"],
            current_params=current_params,
            job_fn=hypo_check_job,
            job_kwargs={"params": current_params},
            new_string=f"Ready to check **{unique_species:,}** species against NCBI.",
            btn_key="hypo_check_apply",
            logs_prefix="hypo_check_",
            threaded=True,
            status_label="🔍 Running NCBI BLAST...",
        )

        # Results
        if mode != "NEW" and st.session_state.hypo_check_df is not None:
            st.divider()
            st.markdown("#### Verification Results")
            hypo_check_display()


# --------------------------------------------------------------------
# DISPLAY FUNCTIONS
# -------------------------------------------------------------------


def render_hypo_metrics():
    df = st.session_state.hypo_df

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Total Records",
        len(df),
        help="Total number of validated HYPO records",
    )
    c2.metric(
        "Confirmed Species",
        df["scientificName"].nunique(),
        help="Unique species validated via the HYPO pipeline",
    )
    c3.metric(
        "Sequences with Confirmed Species",
        df["seq_id"].nunique(),
        help="Number of initial sequences that have at least one confirmed species via HYPO",
    )


def display_hypo_sequence_metrics(seq_df):
    c1, c2 = st.columns(2)
    c1.metric(
        "Number of Records",
        len(seq_df),
        help="Number of BOLD sequences validated via NCBI for this sequence ID",
    )
    c2.metric(
        "Confirmed Species",
        seq_df["scientificName"].nunique(),
        help="Unique species validated via the HYPO pipeline for this sequence ID",
    )


def display_hypo_sequence_table(seq_df):
    display_df = add_below_mol(
        seq_df.copy(),
        st.session_state.mol_df,
        st.session_state.ncbi_search_df,
    )
    if display_df["below_mol"].any():
        st.caption(":orange[■] Found in NCBI but would not have passed the MOL filter")
    st.dataframe(
        styled_or_plain(display_df, highlight_below_mol),
        width="stretch",
        hide_index=True,
        column_order=[
            "scientificName",
            "taxonURL",
            "genus",
            "specificEpithet",
            "family",
            "order",
            "class",
            "phylum",
            "kingdom",
            "scientificNameAuthorship",
        ],
        column_config={
            "seq_id": None,
            "dna_sequence": None,
            "scientificName": COL["scientificName"],
            "taxonURL": COL["taxonURL"],
            "family": COL["family"],
            "genus": COL["genus"],
            "specificEpithet": COL["specificEpithet"],
            "scientificNameAuthorship": COL["scientificNameAuthorship"],
            "taxonID": None,
            "kingdom": COL["kingdom"],
            "phylum": COL["phylum"],
            "class": COL["class"],
            "order": COL["order"],
            "below_mol": None,
        },
    )


def display_hypo_sequence():
    """Display detailed HYPO results for a selected sequence."""

    selected_seq = sequence_selector()
    df = st.session_state.hypo_df
    seq_df = df[(df["seq_id"] == selected_seq)]
    if seq_df.empty:
        st.info("No sequences retained for this step.")
        return

    display_hypo_sequence_metrics(seq_df)
    display_hypo_sequence_table(seq_df)


def display_hypo_summary():
    """Show HYPO summary: each confirmed (seq_id, species) with validating proxies."""
    hypo_df = st.session_state.hypo_df
    hypo_filter_df = st.session_state.hypo_filter_df

    # All unique proxy species per (seq_id, scientificName)
    proxies = (
        hypo_filter_df.groupby(["seq_id", "scientificName"], observed=False)[
            "scientificName_hit"
        ]
        .apply(lambda s: ", ".join(sorted(s.unique())))
        .reset_index(name="validated_by")
    )

    # NCBI check marker status
    check_cols = ["seq_id", "scientificName", "ncbi_top_identity_percentage"]
    check_status = hypo_df[[c for c in check_cols if c in hypo_df.columns]].copy()
    if "ncbi_top_identity_percentage" not in check_status.columns:
        check_status["ncbi_top_identity_percentage"] = pd.NA
    check_status["in_ncbi"] = check_status["ncbi_top_identity_percentage"].notna()
    check_status = add_below_mol(
        check_status,
        st.session_state.mol_df,
        st.session_state.ncbi_search_df,
    )

    summary = proxies.merge(
        check_status[
            [
                "seq_id",
                "scientificName",
                "in_ncbi",
                "ncbi_top_identity_percentage",
                "below_mol",
            ]
        ],
        on=["seq_id", "scientificName"],
        how="left",
    )
    summary["in_ncbi"] = summary["in_ncbi"].fillna(False)
    summary["below_mol"] = summary["below_mol"].fillna(False)

    if summary["below_mol"].any():
        st.caption(":orange[■] Found in NCBI but would not have passed the MOL filter")

    st.dataframe(
        styled_or_plain(summary, highlight_below_mol),
        width="stretch",
        hide_index=True,
        column_config={
            "seq_id": COL["seq_id"],
            "scientificName": COL["scientificName"],
            "validated_by": st.column_config.TextColumn(
                "Validated By (CO1 proxy)",
                help="BOLD proxy species whose CO1 sequence confirmed this candidate.",
            ),
            "in_ncbi": st.column_config.CheckboxColumn(
                "In NCBI",
                help="Whether the species was also found in the NCBI marker check.",
            ),
            "ncbi_top_identity_percentage": st.column_config.NumberColumn(
                "NCBI Identity (%)",
                format="%.2f",
                help="Identity of the best NCBI hit for the target marker.",
            ),
            "below_mol": None,
        },
    )


def display_hypo_results():
    """Display final HYPO results after check."""

    with st.container(border=True):
        st.markdown("### 📊 HYPO List Overview")
        render_hypo_metrics()
        display_hypo_summary()

        with st.expander(
            f"View full HYPO list ({len(st.session_state.hypo_df)} records)",
            expanded=False,
        ):
            st.dataframe(
                st.session_state.hypo_df,
                width="stretch",
                hide_index=True,
                column_order=[
                    "seq_id",
                    "scientificName",
                    "taxonURL",
                    "scientificName_hit",
                    "identity_percentage",
                    "query_cover",
                    "ncbi_top_identity_percentage",
                    "ncbi_top_query_cover",
                    "family",
                    "order",
                    "class",
                    "phylum",
                    "kingdom",
                ],
                column_config={
                    "seq_id": COL["seq_id"],
                    "scientificName": COL["scientificName"],
                    "taxonURL": COL["taxonURL"],
                    "scientificName_hit": st.column_config.TextColumn(
                        "Validated By (CO1 proxy)"
                    ),
                    "identity_percentage": st.column_config.NumberColumn(
                        "Proxy Identity (%)",
                        format="%.2f",
                        help="BLAST identity of the CO1 proxy hit.",
                    ),
                    "query_cover": st.column_config.NumberColumn(
                        "Proxy Query Cover (%)",
                        format="%.2f",
                        help="Query coverage of the CO1 proxy hit.",
                    ),
                    "ncbi_top_identity_percentage": st.column_config.NumberColumn(
                        "NCBI Identity (%)",
                        format="%.2f",
                    ),
                    "ncbi_top_query_cover": st.column_config.NumberColumn(
                        "NCBI Query Cover (%)",
                        format="%.2f",
                    ),
                    "family": COL["family"],
                    "order": COL["order"],
                    "class": COL["class"],
                    "phylum": COL["phylum"],
                    "kingdom": COL["kingdom"],
                },
            )

    with st.container(border=True):
        st.markdown("### 🔬 HYPO List per (M)OTU/ASV")
        display_hypo_sequence()


# --------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------


def hypo_analysis():
    """Main HYPO analysis workflow — hypothetical species validation."""

    st.header("🔗 HYPO: BOLD/NCBI BLAST Search")
    st.caption(
        "Step 5: BOLD sequences from the EXTRA list are BLASTed against the NCBI database "
        "to check whether each species can be confirmed for its associated input sequence. "
        "A species is validated only if the BLAST hit matches one already linked to that sequence ID "
        "from the earlier pipeline steps (MOL → TAX → GEO)."
    )
    st.divider()

    if not require_prerequisite(
        "extra_df",
        "👈 **EXTRA list not found.** Please complete **Step 4 · EXTRA** "
        "before running the HYPO sequences search.",
    ):
        return

    # Early exit: no BOLD sequences to validate
    if st.session_state.extra_df.empty:
        st.info(
            "ℹ️ **No BOLD sequences to validate.**\n\n"
            "All species were previously validated via NCBI and GBIF in steps 1 to 3."
        )
        st.session_state.hypo_df = pd.DataFrame()
        st.session_state.hypo_params = None
        reset_state_after("hypo_df")
        return

    # HYPO Data Acquisition
    hypo_search_workflow()

    # HYPO Data Filtering
    if st.session_state.hypo_merge_df is not None:
        st.divider()
        hypo_filter_workflow()

    # Final NCBI Check
    if st.session_state.hypo_filter_df is not None:
        st.divider()
        hypo_final_check_workflow()

    # Final HYPO Results (only after check completes)
    if st.session_state.hypo_df is not None and not st.session_state.hypo_df.empty:
        st.divider()
        display_hypo_results()
        next_step_button("Open Results ➔", "results")

    st.divider()
