"""
NCBI Analysis UI Module

This module handles only UI rendering and user interaction.
"""

import math

import streamlit as st


from trident.core.utils import pluralize
from trident.clients import ncbi
from trident.clients.ncbi import set_ncbi_email
from trident.core import config
import trident.pipelines as pipe


from trident.ui.ui import (
    failed_items,
    show_failed_items_banner,
    styled_or_plain,
    COL,
    BLAST_HELP,
    get_recent_param_sets,
    param_preset_selector,
    coerce_stored_value,
    mol_query_cover,
    persist_value,
    save_widget,
    reset_state_after,
    highlight_low_identity,
    StreamlitLogSink,
    run_with_progress,
    sequence_selector,
    run_step_workflow,
    next_step_button,
    require_prerequisite,
)

from trident.ui.defaults import (
    BLAST_GIVE_UP_MINUTES_DEFAULT,
    BLAST_GIVE_UP_MINUTES_MIN,
    BLAST_MAX_THREADS,
    NCBI_MAX_HITS_DEFAULT,
    NCBI_EV_EXPONENT_DEFAULT,
    NCBI_NUM_THREADS_DEFAULT,
    NCBI_BATCH_SIZE_DEFAULT,
    NCBI_METHOD_DEFAULT,
    NCBI_QUERY_COVER_DEFAULT,
    NCBI_GAP_SIZE_DEFAULT,
    NCBI_GAP_MIN_TOP_DEFAULT,
    NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT,
    NCBI_ENFORCE_LOW_IDENTITY_DEFAULT,
)

FILTER_METHOD_LABELS = {"barcoding_gap": "Barcoding Gap", "similarity": "Similarity"}

# --------------------------------------------------------------------
# SEARCH FUNCTIONS
# -------------------------------------------------------------------


def display_ncbi_parameters():
    """Renders parameters for NCBI BLAST search."""

    db_path = st.session_state.db_path
    db_cols = ["ncbi_max_hits", "ncbi_ev_exponent"]

    presets = get_recent_param_sets(db_path, "ncbi_search_inputs", db_cols)
    param_preset_selector(
        presets,
        format_fn=lambda p: (
            f"E-value: 1e-{p['ncbi_ev_exponent']}, Max hits: {p['ncbi_max_hits']}"
        ),
        key="ncbi_search_preset",
    )

    stored = st.session_state.ncbi_search_params

    default_hits = stored.get("ncbi_max_hits") or NCBI_MAX_HITS_DEFAULT
    default_ev_exponent = stored.get("ncbi_ev_exponent") or NCBI_EV_EXPONENT_DEFAULT

    # --- Row 1: Primary Scientific Parameters ---
    col1, col2 = st.columns(2)

    with col1:
        persist_value("ncbi_max_hits", int(default_hits))
        max_hits = st.number_input(
            "Max hits per query",
            1,
            1000,
            help=BLAST_HELP["max_hits"],
            key="ncbi_max_hits",
        )
        save_widget("ncbi_max_hits")
        active_hits = stored.get("ncbi_max_hits", "None")
        st.caption(f"Active in results: **{active_hits}**")

    with col2:
        persist_value("ncbi_ev_exponent", int(default_ev_exponent))
        ev_exponent = st.number_input(
            "E-value exponent ($10^{-x}$)",
            0,
            50,
            help=BLAST_HELP["ev_exponent"],
            key="ncbi_ev_exponent",
        )
        save_widget("ncbi_ev_exponent")
        active_ev = stored.get("ncbi_ev_exponent", "None")
        threshold_str = (
            f"(threshold: $10^{{-{active_ev}}}$)" if active_ev != "None" else ""
        )
        st.caption(f"Active in results: **{active_ev}** {threshold_str}")

    # --- Row 2: Advanced Settings  ---
    with st.expander(
        "🛠️ Advanced Search Settings",
        expanded=False,
    ):
        st.markdown(
            "These settings affect processing speed but not biological results, and should be left at default values most of the time."
        )
        tcol1, tcol2, tcol3 = st.columns(3)

        with tcol1:
            persist_value("ncbi_batch_size", NCBI_BATCH_SIZE_DEFAULT)
            batch_size = st.number_input(
                "Batch size",
                1,
                100,
                help=BLAST_HELP["batch_size"],
                key="ncbi_batch_size",
            )
            save_widget("ncbi_batch_size")
        with tcol2:
            persist_value("ncbi_num_threads", NCBI_NUM_THREADS_DEFAULT)
            # A value saved before the cap would be out of the widget's range.
            st.session_state.ncbi_num_threads = min(
                st.session_state.ncbi_num_threads, BLAST_MAX_THREADS
            )
            threads = st.number_input(
                "Parallel Threads",
                1,
                BLAST_MAX_THREADS,
                help=BLAST_HELP["n_threads"],
                key="ncbi_num_threads",
            )
            save_widget("ncbi_num_threads")
        with tcol3:
            persist_value("ncbi_give_up", BLAST_GIVE_UP_MINUTES_DEFAULT)
            give_up = st.number_input(
                "Give up after (min)",
                BLAST_GIVE_UP_MINUTES_MIN,
                None,  # no upper limit: a job waiting longer is still polled
                help=BLAST_HELP["give_up"],
                key="ncbi_give_up",
            )
            save_widget("ncbi_give_up")

    return max_hits, ev_exponent, batch_size, threads, give_up


def ncbi_search_job(*, force_rerun, retry_empty=False, status, progress_bar, params):
    """Run NCBI BLAST with a functioning progress bar and multi-mode support."""

    # 1. Setup Logging
    logsink = StreamlitLogSink(container=status, prefix="ncbi_")

    # 2. Prepare Inputs and Progress Tracker
    sequences = pipe.prepare_ncbi_input(st.session_state.sequences_df)
    email = config.contact_email()
    total_sequences = len(sequences)
    progress_info = {"current": 0, "total": total_sequences}
    progress_placeholder = st.empty()

    # 3. Run NCBI search in a background thread
    def _poll():
        curr_seq, total = progress_info["current"], progress_info["total"]
        if total > 0:
            percent = min(curr_seq / total, 1.0)
            progress_bar.progress(
                percent,
                text=f"Search: {curr_seq}/{total} "
                f"{pluralize(total, 'sequence')} blasted...",
            )

    ncbi_search_df, ncbi_search_params = run_with_progress(
        logsink.wrap(pipe.run_ncbi_search, before=lambda: set_ncbi_email(email)),
        sequences,
        batch_size=params["batch_size"],
        num_threads=params["num_threads"],
        give_up_minutes=params["give_up_minutes"],
        ev_exponent=params["ncbi_ev_exponent"],
        max_hits=params["ncbi_max_hits"],
        force_rerun=force_rerun,
        retry_empty=retry_empty,
        db_path=st.session_state.db_path,
        progress_handler=progress_info,
        logsink=logsink,
        progress=progress_info,
        on_poll=_poll,
    )

    # Final UI update before clearing
    progress_bar.empty()
    progress_placeholder.empty()

    st.session_state.ncbi_search_df = ncbi_search_df
    st.session_state.ncbi_search_params = ncbi_search_params
    st.session_state.ncbi_search_failed = failed_items(ncbi_search_df)

    reset_state_after("ncbi_search_df")
    logsink.close()


def ncbi_search_workflow():
    """Run NCBI BLAST search workflow"""
    with st.container(border=True):
        st.markdown("### ⚙️ Search Configuration and Execution")

        # 1. Parameter inputs
        max_hits, ev_exponent, batch_size, num_threads, give_up = (
            display_ncbi_parameters()
        )
        current_params = {
            "ncbi_max_hits": max_hits,
            "ncbi_ev_exponent": ev_exponent,
            "batch_size": batch_size,
            "num_threads": num_threads,
            "give_up_minutes": give_up,
        }

        st.divider()

        n_seqs = len(st.session_state.sequences_df)
        mode = run_step_workflow(
            df_key="ncbi_search_df",
            params_key="ncbi_search_params",
            flag_key="ncbi_search_flag",
            tab_id="mol",
            compare_keys=["ncbi_max_hits", "ncbi_ev_exponent"],
            current_params=current_params,
            job_fn=ncbi_search_job,
            job_kwargs={"params": current_params},
            new_string=f"Ready to analyze **{n_seqs}** sequences.",
            btn_key="ncbi_search_apply",
            logs_prefix="ncbi_",
            threaded=True,
            status_label=f"🔍 Running NCBI BLAST on **{n_seqs}** sequences...",
        )

    # Search overview and raw results
    if mode == "IDENTICAL":
        show_failed_items_banner(
            st.session_state.ncbi_search_failed,
            "sequence",
            "sequences",
            source="NCBI BLAST",
            gives_up=True,
        )
    if mode != "NEW":
        display_ncbi_search_overview()

        with st.expander(
            f"Inspect raw NCBI/BLAST results ({len(st.session_state.ncbi_search_df)} hits)",
            expanded=False,
        ):
            cols = st.session_state.ncbi_search_df.columns.tolist()
            priority = [
                "seq_id",
                "scientificName",
                "hit_url",
                "identity_percentage",
                "query_cover",
                "hit_def",
            ]
            ordered = priority + [c for c in cols if c not in priority]
            st.dataframe(
                st.session_state.ncbi_search_df[ordered],
                width="stretch",
                hide_index=True,
                column_config={
                    "seq_id": COL["seq_id"],
                    "scientificName": COL["scientificName"],
                    "hit_url": COL["hit_url"],
                    "identity_percentage": COL["identity_percentage"],
                    "query_cover": COL["query_cover"],
                    "hit_def": COL["hit_def"],
                    "dna_sequence": None,
                    "genus": COL["genus"],
                    "specificEpithet": COL["specificEpithet"],
                    "align_length": COL["align_length"],
                    "identities": COL["identities"],
                    "gaps": COL["gaps"],
                    "query_start": COL["query_start"],
                    "query_end": COL["query_end"],
                },
            )


# --------------------------------------------------------------------
# FILTERING FUNCTIONS
# -------------------------------------------------------------------


def ncbi_filter_controls():
    """Render NCBI filtering parameter controls and return current values."""

    stored = st.session_state.mol_params
    db_cols = [
        "ncbi_method",
        "ncbi_query_cover",
        "ncbi_gap_size",
        "ncbi_gap_min_top",
        "ncbi_low_identity_threshold",
        "ncbi_enforce_low_identity",
    ]
    presets = get_recent_param_sets(
        st.session_state.db_path, "results_mol_inputs", db_cols
    )
    param_preset_selector(
        presets,
        format_fn=lambda p: (
            f"{p['ncbi_method']}, QC: {p['ncbi_query_cover']}%, gap: {p['ncbi_gap_size']}"
            + (
                f", top: {p['ncbi_gap_min_top']}%"
                if p["ncbi_method"] == "barcoding_gap"
                else ""
            )
            + f", low identity: {p['ncbi_low_identity_threshold']}%"
            + (
                " (enforced)"
                if coerce_stored_value(p["ncbi_enforce_low_identity"], False)
                else ""
            )
        ),
        key="ncbi_filter_preset",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        methods = list(FILTER_METHOD_LABELS.keys())
        default_method = stored.get("ncbi_method") or NCBI_METHOD_DEFAULT
        persist_value("ncbi_method", default_method)
        method = st.selectbox(
            "Filtering Method",
            methods,
            format_func=lambda m: FILTER_METHOD_LABELS[m],
            help="Method used to filter NCBI hits per sequence, either by detecting a barcoding gap or using a similarity threshold",
            key="ncbi_method",
        )
        save_widget("ncbi_method")
        active_method = stored.get("ncbi_method", "None")
        active_label = FILTER_METHOD_LABELS.get(active_method, active_method)
        st.caption(f"Active in results: **{active_label}**")
    with col2:
        default_qc = int(stored.get("ncbi_query_cover") or NCBI_QUERY_COVER_DEFAULT)
        persist_value("ncbi_query_cover", default_qc)
        qc = st.slider(
            "Query Coverage (%)",
            1,
            100,
            help="Minimum percentage of the query sequence that must be covered by the hit",
            key="ncbi_query_cover",
        )
        save_widget("ncbi_query_cover")
        active_qc = stored.get("ncbi_query_cover", "None")
        st.caption(f"Active in results: **{active_qc}**")
    with col3:
        default_gap_size = int(stored.get("ncbi_gap_size") or NCBI_GAP_SIZE_DEFAULT)
        persist_value("ncbi_gap_size", default_gap_size)
        gap_size = st.slider(
            "Gap Size (%)",
            0,
            20,
            help="Minimum drop in identity percentage that defines the barcoding gap, or the maximum difference from the best hit when using similarity filtering",
            key="ncbi_gap_size",
        )
        save_widget("ncbi_gap_size")
        active_gap_size = stored.get("ncbi_gap_size", "None")
        st.caption(f"Active in results: **{active_gap_size}**")

    with st.expander("🛠️ Advanced Filter Settings", expanded=False):
        acol1, acol2 = st.columns(2)
        with acol1:
            if method == "barcoding_gap":
                default_gap_min_top = int(
                    stored.get("ncbi_gap_min_top") or NCBI_GAP_MIN_TOP_DEFAULT
                )
                persist_value("ncbi_gap_min_top", default_gap_min_top)
                gap_min_top = st.slider(
                    "Gap Min Top (%)",
                    50,
                    100,
                    help="The top of the barcoding gap must begin at or above this identity percentage",
                    key="ncbi_gap_min_top",
                )
                save_widget("ncbi_gap_min_top")
                active_gap_min_top = stored.get("ncbi_gap_min_top", "None")
                st.caption(f"Active in results: **{active_gap_min_top}**")
            else:
                gap_min_top = NCBI_GAP_MIN_TOP_DEFAULT
        with acol2:
            default_lit = int(
                stored.get("ncbi_low_identity_threshold")
                or NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT
            )
            persist_value("ncbi_low_identity_threshold", default_lit)
            low_identity_threshold = st.slider(
                "Low Identity Warning (%)",
                50,
                100,
                help="Sequences with a top identity below this threshold are highlighted as warnings",
                key="ncbi_low_identity_threshold",
            )
            save_widget("ncbi_low_identity_threshold")
            active_lit = stored.get("ncbi_low_identity_threshold", "None")
            st.caption(f"Active in results: **{active_lit}**")

            default_enforce = (
                stored.get("ncbi_enforce_low_identity")
                if stored.get("ncbi_enforce_low_identity") is not None
                else NCBI_ENFORCE_LOW_IDENTITY_DEFAULT
            )
            persist_value("ncbi_enforce_low_identity", default_enforce)
            enforce_low_identity = st.checkbox(
                "Enforce as hard filter",
                help="Remove hits below the threshold instead of just highlighting them",
                key="ncbi_enforce_low_identity",
            )
            save_widget("ncbi_enforce_low_identity")

    return (
        method,
        qc,
        gap_size,
        gap_min_top,
        low_identity_threshold,
        enforce_low_identity,
    )


def ncbi_filter_job(*, force_rerun, retry_empty=False, params):
    """Execution logic for NCBI filtering."""
    raw_df = st.session_state.ncbi_search_df

    filter_df, params_filter = pipe.run_ncbi_filter(
        raw_df,
        query_cover=params["ncbi_query_cover"],
        gap_size=params["ncbi_gap_size"],
        method=params["ncbi_method"],
        gap_min_top=params["ncbi_gap_min_top"],
        search_params=st.session_state.ncbi_search_params,
        db_path=st.session_state.db_path,
        force_rerun=force_rerun,
    )

    finalized_df, params_final = pipe.finalize_mol_results(
        filter_df,
        db_path=st.session_state.db_path,
        filter_params=params_filter,
        force_rerun=force_rerun,
        threshold=params["ncbi_low_identity_threshold"],
        enforce_threshold=params["ncbi_enforce_low_identity"],
    )

    summary_df = pipe.build_mol_summary(
        finalized_df,
        sequences_df=st.session_state.sequences_df,
        ncbi_filter_df=filter_df,
    )

    st.session_state.mol_df = finalized_df
    st.session_state.mol_params = params_final
    st.session_state.mol_summary_df = summary_df
    st.session_state.mol_species = (
        finalized_df["scientificName"].dropna().unique().tolist()
        if not finalized_df.empty
        else []
    )
    reset_state_after("mol_df")


def _filter_success_message():
    """Show success message when filter results exist."""
    mol_df = st.session_state.mol_df
    if mol_df is not None:
        st.success(
            f"✅ Filtering complete: {len(mol_df):,}/{len(st.session_state.ncbi_search_df):,} hits retained"
        )


def ncbi_filter_workflow():
    """Main NCBI filter controller."""

    with st.container(border=True):
        st.markdown("### 🔍 Filtering Search Results")

        # 1. Parameter inputs
        (
            method,
            query_cover,
            gap_size,
            gap_min_top,
            low_identity_threshold,
            enforce_low_identity,
        ) = ncbi_filter_controls()
        current_params = {
            "ncbi_method": method,
            "ncbi_query_cover": query_cover,
            "ncbi_gap_size": gap_size,
            "ncbi_gap_min_top": gap_min_top,
            "ncbi_low_identity_threshold": low_identity_threshold,
            "ncbi_enforce_low_identity": enforce_low_identity,
        }

        st.divider()

        n_seqs = len(st.session_state.sequences_df)
        run_step_workflow(
            df_key="mol_df",
            params_key="mol_params",
            flag_key="ncbi_filter_flag",
            tab_id="mol",
            compare_keys=[
                "ncbi_method",
                "ncbi_query_cover",
                "ncbi_gap_size",
                "ncbi_gap_min_top",
                "ncbi_low_identity_threshold",
                "ncbi_enforce_low_identity",
            ],
            current_params=current_params,
            job_fn=ncbi_filter_job,
            job_kwargs={"params": current_params},
            new_string=f"Ready to filter **{n_seqs}** sequences.",
            btn_key="ncbi_filter_apply",
            status_label="🔍 Filtering NCBI/BLAST results...",
            before_button=_filter_success_message,
        )


# --------------------------------------------------------------------
# DISPLAY FUNCTIONS
# -------------------------------------------------------------------


def display_ncbi_search_overview():
    """Show per-sequence identity range from raw BLAST results, flagging saturated searches."""
    max_hits = st.session_state.ncbi_search_params.get(
        "ncbi_max_hits", NCBI_MAX_HITS_DEFAULT
    )
    overview_df = pipe.build_ncbi_search_overview(
        st.session_state.ncbi_search_df,
        sequences_df=st.session_state.sequences_df,
        max_hits=max_hits,
    )

    with st.container(border=True):
        st.markdown("### Search Overview")

        saturated = overview_df[overview_df["max_hits_reached"]]
        if not saturated.empty:
            NARROW_RANGE_THRESHOLD = 5
            narrow = saturated[saturated["identity_range"] <= NARROW_RANGE_THRESHOLD]
            if not narrow.empty:
                seq_list = ", ".join(f"**{s}**" for s in narrow["seq_id"])
                st.warning(
                    f"The maximum number of hits ({max_hits}) was reached for {seq_list} "
                    f"within a narrow identity range ({NARROW_RANGE_THRESHOLD}%). Consider increasing **Max Hits** "
                    f"to capture more diverse matches."
                )

        st.dataframe(
            overview_df[
                ["seq_id", "n_hits", "max_identity", "min_identity", "identity_range"]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "seq_id": COL["seq_id"],
                "n_hits": st.column_config.NumberColumn(
                    "Hits", help="Number of BLAST hits returned"
                ),
                "max_identity": st.column_config.NumberColumn(
                    "Max Identity (%)",
                    format="%.2f",
                    help="Highest identity percentage among all hits",
                ),
                "min_identity": st.column_config.NumberColumn(
                    "Min Identity (%)",
                    format="%.2f",
                    help="Lowest identity percentage among all hits",
                ),
                "identity_range": st.column_config.NumberColumn(
                    "Identity Range (%)",
                    format="%.2f",
                    help="Spread between highest and lowest identity (max - min)",
                ),
            },
        )


def display_ncbi_summary():
    summary_df = st.session_state.mol_summary_df.copy()
    summary_df["filter_method"] = summary_df["filter_method"].replace(
        FILTER_METHOD_LABELS
    )
    threshold = st.session_state.mol_params.get(
        "ncbi_low_identity_threshold", NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT
    )

    if summary_df["low_identity_warning"].any():
        st.caption(f":red[■] Low Top Identity (<{threshold}%)")
    styled_df = styled_or_plain(summary_df, highlight_low_identity)
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_config={
            "low_identity_warning": None,
            "seq_id": COL["seq_id"],
            "top_identity": COL["top_identity"],
            "hits_count": COL["hits_count"],
            "species_count": COL["species_count"],
            "filter_method": COL["filter_method"],
        },
    )


def display_ncbi_sequence_summary(sequence_data):
    """Display detailed results summary table"""
    # 1. Sort by:
    #   - scientificName: Alphabetical (A-Z)
    #   - identity_percentage: Highest first (99% > 98%)
    #   - query_cover: Highest first (to break ties in identity)
    df_sorted = sequence_data.sort_values(
        ["scientificName", "identity_percentage", "query_cover"],
        ascending=[True, False, False],
    )

    # 2. Keep the first of each species (the one with highest identity)
    df_display = df_sorted.drop_duplicates(subset="scientificName")

    # 3. Display selected columns
    display_cols = [
        "scientificName",
        "hit_url",
        "hit_count",
        "identity_percentage",
        "query_cover",
        "low_identity_warning",
    ]
    df_display = df_display[display_cols]

    # 4. # Apply highlight_low_identity style to the dataframe
    styled_df = styled_or_plain(df_display, highlight_low_identity)

    # 5. Display with Streamlit
    threshold = st.session_state.mol_params.get(
        "ncbi_low_identity_threshold", NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT
    )
    if df_display["low_identity_warning"].any():
        st.caption(f":red[■] Low Identity (<{threshold}%)")
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_config={
            "low_identity_warning": None,
            "scientificName": COL["scientificName"],
            "identity_percentage": COL["identity_percentage"],
            "hit_count": COL["hit_count"],
            "query_cover": COL["query_cover"],
            "hit_url": COL["hit_url"],
        },
    )


def _style_slider(label, key, default, *args, **kwargs):
    """A plot style slider whose value is kept across sequences and tabs."""
    persist_value(key, default)
    value = st.slider(label, *args, key=key, **kwargs)
    save_widget(key)
    return value


def display_ncbi_sequence_plots(
    selected_sequence, ncbi_full_df, ncbi_filtered_df, mol_params
):
    """Display identity plot for a sequence."""
    sequence_ncbi_data = ncbi_full_df[ncbi_full_df["seq_id"] == selected_sequence]
    cut = ncbi.mol_filter_cut(
        sequence_ncbi_data,
        query_cover=mol_query_cover(),
        gap_size=mol_params["ncbi_gap_size"],
        method=mol_params["ncbi_method"],
        gap_min_top=mol_params.get("ncbi_gap_min_top") or NCBI_GAP_MIN_TOP_DEFAULT,
    )
    filtered_species = set(
        ncbi_filtered_df[ncbi_filtered_df["seq_id"] == selected_sequence][
            "scientificName"
        ].unique()
    )

    # The identity range starts just below the sequence's lowest hit; it is
    # kept per sequence, the style settings below are shared by all sequences
    all_identities = sequence_ncbi_data["identity_percentage"]
    range_min = (
        max(math.floor(all_identities.min()) - 1, 0) if not all_identities.empty else 80
    )

    persist_value("mol_plot_view", "Bars")
    view = st.radio(
        "View",
        ["Bars", "Points"],
        horizontal=True,
        key="mol_plot_view",
        help="Bars: number of hits per identity bin, stacked by species. "
        "Points: one point per hit.",
    )
    save_widget("mol_plot_view")

    # Plot style controls
    with st.expander("Plot settings", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            font_size = _style_slider("Axis text", "mol_plot_font", 14, 8, 24)
        with col2:
            legend_size = _style_slider("Legend text", "mol_plot_legend", 12, 6, 20)
        with col3:
            if view == "Bars":
                bin_width = _style_slider(
                    "Bin width (%)",
                    "mol_plot_bin",
                    0.2,
                    0.1,
                    1.0,
                    step=0.1,
                    help="A bin merges identities closer than its width. Below "
                    "one mismatch (about 0.25% on 400 bp), it only merges hits "
                    "with the same number of mismatches.",
                )
            else:
                marker_size = _style_slider("Marker size", "mol_plot_marker", 10, 4, 20)
        with col4:
            plot_height = _style_slider(
                "Plot height", "mol_plot_height", 500, 300, 900, step=50
            )

        range_key = f"yrange_{selected_sequence}"
        stored_range = st.session_state.get(range_key)
        # The hits can change under a kept range (a new search): start again
        if stored_range and stored_range[0] < range_min:
            del st.session_state[range_key]
        y_range = st.slider(
            "Identity range (%)",
            float(range_min),
            100.0,
            (float(range_min), 100.0),
            step=0.1,
            key=range_key,
        )

    # Only the hits in the identity range are plotted, so the hit axis fits
    # them; the bracket still comes from all the hits (cut above)
    shown_hits = sequence_ncbi_data[
        sequence_ncbi_data["identity_percentage"].between(*y_range)
    ]
    if view == "Bars":
        fig_identity = ncbi.plot_identity_bars(
            shown_hits,
            cut=cut,
            filtered_species=filtered_species,
            title="",
            bin_width=bin_width,
        )
        # Bars are centred on their identity: half a bin more keeps 100% whole
        y_axis_range = [y_range[0] - bin_width / 2, y_range[1] + bin_width / 2]
    else:
        fig_identity = ncbi.plot_identity_percentage(
            shown_hits,
            cut=cut,
            filtered_species=filtered_species,
            title="",
            marker_size=marker_size,
        )
        y_axis_range = list(y_range)
    fig_identity.update_layout(
        margin=dict(t=20),
        height=plot_height,
        font=dict(size=font_size),
        xaxis=dict(
            title=dict(font=dict(size=font_size + 1)), tickfont=dict(size=font_size)
        ),
        yaxis=dict(
            title=dict(font=dict(size=font_size + 1)),
            tickfont=dict(size=font_size),
            range=y_axis_range,
        ),
        legend=dict(font=dict(size=legend_size)),
        modebar=dict(orientation="v"),
    )
    for annotation in fig_identity.layout.annotations:
        annotation.font.size = font_size
    st.plotly_chart(
        fig_identity,
        width="stretch",
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "scale": 4,
                "filename": f"identity_plot_{selected_sequence}",
            },
        },
    )


def display_ncbi_sequence_table(sequence_data):
    """Display detailed results table"""
    display_cols = [
        "scientificName",
        "identity_percentage",
        "query_cover",
        "hit_url",
        "hit_def",
        "align_length",
        "identities",
        "low_identity_warning",
    ]
    cols_in_df = [c for c in display_cols if c in sequence_data.columns]
    df_display = sequence_data[cols_in_df]

    styled_df = styled_or_plain(df_display, highlight_low_identity)

    threshold = st.session_state.mol_params.get(
        "ncbi_low_identity_threshold", NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT
    )
    if (
        "low_identity_warning" in df_display.columns
        and df_display["low_identity_warning"].any()
    ):
        st.caption(f":red[■] Low Identity (<{threshold}%)")
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_config={
            "low_identity_warning": None,
            "scientificName": COL["scientificName"],
            "identity_percentage": COL["identity_percentage"],
            "query_cover": COL["query_cover"],
            "hit_url": COL["hit_url"],
            "hit_def": COL["hit_def"],
            "align_length": COL["align_length"],
            "identities": COL["identities"],
        },
    )


def display_ncbi_filter_warnings(filter_method, gap_analysis, has_warnings):
    """Display filtering method results and warnings"""
    if filter_method == "barcoding_gap" and gap_analysis:
        gap_range = gap_analysis["gap_range"]
        species_details = gap_analysis["species_details"]

        # Always show gap info (blue box)
        if gap_range[1] is not None:
            gap_size = gap_range[0] - gap_range[1]
            st.info(
                f"📊 **Barcoding Gap Detected**: Gap between {gap_range[0]:.2f}% and {gap_range[1]:.2f}% "
                f"(gap size: {gap_size:.2f}%)"
            )
        else:
            st.info(f"📊 **Barcoding Gap Detected**: Gap at {gap_range[0]:.2f}%")

        # Show warning only if there are problematic species (yellow box)
        if has_warnings and species_details:
            warning_msg = "⚠️ **Warning**, the following species appear both BEFORE and AFTER the barcoding gap:\n\n"
            warning_msg += ncbi.format_species_gap_details(species_details)
            st.warning(warning_msg)

    elif filter_method == "similarity":
        st.info(
            "ℹ️ **Similarity Filtering**: No barcoding gap > threshold found, top hits within similarity range retained"
        )


def _mol_sequence_label(seq_id):
    """Format sequence selector label with species count, hits, top identity, and filter method."""
    df = st.session_state.mol_df
    seq_data = df[df["seq_id"] == seq_id]
    if seq_data.empty:
        return f"{seq_id} (no hits)"
    n_species = seq_data["scientificName"].nunique()
    n_hits = len(seq_data)
    top_id = seq_data["identity_percentage"].max()
    method = FILTER_METHOD_LABELS.get(
        seq_data["filter_method"].iloc[0], seq_data["filter_method"].iloc[0]
    )
    return (
        f"{seq_id} — {n_species} species, {n_hits} hits, top {top_id:.1f}% ({method})"
    )


def display_ncbi_per_sequence():
    """Display detailed results per sequence."""
    selected_sequence = sequence_selector(format_fn=_mol_sequence_label)

    if st.session_state.mol_df[
        st.session_state.mol_df["seq_id"] == selected_sequence
    ].empty:
        st.info("No validated records for this sequence.")
        return

    seq_analysis = pipe.build_mol_sequence_report(
        st.session_state.mol_df,
        selected_sequence,
        gap_size=st.session_state.mol_params["ncbi_gap_size"],
    )

    tab_overview, tab_table, tab_plots = st.tabs(
        ["Overview (top hit)", "Table (all hits)", "Plot"]
    )

    with tab_overview:
        # Metrics + gap info + species list
        display_ncbi_filter_warnings(
            seq_analysis["filter_method"],
            seq_analysis["gap_analysis"],
            seq_analysis["has_warnings"],
        )
        display_ncbi_sequence_summary(seq_analysis["sequence_data"])

    with tab_table:
        # Detailed table
        display_ncbi_sequence_table(seq_analysis["sequence_data"])
    with tab_plots:
        display_ncbi_sequence_plots(
            selected_sequence,
            ncbi_full_df=st.session_state.ncbi_search_df,
            ncbi_filtered_df=st.session_state.mol_df,
            mol_params=st.session_state.mol_params,
        )


def display_full_mol_list():
    """Display the complete MOL list in a flat table."""
    mol_df = st.session_state.mol_df
    cols = mol_df.columns.tolist()
    priority = [
        "seq_id",
        "filter_method",
        "scientificName",
        "hit_url",
        "identity_percentage",
        "query_cover",
        "hit_def",
    ]
    ordered = priority + [c for c in cols if c not in priority]
    threshold = st.session_state.mol_params.get(
        "ncbi_low_identity_threshold", NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT
    )
    if mol_df["low_identity_warning"].any():
        st.caption(f":red[■] Low Identity (<{threshold}%)")
    styled_df = styled_or_plain(mol_df[ordered], highlight_low_identity)
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_config={
            "seq_id": COL["seq_id"],
            "filter_method": COL["filter_method"],
            "scientificName": COL["scientificName"],
            "hit_url": COL["hit_url"],
            "identity_percentage": COL["identity_percentage"],
            "query_cover": COL["query_cover"],
            "hit_def": COL["hit_def"],
            "dna_sequence": None,
            "genus": COL["genus"],
            "specificEpithet": COL["specificEpithet"],
            "align_length": COL["align_length"],
            "identities": COL["identities"],
            "gaps": COL["gaps"],
            "query_start": COL["query_start"],
            "query_end": COL["query_end"],
            "identity_drop": None,
            "low_identity_warning": None,
        },
    )


def display_ncbi_results():
    # Summary
    with st.container(border=True):
        st.markdown("### 📊 MOL List Overview")

        final_df = st.session_state.mol_df
        summary_df = st.session_state.mol_summary_df

        if final_df.empty:
            st.info("No MOL results to display.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Unique Species", final_df["scientificName"].nunique())
        col2.metric("Total Sequences", len(summary_df))
        col3.metric("Total Hits", f"{len(final_df):,}")

        display_ncbi_summary()

        with st.expander(
            f"View full MOL list ({len(final_df)} hits)",
            expanded=False,
        ):
            display_full_mol_list()

    # Per-sequence details
    with st.container(border=True):
        st.markdown("### 🔬 MOL List per (M)OTU/ASV")
        display_ncbi_per_sequence()


# --------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------


def mol_analysis():
    """Main NCBI analysis workflow"""

    st.header("🧬 MOL: NCBI BLAST Search")
    st.caption(
        "Step 1: Use BLAST to compare your (M)OTU/ASV sequences against the NCBI database and create the MOL list."
    )
    st.divider()

    if not require_prerequisite(
        "sequences_df",
        "👋 **No sequences found.** Please upload a FASTA file in **Start Analysis** to begin.",
    ):
        return

    # Search Card
    ncbi_search_workflow()

    # Filtering Card
    if st.session_state.ncbi_search_df is not None:
        st.divider()
        ncbi_filter_workflow()

    # Results Display
    if st.session_state.mol_df is not None:
        st.divider()
        display_ncbi_results()
        next_step_button("Next: TAX ➔", "tax")

    st.divider()
