"""MOL tab: the NCBI BLAST search, the MOL filter and the MOL list."""

import math

import streamlit as st


from trident.core.utils import pluralize, species_for_sequences
from trident.clients import ncbi
from trident.clients.ncbi import set_ncbi_email
from trident.core import config
import trident.pipelines as pipe


from trident.ui.ui import (
    SETTING_LABELS,
    failed_items,
    show_failed_items_banner,
    show_filter_success,
    lazy_expander,
    styled_or_plain,
    column_config,
    BLAST_HELP,
    get_recent_param_sets,
    param_preset_selector,
    coerce_stored_value,
    mol_filter_settings,
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

FILTER_METHOD_LABELS = {
    "barcoding_gap": "Barcoding gap",
    "similarity": "Similarity range",
}

# The columns of a BLAST hit the MOL tables show, after the sequence.
HIT_COLUMNS = [
    "scientificName",
    "hit_url",
    "identity_percentage",
    "query_cover",
    "hit_def",
    "genus",
    "specificEpithet",
    "ncbi_taxid",
    "align_length",
    "identities",
    "gaps",
    "query_start",
    "query_end",
]

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
            SETTING_LABELS["ncbi_max_hits"],
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
            f"{SETTING_LABELS['ncbi_ev_exponent']} ($10^{{-x}}$)",
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
                "Parallel threads",
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
    """Run the NCBI BLAST search (a step job, see run_step_workflow)."""

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

        with lazy_expander(
            f"Inspect raw NCBI/BLAST results ({len(st.session_state.ncbi_search_df):,} hits)",
            key="mol_search_results",
        ) as is_open:
            if is_open:
                shown = ["seq_id", *HIT_COLUMNS]
                st.dataframe(
                    st.session_state.ncbi_search_df,
                    width="stretch",
                    hide_index=True,
                    column_order=shown,
                    column_config=column_config(shown),
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
            f"{FILTER_METHOD_LABELS.get(p['ncbi_method'], p['ncbi_method'])}, "
            f"query cover: {p['ncbi_query_cover']}%, gap: {p['ncbi_gap_size']}"
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
            SETTING_LABELS["ncbi_method"],
            methods,
            format_func=lambda m: FILTER_METHOD_LABELS[m],
            help="Barcoding gap: keep the hits above a barcoding gap, or within "
            "the similarity range of the best hit when there is none. Similarity "
            "range: always the latter.",
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
            SETTING_LABELS["ncbi_query_cover"],
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
            SETTING_LABELS["ncbi_gap_size"],
            0,
            20,
            help="Minimum drop in identity percentage that defines the barcoding gap, or the maximum difference from the best hit with the similarity range",
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
                    SETTING_LABELS["ncbi_gap_min_top"],
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
                SETTING_LABELS["ncbi_low_identity_threshold"],
                50,
                100,
                help="Hits below this identity are highlighted (red), and in the summary, the sequences with a species whose best hit is below it.",
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
                SETTING_LABELS["ncbi_enforce_low_identity"],
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
        show_filter_success(len(mol_df), len(st.session_state.ncbi_search_df), "hits")


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
                    f"within a narrow identity range ({NARROW_RANGE_THRESHOLD}%). Raise "
                    "**Max hits per query** to get lower-identity hits."
                )

        shown = [
            "seq_id",
            "n_blast_hits",
            "max_identity",
            "min_identity",
            "identity_range",
        ]
        st.dataframe(
            overview_df,
            width="stretch",
            hide_index=True,
            column_order=shown,
            column_config=column_config(shown),
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
        st.caption(f":red[■] Sequences with a species below {threshold}%")
    styled_df = styled_or_plain(summary_df, highlight_low_identity)
    shown = ["seq_id", "filter_method", "n_hits", "n_species", "top_identity"]
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_order=shown,
        column_config=column_config(
            shown,
            n_hits=st.column_config.NumberColumn(
                "Hits Kept", format="%d", help="Hits the MOL filter kept."
            ),
        ),
    )


def display_ncbi_sequence_summary(sequence_data):
    """Display detailed results summary table"""
    df_sorted = sequence_data.sort_values(
        ["scientificName", "identity_percentage", "query_cover"],
        ascending=[True, False, False],
    )

    # The best hit of each species
    df_display = df_sorted.drop_duplicates(subset="scientificName")

    display_cols = [
        "scientificName",
        "hit_url",
        "n_hits",
        "identity_percentage",
        "query_cover",
        "low_identity_warning",
    ]
    df_display = df_display[display_cols]

    styled_df = styled_or_plain(df_display, highlight_low_identity)

    threshold = st.session_state.mol_params.get(
        "ncbi_low_identity_threshold", NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT
    )
    if df_display["low_identity_warning"].any():
        st.caption(f":red[■] Low Identity (<{threshold}%)")
    shown = [
        "scientificName",
        "hit_url",
        "n_hits",
        "identity_percentage",
        "query_cover",
    ]
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_order=shown,
        column_config=column_config(shown),
    )


def _style_slider(label, key, default, *args, **kwargs):
    """A plot style slider whose value is kept across sequences and tabs."""
    persist_value(key, default)
    value = st.slider(label, *args, key=key, **kwargs)
    save_widget(key)
    return value


def display_ncbi_sequence_plots(
    selected_sequence, sequence_ncbi_data, ncbi_filtered_df, cut
):
    """Display identity plot of a sequence's BLAST hits, with the MOL filter's *cut*."""
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
    threshold = st.session_state.mol_params.get(
        "ncbi_low_identity_threshold", NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT
    )
    if sequence_data["low_identity_warning"].any():
        st.caption(f":red[■] Low Identity (<{threshold}%)")
    shown = [
        "scientificName",
        "identity_percentage",
        "query_cover",
        "hit_url",
        "hit_def",
        "align_length",
        "identities",
    ]
    st.dataframe(
        styled_or_plain(sequence_data, highlight_low_identity),
        width="stretch",
        hide_index=True,
        column_order=shown,
        column_config=column_config(shown),
    )


def display_ncbi_filter_warnings(cut, species_details, settings):
    """Display the gap or range the MOL filter used (*settings*: its
    settings, see mol_filter_settings), and species across the gap."""
    if cut is None:
        return
    if cut.is_gap:
        st.info(
            f"📊 **Barcoding Gap Detected**: Gap between {cut.top:.2f}% and "
            f"{cut.bottom:.2f}% (gap size: {cut.top - cut.bottom:.2f}%)"
        )
        if species_details:
            st.warning(
                "⚠️ These species have hits on both sides of the barcoding gap, "
                "so the gap does not separate them:\n\n"
                + ncbi.format_species_gap_details(species_details)
            )
    else:
        # Only the gap method looked for a gap before taking the range.
        gap_method = settings["method"] == "barcoding_gap"
        no_gap = " (no barcoding gap found)" if gap_method else ""
        st.info(
            f"ℹ️ **Similarity range**: hits within {settings['gap_size']:g}% of "
            f"the best hit kept{no_gap}."
        )


def _mol_sequence_labels():
    """Sequence selector label of each sequence: species count, hits, top
    identity and filter method (one pass over the MOL list for all)."""
    df = st.session_state.mol_df
    by_seq = df.groupby("seq_id", sort=False)
    n_species = by_seq["scientificName"].nunique()
    n_hits = by_seq.size()
    top_id = by_seq["identity_percentage"].max()
    # The method of the sequence's first row
    method = df.drop_duplicates("seq_id").set_index("seq_id")["filter_method"]
    labels = {}
    for seq_id in st.session_state.sequences_id:
        if seq_id not in n_hits.index:
            labels[seq_id] = f"{seq_id} (no hits)"
            continue
        label = FILTER_METHOD_LABELS.get(method[seq_id], method[seq_id])
        labels[seq_id] = (
            f"{seq_id}: {n_species[seq_id]} species, {n_hits[seq_id]} hits, "
            f"top {top_id[seq_id]:.1f}% ({label})"
        )
    return labels


def display_ncbi_per_sequence():
    """Display detailed results per sequence."""
    labels = _mol_sequence_labels()
    selected_sequence = sequence_selector(labels=labels)

    if st.session_state.mol_df[
        st.session_state.mol_df["seq_id"] == selected_sequence
    ].empty:
        st.info("The MOL filter kept no hit for this sequence.")
        return

    settings = mol_filter_settings()
    seq_analysis = pipe.build_mol_sequence_report(
        st.session_state.mol_df,
        selected_sequence,
        st.session_state.ncbi_search_df,
        **settings,
    )

    tab_overview, tab_table, tab_plots = st.tabs(
        ["Overview (top hit)", "Table (all hits)", "Plot"]
    )

    with tab_overview:
        # Metrics + gap info + species list
        display_ncbi_filter_warnings(
            seq_analysis["cut"],
            seq_analysis["species_across_gap"],
            settings,
        )
        display_ncbi_sequence_summary(seq_analysis["sequence_data"])

    with tab_table:
        # Detailed table
        display_ncbi_sequence_table(seq_analysis["sequence_data"])
    with tab_plots:
        display_ncbi_sequence_plots(
            selected_sequence,
            sequence_ncbi_data=seq_analysis["search_hits"],
            ncbi_filtered_df=st.session_state.mol_df,
            cut=seq_analysis["cut"],
        )


def display_full_mol_list():
    """Display the complete MOL list in a flat table."""
    mol_df = st.session_state.mol_df.assign(
        filter_method=lambda df: df["filter_method"].replace(FILTER_METHOD_LABELS)
    )
    threshold = st.session_state.mol_params.get(
        "ncbi_low_identity_threshold", NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT
    )
    if mol_df["low_identity_warning"].any():
        st.caption(f":red[■] Low Identity (<{threshold}%)")
    shown = ["seq_id", "filter_method", *HIT_COLUMNS]
    st.dataframe(
        styled_or_plain(mol_df, highlight_low_identity),
        width="stretch",
        hide_index=True,
        column_order=shown,
        column_config=column_config(shown),
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

        with lazy_expander(
            f"View full MOL list ({len(final_df):,} hits: {species_for_sequences(final_df)})",
            key="mol_full_list",
        ) as is_open:
            if is_open:
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
        "Step 1: Use BLAST to compare your (M)OTU/ASV sequences against GenBank and create the MOL list."
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
