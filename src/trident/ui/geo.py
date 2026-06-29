"""
GEO Analysis UI Module

This module handles only UI rendering and user interaction.
All logic is delegated to the trident.pipelines.geo_pipeline module.
"""

import pandas as pd
import streamlit as st

from loguru import logger

from trident.core import config
import trident.pipelines as pipe
from trident.logging import get_log_level
from trident.clients.gbif import (
    get_extent_column_name,
    parse_coordinate,
    plot_bounding_boxes,
)

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
    highlight_in_mol,
    sequence_selector,
    run_step_workflow,
    next_step_button,
    require_prerequisite,
)

from trident.ui.defaults import (
    GBIF_LATITUDE_DEFAULT,
    GBIF_LONGITUDE_DEFAULT,
    GBIF_EXTENT_DEFAULT,
    GBIF_MIN_OCCURRENCES_DEFAULT,
)


# --------------------------------------------------------------------
# SEARCH FUNCTIONS
# -------------------------------------------------------------------


def display_gbif_parameters():
    st.markdown("#### 📍 Sampling Site and Extent")
    st.info(
        "📊 **Area of Interest:** Search areas in **GBIF** are defined by rectangular "
        "**bounding boxes** (latitude and longitude ranges). We calculate these boundaries "
        "from your sampling site using a **Search Extent**. This value represents the minimum "
        "distance (km) from the center to the box edges, ensuring the search covers your area "
        "of interest in every direction."
    )

    # Previous parameter sets selector
    db_cols = ["gbif_latitude", "gbif_longitude", "gbif_extent"]
    presets = get_recent_param_sets(
        st.session_state.db_path, "gbif_search_inputs", db_cols
    )
    param_preset_selector(
        presets,
        format_fn=lambda p: (
            f"{p['gbif_latitude']}°, {p['gbif_longitude']}° — {p['gbif_extent']} km"
        ),
        key="gbif_search_preset",
    )

    stored = st.session_state.gbif_search_params
    def_lat = str(stored.get("gbif_latitude") or GBIF_LATITUDE_DEFAULT)
    def_lon = str(stored.get("gbif_longitude") or GBIF_LONGITUDE_DEFAULT)
    def_extent = str(stored.get("gbif_extent") or GBIF_EXTENT_DEFAULT)

    col1, col2, col3 = st.columns(3)
    with col1:
        persist_value("gbif_latitude", def_lat)
        latitude_input = st.text_input(
            "Latitude",
            help="DD or DMS format, e.g., -7.543 or 7°32'37\"S",
            key="gbif_latitude",
        )
        save_widget("gbif_latitude")
        try:
            latitude = parse_coordinate(latitude_input, "lat")
            st.caption(f"✅ Parsed as: {latitude:.4f}°")
        except ValueError as e:
            st.error(str(e))
            return None

    with col2:
        persist_value("gbif_longitude", def_lon)
        longitude_input = st.text_input(
            "Longitude",
            help="DD or DMS format, e.g., -35.423 or 35°25'24\"W",
            key="gbif_longitude",
        )
        save_widget("gbif_longitude")
        try:
            longitude = parse_coordinate(longitude_input, "lon")
            st.caption(f"✅ Parsed as: {longitude:.4f}°")
        except ValueError as e:
            st.error(str(e))
            return None

    with col3:
        persist_value("gbif_extent", def_extent)
        extent_input = st.text_input(
            "Search extents (km)",
            help="Enter one or more search extents in km, separated by commas (e.g. 100, 250).",
            key="gbif_extent",
        )
        save_widget("gbif_extent")

        try:
            extents = [
                int(float(x.strip())) for x in extent_input.split(",") if x.strip()
            ]

            if not extents:
                st.error(
                    "Please enter at least one valid integer for the search extent."
                )
                return None

            extents_str = ", ".join(map(str, extents))
            st.caption(f"✅ Parsed extents: **{extents_str} km**")

            extents += ["global"]  # Always include global search option

        except ValueError:
            st.error(
                "Please enter valid integers separated by commas (e.g., 100, 250)."
            )
            return None

    return latitude, longitude, extents


def display_gbif_map(latitude: float, longitude: float, extents: list[int]) -> None:
    """Render a Plotly map showing the calculated bounding box."""
    show_map = st.checkbox(
        "📍 Visualize Search Bounding Box",
        value=False,
        key="gbif_show_map",
        help="Display the center point and the calculated coordinate boundaries on a map.",
    )

    if not show_map:
        return

    try:
        numeric_extents = [int(x) for x in extents if x != "global"]

        with st.expander("Map settings", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                plot_height = st.slider(
                    "Plot height", 300, 800, 500, step=50, key="geo_map_height"
                )
            with col2:
                map_padding = st.slider(
                    "Padding",
                    0.1,
                    2.0,
                    0.5,
                    step=0.1,
                    key="geo_map_padding",
                    help="Space around the bounding boxes (as fraction of extent)",
                )
            with col3:
                legend_size = st.slider("Legend text", 6, 20, 12, key="geo_map_legend")
            with col4:
                show_circles = st.checkbox(
                    "Show circles",
                    value=False,
                    key="geo_map_circles",
                    help="Draw the circular perimeter for each extent",
                )

        fig = plot_bounding_boxes(
            latitude,
            longitude,
            numeric_extents,
            show_circles=show_circles,
            padding=map_padding,
        )
        fig.update_layout(
            height=plot_height,
            modebar=dict(orientation="v"),
            legend=dict(font=dict(size=legend_size)),
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 4,
                    "filename": "geo_bounding_boxes",
                },
                "scrollZoom": True,
            },
        )

        st.caption(
            "The bounding boxes represent the coordinate range that will be sent to GBIF. "
        )

    except Exception as e:
        st.warning(f"Could not generate map: {str(e)}")


def _gbif_search_and_merge(
    species_list,
    tax_df,
    *,
    force_rerun,
    retry_empty,
    params,
    db_path,
    tax_params,
    user_agent,
    progress_info,
):
    """Run GBIF search + merge in a background thread."""

    def _update_progress(fraction, text):
        progress_info["fraction"] = fraction
        progress_info["text"] = text

    gbif_search_df, gbif_search_params = pipe.run_gbif_search(
        species_list,
        latitude=params["gbif_latitude"],
        longitude=params["gbif_longitude"],
        extents=params["gbif_extents"],
        db_path=db_path,
        force_rerun=force_rerun,
        retry_empty=retry_empty,
        user_agent=user_agent,
        progress_handler=_update_progress,
    )

    gbif_merge_df, gbif_merge_params = pipe.run_gbif_merge(
        gbif_search_df,
        tax_df,
        force_rerun=force_rerun,
        db_path=db_path,
        tax_params=tax_params,
        gbif_search_params=gbif_search_params,
    )

    if "gbif_extent" in gbif_merge_df.columns:
        gbif_merge_df["gbif_extent"] = gbif_merge_df["gbif_extent"].astype(str)

    return gbif_merge_df, gbif_merge_params


def gbif_search_job(*, force_rerun, retry_empty=False, status, progress_bar, params):
    """Run GBIF search and merge with TAX list."""
    tax_df = st.session_state.tax_df

    # 1. Setup Logging
    logsink = StreamlitLogSink(status, prefix="gbif_")
    handler_id = logger.add(
        logsink.write, filter=logsink.thread_filter, level=get_log_level()
    )

    # 2. Prepare Inputs
    species_list = pipe.prepare_gbif_input(tax_df)
    progress_info = {"fraction": 0.0, "text": ""}

    # 3. Background Execution
    def _poll():
        progress_bar.progress(progress_info["fraction"], text=progress_info["text"])

    gbif_merge_df, gbif_merge_params = run_with_progress(
        logsink.wrap(_gbif_search_and_merge),
        species_list,
        tax_df,
        force_rerun=force_rerun,
        retry_empty=retry_empty,
        params=params,
        db_path=st.session_state.db_path,
        tax_params=st.session_state.tax_params,
        user_agent=config.user_agent(),
        progress_info=progress_info,
        logsink=logsink,
        on_poll=_poll,
    )

    # 4. Save to State
    st.session_state.gbif_search_df = gbif_merge_df
    st.session_state.gbif_search_params = gbif_merge_params

    logsink.flush_to_ui()
    logger.remove(handler_id)
    progress_bar.empty()
    reset_state_after("gbif_search_df")


def gbif_search_workflow():
    """Handle GBIF search execution."""

    with st.container(border=True):
        st.markdown("### ⚙️ Search Configuration and Execution")

        # Parameter Inputs
        params = display_gbif_parameters()
        if params is None:
            return

        latitude, longitude, extents = params
        display_gbif_map(latitude, longitude, extents)

        current_params = {
            "gbif_latitude": latitude,
            "gbif_longitude": longitude,
            "gbif_extents": extents,
        }

        unique_sp = len(st.session_state.tax_species)

        mode = run_step_workflow(
            df_key="gbif_search_df",
            params_key="gbif_search_params",
            flag_key="gbif_search_flag",
            tab_id="geo",
            compare_keys=["gbif_latitude", "gbif_longitude", "gbif_extents"],
            current_params=current_params,
            job_fn=gbif_search_job,
            job_kwargs={"params": current_params},
            new_string=f"Ready to analyze GBIF distribution for **{unique_sp}** species.",
            btn_key="gbif_search_apply",
            logs_prefix="gbif_",
            threaded=True,
            status_label="🗺️ Running GBIF search...",
        )

    # Results Preview
    if mode != "NEW" and st.session_state.gbif_search_df is not None:
        with st.expander(
            f"Inspect GBIF search results ({len(st.session_state.gbif_search_df)} records)",
            expanded=False,
        ):
            df = st.session_state.gbif_search_df
            cols = df.columns.tolist()
            priority = [
                "seq_id",
                "scientificName",
                "gbif_taxonURL",
                "gbif_extent",
                "occurrences",
                "genus",
                "specificEpithet",
                "family",
                "order",
                "class",
                "phylum",
                "kingdom",
            ]
            ordered = priority + [c for c in cols if c not in priority]
            if df["in_mol"].astype(bool).any():
                st.caption(":green[■] Species also in MOL list")
            styled_df = styled_or_plain(df, highlight_in_mol)
            st.dataframe(
                styled_df,
                width="stretch",
                hide_index=True,
                column_order=ordered,
                column_config={
                    "seq_id": COL["seq_id"],
                    "scientificName": COL["scientificName"],
                    "gbif_taxonURL": COL["gbif_taxonURL"],
                    "gbif_extent": COL["gbif_extent"],
                    "occurrences": COL["occurrences"],
                    "genus": COL["genus"],
                    "specificEpithet": COL["specificEpithet"],
                    "family": COL["family"],
                    "order": COL["order"],
                    "class": COL["class"],
                    "phylum": COL["phylum"],
                    "kingdom": COL["kingdom"],
                    "dna_sequence": None,
                    "in_mol": None,
                    "mol_top_identity_percentage": None,
                    "mol_top_query_cover": None,
                    "taxonURL": None,
                    "scientificNameAuthorship": None,
                    "taxonRank": None,
                    "taxonID": None,
                    "taxonID_db": None,
                },
            )


# --------------------------------------------------------------------
# FILTERING FUNCTIONS
# -------------------------------------------------------------------


def gbif_filter_controls() -> tuple[int, int]:
    """Render GBIF filtering parameter controls and return current values."""

    db_cols = ["gbif_min_occurrences", "gbif_filter_extent"]
    presets = get_recent_param_sets(
        st.session_state.db_path, "gbif_filter_inputs", db_cols
    )
    param_preset_selector(
        presets,
        format_fn=lambda p: (
            f"Min. occurrences: {p['gbif_min_occurrences']}, extent: {p['gbif_filter_extent']}"
        ),
        key="gbif_filter_preset",
    )
    stored = st.session_state.gbif_search_params

    def_min_occ = int(
        stored.get("gbif_min_occurrences") or GBIF_MIN_OCCURRENCES_DEFAULT
    )
    default_extent = stored.get("gbif_filter_extent") or GBIF_EXTENT_DEFAULT
    available_extents = st.session_state.gbif_search_params["gbif_extents"]

    col1, col2 = st.columns(2)
    with col1:
        persist_value("gbif_min_occurrences", def_min_occ)
        min_occurrences = st.number_input(
            "Minimum occurrences:",
            min_value=0,
            key="gbif_min_occurrences",
            help="Set the minimum number of occurrences a species must have within the area of interest to be retained.",
        )
        save_widget("gbif_min_occurrences")
        active_min_occ = stored.get("gbif_min_occurrences", "None")
        st.caption(f"Active in results: **{active_min_occ}**")
    with col2:
        options = available_extents
        active_extent = stored.get("gbif_filter_extent", "None")
        # Ensure default is valid for current options
        if default_extent not in options:
            default_extent = options[0]
        persist_value("gbif_filter_extent", default_extent)
        extent = st.selectbox(
            "Select filtering area:",
            options,
            key="gbif_filter_extent",
            help="Choose which search area to use for the occurrence threshold.",
        )
        save_widget("gbif_filter_extent")

        if active_extent == "None" or active_extent is None:
            st.caption("Active in results: **None**")
        else:
            st.caption(
                f"Active in results: **{get_extent_column_name(active_extent)}**"
            )

    return min_occurrences, extent


def gbif_filter_job(*, force_rerun, retry_empty=False, min_occurrences, extent):
    """Execution logic for GBIF filtering."""

    gbif_filter_df, gbif_filter_params = pipe.run_gbif_filter(
        st.session_state.gbif_search_df,
        extent=extent,
        min_occurrences=min_occurrences,
        db_path=st.session_state.db_path,
        gbif_merge_params=st.session_state.gbif_search_params,
        force_rerun=force_rerun,
    )

    ncbi_search_df = st.session_state.ncbi_search_df
    ncbi_search_pairs = (
        set(zip(ncbi_search_df["seq_id"], ncbi_search_df["scientificName"]))
        if ncbi_search_df is not None
        else None
    )

    geo_df, geo_params = pipe.finalize_geo_results(
        gbif_filter_df,
        db_path=st.session_state.db_path,
        gbif_filter_params=gbif_filter_params,
        force_rerun=force_rerun,
        ncbi_search_pairs=ncbi_search_pairs,
    )

    geo_summary_df = pipe.build_geo_summary(
        st.session_state.gbif_search_df, min_occurrences
    )

    st.session_state.gbif_filter_df = gbif_filter_df
    st.session_state.geo_df = geo_df
    st.session_state.geo_params = geo_params
    st.session_state.geo_summary_df = geo_summary_df
    st.session_state.geo_species = geo_df["scientificName"].unique().tolist()
    st.session_state.geo_and_mol_species = (
        geo_df.loc[geo_df["in_mol"].astype(bool), "scientificName"].unique().tolist()
    )

    reset_state_after("geo_df")


def gbif_filter_workflow():
    """Main GBIF filter controller."""

    with st.container(border=True):
        st.markdown("### 🔍 Filtering Search Results")

        # Parameter inputs
        min_occurrences, extent = gbif_filter_controls()
        current_params = {
            "gbif_min_occurrences": min_occurrences,
            "gbif_filter_extent": extent,
        }
        st.divider()

        run_step_workflow(
            df_key="geo_df",
            params_key="geo_params",
            flag_key="gbif_filter_flag",
            tab_id="geo",
            compare_keys=["gbif_min_occurrences", "gbif_filter_extent"],
            current_params=current_params,
            job_fn=gbif_filter_job,
            job_kwargs={"min_occurrences": min_occurrences, "extent": extent},
            new_string=f"Ready to filter GBIF results for **{len(st.session_state.gbif_search_df)}** records.",
            btn_key="gbif_filter_apply",
            status_label="🔍 Filtering GBIF results...",
        )

    # Empty results warning
    geo_df = st.session_state.geo_df
    if geo_df is not None and len(geo_df) == 0:
        st.warning("⚠️ No species found with the applied filter.")


# --------------------------------------------------------------------
# DISPLAY FUNCTIONS
# -------------------------------------------------------------------


def render_gbif_metrics():
    """Render GBIF metrics in a 3-column layout."""
    if st.session_state.geo_df.empty:
        st.info("No GEO results to display.")
        return

    ext = st.session_state.geo_params.get("gbif_filter_extent", "global")
    ext_label = get_extent_column_name(ext)
    lat, lon = (
        st.session_state.geo_params.get("gbif_latitude"),
        st.session_state.geo_params.get("gbif_longitude"),
    )

    total_species = st.session_state.geo_df["scientificName"].nunique()
    seq_with_hits = (st.session_state.geo_summary_df[ext_label] > 0).sum()
    total_seqs = len(st.session_state.geo_summary_df)

    if ext == "global":
        loc_str = "🌍 Global"
    else:
        loc_str = (
            f"📍 {lat:.2f}°, {lon:.2f}° — {ext_label}"
            if lat is not None
            else f"📍 {ext_label}"
        )

    st.markdown(f"**{loc_str}**")
    confirmed = len(st.session_state.geo_and_mol_species)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Species", total_species, help="Unique species in the GEO list.")
    c2.metric(
        "Confirmed Species",
        confirmed,
        help="Species with both a direct NCBI hit (MOL) and geographic validation (GEO).",
    )
    c3.metric(
        "Sequences with Hits",
        f"{seq_with_hits} / {total_seqs}",
        help="Sequences that matched at least one species in this area.",
    )


def display_gbif_summary():
    extent_cols = [c for c in st.session_state.geo_summary_df.columns if c != "seq_id"]
    st.dataframe(
        st.session_state.geo_summary_df,
        width="stretch",
        hide_index=True,
        column_order=["seq_id"] + extent_cols,
        column_config={
            "seq_id": COL["seq_id"],
            "global": "🌍 Global Hits",
            **{c: f"📍 {c}" for c in extent_cols if c != "global"},
        },
    )


def display_missing_species():
    if st.session_state.gbif_search_df.empty:
        return
    gbif_species = set(st.session_state.gbif_search_df["scientificName"].unique())
    missing = sorted(list(set(st.session_state.tax_species) - gbif_species))
    if missing:
        with st.expander(f"⚠️ {len(missing)} TAX species not found in GBIF"):
            st.warning("Species from the TAX list with no GBIF match:")
            st.dataframe([{"Species Name": n} for n in missing], hide_index=True)


def display_rejected_species():
    """Show rows removed during finalization (found by NCBI but rejected by MOL)."""
    gbif_filter_df = st.session_state.gbif_filter_df
    if gbif_filter_df is None:
        return
    rejected_df = pipe.get_ncbi_rejected_rows(
        gbif_filter_df, st.session_state.geo_df, st.session_state.ncbi_search_df
    )
    if rejected_df.empty:
        return
    n_rows = len(rejected_df)
    n_species = rejected_df["scientificName"].nunique()
    row_word = "row" if n_rows == 1 else "rows"
    species_word = "species" if n_species == 1 else "species"
    with st.expander(
        f"🚫 {n_rows} {row_word} ({n_species} {species_word}) removed — already rejected by MOL"
    ):
        st.info(
            "These (sequence, species) pairs appeared in NCBI BLAST results but did not "
            "survive MOL filtering. They are excluded from the GEO list."
        )
        cols = rejected_df.columns.tolist()
        priority = [
            "seq_id",
            "scientificName",
            "mol_top_identity_percentage",
            "gbif_occurrences",
            "gbif_taxonURL",
            "taxonURL",
            "genus",
            "specificEpithet",
            "family",
            "order",
            "class",
            "phylum",
            "kingdom",
        ]
        ordered = priority + [c for c in cols if c not in priority]
        st.dataframe(
            rejected_df,
            hide_index=True,
            width="stretch",
            column_order=ordered,
            column_config={
                "seq_id": COL["seq_id"],
                "scientificName": COL["scientificName"],
                "mol_top_identity_percentage": COL["mol_top_identity_percentage"],
                "mol_top_query_cover": None,
                "gbif_occurrences": COL["gbif_occurrences"],
                "gbif_taxonURL": COL["gbif_taxonURL"],
                "taxonURL": COL["taxonURL"],
                "genus": COL["genus"],
                "specificEpithet": COL["specificEpithet"],
                "family": COL["family"],
                "order": COL["order"],
                "class": COL["class"],
                "phylum": COL["phylum"],
                "kingdom": COL["kingdom"],
                "in_mol": None,
                "dna_sequence": None,
                "scientificNameAuthorship": None,
                "taxonRank": None,
                "taxonID": None,
                "taxonID_db": None,
            },
        )


def display_gbif_sequence_table(
    seq_df: pd.DataFrame,
    min_occurrences: int,
    extents: list[float],
    priority_extent: float | str = None,
):
    df_wide, classification = pipe.classify_gbif_extents(
        seq_df, min_occurrences, extents, priority_extent
    )
    priority_col = classification["priority_col"]
    extent_cols = classification["extent_cols"]
    has_identity = classification["has_identity"]

    col_order = (
        ["scientificName", "gbif_taxonURL"]
        + [c for c in extent_cols if c in df_wide.columns]
        + (["mol_top_identity_percentage"] if has_identity else [])
    )

    col_config = {
        "scientificName": COL["scientificName"],
        "in_mol": None,
        "mol_top_identity_percentage": COL["mol_top_identity_percentage"]
        if has_identity
        else None,
        "gbif_taxonURL": COL["gbif_taxonURL"],
        **{c: st.column_config.NumberColumn(c, format="%d") for c in extent_cols},
    }

    def render_styled_table(data_subset):
        df_to_show = data_subset.sort_values("scientificName", ascending=True)
        styled = styled_or_plain(df_to_show, highlight_in_mol)
        st.dataframe(
            styled,
            column_order=col_order,
            column_config=col_config,
            width="stretch",
            hide_index=True,
        )

    st.caption(":green[■] Species also in MOL list")

    for bucket in classification["buckets"]:
        kind, ext_col, df = bucket["kind"], bucket["extent_col"], bucket["df"]

        if kind == "local":
            if not df.empty:
                render_styled_table(df)
            else:
                st.info(
                    f"No species found matching the threshold in the "
                    f"{priority_col} extent."
                )
        elif kind == "smaller":
            with st.expander(
                f"📍 {len(df)} / {bucket['total']} species also validated "
                f"in **{ext_col}**"
            ):
                render_styled_table(df)
        elif kind == "larger":
            with st.expander(
                f"🌍 {len(df)} species validated in **{ext_col}** "
                f"(not in {priority_col})"
            ):
                render_styled_table(df)
        elif kind == "never":
            with st.expander(f"❌ {len(df)} species below threshold at all extents"):
                render_styled_table(df)


def _geo_sequence_label(seq_id):
    """Format sequence selector label with species counts."""
    params = st.session_state.geo_params
    target_extent = str(params["gbif_filter_extent"])
    min_occ = params["gbif_min_occurrences"]

    df = st.session_state.gbif_search_df
    seq_df = df[df["seq_id"] == seq_id]

    if seq_df.empty:
        return f"{seq_id} (no data)"

    total = seq_df["scientificName"].nunique()

    # Count species in GEO list meeting threshold in priority extent
    geo_species = set(
        st.session_state.geo_df[st.session_state.geo_df["seq_id"] == seq_id][
            "scientificName"
        ]
    )
    geo_seq_df = seq_df[seq_df["scientificName"].isin(geo_species)]
    local_mask = (geo_seq_df["gbif_extent"].astype(str) == target_extent) & (
        geo_seq_df["occurrences"] >= min_occ
    )
    local_count = geo_seq_df.loc[local_mask, "scientificName"].nunique()

    label = get_extent_column_name(params["gbif_filter_extent"])
    return f"{seq_id} — {local_count}/{total} species in {label}"


def display_gbif_sequence():
    """Display detailed GBIF results for a selected sequence."""

    selected_seq = sequence_selector(format_fn=_geo_sequence_label)
    df = st.session_state.gbif_search_df

    seq_df = df[(df["seq_id"] == selected_seq)].copy()

    if seq_df.empty:
        st.info(f"No GBIF data available for sequence: **{selected_seq}**")
        return

    params = st.session_state.geo_params
    target_extent = params["gbif_filter_extent"]
    min_occ = params["gbif_min_occurrences"]

    display_gbif_sequence_table(
        seq_df,
        min_occ,
        st.session_state.gbif_search_params["gbif_extents"],
        target_extent,
    )

    # Show species rejected by MOL for this sequence
    gbif_filter_df = st.session_state.gbif_filter_df
    if gbif_filter_df is not None:
        rejected_df = pipe.get_ncbi_rejected_rows(
            gbif_filter_df, st.session_state.geo_df, st.session_state.ncbi_search_df
        )
        rejected_seq = rejected_df[rejected_df["seq_id"] == selected_seq]
        if not rejected_seq.empty:
            n_species = rejected_seq["scientificName"].nunique()
            with st.expander(f"🚫 {n_species} species rejected by MOL filtering"):
                st.info(
                    "These species appeared in NCBI BLAST results but did not "
                    "survive MOL filtering. They are excluded from the GEO list."
                )
                st.dataframe(
                    rejected_seq,
                    hide_index=True,
                    width="stretch",
                    column_order=[
                        "scientificName",
                        "mol_top_identity_percentage",
                        "gbif_occurrences",
                        "gbif_taxonURL",
                    ],
                    column_config={
                        "scientificName": COL["scientificName"],
                        "mol_top_identity_percentage": COL[
                            "mol_top_identity_percentage"
                        ],
                        "gbif_occurrences": COL["gbif_occurrences"],
                        "gbif_taxonURL": COL["gbif_taxonURL"],
                    },
                )


def display_full_geo_list():
    """Display the complete GEO list in a flat table."""
    geo_df = st.session_state.geo_df
    cols = geo_df.columns.tolist()
    priority = [
        "seq_id",
        "scientificName",
        "gbif_occurrences",
        "mol_top_identity_percentage",
        "gbif_taxonURL",
        "taxonURL",
        "genus",
        "specificEpithet",
        "family",
        "order",
        "class",
        "phylum",
        "kingdom",
    ]
    ordered = priority + [c for c in cols if c not in priority]
    st.caption(":green[■] Species also in MOL list")
    styled_df = styled_or_plain(geo_df, highlight_in_mol)
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_order=ordered,
        column_config={
            "seq_id": COL["seq_id"],
            "scientificName": COL["scientificName"],
            "gbif_taxonURL": COL["gbif_taxonURL"],
            "taxonURL": COL["taxonURL"],
            "gbif_occurrences": COL["gbif_occurrences"],
            "genus": COL["genus"],
            "specificEpithet": COL["specificEpithet"],
            "family": COL["family"],
            "order": COL["order"],
            "class": COL["class"],
            "phylum": COL["phylum"],
            "kingdom": COL["kingdom"],
            "in_mol": None,
            "mol_top_identity_percentage": COL["mol_top_identity_percentage"],
            "mol_top_query_cover": None,
            "dna_sequence": None,
            "scientificNameAuthorship": None,
            "taxonRank": None,
            "taxonID": None,
            "taxonID_db": None,
        },
    )


def display_gbif_results():
    """Main GBIF results controller."""

    with st.container(border=True):
        st.markdown("### 📊 GEO List Overview")
        render_gbif_metrics()
        display_gbif_summary()
        display_missing_species()
        display_rejected_species()

        with st.expander(
            f"View full GEO list ({len(st.session_state.geo_df)} species)",
            expanded=False,
        ):
            display_full_geo_list()

    with st.container(border=True):
        st.markdown("### 🔬 GEO List per (M)OTU/ASV")
        display_gbif_sequence()


# --------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------


def geo_analysis():
    """Main GBIF analysis workflow"""

    st.header("🌍 GEO: GBIF Species Validation")
    st.caption(
        "Define a geographic search window by latitude, longitude, and radius. Species present within this area, meeting the optional minimum occurrence threshold (default: n = 3), are retained to create the GEO list."
    )
    st.divider()

    if not require_prerequisite(
        "tax_df",
        "👈 **TAX list not found.** Please complete **Step 2 · TAX** "
        "to provide a species list for GBIF validation.",
    ):
        return

    # GBIF Data Acquisition
    gbif_search_workflow()

    # Filtering
    if st.session_state.gbif_search_df is not None:
        st.divider()
        gbif_filter_workflow()

    # Results
    if st.session_state.geo_df is not None:
        st.divider()
        display_gbif_results()
        next_step_button("Next: EXTRA ➔", "extra")

    st.divider()
