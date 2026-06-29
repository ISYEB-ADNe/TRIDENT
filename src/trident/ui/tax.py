"""
WoRMS Analysis UI Module

This module handles only UI rendering and user interaction.
"""

import streamlit as st

from loguru import logger

from trident.core import config
import trident.pipelines as pipe
from trident.logging import get_log_level

from trident.ui.ui import (
    styled_or_plain,
    COL,
    StreamlitLogSink,
    run_with_progress,
    reset_state_after,
    highlight_in_mol,
    sequence_selector,
    run_step_workflow,
    next_step_button,
    require_prerequisite,
    persist_value,
    save_widget,
)


# --------------------------------------------------------------------
# SEARCH FUNCTIONS
# -------------------------------------------------------------------


def worms_search_job(*, force_rerun, retry_empty=False, status, progress_bar, params):
    """Run WoRMS Genus Expansion."""
    mol_df = st.session_state.mol_df

    # 1. Setup Logging
    logsink = StreamlitLogSink(status, prefix="worms_")
    handler_id = logger.add(
        logsink.write, filter=logsink.thread_filter, level=get_log_level()
    )

    # 2a. Name resolution (R3): canonicalise MOL names to WoRMS accepted names.
    # Cached per name (marine-independent), so re-runs reuse it. When the toggle
    # is off, acceptedName falls back to the raw name and matching is unchanged.
    resolve = params["worms_resolve_names"]
    if resolve:
        status.update(label="🔤 Resolving names to WoRMS accepted...", state="running")
        res_names = pipe.prepare_resolution_input(mol_df)
        res_info = {"current": 0, "total": len(res_names)}

        def _res_poll():
            curr = res_info["current"]
            if res_names:
                progress_bar.progress(
                    min(curr / len(res_names), 1.0),
                    text=f"Resolving {curr}/{len(res_names)} names against WoRMS",
                )

        resolution_df, _ = run_with_progress(
            logsink.wrap(pipe.run_name_resolution),
            res_names,
            db_path=st.session_state.db_path,
            retry_empty=retry_empty,
            user_agent=config.user_agent(),
            progress_handler=res_info,
            logsink=logsink,
            on_poll=_res_poll,
        )
    else:
        resolution_df = None

    # Attach acceptedName/acceptedNameUsageID (raw scientificName kept). Stored
    # back so Results' MOL+GEO match also keys on the accepted name.
    mol_df = pipe.apply_name_resolution(mol_df, resolution_df)
    st.session_state.mol_df = mol_df

    # 2b. Prepare Inputs and Progress Tracker (accepted genera when resolved)
    genera_list = pipe.prepare_worms_input(mol_df)
    total_names = len(genera_list)
    progress_info = {"current": 0, "total": total_names}
    name_str = "genus" if total_names <= 1 else "genera"

    # 3. Background Execution
    def _poll():
        curr = progress_info["current"]
        if total_names > 0:
            percent = min(curr / total_names, 1.0)
            progress_bar.progress(
                percent,
                text=f"Search: {curr}/{total_names} {name_str} expanded",
            )

    worms_search_df, worms_search_params = run_with_progress(
        logsink.wrap(pipe.run_worms_search),
        genera_list,
        db_path=st.session_state.db_path,
        force_rerun=force_rerun,
        retry_empty=retry_empty,
        progress_handler=progress_info,
        user_agent=config.user_agent(),
        marine_only=params["worms_marine_only"],
        logsink=logsink,
        on_poll=_poll,
    )

    # 4. Post-Processing (Merge and Finalize)
    status.update(label="🧬 Merging with MOL list...", state="running")
    worms_merge_df, merge_params = pipe.run_worms_merge(
        worms_search_df,
        mol_df,
        force_rerun=force_rerun,
        db_path=st.session_state.db_path,
        mol_params=st.session_state.mol_params,
        worms_search_params=worms_search_params,
        resolve_names=resolve,
    )

    status.update(label="📊 Finalizing taxonomic expansion...", state="running")
    tax_df, final_params = pipe.finalize_tax_results(
        worms_merge_df,
        force_rerun=force_rerun,
        db_path=st.session_state.db_path,
        worms_merge_params=merge_params,
    )

    summary_df = pipe.build_tax_summary(tax_df, mol_df)

    # 5. Save to State. worms_marine_only propagates from worms_search_params
    # through worms_merge -> finalize into tax_params, so the step-status
    # comparison (compare_keys=["worms_marine_only"]) matches after a run.
    st.session_state.tax_params = final_params
    st.session_state.tax_df = tax_df
    st.session_state.tax_summary_df = summary_df
    st.session_state.tax_species = tax_df["scientificName"].unique().tolist()

    logger.remove(handler_id)
    # Final UI cleanup
    progress_bar.empty()
    reset_state_after("tax_df")


def worms_search_workflow():
    with st.container(border=True):
        st.markdown("### ⚙️ Search Execution")

        with st.expander("🛠️ Advanced Search Settings", expanded=False):
            persist_value("worms_marine_only", True)
            marine_only = st.checkbox(
                "Marine species only",
                key="worms_marine_only",
                help=(
                    "Restrict WoRMS expansion to marine-flagged species "
                    "(default). Uncheck to also include non-marine WoRMS records."
                ),
            )
            save_widget("worms_marine_only")

            persist_value("worms_resolve_names", True)
            resolve_names = st.checkbox(
                "Resolve names to WoRMS accepted",
                key="worms_resolve_names",
                help=(
                    "Map MOL names to their WoRMS-accepted name before matching "
                    "(default), so synonyms (e.g. Gadus ogac -> Gadus macrocephalus) "
                    "are credited and expanded under the correct genus. The original "
                    "NCBI name is kept as verbatimIdentification. Uncheck to match on "
                    "the raw NCBI names only."
                ),
            )
            save_widget("worms_resolve_names")

        current_params = {
            "worms_marine_only": marine_only,
            "worms_resolve_names": resolve_names,
        }

        mol_df = st.session_state.mol_df
        unique_genera = len(pipe.prepare_worms_input(mol_df))

        run_step_workflow(
            df_key="tax_df",
            params_key="tax_params",
            flag_key="worms_search_flag",
            tab_id="tax",
            compare_keys=["worms_marine_only", "worms_resolve_names"],
            current_params=current_params,
            job_fn=worms_search_job,
            job_kwargs={"params": current_params},
            new_string=f"Ready to expand **{unique_genera}** genera into full species lists.",
            btn_key="worms_search_apply",
            logs_prefix="worms_",
            threaded=True,
            status_label="🔍 Expanding genera from MOL list...",
        )


# --------------------------------------------------------------------
# DISPLAY FUNCTIONS
# -------------------------------------------------------------------


def render_worms_metrics():
    """Renders the 3-column metric dashboard."""
    # Compare MOL against TAX on the WoRMS-accepted name when resolution ran, so
    # synonyms (e.g. Gadus ogac -> Gadus macrocephalus) are not falsely flagged
    # as missing. Falls back to the raw MOL names when resolution is off.
    mol_df = st.session_state.mol_df
    if "acceptedName" in mol_df.columns:
        mol_species = set(mol_df["acceptedName"].dropna())
    else:
        mol_species = set(st.session_state.mol_species)
    tax_species = set(st.session_state.tax_species)
    new_count = len(tax_species - mol_species)
    missing_species = sorted(list(mol_species - tax_species))

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Unique Species",
        f"{len(tax_species):,}",
        help="Total unique species found after WoRMS expansion",
    )
    col2.metric(
        "Expanded Species",
        f"+{new_count:,}",
        help="Species added by WoRMS expansion that weren't in the MOL list",
    )
    col3.metric(
        "Missing MOL Hits",
        len(missing_species),
        help="MOL species names not found/accepted in WoRMS, often due to badly formed names or synonyms",
        delta=f"{len(missing_species)} unresolved" if missing_species else "None",
        delta_color="inverse",
    )

    if missing_species:
        with st.expander("⚠️ View Unresolved MOL Names", expanded=False):
            st.warning("The following names from the MOL list were not found in WoRMS:")

            missing_table = [{"Species Name": name} for name in missing_species]
            st.dataframe(
                missing_table,
                width="stretch",
                hide_index=True,
            )


def display_worms_summary():
    st.dataframe(
        st.session_state.tax_summary_df,
        width="stretch",
        hide_index=True,
        column_config={
            "seq_id": COL["seq_id"],
            "MOL Species": st.column_config.NumberColumn(
                "MOL hits",
                help="Total number of unique species identified during the MOL step.",
            ),
            "TAX Species": st.column_config.NumberColumn(
                "TAX Total",
                help="Total species identified in WoRMS for all genera associated with this sequence.",
            ),
            "Overlap": st.column_config.NumberColumn(
                "Verified",
                help="Species that appeared in both the MOL and TAX lists.",
            ),
            "New Species": st.column_config.NumberColumn(
                "Expanded",
                help="New species added by WoRMS genus expansion (not present in MOL).",
            ),
        },
    )


def display_worms_sequence_table(sequence_worms):
    """Display WoRMS species table with formatted columns and color highlighting."""

    styled_df = styled_or_plain(sequence_worms, highlight_in_mol)

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_order=[
            "scientificName",
            "taxonURL",
            "scientificNameAuthorship",
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "specificEpithet",
            "taxonRank",
            "taxonID",
            "taxonID_db",
            "verbatimIdentification",
        ],
        column_config={
            "seq_id": None,
            "dna_sequence": None,
            "scientificName": COL["scientificName"],
            "verbatimIdentification": COL["verbatimIdentification"],
            "taxonURL": COL["taxonURL"],
            "family": COL["family"],
            "genus": COL["genus"],
            "specificEpithet": COL["specificEpithet"],
            "scientificNameAuthorship": COL["scientificNameAuthorship"],
            "taxonRank": COL["taxonRank"],
            "taxonID": COL["taxonID"],
            "taxonID_db": COL["taxonID_db"],
            "kingdom": COL["kingdom"],
            "phylum": COL["phylum"],
            "class": COL["class"],
            "order": COL["order"],
            "in_mol": None,
            "mol_top_identity_percentage": None,
            "mol_top_query_cover": None,
        },
    )


def _tax_sequence_label(seq_id):
    """Format sequence selector label with species count."""
    tax_df = st.session_state.tax_df
    seq_data = tax_df[tax_df["seq_id"] == seq_id]
    if seq_data.empty:
        return f"{seq_id} (no species)"
    n_species = seq_data["scientificName"].nunique()
    in_mol = (
        seq_data.loc[seq_data["in_mol"].astype(bool), "scientificName"].nunique()
        if "in_mol" in seq_data.columns
        else 0
    )
    new = n_species - in_mol
    mol_df = st.session_state.mol_df
    mol_name_col = (
        "acceptedName" if "acceptedName" in mol_df.columns else "scientificName"
    )
    mol_species = set(mol_df[mol_df["seq_id"] == seq_id][mol_name_col].unique())
    tax_species = set(seq_data["scientificName"].unique())
    missing = len(mol_species - tax_species)
    label = f"{seq_id} — {n_species} species ({in_mol} from MOL, +{new} new"
    if missing:
        label += f", {missing} missing"
    return label + ")"


def display_worms_sequence():
    """Display detailed WoRMS results for a selected sequence."""

    selected_seq = sequence_selector(format_fn=_tax_sequence_label)

    df = st.session_state.tax_df
    seq_df = df[(df["seq_id"] == selected_seq)]

    if seq_df.empty:
        st.info("No WoRMS data available for this sequence.")
        return

    st.caption(":green[■] Species also in MOL list")
    display_worms_sequence_table(seq_df)


def display_full_tax_list():
    """Display the complete TAX list in a flat table."""
    df = st.session_state.tax_df
    styled_df = styled_or_plain(df, highlight_in_mol)
    cols = df.columns.tolist()
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
    # NCBI verbatim name goes last.
    middle = [c for c in cols if c not in priority and c != "verbatimIdentification"]
    tail = ["verbatimIdentification"] if "verbatimIdentification" in cols else []
    ordered = priority + middle + tail
    st.caption(":green[■] Species also in MOL list")
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_order=ordered,
        column_config={
            "seq_id": COL["seq_id"],
            "scientificName": COL["scientificName"],
            "verbatimIdentification": COL["verbatimIdentification"],
            "taxonURL": COL["taxonURL"],
            "in_mol": None,
            "mol_top_identity_percentage": None,
            "mol_top_query_cover": None,
            "genus": COL["genus"],
            "specificEpithet": COL["specificEpithet"],
            "family": COL["family"],
            "order": COL["order"],
            "class": COL["class"],
            "phylum": COL["phylum"],
            "kingdom": COL["kingdom"],
            "scientificNameAuthorship": COL["scientificNameAuthorship"],
            "taxonRank": COL["taxonRank"],
            "taxonID": COL["taxonID"],
            "taxonID_db": COL["taxonID_db"],
            "dna_sequence": None,
        },
    )


def display_worms_results():
    """Standardized display for WoRMS expansion results"""

    with st.container(border=True):
        st.markdown("### 📊 TAX List Overview")
        render_worms_metrics()
        display_worms_summary()

        with st.expander(
            f"View full TAX list ({len(st.session_state.tax_df)} species)",
            expanded=False,
        ):
            display_full_tax_list()

    with st.container(border=True):
        st.markdown("### 🔬 TAX List per (M)OTU/ASV")
        display_worms_sequence()


# --------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------


def tax_analysis():
    """Main WoRMS analysis workflow"""

    st.header("🐚 TAX: WoRMS Genus Expansion")
    st.caption(
        "Step 2: For all genera represented in the MOL list, retrieve all valid species using the World Register of Marine Species (WoRMS), forming the TAX list."
    )
    st.divider()

    if not require_prerequisite(
        "mol_df",
        "👈 **MOL list not found.** Please complete **Step 1 · MOL** "
        "before running the WoRMS genus expansion.",
    ):
        return

    # Search Card (No extra filtering step for WoRMS)
    worms_search_workflow()

    # Results Display
    if st.session_state.tax_df is not None:
        st.divider()
        display_worms_results()
        next_step_button("Next: GEO ➔", "geo")

    st.divider()
