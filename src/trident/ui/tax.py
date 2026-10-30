"""TAX tab: the WoRMS name check of the MOL names and the expansion of their
genera."""

import pandas as pd
import streamlit as st


from trident.core.utils import (
    SETTING_LABELS,
    failed_items,
    pluralize,
    species_for_sequences,
)
from trident.core import config
import trident.pipelines as pipe

from trident.ui.ui import (
    ready,
    record_empty_step,
    lazy_expander,
    styled_or_plain,
    column_config,
    StreamlitLogSink,
    run_with_progress,
    reset_state_after,
    highlight_in_mol,
    mol_query_cover,
    display_same_species,
    with_same_species,
    sequence_selector,
    run_step_workflow,
    next_step_button,
    require_prerequisite,
    persist_value,
    save_widget,
)


# The taxonomy columns the TAX tables show, after the sequence.
TAXONOMY_COLUMNS = [
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
]
# The name column of the panels about MOL names.
MOL_NAME = st.column_config.TextColumn("MOL Name (NCBI)")

# --------------------------------------------------------------------
# SEARCH FUNCTIONS
# -------------------------------------------------------------------


def worms_search_job(*, force_rerun, retry_empty=False, status, progress_bar, params):
    """Run WoRMS Genus Expansion."""
    mol_df = st.session_state.mol_df

    # 1. Setup Logging
    logsink = StreamlitLogSink(status, prefix="worms_")

    # 2a. Name resolution: canonicalise MOL names to WoRMS accepted names.
    # Cached per name (marine-independent), so re-runs reuse it.
    status.update(label="🔤 Checking the MOL names in WoRMS...", state="running")
    res_names = pipe.prepare_resolution_input(mol_df)
    ncbi_taxids = pipe.prepare_resolution_taxids(mol_df)
    res_info = {"current": 0, "total": len(res_names)}

    def _res_poll():
        curr, total = res_info["current"], res_info["total"]
        if total > 0:
            progress_bar.progress(
                min(curr / total, 1.0),
                text=f"Name check: {curr}/{total} MOL {pluralize(total, 'name')}",
            )

    resolution_df, _ = run_with_progress(
        logsink.wrap(pipe.run_name_resolution),
        res_names,
        ncbi_taxids=ncbi_taxids,
        db_path=st.session_state.db_path,
        force_rerun=force_rerun,
        retry_empty=retry_empty,
        user_agent=config.user_agent(),
        progress_handler=res_info,
        logsink=logsink,
        progress=res_info,
        on_poll=_res_poll,
    )
    tax_failed = failed_items(resolution_df)

    name_table = pipe.build_name_table(resolution_df, ncbi_taxids)

    # 2b. Prepare Inputs and Progress Tracker (accepted genera)
    genera_list = pipe.prepare_worms_input(mol_df, name_table)
    total_names = len(genera_list)
    progress_info = {"current": 0, "total": total_names}

    # 3. Background Execution
    def _poll():
        curr, total = progress_info["current"], progress_info["total"]
        if total > 0:
            percent = min(curr / total, 1.0)
            progress_bar.progress(
                percent,
                text=f"Search: {curr}/{total} {pluralize(total, 'genus')} expanded",
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
        progress=progress_info,
        on_poll=_poll,
    )
    tax_failed += failed_items(worms_search_df)

    # 4. Post-Processing (Merge and Finalize)
    status.update(label="🧬 Merging with MOL list...", state="running")
    worms_merge_df, merge_params = pipe.run_worms_merge(
        worms_search_df,
        mol_df,
        name_table=name_table,
        force_rerun=force_rerun,
        db_path=st.session_state.db_path,
        mol_params=st.session_state.mol_params,
        worms_search_params=worms_search_params,
    )

    status.update(label="📊 Finalizing taxonomic expansion...", state="running")
    ncbi_search_df = st.session_state.ncbi_search_df
    tax_df, final_params = pipe.finalize_tax_results(
        worms_merge_df,
        force_rerun=force_rerun,
        db_path=st.session_state.db_path,
        worms_merge_params=merge_params,
        ncbi_search_df=ncbi_search_df,
        name_table=name_table,
        drop_mol_rejected=params["tax_drop_mol_rejected"],
    )

    summary_df = pipe.build_tax_summary(tax_df, mol_df, name_table)

    # 5. Save to State. The compared settings (worms_marine_only from
    # worms_search_params, tax_drop_mol_rejected from finalize) end up in
    # tax_params, so the step-status comparison matches after a run.
    st.session_state.tax_params = final_params
    st.session_state.tax_df = tax_df
    st.session_state.tax_failed = tax_failed
    st.session_state.tax_homonyms = pipe.genus_homonyms(worms_search_df, tax_df)
    st.session_state.tax_mol_rejected = pipe.mol_rejected_rows(
        worms_merge_df, ncbi_search_df, name_table, mol_query_cover()
    )
    st.session_state.name_table = name_table
    st.session_state.tax_summary_df = summary_df
    st.session_state.tax_species = tax_df["scientificName"].unique().tolist()

    logsink.close()
    # Final UI cleanup
    progress_bar.empty()
    reset_state_after("tax_df")


def worms_search_workflow():
    with st.container(border=True):
        st.markdown("### ⚙️ Search Execution")

        with st.expander("🛠️ Advanced Search Settings", expanded=False):
            persist_value("worms_marine_only", True)
            marine_only = st.checkbox(
                SETTING_LABELS["worms_marine_only"],
                key="worms_marine_only",
                help=(
                    "Restrict WoRMS expansion to marine-flagged species "
                    "(default). WoRMS counts brackish species as marine too. "
                    "Uncheck to also include non-marine WoRMS records."
                ),
            )
            save_widget("worms_marine_only")

            persist_value("tax_drop_mol_rejected", True)
            drop_mol_rejected = st.checkbox(
                SETTING_LABELS["tax_drop_mol_rejected"],
                key="tax_drop_mol_rejected",
                help=(
                    "For each sequence, leave out the species added by TAX that BLAST "
                    "returned for it but that failed the MOL filter (default). "
                    "This applies per sequence: a species rejected for one "
                    "sequence is kept for the others. Uncheck to keep them."
                ),
            )
            save_widget("tax_drop_mol_rejected")

        current_params = {
            "worms_marine_only": marine_only,
            "tax_drop_mol_rejected": drop_mol_rejected,
        }

        mol_df = st.session_state.mol_df
        unique_genera = len(
            pipe.prepare_worms_input(mol_df, st.session_state.name_table)
        )

        run_step_workflow(
            df_key="tax_df",
            params_key="tax_params",
            flag_key="worms_search_flag",
            tab_id="tax",
            compare_keys=[
                "worms_marine_only",
                "tax_drop_mol_rejected",
            ],
            current_params=current_params,
            job_fn=worms_search_job,
            job_kwargs={"params": current_params},
            new_string=f"Ready to expand **{unique_genera}** genera into full species lists.",
            btn_key="worms_search_apply",
            failed_key="tax_failed",
            logs_prefix="worms_",
            threaded=True,
            status_label="🔍 Expanding genera from MOL list...",
        )


def render_worms_metrics():
    """Renders the 3-column metric dashboard."""
    # Compare MOL against TAX on the WoRMS-accepted name, so synonyms
    # (e.g. Gadus ogac -> Gadus macrocephalus) are not falsely flagged as missing.
    mol_df = st.session_state.mol_df
    name_table = st.session_state.name_table
    mol_names = pd.Series(mol_df["scientificName"].dropna().unique(), dtype=object)
    mol_species = dict(zip(pipe.species_key(mol_names, name_table), mol_names))
    tax_species = set(st.session_state.tax_species)
    tax_keys = set(pipe.species_key(pd.Series(list(tax_species), dtype=object), None))
    new_count = len(tax_keys - set(mol_species))

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Unique Species",
        f"{len(tax_species):,}",
        help="Total unique species found after WoRMS expansion",
    )
    col2.metric(
        "Added by TAX",
        f"+{new_count:,}",
        help="Species TAX added that are not MOL species.",
    )

    summary = pipe.summarize_name_resolution(
        mol_df,
        name_table,
        tax_df=st.session_state.tax_df,
    )
    renamed, not_found = summary.renamed, summary.not_found
    col3.metric(
        "Not found in WoRMS",
        len(not_found),
        help=(
            "MOL names that match no WoRMS record (taxa outside WoRMS, or open "
            "nomenclature such as 'Genus sp.'), or whose records give no single "
            "accepted species. 'Genus in TAX' in the list tells "
            "whether their genus was expanded anyway. Names whose lookup failed "
            "are not counted: they are retried on the next run."
        ),
    )

    if not renamed.empty:
        merged = summary.n_names - summary.n_species
        merged_text = (
            f" ({merged} {pluralize(merged, 'synonym')} merged)" if merged else ""
        )
        st.caption(
            f"{summary.n_names} MOL names = {summary.n_species} WoRMS accepted "
            f"species{merged_text}. TAX works on the WoRMS accepted names."
        )
        n_renamed = len(renamed)
        with st.expander(
            f"View the {n_renamed:,} MOL {pluralize(n_renamed, 'name')} renamed by WoRMS",
            expanded=False,
        ):
            shown = ["scientificName", "accepted_name"]
            st.dataframe(
                renamed,
                width="stretch",
                hide_index=True,
                column_order=shown,
                column_config=column_config(shown, scientificName=MOL_NAME),
            )

    if not not_found.empty:
        n_not_found = len(not_found)
        with st.expander(
            f"View the {n_not_found:,} MOL {pluralize(n_not_found, 'name')} "
            "WoRMS did not resolve",
            expanded=False,
        ):
            shown = ["scientificName", "note", "genus_in_tax"]
            st.dataframe(
                not_found,
                width="stretch",
                hide_index=True,
                column_order=shown,
                column_config=column_config(shown, scientificName=MOL_NAME),
            )


def display_genus_homonyms():
    """List the genus names that are an accepted genus in several groups."""
    homonyms = st.session_state.tax_homonyms
    if homonyms is None or homonyms.empty:
        return
    n_genera = homonyms["genus"].nunique()
    with st.expander(
        f"⚠️ {n_genera} {pluralize(n_genera, 'genus')} with homonyms in WoRMS",
        expanded=False,
    ):
        st.caption(
            "These names are an accepted genus in several groups. For each "
            "sequence, TAX keeps the ones its MOL species belong to, or all of "
            "them when none of its MOL species is in any: check that the "
            "species kept make sense."
        )
        shown = [
            "genus",
            "genusID",
            "kingdom",
            "class",
            "n_species",
            "mol_species",
            "sequences",
        ]
        st.dataframe(
            homonyms,
            width="stretch",
            hide_index=True,
            column_order=shown,
            column_config=column_config(shown),
        )


def display_mol_rejected(seq_id=None):
    """List the expanded species that BLAST returned but MOL rejected.

    Per (sequence, species) pair; restricted to one sequence when seq_id is given.
    """
    rejected = st.session_state.tax_mol_rejected
    if rejected is None or rejected.empty:
        return
    if seq_id is not None:
        rejected = rejected[rejected["seq_id"] == seq_id]
        if rejected.empty:
            return
    n_species = rejected["scientificName"].nunique()
    dropped = st.session_state.tax_params.get("tax_drop_mol_rejected", False)
    if seq_id is None:
        title = f"🚫 {species_for_sequences(rejected)} already rejected by MOL"
    else:
        title = f"🚫 {n_species} species already rejected by MOL for this sequence"
    with st.expander(title, expanded=False):
        st.caption(
            "BLAST returned these species for the sequence, but they failed the "
            "MOL filter. "
            + (
                "They are excluded from the TAX list for that sequence only."
                if dropped
                else "They are kept in the TAX list (setting off)."
            )
        )
        shown = [
            *(["seq_id"] if seq_id is None else []),
            "scientificName",
            "mol_identity",
            "mol_query_cover",
        ]
        st.dataframe(
            rejected.sort_values(["seq_id", "mol_identity"], ascending=[True, False]),
            width="stretch",
            hide_index=True,
            column_order=shown,
            # The best hit the MOL filter rejected, not a MOL hit.
            column_config=column_config(
                shown,
                mol_identity=st.column_config.NumberColumn(
                    "BLAST Identity (%)",
                    format="%.2f",
                    help="Identity of the best BLAST hit of this species for the "
                    "sequence, which the MOL filter rejected.",
                ),
                mol_query_cover=st.column_config.NumberColumn(
                    "BLAST Query Cover (%)",
                    format="%.2f",
                    help="Query cover of that hit.",
                ),
            ),
        )


def display_worms_summary():
    shown = ["seq_id", "n_mol_species", "n_tax_species", "n_both", "n_added"]
    st.dataframe(
        st.session_state.tax_summary_df,
        width="stretch",
        hide_index=True,
        column_order=shown,
        column_config=column_config(shown),
    )


def _with_ncbi_synonyms(df):
    """df with ncbi_synonyms: the NCBI names its MOL hits gave the species,
    when they differ from the species name."""
    synonyms = pipe.ncbi_names(
        df, st.session_state.mol_df, st.session_state.name_table, synonyms_only=True
    )
    return df.assign(ncbi_synonyms=synonyms.str.join(", "))


def display_worms_sequence_table(sequence_worms):
    """Display WoRMS species table with formatted columns and color highlighting."""

    styled_df = styled_or_plain(_with_ncbi_synonyms(sequence_worms), highlight_in_mol)

    shown = [*TAXONOMY_COLUMNS, "ncbi_synonyms"]
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_order=shown,
        column_config=column_config(shown),
    )


def _tax_sequence_labels():
    """Sequence selector label of each sequence: species counts (one pass
    over the TAX list for all)."""
    tax_df = st.session_state.tax_df
    sequences = st.session_state.sequences_id
    if tax_df.empty:  # its summary may have no columns
        return {seq_id: f"{seq_id} (no species)" for seq_id in sequences}
    n_species = tax_df.groupby("seq_id")["scientificName"].nunique()
    if "in_mol" in tax_df.columns:
        from_mol = tax_df[tax_df["in_mol"]]
        in_mol = from_mol.groupby("seq_id")["scientificName"].nunique()
    else:
        in_mol = pd.Series(dtype=int)
    # MOL species not in the TAX list (see build_tax_summary).
    summary = st.session_state.tax_summary_df
    missing = (
        (summary["n_mol_species"] - summary["n_both"]).groupby(summary["seq_id"]).sum()
    )
    labels = {}
    for seq_id in sequences:
        if seq_id not in n_species.index:
            labels[seq_id] = f"{seq_id} (no species)"
            continue
        n_in_mol = in_mol.get(seq_id, 0)
        new = n_species[seq_id] - n_in_mol
        label = (
            f"{seq_id}: {n_species[seq_id]} species ({n_in_mol} from MOL, +{new} added"
        )
        if missing.get(seq_id, 0):
            label += f", {missing[seq_id]} missing"
        labels[seq_id] = label + ")"
    return labels


def display_worms_sequence():
    """Display detailed WoRMS results for a selected sequence."""

    labels = _tax_sequence_labels()
    selected_seq = sequence_selector(labels=labels)

    df = st.session_state.tax_df
    seq_df = df[(df["seq_id"] == selected_seq)]

    if seq_df.empty:
        st.info("No WoRMS data available for this sequence.")
        return

    st.caption(":green[■] Species also in MOL list")
    display_worms_sequence_table(seq_df)
    display_mol_rejected(selected_seq)


def display_full_tax_list():
    """Display the complete TAX list in a flat table."""
    df = with_same_species(_with_ncbi_synonyms(st.session_state.tax_df))
    shown = ["seq_id", *TAXONOMY_COLUMNS, "genusID", "ncbi_synonyms", "same_species_as"]
    st.caption(":green[■] Species also in MOL list")
    st.dataframe(
        styled_or_plain(df, highlight_in_mol),
        width="stretch",
        hide_index=True,
        column_order=shown,
        column_config=column_config(shown),
    )


def display_worms_results():
    """Standardized display for WoRMS expansion results"""

    with st.container(border=True):
        st.markdown("### 📊 TAX List Overview")
        render_worms_metrics()
        display_genus_homonyms()
        display_same_species(st.session_state.tax_df)
        display_mol_rejected()
        display_worms_summary()

        with lazy_expander(
            f"View full TAX list ({species_for_sequences(st.session_state.tax_df)})",
            key="tax_full_list",
        ) as is_open:
            if is_open:
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
        "Step 2: For all genera represented in the MOL list, retrieve all accepted species using the World Register of Marine Species (WoRMS), forming the TAX list."
    )
    st.divider()

    if not require_prerequisite(
        "mol_df",
        "👈 **MOL list not found.** Please complete **Step 1 · MOL** "
        "before running the WoRMS genus expansion.",
    ):
        return

    if st.session_state.mol_df.empty:
        st.info("The MOL list is empty: no genus to expand in WoRMS.")
        record_empty_step(
            "tax_df",
            "tax_params",
            lambda: pipe.empty_tax_results(st.session_state.mol_df),
        )
        return

    # Search Card (No extra filtering step for WoRMS)
    worms_search_workflow()

    # Results Display
    # The step's own list shows while its failures wait; GEO waits for them.
    if st.session_state.tax_df is not None:
        st.divider()
        display_worms_results()
        if ready("tax_df") is not None:
            next_step_button("Next: GEO ➔", "geo")

    st.divider()
