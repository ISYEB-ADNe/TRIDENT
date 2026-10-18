"""
Results UI Module (new layout)

Same results as the Results tab, laid out per sequence: a summary first, one
row per (M)OTU/ASV with its assignment, the selected sequence's candidate
species (curated there), and every download at the end.
"""

from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st

from trident.pipelines.results_pipeline import (
    build_results_df,
    find_sequence_exclusion_step,
    filter_excluded_results,
    resolve_sequences,
    RESULT_COLS,
    EXPORT_COLS,
    build_gbif_export_df,
)
from trident.ui.defaults import NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT
from trident.ui.results import (
    RESULTS_COLUMN_CONFIG,
    _show_all_hits_expander,
    _species_breakdown_columns,
    show_provenance_section,
)
from trident.ui.ui import (
    COL,
    SELECTED_SEQUENCE_KEY,
    persist_value,
    save_widget,
    styled_or_plain,
    highlight_low_identity,
    highlight_below_mol,
    sequence_selector,
)


THRESHOLD_KEY = "flag_identity"
FILTER_KEY = "results_filter"
ASSIGNMENTS_KEY = "results_assignments"

FILTERS = ["All", "Species level", "Higher rank", "Unassigned", "Low identity"]

# Why a sequence ends without a species, by the step that lost it.
LOSS_REASONS = {
    "MOL Search": "no NCBI hit (MOL)",
    "MOL Filter": "no hit passing the MOL filter",
    "TAX": "no species found in WoRMS (TAX)",
    "GEO": "no candidate recorded in the area (GEO)",
    "HYPO": "only expanded candidates, none confirmed by HYPO",
    "HYPO not run": "only expanded candidates, HYPO not run",
    "Curation": "all candidates excluded by hand",
}


# --------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------


def _best_identity(df):
    """Best NCBI identity of each sequence's MOL+GEO species.

    HYPO species are left out: they passed the HYPO thresholds, and their NCBI
    identity is the marker check, not the assignment.
    """
    mol_geo = df[df["validation_step"] == "MOL+GEO"]
    return mol_geo.groupby("seq_id")["ncbi_top_identity_percentage"].max()


def _flag_low_identity(df, threshold, best_identity):
    """Flag the MOL+GEO rows of sequences whose best identity is below threshold."""
    df = df.copy()
    best = df["seq_id"].map(best_identity)
    df["low_identity_warning"] = (
        (df["validation_step"] == "MOL+GEO") & best.notna() & (best < threshold)
    )
    return df


def _lost_step(seq_ids, results_df):
    """Step at which each sequence without a species lost it."""
    exclusion = find_sequence_exclusion_step(
        all_seq_ids=seq_ids,
        ncbi_search_df=st.session_state.get("ncbi_search_df"),
        mol_df=st.session_state.get("mol_df"),
        tax_df=st.session_state.get("tax_df"),
        geo_df=st.session_state.get("geo_df"),
    )
    steps = dict(zip(exclusion.get("seq_id", []), exclusion.get("pipeline_step", [])))
    had_species = set(results_df.dropna(subset=["scientificName"])["seq_id"])
    # A sequence with validated species lost them all by hand. One that passed
    # GEO without any had only expanded species, which HYPO did not confirm or
    # has not tested yet.
    after_geo = (
        "HYPO" if st.session_state.get("hypo_df") is not None else "HYPO not run"
    )
    return {
        seq_id: "Curation" if seq_id in had_species else steps.get(seq_id, after_geo)
        for seq_id in seq_ids
    }


def _assignments(curated_df, results_df, threshold):
    """One row per sequence: its assignment after curation, and what to check."""
    table = resolve_sequences(curated_df)
    species_rows = curated_df.dropna(subset=["scientificName"])
    by_seq = species_rows.groupby("seq_id")

    table["best_identity"] = table["seq_id"].map(_best_identity(species_rows))
    table["low_identity"] = table["best_identity"] < threshold
    table["paths"] = table["seq_id"].map(
        by_seq["validation_step"].agg(lambda s: " + ".join(sorted(s.unique())))
    )
    below_mol = table["seq_id"].map(by_seq["below_mol"].any())
    table["below_mol"] = below_mol.fillna(False).astype(bool)

    unassigned = table.loc[table["n_species"] == 0, "seq_id"].tolist()
    lost = _lost_step(unassigned, results_df)
    notes = []
    for _, row in table.iterrows():
        if row["n_species"] == 0:
            notes.append(LOSS_REASONS.get(lost[row["seq_id"]], lost[row["seq_id"]]))
            continue
        note = []
        if row["low_identity"]:
            note.append(f"best identity below {threshold}%")
        if row["below_mol"]:
            note.append("HYPO species below the MOL filter")
        if row["n_species"] > 1 and pd.isna(row["taxonRank"]):
            note.append("no shared taxon")
        notes.append(", ".join(note))
    table["note"] = notes
    return table


def _filtered_seq_ids(assignments, choice):
    """Sequence IDs shown for a filter choice (all of them for "All" or no choice)."""
    n_species = assignments["n_species"]
    species_level = assignments["species_level"]
    mask = {
        "Species level": species_level,
        "Higher rank": (n_species > 0) & ~species_level,
        "Unassigned": n_species == 0,
        "Low identity": assignments["low_identity"],
    }.get(choice)
    return assignments["seq_id"] if mask is None else assignments.loc[mask, "seq_id"]


def _count_phrase(counts):
    """Counts as "2 family, 1 order"."""
    return ", ".join(f"{n} {label}" for label, n in counts.items())


def _default_threshold():
    """The MOL filter's low identity warning, so both steps flag alike."""
    stored = st.session_state.get("mol_params") or {}
    value = stored.get("ncbi_low_identity_threshold")
    return int(float(value)) if value else NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT


# --------------------------------------------------------------------
# CALLBACKS (run before the page, so the summary at the top is current)
# -------------------------------------------------------------------


def _apply_keep_edits(editor_key, row_keys):
    """Turn the Keep checkboxes into exclusions."""
    edits = st.session_state[editor_key]["edited_rows"]
    excluded = set(st.session_state.excluded_results)
    for i, change in edits.items():
        if "keep" in change:
            key = row_keys[int(i)]
            if change["keep"]:
                excluded.discard(key)
            else:
                excluded.add(key)
    st.session_state.excluded_results = sorted(excluded)
    # A new editor key drops the edits the table holds, so it shows the
    # exclusions as they are now rather than replaying old clicks.
    st.session_state._keep_version = st.session_state.get("_keep_version", 0) + 1


def _select_sequence(table_key, seq_ids):
    """Show the sequence clicked in the assignment table below it."""
    rows = st.session_state[table_key]["selection"]["rows"]
    if rows:
        seq_id = seq_ids[rows[0]]
        st.session_state[SELECTED_SEQUENCE_KEY] = seq_id
        st.session_state._widget_overrides[SELECTED_SEQUENCE_KEY] = seq_id


# --------------------------------------------------------------------
# SECTIONS
# -------------------------------------------------------------------


def show_summary(assignments, curated_df, results_df, threshold):
    """Counts of sequences by assignment level and of candidate species."""
    st.markdown("### 📋 Summary")

    n_species = assignments["n_species"]
    species_level = assignments["species_level"]
    higher = assignments[(n_species > 0) & ~species_level]
    unassigned = assignments[n_species == 0]

    st.markdown("**Sequences**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", len(assignments))
    c2.metric(
        "Species level",
        int(species_level.sum()),
        help="Sequences whose candidates resolve to one species.",
    )
    c3.metric(
        "Higher rank",
        len(higher),
        help=(
            "Sequences with several candidate species, assigned to the lowest "
            "rank they share (as in the GBIF export)."
        ),
    )
    c4.metric(
        "Unassigned",
        len(unassigned),
        help="Sequences without a validated species.",
    )
    c5.metric(
        f"Identity < {threshold}%",
        int(assignments["low_identity"].sum()),
        help="Sequences whose best NCBI identity (MOL+GEO) is below the threshold.",
    )
    if not higher.empty:
        ranks = higher["taxonRank"].str.lower().fillna("no shared taxon")
        st.caption(f"Higher rank: {_count_phrase(ranks.value_counts())}.")
    if not unassigned.empty:
        reasons = unassigned["note"].value_counts()
        st.caption(
            "Unassigned: "
            + ", ".join(f"{n} with {reason}" for reason, n in reasons.items())
            + "."
        )

    species_rows = curated_df.dropna(subset=["scientificName"])
    steps = species_rows.groupby("validation_step")["scientificName"].nunique()
    resolved_species = assignments.loc[species_level, "species"].dropna().unique()

    st.markdown("**Species**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Candidate species",
        species_rows["scientificName"].nunique(),
        help=(
            "Every species kept for at least one sequence, including the "
            "candidates of sequences assigned at a higher rank."
        ),
    )
    c2.metric(
        "MOL+GEO",
        steps.get("MOL+GEO", 0),
        help="Candidates confirmed by a direct NCBI hit and by geography.",
    )
    c3.metric(
        "HYPO",
        steps.get("HYPO", 0),
        help=(
            "Candidates confirmed by a CO1 proxy (BOLD/NCBI), with the thresholds "
            "of the HYPO step. A species can be both MOL+GEO and HYPO."
        ),
    )
    c4.metric(
        "At species level",
        len(resolved_species),
        help="Species to which at least one sequence resolves.",
    )

    excluded = results_df.dropna(subset=["scientificName"])
    excluded = excluded[
        (excluded["seq_id"] + "||" + excluded["scientificName"]).isin(
            st.session_state.excluded_results
        )
    ].drop_duplicates(subset=["seq_id", "scientificName"])
    if not excluded.empty:
        with st.expander(f"Excluded by hand ({len(excluded)})"):
            st.caption("Select the sequence and check Keep to restore a species.")
            st.dataframe(
                excluded,
                width="stretch",
                hide_index=True,
                column_order=[
                    "seq_id",
                    "scientificName",
                    "validation_step",
                    "ncbi_top_identity_percentage",
                ],
                column_config=RESULTS_COLUMN_CONFIG,
            )


def show_assignments(assignments, threshold):
    """One row per sequence, filtered; clicking a row selects that sequence."""
    with st.container(border=True):
        st.markdown("### 🧬 Assignment per (M)OTU/ASV")
        counts = {
            choice: len(_filtered_seq_ids(assignments, choice)) for choice in FILTERS
        }
        persist_value(FILTER_KEY, "All")
        choice = st.segmented_control(
            "Show",
            FILTERS,
            key=FILTER_KEY,
            format_func=lambda c: f"{c} ({counts[c]})",
            help=f"Low identity: best NCBI identity (MOL+GEO) below {threshold}%.",
        )
        save_widget(FILTER_KEY)
        # Clicking the active button deselects it: show everything then.
        choice = choice or "All"
        shown = assignments[
            assignments["seq_id"].isin(_filtered_seq_ids(assignments, choice))
        ].reset_index(drop=True)
        st.caption("Click a row to see and curate its candidate species below.")
        # A selection belongs to one table: a new key per filter and per shown
        # sequence keeps it from pointing at another row once either changes.
        table_key = (
            f"{ASSIGNMENTS_KEY}_{choice}_{st.session_state.get(SELECTED_SEQUENCE_KEY)}"
        )
        st.dataframe(
            shown,
            width="stretch",
            hide_index=True,
            on_select=partial(_select_sequence, table_key, shown["seq_id"].tolist()),
            selection_mode="single-row",
            key=table_key,
            column_order=[
                "seq_id",
                "scientificName",
                "taxonRank",
                "n_species",
                "best_identity",
                "paths",
                "note",
            ],
            column_config={
                "seq_id": COL["seq_id"],
                "scientificName": st.column_config.TextColumn(
                    "Assignment",
                    help="The species, or the lowest taxon its candidates share.",
                ),
                "taxonRank": st.column_config.TextColumn("Rank"),
                "n_species": st.column_config.NumberColumn(
                    "Candidates", help="Candidate species kept for this sequence."
                ),
                "best_identity": st.column_config.NumberColumn(
                    "Best identity (%)",
                    format="%.2f",
                    help="Best NCBI identity among its MOL+GEO candidates.",
                ),
                "paths": RESULTS_COLUMN_CONFIG["validation_step"],
                "note": st.column_config.TextColumn("To check"),
            },
        )


def show_sequence_detail(results_df, assignments, threshold):
    """The selected sequence's assignment and candidates, with Keep checkboxes."""
    with st.container(border=True):
        st.markdown("### 🔬 Candidate species of a (M)OTU/ASV")
        by_seq = assignments.set_index("seq_id")

        def _seq_label(seq_id):
            row = by_seq.loc[seq_id]
            if row["n_species"] == 0:
                return f"{seq_id} ∅"
            return f"{seq_id} ⚠" if row["note"] else seq_id

        selected_seq = sequence_selector(format_fn=_seq_label)
        row = by_seq.loc[selected_seq]
        if row["species_level"] and row["n_species"] == 1:
            st.markdown(f"Assigned to **{row['scientificName']}** (species).")
        elif pd.notna(row["taxonRank"]):
            st.markdown(
                f"Assigned to **{row['scientificName']}** "
                f"({row['taxonRank'].lower()}), {row['n_species']} candidate species."
            )
        elif row["n_species"] > 1:
            st.markdown(f"{row['n_species']} candidate species with no shared taxon.")
        else:
            st.info(f"Unassigned: {row['note']}.")
        if row["note"] and row["n_species"]:
            st.caption(f"⚠ {row['note'][:1].upper()}{row['note'][1:]}.")

        species_rows = results_df.dropna(subset=["scientificName"])
        table = species_rows[species_rows["seq_id"] == selected_seq].reset_index(
            drop=True
        )
        if table.empty:
            return
        table = _flag_low_identity(
            table, threshold, assignments.set_index("seq_id")["best_identity"]
        )
        row_keys = (table["seq_id"] + "||" + table["scientificName"]).tolist()
        excluded = set(st.session_state.excluded_results)
        table.insert(0, "keep", [key not in excluded for key in row_keys])

        legends = ["Uncheck Keep to exclude a candidate from this sequence."]
        if table["low_identity_warning"].any():
            legends.append(f":red[■] Best identity below {threshold}%")
        if table["below_mol"].any():
            legends.append(
                ":orange[■] Found in NCBI but would not have passed the MOL filter"
            )
        st.caption(" · ".join(legends))

        editor_key = f"keep_{selected_seq}_{st.session_state.get('_keep_version', 0)}"
        st.data_editor(
            styled_or_plain(table, highlight_below_mol, highlight_low_identity),
            width="stretch",
            hide_index=True,
            key=editor_key,
            on_change=_apply_keep_edits,
            args=(editor_key, row_keys),
            disabled=[c for c in table.columns if c != "keep"],
            column_order=["keep"] + [c for c in RESULT_COLS if c != "seq_id"],
            column_config={
                **RESULTS_COLUMN_CONFIG,
                "keep": st.column_config.CheckboxColumn("Keep"),
            },
        )


def show_results_per_species(results_df, assignments, threshold):
    """Taxonomy, links and sequences of one species."""
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

        tax_parts = []
        for rank in ["kingdom", "phylum", "class", "order", "family", "genus"]:
            val = row.get(rank)
            if pd.notna(val):
                tax_parts.append(f"**{rank.capitalize()}:** {val}")
        if pd.notna(row.get("scientificNameAuthorship")):
            tax_parts.append(f"**Authorship:** {row['scientificNameAuthorship']}")
        st.markdown(" · ".join(tax_parts))

        links = []
        for col, label in (
            ("taxonURL", "WoRMS"),
            ("gbif_taxonURL", "GBIF"),
            ("ncbi_top_hit_url", "NCBI"),
            ("bold_seq_url", "BOLD"),
        ):
            if pd.notna(row.get(col)):
                links.append(f"[🔗 {label}]({row[col]})")
        if links:
            st.markdown(" · ".join(links))

        st.divider()

        best_identity = assignments.set_index("seq_id")["best_identity"]
        for step, is_hypo in (("MOL+GEO", False), ("HYPO", True)):
            step_df = sp_df[sp_df["validation_step"] == step]
            if step_df.empty:
                continue
            st.markdown(f"#### {step} Sequences")
            col_order, col_config = _species_breakdown_columns(step_df, is_hypo)
            col_config["below_mol"] = None
            st.dataframe(
                styled_or_plain(
                    _flag_low_identity(step_df, threshold, best_identity),
                    highlight_below_mol,
                    highlight_low_identity,
                ),
                width="stretch",
                hide_index=True,
                column_order=col_order,
                column_config=col_config,
            )
            _show_all_hits_expander(selected_species, is_hypo=is_hypo)


def show_exports(curated_df):
    """Every download in one place, with the provenance of the analysis."""
    with st.container(border=True):
        st.markdown("### 💾 Export")
        analysis_name = st.session_state.get("analysis_name", "results")

        if curated_df is not None:
            st.caption(
                "Results after curation: one row per sequence and candidate "
                "species (CSV), or one row per sequence at its assigned rank (GBIF)."
            )
            export_cols = [c for c in EXPORT_COLS if c in curated_df.columns]
            dl1, dl2 = st.columns(2)
            # Built when clicked, from the results shown now.
            dl1.download_button(
                label="📥 Full results (CSV)",
                data=lambda: curated_df[export_cols].to_csv(index=False),
                file_name=f"{analysis_name}_results.csv",
                mime="text/csv",
                width="stretch",
                key="new_results_csv",
            )
            with dl2:
                # Built in the script run, where the session's contact email
                # and the spinner apply; offered for the results it was built from.
                fingerprint = int(
                    pd.util.hash_pandas_object(
                        curated_df[export_cols], index=False
                    ).sum()
                )
                if st.button(
                    "📥 GBIF export (CSV)", width="stretch", key="new_gbif_csv"
                ):
                    with st.spinner("Looking up higher-level taxa in WoRMS..."):
                        gbif_df = build_gbif_export_df(curated_df)
                    st.session_state._gbif_export_new = (
                        fingerprint,
                        gbif_df.to_csv(index=False),
                    )
                built = st.session_state.get("_gbif_export_new")
                if built and built[0] == fingerprint:
                    st.download_button(
                        label="⬇️ Download GBIF export",
                        data=built[1],
                        file_name=f"{analysis_name}_gbif_export.csv",
                        mime="text/csv",
                        width="stretch",
                        key="new_gbif_download",
                    )

        db_path = st.session_state.get("db_path")
        if not db_path or not Path(db_path).is_file():
            return
        st.caption(
            "The analysis is one SQLite file: every step's results, the cached "
            "searches and the settings used. Load it at the Start tab to resume "
            "the analysis or to share it; any SQLite tool can also read it."
        )
        st.download_button(
            label=f"📥 Download the analysis ({Path(db_path).name}, SQLite)",
            data=Path(db_path).read_bytes,
            file_name=f"{analysis_name}.db",
            mime="application/x-sqlite3",
            width="stretch",
            key="new_db_download",
        )
    show_provenance_section()


# --------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


def results_main():
    """Render the Results (new) tab."""
    st.title("📦 Analysis Results")
    st.divider()

    if st.session_state.get("geo_df") is None:
        st.info(
            "📊 No results yet. Complete at least steps 1–3 "
            "(MOL → TAX → GEO) to see results."
        )
        show_exports(None)
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
        show_exports(None)
        return

    curated_df = filter_excluded_results(
        results_df, set(st.session_state.excluded_results)
    )

    with st.container(border=True):
        # Follow the MOL warning when it changes, else keep the user's value.
        default = _default_threshold()
        if st.session_state.get("_flag_identity_seed") != default:
            st.session_state._flag_identity_seed = default
            st.session_state[THRESHOLD_KEY] = default
            st.session_state._widget_overrides[THRESHOLD_KEY] = default
        persist_value(THRESHOLD_KEY, default)
        threshold = st.slider(
            "Flag sequences whose best identity is below (%)",
            50,
            100,
            key=THRESHOLD_KEY,
            help=(
                "Starts at the MOL filter's low identity warning. A sequence whose "
                "best NCBI identity (MOL+GEO) is below it is flagged; nothing is "
                "removed. HYPO candidates keep the thresholds of the HYPO step."
            ),
        )
        save_widget(THRESHOLD_KEY)
        assignments = _assignments(curated_df, results_df, threshold)
        show_summary(assignments, curated_df, results_df, threshold)

    show_assignments(assignments, threshold)
    show_sequence_detail(results_df, assignments, threshold)
    show_results_per_species(results_df, assignments, threshold)
    show_exports(curated_df)
