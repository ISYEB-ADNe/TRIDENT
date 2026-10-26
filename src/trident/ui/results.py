"""
Results UI Module

The combined MOL+GEO and HYPO results, laid out per sequence: a summary first,
one row per (M)OTU/ASV with its assignment, the selected sequence's candidate
species (curated there) and their path through the steps, and every download
at the end.
"""

from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st

from trident.clients.gbif import area_label
from trident.core.config import is_streamlit_cloud
from trident.core.database import fingerprint, load_provenance
from trident.core.utils import (
    count_phrase,
    name_list,
    normalize_name,
    pluralize,
    species_for_sequences,
)
from trident.pipelines.names import resolved_names, species_key, species_of

from trident.pipelines.results_pipeline import (
    build_species_flows,
    plot_species_flow,
    species_flow_table,
    build_results_df,
    exclusion_keys,
    flow_lost_at,
    filter_excluded_results,
    resolve_sequences,
    EXPORT_COLS,
    build_gbif_export_df,
    best_identity_per_sequence,
    flag_low_identity,
)
from trident.ui.defaults import NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT
from trident.ui.geo import display_gbif_sequence_table
from trident.ui.ui import (
    column_config,
    SELECTED_SEQUENCE_KEY,
    failed_items,
    persist_value,
    save_widget,
    styled_or_plain,
    highlight_low_identity,
    highlight_check_hit,
    highlight_empty_sequence,
    CHECK_HIT_LEGEND,
    CHECK_HIT_NOTE,
    mol_query_cover,
    sequence_selector,
    display_same_species,
    same_species_as,
)


THRESHOLD_KEY = "flag_identity"

# The columns of a candidate the Results tables show, after the sequence:
# for a HYPO species, also the MOL species its CO1 proxy matched and the
# sequence's MOL identity to it.
CANDIDATE_COLS = [
    "validation_step",
    "scientificName",
    "taxonURL",
    "geo_url",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "specificEpithet",
    "scientificNameAuthorship",
    "mol_identity",
    "mol_query_cover",
    "mol_hit_url",
    "hypo_check_identity",
    "hypo_check_query_cover",
    "hypo_check_hit_url",
    "proxy_hit_species",
    "proxy_identity",
    "proxy_query_cover",
    "via_mol_identity",
    "proxy_url",
    "geo_occurrences",
]
# The columns of a species' NCBI hits.
NCBI_HIT_COLS = [
    "seq_id",
    "scientificName",
    "hit_def",
    "identity_percentage",
    "query_cover",
    "hit_url",
]
# The columns of the assignment table, one row per sequence.
ASSIGNMENT_COLS = [
    "seq_id",
    "scientificName",
    "taxonRank",
    "n_species",
    "best_identity",
    "paths",
    "note",
]
FILTER_KEY = "results_filter"
ASSIGNMENTS_KEY = "results_assignments"

FILTERS = [
    "All",
    "Species level",
    "Higher rank",
    "Unassigned",
    "HYPO check",
    "Low identity",
]

# Why a sequence ends without a species, by the step that lost it.
LOSS_REASONS = {
    "MOL Search": "no NCBI hit (MOL)",
    "MOL Filter": "no hit passing the MOL filter",
    "TAX": "no species found in WoRMS (TAX)",
    "GEO": "no candidate recorded in the area (GEO)",
    "GEO not run": "GEO not run",
    "EXTRA": "only expanded candidates, none with a BOLD proxy (EXTRA)",
    "EXTRA not run": "only expanded candidates, EXTRA not run",
    "HYPO": "only expanded candidates, none confirmed by HYPO",
    "HYPO not run": "only expanded candidates, HYPO not run",
    "Curation": "all its validated species excluded by hand",
}


# --------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------


# The session tables of the species flow, named as build_species_flow's arguments.
FLOW_TABLES = ["ncbi_search_df", "mol_df", "tax_df", "geo_df", "extra_df", "hypo_df"]


def _species_flows(seq_ids):
    """The species flow of each sequence (see build_species_flows) for the
    steps run."""
    state = st.session_state
    return build_species_flows(
        seq_ids,
        **{key: state.get(key) for key in FLOW_TABLES},
        excluded=state.get("excluded_results", []),
        name_table=state.get("name_table"),
    )


def _lost_at(seq_ids):
    """flow_lost_at of each sequence, kept until a step's table or the name
    table changes: rebuilding a flow per sequence on every rerun takes seconds
    on thousands of sequences. Steps replace their tables, never edit them, so
    the same objects mean the same data. An exclusion changes only its own
    sequence's flow."""
    state = st.session_state
    tables = [state.get(key) for key in [*FLOW_TABLES, "name_table"]]
    excluded = set(state.get("excluded_results", []))
    memo = state.get("lost_at_memo")
    if memo is None or any(old is not new for old, new in zip(memo["tables"], tables)):
        memo = {"tables": tables, "excluded": excluded, "lost": {}}
        state.lost_at_memo = memo
    lost = memo["lost"]
    for key in memo["excluded"] ^ excluded:
        lost.pop(key.split("||", 1)[0], None)
    memo["excluded"] = excluded
    missing = [seq_id for seq_id in seq_ids if seq_id not in lost]
    if missing:
        flows = _species_flows(missing)
        lost.update({seq_id: flow_lost_at(links) for seq_id, links in flows.items()})
    return {seq_id: lost[seq_id] for seq_id in seq_ids}


def _assignments(curated_df, threshold):
    """One row per sequence: its assignment after curation, and what to check."""
    table = resolve_sequences(curated_df)
    species_rows = curated_df.dropna(subset=["scientificName"])
    by_seq = species_rows.groupby("seq_id")

    best = best_identity_per_sequence(species_rows, mol_query_cover())
    table["best_identity"] = table["seq_id"].map(best["identity"])
    table["best_via"] = table["seq_id"].map(best["via"])
    table["low_identity"] = table["best_identity"] < threshold
    table["paths"] = table["seq_id"].map(
        by_seq["validation_step"].agg(lambda s: " + ".join(sorted(s.unique())))
    )
    check_hit = table["seq_id"].map(by_seq["hypo_check_hit"].any())
    table["hypo_check_hit"] = check_hit.fillna(False).astype(bool)

    unassigned = table.loc[table["n_species"] == 0, "seq_id"].tolist()
    # Where it lost its last species, as its species flow shows it.
    lost = {seq_id: step for seq_id, (step, _) in _lost_at(unassigned).items()}
    table["lost_at"] = table["seq_id"].map(lost)
    notes = []
    for _, row in table.iterrows():
        if row["n_species"] == 0:
            lost_at = row["lost_at"]
            notes.append(
                "no candidate left"
                if lost_at is None
                else LOSS_REASONS.get(lost_at, lost_at)
            )
            continue
        note = []
        if row["low_identity"]:
            via = f" (via {row['best_via']})" if pd.notna(row["best_via"]) else ""
            note.append(f"best identity below {threshold}%{via}")
        if row["hypo_check_hit"]:
            note.append(CHECK_HIT_NOTE)
        if pd.isna(row["taxonRank"]) and not row["species_level"]:
            note.append("no shared taxon" if row["n_species"] > 1 else "rank unknown")
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
        "HYPO check": assignments["hypo_check_hit"],
    }.get(choice)
    return assignments["seq_id"] if mask is None else assignments.loc[mask, "seq_id"]


def _default_threshold():
    """The MOL filter's low identity warning, so both steps flag alike."""
    stored = st.session_state.get("mol_params") or {}
    value = stored.get("ncbi_low_identity_threshold")
    return int(float(value)) if value else NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT


def _threshold_slider():
    """The identity below which sequences are flagged (Low identity). Starts
    at the MOL filter's low identity warning and follows it when it changes,
    else keeps the user's value."""
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
            "best NCBI identity is below it is flagged; nothing is removed. A "
            "HYPO species without check hit counts with the MOL hit of the "
            "species its CO1 proxy matched."
        ),
    )
    save_widget(THRESHOLD_KEY)
    return threshold


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
    _set_excluded(excluded)


def _toggle_bulk(name, keys):
    """Exclude a group of candidates at once, or restore the ones that group's
    button excluded. Candidates excluded by hand stay excluded, and so do the
    ones the other button also excluded."""
    excluded = set(st.session_state.excluded_results)
    bulk = dict(st.session_state.bulk_excluded)
    if name in bulk:
        mine = set(bulk.pop(name))
        excluded -= mine - set().union(*bulk.values())
    else:
        by_hand = excluded - set().union(*bulk.values())
        bulk[name] = sorted(set(keys) - by_hand)
        excluded |= set(keys)
    st.session_state.bulk_excluded = bulk
    _set_excluded(excluded)


def _set_excluded(excluded):
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
    c1, c2, c3, c4, c5, c6 = st.columns(6)
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
            "rank they share (as in the one-row-per-sequence table)."
        ),
    )
    c4.metric(
        "Unassigned",
        len(unassigned),
        help="Sequences without a validated species.",
    )
    c5.metric(
        "HYPO check",
        int(assignments["hypo_check_hit"].sum()),
        help=(
            "Sequences with a HYPO species whose marker the HYPO check found in "
            "NCBI, although a hypothetical species has no marker in GenBank: "
            "check them and keep or exclude the species."
        ),
    )
    c6.metric(
        f"Identity < {threshold}%",
        int(assignments["low_identity"].sum()),
        help="Sequences whose best NCBI identity is below the threshold.",
    )
    if not higher.empty:
        ranks = pd.Series(
            [
                rank.lower()
                if pd.notna(rank)
                else ("no shared taxon" if n > 1 else "rank unknown")
                for rank, n in zip(higher["taxonRank"], higher["n_species"])
            ]
        )
        st.caption(f"Higher rank: {count_phrase(ranks)}.")
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
            "Hypothetical candidates: no target marker in GenBank, supported by "
            "a CO1 proxy (BOLD/NCBI) with the thresholds of the HYPO step. A "
            "species can be MOL+GEO for one sequence and HYPO for another."
        ),
    )
    c4.metric(
        "Species identified",
        len(resolved_species),
        help="Species to which at least one sequence resolves.",
    )

    excluded = results_df.dropna(subset=["scientificName"])
    excluded = excluded[
        exclusion_keys(excluded).isin(st.session_state.excluded_results)
    ].drop_duplicates(subset=["seq_id", "scientificName"])
    if not excluded.empty:
        with st.expander(f"Excluded by hand ({species_for_sequences(excluded)})"):
            st.caption("Select the sequence and check Keep to restore a species.")
            st.dataframe(
                excluded,
                width="stretch",
                hide_index=True,
                column_order=[
                    "seq_id",
                    "scientificName",
                    "validation_step",
                    "mol_identity",
                    "hypo_check_identity",
                ],
                column_config=column_config(
                    [
                        "seq_id",
                        "scientificName",
                        "validation_step",
                        "mol_identity",
                        "hypo_check_identity",
                    ]
                ),
            )


def _bulk_button(col, name, rows, label, what, criterion):
    """One exclude button, labelled Restore once it has excluded its rows."""
    done = st.session_state.bulk_excluded.get(name)
    if done is not None:
        excluded = set(st.session_state.excluded_results)
        label = f"Restore the {len(set(done) & excluded)} {what} excluded"
    col.button(
        label,
        key=f"exclude_{name}",
        on_click=_toggle_bulk,
        args=(name, exclusion_keys(rows).tolist()),
        disabled=done is None and rows.empty,
        width="stretch",
        help=(
            f"{criterion} They leave the curated results and the exports. "
            "Click again to restore them, or restore one by selecting its "
            "sequence and checking Keep."
        ),
    )


def show_bulk_buttons(assignments, curated_df, threshold):
    """Two buttons excluding at once what a filter flags: the flagged HYPO
    species, or every candidate of the low identity sequences (which then
    become unassigned). A second click restores what the button excluded."""
    species_rows = curated_df.dropna(subset=["scientificName"])
    flagged = species_rows[species_rows["hypo_check_hit"]]
    low = assignments.loc[assignments["low_identity"], "seq_id"]
    low_rows = species_rows[species_rows["seq_id"].isin(low)]
    col1, col2 = st.columns(2)
    _bulk_button(
        col1,
        "HYPO check",
        flagged,
        label=f"Exclude the {len(flagged)} flagged HYPO species",
        what="flagged HYPO species",
        criterion=(
            "Excludes each HYPO species whose marker the HYPO check found in "
            "NCBI, whatever the identity or cover of the hit."
        ),
    )
    _bulk_button(
        col2,
        "Low identity",
        low_rows,
        label=(
            f"Exclude all {len(low_rows)} candidates of the {low.nunique()} "
            "low identity sequences"
        ),
        what="low identity candidates",
        criterion=(
            "Excludes every candidate species of the sequences whose best NCBI "
            f"identity is below {threshold}% (the slider above). These "
            "sequences become unassigned."
        ),
    )


def show_assignments(assignments, curated_df, threshold):
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
            help=(
                f"HYPO check: {CHECK_HIT_NOTE}. "
                f"Low identity: best NCBI identity below {threshold}%."
            ),
        )
        save_widget(FILTER_KEY)
        # Clicking the active button deselects it: show everything then.
        choice = choice or "All"
        show_bulk_buttons(assignments, curated_df, threshold)
        shown = assignments[
            assignments["seq_id"].isin(_filtered_seq_ids(assignments, choice))
        ].reset_index(drop=True)
        st.caption(
            "Click a row to see and curate its candidate species in the "
            '"Candidate species of a (M)OTU/ASV" panel below.'
        )
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
            column_order=ASSIGNMENT_COLS,
            column_config=column_config(
                ASSIGNMENT_COLS,
                scientificName=st.column_config.TextColumn(
                    "Assignment",
                    help="The species, or the lowest taxon its candidates share.",
                ),
                n_species=st.column_config.NumberColumn(
                    "Candidates", help="Candidate species kept for this sequence."
                ),
                note=st.column_config.TextColumn("To Check"),
            ),
        )


# What an unassigned sequence had at the step that lost it: the session table
# that step read, and the sentence above it (the note above says why).
LOST_AT_TABLES = {
    "MOL Filter": ("ncbi_search_df", "This sequence's NCBI hits:"),
    "TAX": ("mol_df", "This sequence's MOL hits:"),
    "EXTRA": ("geo_df", "This sequence's expanded candidates:"),
    "EXTRA not run": ("geo_df", "This sequence's expanded candidates:"),
    "HYPO": ("geo_df", "The expanded candidates HYPO tested:"),
    "HYPO not run": ("geo_df", "This sequence's expanded candidates:"),
}
LOST_AT_COLS = [
    "scientificName",
    "accepted_name",
    "identity_percentage",
    "query_cover",
    "hit_def",
    "hit_url",
    "geo_occurrences",
    "geo_url",
    "taxonURL",
]


def _show_lost_at(seq_id, step):
    """Show what an unassigned sequence had at the step that lost it, as
    that step recorded it."""
    names = _lost_at([seq_id])[seq_id][1]
    if step == "GEO":
        params = st.session_state.geo_params
        min_occ = params["gbif_min_occurrences"]
        extent = params["gbif_filter_extent"]
        df = st.session_state.gbif_search_df
        # Only the candidates lost at GEO, not a congener the MOL filter
        # rejected that TAX kept.
        seq_df = df[(df["seq_id"] == seq_id) & df["scientificName"].isin(names)]
        if seq_df.empty:
            st.caption("GBIF returned no counts for its candidate species.")
            return
        area = area_label(extent, params.get("gbif_area"))
        st.caption(
            f"This sequence's candidates and their GBIF occurrences. GEO keeps a "
            f"species with at least {min_occ} {pluralize(min_occ, 'occurrence')} "
            f"{area}."
        )
        display_gbif_sequence_table(
            seq_df,
            min_occ,
            st.session_state.gbif_search_params["gbif_extents"],
            extent,
            note_empty_area=False,
        )
        return
    if step not in LOST_AT_TABLES:
        return
    key, caption = LOST_AT_TABLES[step]
    df = st.session_state[key]
    rows = df[df["seq_id"] == seq_id]
    if key == "geo_df":
        # Only the candidates that ended at this step: GEO also lists species
        # lost earlier, such as a congener the MOL filter rejected and TAX kept.
        rows = rows[rows["scientificName"].isin(names)]
    name_table = st.session_state.name_table
    if key == "mol_df" and not resolved_names(name_table).empty:
        rows = rows.assign(accepted_name=species_of(rows["scientificName"], name_table))
    if "identity_percentage" in rows.columns:
        rows = rows.sort_values("identity_percentage", ascending=False)
    st.caption(caption)
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_order=[c for c in LOST_AT_COLS if c in rows.columns],
        column_config=column_config(LOST_AT_COLS),
    )


def show_species_flow(seq_id):
    """The selected sequence's candidates at each step, as a Sankey."""
    state = st.session_state
    links = _species_flows([seq_id])[seq_id]
    if not links:
        st.info("No NCBI hit for this sequence.")
        return
    st.caption(
        "Candidate species of this sequence at each step. Hover over a box or "
        "a flow to see its species."
    )
    st.plotly_chart(
        plot_species_flow(links),
        width="stretch",
        config={
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format": "png",
                "scale": 4,
                "filename": f"species_flow_{seq_id}",
            },
        },
    )
    table = species_flow_table(
        seq_id, links, state.get("ncbi_search_df"), state.get("name_table")
    )
    with st.expander(
        f"Species at each step ({len(table)} {pluralize(len(table), 'name')})"
    ):
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config=column_config(["species", "origin", "outcome", "identity"]),
        )


def show_sequence_detail(results_df, assignments, threshold):
    """The selected sequence's assignment and candidates, with Keep checkboxes."""
    with st.container(border=True):
        st.markdown("### 🔬 Candidate species of a (M)OTU/ASV")
        by_seq = assignments.set_index("seq_id")

        labels = {
            seq_id: f"{seq_id} ∅" if n == 0 else f"{seq_id} ⚠" if note else seq_id
            for seq_id, n, note in zip(
                assignments["seq_id"], assignments["n_species"], assignments["note"]
            )
        }
        selected_seq = sequence_selector(labels=labels)
        row = by_seq.loc[selected_seq]
        n = row["n_species"]
        rank = row["taxonRank"].lower() if pd.notna(row["taxonRank"]) else None
        candidates = f", {n} candidate species" if n > 1 else ""
        if n == 0:
            st.info(f"Unassigned: {row['note']}.")
            _show_lost_at(selected_seq, row["lost_at"])
        elif row["species_level"] or rank:
            st.markdown(
                f"Assigned to **{row['scientificName']}** "
                f"({rank or 'species'}){candidates}."
            )
        elif n > 1:
            st.markdown(f"{n} candidate species with no shared taxon.")
        else:
            st.markdown(f"Assigned to **{row['scientificName']}** (rank unknown).")
        if row["note"] and row["n_species"]:
            st.caption(f"⚠ {row['note'][:1].upper()}{row['note'][1:]}.")

        tab_table, tab_flow = st.tabs(["Candidates", "Plot (steps)"])
        with tab_table:
            _show_candidates(results_df, assignments, threshold, selected_seq)
        with tab_flow:
            show_species_flow(selected_seq)


def _show_candidates(results_df, assignments, threshold, selected_seq):
    """The selected sequence's candidates, with Keep checkboxes."""
    species_rows = results_df.dropna(subset=["scientificName"])
    table = species_rows[species_rows["seq_id"] == selected_seq].reset_index(drop=True)
    if table.empty:
        st.caption("No candidate species.")
        return
    table = flag_low_identity(
        table, threshold, assignments.set_index("seq_id")["best_identity"]
    )
    row_keys = exclusion_keys(table).tolist()
    excluded = set(st.session_state.excluded_results)
    table.insert(0, "keep", [key not in excluded for key in row_keys])

    legends = ["Uncheck Keep to exclude a candidate from this sequence."]
    if table["low_identity_warning"].any():
        legends.append(f":red[■] Best identity below {threshold}%")
    if table["hypo_check_hit"].any():
        legends.append(CHECK_HIT_LEGEND)
    st.caption(" · ".join(legends))

    editor_key = f"keep_{selected_seq}_{st.session_state.get('_keep_version', 0)}"
    st.data_editor(
        styled_or_plain(table, highlight_check_hit, highlight_low_identity),
        width="stretch",
        hide_index=True,
        key=editor_key,
        on_change=_apply_keep_edits,
        args=(editor_key, row_keys),
        disabled=[c for c in table.columns if c != "keep"],
        column_order=["keep", *CANDIDATE_COLS],
        column_config=column_config(["keep", *CANDIDATE_COLS]),
    )


def _species_breakdown_columns(sp_df, is_hypo):
    """The columns of the per-species breakdown table of one validation."""
    if not is_hypo:
        return [
            "seq_id",
            "validation_step",
            "mol_identity",
            "mol_query_cover",
            "geo_occurrences",
        ]
    shown = [
        "seq_id",
        "validation_step",
        "proxy_identity",
        "proxy_query_cover",
        "geo_occurrences",
    ]
    if sp_df["hypo_check_identity"].notna().any():
        shown += ["hypo_check_identity", "hypo_check_query_cover"]
    return shown


def _show_all_hits_expander(species, is_hypo):
    """Expander showing all individual validation hits for a species."""
    if is_hypo:
        hypo_filter_df = st.session_state.get("hypo_filter_df")
        if hypo_filter_df is None:
            return
        hits = hypo_filter_df[hypo_filter_df["scientificName"] == species]
        if hits.empty:
            return

        shown = [
            "seq_id",
            "proxy_hit_species",
            "proxy_identity",
            "proxy_query_cover",
            "proxy_hit_url",
            "proxy_url",
        ]
        with st.expander(f"All proxy validation hits ({len(hits)})"):
            st.dataframe(
                hits,
                width="stretch",
                hide_index=True,
                column_order=shown,
                column_config=column_config(shown),
            )
    else:
        mol_df = st.session_state.get("mol_df")
        if mol_df is None:
            return
        # The species' hits under any of its NCBI names.
        keys = species_key(mol_df["scientificName"], st.session_state.name_table)
        hits = mol_df[keys == normalize_name(species)]
        if hits.empty:
            return

        with st.expander(f"All NCBI hits ({len(hits)})"):
            st.dataframe(
                hits,
                width="stretch",
                hide_index=True,
                column_order=NCBI_HIT_COLS,
                column_config=column_config(
                    NCBI_HIT_COLS,
                    scientificName=st.column_config.TextColumn("NCBI Name"),
                ),
            )


def show_results_per_species(results_df, assignments, threshold, same_species):
    """Taxonomy, links and sequences of one species. same_species: see
    same_species_as."""
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
        same = same_species.get(selected_species)
        if same:
            st.warning(
                f"Possibly the same species as {same}: another source counts "
                "them as one species.",
                icon="⚠️",
            )

        links = []
        for col, label in (
            ("taxonURL", "WoRMS"),
            ("geo_url", "GBIF"),
            ("mol_hit_url", "NCBI"),
            ("proxy_url", "BOLD"),
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
            shown = _species_breakdown_columns(step_df, is_hypo)
            st.dataframe(
                styled_or_plain(
                    flag_low_identity(step_df, threshold, best_identity),
                    highlight_check_hit,
                    highlight_low_identity,
                ),
                width="stretch",
                hide_index=True,
                column_order=shown,
                column_config=column_config(shown),
            )
            _show_all_hits_expander(selected_species, is_hypo=is_hypo)


def show_curated_results(curated_df, assignments, threshold):
    """Every kept candidate of every sequence: the rows the CSV export writes."""
    with st.container(border=True):
        st.markdown("### ✅ Curated results")
        table = flag_low_identity(
            curated_df, threshold, assignments.set_index("seq_id")["best_identity"]
        )
        legends = ["One row per sequence and kept candidate species."]
        if table["low_identity_warning"].any():
            legends.append(f":red[■] Best identity below {threshold}%")
        if table["hypo_check_hit"].any():
            legends.append(CHECK_HIT_LEGEND)
        if table["scientificName"].isna().any():
            legends.append(":gray[■] No species assigned")
        st.caption(" · ".join(legends))
        st.dataframe(
            styled_or_plain(
                table,
                highlight_check_hit,
                highlight_low_identity,
                highlight_empty_sequence,
            ),
            width="stretch",
            hide_index=True,
            column_order=["seq_id", *CANDIDATE_COLS],
            column_config=column_config(["seq_id", *CANDIDATE_COLS]),
        )


# Search steps whose query date reflects the state of a live external database.
_SOURCE_STEPS = [
    ("ncbi_search", "NCBI", "MOL"),
    ("worms_search", "WoRMS", "TAX"),
    ("gbif_search", "GBIF", "GEO"),
    ("bold_search", "BOLD", "EXTRA"),
    ("hypo_search", "NCBI", "HYPO"),
    ("worms_higher_taxa", "WoRMS", "One row per sequence table"),
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


def show_exports(curated_df):
    """Every download in one place, with the provenance of the analysis."""
    with st.container(border=True):
        st.markdown("### 💾 Export")
        analysis_name = st.session_state.get("analysis_name", "results")

        if curated_df is not None:
            st.caption(
                "Results after curation: one row per sequence and candidate "
                "species, or one row per sequence at its assigned rank. The "
                "second follows Darwin Core and the FAIRe taxa table, ready for "
                "a GBIF submission."
            )
            export_cols = [c for c in EXPORT_COLS if c in curated_df.columns]
            dl1, dl2 = st.columns(2)
            # Built when clicked, from the results shown now.
            dl1.download_button(
                label="📥 One row per candidate species (CSV)",
                data=lambda: curated_df[export_cols].to_csv(index=False),
                file_name=f"{analysis_name}_candidates.csv",
                mime="text/csv",
                width="stretch",
                key="results_csv",
            )
            with dl2:
                # Built in the script run, where the session's contact email
                # and the spinner apply; offered for the results it was built
                # from, which are hashed only once an export exists.
                shown = None
                if st.button(
                    "📥 One row per sequence (Darwin Core CSV)",
                    width="stretch",
                    key="gbif_csv",
                ):
                    shown = fingerprint(curated_df[export_cols])
                    with st.spinner("Building the table..."):
                        gbif_df = build_gbif_export_df(
                            curated_df, db_path=st.session_state.get("db_path")
                        )
                    st.session_state._gbif_export = (
                        shown,
                        gbif_df.to_csv(index=False),
                        failed_items(gbif_df),
                    )
                built = st.session_state.get("_gbif_export")
                if built and built[0] == (
                    shown or fingerprint(curated_df[export_cols])
                ):
                    if built[2]:
                        n = len(built[2])
                        st.warning(
                            f"WoRMS could not be reached for {n} higher-level "
                            f"{pluralize(n, 'taxon')} ({name_list(built[2])}): "
                            "their ID and author are left empty. Build the "
                            "one-row-per-sequence table again later to "
                            "complete them."
                        )
                    st.download_button(
                        label="⬇️ Download the table",
                        data=built[1],
                        file_name=f"{analysis_name}_assignments.csv",
                        mime="text/csv",
                        width="stretch",
                        key="gbif_download",
                    )

        db_path = st.session_state.get("db_path")
        if not db_path or not Path(db_path).is_file():
            return
        # On Cloud the analysis is on the server, not on the user's computer.
        keep = (
            "Download it to keep it or to share it."
            if is_streamlit_cloud()
            else "It is already saved on this computer: download it to keep a "
            "copy elsewhere or to share it."
        )
        st.caption(
            "The whole analysis in one file: the results of every step, the "
            f"database searches and the settings used. {keep}"
        )
        st.download_button(
            label=f"📥 Analysis file ({Path(db_path).name})",
            data=Path(db_path).read_bytes,
            file_name=f"{analysis_name}.db",
            mime="application/x-sqlite3",
            help="One SQLite file: it can be uploaded again on the Start tab, or "
            "read with any SQLite tool.",
            width="stretch",
            key="db_download",
        )
    show_provenance_section()


# --------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


def results_main():
    """Render the Results tab."""
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
        name_table=st.session_state.get("name_table"),
    )
    if results_df.empty:
        st.info("📊 No results available.")
        show_exports(None)
        return

    curated_df = filter_excluded_results(
        results_df, set(st.session_state.excluded_results)
    )

    with st.container(border=True):
        threshold = _threshold_slider()
        assignments = _assignments(curated_df, threshold)
        show_summary(assignments, curated_df, results_df, threshold)
        display_same_species(curated_df)

    show_assignments(assignments, curated_df, threshold)
    show_sequence_detail(results_df, assignments, threshold)
    show_curated_results(curated_df, assignments, threshold)
    show_results_per_species(
        results_df,
        assignments,
        threshold,
        same_species_as(curated_df["scientificName"]),
    )
    show_exports(curated_df)
