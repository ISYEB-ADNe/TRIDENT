"""Streamlit UI components and helpers shared by the tabs."""

import concurrent.futures
import contextlib
import functools
import io
import logging
import os
import queue
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, fields, asdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from loguru import logger
from PIL import Image

from trident.clients.ncbi import BLAST_SLOW_THREADS
from trident.core.utils import pluralize
from trident.core import stop
from trident.core.database import get_connection
from trident.pipelines.names import same_species_groups
from trident.logging import get_log_level
from trident.ui.defaults import (
    NCBI_GAP_MIN_TOP_DEFAULT,
    NCBI_GAP_SIZE_DEFAULT,
    NCBI_METHOD_DEFAULT,
    NCBI_QUERY_COVER_DEFAULT,
)


# Suppress Streamlit's thread context warnings (Streamlit uses standard logging internally)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)

# --------------------------------------------------------------------
# SHARED COLUMN CONFIGS
# -------------------------------------------------------------------

_Text = st.column_config.TextColumn
_Number = st.column_config.NumberColumn
_Link = st.column_config.LinkColumn
_Check = st.column_config.CheckboxColumn


def _percent(label: str, help: str) -> st.column_config.NumberColumn:
    return _Number(label, format="%.2f", help=help)


def _count(label: str, help: str) -> st.column_config.NumberColumn:
    return _Number(label, format="%d", help=help)


def _link(label: str) -> st.column_config.LinkColumn:
    return _Link(label, display_text="🔗 View")


# The display settings of every column the app shows, by column name: a tab
# lists the columns it shows and takes their settings with column_config().
COL = {
    # The (M)OTU/ASV
    "seq_id": _Text(
        "Sequence ID", help="Identifier of the (M)OTU/ASV, from the FASTA header."
    ),
    "dna_sequence": _Text("DNA Sequence", help="The (M)OTU/ASV sequence."),
    "seq_length": _count("Length (bp)", "Length of the sequence in base pairs."),
    # Taxonomy (Darwin Core)
    "scientificName": _Text(
        "Scientific Name",
        help="Species name: as NCBI gives it in the MOL step, the WoRMS "
        "accepted name from TAX on.",
    ),
    "kingdom": _Text("Kingdom"),
    "phylum": _Text("Phylum"),
    "class": _Text("Class"),
    "order": _Text("Order"),
    "family": _Text("Family"),
    "genus": _Text("Genus"),
    "specificEpithet": _Text("Epithet", help="Species epithet."),
    "scientificNameAuthorship": _Text(
        "Authorship",
        help="Author and year of the name, as WoRMS gives them; in parentheses "
        "when the species is no longer in the genus it was described in.",
    ),
    "taxonRank": _Text("Rank", help="Taxonomic rank of the species."),
    "taxonID": _Text(
        "Taxon ID", help="Identifier of the taxon in the taxonomy database."
    ),
    "taxonID_db": _Text("Taxon DB", help="Taxonomy database of the taxon ID."),
    "taxonURL": _link("WoRMS Link"),
    "genusID": _count(
        "WoRMS Genus ID", "WoRMS ID of the genus the species was listed under."
    ),
    "ncbi_synonyms": _Text(
        "NCBI synonym",
        help="Names NCBI uses for this species that differ from its WoRMS name "
        "(synonyms or spelling variants).",
    ),
    "same_species_as": _Text(
        "Possibly same species as",
        help="Another source (in brackets) counts these species as one. WoRMS "
        "may keep one species under two names, or the sources may disagree. "
        "The names are kept and counted apart.",
    ),
    # BLAST hits (MOL search and list, HYPO check)
    "identity_percentage": _percent(
        "Identity (%)",
        "Percentage of matching bases in the best alignment with the GenBank record.",
    ),
    "query_cover": _percent(
        "Query Cover (%)",
        "Percentage of the query sequence covered by the best alignment with the "
        "GenBank record.",
    ),
    "hit_url": _link("NCBI Link"),
    "hit_def": _Text("Hit Definition", help="Title of the GenBank record."),
    "ncbi_taxid": _count("NCBI Taxon ID", "NCBI taxonomy ID of the hit's species."),
    "align_length": _count(
        "Alignment Length", "Length of the aligned region in base pairs."
    ),
    "identities": _count("Identities", "Number of identical bases in the alignment."),
    "gaps": _count("Gaps", "Number of gaps introduced in the alignment."),
    "query_start": _count(
        "Query Start", "Start position of the alignment on the query sequence."
    ),
    "query_end": _count(
        "Query End", "End position of the alignment on the query sequence."
    ),
    "filter_method": _Text(
        "Filter Method",
        help="How the MOL filter kept the hits of the sequence: above a "
        "barcoding gap, or within a similarity range of the best hit.",
    ),
    # MOL summaries
    "n_blast_hits": _count("BLAST Hits", "Hits NCBI BLAST returned for this sequence."),
    "max_identity": _percent("Max Identity (%)", "Highest identity among the hits."),
    "min_identity": _percent("Min Identity (%)", "Lowest identity among the hits."),
    "identity_range": _percent(
        "Identity Range (%)", "Spread between the highest and lowest identity."
    ),
    "n_hits": _count("Hits", "Number of hits."),
    "n_species": _count("Species", "Number of species."),
    "top_identity": _percent(
        "Top Identity (%)", "Highest identity found for this sequence."
    ),
    # Best MOL hit of a species (TAX on)
    "mol_identity": _percent(
        "MOL Identity (%)", "Identity of the sequence's best MOL hit on this species."
    ),
    "mol_query_cover": _percent("MOL Query Cover (%)", "Query cover of that hit."),
    "mol_hit_url": _link("NCBI Link"),
    "in_mol": _Check("In MOL", help="The species is a MOL species of this sequence."),
    # TAX
    "n_mol_species": _count(
        "MOL Species", "Species of the sequence's MOL list (by their WoRMS name)."
    ),
    "n_tax_species": _count("TAX Species", "Species of the sequence's TAX list."),
    "n_both": _count("In Both", "Species in both the MOL and the TAX lists."),
    "n_added": _count("Added", "Species added by TAX."),
    "accepted_name": _Text(
        "WoRMS Accepted Name",
        help="WoRMS lists the MOL name as a synonym of this name, or links them "
        "through the NCBI taxid.",
    ),
    "note": _Text(
        "Why",
        help="Empty: WoRMS has no record of the name. Otherwise WoRMS has "
        "records but none gives one accepted species, so the name is kept as "
        "NCBI gives it.",
    ),
    "genus_in_tax": _Check(
        "Genus in TAX",
        help="The genus of this name is in the TAX list: its species are "
        "candidates for the sequence anyway.",
    ),
    "mol_species": _Text(
        "MOL Species in It", help="The MOL species that belong to this genus."
    ),
    "sequences": _Text(
        "Kept for Sequences",
        help="Sequences whose TAX list keeps this genus, out of the sequences "
        "with this genus name.",
    ),
    # GEO
    "geo_url": _link("GBIF Link"),
    "geo_occurrences": _count(
        "GBIF Occurrences",
        "Number of GBIF occurrence records with coordinates of this species in "
        "the search area "
        "(the filtering area in the GEO list).",
    ),
    "geo_extent": _Text("Extent", help="Search area of the occurrence count."),
    "gbif_scientificName": _Text(
        "Name in GBIF",
        help="The name GBIF gives the species, when it differs from WoRMS; a "
        "subspecies name when GBIF holds it as one (see the GBIF match log).",
    ),
    # EXTRA
    "proxy_id": _Text("BOLD Process ID", help="The BOLD record of the proxy."),
    "proxy_sequence": _Text(
        "BOLD DNA Sequence", width="large", help="The proxy's CO1 sequence."
    ),
    "proxy_url": _link("BOLD Link"),
    "bold_scientificName": _Text(
        "Name in BOLD",
        help="The name BOLD files the records under, when the species was "
        "found under another of its names (NCBI or GBIF).",
    ),
    "bold_outcome": _Text("Why", help="Why the species has no usable BOLD sequence."),
    "bold_url": _link("BOLD Link"),
    "n_bold_sequences": _count(
        "BOLD Sequences", "BOLD sequences kept by the filter, as proxies."
    ),
    "n_species_queried": _count(
        "Species Queried", "Species of this sequence searched in BOLD."
    ),
    "n_species_found": _count(
        "Species Found", "Species with at least one usable BOLD sequence."
    ),
    "n_species_missing": _count(
        "Species Missing", "Species without a usable BOLD sequence."
    ),
    # HYPO
    "proxy_hit_species": _Text(
        "MOL Species Hit",
        help="The MOL species of this sequence that the BLAST of the candidate's "
        "BOLD CO1 sequence hit.",
    ),
    "proxy_identity": _percent(
        "Proxy Identity (%)", "Identity of the CO1 proxy to that MOL species."
    ),
    "proxy_query_cover": _percent("Proxy Query Cover (%)", "Query cover of that hit."),
    "proxy_hit_def": _Text("Hit Definition", help="Title of the GenBank record."),
    "proxy_hit_url": _link("NCBI Link"),
    "hit_found": _Check(
        "Hit Found", help="The check found the target marker of the species in GenBank."
    ),
    "scientificName_NCBI": _Text(
        "Check Hit Species", help="The GenBank species of the check hit."
    ),
    "hypo_check_identity": _percent(
        "Check Identity (%)",
        "Identity of the best hit the HYPO check found for the target marker "
        "(empty: marker not found in GenBank).",
    ),
    "hypo_check_query_cover": _percent(
        "Check Query Cover (%)", "Query cover of that hit."
    ),
    "hypo_check_hit_url": _link("NCBI Link"),
    "hypo_check_hit": _Check(
        "Marker in GenBank",
        help="The HYPO check found the species' target marker in GenBank, "
        "although a hypothetical species has none: check it.",
    ),
    # Results
    "validation_step": _Text(
        "Validation",
        help="MOL+GEO: a MOL species of the sequence that GEO kept. HYPO: a "
        "hypothetical species, validated through its BOLD CO1 proxy.",
    ),
    "via_mol_identity": _percent(
        "Via Identity (%)",
        "HYPO: this sequence's best MOL identity to the MOL species its proxy hit.",
    ),
    "keep": _Check("Keep", help="Uncheck to exclude the candidate from this sequence."),
    "best_identity": _percent(
        "Best Identity (%)",
        "Best NCBI identity among the sequence's candidates: the MOL hit, or "
        "the HYPO check hit of a HYPO species (hits covering at least the MOL "
        "query cover). A HYPO species without check hit counts with the MOL "
        "hit of the species its CO1 proxy hit.",
    ),
    "paths": _Text(
        "Validation",
        help="The validation of the sequence's candidates (see Validation).",
    ),
    # Species flow table
    "species": _Text("Species"),
    "origin": _Text("Origin", help="NCBI hit, or added by TAX."),
    "outcome": _Text("Ended At", help="The step where the name ended."),
    "identity": _percent(
        "Best NCBI Identity (%)",
        "Best identity among this sequence's NCBI hits on the name.",
    ),
}


# The classification columns the tables show, from the genus up.
RANK_COLUMNS = [
    "genus",
    "specificEpithet",
    "family",
    "order",
    "class",
    "phylum",
    "kingdom",
]


def column_config(columns, **overrides) -> dict:
    """The display settings of *columns* (from COL), with *overrides* for a
    table that labels a column otherwise. A column without settings raises:
    Streamlit would hide it."""
    return {col: overrides[col] if col in overrides else COL[col] for col in columns}


# The names the Settings panel gives the log levels.
LOG_LEVEL_LABELS = {"INFO": "Standard", "DEBUG": "Verbose"}


# Help text for the NCBI BLAST settings shared by the MOL and HYPO tabs.
BLAST_HELP = {
    "max_hits": (
        "Maximum number of GenBank records returned per query sequence; "
        "lower-scoring hits beyond this limit are not retrieved."
    ),
    "ev_exponent": (
        "Maximum E-value, as 10^-x: only hits at or below this "
        "threshold are kept. A larger x is stricter."
    ),
    "batch_size": (
        "Number of query sequences sent together in one NCBI BLAST search. NCBI "
        "asks for fewer, larger searches: above about 100 searches a day it moves "
        "a user to a slower queue. A search too heavy for NCBI is split into "
        "smaller ones; if many batches fail, lower the batch size."
    ),
    "n_threads": (
        "Number of BLAST searches run at NCBI at the same time. More can speed up "
        "large inputs, but NCBI throttles heavy use, so too many can make batches "
        "fail. A search left waiting at NCBI makes way for another one, up to "
        f"{BLAST_SLOW_THREADS} more."
    ),
    "give_up": (
        "Minutes a BLAST search may wait at NCBI. NCBI sometimes leaves a search "
        "waiting for hours: past this time, it is sent again once, then counted "
        "as failed (re-run the step to retry it). Most searches take 1 to 3 "
        "min; raise it if large batches keep failing this way."
    ),
}

# --------------------------------------------------------------------
# SESSION STATE MANAGEMENT
# -------------------------------------------------------------------


@dataclass
class AppState:
    """Session state of an analysis, in pipeline order: reset_state_after
    resets the fields after a given one, so the order matters."""

    fasta_uploader_key: int = 0
    analysis_name: str = ""
    db_path: str = ""
    sequences_id: list[str] | None = None
    sequences_df: "pd.DataFrame | None" = None

    # The *_logs fields hold each step's log lines, written and read by
    # prefix (StreamlitLogSink, display_run_button).
    ncbi_logs: list[str] = field(default_factory=list)
    ncbi_search_flag: bool = False
    ncbi_search_params: dict[str, Any] = field(default_factory=dict)
    ncbi_search_failed: list[str] = field(default_factory=list)
    ncbi_search_df: "pd.DataFrame | None" = None
    ncbi_filter_flag: bool = False
    mol_params: dict[str, Any] = field(default_factory=dict)
    mol_species: list[str] = field(default_factory=list)
    mol_summary_df: "pd.DataFrame | None" = None
    mol_df: "pd.DataFrame | None" = None

    worms_logs: list[str] = field(default_factory=list)
    worms_search_flag: bool = False
    tax_params: dict[str, Any] = field(default_factory=dict)
    tax_species: list[str] = field(default_factory=list)
    tax_summary_df: "pd.DataFrame | None" = None
    tax_failed: list[str] = field(default_factory=list)
    tax_homonyms: "pd.DataFrame | None" = None
    tax_mol_rejected: "pd.DataFrame | None" = None
    name_table: "pd.DataFrame | None" = None
    tax_df: "pd.DataFrame | None" = None

    gbif_logs: list[str] = field(default_factory=list)
    gbif_search_flag: bool = False
    gbif_search_params: dict[str, Any] = field(default_factory=dict)
    gbif_search_failed: list[str] = field(default_factory=list)
    gbif_search_df: "pd.DataFrame | None" = None
    gbif_filter_flag: bool = False
    gbif_filter_df: "pd.DataFrame | None" = None
    geo_params: dict[str, Any] = field(default_factory=dict)
    geo_and_mol_species: list[str] = field(default_factory=list)
    geo_summary_df: "pd.DataFrame | None" = None
    geo_df: "pd.DataFrame | None" = None

    bold_logs: list[str] = field(default_factory=list)
    bold_search_flag: bool = False
    bold_search_params: dict[str, Any] = field(default_factory=dict)
    bold_input_species: list[str] = field(default_factory=list)
    bold_missing_species: list[str] = field(default_factory=list)
    bold_search_failed: list[str] = field(default_factory=list)
    bold_search_df: "pd.DataFrame | None" = None
    bold_filter_logs: list[str] = field(default_factory=list)
    bold_filter_flag: bool = False
    bold_filter_df: "pd.DataFrame | None" = None
    bold_species: list[str] = field(default_factory=list)
    extra_summary_df: "pd.DataFrame | None" = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    extra_df: "pd.DataFrame | None" = None

    hypo_logs: list[str] = field(default_factory=list)
    hypo_search_flag: bool = False
    hypo_merge_params: dict[str, Any] = field(default_factory=dict)
    hypo_search_failed: list[str] = field(default_factory=list)
    hypo_merge_df: "pd.DataFrame | None" = None
    hypo_filter_flag: bool = False
    hypo_filter_params: dict[str, Any] = field(default_factory=dict)
    hypo_filter_df: "pd.DataFrame | None" = None
    hypo_check_logs: list[str] = field(default_factory=list)
    hypo_check_flag: bool = False
    hypo_check_params: dict[str, Any] = field(default_factory=dict)
    hypo_check_failed: list[str] = field(default_factory=list)
    hypo_check_df: "pd.DataFrame | None" = None
    hypo_check_summary_df: "pd.DataFrame | None" = None
    hypo_params: dict[str, Any] = field(default_factory=dict)
    hypo_df: "pd.DataFrame | None" = None

    excluded_results: list[str] = field(default_factory=list)
    # The keys each bulk button excluded, so a second click restores them.
    bulk_excluded: dict[str, list[str]] = field(default_factory=dict)
    # Results' lost-at steps and the tables they came from (see _lost_at in
    # results.py); any reset drops it, and with it the old tables.
    lost_at_memo: dict[str, Any] | None = None

    # reset=False: not a pipeline output, so reset_state_after leaves it intact
    # (it is the cross-tab widget persistence store).
    _widget_overrides: dict[str, Any] = field(
        default_factory=dict, metadata={"reset": False}
    )


def init_session_state():
    defaults = asdict(AppState())
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def coerce_stored_value(value, default):
    """Restore a persisted override to the type of *default*.

    Cache inputs round-trip through SQLite as strings, and the value written
    depends on the caller: a notebook passing ``98.0`` stores ``"98.0"`` where
    the app's int widget stores ``"98"``. A plain ``type(default)(value)`` cast
    fails on the former, so numbers go through ``float`` first.

    Anything that cannot be represented as the default's type falls back to
    *default*, since feeding a widget a value it cannot render corrupts the
    parameter it is bound to.
    """
    if default is None:
        return value
    if value is None:
        return default

    target = type(default)

    # bool before int: bool is an int subclass, and bool("False") is True.
    if target is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return default

    if isinstance(value, target) and not isinstance(value, bool):
        return value

    try:
        return int(float(value)) if target is int else target(value)
    except (TypeError, ValueError):
        return default


def persist_value(key: str, default):
    """Pre-seed a widget key from the persistent override store.

    Call BEFORE creating the widget. Only acts when the key is absent
    from session_state (e.g. after a tab switch removed it). When the
    widget is already on screen, the key exists and this is a no-op.
    """
    if key not in st.session_state:
        overrides = st.session_state._widget_overrides
        if key in overrides:
            st.session_state[key] = coerce_stored_value(overrides[key], default)
        else:
            st.session_state[key] = default


def save_widget(key: str):
    """Copy the current widget value into the persistent override store.

    Call AFTER creating the widget.
    """
    st.session_state._widget_overrides[key] = st.session_state[key]


def reset_state_after(key_name: str):
    """Reset all AppState fields *after* key_name (exclusive).

    ``_widget_overrides`` is the cross-tab widget persistence store, not a
    pipeline output, so it is left untouched (it happens to be the last field,
    which would otherwise put it in every reset's path and forget remembered
    parameters for not-yet-rerun downstream tabs).
    """
    default = asdict(AppState())
    flds = fields(AppState())
    ordered = [f.name for f in flds]
    no_reset = {f.name for f in flds if f.metadata.get("reset") is False}

    if key_name not in ordered:
        raise ValueError(f"{key_name} is not a field of AppState")

    start_idx = ordered.index(key_name) + 1  # exclusive
    for name in ordered[start_idx:]:
        if name in no_reset:
            continue
        st.session_state[name] = default[name]
        if name.endswith("_flag"):
            drop_jobs(name)


# --------------------------------------------------------------------
# OTHER UI HELPERS
# -------------------------------------------------------------------


def get_analysis_status(
    df_name: str, params_name: str, current_params: dict, keys: list[str] = None
) -> str:
    """NEW when the step has no result, IDENTICAL when *current_params* equal
    the ones of its result (only *keys*, if given), CHANGED otherwise."""
    stored_df = st.session_state.get(df_name)
    stored_params = st.session_state.get(params_name)

    if stored_df is None or stored_params is None:
        return "NEW"

    is_identical = True

    if keys is not None:
        current_params = {k: current_params[k] for k in keys}
        stored_params = {k: stored_params.get(k) for k in keys}

    for k, v in current_params.items():
        if v != stored_params.get(k):
            is_identical = False

    return "IDENTICAL" if is_identical else "CHANGED"


# The choices next to a step's run button: reuse what is saved for the
# settings, also ask again for the items saved without a result, or ask for
# everything again.
REUSE_SAVED = "Reuse saved results"
RETRY_EMPTY = "Ask again for empty results"
ERASE_SAVED = "Erase saved results"
CACHE_MODES = [REUSE_SAVED, RETRY_EMPTY, ERASE_SAVED]


def display_run_button(
    mode: str, new_string: str, btn_key: str, logs_prefix: str = "", starts: int = 0
) -> tuple[bool, str]:
    """Renders the standardized status message and button.

    The button's key is *btn_key* with the number of times the step was
    started (*starts*): a click on the button still on screen from before the
    step started (a double click) has the old key, and Streamlit ignores it
    instead of starting the step again.

    Returns:
        Tuple of (clicked, cache_mode), cache_mode one of CACHE_MODES.
    """

    # Initialize variables with defaults for the button
    btn_label = "▶️ Start Analysis"
    btn_type = "primary"
    btn_help = ""

    if mode == "NEW":
        st.info(new_string)
        btn_label = "▶️ Start Analysis"
        btn_type = "primary"
        btn_help = "Begin fresh analysis"

    elif mode == "IDENTICAL":
        if logs_prefix:
            with st.expander("✅ Step done: view its log", expanded=False):
                if st.session_state.get(f"{logs_prefix}logs"):
                    st.text("\n".join(st.session_state[f"{logs_prefix}logs"]))
                else:
                    st.info("No logs available for this session.")
        # Define button for re-running even if identical
        btn_label = "🔄 Re-run with Same Settings"
        btn_type = "secondary"
        btn_help = (
            "Reuses the saved results and asks only for the failed or missing ones"
        )

    else:  # This handles "CHANGED"
        st.warning("⚠️ Settings have changed since the last run.")
        btn_label = "🚀 Run with New Settings"
        btn_type = "primary"
        btn_help = "Run with the new settings"

    c1, c2 = st.columns([2, 1])
    with c1:
        clicked = st.button(
            btn_label,
            type=btn_type,
            help=btn_help,
            width="stretch",
            key=f"{btn_key}_{starts}",
        )
    with c2:
        cache_mode = st.selectbox(
            "Saved results",
            options=CACHE_MODES,
            index=0,
            key=f"{btn_key}_cache_mode",
            label_visibility="collapsed",
        )
    return clicked, cache_mode


def run_step_workflow(
    *,
    df_key: str,
    params_key: str,
    flag_key: str,
    tab_id: str,
    compare_keys: list[str] | None,
    current_params: dict,
    job_fn: Callable,
    job_kwargs: dict | None = None,
    new_string: str,
    btn_key: str,
    logs_prefix: str = "",
    threaded: bool = False,
    status_label: str = "",
    before_button: Callable[[], None] | None = None,
    stoppable: bool = True,
) -> str:
    """Handle execution + button for a pipeline step. Returns mode.

    A threaded step gets a stop button while it runs, unless not *stoppable*
    (a local computation, which has no queries to keep).
    """
    mode = get_analysis_status(df_key, params_key, current_params, keys=compare_keys)

    if st.session_state.get(flag_key):
        # The run uses the settings of the click, not the widgets' current
        # values: a setting edited while it runs (a rerun) leaves it going, and
        # shows as "Settings have changed" once it is done.
        run = st.session_state[STEP_RUNS][flag_key]
        st.session_state[_JOB_STEP] = flag_key
        # A stopped step keeps no result (run_with_progress raises StepStopped):
        # its finished queries are cached, and running it again with the same
        # settings continues from there.
        try:
            if threaded:
                if stoppable:
                    _stop_button(flag_key, btn_key)
                progress_bar = st.progress(0, text="Initializing...")
                with st.status(status_label, expanded=False) as status:
                    job_fn(
                        force_rerun=run["force"],
                        retry_empty=run["retry"],
                        status=status,
                        progress_bar=progress_bar,
                        **run["job_kwargs"],
                    )
            else:
                with st.status(status_label, expanded=False):
                    job_fn(
                        force_rerun=run["force"],
                        retry_empty=run["retry"],
                        **run["job_kwargs"],
                    )
        except stop.StepStopped:
            pass
        else:
            # A stop clicked once the jobs had all finished came too late.
            st.session_state.get(STOPPED_STEPS, set()).discard(flag_key)

        st.session_state[flag_key] = False
        st.session_state["requested_tab"] = tab_id
        st.rerun()

    if before_button:
        before_button()

    if flag_key in st.session_state.get(STOPPED_STEPS, ()):
        st.info(
            "⏹️ Search stopped. The results already received are kept: run the "
            "step again to continue where it stopped."
        )
    runs = st.session_state.setdefault(STEP_RUNS, {})
    starts = runs.get(flag_key, {}).get("starts", 0)
    clicked, cache_mode = display_run_button(
        mode, new_string, btn_key=btn_key, logs_prefix=logs_prefix, starts=starts
    )
    if clicked:
        # Only ERASE_SAVED forces a fresh run. Otherwise the step
        # reuses the items saved for its current settings and runs the others,
        # whether the settings changed or a stopped run continues: every
        # setting in *compare_keys* is part of the step's cache key.
        runs[flag_key] = {
            "force": cache_mode == ERASE_SAVED,
            "retry": cache_mode == RETRY_EMPTY,
            "job_kwargs": job_kwargs or {},
            "starts": starts + 1,
        }
        drop_jobs(flag_key)
        st.session_state[flag_key] = True
        st.rerun()

    return mode


def reset_for_new_analysis(key_name: str):
    """Reset the app for a newly loaded analysis.

    Clears the pipeline state after *key_name*, as ``reset_state_after`` does,
    and also the settings remembered across tabs (``_widget_overrides``), so
    nothing from the previous analysis (coordinates, thresholds, selected
    sequence) carries over to the new one.
    """
    reset_state_after(key_name)
    st.session_state._widget_overrides = {}
    # Forget which preset each selector last applied, or a preset equal to the
    # last one would not be applied again to the now empty settings.
    for key in [k for k in st.session_state if str(k).startswith(PRESET_MARKER)]:
        del st.session_state[key]


def show_filter_success(kept: int, total: int, noun: str, tail: str = "") -> None:
    """Say how many of the search's *total* rows (*noun*) a filter kept."""
    st.success(f"✅ Filtering complete: {kept:,}/{total:,} {noun} kept{tail}")


# Green, the colour of a step done: a next-step button shows once its step has
# a result. Streamlit gives a keyed widget the class st-key-<key>.
NEXT_STEP_STYLE = """<style>
[class*="st-key-next_step_"] button {
    background-color: #1e8e3e; border-color: #1e8e3e; color: white;
}
[class*="st-key-next_step_"] button:hover,
[class*="st-key-next_step_"] button:focus,
[class*="st-key-next_step_"] button:active {
    background-color: #176f30; border-color: #176f30; color: white;
}
</style>"""


def next_step_button(label: str, tab_id: str):
    """Render a right-aligned navigation button, green: its step is done."""
    st.html(NEXT_STEP_STYLE)
    cols = st.columns([5, 1])
    with cols[1]:
        if st.button(label, key=f"next_step_{tab_id}", width="stretch"):
            st.session_state["requested_tab"] = tab_id
            st.rerun()


def require_prerequisite(key: str, message: str) -> bool:
    """Check that a session state key exists, show info message if not."""
    if st.session_state.get(key) is None:
        st.info(message)
        return False
    return True


def record_empty_step(
    df_key: str,
    params_key: str,
    build_empty: Callable[[], pd.DataFrame],
    *,
    failed_key: str | None,
    params: dict | None = None,
) -> None:
    """Record the empty result of a step that has nothing to process.

    *build_empty* returns the step's empty result with its usual columns (the
    pipelines' ``empty_*_results``). Once recorded, later tabs and Results see
    a finished, empty step rather than one not run yet. *params* are the
    settings recorded for the step; None (the default) shows it as skipped in
    the dashboard. Recorded once; a change upstream resets it.

    If queries of the previous step failed (*failed_key*), the empty input may
    come from those failures: nothing is recorded, and the user is asked to
    re-run that step. Pass None when the previous list is not empty.
    """
    failed = (st.session_state.get(failed_key) or []) if failed_key else []
    if failed:
        query_word = pluralize(len(failed), "query")
        st.warning(
            f"⚠️ {len(failed)} {query_word} of the previous step failed, so its "
            "list may be empty only because of them. Re-run that step to retry "
            "them before going on."
        )
        return
    if st.session_state.get(df_key) is not None:
        return
    st.session_state[df_key] = build_empty()
    st.session_state[params_key] = params
    reset_state_after(df_key)


def failed_items(df) -> list[str]:
    """Items whose query failed in the call that returned *df*.

    Read from ``df.attrs["failed_items"]``, set by ``save_to_db``. Empty when
    *df* is missing or did not come from a cached step. An item reported as a
    full cache key (a tuple) is shown by its first value, e.g. the sequence id.
    """
    if df is None:
        return []
    names = (
        str(x[0] if isinstance(x, tuple) else x)
        for x in df.attrs.get("failed_items", [])
    )
    return list(dict.fromkeys(names))


def show_failed_items_banner(
    failed,
    item: str = "item",
    items: str = "items",
    source: str = "",
    gives_up: bool = False,
) -> None:
    """Warn about items whose query failed in the last run.

    Failed items are not cached (see ``failure_sink`` in ``save_to_db``), so a
    plain re-run queries them again while everything else is reused.
    *gives_up* says the source also fails a query it gave up waiting for (NCBI
    BLAST jobs stuck in NCBI's queue).
    """
    failed = list(failed or [])
    if not failed:
        return
    shown = ", ".join(failed[:10])
    if len(failed) > 10:
        shown += f", ... (+{len(failed) - 10} more)"
    request = f"the {source} request" if source else "the request"
    label = pluralize(len(failed), item, items)
    outcome = (
        "failed after retries or was given up" if gives_up else "failed after retries"
    )
    st.warning(
        f"⚠️ {len(failed)} {label} could not be queried: {request} {outcome}, "
        "so they were not saved. Run the step again to retry only these; the "
        "rest is reused.\n\n"
        f"{shown}"
    )


# Widget key of the selected sequence, shared by every tab's selector.
SELECTED_SEQUENCE_KEY = "selected_sequence"


def sequence_selector(
    show_sequence: bool = True,
    labels: dict[str, str] | None = None,
) -> str:
    """Select a sequence, searchable by ID, kept from one tab to the next.
    *labels* gives the text shown for each sequence (its ID by default).
    """
    sequences = st.session_state.sequences_id
    persist_value(SELECTED_SEQUENCE_KEY, sequences[0])
    # The labels are computed before the selectbox, which may call format_func
    # outside the script run (e.g. in tests), where session state is empty.
    labels = labels or {}
    selected_sequence = st.selectbox(
        "Select a sequence (type to search):",
        sequences,
        key=SELECTED_SEQUENCE_KEY,
        format_func=lambda seq: labels.get(seq, str(seq)),
        filter_mode="contains",
    )
    save_widget(SELECTED_SEQUENCE_KEY)
    sequences_df = st.session_state.sequences_df
    dna_sequence = sequences_df.loc[
        sequences_df["seq_id"] == selected_sequence, "dna_sequence"
    ].iloc[0]

    if show_sequence:
        with st.expander(
            f"🧬 View Raw DNA Sequence ({len(dna_sequence)} bp)", expanded=False
        ):
            st.code(dna_sequence, language="text", wrap_lines=True)

    return selected_sequence


class StreamlitLogSink:
    """Loguru sink with thread-scoped filtering for Streamlit Cloud.

    On Cloud, concurrent sessions share one process and one global
    ``loguru.logger``.  This sink only accepts log messages from threads
    registered via ``allow_current_thread()`` or ``wrap()``, preventing
    cross-session log leakage. ``wrap()`` also tags the call's log context
    with this sink, so threads the call spawns in a copy of its context (the
    BLAST batch workers, see ``stop.run_threads``) are accepted too.
    """

    def __init__(self, container=None, prefix=""):
        self.container = container
        self.prefix = prefix
        self.log_queue = queue.Queue()
        self._allowed_threads: set[int] = {threading.get_ident()}
        # Unique per sink (id() can be reused once a sink is freed).
        self._tag = uuid.uuid4().hex

        # Initialize session state logs
        self.lines: list[str] = []
        st.session_state[f"{self.prefix}logs"] = self.lines

        self._taken_over = False
        self._handler_id = logger.add(
            self.write, filter=self.thread_filter, level=get_log_level()
        )

    def close(self):
        """Stop receiving log messages (call when the job is over)."""
        if self._handler_id is not None:
            logger.remove(self._handler_id)
            self._handler_id = None

    def take_over(self, old: "StreamlitLogSink"):
        """Continue the log of a job started by an interrupted script run.

        The log becomes the job's log so far (the lines this sink got before
        repeat steps the job already logged), then this sink receives the job's
        next lines in place of *old*, which is closed.
        """
        # Accept the job's records before closing old, so none falls between.
        self._tag = old._tag
        self._allowed_threads |= old._allowed_threads
        # The abandoned run may still be polling old: stop it draining.
        old._taken_over = True
        old.close()
        self.lines[:] = old.lines + old._drain()
        for line in self.lines:
            self._show(line)

    def allow_current_thread(self):
        """Register the calling thread so this sink accepts its log messages."""
        self._allowed_threads.add(threading.get_ident())

    def thread_filter(self, record):
        """Loguru filter: accept only 'trident' messages from registered
        threads, or tagged with this sink by ``wrap()``."""
        if not record["name"].startswith("trident") or not stop.log_filter(record):
            return False
        if record["extra"].get("log_sink") == self._tag:
            return True
        return record["thread"].id in self._allowed_threads

    def wrap(self, fn, before=None):
        """Return a wrapper that registers its thread before calling *fn*.

        Use with ``executor.submit(logsink.wrap(fn), ...)`` so the worker
        thread's logs are captured by this sink.  *before* is an optional
        zero-arg callable run after registration (e.g. set NCBI email).
        """

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            self.allow_current_thread()
            if before:
                before()
            with logger.contextualize(log_sink=self._tag):
                return fn(*args, **kwargs)

        return wrapper

    _LEVEL_ICONS = {"DEBUG": "·", "INFO": "🔹", "WARNING": "🔶", "ERROR": "🔴"}

    def write(self, message):
        """Called by loguru from any thread - adds to queue"""
        text = message.record["message"]
        dt = message.record["time"].strftime("%H:%M:%S")
        icon = self._LEVEL_ICONS.get(message.record["level"].name, "🔹")
        line = f"{icon} [{dt}] {text}"

        # Put in queue (thread-safe)
        self.log_queue.put(line)

    def _show(self, line) -> bool:
        try:
            self.container.write(line)
            return True
        except Exception:
            return False

    def _drain(self) -> list[str]:
        """Return and remove the queued lines."""
        lines = []
        while True:
            try:
                lines.append(self.log_queue.get_nowait())
            except queue.Empty:
                return lines

    def flush_to_ui(self):
        """Called from main thread to display queued logs"""
        if self._taken_over:
            return 0
        lines = self._drain()
        self.lines.extend(lines)
        return sum(self._show(line) for line in lines)


# Pandas' Styler cap (262144 cells by default, above which st.dataframe
# raises), raised by apply_styler_limit, and the size above which
# styled_or_plain shows a plain table: one value, so they cannot drift.
STYLER_MAX_CELLS = 2_000_000


@functools.cache
def logo_png(width: int = 480) -> bytes:
    """The app logo as a PNG at most *width* pixels wide, made once: given the
    1920 px file, Streamlit would decode and shrink it on every run."""
    with Image.open(Path(__file__).with_name("logo.png")) as image:
        image.thumbnail((width, width))
        png = io.BytesIO()
        image.save(png, format="PNG")
    return png.getvalue()


def apply_styler_limit() -> None:
    """Raise pandas' Styler render cap to STYLER_MAX_CELLS (call once at startup)."""
    pd.set_option("styler.render.max_elements", STYLER_MAX_CELLS)


def styled_or_plain(df, *row_style_fns, max_cells: int = STYLER_MAX_CELLS):
    """Row-highlighted Styler, or the plain DataFrame when too large to style.

    Colors are kept for every realistic table (the cap is raised to match). The
    fallback only triggers on a pathologically huge single table, where it
    returns the unstyled frame instead of letting st.dataframe crash. st.dataframe
    accepts either a Styler or a DataFrame, so callers pass the result straight
    through.

    Args:
        df: The DataFrame to render.
        *row_style_fns: Zero or more row-wise (axis=1) style functions, applied
            in order (mirrors a chained ``.style.apply(...).apply(...)``).
        max_cells: Cell-count ceiling above which styling is skipped.

    Returns:
        A pandas Styler when df.size <= max_cells, else df unchanged.
    """
    if df.size > max_cells:
        return df
    styler = df.style
    for fn in row_style_fns:
        styler = styler.apply(fn, axis=1)
    return styler


@contextlib.contextmanager
def lazy_expander(label: str, key: str, icon: str | None = None):
    """A closed expander that yields whether it is open, for content drawn only
    then (``if is_open:``). Streamlit otherwise runs the content of a closed
    expander on every rerun, and a large table costs seconds to draw and send.
    Opening or closing it reruns the app."""
    section = st.expander(label, key=key, icon=icon, on_change="rerun")
    with section:
        yield section.open


def run_with_progress(
    fn,
    *args,
    logsink: "StreamlitLogSink | None" = None,
    progress: dict | None = None,
    on_poll: Callable[[], None] | None = None,
    poll_interval: float = 0.5,
    **kwargs,
):
    """Run ``fn(*args, **kwargs)`` in a worker thread, polling until it finishes.

    Used by the step tabs to keep the UI responsive during a long search/merge:
    while the future is pending, ``on_poll`` is called each tick (to refresh the
    progress bar) and ``logsink`` is flushed to the UI. The sink is flushed once
    more after completion so the final log lines appear.

    Args:
        fn: The callable to run in the background (e.g. a ``logsink.wrap(...)``).
        *args, **kwargs: Passed straight through to ``fn``.
        logsink: Log sink to flush each tick and once at the end, if provided.
        progress: The dict the job updates and ``on_poll`` reads, if any.
        on_poll: Zero-arg callback invoked each tick (typically updates the
            progress bar from a shared ``progress_info`` dict).
        poll_interval: Longest wait between two ticks, in seconds.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        StepStopped: the step was stopped (``_stop_button``). The job's
            finished items are cached, but its result is dropped.
    """
    # A rerun requested while the job runs (e.g. a widget event) abandons this
    # script run but not the job's thread. The next run calls here again with
    # the step's run flag still set: wait on the same job instead of starting
    # a second one, and take over its log and progress. A step runs one job at
    # a time: a dropped job (its step reset) is waited for first. It is only
    # stopped by the user (the stop button, which covers it too).
    jobs = st.session_state.setdefault(RUNNING_JOBS, {})
    step = st.session_state.get(_JOB_STEP)
    key = (step, fn.__name__)
    job = jobs.get(key)
    if job is not None and job["dropped"]:
        if not job["future"].done():
            waiting = st.empty()
            # Polled with a Streamlit call each tick, so a click on the stop
            # button (a rerun) takes effect instead of waiting for the job.
            while not job["future"].done():
                waiting.info("Finishing the previous search first...")
                concurrent.futures.wait([job["future"]], timeout=poll_interval)
            waiting.empty()
        job = None
    if job is None:
        if step in st.session_state.get(STOPPED_STEPS, ()):
            # Stopped between two jobs of the step: start no further one.
            if logsink is not None:
                logsink.close()
            raise stop.StepStopped()
        event = threading.Event()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_stoppable, event, fn, args, kwargs)
        executor.shutdown(wait=False)
        job = jobs[key] = {
            "future": future,
            "progress": progress,
            "dropped": False,
            "stop": event,
        }
    elif logsink is not None and job["logsink"] is not None:
        logsink.take_over(job["logsink"])
    job["logsink"] = logsink

    future = job["future"]
    while not future.done():
        if progress is not None and job["progress"] is not progress:
            progress.update(job["progress"])
        if on_poll is not None:
            on_poll()
        if logsink is not None:
            logsink.flush_to_ui()
        # Returns as soon as the job ends: a step whose results are all saved
        # does not wait a whole tick.
        concurrent.futures.wait([future], timeout=poll_interval)
    # A finished job stays until its step is run again or reset (drop_jobs):
    # the abandoned run may still be polling it, and either run may finish first.
    error = future.exception()
    stopped = job["stop"].is_set()
    if error is None and not stopped:
        if logsink is not None:
            logsink.flush_to_ui()
        return future.result()
    # Not kept (a stopped job's result lacks the items it skipped): the next
    # run of the step tries again.
    if jobs.get(key) is job:
        del jobs[key]
    if logsink is not None:
        logsink.flush_to_ui()
        logsink.close()
    if not stopped:
        raise error
    if error is not None and not isinstance(error, stop.StepStopped):
        # Most likely caused by the stop (its items all failed), but kept in
        # the console and log file (this thread is not stopped).
        logger.opt(exception=error).warning("The stopped search ended with an error")
    raise stop.StepStopped() from error


def _run_stoppable(event: threading.Event, fn, args, kwargs):
    """Run a job with *event* as its stop signal (see ``trident.core.stop``)."""
    with stop.stoppable(event):
        return fn(*args, **kwargs)


def _stop_button(flag_key: str, btn_key: str):
    """Stop button of a running step. Its running jobs send no further request
    (the next one raises StepStopped, see ``trident.core.stop``); a request
    already sent finishes, and a BLAST job being polled is dropped."""
    stopped = st.session_state.setdefault(STOPPED_STEPS, set())
    # Not a bare "Stop": Streamlit's own Stop (top right) ends the script run,
    # not the job, which goes on and is re-attached to by the next run.
    if flag_key not in stopped and st.button(
        "Stop the search",
        key=f"{btn_key}_stop",
        icon=":material/stop_circle:",
        help=(
            "Stops the search now: queries still running are dropped. Results "
            "already received are kept: run the step again to continue where "
            "it stopped."
        ),
    ):
        stopped.add(flag_key)
        # A finished job keeps its result (the post-processing may still use it).
        for (step, _), job in st.session_state.get(RUNNING_JOBS, {}).items():
            if step == flag_key and not job["future"].done():
                job["stop"].set()
    if flag_key in stopped:
        st.info("Stopping the search...")


# Session key of the jobs of each step, by (step run flag, job function).
RUNNING_JOBS = "_running_jobs"
# Session key of the step whose job is being run (its run flag).
_JOB_STEP = "_job_step"
# Session key of each step's run settings, fixed at the click (by run flag).
STEP_RUNS = "_step_runs"
# Session key of the steps stopped by the user (run flags), until run again.
STOPPED_STEPS = "_stopped_steps"


def drop_jobs(flag_key: str | None = None):
    """Forget the jobs of one step (by its run flag), or of all steps, and
    whether they were stopped.

    The next run of the step starts a new job instead of re-attaching. A job
    still running is kept, marked dropped, so the next run waits for it to end
    rather than run beside it (or the user stops it).
    """
    jobs = st.session_state.get(RUNNING_JOBS, {})
    for key in [k for k in jobs if flag_key is None or k[0] == flag_key]:
        job = jobs[key]
        if job["logsink"] is not None:
            job["logsink"].close()
        if job["future"].done():
            del jobs[key]
        else:
            job["dropped"] = True
    stopped = st.session_state.get(STOPPED_STEPS, set())
    if flag_key is None:
        stopped.clear()
    else:
        stopped.discard(flag_key)


def highlight_low_identity(row):
    """Apply highlight style for valid hits with low identity."""
    style = (
        "background-color: #ffc7ce; color: black;"
        if row["low_identity_warning"]
        else ""
    )
    return [style] * len(row)


# What the hypo_check_hit flag means (see add_check_hit), and the legend of the
# rows highlight_check_hit colours.
CHECK_HIT_NOTE = "HYPO species whose marker the HYPO check found in GenBank"
CHECK_HIT_LEGEND = f":orange[■] {CHECK_HIT_NOTE}"


def highlight_check_hit(row):
    """Apply highlight style for HYPO species with a check hit (see add_check_hit)."""
    style = "background-color: #ffe0b2;" if row.get("hypo_check_hit", False) else ""
    return [style] * len(row)


def mol_query_cover():
    """The MOL filter's query cover, which the best HYPO check hit and the
    best identity of a sequence also require (a partial hit says less)."""
    stored = (st.session_state.get("mol_params") or {}).get("ncbi_query_cover")
    return coerce_stored_value(stored, float(NCBI_QUERY_COVER_DEFAULT))


def mol_filter_settings():
    """The MOL filter settings of the current results, as filter arguments."""
    stored = st.session_state.get("mol_params") or {}
    return {
        "query_cover": mol_query_cover(),
        "gap_size": coerce_stored_value(
            stored.get("ncbi_gap_size"), float(NCBI_GAP_SIZE_DEFAULT)
        ),
        "method": stored.get("ncbi_method") or NCBI_METHOD_DEFAULT,
        "gap_min_top": coerce_stored_value(
            stored.get("ncbi_gap_min_top"), float(NCBI_GAP_MIN_TOP_DEFAULT)
        ),
    }


def highlight_empty_sequence(row):
    """Apply highlight style for sequences with no species assigned."""

    is_empty = (
        pd.isna(row["scientificName"]) or str(row["scientificName"]).strip() == ""
    )

    style = "background-color: #c0c0c0; color: black;" if is_empty else ""
    return [style] * len(row)


def same_species_as(names: pd.Series, groups: pd.DataFrame | None = None) -> pd.Series:
    """The species each of *names* is possibly the same as, indexed by name;
    only the linked ones. *groups*: same_species_groups of *names*, when
    already computed."""
    if groups is None:
        groups = same_species_groups(names, st.session_state.name_table)
    return groups.set_index("scientificName")["same_species_as"]


def with_same_species(df: pd.DataFrame) -> pd.DataFrame:
    """df with same_species_as (see same_species_as); empty for other rows."""
    names = df["scientificName"]
    return df.assign(same_species_as=names.map(same_species_as(names)))


def display_same_species(df: pd.DataFrame) -> pd.DataFrame:
    """List the species of df another source counts as one species; returns
    their same_species_groups."""
    groups = same_species_groups(df["scientificName"], st.session_state.name_table)
    if groups.empty:
        return groups
    n_groups = groups["group"].nunique()
    with st.expander(
        f"⚠️ {n_groups} species possibly listed under several names",
        expanded=False,
    ):
        st.caption(
            "Another source counts these names as one species. WoRMS may keep "
            "one species under two names, or the sources may disagree. The "
            "names are kept and counted apart: check them."
        )
        st.dataframe(
            groups,
            width="stretch",
            hide_index=True,
            column_order=["scientificName", "same_species_as"],
            column_config=column_config(["scientificName", "same_species_as"]),
        )
    return groups


def highlight_in_mol(row):
    """Apply highlight style for MOL species."""
    style = "background-color: #c6efce; color: black;" if row["in_mol"] else ""
    return [style] * len(row)


def get_recent_param_sets(db_path, table_name, columns, limit=5, only=None):
    """Return distinct recent parameter combinations from cache history.

    *limit* None returns them all. *only*: values the rows must hold (e.g. the
    BLAST database), so a preset never names results the step cannot reuse.
    """
    only = only or {}
    if not os.path.exists(db_path):
        return []
    try:
        with get_connection(db_path) as conn:
            # A table without one of these columns predates that setting: no preset
            info = conn.execute(f"PRAGMA table_info({table_name})")
            stored = {row[1] for row in info}
            if stored and not set(columns) | set(only) <= stored:
                logger.debug(f"No presets from {table_name}: written before {columns}")
                return []
            cols_sql = ", ".join(columns)
            where_clauses = " AND ".join(
                f"{col} IS NOT NULL AND {col} <> 'None' AND {col} <> ''"
                for col in columns
            )
            for col in only:
                where_clauses += f" AND {col} = ?"
            query = f"""
                SELECT {cols_sql} FROM {table_name}
                WHERE {where_clauses}
                GROUP BY {cols_sql} ORDER BY MAX(run_id) DESC LIMIT ?
            """
            rows = conn.execute(
                query, (*only.values(), -1 if limit is None else limit)
            ).fetchall()
        return [dict(zip(columns, row)) for row in rows]
    except sqlite3.OperationalError as exc:
        # A missing table is normal before its step has run for the first time,
        # so it stays at debug; anything else (unknown column, corrupt file)
        # means the query itself is wrong and should be visible.
        level = "debug" if "no such table" in str(exc) else "warning"
        getattr(logger, level)(f"No presets from {table_name}: {exc}")
        return []
    except Exception as exc:
        logger.warning(f"No presets from {table_name}: {exc}")
        return []


# Prefix of the session keys recording the preset each selector last applied.
PRESET_MARKER = "_prev_"


def prefill_settings(values: dict):
    """Pre-fill widgets of the later steps (widget key -> value).

    Call after reset_for_new_analysis. A preset selector covering these
    widgets keeps the values on its first render instead of applying the
    latest history preset; picking a preset in its menu applies it as usual.
    """
    st.session_state._widget_overrides.update(values)


def param_preset_selector(presets, format_fn, key):
    """Selectbox of previous parameter sets. Updates widget overrides on change.

    Widget keys must match the DB column names in *presets*.
    """
    if not presets:
        return
    cols = [k for k in presets[0] if k != "run_id"]

    if len(presets) == 1:
        # Single preset: apply silently, no UI
        preset = presets[0]
    else:
        labels = [format_fn(p) for p in presets]

        # Match current overrides against presets to find active index
        current = {
            col: str(st.session_state._widget_overrides.get(col, "")) for col in cols
        }
        selected_idx = 0
        for i, p in enumerate(presets):
            if all(str(p.get(k, "")) == v for k, v in current.items()):
                selected_idx = i
                break

        choice = st.selectbox(
            "Previously used settings",
            labels,
            index=selected_idx,
            key=key,
            help="Pre-fill the fields below with a previously used parameter set.",
        )
        preset = presets[labels.index(choice)]

    # Only apply preset when the selection actually changed
    prev_key = f"{PRESET_MARKER}{key}"
    preset_id = tuple(str(preset.get(c, "")) for c in cols)
    overrides = st.session_state._widget_overrides
    if prev_key not in st.session_state and any(c in overrides for c in cols):
        # First render since the analysis was loaded, and the fields already
        # hold values (pre-filled, or typed before any history existed): keep
        # them, and record the preset as seen.
        st.session_state[prev_key] = preset_id
    elif st.session_state.get(prev_key) != preset_id:
        st.session_state[prev_key] = preset_id
        for col in cols:
            st.session_state._widget_overrides[col] = preset[col]
            # Delete widget key so persist_value re-seeds from override
            if col in st.session_state:
                del st.session_state[col]
