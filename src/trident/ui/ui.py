"""
Streamlit UI components and helpers.
Contains all Streamlit-specific UI functions that are reusable across the app.
"""

import concurrent.futures
import functools
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, fields, asdict
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st
from loguru import logger

from trident.clients.ncbi import BLAST_SLOW_THREADS
from trident.core.utils import pluralize
from trident.core import stop
from trident.logging import get_log_level
from trident.ui.defaults import NCBI_QUERY_COVER_DEFAULT


# Suppress Streamlit's thread context warnings (Streamlit uses standard logging internally)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)

# --------------------------------------------------------------------
# SHARED COLUMN CONFIGS
# -------------------------------------------------------------------

COL = {
    "seq_id": st.column_config.TextColumn(
        "Sequence ID", help="Unique identifier from the FASTA header."
    ),
    "dna_sequence": st.column_config.TextColumn(
        "DNA Sequence", help="Raw genetic sequence (A, T, C, G)."
    ),
    "seq_length": st.column_config.NumberColumn(
        "Nucleotides", format="%d bp", help="Number of base pairs in the sequence."
    ),
    "scientificName": st.column_config.TextColumn(
        "Scientific Name",
    ),
    "verbatimIdentification": st.column_config.TextColumn(
        "NCBI synonym",
        help="NCBI/BLAST name(s) for this species that differ from the WoRMS-accepted name (synonyms/misspellings). Blank when NCBI already used the accepted name.",
    ),
    "identity_percentage": st.column_config.NumberColumn(
        "Identity (%)",
        format="%.2f",
        help="Percentage of matching bases in the alignment.",
    ),
    "query_cover": st.column_config.NumberColumn(
        "Query Coverage (%)",
        format="%.2f",
        help="Percentage of the query sequence covered by the hit.",
    ),
    "hit_url": st.column_config.LinkColumn(
        "NCBI Link",
        display_text="🔗 View",
    ),
    "ncbi_top_hit_url": st.column_config.LinkColumn(
        "NCBI Link",
        display_text="🔗 View",
    ),
    "hit_def": st.column_config.TextColumn(
        "Hit Definition",
    ),
    "hit_count": st.column_config.NumberColumn(
        "Hit Count", format="%d", help="Total number of hits found for this species."
    ),
    "top_identity": st.column_config.NumberColumn(
        "Top Identity (%)",
        format="%.2f",
        help="Highest identity percentage found for this sequence.",
    ),
    "hits_count": st.column_config.NumberColumn(
        "Hits", format="%d", help="Total number of hits for this sequence."
    ),
    "species_count": st.column_config.NumberColumn(
        "Species", format="%d", help="Total number of unique species for this sequence."
    ),
    "filter_method": st.column_config.TextColumn("Filter Method"),
    "genus": st.column_config.TextColumn("Genus"),
    "specificEpithet": st.column_config.TextColumn("Epithet"),
    "align_length": st.column_config.NumberColumn(
        "Alignment Length",
        format="%d",
        help="Length of the aligned region in base pairs.",
    ),
    "identities": st.column_config.NumberColumn(
        "Identities", format="%d", help="Number of identical bases in the alignment."
    ),
    "gaps": st.column_config.NumberColumn(
        "Gaps", format="%d", help="Number of gaps introduced in the alignment."
    ),
    "query_start": st.column_config.NumberColumn(
        "Query Start",
        format="%d",
        help="Start position of the alignment on the query sequence.",
    ),
    "query_end": st.column_config.NumberColumn(
        "Query End",
        format="%d",
        help="End position of the alignment on the query sequence.",
    ),
    "mol_top_identity_percentage": st.column_config.NumberColumn(
        "MOL Identity (%)",
        format="%.2f",
        help="Identity percentage of the best NCBI hit for this species in the MOL step.",
    ),
    "mol_top_query_cover": st.column_config.NumberColumn(
        "MOL Query Coverage (%)",
        format="%.2f",
        help="Query coverage of the best NCBI hit for this species in the MOL step.",
    ),
    "taxonURL": st.column_config.LinkColumn("WoRMS Link", display_text="🔗 View"),
    "scientificNameAuthorship": st.column_config.TextColumn("Authorship"),
    "family": st.column_config.TextColumn("Family"),
    "kingdom": st.column_config.TextColumn("Kingdom"),
    "phylum": st.column_config.TextColumn("Phylum"),
    "class": st.column_config.TextColumn("Class"),
    "order": st.column_config.TextColumn("Order"),
    "taxonRank": st.column_config.TextColumn(
        "Rank", help="Taxonomic rank of the species."
    ),
    "taxonID": st.column_config.NumberColumn(
        "Taxon ID",
        format="%d",
        help="Unique identifier in the source taxonomy database.",
    ),
    "taxonID_db": st.column_config.TextColumn(
        "Taxon DB", help="Source taxonomy database for the taxon ID."
    ),
    "bold_seq_url": st.column_config.LinkColumn("BOLD Link", display_text="🔗 View"),
    "bold_dna_sequence": st.column_config.TextColumn(
        "BOLD DNA Sequence", width="large"
    ),
    "gbif_taxonURL": st.column_config.LinkColumn("GBIF Link", display_text="🔗 View"),
    "occurrences": st.column_config.NumberColumn(
        "Occurrences",
        format="%d",
        help="Number of GBIF occurrence records for this species in the search area.",
    ),
    "gbif_occurrences": st.column_config.NumberColumn(
        "GBIF Occurrences",
        format="%d",
        help="Number of GBIF occurrence records for this species in the selected filtering area.",
    ),
    "gbif_extent": st.column_config.TextColumn(
        "Extent", help="Geographic search extent used for this record."
    ),
    "total_records": st.column_config.NumberColumn(
        "BOLD Records", format="%d", help="Total BOLD sequences found."
    ),
    "species_queried": st.column_config.NumberColumn(
        "Species Queried",
        format="%d",
        help="Total species queried in BOLD for this sequence.",
    ),
    "species_found": st.column_config.NumberColumn(
        "Species Found", format="%d", help="Species with at least one BOLD record."
    ),
    "species_missing": st.column_config.NumberColumn(
        "Species Missing", format="%d", help="Species queried but not found in BOLD."
    ),
}

# Help text for the NCBI BLAST settings shared by the MOL and HYPO tabs.
BLAST_HELP = {
    "max_hits": (
        "Maximum number of NCBI database sequences returned per query sequence; "
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
    fasta_uploader_key: int = 0
    analysis_name: str = ""
    db_path: str = ""
    sequences_id: list[str] | None = None
    sequences_df: "pd.DataFrame | None" = None

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
    tax_failed_names: list[str] = field(default_factory=list)
    tax_homonyms: "pd.DataFrame | None" = None
    tax_mol_rejected: "pd.DataFrame | None" = None
    tax_df: "pd.DataFrame | None" = None

    gbif_logs: list[str] = field(default_factory=list)
    gbif_search_flag: bool = False
    gbif_search_params: dict[str, Any] = field(default_factory=dict)
    gbif_search_failed: list[str] = field(default_factory=list)
    gbif_search_df: "pd.DataFrame | None" = None
    gbif_filter_flag: bool = False
    gbif_filter_df: "pd.DataFrame | None" = None
    geo_params: dict[str, Any] = field(default_factory=dict)
    geo_species: list[str] = field(default_factory=list)
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


def display_run_button(
    mode: str, new_string: str, btn_key: str, logs_prefix: str = ""
) -> tuple[bool, str]:
    """Renders the standardized status message and button.

    Returns:
        Tuple of (clicked, cache_mode) where cache_mode is the selected cache
        option string ("Use cached results" / "Retry empty results" /
        "Erase cached results").
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
            with st.expander(
                "✅ Analysis Run Complete, View Execution Logs", expanded=False
            ):
                if st.session_state.get(f"{logs_prefix}logs"):
                    st.text("\n".join(st.session_state[f"{logs_prefix}logs"]))
                else:
                    st.info("No logs available for this session.")
        # Define button for re-running even if identical
        btn_label = "🔄 Re-run with Same Settings"
        btn_type = "secondary"
        btn_help = "Re-run using cache — only retries failed or missing items"

    else:  # This handles "CHANGED"
        st.warning("⚠️ Settings have changed since the last run.")
        btn_label = "🚀 Run with New Settings"
        btn_type = "primary"
        btn_help = "Overwrite previous results with new parameters"

    c1, c2 = st.columns([2, 1])
    with c1:
        clicked = st.button(
            btn_label, type=btn_type, help=btn_help, width="stretch", key=btn_key
        )
    with c2:
        cache_mode = st.selectbox(
            "Cache",
            options=[
                "Use cached results",
                "Retry empty results",
                "Erase cached results",
            ],
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
    clicked, cache_mode = display_run_button(
        mode, new_string, btn_key=btn_key, logs_prefix=logs_prefix
    )
    if clicked:
        # Force a fresh run when the settings changed or the cache is erased.
        # IDENTICAL re-runs reuse cached items: only failed or missing items
        # are re-executed (important for PartialCache steps). Continuing a
        # stopped run with the same settings does not force: it already
        # applied them, and the items it cached are the ones to keep.
        runs = st.session_state.setdefault(STEP_RUNS, {})
        # The settings that make the results (as for "Settings have changed"):
        # others, such as threads or batch size, do not stop a run continuing.
        result_settings = (
            {k: current_params.get(k) for k in compare_keys}
            if compare_keys
            else current_params
        )
        resuming = (
            flag_key in st.session_state.get(STOPPED_STEPS, ())
            and runs[flag_key]["result_settings"] == result_settings
            and "Erase" not in cache_mode
        )
        runs[flag_key] = {
            "result_settings": result_settings,
            "force": (mode == "CHANGED" or "Erase" in cache_mode) and not resuming,
            "retry": "Retry" in cache_mode,
            "job_kwargs": job_kwargs or {},
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


def next_step_button(label: str, tab_id: str):
    """Render a right-aligned navigation button."""
    cols = st.columns([5, 1])
    with cols[1]:
        if st.button(label, type="secondary", width="stretch"):
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
        query_word = pluralize(len(failed), "query", "queries")
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
    plain re-run queries them again while everything else comes from the cache.
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
        "so they were not saved. Click **Re-run with Same Settings** to retry "
        "only these; everything else comes from the cache.\n\n"
        f"{shown}"
    )


def show_missing_results_banner(
    input_values, result_df, key_col: str, item_label: str = "items", failed=None
) -> None:
    """Warn when items from THIS run's input are absent from its result.

    Compares the current input against the current result (not the whole cache
    history). Items in *failed* are left out: they are reported by
    ``show_failed_items_banner``. The remaining missing items returned no
    records and are cached as empty; "Retry empty results" re-queries them
    (useful if the source changed, or for a database written before failures
    were tracked, where a failed query was cached as empty).
    """
    inputs = set(input_values or [])
    if not inputs:
        return
    found = set()
    if result_df is not None and not result_df.empty and key_col in result_df.columns:
        found = set(result_df[key_col].dropna().unique())
    n_missing = len(inputs - found - set(failed or []))
    if n_missing:
        st.info(
            f"{n_missing}/{len(inputs)} {item_label} returned no records (saved "
            "as empty). To query them again, set Cache to **Retry empty "
            "results** and re-run."
        )


# Widget key of the selected sequence, shared by every tab's selector.
SELECTED_SEQUENCE_KEY = "selected_sequence"


def sequence_selector(
    show_sequence: bool = True,
    format_fn: Callable[[str], str] | None = None,
) -> str:
    """Select a sequence, searchable by ID, kept from one tab to the next."""
    sequences = st.session_state.sequences_id
    persist_value(SELECTED_SEQUENCE_KEY, sequences[0])
    # Labels are computed here because the selectbox may call format_func
    # outside the script run (e.g. in tests), where session state is empty.
    labels = {seq: (format_fn or str)(seq) for seq in sequences}
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


# Streamlit's default Styler cap is 262144 cells; above it st.dataframe raises.
# We raise it (apply_styler_limit) so the colored full-list views render for
# realistic large runs, and use the SAME ceiling as the graceful-degrade point
# in styled_or_plain. One constant drives both, so the config and the fallback
# never drift: <= ceiling -> colored Styler renders; > ceiling -> plain frame
# (no crash) even on a pathologically huge run.
STYLER_MAX_CELLS = 2_000_000


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
        poll_interval: Seconds to sleep between ticks.

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
                time.sleep(poll_interval)
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
        time.sleep(poll_interval)
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


# What the check_hit flag means (see add_check_hit), and the legend of the
# rows highlight_check_hit colours.
CHECK_HIT_NOTE = "HYPO species whose marker the HYPO check found in NCBI"
CHECK_HIT_LEGEND = f":orange[■] {CHECK_HIT_NOTE}"


def highlight_check_hit(row):
    """Apply highlight style for HYPO species with a check hit (see add_check_hit)."""
    style = "background-color: #ffe0b2;" if row.get("check_hit", False) else ""
    return [style] * len(row)


def mol_query_cover():
    """The MOL filter's query cover, which the best HYPO check hit and the
    best identity of a sequence also require (a partial hit says less)."""
    stored = (st.session_state.get("mol_params") or {}).get("ncbi_query_cover")
    return coerce_stored_value(stored, float(NCBI_QUERY_COVER_DEFAULT))


def highlight_empty_sequence(row):
    """Apply highlight style for sequences with no species assigned."""

    is_empty = (
        pd.isna(row["scientificName"]) or str(row["scientificName"]).strip() == ""
    )

    style = "background-color: #c0c0c0; color: black;" if is_empty else ""
    return [style] * len(row)


def highlight_in_mol(row):
    """Apply highlight style for species found in NCBI."""
    style = "background-color: #c6efce; color: black;" if row["in_mol"] else ""
    return [style] * len(row)


def get_recent_param_sets(db_path, table_name, columns, limit=5):
    """Return distinct recent parameter combinations from cache history.

    *limit* None returns them all.
    """
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cols_sql = ", ".join(columns)
        where_clauses = " AND ".join(
            f"{col} IS NOT NULL AND {col} <> 'None' AND {col} <> ''" for col in columns
        )
        query = f"""
            SELECT {cols_sql} FROM {table_name}
            WHERE {where_clauses}
            GROUP BY {cols_sql} ORDER BY MAX(run_id) DESC LIMIT ?
        """
        cursor.execute(query, (-1 if limit is None else limit,))
        rows = cursor.fetchall()
        conn.close()
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
