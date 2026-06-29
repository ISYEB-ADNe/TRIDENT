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
from dataclasses import dataclass, field, fields, asdict
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st


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
    tax_df: "pd.DataFrame | None" = None

    gbif_logs: list[str] = field(default_factory=list)
    gbif_search_flag: bool = False
    gbif_search_params: dict[str, Any] = field(default_factory=dict)
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
    hypo_merge_df: "pd.DataFrame | None" = None
    hypo_filter_flag: bool = False
    hypo_filter_params: dict[str, Any] = field(default_factory=dict)
    hypo_filter_df: "pd.DataFrame | None" = None
    hypo_check_logs: list[str] = field(default_factory=list)
    hypo_check_flag: bool = False
    hypo_check_params: dict[str, Any] = field(default_factory=dict)
    hypo_check_df: "pd.DataFrame | None" = None
    hypo_check_summary_df: "pd.DataFrame | None" = None
    hypo_params: dict[str, Any] = field(default_factory=dict)
    hypo_df: "pd.DataFrame | None" = None

    excluded_results: list[str] = field(default_factory=list)

    # reset=False: not a pipeline output, so reset_state_after leaves it intact
    # (it is the cross-tab widget persistence store).
    _widget_overrides: dict[str, Any] = field(
        default_factory=dict, metadata={"reset": False}
    )


def init_session_state():
    defaults = asdict(AppState())
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def persist_value(key: str, default):
    """Pre-seed a widget key from the persistent override store.

    Call BEFORE creating the widget. Only acts when the key is absent
    from session_state (e.g. after a tab switch removed it). When the
    widget is already on screen, the key exists and this is a no-op.
    """
    if key not in st.session_state:
        overrides = st.session_state._widget_overrides
        if key in overrides:
            value = overrides[key]
            # Cast override to match the default's type (DB stores strings)
            if default is not None and value is not None:
                try:
                    value = type(default)(value)
                except (ValueError, TypeError):
                    pass
            st.session_state[key] = value
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
) -> str:
    """Handle execution + button for a pipeline step. Returns mode."""
    mode = get_analysis_status(df_key, params_key, current_params, keys=compare_keys)

    if st.session_state.get(flag_key):
        cache_mode = st.session_state.get(f"{btn_key}_cache_mode", "Use cached results")

        # Force re-run when settings changed or user explicitly ignores cache.
        # IDENTICAL re-runs reuse cached items by default — only failed/missing
        # items are re-executed (important for PartialCache steps).
        force = mode == "CHANGED" or "Erase" in cache_mode
        retry = "Retry" in cache_mode
        if threaded:
            progress_bar = st.progress(0, text="Initializing...")
            with st.status(status_label, expanded=False) as status:
                job_fn(
                    force_rerun=force,
                    retry_empty=retry,
                    status=status,
                    progress_bar=progress_bar,
                    **(job_kwargs or {}),
                )
        else:
            with st.status(status_label, expanded=False):
                job_fn(force_rerun=force, retry_empty=retry, **(job_kwargs or {}))

        st.session_state[flag_key] = False
        st.session_state["requested_tab"] = tab_id
        st.rerun()

    if before_button:
        before_button()

    clicked, _ = display_run_button(
        mode, new_string, btn_key=btn_key, logs_prefix=logs_prefix
    )
    if clicked:
        st.session_state[flag_key] = True
        st.rerun()

    return mode


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


def show_missing_results_banner(
    input_values, result_df, key_col: str, item_label: str = "items"
) -> None:
    """Warn when items from THIS run's input are absent from its result.

    Compares the current input against the current result (not the whole cache
    history). A missing item means the source returned no records this run,
    either a genuine empty or a failed query; "Retry empty results" re-queries.
    """
    inputs = set(input_values or [])
    if not inputs:
        return
    found = set()
    if result_df is not None and not result_df.empty and key_col in result_df.columns:
        found = set(result_df[key_col].dropna().unique())
    n_missing = len(inputs - found)
    if n_missing:
        st.warning(
            f"⚠️ {n_missing}/{len(inputs)} {item_label} returned no records. If you "
            "expected data (e.g. after a failed query), set Cache to "
            "**Retry empty results** and re-run."
        )


def sequence_selector(
    show_sequence: bool = True,
    format_fn: Callable[[str], str] | None = None,
) -> str:
    sequences = st.session_state.sequences_id
    if format_fn:
        options = {seq: format_fn(seq) for seq in sequences}
        selected_sequence = st.selectbox(
            "Select a sequence:",
            sequences,
            index=0,
            format_func=lambda s: options[s],
        )
    else:
        selected_sequence = st.selectbox(
            "Select a sequence:",
            sequences,
            index=0,
        )
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
    cross-session log leakage.
    """

    def __init__(self, container=None, prefix=""):
        self.container = container
        self.prefix = prefix
        self.log_queue = queue.Queue()
        self._allowed_threads: set[int] = {threading.get_ident()}

        # Initialize session state logs
        log_key = f"{self.prefix}logs"
        st.session_state[log_key] = []

    def allow_current_thread(self):
        """Register the calling thread so this sink accepts its log messages."""
        self._allowed_threads.add(threading.get_ident())

    def thread_filter(self, record):
        """Loguru filter: accept only 'trident' messages from registered threads."""
        if not record["name"].startswith("trident"):
            return False
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

    def flush_to_ui(self):
        """Called from main thread to display queued logs"""
        log_key = f"{self.prefix}logs"
        displayed_count = 0

        if log_key not in st.session_state or st.session_state[log_key] is None:
            st.session_state[log_key] = []

        # Process all queued logs
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break

            st.session_state[log_key].append(line)
            try:
                self.container.write(line)
                displayed_count += 1
            except Exception:
                pass

        return displayed_count


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
        on_poll: Zero-arg callback invoked each tick (typically updates the
            progress bar from a shared ``progress_info`` dict).
        poll_interval: Seconds to sleep between ticks.

    Returns:
        Whatever ``fn`` returns.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        while not future.done():
            if on_poll is not None:
                on_poll()
            if logsink is not None:
                logsink.flush_to_ui()
            time.sleep(poll_interval)
        result = future.result()
        if logsink is not None:
            logsink.flush_to_ui()
    return result


def highlight_low_identity(row):
    """Apply highlight style for valid hits with low identity."""
    style = (
        "background-color: #ffc7ce; color: black;"
        if row["low_identity_warning"]
        else ""
    )
    return [style] * len(row)


def highlight_below_mol(row):
    """Apply highlight style for species that would not have passed the MOL filter."""
    style = "background-color: #ffe0b2;" if row.get("below_mol", False) else ""
    return [style] * len(row)


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
    """Return distinct recent parameter combinations from cache history."""
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
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(zip(columns, row)) for row in rows]
    except Exception:
        return []


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
    prev_key = f"_prev_{key}"
    preset_id = tuple(str(preset.get(c, "")) for c in cols)
    if st.session_state.get(prev_key) != preset_id:
        st.session_state[prev_key] = preset_id
        for col in cols:
            st.session_state._widget_overrides[col] = preset[col]
            # Delete widget key so persist_value re-seeds from override
            if col in st.session_state:
                del st.session_state[col]
