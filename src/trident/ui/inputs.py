"""Start tab: FASTA upload, opening a saved analysis, and the sequences."""

import io
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

import trident.pipelines as pipe
from trident.clients.fasta import load_fasta, records_to_dataframe
from trident.core.config import (
    contact_email,
    is_streamlit_cloud,
    results_dir,
    source_root,
)
from trident.core.database import table_exists, check_schema_version, get_connection
from trident.core.utils import pluralize
from trident.ui.ui import (
    column_config,
    lazy_expander,
    next_step_button,
    prefill_settings,
    reset_for_new_analysis,
)

FASTA_EXTENSIONS = {"fasta", "fa", "txt", "fas"}
DB_EXTENSIONS = {"db", "sqlite", "sqlite3"}

# The example ships at the repository root (example/), next to src/, in every
# documented install (git clone or downloaded archive); None when installed.
EXAMPLE_FASTA = root / "example" / "example.fasta" if (root := source_root()) else None
# The example's sequences come from Svalbard: pre-fill the GEO area of the
# paper's worked example (kept on the counts within 250 km), or GBIF would be
# searched around 0°, 0°.
EXAMPLE_SETTINGS = {
    "gbif_latitude": "78.5",
    "gbif_longitude": "14.5",
    "gbif_extent": "250, 500",
    "gbif_filter_extent": 250,
}


def _db_path(base_name: str) -> str:
    """Return a DB path, scoped to the session on Streamlit Cloud."""
    if is_streamlit_cloud():
        import random

        sid = st.session_state.setdefault(
            "_session_id", random.randint(100_000_000, 999_999_999)
        )
        base_name = f"{sid}_{base_name}"
    return str(results_dir() / f"{base_name}.db")


# The columns of the sequences table.
SEQUENCE_COLS = ["seq_id", "seq_length", "dna_sequence"]


def display_fasta_sequence_table():
    """Renders the loaded sequences in an expandable table."""
    df = st.session_state.sequences_df
    with st.expander(
        f"📄 View FASTA File Content ({len(df):,} {pluralize(len(df), 'sequence')})"
    ):
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_order=SEQUENCE_COLS,
            column_config=column_config(SEQUENCE_COLS),
        )


def _fasta_handle(file_name: str, data: bytes) -> io.StringIO:
    r"""An uploaded FASTA as a named text handle, parsed from memory: a shared
    temp file would race between sessions on Cloud. The cache keys the FASTA
    by this name; newline=None translates \r and \r\n line endings, as
    open() does. A UTF-8 byte order mark is dropped, or the first header would
    not start with ">"; a file that is not UTF-8 raises UnicodeDecodeError."""
    handle = io.StringIO(data.decode("utf-8-sig"), newline=None)
    handle.name = file_name
    return handle


def _start_fasta_analysis(file_name: str, data: bytes, settings: dict | None = None):
    """Load a FASTA as the active analysis, named after the file.

    *settings* pre-fill widgets of the later steps (widget key -> value).
    The file is read before anything is saved: a file without sequences
    leaves no analysis behind.
    """
    try:
        records = load_fasta(_fasta_handle(file_name, data))
    except UnicodeDecodeError:
        st.error(
            f"❌ {file_name} is not UTF-8 text: save it as UTF-8 and upload it again."
        )
        return
    except ValueError as e:
        st.error(
            f"❌ {file_name} is not a FASTA file TRIDENT can read: {str(e).splitlines()[0]}"
        )
        return
    if not records:
        st.error("❌ No valid sequences found.")
        return
    seqs_df = records_to_dataframe(records)

    base_name = Path(file_name).stem
    name = _fasta_analysis_name(base_name, seqs_df)
    if _is_active(name):
        _clear_upload(f"ℹ️ {name} already active.")
    db_path = _db_path(name)
    # An analysis with these sequences is opened as it is: they are saved by
    # filename, so saving them from a file named otherwise would add a copy.
    if Path(db_path).exists():
        compatible, message = check_schema_version(db_path)
        if not compatible:
            st.error(message)
            return
        logger.info(f"Analysis opened from {db_path}")
    else:
        logger.info(f"Analysis saved in {db_path}")
        with st.spinner(f"Processing {file_name}...", show_time=True):
            seqs_df, _ = pipe.run_fasta_workflow(
                _fasta_handle(file_name, data), db_path=db_path
            )

    st.session_state.sequences_df = seqs_df
    st.session_state.sequences_id = seqs_df["seq_id"].unique().tolist()
    st.session_state.analysis_name = name
    st.session_state.db_path = db_path

    reset_for_new_analysis("sequences_df")
    if settings:
        prefill_settings(settings)
    if name != base_name:
        st.session_state._upload_notice = (
            f"ℹ️ An analysis named {base_name} already holds other sequences: "
            f"this file is saved as **{name}**."
        )
    st.session_state.fasta_uploader_key += 1
    st.session_state.requested_tab = "start"
    st.rerun()


def _fasta_analysis_name(name: str, seqs_df: pd.DataFrame) -> str:
    """*name*, or "name (2)", "name (3)"... while that analysis holds other
    sequences: the sequences are saved by filename, so an edited file under the
    same name would otherwise reopen the old analysis. The first analysis with
    these sequences is reused."""
    return _free_name(name, lambda path: _holds_sequences(path, seqs_df))


def _holds_sequences(db_path: Path, seqs_df: pd.DataFrame) -> bool:
    """Whether the analysis at *db_path* is free (no file) or holds exactly
    these sequences."""
    if not db_path.exists():
        return True
    if not _has_sequences(db_path):
        return False
    try:
        with get_connection(db_path) as conn:
            saved = pd.read_sql("SELECT seq_id, dna_sequence FROM sequences", conn)
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return False  # not a TRIDENT sequences table
    columns = ["seq_id", "dna_sequence"]
    return saved[columns].values.tolist() == seqs_df[columns].values.tolist()


def _free_name(name: str, usable=lambda path: not path.exists()) -> str:
    """*name*, or the first of "name (2)", "name (3)"... whose analysis file is
    *usable* (by default: not there yet)."""
    candidate, n = name, 1
    while not usable(Path(_db_path(candidate))):
        n += 1
        candidate = f"{name} ({n})"
    return candidate


def _same_sqlite_content(path: Path, data: bytes) -> bool:
    """Whether the SQLite file at *path* holds these bytes, outside the header
    fields any commit rewrites even when no row changed: the change counter
    (bytes 24-27), and the counter and SQLite version of that commit (92-99).
    Read only when the sizes match, then compared by 1 MB blocks: as fast as
    one comparison, without a second copy of the file in memory."""
    if path.stat().st_size != len(data):
        return False
    saved = path.read_bytes()
    if saved[:24] != data[:24] or saved[28:92] != data[28:92]:
        return False
    block = 1 << 20
    return all(
        saved[i : i + block] == data[i : i + block]
        for i in range(100, len(data), block)
    )


def _saved_copy(data: bytes, name: str, active: str | None = None) -> Path | None:
    """The saved analysis of the results folder with the same content as these
    bytes (_same_sqlite_content), or None: uploading it again opens it instead of
    copying it. Among identical copies, the *active* one comes first (its
    upload is then seen as already active), then the one named *name*. A file
    without sequences is not an analysis to open.

    Not on Streamlit Cloud, where the folder holds every user's analyses.
    """
    if is_streamlit_cloud():
        return None
    paths = sorted(
        results_dir().glob("*.db"),
        key=lambda path: (path.stem != active, path.stem != name),
    )
    for path in paths:
        if _same_sqlite_content(path, data) and _has_sequences(path):
            return path
    return None


def _has_sequences(db_path: Path) -> bool:
    """Whether the database holds the sequences an analysis is restored from
    (an empty file, a file that is not SQLite or another tool's SQLite file
    does not)."""
    try:
        with get_connection(db_path) as conn:
            return table_exists(conn, "sequences")
    except sqlite3.DatabaseError:
        return False


def _restore_db(uploaded_file, name: str):
    """Restore analysis from an uploaded .db file, saved as *name*."""
    target_path = Path(_db_path(name))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Checked under another name first, so an incompatible file never
    # replaces the analysis it was meant to overwrite.
    upload_path = target_path.with_suffix(".upload")
    upload_path.write_bytes(uploaded_file.getvalue())

    compatible, message = check_schema_version(str(upload_path))
    if compatible and not _has_sequences(upload_path):
        compatible, message = False, "This file is not a TRIDENT analysis file."
    if not compatible:
        upload_path.unlink(missing_ok=True)
        st.error(message)
        return
    upload_path.replace(target_path)
    _open_analysis(name)


def _open_saved(path: Path):
    """Open the saved analysis at *path*, if it is still there and this version
    of TRIDENT can read it."""
    # Checked first: opening a missing file would create an empty one.
    if not path.is_file():
        st.error("This analysis is no longer in the results folder.")
        return
    compatible, message = check_schema_version(str(path))
    if not compatible:
        st.error(message)
        return
    _open_analysis(path.stem)


def _open_analysis(name: str):
    """Make the saved analysis *name* (in the results folder) the active one."""
    reset_for_new_analysis("analysis_name")
    st.session_state.analysis_name = name
    st.session_state.db_path = _db_path(name)
    st.session_state.sequences_df = None  # Will trigger auto-restore on rerun
    # A fresh uploader: a restored file would otherwise stay in it, offering
    # to restore it again under a new name.
    st.session_state.fasta_uploader_key += 1
    st.session_state.requested_tab = "start"
    st.rerun()


def _saved_analyses() -> list[Path]:
    """The saved analyses of the results folder, newest first, the active one
    left out."""
    paths = [
        path
        for path in results_dir().glob("*.db")
        if path.stem != st.session_state.get("analysis_name") and _has_sequences(path)
    ]
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def _analysis_label(path: Path) -> str:
    """A saved analysis as the list shows it, e.g. "name (20 Oct 2026, 86 MB)":
    the date and size tell apart copies of one analysis."""
    stat = path.stat()
    day = datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y")
    mb = stat.st_size / 1e6
    size = f"{mb:,.0f} MB" if mb >= 1 else f"{stat.st_size / 1e3:,.0f} kB"
    return f"{path.stem} ({day}, {size})"


def open_saved_analysis():
    """Open an analysis already in the results folder, without copying it.

    Not on Streamlit Cloud, where the folder holds every user's analyses.
    """
    if is_streamlit_cloud():
        return
    paths = _saved_analyses()
    if not paths:
        return
    labels = {str(path): _analysis_label(path) for path in paths}
    choice = st.selectbox(
        "Or open an analysis saved on this computer",
        list(labels),
        index=None,
        format_func=labels.get,
        placeholder="Choose an analysis",
        key=f"open_analysis_{st.session_state.fasta_uploader_key}",
    )
    if st.button(
        "Open Analysis",
        icon=":material/folder_open:",
        type="primary",
        disabled=choice is None,
        width="stretch",
    ):
        _open_saved(Path(choice))


def _auto_restore_from_db():
    """If analysis_name is set but sequences are missing, restore from DB."""
    if not (
        st.session_state.get("analysis_name")
        and st.session_state.get("sequences_df") is None
    ):
        return

    db_path = Path(_db_path(st.session_state.analysis_name))
    if not db_path.exists():
        return

    with st.spinner("Opening the analysis...", show_time=True):
        try:
            with get_connection(db_path) as conn:
                seqs_df = pd.read_sql(
                    "SELECT seq_id, seq_length, dna_sequence FROM sequences", conn
                )

            if not seqs_df.empty:
                st.session_state.sequences_df = seqs_df
                st.session_state.sequences_id = seqs_df["seq_id"].unique().tolist()
                # Restore the DB path too, otherwise later steps run with the
                # empty default and fail to open the database.
                st.session_state.db_path = str(db_path)
                logger.info(f"Analysis opened from {db_path}")
                st.rerun()
        except Exception as e:
            st.error(f"Could not restore sequences: {e}")


def _clear_upload(notice: str):
    """Clear the uploader, *notice* shown above the choices: a file kept there
    would be read again on every rerun, and offered as an old copy once a step
    runs."""
    st.session_state._upload_notice = notice
    st.session_state.fasta_uploader_key += 1
    st.rerun()


def _is_active(name: str) -> bool:
    """Whether *name* is the analysis open now, its sequences loaded."""
    return (
        st.session_state.get("sequences_df") is not None
        and st.session_state.get("analysis_name") == name
    )


def handle_file_upload():
    """Single file uploader handling both FASTA and DB files."""
    uploaded_file = st.file_uploader(
        "Upload a FASTA file to start a new analysis, or an analysis file (.db) "
        "to import",
        type=sorted(FASTA_EXTENSIONS | DB_EXTENSIONS),
        key=f"file_upload_{st.session_state.fasta_uploader_key}",
    )

    if not uploaded_file:
        return

    ext = Path(uploaded_file.name).suffix.lstrip(".").lower()
    name = Path(uploaded_file.name).stem

    if ext in DB_EXTENSIONS:
        # Compared once per upload: the folder's files are read in full.
        cached = st.session_state.get("_upload_saved_copy")
        if cached and cached[0] == uploaded_file.file_id:
            saved = cached[1]
        else:
            saved = _saved_copy(
                uploaded_file.getvalue(), name, st.session_state.get("analysis_name")
            )
            st.session_state._upload_saved_copy = (uploaded_file.file_id, saved)
        if saved is not None:
            if _is_active(saved.stem):
                _clear_upload(f"ℹ️ {saved.stem} already active.")
            st.caption(
                f"This analysis is already saved on this computer as "
                f"**{saved.stem}**: TRIDENT opens it, without making a copy."
            )
            if st.button(
                "Open Analysis",
                icon=":material/folder_open:",
                type="primary",
                width="stretch",
                key="open_uploaded_analysis",
            ):
                _open_saved(saved)
            return

        free_name = _free_name(name)
        restore_name = st.text_input(
            "Save as:",
            value=free_name,
            key=f"restore_name_{st.session_state.fasta_uploader_key}",
        ).strip()
        # Not on Cloud: the folder holds every user's analyses, never compared.
        if free_name != name and restore_name == free_name and not is_streamlit_cloud():
            st.caption(
                f"A file named **{name}** is already in the results folder, "
                f"so this one is saved as **{free_name}**. Type **{name}** to "
                "replace it."
            )
        if not restore_name:
            st.error("Enter a name for the analysis.")
            return
        if Path(restore_name).name != restore_name:
            st.error("Enter a name without folders.")
            return
        # On Cloud the folder is on the server and holds no list to reopen from.
        if is_streamlit_cloud():
            st.caption("TRIDENT works on a copy: your file is left unchanged.")
        else:
            st.caption(
                f"TRIDENT keeps a copy in its results folder "
                f"(`{_db_path(restore_name)}`) and works on that copy: your file "
                "is left unchanged. To continue it later, open it from the saved "
                "analyses below instead of uploading it again."
            )
        if Path(_db_path(restore_name)).exists():
            st.warning(
                f"An analysis named **{restore_name}** already exists: "
                "restoring replaces it. Change the name to keep both."
            )
        if st.button(
            "Restore Analysis",
            icon=":material/restore:",
            type="primary",
            width="stretch",
        ):
            _restore_db(uploaded_file, restore_name)

    elif ext in FASTA_EXTENSIONS:
        if st.button(
            "Process FASTA",
            icon=":material/play_arrow:",
            type="primary",
            width="stretch",
        ):
            _start_fasta_analysis(uploaded_file.name, uploaded_file.getvalue())


def load_example_button():
    """Offer the bundled example FASTA, for users without a file at hand."""
    if EXAMPLE_FASTA is None or not EXAMPLE_FASTA.exists():
        return

    active = _is_active(EXAMPLE_FASTA.stem)
    if st.button(
        f"Try the example ({EXAMPLE_FASTA.name})",
        icon=":material/science:",
        type="tertiary",
        disabled=active,
        help=(
            "The four Svalbard 16S sequences of the paper's worked example. "
            "The GEO area is set to the paper's (78.5° N, 14.5° E, 250 km; 500 km is "
            "also searched for comparison)."
        ),
    ):
        _start_fasta_analysis(
            EXAMPLE_FASTA.name, EXAMPLE_FASTA.read_bytes(), EXAMPLE_SETTINGS
        )


def display_analysis_status():
    """Display current analysis status and preview."""
    active_name = st.session_state.analysis_name

    with st.container(border=True):
        st.markdown(f"### 📊 Active Analysis: **{active_name}**")
        # On Cloud the path is on the server, of no use to the user.
        if not is_streamlit_cloud() and st.session_state.get("db_path"):
            st.caption(f"Saved in `{st.session_state.db_path}`")

        display_fasta_sequence_table()

        next_step_button("Next: MOL ➔", "mol")


def upload_and_start_analysis():
    """Main entry point for starting analysis."""
    st.header("🚀 **Start Analysis**")
    st.caption(
        "Start an analysis from a FASTA file with (M)OTUs/ASVs sequences, or "
        "open a saved one."
    )
    st.divider()

    _auto_restore_from_db()
    # Set when an upload turned out to be the active analysis: shown here, as
    # the upload's own section is folded again.
    if notice := st.session_state.pop("_upload_notice", None):
        st.info(notice)

    if st.session_state.get("sequences_df") is None:
        _analysis_choices()
        return

    display_analysis_status()
    # Folded while working, so the example is not one click away, and drawn
    # only when open: the saved analyses are read from disk. Keyed by the
    # uploader key, which every switch changes: it opens folded again after one.
    with lazy_expander(
        "Start or open another analysis",
        key=f"switch_analysis_{st.session_state.fasta_uploader_key}",
        icon=":material/swap_horiz:",
    ) as is_open:
        if is_open:
            _analysis_choices()


def _analysis_choices():
    """The ways to start or open an analysis, the most common first: a file,
    a saved analysis, then the example. Only once a contact email is set:
    without it, NCBI and the other services get anonymous requests."""
    if not contact_email():
        st.warning(
            "Enter your contact email in Settings above to start or open an analysis."
        )
        return
    handle_file_upload()
    open_saved_analysis()
    load_example_button()
