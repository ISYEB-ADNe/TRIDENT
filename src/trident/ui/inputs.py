"""
Input handling UI: FASTA upload, DB restore, and sequence preview.
"""

import io
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

import trident.pipelines as pipe
from trident.core.config import is_streamlit_cloud
from trident.core.database import check_schema_version
from trident.ui.ui import prefill_settings, reset_for_new_analysis, COL

FASTA_EXTENSIONS = {"fasta", "fa", "txt", "fas"}
DB_EXTENSIONS = {"db", "sqlite", "sqlite3"}

# The example ships at the repository root (example/), next to src/, in every
# documented install (git clone or downloaded archive).
EXAMPLE_FASTA = Path(__file__).resolve().parents[3] / "example" / "example.fasta"
# The example's sequences come from Svalbard: pre-fill the GEO window of the
# paper's worked example, or GBIF would be searched around 0°, 0°.
EXAMPLE_SETTINGS = {
    "gbif_latitude": "78.5",
    "gbif_longitude": "14.5",
    "gbif_extent": "250, 500",
}


def _db_path(base_name: str) -> str:
    """Return a DB path, scoped to the session on Streamlit Cloud."""
    if is_streamlit_cloud():
        import random

        sid = st.session_state.setdefault(
            "_session_id", random.randint(100_000_000, 999_999_999)
        )
        path = f"./results/{sid}_{base_name}.db"
    else:
        path = f"./results/{base_name}.db"
    return path


def display_fasta_sequence_table():
    """Renders the loaded sequences in an expandable table."""
    df = st.session_state.sequences_df
    with st.expander(f"📄 View FASTA File Content ({len(df)} sequences)"):
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "seq_id": COL["seq_id"],
                "dna_sequence": COL["dna_sequence"],
                "seq_length": COL["seq_length"],
            },
        )


def _start_fasta_analysis(file_name: str, data: bytes, settings: dict | None = None):
    """Load a FASTA as the active analysis, named after the file.

    *settings* pre-fill widgets of the later steps (widget key -> value).
    """
    base_name = Path(file_name).stem
    # Parsed from memory: a shared temp file would race between sessions on
    # Cloud. The cache keys the FASTA by this name.
    # newline=None translates \r and \r\n line endings, as open() does.
    handle = io.StringIO(data.decode(), newline=None)
    handle.name = file_name

    db_path = _db_path(base_name)
    logger.info(f"Full analysis saved in {db_path}")
    with st.spinner(f"Processing {file_name}...", show_time=True):
        seqs_df, _ = pipe.run_fasta_workflow(handle, db_path=db_path)

    if seqs_df.empty:
        st.error("❌ No valid sequences found.")
        return

    st.session_state.sequences_df = seqs_df
    st.session_state.sequences_id = seqs_df["seq_id"].unique().tolist()
    st.session_state.analysis_name = base_name
    st.session_state.db_path = db_path

    reset_for_new_analysis("sequences_df")
    if settings:
        prefill_settings(settings)
    st.session_state.fasta_uploader_key += 1
    st.session_state.requested_tab = "start"
    st.rerun()


def _restore_db(uploaded_file):
    """Restore analysis from an uploaded .db file."""
    name = Path(uploaded_file.name).stem
    db = _db_path(name)
    target_path = Path(db)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(uploaded_file.getvalue())

    compatible, message = check_schema_version(db)
    if not compatible:
        target_path.unlink(missing_ok=True)
        st.error(message)
        return

    reset_for_new_analysis("analysis_name")
    st.session_state.analysis_name = name
    st.session_state.db_path = db
    st.session_state.sequences_df = None  # Will trigger auto-restore on rerun
    st.session_state.requested_tab = "start"
    st.rerun()


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

    with st.spinner("Restoring sequences from database...", show_time=True):
        try:
            conn = sqlite3.connect(db_path)
            seqs_df = pd.read_sql("SELECT * FROM sequences", conn)
            conn.close()

            if not seqs_df.empty:
                st.session_state.sequences_df = seqs_df
                st.session_state.sequences_id = seqs_df["seq_id"].unique().tolist()
                # Restore the DB path too, otherwise later steps run with the
                # empty default and fail to open the database.
                st.session_state.db_path = str(db_path)
                logger.info(f"Full analysis opened from {db_path}")
                st.rerun()
        except Exception as e:
            st.error(f"Could not restore sequences: {e}")


def handle_file_upload():
    """Single file uploader handling both FASTA and DB files."""
    uploaded_file = st.file_uploader(
        "Upload a FASTA file or a previously exported .db file",
        type=sorted(FASTA_EXTENSIONS | DB_EXTENSIONS),
        key=f"file_upload_{st.session_state.fasta_uploader_key}",
    )

    if not uploaded_file:
        return

    ext = Path(uploaded_file.name).suffix.lstrip(".").lower()
    name = Path(uploaded_file.name).stem

    if ext in DB_EXTENSIONS:
        if st.button(
            "Restore Analysis",
            icon=":material/restore:",
            type="primary",
            use_container_width=True,
        ):
            _restore_db(uploaded_file)

    elif ext in FASTA_EXTENSIONS:
        # Skip if already loaded
        if st.session_state.get(
            "sequences_df"
        ) is not None and name == st.session_state.get("analysis_name"):
            st.info(f"ℹ️ {uploaded_file.name} already active.")
            return

        if st.button(
            "Process FASTA",
            icon=":material/play_arrow:",
            type="primary",
            use_container_width=True,
        ):
            _start_fasta_analysis(uploaded_file.name, uploaded_file.getvalue())


def load_example_button():
    """Offer the bundled example FASTA, for users without a file at hand."""
    if not EXAMPLE_FASTA.exists():
        return

    active = (
        st.session_state.get("sequences_df") is not None
        and st.session_state.get("analysis_name") == EXAMPLE_FASTA.stem
    )
    if st.button(
        f"Load the example ({EXAMPLE_FASTA.name})",
        icon=":material/science:",
        disabled=active,
        help=(
            "The four Svalbard 16S sequences of the paper's worked example. "
            "The GEO window is set to the sampling area (78.5° N, 14.5° E; 250 and 500 km)."
        ),
        use_container_width=True,
    ):
        _start_fasta_analysis(
            EXAMPLE_FASTA.name, EXAMPLE_FASTA.read_bytes(), EXAMPLE_SETTINGS
        )


def display_analysis_status():
    """Display current analysis status and preview."""
    active_name = st.session_state.analysis_name

    with st.container(border=True):
        st.markdown(f"### 📊 Active Analysis: **{active_name}**")

        display_fasta_sequence_table()

        if st.button(
            "Proceed to MOL",
            icon=":material/arrow_forward:",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state.requested_tab = "mol"
            st.rerun()


def upload_and_start_analysis():
    """Main entry point for starting analysis."""
    st.header("🚀 **Start Analysis**")
    st.caption(
        "Upload a FASTA file with (M)OTUs/ASVs sequences, or restore a previous analysis from a .db file."
    )
    st.divider()

    _auto_restore_from_db()

    handle_file_upload()
    load_example_button()

    st.divider()

    if st.session_state.get("sequences_df") is not None:
        display_analysis_status()
    else:
        st.info("👆 **Get started** by uploading a file above, or load the example.")
