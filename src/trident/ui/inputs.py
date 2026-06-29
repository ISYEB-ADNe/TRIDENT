"""
Input handling UI: FASTA upload, DB restore, and sequence preview.
"""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

import trident.pipelines as pipe
from trident.core.config import is_streamlit_cloud
from trident.core.database import check_schema_version
from trident.ui.ui import reset_state_after, COL

FASTA_EXTENSIONS = {"fasta", "fa", "txt", "fas"}
DB_EXTENSIONS = {"db", "sqlite", "sqlite3"}


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
    logger.info(f"Database path: {path}")
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


def _process_fasta(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Saves upload to temp, parses sequences via trident.fasta, and cleans up."""
    base_name = Path(uploaded_file.name).stem
    temp_path = Path(tempfile.gettempdir()) / f"{base_name}.fasta"

    temp_path.write_bytes(uploaded_file.getvalue())

    try:
        sequences_df, _ = pipe.run_fasta_workflow(
            str(temp_path), db_path=_db_path(base_name)
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return sequences_df, base_name


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

    st.session_state.analysis_name = name
    st.session_state.db_path = db
    st.session_state.sequences_df = None  # Will trigger auto-restore on rerun
    reset_state_after("analysis_name")
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
            with st.spinner(f"Processing {uploaded_file.name}...", show_time=True):
                seqs_df, base_name = _process_fasta(uploaded_file)

                if not seqs_df.empty:
                    st.session_state.sequences_df = seqs_df
                    st.session_state.sequences_id = seqs_df["seq_id"].unique().tolist()
                    st.session_state.analysis_name = base_name
                    st.session_state.db_path = _db_path(base_name)

                    reset_state_after("sequences_df")
                    st.session_state.fasta_uploader_key += 1
                    st.session_state.requested_tab = "start"
                    st.rerun()
                else:
                    st.error("❌ No valid sequences found.")


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

    st.divider()

    if st.session_state.get("sequences_df") is not None:
        display_analysis_status()
    else:
        st.info("👆 **Get started** by uploading a file above.")
