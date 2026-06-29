"""FASTA input pipeline — loads and caches input sequences.

Step order:
    1. run_fasta_workflow(fasta_file) → (sequences_df, params)
"""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from trident.clients.fasta import records_to_dataframe, load_fasta
from trident.core.database import save_to_db, CustomCache


def _get_fasta_filename(fasta_file: Any) -> str:
    """Extract a display name from a fasta_file argument.

    Args:
        fasta_file: File path (str or Path) or file-like object.

    Returns:
        Filename string for logging and caching.
    """
    if isinstance(fasta_file, (str, Path)):
        return Path(fasta_file).name
    return getattr(fasta_file, "name", "uploaded_buffer")


def _fasta_workflow_params(fasta_file: Any, **kwargs) -> tuple[list[dict], dict]:
    """Prepare cache input record for the FASTA workflow step.

    Uses the filename (not the full path) as the cache key so that the same
    file loaded from different locations hits the same cache entry.
    """
    return [{"filename": _get_fasta_filename(fasta_file)}], {}


@save_to_db(
    table_name="sequences",
    # CustomCache because the cache key is derived from fasta_file (not a plain kwarg value).
    cache=CustomCache(prepare_fn=_fasta_workflow_params),
)
def run_fasta_workflow(fasta_file: Any) -> pd.DataFrame:
    """Load a FASTA file and return a sequences DataFrame.

    Accepts local file paths (CLI) and file-like objects (Streamlit).

    Args:
        fasta_file: File path (str or Path) or file-like object.

    Returns:
        DataFrame with seq_id, seq_length, and dna_sequence columns.
        Returns None if no sequences found (the caching decorator
        converts this to an empty DataFrame).
    """
    filename = _get_fasta_filename(fasta_file)
    logger.info(f"Starting FASTA workflow: {filename}")

    sequences = load_fasta(fasta_file)

    if not sequences:
        logger.warning(f"No sequences found in {filename}. Aborting.")
        return None

    df = records_to_dataframe(sequences)
    logger.success(f"Read {len(df)} sequences from {filename}")
    return df
