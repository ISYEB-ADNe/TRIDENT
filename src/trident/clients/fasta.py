"""FASTA I/O — load, build, and serialize FASTA sequences via BioPython."""

import io
from typing import Any

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from loguru import logger


def load_fasta(fasta_file: Any) -> list[SeqRecord]:
    """Load sequences from a FASTA source.

    Args:
        fasta_file: File path (str or Path), file-like object, or Streamlit
            UploadedFile. Any object accepted by Bio.SeqIO.parse.

    Returns:
        List of SeqRecord objects parsed from the FASTA source.
    """
    display_name = getattr(fasta_file, "name", "buffer")
    sequences = list(SeqIO.parse(fasta_file, "fasta"))

    if not sequences:
        logger.debug(f"No sequences found in {display_name}")
    else:
        logger.debug(f"Loaded {len(sequences)} sequences from {display_name}")

    return sequences


def records_to_dataframe(sequences: list[SeqRecord]) -> pd.DataFrame:
    """Convert SeqRecord objects to a DataFrame.

    Args:
        sequences: List of SeqRecord objects.

    Returns:
        DataFrame with columns seq_id, seq_length, and dna_sequence.
    """
    logger.debug(f"Building DataFrame for {len(sequences)} sequences")
    return pd.DataFrame(
        {
            "seq_id": [seq.id for seq in sequences],
            "seq_length": [len(seq) for seq in sequences],
            "dna_sequence": [str(seq.seq) for seq in sequences],
        }
    )


def records_to_fasta_string(sequences: list[SeqRecord]) -> str:
    """Serialize SeqRecord objects to a FASTA-formatted string.

    Args:
        sequences: List of SeqRecord objects.

    Returns:
        FASTA-formatted string.
    """
    logger.debug(f"Building FASTA string for {len(sequences)} sequences")
    with io.StringIO() as output:
        SeqIO.write(sequences, output, "fasta")
        return output.getvalue()


def dataframe_to_records(sequences_df: pd.DataFrame) -> list[SeqRecord]:
    """Convert a sequences DataFrame back to SeqRecord objects.

    Gaps ('-') in sequences are replaced with 'N' before conversion.

    Args:
        sequences_df: DataFrame with seq_id and dna_sequence columns,
            as produced by records_to_dataframe().

    Returns:
        List of SeqRecord objects.
    """
    logger.debug(f"Converting DataFrame to SeqRecord list ({len(sequences_df)} rows)")
    return [
        SeqRecord(Seq(dna_sequence.replace("-", "N")), id=str(seq_id), description="")
        for seq_id, dna_sequence in zip(
            sequences_df["seq_id"], sequences_df["dna_sequence"]
        )
    ]
