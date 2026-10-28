"""FASTA I/O: load, build and serialize FASTA sequences via Biopython."""

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
    with io.StringIO() as output:
        SeqIO.write(sequences, output, "fasta")
        return output.getvalue()


def clean_sequence(dna_sequence: str) -> str:
    """The bases of a sequence: uppercase, without gaps ('-') and without the
    Ns at its ends. A gap is a position where the sequence has no base (BOLD
    keeps some records aligned, as submitted), and end Ns hold no base either:
    both would count in the sequence's length, and so in the query cover.
    Lowercase is the same base: BLAST reads it as uppercase, and the BOLD
    filter must too. Inner Ns and other ambiguity codes stay, as bases that
    could not be read. A sequence of gaps and Ns only keeps its length, as
    Ns."""
    sequence = dna_sequence.strip().upper()
    return sequence.replace("-", "").strip("N") or sequence.replace("-", "N")


def blast_record(seq_id, dna_sequence: str) -> SeqRecord:
    """A sequence as a BLAST query, with no description so the FASTA header
    is the id alone. It is cleaned only when sent (see submit_blast_batch), so
    the hits keep the sequence as given."""
    return SeqRecord(Seq(dna_sequence), id=str(seq_id), description="")


def dataframe_to_records(sequences_df: pd.DataFrame) -> list[SeqRecord]:
    """Convert a sequences DataFrame back to SeqRecord objects.

    Each sequence is made a BLAST record (see blast_record).

    Args:
        sequences_df: DataFrame with seq_id and dna_sequence columns,
            as produced by records_to_dataframe().

    Returns:
        List of SeqRecord objects.
    """
    return [
        blast_record(seq_id, dna_sequence)
        for seq_id, dna_sequence in zip(
            sequences_df["seq_id"], sequences_df["dna_sequence"]
        )
    ]
