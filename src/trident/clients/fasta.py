"""FASTA I/O: load, build and serialize FASTA sequences via Biopython."""

import io
import os
from collections import Counter
from typing import Any

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from loguru import logger

from trident.core.utils import pluralize


class RepeatedSequenceNames(ValueError):
    """A FASTA whose sequences do not all have their own name: every step
    keys a sequence by it, so two sharing one would be merged."""

    SHOWN = 5  # names listed; the others are counted

    def __init__(self, repeated: dict[str, int]):
        self.repeated = repeated
        names = ", ".join(
            f"{name or '(empty header)'} ({n} sequences)"
            for name, n in list(repeated.items())[: self.SHOWN]
        )
        if (more := len(repeated) - self.SHOWN) > 0:
            names += f" and {more} more {pluralize(more, 'name')}"
        super().__init__(
            f"Some sequences share a name: {names}. A sequence is named by its "
            "header up to the first space or tab; give each its own name."
        )


def load_fasta(fasta_file: Any) -> list[SeqRecord]:
    """Load sequences from a FASTA source.

    A file at a path and binary content are read as UTF-8. A byte order mark
    is dropped, and Windows and old Mac line endings are accepted.

    Args:
        fasta_file: File path (str or Path), or text or binary file-like
            object.

    Returns:
        List of SeqRecord objects parsed from the FASTA source.

    Raises:
        RepeatedSequenceNames: two or more sequences share a name (the header
            up to its first space or tab).
        ValueError: the source has text before its first sequence header
            (blank lines there are ignored).
        UnicodeDecodeError: the file or the binary content is not UTF-8.
    """
    if isinstance(fasta_file, (str, os.PathLike)):
        display_name = str(fasta_file)
        with open(fasta_file, "rb") as handle:
            text = handle.read()
    else:
        display_name = getattr(fasta_file, "name", "buffer")
        text = fasta_file.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    # Checked here because Biopython versions differ on text before the first
    # header: some skip it with a warning, others refuse the file.
    text = text.removeprefix("\ufeff").lstrip()
    if text and not text.startswith(">"):
        raise ValueError("text before the first sequence header (>)")
    # newline=None translates \r and \r\n line endings, as open() does.
    sequences = list(SeqIO.parse(io.StringIO(text, newline=None), "fasta"))
    counts = Counter(seq.id for seq in sequences)
    repeated = {name: n for name, n in counts.items() if n > 1}
    if repeated:
        raise RepeatedSequenceNames(repeated)

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
