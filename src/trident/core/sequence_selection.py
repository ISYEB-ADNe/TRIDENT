"""Non-redundant sequence selection via pairwise identity."""

from __future__ import annotations

from loguru import logger

from Bio.Align import PairwiseAligner


# --------------------------------------------------------------------
# Sequence Similarity Functions
# --------------------------------------------------------------------

# Module-level aligner instance (reused across all comparisons)
_ALIGNER = PairwiseAligner()
_ALIGNER.mode = "global"
_ALIGNER.match_score = 1
_ALIGNER.mismatch_score = -1
_ALIGNER.open_gap_score = -2
_ALIGNER.extend_gap_score = -0.1


def _length_ratio(seq1: str, seq2: str) -> float:
    """Return length ratio (0-1) of the shorter to the longer sequence."""
    shorter, longer = sorted((len(seq1), len(seq2)))
    return shorter / longer if longer > 0 else 0.0


def compute_pairwise_identity(
    seq1: str,
    seq2: str,
    identity_threshold: float | None = None,
) -> float:
    """Percent identity (0-100) via global pairwise alignment.

    If *identity_threshold* is given, skips pairs whose length ratio rules them out.

    Args:
        seq1: First DNA sequence.
        seq2: Second DNA sequence.
        identity_threshold: Skip pair if length ratio rules it out.

    Returns:
        Percent identity (0-100).
    """
    if not seq1 or not seq2:
        logger.warning("Empty sequence provided to compute_pairwise_identity")
        return 0.0

    if identity_threshold is not None:
        if _length_ratio(seq1, seq2) * 100 < identity_threshold:
            return 0.0

    try:
        alignments = _ALIGNER.align(seq1, seq2)
        best = next(iter(alignments))
    except StopIteration:
        logger.warning(
            f"No alignment found for sequences of length {len(seq1)} and {len(seq2)}"
        )
        return 0.0
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return 0.0

    aligned = [(a, b) for a, b in zip(best[0], best[1]) if a != "-" or b != "-"]
    if not aligned:
        return 0.0
    matches = sum(1 for a, b in aligned if a == b)
    return matches / len(aligned) * 100


def select_longest_sequences(
    sequences: list[str],
    n_longest: int | None = None,
    identity_threshold: float = 98.0,
) -> tuple[list[int], list[str]]:
    """Select non-redundant sequences, longest first.

    Skips sequences above *identity_threshold* to an already-selected one.

    Args:
        sequences: DNA sequence strings.
        n_longest: Max selections (None = keep all non-redundant).
        identity_threshold: % identity above which sequences are redundant.

    Returns:
        (selected_indices, selected_sequences).
    """
    if n_longest is not None and n_longest <= 0:
        return [], []

    seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
    seq_lengths.sort(key=lambda x: x[1], reverse=True)

    selected_sequences = []
    selected_indices = []

    for idx, _ in seq_lengths:
        current_seq = sequences[idx]
        is_redundant = False

        for sel_seq in selected_sequences:
            identity = compute_pairwise_identity(
                current_seq, sel_seq, identity_threshold=identity_threshold
            )
            if identity >= identity_threshold:
                is_redundant = True
                break

        if not is_redundant:
            selected_indices.append(idx)
            selected_sequences.append(current_seq)

        if n_longest is not None and len(selected_sequences) >= n_longest:
            break

    return selected_indices, selected_sequences
