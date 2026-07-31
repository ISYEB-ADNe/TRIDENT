"""Alignment failures must not masquerade as legitimate identity scores.

``compute_pairwise_identity`` used to swallow any alignment error and return
0.0, which reads as "not redundant" and silently keeps both sequences. Errors
now propagate to the per-species handler in ``run_bold_filter``, which is
where the deliberate keep-everything fallback lives.
"""

import pandas as pd
import pytest

import trident.core.sequence_selection as sequence_selection
from trident.pipelines.extra_pipeline import run_bold_filter

SEQ_A = "ATCGATCGATCGATCGATCGATCG"
SEQ_B = "ATCGATCGATCGATCGATCGATCT"


def _search_df(species: list[str]) -> pd.DataFrame:
    # Identical pairs per species, so filtering keeps exactly one of each.
    return pd.DataFrame(
        {
            "scientificName": [s for s in species for _ in range(2)],
            "dna_sequence": [SEQ_A, SEQ_A] * len(species),
        }
    )


def test_alignment_errors_propagate_instead_of_scoring_zero():
    class ExplodingAligner:
        def align(self, seq1, seq2):
            raise ValueError("aligner exploded")

    original = sequence_selection._ALIGNER
    sequence_selection._ALIGNER = ExplodingAligner()
    try:
        with pytest.raises(ValueError, match="aligner exploded"):
            sequence_selection.compute_pairwise_identity(SEQ_A, SEQ_B)
    finally:
        sequence_selection._ALIGNER = original


def test_bold_filter_keeps_a_single_failing_species_unfiltered():
    calls = {"n": 0}
    original = sequence_selection._ALIGNER

    class FlakyAligner:
        def align(self, seq1, seq2):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("one bad pair")
            return original.align(seq1, seq2)

    sequence_selection._ALIGNER = FlakyAligner()
    try:
        result_df, _ = run_bold_filter(_search_df(["A", "B"]), similarity=98)
    finally:
        sequence_selection._ALIGNER = original

    # Species A kept whole (2 rows, filtering failed), species B deduplicated.
    assert sorted(result_df["scientificName"].tolist()) == ["A", "A", "B"]
