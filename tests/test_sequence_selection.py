"""Tests for trident.core.sequence_selection: pairwise identity and longest selection."""

from trident.core.sequence_selection import (
    compute_pairwise_identity,
    select_longest_sequences,
)


def test_identity_is_100_for_equal_sequences_and_0_without_alignment():
    """An empty sequence, or lengths too different for the threshold, score 0
    without aligning."""
    assert compute_pairwise_identity("ATCGATCG", "ATCGATCG") == 100.0
    assert 0 < compute_pairwise_identity("ATCGATCG", "TTTTTTTT") < 100
    assert compute_pairwise_identity("", "ATCG") == 0.0
    assert compute_pairwise_identity("ATCG", "") == 0.0
    assert compute_pairwise_identity("AT", "ATCG" * 4, identity_threshold=90) == 0.0


def test_longest_sequences_are_kept_first_and_identical_ones_once():
    indices, selected = select_longest_sequences(
        ["AT", "ATCGATCG", "ATCG"], n_longest=2, identity_threshold=99
    )
    assert indices[0] == 1 and len(selected) <= 2
    assert len(select_longest_sequences(["ATCG"] * 3, identity_threshold=90)[1]) == 1
    assert select_longest_sequences(["ATCG"], n_longest=0) == ([], [])


def test_a_shorter_fragment_of_a_longer_sequence_is_kept():
    """A fragment much shorter than another sequence is not collapsed into it:
    the length ratio counts very different lengths as non-redundant."""
    longer = "ATCGATCGATCGATCGATCG"  # 20 bp
    fragment = "ATCGATCGAT"  # 10 bp, a prefix of `longer`
    _, selected = select_longest_sequences([longer, fragment], identity_threshold=98)
    assert len(selected) == 2
