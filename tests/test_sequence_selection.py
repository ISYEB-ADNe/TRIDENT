"""Tests for trident.core.sequence_selection: pairwise identity and longest selection."""

from trident.core.sequence_selection import (
    compute_pairwise_identity,
    select_longest_sequences,
)


# --- compute_pairwise_identity ---


def test_identity_identical_sequences():
    identity = compute_pairwise_identity("ATCGATCG", "ATCGATCG")
    assert identity == 100.0


def test_identity_empty_sequence():
    assert compute_pairwise_identity("", "ATCG") == 0.0
    assert compute_pairwise_identity("ATCG", "") == 0.0


def test_identity_different_sequences():
    identity = compute_pairwise_identity("ATCGATCG", "TTTTTTTT")
    assert 0 < identity < 100


def test_identity_threshold_skips_short():
    """Length ratio shortcut: very different lengths → 0.0 when threshold is high."""
    identity = compute_pairwise_identity(
        "AT", "ATCGATCGATCGATCG", identity_threshold=90
    )
    assert identity == 0.0


# --- select_longest_sequences ---


def test_longest_basic():
    seqs = ["AT", "ATCGATCG", "ATCG"]
    indices, selected = select_longest_sequences(
        seqs, n_longest=2, identity_threshold=99
    )
    assert len(selected) <= 2
    # Longest should be first selected
    assert indices[0] == 1


def test_longest_all_identical():
    """All identical → only one kept."""
    seqs = ["ATCG", "ATCG", "ATCG"]
    indices, selected = select_longest_sequences(seqs, identity_threshold=90)
    assert len(selected) == 1


def test_longest_zero():
    indices, selected = select_longest_sequences(["ATCG"], n_longest=0)
    assert indices == []
    assert selected == []


def test_longest_conservative_keeps_length_mismatched_fragment():
    """Deliberate conservative choice: a shorter fragment of a
    longer sequence is NOT collapsed, because the length-ratio gate treats very
    different lengths as non-redundant. Global alignment never over-merges.
    Pinning this so any future switch to local/semi-global is intentional.
    """
    longer = "ATCGATCGATCGATCGATCG"  # 20 bp
    fragment = "ATCGATCGAT"  # 10 bp, a prefix substring of `longer`
    indices, selected = select_longest_sequences(
        [longer, fragment], identity_threshold=98
    )
    assert len(selected) == 2  # both kept: ratio 0.5 < 0.98 -> not redundant
