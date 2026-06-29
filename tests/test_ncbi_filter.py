"""Tests for the NCBI barcoding-gap and similarity filtering (the core MOL method).

All functions here are pure (no network), so they are exercised directly.
"""

import numpy as np
import pandas as pd
import pytest

from trident.clients.ncbi import (
    compute_barcoding_gap,
    filter_by_barcoding_gap,
    filter_by_similarity,
    filter_blast_results,
)


def _hits(rows):
    """Build a BLAST-like frame; rows are (seq_id, identity, query_cover, name)."""
    return pd.DataFrame(
        rows,
        columns=["seq_id", "identity_percentage", "query_cover", "scientificName"],
    )


# --- compute_barcoding_gap ---


def test_compute_barcoding_gap_drop_to_next():
    df = _hits(
        [
            ("s1", 99.0, 100, "A"),
            ("s1", 96.0, 100, "B"),
            ("s1", 90.0, 100, "C"),
        ]
    )
    out = compute_barcoding_gap(df)
    drops = out["identity_drop"].tolist()
    assert drops[0] == pytest.approx(3.0)  # 99 -> 96
    assert drops[1] == pytest.approx(6.0)  # 96 -> 90
    assert np.isnan(drops[2])  # last row has no next hit


# --- filter_by_barcoding_gap ---


def test_gap_found_keeps_above_gap():
    # 99, 98 then an 8-point drop to 90: gap after the 98 hit.
    df = _hits(
        [
            ("s1", 99.0, 100, "A"),
            ("s1", 98.0, 100, "B"),
            ("s1", 90.0, 100, "C"),
        ]
    )
    gap_df, no_gap_df, method = filter_by_barcoding_gap(df, gap_size=2, gap_min_top=97)
    assert method["s1"] == "barcoding_gap"
    assert sorted(gap_df["identity_percentage"].tolist()) == [98.0, 99.0]
    assert no_gap_df.empty


def test_no_gap_falls_back_to_similarity():
    # All drops < gap_size -> no detectable gap.
    df = _hits(
        [
            ("s1", 99.0, 100, "A"),
            ("s1", 98.5, 100, "B"),
            ("s1", 98.0, 100, "C"),
        ]
    )
    gap_df, no_gap_df, method = filter_by_barcoding_gap(df, gap_size=2, gap_min_top=97)
    assert method["s1"] == "similarity"
    assert gap_df.empty
    assert len(no_gap_df) == 3


def test_gap_below_min_top_is_not_a_gap():
    # A big drop, but the top hit is below gap_min_top -> not a barcoding gap.
    df = _hits(
        [
            ("s1", 95.0, 100, "A"),
            ("s1", 88.0, 100, "B"),
        ]
    )
    _, _, method = filter_by_barcoding_gap(df, gap_size=2, gap_min_top=97)
    assert method["s1"] == "similarity"


# --- filter_by_similarity ---


def test_similarity_keeps_within_window():
    df = _hits(
        [
            ("s1", 99.0, 100, "A"),
            ("s1", 98.0, 100, "B"),
            ("s1", 96.0, 100, "C"),  # 3 below top -> dropped at gap_size=2
        ]
    )
    out = filter_by_similarity(df, gap_size=2)
    assert sorted(out["identity_percentage"].tolist()) == [98.0, 99.0]


# --- filter_blast_results (dispatch + query_cover) ---


def test_filter_blast_results_query_cover_threshold():
    df = _hits(
        [
            ("s1", 99.0, 95, "A"),
            ("s1", 98.0, 50, "B"),  # below query_cover -> dropped
        ]
    )
    out = filter_blast_results(df, query_cover=90, method="similarity", gap_size=2)
    assert out["query_cover"].min() >= 90
    assert "filter_method" in out.columns


def test_filter_blast_results_unknown_method_raises():
    df = _hits([("s1", 99.0, 100, "A")])
    with pytest.raises(ValueError, match="Unknown filtering method"):
        filter_blast_results(df, method="nonsense")


def test_filter_blast_results_barcoding_gap_tags_methods():
    # s1 has a gap; s2 has none -> falls back to similarity.
    df = _hits(
        [
            ("s1", 99.0, 100, "A"),
            ("s1", 98.0, 100, "B"),
            ("s1", 90.0, 100, "C"),
            ("s2", 99.0, 100, "D"),
            ("s2", 98.5, 100, "E"),
        ]
    )
    out = filter_blast_results(df, query_cover=90, method="barcoding_gap", gap_size=2)
    methods = dict(zip(out["seq_id"], out["filter_method"]))
    assert methods["s1"] == "barcoding_gap"
    assert methods["s2"] == "similarity"
