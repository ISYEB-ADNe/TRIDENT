"""Tests for the NCBI barcoding-gap and similarity filtering (the core MOL method).

All functions here are pure (no network), so they are exercised directly.
"""

import pandas as pd
import pytest

from trident.clients.ncbi import (
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


# --- filter_by_barcoding_gap ---


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


@pytest.mark.parametrize("method", ["barcoding_gap", "similarity"])
def test_filter_blast_results_rejecting_every_hit_keeps_columns(method):
    df = _hits([("s1", 99.0, 10.0, "A a"), ("s2", 98.0, 10.0, "B b")])
    out = filter_blast_results(df, query_cover=90, method=method)
    assert out.empty
    assert {"seq_id", "identity_percentage", "scientificName"} <= set(out.columns)


def test_filter_blast_results_unknown_method_raises():
    """A misspelled method is refused, never replaced by another filter."""
    with pytest.raises(ValueError, match="Unknown filtering method"):
        filter_blast_results(_hits([("s1", 99.0, 100, "A")]), method="barcode_gap")


def test_gap_found_keeps_the_hits_above_it_else_similarity_is_used():
    # s1: 99, 98 then an 8-point drop to 90, a gap after the 98 hit; s2 has none.
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
    assert methods == {"s1": "barcoding_gap", "s2": "similarity"}
    s1 = out[out["seq_id"] == "s1"]["identity_percentage"]
    assert sorted(s1) == [98.0, 99.0]
