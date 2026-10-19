"""Tests for the MOL identity plots: the x-axis counts hits, the bracket is the filter's cut."""

import pandas as pd

from trident.clients.ncbi import (
    mol_filter_cut,
    plot_identity_bars,
    plot_identity_percentage,
)


def _hits(rows):
    """Build one sequence's hits; rows are (identity, query_cover, name)."""
    return pd.DataFrame(
        [("s1", *row) for row in rows],
        columns=["seq_id", "identity_percentage", "query_cover", "scientificName"],
    )


def test_identity_plots_x_axis_is_the_number_of_hits():
    hits = _hits(
        [(100.0, 100, "A"), (100.0, 100, "A"), (100.0, 100, "B")]
        + [(90.0, 100, "C"), (90.0, 100, "C")]
    )
    points = plot_identity_percentage(hits, filtered_species={"A", "B"})
    last_x = {}
    for trace in points.data:
        for x, y in zip(trace.x, trace.y, strict=True):
            last_x[y] = max(last_x.get(y, 0), x)
    assert last_x == {100.0: 3, 90.0: 2}

    bars = plot_identity_bars(hits, filtered_species={"A", "B"}, bin_width=0.2)
    segments = {
        (t.name, y): x for t in bars.data for x, y in zip(t.x, t.y, strict=True)
    }
    assert segments == {
        ("A", 100.0): 2,
        ("B", 100.0): 1,
        ("Filtered out", 90.0): 2,
    }


def test_bracket_is_the_mol_filter_cut():
    # The 92 -> 88 drop is a gap but below gap_min_top: the filter uses the range
    low_gap = _hits([(93.0, 100, "A"), (92.0, 100, "B"), (88.0, 100, "C")])
    cut = mol_filter_cut(low_gap, query_cover=90, gap_size=2, gap_min_top=95)
    assert (cut.top, cut.bottom, cut.is_gap) == (93.0, 91.0, False)

    # A low-cover hit inside the gap does not shorten it
    gap = _hits(
        [(100.0, 100, "A"), (99.9, 100, "A"), (99.85, 100, "B")]
        + [(98.0, 50, "D"), (97.0, 100, "C")]
    )
    cut = mol_filter_cut(gap, query_cover=90, gap_size=2, gap_min_top=95)
    assert (cut.top, cut.bottom, cut.is_gap) == (99.85, 97.0, True)

    # The bars on each side of the gap sit at its ends
    bars = plot_identity_bars(gap, cut=cut, filtered_species={"A", "B"})
    assert {y for t in bars.data for y in t.y} == {99.85, 98.0, 97.0}
