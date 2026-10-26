"""Tests for geo_pipeline.classify_gbif_extents: one sequence's species put
in buckets (local, smaller, larger, never) relative to the priority extent."""

import pandas as pd

from trident.pipelines.geo_pipeline import classify_gbif_extents


def _long_df(rows):
    """rows: list of (species, extent, occurrences). in_mol/url filled."""
    return pd.DataFrame(
        [
            {
                "scientificName": sp,
                "in_mol": False,
                "geo_url": f"http://gbif/{sp}",
                "geo_extent": ext,
                "geo_occurrences": occ,
            }
            for sp, ext, occ in rows
        ]
    )


def _buckets_by_kind(classification):
    out = {}
    for b in classification["buckets"]:
        out.setdefault(b["kind"], []).append(b)
    return out


def test_local_bucket_holds_priority_validated_species():
    # A: 5 @ 500km (passes), B: 1 @ 500km (fails), both 5 @ global
    seq_df = _long_df(
        [
            ("A", 500, 5),
            ("A", "global", 5),
            ("B", 500, 1),
            ("B", "global", 5),
        ]
    )
    df_wide, classification = classify_gbif_extents(
        seq_df, min_occurrences=3, extents=[500, "global"], priority_extent=500
    )
    assert classification["priority_col"] == "500 km"
    buckets = _buckets_by_kind(classification)
    local = buckets["local"][0]["df"]
    assert local["scientificName"].tolist() == ["A"]
    # B validated only further out -> larger bucket
    larger = buckets["larger"]
    assert any(set(b["df"]["scientificName"]) == {"B"} for b in larger)


def test_never_bucket_for_species_below_threshold_everywhere():
    seq_df = _long_df([("C", 500, 1), ("C", "global", 2)])
    _, classification = classify_gbif_extents(
        seq_df, min_occurrences=3, extents=[500, "global"], priority_extent=500
    )
    buckets = _buckets_by_kind(classification)
    assert "local" in buckets and buckets["local"][0]["df"].empty
    never = buckets["never"][0]["df"]
    assert never["scientificName"].tolist() == ["C"]


def test_smaller_bucket_subsets_local_with_total():
    # A & B local at 500; A also passes 100 (smaller). C not local, so the
    # smaller breakdown is computed (it is skipped only when all species local).
    seq_df = _long_df(
        [
            ("A", 100, 4),
            ("A", 500, 6),
            ("B", 100, 0),
            ("B", 500, 5),
            ("C", 100, 0),
            ("C", 500, 1),
        ]
    )
    _, classification = classify_gbif_extents(
        seq_df, min_occurrences=3, extents=[100, 500], priority_extent=500
    )
    buckets = _buckets_by_kind(classification)
    assert set(buckets["local"][0]["df"]["scientificName"]) == {"A", "B"}
    smaller = buckets["smaller"][0]
    assert smaller["extent_col"] == "100 km"
    assert smaller["df"]["scientificName"].tolist() == ["A"]
    assert smaller["total"] == 2  # denominator = local count (A, B)


def test_species_go_to_the_nearest_extent_where_they_reach_the_threshold():
    """Reaching the threshold exactly counts; smaller extents only split the
    local species, larger ones (nearest first, then global) take the others
    once each, in km order whatever the order the extents are given in."""
    rows = [("A", 100, 3), ("A", 500, 3)]
    rows += [("B", 500, 2), ("B", 1000, 3), ("B", "global", 3)]
    rows += [("C", 500, 2), ("C", 1000, 2), ("C", "global", 3)]
    rows += [("D", 100, 2), ("D", 500, 2), ("D", "global", 2)]
    _, classification = classify_gbif_extents(
        _long_df(rows),
        min_occurrences=3,
        extents=[1000, 100, "global", 500],
        priority_extent=500,
    )
    layout = [
        (b["kind"], b["extent_col"], b["df"]["scientificName"].tolist())
        for b in classification["buckets"]
    ]
    assert layout == [
        ("local", "500 km", ["A"]),
        ("smaller", "100 km", ["A"]),
        ("larger", "1000 km", ["B"]),
        ("larger", "global", ["C"]),
        ("never", None, ["D"]),
    ]
