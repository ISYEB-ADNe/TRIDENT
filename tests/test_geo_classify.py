"""Tests for geo_pipeline.classify_gbif_extents — per-sequence extent bucketing.

Pure data logic extracted from the GEO UI: pivot one sequence's GBIF
occurrences to wide form and bucket species into local / smaller / larger /
never relative to the priority extent.
"""

import inspect

import pandas as pd

from trident.pipelines.geo_pipeline import classify_gbif_extents, finalize_geo_results

_finalize_geo = inspect.unwrap(finalize_geo_results)


def _long_df(rows):
    """rows: list of (species, extent, occurrences). in_mol/url filled."""
    return pd.DataFrame(
        [
            {
                "scientificName": sp,
                "in_mol": False,
                "gbif_taxonURL": f"http://gbif/{sp}",
                "gbif_extent": ext,
                "occurrences": occ,
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


def test_larger_dedup_does_not_repeat_species():
    # B fails at 500 (priority) but passes at both 1000 and global.
    # It must appear once (first larger extent), not in every larger bucket.
    seq_df = _long_df(
        [
            ("A", 500, 5),
            ("B", 500, 1),
            ("B", 1000, 5),
            ("B", "global", 9),
        ]
    )
    _, classification = classify_gbif_extents(
        seq_df,
        min_occurrences=3,
        extents=[500, 1000, "global"],
        priority_extent=500,
    )
    larger = _buckets_by_kind(classification)["larger"]
    appearances = [
        sp for b in larger for sp in b["df"]["scientificName"].tolist() if sp == "B"
    ]
    assert appearances == ["B"]


# --- finalize_geo_results: drop NCBI-seen-but-MOL-rejected pairs ---
#
# A (seq_id, species) pair is removed only when in_mol is False AND that exact
# pair appeared in the raw NCBI search (NCBI evaluated it and MOL rejected it).


def _geo_rows():
    return pd.DataFrame(
        {
            "seq_id": ["t1", "t1", "t1"],
            "scientificName": ["A", "B", "C"],
            "in_mol": [True, False, False],
        }
    )


def test_finalize_removes_only_rejected_non_mol_pairs():
    out = _finalize_geo(_geo_rows(), ncbi_search_pairs={("t1", "B")})
    # B: not in_mol AND seen by NCBI -> removed. A: in_mol -> kept.
    # C: not in_mol but never seen by NCBI -> kept.
    assert out["scientificName"].tolist() == ["A", "C"]


def test_finalize_keeps_in_mol_pair_even_if_in_ncbi_pairs():
    out = _finalize_geo(_geo_rows(), ncbi_search_pairs={("t1", "A")})
    assert "A" in out["scientificName"].tolist()  # in_mol shields it


def test_finalize_noop_without_pairs():
    out = _finalize_geo(_geo_rows(), ncbi_search_pairs=None)
    assert len(out) == 3
