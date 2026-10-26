"""The low identity warnings of the results pipeline: the Results tab's (by
each sequence's best identity) and the notebook's (by row)."""

import pandas as pd

from trident.pipelines.results_pipeline import (
    add_low_identity_warning,
    best_identity_per_sequence,
    flag_low_identity,
)


def test_a_sequence_is_flagged_by_the_best_identity_of_its_species():
    """Only hits covering the MOL query cover count. A HYPO species counts its
    check hit, else the sequence's MOL hit on the species its proxy matched
    (via); on a tie the species' own hit wins. A sequence without any such
    hit is not flagged."""
    na = pd.NA
    curated = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2", "s3", "s3", "s4", "s5"],
            "validation_step": [
                "MOL+GEO",
                "HYPO",
                "MOL+GEO",
                "HYPO",
                "MOL+GEO",
                "MOL+GEO",
                "HYPO",
            ],
            "mol_identity": [96.0, na, 99.0, na, 98.0, 99.0, na],
            "mol_query_cover": [100, na, 100, na, 100, 40, na],
            "hypo_check_identity": [na, 98.5, na, na, na, na, na],
            "hypo_check_query_cover": [na, 100, na, na, na, na, na],
            "via_mol_identity": [na, na, na, 98.0, na, na, 97.0],
            "proxy_hit_species": [na, "X", na, "Y", na, na, "Z"],
        }
    )
    best = best_identity_per_sequence(curated, query_cover=90)
    assert best["identity"].to_dict() == {
        "s1": 98.5,
        "s2": 99.0,
        "s3": 98.0,
        "s5": 97.0,
    }
    assert best["via"].dropna().to_dict() == {"s5": "Z"}

    flagged = flag_low_identity(curated, 98.6, best["identity"])
    by_seq = flagged.groupby("seq_id")["low_identity_warning"].all()
    assert by_seq.to_dict() == {
        "s1": True,
        "s2": False,
        "s3": True,
        "s4": False,
        "s5": True,
    }


def test_a_row_is_flagged_only_when_its_identity_is_below_the_threshold():
    """Strictly below: a row at the threshold, without identity, or with one
    that is not a number is not flagged; identities read back as text count
    as numbers."""
    df = pd.DataFrame(
        {"mol_identity": [99.0, 93.0, 97.0, pd.NA, "95.5", "98", "not_a_number"]}
    )
    result = add_low_identity_warning(df, threshold=97)
    assert result["low_identity_warning"].tolist() == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
    ]
