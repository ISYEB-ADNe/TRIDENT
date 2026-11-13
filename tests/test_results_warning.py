"""The low identity warning of the Results tab: by each sequence's best
identity."""

import pandas as pd

from trident.pipelines.results_pipeline import (
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
