"""Tests for the EXTRA pipeline's network-free logic.

prepare_bold_input decides which species still need a BOLD proxy: a species is
only excluded when EVERY one of its (seq_id, species) pairs was already covered
by a direct NCBI BLAST hit (falling back to the in_mol flag when ncbi_search_df
is absent).
"""

import pandas as pd

from trident.pipelines.extra_pipeline import prepare_bold_input


def test_excludes_pairs_already_covered_by_ncbi():
    geo_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1", "t2"],
            "scientificName": ["A", "B", "C"],
            "in_mol": [True, False, False],
        }
    )
    ncbi_search_df = pd.DataFrame({"seq_id": ["t1"], "scientificName": ["A"]})

    species, n_excluded = prepare_bold_input(geo_df, ncbi_search_df)
    assert set(species) == {"B", "C"}  # A covered by NCBI -> dropped
    assert n_excluded == 1


def test_species_kept_when_uncovered_for_another_sequence():
    # A is NCBI-covered for t1 but NOT for t2, so it still needs BOLD.
    geo_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t2"],
            "scientificName": ["A", "A"],
            "in_mol": [True, False],
        }
    )
    ncbi_search_df = pd.DataFrame({"seq_id": ["t1"], "scientificName": ["A"]})

    species, n_excluded = prepare_bold_input(geo_df, ncbi_search_df)
    assert species == ["A"]
    assert n_excluded == 0


def test_fallback_to_in_mol_without_ncbi_df():
    geo_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1"],
            "scientificName": ["A", "B"],
            "in_mol": [True, False],
        }
    )
    species, n_excluded = prepare_bold_input(geo_df, ncbi_search_df=None)
    assert species == ["B"]  # in_mol species A excluded
    assert n_excluded == 1
