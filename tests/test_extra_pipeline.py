"""Tests for the EXTRA pipeline's network-free logic.

prepare_bold_input decides which species still need a BOLD proxy: a species is
only excluded when EVERY one of its (seq_id, species) pairs was already found
by NCBI, either kept by MOL (in_mol, which also matches synonyms) or among the
raw BLAST hits. run_bold_merge then attaches the BOLD proxies to the other
pairs only.
"""

import pandas as pd

from trident.pipelines.extra_pipeline import (
    build_extra_summary,
    prepare_bold_input,
    run_bold_merge,
)


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


def test_proxy_only_where_ncbi_did_not_find_the_species():
    # A is a candidate for three sequences: MOL kept it for t1 (under a
    # synonym, so only in_mol says so), BLAST returned it for t3 but MOL
    # rejected it, and NCBI did not find it for t2. Only t2 receives the BOLD
    # proxy or counts as queried.
    taxonomy = dict.fromkeys(
        [
            "dna_sequence",
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "specificEpithet",
            "scientificNameAuthorship",
            "taxonRank",
            "taxonID",
            "taxonID_db",
            "taxonURL",
        ]
    )
    geo_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t2", "t3"],
            "scientificName": ["A", "A", "A"],
            "in_mol": [True, False, False],
            **taxonomy,
        }
    )
    bold_filter_df = pd.DataFrame(
        {"scientificName": ["A"], "seq_id": ["BOLD1"], "dna_sequence": ["ACGT"]}
    )
    ncbi_search_df = pd.DataFrame(
        {"seq_id": ["t1", "t3"], "scientificName": ["A syn", "A"]}
    )
    merged, _ = run_bold_merge(
        bold_filter_df, geo_df, ncbi_search_df=ncbi_search_df, db_path=None
    )
    assert merged["seq_id"].tolist() == ["t2"]
    assert merged["seq_id_bold"].tolist() == ["BOLD1"]
    summary = build_extra_summary(merged, geo_df, ncbi_search_df=ncbi_search_df)
    assert summary["seq_id"].tolist() == ["t2"]
