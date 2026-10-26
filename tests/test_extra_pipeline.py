"""Tests for the EXTRA pipeline's network-free logic.

prepare_bold_input decides which species still need a BOLD proxy: a species is
only excluded when EVERY one of its (seq_id, species) pairs was already found
by NCBI, either kept by MOL (in_mol, which also matches synonyms) or among the
raw BLAST hits, under any of the species' names. run_bold_merge then attaches
the BOLD proxies to the other pairs only.
"""

import pandas as pd

from trident.pipelines import extra_pipeline
from trident.pipelines.names import build_name_table
from trident.pipelines.extra_pipeline import (
    build_extra_summary,
    prepare_bold_input,
    run_bold_filter,
    run_bold_merge,
)


def test_excludes_pairs_already_covered_by_ncbi():
    # BLAST returned D for t2 under its synonym D old, which the MOL filter
    # rejected (the name table knows it from another sequence's MOL list).
    geo_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1", "t2", "t2"],
            "scientificName": ["A", "B", "C", "D"],
            "in_mol": [True, False, False, False],
        }
    )
    ncbi_search_df = pd.DataFrame(
        {"seq_id": ["t1", "t2"], "scientificName": ["A", "D old"]}
    )
    name_table = build_name_table(
        pd.DataFrame(
            {
                "scientificName": ["D old"],
                "acceptedName": ["D"],
                "acceptedNameUsageID": [1],
                "matchedBy": ["name"],
            }
        )
    )

    species, n_excluded = prepare_bold_input(geo_df, ncbi_search_df, name_table)
    assert set(species) == {"B", "C"}  # A and D covered by NCBI -> dropped
    assert n_excluded == 2


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

    species, n_excluded = prepare_bold_input(geo_df, ncbi_search_df, None)
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
    species, n_excluded = prepare_bold_input(geo_df, None, None)
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
        {
            "scientificName": ["A"],
            "seq_id": ["BOLD1"],
            "dna_sequence": ["ACGT"],
            "seq_url": ["https://portal.boldsystems.org/record/BOLD1"],
        }
    )
    ncbi_search_df = pd.DataFrame(
        {"seq_id": ["t1", "t3"], "scientificName": ["A syn", "A"]}
    )
    merged, _ = run_bold_merge(
        bold_filter_df, geo_df, ncbi_search_df, None, db_path=None
    )
    assert merged["seq_id"].tolist() == ["t2"]
    assert merged["proxy_id"].tolist() == ["BOLD1"]
    summary = build_extra_summary(merged, geo_df, ncbi_search_df, None)
    assert summary["seq_id"].tolist() == ["t2"]


def test_resumed_bold_filter_filters_only_the_new_species(tmp_path, monkeypatch):
    """Species filtered before are read back; only a new one is filtered, and
    each keeps one of its identical sequences."""
    filtered = []
    real = extra_pipeline.select_longest_sequences

    def select(seqs, **kwargs):
        filtered.append(len(seqs))
        return real(seqs, **kwargs)

    monkeypatch.setattr(extra_pipeline, "select_longest_sequences", select)

    def search_df(species):
        return pd.DataFrame(
            {
                "scientificName": [s for s in species for _ in range(2)],
                "dna_sequence": ["ACGTACGTAC"] * 2 * len(species),
            }
        )

    db_path = str(tmp_path / "analysis.db")
    run_bold_filter(search_df(["A", "B"]), similarity=98, db_path=db_path)
    filtered.clear()
    df, _ = run_bold_filter(search_df(["A", "B", "C"]), similarity=98, db_path=db_path)
    assert filtered == [2]
    assert sorted(df["scientificName"]) == ["A", "B", "C"]
