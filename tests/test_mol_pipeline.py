"""Tests for trident.pipelines.mol_pipeline — summary and finalize logic."""

import pandas as pd
import pytest

from trident.clients.ncbi import filter_blast_results
from trident.pipelines.mol_pipeline import (
    build_mol_sequence_report,
    build_mol_summary,
    finalize_mol_results,
)


@pytest.fixture
def ncbi_filter_df():
    """Simulates output from run_ncbi_filter with two sequences."""
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2", "s2"],
            "scientificName": ["SpecA", "SpecB", "SpecC", "SpecC"],
            "identity_percentage": [99.0, 97.0, 93.0, 91.0],
            "query_cover": [100, 100, 98, 95],
            "filter_method": [
                "barcoding_gap",
                "barcoding_gap",
                "similarity",
                "similarity",
            ],
        }
    )


@pytest.fixture
def sequences_df():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s2", "s3"],
            "dna_sequence": ["ATCG", "GCTA", "TTTT"],
        }
    )


def _finalize(ncbi_filter_df, **kwargs):
    """Call finalize_mol_results without db caching."""
    kwargs.setdefault("threshold", 95.0)
    return finalize_mol_results(ncbi_filter_df, **kwargs)


# --- finalize_mol_results and build_mol_summary ---


class TestMolResults:
    def test_enforced_threshold_removes_the_hits_below_it(
        self, sequences_df, ncbi_filter_df
    ):
        """Without enforcement every hit is kept; with it, a sequence whose hits
        are all below loses them but keeps the filter method in the summary."""
        assert len(_finalize(ncbi_filter_df, enforce_threshold=False)[0]) == 4
        mol_df, _ = _finalize(ncbi_filter_df, enforce_threshold=True)
        assert mol_df["seq_id"].tolist() == ["s1", "s1"]
        summary = build_mol_summary(mol_df, sequences_df, ncbi_filter_df=ncbi_filter_df)
        s2 = summary[summary["seq_id"] == "s2"]
        assert s2.iloc[0]["filter_method"] == "similarity"

    def test_hits_are_flagged_by_their_sequences_best_identity(self, ncbi_filter_df):
        mol_df, _ = _finalize(ncbi_filter_df)
        flags = mol_df.groupby("seq_id")["low_identity_warning"].agg(["any", "all"])
        assert not flags.loc["s1", "any"]
        assert flags.loc["s2", "all"]

    def test_summary_has_one_row_per_sequence_in_input_order(self, ncbi_filter_df):
        """A sequence without hits gets a row with no hits and no species; the
        top identity is the best hit."""
        sequences_df = pd.DataFrame(
            {"seq_id": ["s3", "s2", "s1"], "dna_sequence": ["T", "G", "A"]}
        )
        mol_df, _ = _finalize(ncbi_filter_df)
        summary = build_mol_summary(mol_df, sequences_df).set_index("seq_id")
        assert summary.index.tolist() == ["s3", "s2", "s1"]
        assert summary.loc["s3", ["n_hits", "n_species"]].tolist() == [0, 0]
        assert summary.loc["s1", "top_identity"] == 99.0

    def test_low_identity_warning_flags_a_species_below(self, sequences_df):
        """A sequence is flagged when a species kept for it has its best hit
        below the threshold, not when only a lower hit of a species is."""
        ncbi_df = pd.DataFrame(
            {
                "seq_id": ["s1", "s1", "s2", "s2"],
                "scientificName": ["A", "B", "C", "C"],
                "identity_percentage": [99.0, 93.0, 96.0, 94.0],
                "query_cover": [100, 100, 100, 100],
                "filter_method": ["similarity"] * 4,
            }
        )
        mol_df, _ = _finalize(ncbi_df)
        summary = build_mol_summary(mol_df, sequences_df).set_index("seq_id")
        assert summary.loc["s1", "low_identity_warning"]
        assert not summary.loc["s2", "low_identity_warning"]


def test_sequence_report_finds_the_gap_on_all_hits():
    # The MOL list holds only the hits above the gap: the report must look at
    # the search hits to find it and the species matching on both sides
    hits = pd.DataFrame(
        {
            "seq_id": "s1",
            "scientificName": ["SpecA", "SpecA", "SpecB", "SpecA", "SpecC"],
            "identity_percentage": [100.0, 99.5, 99.5, 96.0, 95.0],
            "query_cover": 100,
        }
    )
    mol_df = filter_blast_results(hits, query_cover=90, gap_size=2, gap_min_top=95)

    report = build_mol_sequence_report(
        mol_df, "s1", hits, query_cover=90, gap_size=2, gap_min_top=95
    )

    assert report["cut"].is_gap
    assert (report["cut"].top, report["cut"].bottom) == (99.5, 96.0)
    assert report["species_across_gap"] == {
        "SpecA": {"before_gap": [100.0, 99.5], "after_gap": [96.0]}
    }
