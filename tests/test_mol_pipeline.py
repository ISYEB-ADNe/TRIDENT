"""Tests for trident.pipelines.mol_pipeline — summary and finalize logic."""

import pandas as pd
import pytest

from trident.pipelines.mol_pipeline import build_mol_summary, finalize_mol_results


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


# --- finalize_mol_results ---


class TestFinalizeMolResults:
    def test_adds_warning_column(self, ncbi_filter_df):
        mol_df, params = _finalize(ncbi_filter_df)
        assert "low_identity_warning" in mol_df.columns
        s1 = mol_df[mol_df["seq_id"] == "s1"]
        assert not s1["low_identity_warning"].any()
        s2 = mol_df[mol_df["seq_id"] == "s2"]
        assert s2["low_identity_warning"].all()

    def test_enforce_removes_low(self, ncbi_filter_df):
        mol_df, _ = _finalize(ncbi_filter_df, enforce_threshold=True)
        assert "s2" not in mol_df["seq_id"].values
        assert len(mol_df) == 2

    def test_enforce_false_keeps_all(self, ncbi_filter_df):
        mol_df, _ = _finalize(ncbi_filter_df, enforce_threshold=False)
        assert len(mol_df) == 4

    def test_returns_tuple(self, ncbi_filter_df):
        result = _finalize(ncbi_filter_df)
        assert isinstance(result, tuple)
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], dict)


# --- build_mol_summary ---


class TestBuildMolSummary:
    def test_all_sequences_present(self, sequences_df, ncbi_filter_df):
        mol_df, _ = _finalize(ncbi_filter_df)
        summary = build_mol_summary(mol_df, sequences_df)
        assert set(summary["seq_id"]) == {"s1", "s2", "s3"}

    def test_empty_sequence_row(self, sequences_df, ncbi_filter_df):
        mol_df, _ = _finalize(ncbi_filter_df)
        summary = build_mol_summary(mol_df, sequences_df)
        s3 = summary[summary["seq_id"] == "s3"]
        assert s3.iloc[0]["hits_count"] == 0
        assert s3.iloc[0]["species_count"] == 0

    def test_top_identity_uses_max(self, sequences_df, ncbi_filter_df):
        mol_df, _ = _finalize(ncbi_filter_df)
        summary = build_mol_summary(mol_df, sequences_df)
        s1 = summary[summary["seq_id"] == "s1"]
        assert s1.iloc[0]["top_identity"] == "99.00"

    def test_filter_method_recovery(self, sequences_df, ncbi_filter_df):
        """When enforce_threshold removes all hits, filter_method is recovered."""
        mol_df, _ = _finalize(ncbi_filter_df, enforce_threshold=True)
        summary = build_mol_summary(mol_df, sequences_df, ncbi_filter_df=ncbi_filter_df)
        s2 = summary[summary["seq_id"] == "s2"]
        assert s2.iloc[0]["filter_method"] == "similarity"

    def test_preserves_sequence_order(self, ncbi_filter_df):
        sequences_df = pd.DataFrame(
            {"seq_id": ["s2", "s1"], "dna_sequence": ["G", "A"]}
        )
        mol_df, _ = _finalize(ncbi_filter_df)
        summary = build_mol_summary(mol_df, sequences_df)
        assert list(summary["seq_id"]) == ["s2", "s1"]

    def test_low_identity_warning_any(self, sequences_df):
        """low_identity_warning is True if any hit for that seq has it."""
        ncbi_df = pd.DataFrame(
            {
                "seq_id": ["s1", "s1"],
                "scientificName": ["A", "B"],
                "identity_percentage": [99.0, 93.0],
                "query_cover": [100, 100],
                "filter_method": ["gap", "gap"],
            }
        )
        mol_df, _ = _finalize(ncbi_df)
        summary = build_mol_summary(mol_df, sequences_df)
        s1 = summary[summary["seq_id"] == "s1"]
        assert s1.iloc[0]["low_identity_warning"] == True  # noqa: E712
