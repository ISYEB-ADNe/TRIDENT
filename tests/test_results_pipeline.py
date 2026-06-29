"""Tests for trident.pipelines.results_pipeline — results assembly logic."""

import pandas as pd
import pytest

from trident.pipelines.results_pipeline import (
    EXPORT_COLS,
    build_results_df,
    find_sequence_exclusion_step,
    filter_excluded_results,
    add_below_mol,
    get_rejected_max_identity,
)


def _curated_df():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "dna_sequence": ["AAA", "AAA", "CCC"],
            "scientificName": ["Gadus morhua", "Gadus ogac", "Salmo salar"],
        }
    )


def test_filter_excluded_noop_when_empty():
    expected = _curated_df()
    assert filter_excluded_results(_curated_df(), set()).equals(expected)
    assert filter_excluded_results(_curated_df(), None).equals(expected)


def test_filter_excluded_drops_selected_species():
    out = filter_excluded_results(_curated_df(), {"s1||Gadus ogac"})
    assert "Gadus ogac" not in out["scientificName"].tolist()
    assert "Gadus morhua" in out["scientificName"].tolist()
    assert len(out) == 2


def test_filter_excluded_keeps_emptied_seq_as_blank_row():
    # Excluding the only species of s2 leaves an empty placeholder row.
    out = filter_excluded_results(_curated_df(), {"s2||Salmo salar"})
    s2 = out[out["seq_id"] == "s2"]
    assert len(s2) == 1
    assert pd.isna(s2["scientificName"].iloc[0])
    assert s2["dna_sequence"].iloc[0] == "CCC"


@pytest.fixture
def sequences_df():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s2", "s3"],
            "dna_sequence": ["ATCG", "GCTA", "TTTT"],
        }
    )


@pytest.fixture
def mol_df():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "scientificName": ["SpecA", "SpecB", "SpecC"],
            "identity_percentage": [99.0, 97.0, 95.0],
            "query_cover": [100, 100, 98],
            "hit_url": ["url1", "url2", "url3"],
        }
    )


@pytest.fixture
def geo_df():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "scientificName": ["SpecA", "SpecB", "SpecC"],
            "in_mol": [True, True, True],
            "gbif_occurrences": [100, 200, 50],
            "gbif_taxonURL": ["gurl1", "gurl2", "gurl3"],
            "taxonURL": ["wurl1", "wurl2", "wurl3"],
        }
    )


@pytest.fixture
def hypo_df():
    return pd.DataFrame(
        {
            "seq_id": ["s2"],
            "scientificName": ["SpecD"],
            "identity_percentage": [88.0],
            "query_cover": [95],
            "seq_url": ["bold_url"],
            "ncbi_top_identity_percentage": [92.0],
            "ncbi_top_query_cover": [100],
        }
    )


# --- build_results_df ---


class TestBuildResultsDf:
    def test_mol_geo_only(self, sequences_df, geo_df, mol_df):
        result = build_results_df(sequences_df, geo_df, mol_df)
        assert "validation_step" in result.columns
        mol_geo = result[result["validation_step"] == "MOL+GEO"]
        assert len(mol_geo) == 3

    def test_with_hypo(self, sequences_df, geo_df, mol_df, hypo_df):
        result = build_results_df(sequences_df, geo_df, mol_df, hypo_df=hypo_df)
        hypo = result[result["validation_step"] == "HYPO"]
        assert len(hypo) == 1
        assert hypo.iloc[0]["scientificName"] == "SpecD"

    def test_empty_sequences_get_row(self, sequences_df, geo_df, mol_df):
        """s3 has no species → gets an empty row with seq_id and dna_sequence."""
        result = build_results_df(sequences_df, geo_df, mol_df)
        s3_rows = result[result["seq_id"] == "s3"]
        assert len(s3_rows) == 1
        assert pd.isna(s3_rows.iloc[0]["scientificName"])
        assert s3_rows.iloc[0]["dna_sequence"] == "TTTT"

    def test_preserves_sequence_order(self, geo_df, mol_df):
        """Output follows the order of sequences_df."""
        sequences_df = pd.DataFrame(
            {"seq_id": ["s2", "s1"], "dna_sequence": ["G", "A"]}
        )
        result = build_results_df(sequences_df, geo_df, mol_df)
        seq_order = result["seq_id"].unique().tolist()
        assert seq_order == ["s2", "s1"]

    def test_all_export_cols_present(self, sequences_df, geo_df, mol_df, hypo_df):
        """All EXPORT_COLS should be present (possibly as NA)."""
        result = build_results_df(sequences_df, geo_df, mol_df, hypo_df=hypo_df)
        for col in EXPORT_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_hypo_renames_columns(self, sequences_df, geo_df, mol_df, hypo_df):
        """HYPO columns renamed: identity_percentage → proxy_identity_percentage."""
        result = build_results_df(sequences_df, geo_df, mol_df, hypo_df=hypo_df)
        hypo = result[result["validation_step"] == "HYPO"]
        assert "proxy_identity_percentage" in result.columns
        assert hypo.iloc[0]["proxy_identity_percentage"] == 88.0

    def test_empty_hypo_df(self, sequences_df, geo_df, mol_df):
        """Empty hypo_df is treated like None."""
        result = build_results_df(sequences_df, geo_df, mol_df, hypo_df=pd.DataFrame())
        assert (result["validation_step"].dropna() == "MOL+GEO").all()

    def test_ncbi_identity_from_mol(self, sequences_df, geo_df, mol_df):
        """MOL+GEO rows get best identity from mol_df."""
        result = build_results_df(sequences_df, geo_df, mol_df)
        s1_specA = result[
            (result["seq_id"] == "s1") & (result["scientificName"] == "SpecA")
        ]
        assert s1_specA.iloc[0]["ncbi_top_identity_percentage"] == 99.0


# --- find_sequence_exclusion_step ---


class TestFindExclusionStep:
    def test_basic(self):
        result = find_sequence_exclusion_step(
            ["s1", "s2", "s3"],
            ncbi_search_df=pd.DataFrame({"seq_id": ["s1", "s2", "s3"]}),
            mol_df=pd.DataFrame({"seq_id": ["s1", "s2"]}),
            geo_df=pd.DataFrame({"seq_id": ["s1"]}),
        )
        assert len(result) == 2
        s3 = result[result["seq_id"] == "s3"]
        assert s3.iloc[0]["pipeline_step"] == "MOL Filter"

    def test_no_steps(self):
        result = find_sequence_exclusion_step(["s1"])
        assert result.empty

    def test_all_present(self):
        result = find_sequence_exclusion_step(
            ["s1"],
            ncbi_search_df=pd.DataFrame({"seq_id": ["s1"]}),
            mol_df=pd.DataFrame({"seq_id": ["s1"]}),
        )
        assert result.empty


# --- add_below_mol / get_rejected_max_identity ---
#
# below_mol flags a row when its identity is at or below the strongest hit that
# MOL filtering rejected for that sequence (a hit NCBI found but MOL discarded).


def _mol_kept():
    # MOL kept only SpecA @ 99 for s1.
    return pd.DataFrame(
        {
            "seq_id": ["s1"],
            "scientificName": ["SpecA"],
            "identity_percentage": [99.0],
        }
    )


def _ncbi_all():
    # NCBI also returned SpecX @ 96 and SpecY @ 90 for s1 (both rejected by MOL).
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s1"],
            "scientificName": ["SpecA", "SpecX", "SpecY"],
            "identity_percentage": [99.0, 96.0, 90.0],
        }
    )


class TestBelowMol:
    def test_rejected_max_is_strongest_rejected_hit(self):
        s = get_rejected_max_identity(_mol_kept(), _ncbi_all())
        assert s.loc["s1"] == 96.0  # SpecA(99) kept, so max rejected is SpecX(96)

    def test_flags_at_or_below_rejected_max(self):
        df = pd.DataFrame(
            {
                "seq_id": ["s1", "s1", "s1"],
                "ncbi_top_identity_percentage": [95.0, 96.0, 99.0],
            }
        )
        out = add_below_mol(df.copy(), _mol_kept(), _ncbi_all())
        # rejected_max = 96: 95 <= 96 True, 96 <= 96 True (boundary), 99 > 96 False
        assert out["below_mol"].tolist() == [True, True, False]

    def test_no_rejected_hits_means_no_flags(self):
        # mol == ncbi: nothing was rejected, so nothing is below_mol.
        df = pd.DataFrame({"seq_id": ["s1"], "ncbi_top_identity_percentage": [50.0]})
        out = add_below_mol(df.copy(), _ncbi_all(), _ncbi_all())
        assert not out["below_mol"].any()

    def test_missing_identity_column(self):
        df = pd.DataFrame({"seq_id": ["s1"]})  # no identity column
        out = add_below_mol(df.copy(), _mol_kept(), _ncbi_all())
        assert out["below_mol"].tolist() == [False]

    def test_empty_or_none_inputs(self):
        df = pd.DataFrame({"seq_id": ["s1"], "ncbi_top_identity_percentage": [10.0]})
        assert not add_below_mol(df.copy(), pd.DataFrame(), None)["below_mol"].any()
        assert not add_below_mol(df.copy(), None, _ncbi_all())["below_mol"].any()

    def test_nan_identity_not_flagged(self):
        df = pd.DataFrame({"seq_id": ["s1"], "ncbi_top_identity_percentage": [pd.NA]})
        out = add_below_mol(df.copy(), _mol_kept(), _ncbi_all())
        assert out["below_mol"].tolist() == [False]
